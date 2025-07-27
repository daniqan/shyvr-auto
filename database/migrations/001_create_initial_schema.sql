-- Migration 001: Create Initial Schema for Activity Logging
-- This migration creates the foundational database schema for activity logging
-- Run with: psql -d shyvr_rlte -f database/migrations/001_create_initial_schema.sql

BEGIN;

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS AND TYPES
-- =============================================================================

-- Activity categories for organized logging
CREATE TYPE activity_category AS ENUM (
    'system',          -- System operations, startup, shutdown, errors
    'trading',         -- All trading-related activities
    'user',           -- User interactions and authentication
    'ml_rl',          -- Machine learning and reinforcement learning
    'security',       -- Security events, auth failures, violations
    'api',            -- External API calls and responses
    'performance',    -- Performance monitoring and alerts
    'data',           -- Data processing, validation, corruption
    'configuration',  -- Config changes, settings updates
    'integration',    -- Cross-component integration events
    'dashboard'       -- Dashboard-specific events
);

-- Activity severity levels
CREATE TYPE activity_severity AS ENUM (
    'trace',     -- Fine-grained debugging information
    'debug',     -- Debug information
    'info',      -- General information
    'notice',    -- Notable conditions
    'warning',   -- Warning conditions
    'error',     -- Error conditions
    'critical',  -- Critical conditions
    'alert',     -- Action must be taken immediately
    'emergency'  -- System is unusable
);

-- Activity action types for more granular tracking
CREATE TYPE activity_action AS ENUM (
    'create',     -- Resource creation
    'read',       -- Resource access/read
    'update',     -- Resource modification
    'delete',     -- Resource deletion
    'execute',    -- Action execution (trades, analysis)
    'start',      -- Service/process start
    'stop',       -- Service/process stop
    'pause',      -- Service/process pause
    'resume',     -- Service/process resume
    'error',      -- Error occurrence
    'alert',      -- Alert generation
    'login',      -- User authentication
    'logout',     -- User logout
    'access',     -- Resource access
    'violation',  -- Security or policy violation
    'timeout',    -- Operation timeout
    'retry',      -- Operation retry
    'success',    -- Operation success
    'failure'     -- Operation failure
);

-- Trading mode types
CREATE TYPE trading_mode AS ENUM (
    'analysis',   -- Analysis mode only
    'simulation', -- Simulation trading
    'live'        -- Live trading with real funds
);

-- Chain types for blockchain identification
CREATE TYPE chain_type AS ENUM (
    'ethereum',   -- Ethereum mainnet
    'solana',     -- Solana mainnet
    'base'        -- Base chain
);

-- =============================================================================
-- USERS TABLE (Required for foreign key references)
-- =============================================================================

CREATE TABLE users (
    telegram_user_id BIGINT PRIMARY KEY,
    username VARCHAR(100),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for user lookups
CREATE INDEX idx_users_username ON users (username) WHERE username IS NOT NULL;
CREATE INDEX idx_users_active ON users (is_active, last_seen_at DESC);

-- =============================================================================
-- MAIN ACTIVITY LOGS TABLE
-- =============================================================================

CREATE TABLE activity_logs (
    -- Primary identifiers
    id BIGSERIAL PRIMARY KEY,
    activity_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    
    -- Timestamp with high precision for ordering
    created_at TIMESTAMP(6) WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Activity classification
    category activity_category NOT NULL,
    action activity_action NOT NULL,
    severity activity_severity DEFAULT 'info' NOT NULL,
    
    -- Basic activity information
    source VARCHAR(100) NOT NULL,          -- Component/module generating the log
    event_type VARCHAR(100) NOT NULL,      -- Specific event type
    title VARCHAR(255) NOT NULL,           -- Human-readable title
    description TEXT,                      -- Detailed description
    
    -- Context and relationships
    user_id BIGINT REFERENCES users(telegram_user_id),
    session_id UUID,                       -- Session tracking
    request_id UUID,                       -- Request correlation
    parent_activity_id UUID REFERENCES activity_logs(activity_id), -- Activity hierarchy
    
    -- Trading context
    trading_mode trading_mode,             -- Current trading mode when event occurred
    token_address VARCHAR(64),             -- Related token
    chain chain_type,                      -- Related blockchain
    
    -- Financial context
    amount_usd DECIMAL(20,8),              -- Financial amount if applicable
    fee_usd DECIMAL(20,8),                 -- Fee amount if applicable
    
    -- Technical context
    execution_time_ms INTEGER,             -- Execution duration
    memory_usage_mb INTEGER,               -- Memory usage at time of event
    cpu_usage_pct DECIMAL(5,2),           -- CPU usage percentage
    
    -- Metadata and additional context
    metadata JSONB,                        -- Flexible metadata storage
    tags TEXT[],                           -- Searchable tags
    correlation_id VARCHAR(100),           -- External correlation ID
    
    -- Error context
    error_code VARCHAR(50),                -- Standardized error code
    error_message TEXT,                    -- Error message
    stack_trace TEXT,                      -- Stack trace for errors
    
    -- API context (for API-related activities)
    api_endpoint VARCHAR(200),             -- API endpoint
    http_method VARCHAR(10),               -- HTTP method
    http_status INTEGER,                   -- HTTP status code
    user_agent TEXT,                       -- User agent string
    ip_address INET,                       -- Source IP address
    
    -- Performance metrics
    response_time_ms INTEGER,              -- Response time
    throughput_ops_per_sec DECIMAL(10,2),  -- Throughput metric
    
    -- Security context
    security_level VARCHAR(20) DEFAULT 'normal', -- Security classification
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    
    -- Dashboard specific
    dashboard_component VARCHAR(100),       -- Specific dashboard component
    dashboard_action VARCHAR(100),          -- Dashboard-specific action
    
    -- Data integrity
    checksum VARCHAR(64),                   -- Data integrity checksum
    version INTEGER DEFAULT 1,             -- Schema version for migration
    
    -- Indexing hints
    indexed_at TIMESTAMP WITH TIME ZONE,   -- When this record was indexed
    archived_at TIMESTAMP WITH TIME ZONE   -- When this record was archived
);

-- =============================================================================
-- ACTIVITY SUMMARIES TABLE (for dashboard performance)
-- =============================================================================

CREATE TABLE activity_summaries (
    id BIGSERIAL PRIMARY KEY,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- 'minute', 'hour', 'day', 'week'
    
    -- Aggregated metrics
    category activity_category NOT NULL,
    total_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    
    -- Performance metrics
    avg_execution_time_ms DECIMAL(10,2),
    max_execution_time_ms INTEGER,
    avg_response_time_ms DECIMAL(10,2),
    max_response_time_ms INTEGER,
    
    -- System metrics
    avg_memory_usage_mb DECIMAL(10,2),
    max_memory_usage_mb INTEGER,
    avg_cpu_usage_pct DECIMAL(5,2),
    max_cpu_usage_pct DECIMAL(5,2),
    
    -- Top sources and events
    top_sources JSONB,                     -- Top activity sources in this period
    top_events JSONB,                      -- Top event types in this period
    top_errors JSONB,                      -- Top errors in this period
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(period_start, period_end, period_type, category)
);

-- =============================================================================
-- USER ACTIVITY SESSIONS TABLE
-- =============================================================================

CREATE TABLE user_activity_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id BIGINT REFERENCES users(telegram_user_id) NOT NULL,
    
    -- Session lifecycle
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Session context
    source VARCHAR(50) NOT NULL,           -- 'dashboard', 'api', 'telegram'
    user_agent TEXT,
    ip_address INET,
    
    -- Session metrics
    total_activities INTEGER DEFAULT 0,
    unique_components_accessed INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    
    -- Session metadata
    metadata JSONB,
    
    -- Update triggers will maintain last_activity_at
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMIT;