-- Activity Logs Database Schema Design
-- Comprehensive logging schema for Shyvr RLTE dashboard activity tracking
-- 
-- This schema is designed for production-ready high-volume logging with:
-- - High performance time-based queries
-- - Comprehensive activity categorization
-- - Scalable data retention and partitioning
-- - Rich metadata and context tracking

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

-- =============================================================================
-- PERFORMANCE OPTIMIZED INDEXES
-- =============================================================================

-- Primary time-based query indexes (most important)
CREATE INDEX idx_activity_logs_created_at_desc ON activity_logs (created_at DESC);
CREATE INDEX idx_activity_logs_created_at_category ON activity_logs (created_at DESC, category);
CREATE INDEX idx_activity_logs_created_at_severity ON activity_logs (created_at DESC, severity);

-- Category and action filtering
CREATE INDEX idx_activity_logs_category_action ON activity_logs (category, action);
CREATE INDEX idx_activity_logs_category_created ON activity_logs (category, created_at DESC);
CREATE INDEX idx_activity_logs_action_created ON activity_logs (action, created_at DESC);

-- Severity-based queries
CREATE INDEX idx_activity_logs_severity_created ON activity_logs (severity, created_at DESC) 
WHERE severity IN ('error', 'critical', 'alert', 'emergency');

-- User and session tracking
CREATE INDEX idx_activity_logs_user_id_created ON activity_logs (user_id, created_at DESC) 
WHERE user_id IS NOT NULL;
CREATE INDEX idx_activity_logs_session_id ON activity_logs (session_id) 
WHERE session_id IS NOT NULL;
CREATE INDEX idx_activity_logs_request_id ON activity_logs (request_id) 
WHERE request_id IS NOT NULL;

-- Source and component tracking
CREATE INDEX idx_activity_logs_source_created ON activity_logs (source, created_at DESC);
CREATE INDEX idx_activity_logs_event_type ON activity_logs (event_type);
CREATE INDEX idx_activity_logs_dashboard_component ON activity_logs (dashboard_component) 
WHERE dashboard_component IS NOT NULL;

-- Trading context indexes
CREATE INDEX idx_activity_logs_trading_mode ON activity_logs (trading_mode) 
WHERE trading_mode IS NOT NULL;
CREATE INDEX idx_activity_logs_token_chain ON activity_logs (token_address, chain) 
WHERE token_address IS NOT NULL;

-- Error and performance tracking
CREATE INDEX idx_activity_logs_error_code ON activity_logs (error_code) 
WHERE error_code IS NOT NULL;
CREATE INDEX idx_activity_logs_execution_time ON activity_logs (execution_time_ms DESC) 
WHERE execution_time_ms IS NOT NULL;
CREATE INDEX idx_activity_logs_response_time ON activity_logs (response_time_ms DESC) 
WHERE response_time_ms IS NOT NULL;

-- API tracking
CREATE INDEX idx_activity_logs_api_endpoint ON activity_logs (api_endpoint, http_status) 
WHERE api_endpoint IS NOT NULL;

-- Security and risk
CREATE INDEX idx_activity_logs_risk_score ON activity_logs (risk_score DESC) 
WHERE risk_score IS NOT NULL;
CREATE INDEX idx_activity_logs_ip_address ON activity_logs (ip_address) 
WHERE ip_address IS NOT NULL;

-- Hierarchy and correlation
CREATE INDEX idx_activity_logs_parent_activity ON activity_logs (parent_activity_id) 
WHERE parent_activity_id IS NOT NULL;
CREATE INDEX idx_activity_logs_correlation_id ON activity_logs (correlation_id) 
WHERE correlation_id IS NOT NULL;

-- GIN indexes for JSONB and array data
CREATE INDEX idx_activity_logs_metadata_gin ON activity_logs USING GIN (metadata);
CREATE INDEX idx_activity_logs_tags_gin ON activity_logs USING GIN (tags);

-- Activity summaries indexes
CREATE INDEX idx_activity_summaries_period ON activity_summaries (period_start, period_end, period_type);
CREATE INDEX idx_activity_summaries_category_period ON activity_summaries (category, period_start DESC);

-- User sessions indexes
CREATE INDEX idx_user_sessions_user_started ON user_activity_sessions (user_id, started_at DESC);
CREATE INDEX idx_user_sessions_session_id ON user_activity_sessions (session_id);
CREATE INDEX idx_user_sessions_active ON user_activity_sessions (last_activity_at DESC) 
WHERE ended_at IS NULL;

-- =============================================================================
-- TABLE PARTITIONING FOR SCALABILITY
-- =============================================================================

-- Enable partitioning by time (monthly partitions)
-- This would be implemented as needed for high-volume installations

-- Example: Create monthly partitions for the current year
-- This can be automated with a maintenance script

-- =============================================================================
-- DATA RETENTION POLICIES
-- =============================================================================

-- Create a function to implement data retention
CREATE OR REPLACE FUNCTION cleanup_old_activity_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    retention_days INTEGER := 90; -- Default 90 days retention
BEGIN
    -- Delete logs older than retention period (except critical ones)
    DELETE FROM activity_logs 
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days
    AND severity NOT IN ('critical', 'alert', 'emergency');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Archive critical logs older than 1 year
    UPDATE activity_logs 
    SET archived_at = NOW()
    WHERE created_at < NOW() - INTERVAL '1 year'
    AND severity IN ('critical', 'alert', 'emergency')
    AND archived_at IS NULL;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to generate activity summaries
CREATE OR REPLACE FUNCTION generate_activity_summaries(
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    summary_period VARCHAR(20) DEFAULT 'hour'
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER;
BEGIN
    -- Generate hourly/daily summaries
    INSERT INTO activity_summaries (
        period_start, period_end, period_type, category,
        total_count, error_count, warning_count,
        avg_execution_time_ms, max_execution_time_ms,
        avg_response_time_ms, max_response_time_ms,
        avg_memory_usage_mb, max_memory_usage_mb,
        avg_cpu_usage_pct, max_cpu_usage_pct,
        top_sources, top_events, top_errors
    )
    SELECT 
        start_time,
        end_time,
        summary_period,
        category,
        COUNT(*) as total_count,
        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
        COUNT(*) FILTER (WHERE severity = 'warning') as warning_count,
        AVG(execution_time_ms) as avg_execution_time_ms,
        MAX(execution_time_ms) as max_execution_time_ms,
        AVG(response_time_ms) as avg_response_time_ms,
        MAX(response_time_ms) as max_response_time_ms,
        AVG(memory_usage_mb) as avg_memory_usage_mb,
        MAX(memory_usage_mb) as max_memory_usage_mb,
        AVG(cpu_usage_pct) as avg_cpu_usage_pct,
        MAX(cpu_usage_pct) as max_cpu_usage_pct,
        jsonb_object_agg(source, source_count) FILTER (WHERE source_rank <= 5) as top_sources,
        jsonb_object_agg(event_type, event_count) FILTER (WHERE event_rank <= 5) as top_events,
        jsonb_object_agg(error_code, error_count) FILTER (WHERE error_rank <= 5 AND error_code IS NOT NULL) as top_errors
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY category ORDER BY source_count DESC) as source_rank,
            ROW_NUMBER() OVER (PARTITION BY category ORDER BY event_count DESC) as event_rank,
            ROW_NUMBER() OVER (PARTITION BY category ORDER BY error_count DESC) as error_rank
        FROM (
            SELECT 
                category,
                source,
                event_type,
                error_code,
                execution_time_ms,
                response_time_ms,
                memory_usage_mb,
                cpu_usage_pct,
                severity,
                COUNT(*) OVER (PARTITION BY category, source) as source_count,
                COUNT(*) OVER (PARTITION BY category, event_type) as event_count,
                COUNT(*) OVER (PARTITION BY category, error_code) as error_count
            FROM activity_logs 
            WHERE created_at >= start_time AND created_at < end_time
        ) ranked
    ) final
    GROUP BY category
    ON CONFLICT (period_start, period_end, period_type, category) 
    DO NOTHING;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS AND AUTOMATION
-- =============================================================================

-- Trigger to update user session activity
CREATE OR REPLACE FUNCTION update_user_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Update session last activity time
    UPDATE user_activity_sessions 
    SET 
        last_activity_at = NEW.created_at,
        total_activities = total_activities + 1,
        errors_encountered = errors_encountered + CASE WHEN NEW.severity IN ('error', 'critical', 'alert', 'emergency') THEN 1 ELSE 0 END,
        updated_at = NOW()
    WHERE session_id = NEW.session_id 
    AND ended_at IS NULL;
    
    -- If no session found and user_id exists, create one
    IF NOT FOUND AND NEW.user_id IS NOT NULL AND NEW.session_id IS NOT NULL THEN
        INSERT INTO user_activity_sessions (
            session_id, user_id, source, total_activities, 
            errors_encountered, started_at, last_activity_at
        ) VALUES (
            NEW.session_id, NEW.user_id, NEW.source, 1,
            CASE WHEN NEW.severity IN ('error', 'critical', 'alert', 'emergency') THEN 1 ELSE 0 END,
            NEW.created_at, NEW.created_at
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_user_session_activity
    AFTER INSERT ON activity_logs
    FOR EACH ROW
    WHEN (NEW.session_id IS NOT NULL)
    EXECUTE FUNCTION update_user_session_activity();

-- Trigger to automatically set indexed_at timestamp
CREATE OR REPLACE FUNCTION set_indexed_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.indexed_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_indexed_timestamp
    BEFORE INSERT ON activity_logs
    FOR EACH ROW
    EXECUTE FUNCTION set_indexed_timestamp();

-- =============================================================================
-- VIEWS FOR COMMON DASHBOARD QUERIES
-- =============================================================================

-- Recent activity view for dashboard
CREATE OR REPLACE VIEW recent_activity AS
SELECT 
    activity_id,
    created_at,
    category,
    action,
    severity,
    source,
    event_type,
    title,
    description,
    user_id,
    trading_mode,
    execution_time_ms,
    error_code,
    error_message
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC
LIMIT 1000;

-- Error summary view
CREATE OR REPLACE VIEW error_summary AS
SELECT 
    category,
    source,
    error_code,
    COUNT(*) as error_count,
    MAX(created_at) as last_occurrence,
    array_agg(DISTINCT severity) as severities
FROM activity_logs 
WHERE severity IN ('error', 'critical', 'alert', 'emergency')
AND created_at > NOW() - INTERVAL '7 days'
GROUP BY category, source, error_code
ORDER BY error_count DESC, last_occurrence DESC;

-- Performance metrics view
CREATE OR REPLACE VIEW performance_metrics AS
SELECT 
    source,
    category,
    COUNT(*) as total_activities,
    AVG(execution_time_ms) as avg_execution_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
    ROUND(COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) * 100.0 / COUNT(*), 2) as error_rate_pct
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '1 hour'
AND execution_time_ms IS NOT NULL
GROUP BY source, category
ORDER BY total_activities DESC;

-- User activity overview
CREATE OR REPLACE VIEW user_activity_overview AS
SELECT 
    u.telegram_user_id,
    u.username,
    COUNT(al.*) as total_activities,
    COUNT(DISTINCT DATE(al.created_at)) as active_days,
    MAX(al.created_at) as last_activity,
    COUNT(*) FILTER (WHERE al.severity IN ('error', 'critical', 'alert', 'emergency')) as errors_encountered,
    array_agg(DISTINCT al.category) as categories_used,
    array_agg(DISTINCT al.source) as components_used
FROM users u
LEFT JOIN activity_logs al ON u.telegram_user_id = al.user_id
WHERE al.created_at > NOW() - INTERVAL '30 days' OR al.created_at IS NULL
GROUP BY u.telegram_user_id, u.username
ORDER BY total_activities DESC NULLS LAST;

-- Trading activity summary
CREATE OR REPLACE VIEW trading_activity_summary AS
SELECT 
    trading_mode,
    DATE(created_at) as activity_date,
    COUNT(*) as total_activities,
    COUNT(DISTINCT token_address) as unique_tokens,
    COUNT(*) FILTER (WHERE action = 'execute') as executions,
    COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as errors,
    SUM(amount_usd) as total_volume_usd,
    AVG(execution_time_ms) as avg_execution_time
FROM activity_logs 
WHERE category = 'trading'
AND created_at > NOW() - INTERVAL '30 days'
GROUP BY trading_mode, DATE(created_at)
ORDER BY activity_date DESC, trading_mode;

-- =============================================================================
-- GRANTS AND PERMISSIONS
-- =============================================================================

-- Grant appropriate permissions
GRANT SELECT, INSERT ON activity_logs TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON activity_summaries TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON user_activity_sessions TO rlte_user;
GRANT SELECT ON recent_activity, error_summary, performance_metrics, user_activity_overview, trading_activity_summary TO rlte_user;
GRANT USAGE ON SEQUENCE activity_logs_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE activity_summaries_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE user_activity_sessions_id_seq TO rlte_user;