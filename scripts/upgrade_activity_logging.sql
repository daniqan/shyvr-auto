-- Activity Logging Schema Upgrade Script
-- Upgrades the existing system_events table to the comprehensive activity_logs schema
--
-- This script:
-- 1. Creates new enums and types
-- 2. Creates the new activity_logs table structure
-- 3. Migrates existing system_events data
-- 4. Creates supporting tables and views
-- 5. Maintains backward compatibility

-- =============================================================================
-- BACKUP AND PREPARATION
-- =============================================================================

-- Create backup of existing system_events
CREATE TABLE system_events_backup AS SELECT * FROM system_events;

-- Log the upgrade start
INSERT INTO system_events (event_type, details, severity, source, created_at) 
VALUES (
    'schema_upgrade_start',
    '{"upgrade": "activity_logging", "version": "1.0", "backup_created": true}',
    'info',
    'database_migration',
    NOW()
);

-- =============================================================================
-- CREATE NEW ENUMS AND TYPES
-- =============================================================================

-- Activity categories for organized logging
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'activity_category') THEN
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
    END IF;
END$$;

-- Activity severity levels (extend existing if needed)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'activity_severity') THEN
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
    END IF;
END$$;

-- Activity action types for more granular tracking
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'activity_action') THEN
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
    END IF;
END$$;

-- =============================================================================
-- CREATE NEW ACTIVITY LOGS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS activity_logs (
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
    throughput_ops_per_sec DECIMAL(10,2), -- Throughput metric
    
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
-- MIGRATE EXISTING DATA
-- =============================================================================

-- Function to map old severity to new severity
CREATE OR REPLACE FUNCTION map_severity(old_severity VARCHAR) 
RETURNS activity_severity AS $$
BEGIN
    RETURN CASE LOWER(old_severity)
        WHEN 'debug' THEN 'debug'::activity_severity
        WHEN 'info' THEN 'info'::activity_severity
        WHEN 'warning' THEN 'warning'::activity_severity
        WHEN 'error' THEN 'error'::activity_severity
        WHEN 'critical' THEN 'critical'::activity_severity
        ELSE 'info'::activity_severity
    END;
END;
$$ LANGUAGE plpgsql;

-- Function to determine category from event_type and source
CREATE OR REPLACE FUNCTION determine_category(event_type VARCHAR, source VARCHAR, details JSONB)
RETURNS activity_category AS $$
BEGIN
    -- Trading-related events
    IF event_type ILIKE '%trade%' OR event_type ILIKE '%order%' OR event_type ILIKE '%swap%' THEN
        RETURN 'trading'::activity_category;
    END IF;
    
    -- User-related events
    IF event_type ILIKE '%user%' OR event_type ILIKE '%auth%' OR event_type ILIKE '%login%' THEN
        RETURN 'user'::activity_category;
    END IF;
    
    -- ML/RL events
    IF event_type ILIKE '%ml%' OR event_type ILIKE '%rl%' OR event_type ILIKE '%model%' OR event_type ILIKE '%prediction%' THEN
        RETURN 'ml_rl'::activity_category;
    END IF;
    
    -- API events
    IF event_type ILIKE '%api%' OR source ILIKE '%api%' THEN
        RETURN 'api'::activity_category;
    END IF;
    
    -- Security events
    IF event_type ILIKE '%security%' OR event_type ILIKE '%violation%' OR event_type ILIKE '%breach%' THEN
        RETURN 'security'::activity_category;
    END IF;
    
    -- Performance events
    IF event_type ILIKE '%performance%' OR event_type ILIKE '%latency%' OR event_type ILIKE '%timeout%' THEN
        RETURN 'performance'::activity_category;
    END IF;
    
    -- Dashboard events
    IF source ILIKE '%dashboard%' OR event_type ILIKE '%dashboard%' THEN
        RETURN 'dashboard'::activity_category;
    END IF;
    
    -- Default to system
    RETURN 'system'::activity_category;
END;
$$ LANGUAGE plpgsql;

-- Function to determine action from event_type
CREATE OR REPLACE FUNCTION determine_action(event_type VARCHAR, severity VARCHAR)
RETURNS activity_action AS $$
BEGIN
    -- Error conditions
    IF LOWER(severity) IN ('error', 'critical') THEN
        RETURN 'error'::activity_action;
    END IF;
    
    -- Specific actions based on event type
    IF event_type ILIKE '%start%' OR event_type ILIKE '%startup%' THEN
        RETURN 'start'::activity_action;
    ELSIF event_type ILIKE '%stop%' OR event_type ILIKE '%shutdown%' THEN
        RETURN 'stop'::activity_action;
    ELSIF event_type ILIKE '%create%' OR event_type ILIKE '%new%' THEN
        RETURN 'create'::activity_action;
    ELSIF event_type ILIKE '%update%' OR event_type ILIKE '%modify%' THEN
        RETURN 'update'::activity_action;
    ELSIF event_type ILIKE '%delete%' OR event_type ILIKE '%remove%' THEN
        RETURN 'delete'::activity_action;
    ELSIF event_type ILIKE '%execute%' OR event_type ILIKE '%run%' THEN
        RETURN 'execute'::activity_action;
    ELSIF event_type ILIKE '%login%' THEN
        RETURN 'login'::activity_action;
    ELSIF event_type ILIKE '%logout%' THEN
        RETURN 'logout'::activity_action;
    ELSIF event_type ILIKE '%access%' THEN
        RETURN 'access'::activity_action;
    ELSE
        RETURN 'execute'::activity_action;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Migrate existing system_events data
INSERT INTO activity_logs (
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
    metadata,
    error_message,
    version
)
SELECT 
    uuid_generate_v4() as activity_id,
    created_at,
    determine_category(event_type, COALESCE(source, 'system'), details) as category,
    determine_action(event_type, COALESCE(severity, 'info')) as action,
    map_severity(COALESCE(severity, 'info')) as severity,
    COALESCE(source, 'system') as source,
    event_type,
    event_type as title,  -- Use event_type as title initially
    CASE 
        WHEN details IS NOT NULL THEN details->>'message'
        ELSE NULL 
    END as description,
    user_id,
    details as metadata,
    CASE 
        WHEN LOWER(COALESCE(severity, 'info')) IN ('error', 'critical') 
        THEN details->>'error'
        ELSE NULL 
    END as error_message,
    1 as version
FROM system_events
ORDER BY created_at;

-- Clean up migration functions
DROP FUNCTION IF EXISTS map_severity(VARCHAR);
DROP FUNCTION IF EXISTS determine_category(VARCHAR, VARCHAR, JSONB);
DROP FUNCTION IF EXISTS determine_action(VARCHAR, VARCHAR);

-- =============================================================================
-- CREATE SUPPORTING TABLES
-- =============================================================================

-- Activity summaries table for dashboard performance
CREATE TABLE IF NOT EXISTS activity_summaries (
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

-- User activity sessions table
CREATE TABLE IF NOT EXISTS user_activity_sessions (
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
-- CREATE PERFORMANCE INDEXES
-- =============================================================================

-- Primary time-based query indexes (most important)
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at_desc ON activity_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at_category ON activity_logs (created_at DESC, category);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at_severity ON activity_logs (created_at DESC, severity);

-- Category and action filtering
CREATE INDEX IF NOT EXISTS idx_activity_logs_category_action ON activity_logs (category, action);
CREATE INDEX IF NOT EXISTS idx_activity_logs_category_created ON activity_logs (category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_action_created ON activity_logs (action, created_at DESC);

-- Severity-based queries (partial index for performance)
CREATE INDEX IF NOT EXISTS idx_activity_logs_severity_created ON activity_logs (severity, created_at DESC) 
WHERE severity IN ('error', 'critical', 'alert', 'emergency');

-- User and session tracking (partial indexes)
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id_created ON activity_logs (user_id, created_at DESC) 
WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_session_id ON activity_logs (session_id) 
WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_request_id ON activity_logs (request_id) 
WHERE request_id IS NOT NULL;

-- Source and component tracking
CREATE INDEX IF NOT EXISTS idx_activity_logs_source_created ON activity_logs (source, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_event_type ON activity_logs (event_type);
CREATE INDEX IF NOT EXISTS idx_activity_logs_dashboard_component ON activity_logs (dashboard_component) 
WHERE dashboard_component IS NOT NULL;

-- Trading context indexes (partial indexes)
CREATE INDEX IF NOT EXISTS idx_activity_logs_trading_mode ON activity_logs (trading_mode) 
WHERE trading_mode IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_token_chain ON activity_logs (token_address, chain) 
WHERE token_address IS NOT NULL;

-- Error and performance tracking (partial indexes)
CREATE INDEX IF NOT EXISTS idx_activity_logs_error_code ON activity_logs (error_code) 
WHERE error_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_execution_time ON activity_logs (execution_time_ms DESC) 
WHERE execution_time_ms IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_response_time ON activity_logs (response_time_ms DESC) 
WHERE response_time_ms IS NOT NULL;

-- API tracking (partial index)
CREATE INDEX IF NOT EXISTS idx_activity_logs_api_endpoint ON activity_logs (api_endpoint, http_status) 
WHERE api_endpoint IS NOT NULL;

-- Security and risk (partial indexes)
CREATE INDEX IF NOT EXISTS idx_activity_logs_risk_score ON activity_logs (risk_score DESC) 
WHERE risk_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_ip_address ON activity_logs (ip_address) 
WHERE ip_address IS NOT NULL;

-- Hierarchy and correlation (partial indexes)
CREATE INDEX IF NOT EXISTS idx_activity_logs_parent_activity ON activity_logs (parent_activity_id) 
WHERE parent_activity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_logs_correlation_id ON activity_logs (correlation_id) 
WHERE correlation_id IS NOT NULL;

-- GIN indexes for JSONB and array data
CREATE INDEX IF NOT EXISTS idx_activity_logs_metadata_gin ON activity_logs USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_activity_logs_tags_gin ON activity_logs USING GIN (tags);

-- Activity summaries indexes
CREATE INDEX IF NOT EXISTS idx_activity_summaries_period ON activity_summaries (period_start, period_end, period_type);
CREATE INDEX IF NOT EXISTS idx_activity_summaries_category_period ON activity_summaries (category, period_start DESC);

-- User sessions indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_started ON user_activity_sessions (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_activity_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_activity_sessions (last_activity_at DESC) 
WHERE ended_at IS NULL;

-- =============================================================================
-- CREATE FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Include the functions and triggers from the main schema file
\i database/activity_logs_schema.sql

-- =============================================================================
-- CREATE VIEWS FOR DASHBOARD
-- =============================================================================

-- Recent activity view for dashboard (replace any existing)
DROP VIEW IF EXISTS recent_activity CASCADE;
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
DROP VIEW IF EXISTS error_summary CASCADE;
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
DROP VIEW IF EXISTS performance_metrics CASCADE;
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

-- =============================================================================
-- MAINTAIN BACKWARD COMPATIBILITY
-- =============================================================================

-- Create a view that maintains the old system_events interface
CREATE OR REPLACE VIEW system_events_compat AS
SELECT 
    activity_id::varchar as id,
    event_type,
    user_id,
    metadata as details,
    severity::varchar,
    source,
    created_at
FROM activity_logs 
WHERE category = 'system'
ORDER BY created_at DESC;

-- Create a function to insert into activity_logs using the old system_events format
CREATE OR REPLACE FUNCTION insert_system_event(
    p_event_type VARCHAR,
    p_user_id BIGINT DEFAULT NULL,
    p_details JSONB DEFAULT NULL,
    p_severity VARCHAR DEFAULT 'info',
    p_source VARCHAR DEFAULT 'system'
) RETURNS UUID AS $$
DECLARE
    new_activity_id UUID;
BEGIN
    INSERT INTO activity_logs (
        category,
        action,
        source,
        event_type,
        title,
        user_id,
        metadata,
        severity
    ) VALUES (
        'system'::activity_category,
        'execute'::activity_action,
        p_source,
        p_event_type,
        p_event_type,
        p_user_id,
        p_details,
        p_severity::activity_severity
    ) RETURNING activity_id INTO new_activity_id;
    
    RETURN new_activity_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant permissions on new tables and views
GRANT SELECT, INSERT ON activity_logs TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON activity_summaries TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON user_activity_sessions TO rlte_user;
GRANT SELECT ON recent_activity, error_summary, performance_metrics TO rlte_user;
GRANT SELECT ON system_events_compat TO rlte_user;
GRANT USAGE ON SEQUENCE activity_logs_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE activity_summaries_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE user_activity_sessions_id_seq TO rlte_user;
GRANT EXECUTE ON FUNCTION insert_system_event TO rlte_user;

-- =============================================================================
-- VALIDATION AND COMPLETION
-- =============================================================================

-- Validate migration by checking record counts
DO $$
DECLARE
    old_count INTEGER;
    new_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO old_count FROM system_events_backup;
    SELECT COUNT(*) INTO new_count FROM activity_logs WHERE version = 1;
    
    IF new_count != old_count THEN
        RAISE EXCEPTION 'Migration validation failed: expected % records, got %', old_count, new_count;
    END IF;
    
    RAISE NOTICE 'Migration validation successful: % records migrated', new_count;
END$$;

-- Log successful completion
INSERT INTO activity_logs (
    category,
    action,
    source,
    event_type,
    title,
    severity,
    metadata
) VALUES (
    'system'::activity_category,
    'success'::activity_action,
    'database_migration',
    'schema_upgrade_complete',
    'Activity logging schema upgrade completed successfully',
    'info'::activity_severity,
    jsonb_build_object(
        'upgrade', 'activity_logging',
        'version', '1.0',
        'migrated_records', (SELECT COUNT(*) FROM system_events_backup),
        'backup_table', 'system_events_backup',
        'completion_time', NOW()
    )
);

-- Optional: Comment out the DROP statement for safety
-- After validating everything works correctly, you can uncomment this to clean up
-- DROP TABLE system_events_backup;

COMMIT;