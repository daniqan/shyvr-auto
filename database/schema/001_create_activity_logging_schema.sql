-- Activity Logging Database Schema
-- Comprehensive schema for the Shyvr RLTE activity logging system
-- Based on requirements from test files and activity logger implementation

-- Create enums for activity categorization
CREATE TYPE activity_category AS ENUM (
    'system',
    'trading', 
    'user',
    'ml_rl',
    'security',
    'api',
    'performance',
    'data',
    'configuration',
    'integration',
    'dashboard'
);

CREATE TYPE activity_action AS ENUM (
    'create',
    'read', 
    'update',
    'delete',
    'execute',
    'start',
    'stop',
    'pause',
    'resume',
    'error',
    'alert',
    'login',
    'logout',
    'access',
    'violation',
    'timeout',
    'retry',
    'success',
    'failure'
);

CREATE TYPE activity_severity AS ENUM (
    'trace',
    'debug',
    'info',
    'notice', 
    'warning',
    'error',
    'critical',
    'alert',
    'emergency'
);

CREATE TYPE trading_mode AS ENUM (
    'analysis',
    'simulation',
    'live'
);

CREATE TYPE chain_type AS ENUM (
    'ethereum',
    'solana',
    'base'
);

-- Main activity logs table
CREATE TABLE activity_logs (
    id BIGSERIAL PRIMARY KEY,
    activity_id UUID UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Core activity classification
    category activity_category NOT NULL,
    action activity_action NOT NULL,
    severity activity_severity NOT NULL DEFAULT 'info',
    source VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Context and relationships
    user_id BIGINT, -- References users.telegram_user_id
    session_id UUID,
    request_id UUID,
    parent_activity_id UUID, -- Self-referential FK
    correlation_id VARCHAR(255),
    
    -- Trading-specific context
    trading_mode trading_mode,
    token_address VARCHAR(255),
    chain chain_type,
    
    -- Financial context
    amount_usd NUMERIC(20, 8),
    fee_usd NUMERIC(20, 8),
    
    -- Performance metrics  
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_pct NUMERIC(5, 2),
    
    -- Structured data
    metadata JSONB,
    tags TEXT[],
    
    -- Error context
    error_code VARCHAR(255),
    error_message TEXT,
    stack_trace TEXT,
    
    -- API context
    api_endpoint VARCHAR(255),
    http_method VARCHAR(10),
    http_status INTEGER,
    user_agent TEXT,
    ip_address INET,
    
    -- Additional performance metrics
    response_time_ms INTEGER,
    throughput_ops_per_sec NUMERIC(10, 2),
    
    -- Security context
    security_level VARCHAR(50) NOT NULL DEFAULT 'normal',
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    
    -- Dashboard context
    dashboard_component VARCHAR(255),
    dashboard_action VARCHAR(255),
    
    -- Data integrity and versioning
    checksum VARCHAR(64),
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Indexing and archival timestamps
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archived_at TIMESTAMP WITH TIME ZONE
);

-- Foreign key constraints
ALTER TABLE activity_logs 
ADD CONSTRAINT fk_activity_logs_parent 
FOREIGN KEY (parent_activity_id) REFERENCES activity_logs(activity_id);

-- Create indexes for performance
CREATE INDEX idx_activity_logs_created_at_desc ON activity_logs (created_at DESC);
CREATE INDEX idx_activity_logs_created_at_category ON activity_logs (created_at, category);
CREATE INDEX idx_activity_logs_category_action ON activity_logs (category, action);
CREATE INDEX idx_activity_logs_severity_created ON activity_logs (severity, created_at DESC);
CREATE INDEX idx_activity_logs_user_id_created ON activity_logs (user_id, created_at DESC);
CREATE INDEX idx_activity_logs_session_id ON activity_logs (session_id);
CREATE INDEX idx_activity_logs_source_created ON activity_logs (source, created_at DESC);

-- GIN indexes for JSONB and array columns
CREATE INDEX idx_activity_logs_metadata_gin ON activity_logs USING GIN (metadata);
CREATE INDEX idx_activity_logs_tags_gin ON activity_logs USING GIN (tags);

-- Composite indexes for common query patterns
CREATE INDEX idx_activity_logs_category_severity_created ON activity_logs (category, severity, created_at DESC);
CREATE INDEX idx_activity_logs_trading_mode_created ON activity_logs (trading_mode, created_at DESC) WHERE trading_mode IS NOT NULL;
CREATE INDEX idx_activity_logs_error_severity ON activity_logs (created_at DESC) WHERE severity IN ('error', 'critical', 'alert', 'emergency');

-- Activity summaries table for aggregated metrics
CREATE TABLE activity_summaries (
    id BIGSERIAL PRIMARY KEY,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- 'hour', 'day', 'week', 'month'
    
    -- Aggregation dimensions
    category activity_category,
    source VARCHAR(255),
    user_id BIGINT,
    
    -- Aggregate metrics
    total_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    
    -- Performance aggregates
    avg_execution_time_ms NUMERIC(10, 2),
    max_execution_time_ms INTEGER,
    min_execution_time_ms INTEGER,
    p95_execution_time_ms NUMERIC(10, 2),
    
    -- Trading-specific aggregates
    total_volume_usd NUMERIC(20, 8),
    total_fees_usd NUMERIC(20, 8),
    unique_tokens_count INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for activity summaries
CREATE INDEX idx_activity_summaries_period ON activity_summaries (period_start, period_end, period_type);
CREATE INDEX idx_activity_summaries_category ON activity_summaries (category, period_start DESC);
CREATE INDEX idx_activity_summaries_source ON activity_summaries (source, period_start DESC);

-- User activity sessions table
CREATE TABLE user_activity_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    user_id BIGINT NOT NULL, -- References users.telegram_user_id
    source VARCHAR(255) NOT NULL, -- 'dashboard', 'telegram', 'api', etc.
    
    -- Session metrics
    total_activities INTEGER NOT NULL DEFAULT 0,
    error_activities INTEGER NOT NULL DEFAULT 0,
    categories_accessed TEXT[],
    components_used TEXT[],
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    
    -- Session metadata
    user_agent TEXT,
    ip_address INET,
    metadata JSONB
);

-- Indexes for user sessions
CREATE INDEX idx_user_activity_sessions_user_id ON user_activity_sessions (user_id, started_at DESC);
CREATE INDEX idx_user_activity_sessions_session_id ON user_activity_sessions (session_id);
CREATE INDEX idx_user_activity_sessions_active ON user_activity_sessions (last_activity_at DESC) WHERE ended_at IS NULL;

-- Users table (minimal definition for FK constraints)
-- This assumes a users table exists; if not, it will be created
CREATE TABLE IF NOT EXISTS users (
    telegram_user_id BIGINT PRIMARY KEY,
    username VARCHAR(255),
    first_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT true
);

-- Add foreign key constraints after users table exists
ALTER TABLE activity_logs 
ADD CONSTRAINT fk_activity_logs_user 
FOREIGN KEY (user_id) REFERENCES users(telegram_user_id);

ALTER TABLE user_activity_sessions
ADD CONSTRAINT fk_user_activity_sessions_user
FOREIGN KEY (user_id) REFERENCES users(telegram_user_id);

-- Database functions for automated operations

-- Function to update session activity when new activity logs are inserted
CREATE OR REPLACE FUNCTION update_user_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.session_id IS NOT NULL AND NEW.user_id IS NOT NULL THEN
        INSERT INTO user_activity_sessions (
            session_id, user_id, source, total_activities, last_activity_at
        ) VALUES (
            NEW.session_id, NEW.user_id, NEW.source, 1, NEW.created_at
        ) ON CONFLICT (session_id) DO UPDATE SET
            total_activities = user_activity_sessions.total_activities + 1,
            last_activity_at = NEW.created_at,
            error_activities = CASE 
                WHEN NEW.severity IN ('error', 'critical', 'alert', 'emergency') 
                THEN user_activity_sessions.error_activities + 1
                ELSE user_activity_sessions.error_activities
            END,
            categories_accessed = array(
                SELECT DISTINCT unnest(
                    user_activity_sessions.categories_accessed || ARRAY[NEW.category::text]
                )
            ),
            components_used = array(
                SELECT DISTINCT unnest(
                    user_activity_sessions.components_used || 
                    CASE WHEN NEW.dashboard_component IS NOT NULL 
                         THEN ARRAY[NEW.dashboard_component]
                         ELSE ARRAY[]::text[]
                    END
                )
            );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to set indexed timestamp
CREATE OR REPLACE FUNCTION set_indexed_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.indexed_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old activity logs
CREATE OR REPLACE FUNCTION cleanup_old_activity_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM activity_logs 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL
    AND severity NOT IN ('critical', 'alert', 'emergency')
    AND archived_at IS NULL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to generate activity summaries
CREATE OR REPLACE FUNCTION generate_activity_summaries(
    summary_period VARCHAR DEFAULT 'hour',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    end_time TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    truncate_format TEXT;
    period_start_calc TIMESTAMP WITH TIME ZONE;
    period_end_calc TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Set default time range if not provided
    IF start_time IS NULL THEN
        start_time := DATE_TRUNC(summary_period, NOW() - INTERVAL '1 ' || summary_period);
    END IF;
    
    IF end_time IS NULL THEN
        end_time := DATE_TRUNC(summary_period, NOW());
    END IF;
    
    -- Set truncation format
    truncate_format := summary_period;
    
    -- Generate summaries for each period and category combination
    INSERT INTO activity_summaries (
        period_start, period_end, period_type, category, source,
        total_count, error_count, warning_count, success_count,
        avg_execution_time_ms, max_execution_time_ms, min_execution_time_ms,
        total_volume_usd, total_fees_usd, unique_tokens_count
    )
    SELECT 
        DATE_TRUNC(truncate_format, created_at) as period_start,
        DATE_TRUNC(truncate_format, created_at) + INTERVAL '1 ' || summary_period as period_end,
        summary_period as period_type,
        category,
        source,
        COUNT(*) as total_count,
        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
        COUNT(*) FILTER (WHERE severity = 'warning') as warning_count,
        COUNT(*) FILTER (WHERE severity IN ('info', 'debug', 'trace') AND action IN ('success')) as success_count,
        AVG(execution_time_ms) as avg_execution_time_ms,
        MAX(execution_time_ms) as max_execution_time_ms,
        MIN(execution_time_ms) as min_execution_time_ms,
        SUM(amount_usd) as total_volume_usd,
        SUM(fee_usd) as total_fees_usd,
        COUNT(DISTINCT token_address) as unique_tokens_count
    FROM activity_logs
    WHERE created_at >= start_time 
    AND created_at < end_time
    GROUP BY DATE_TRUNC(truncate_format, created_at), category, source
    ON CONFLICT (period_start, period_end, period_type, category, source) 
    DO UPDATE SET
        total_count = EXCLUDED.total_count,
        error_count = EXCLUDED.error_count,
        warning_count = EXCLUDED.warning_count,
        success_count = EXCLUDED.success_count,
        avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
        max_execution_time_ms = EXCLUDED.max_execution_time_ms,
        min_execution_time_ms = EXCLUDED.min_execution_time_ms,
        total_volume_usd = EXCLUDED.total_volume_usd,
        total_fees_usd = EXCLUDED.total_fees_usd,
        unique_tokens_count = EXCLUDED.unique_tokens_count,
        updated_at = NOW();
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_user_session_activity
    AFTER INSERT ON activity_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_user_session_activity();

CREATE TRIGGER trigger_set_indexed_timestamp
    BEFORE INSERT OR UPDATE ON activity_logs
    FOR EACH ROW
    EXECUTE FUNCTION set_indexed_timestamp();

-- Create materialized views for common queries

-- Recent activity view (last 24 hours)
CREATE MATERIALIZED VIEW recent_activity AS
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
    session_id,
    trading_mode,
    token_address,
    chain,
    amount_usd,
    execution_time_ms,
    error_code,
    error_message
FROM activity_logs
WHERE created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Create unique index on materialized view
CREATE UNIQUE INDEX idx_recent_activity_activity_id ON recent_activity (activity_id);

-- Error summary view
CREATE MATERIALIZED VIEW error_summary AS
SELECT 
    category,
    source,
    error_code,
    COUNT(*) as error_count,
    MAX(created_at) as last_occurrence,
    array_agg(DISTINCT error_message) as error_messages
FROM activity_logs
WHERE severity IN ('error', 'critical', 'alert', 'emergency')
AND created_at >= NOW() - INTERVAL '7 days'
GROUP BY category, source, error_code
ORDER BY error_count DESC;

-- Performance metrics view
CREATE MATERIALIZED VIEW performance_metrics AS
SELECT 
    source,
    category,
    COUNT(*) as total_activities,
    AVG(execution_time_ms) as avg_execution_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
    MAX(execution_time_ms) as max_execution_time,
    COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
    ROUND(
        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) * 100.0 / COUNT(*), 2
    ) as error_rate_pct
FROM activity_logs
WHERE created_at >= NOW() - INTERVAL '24 hours'
AND execution_time_ms IS NOT NULL
GROUP BY source, category
ORDER BY total_activities DESC;

-- User activity overview
CREATE MATERIALIZED VIEW user_activity_overview AS
SELECT 
    u.telegram_user_id,
    u.username,
    COUNT(al.id) as total_activities,
    COUNT(DISTINCT DATE(al.created_at)) as active_days,
    MAX(al.created_at) as last_activity,
    COUNT(*) FILTER (WHERE al.severity IN ('error', 'critical', 'alert', 'emergency')) as errors_encountered,
    array_agg(DISTINCT al.category::text) as categories_used,
    array_agg(DISTINCT al.dashboard_component) FILTER (WHERE al.dashboard_component IS NOT NULL) as components_used
FROM users u
LEFT JOIN activity_logs al ON u.telegram_user_id = al.user_id
WHERE al.created_at >= NOW() - INTERVAL '30 days' OR al.created_at IS NULL
GROUP BY u.telegram_user_id, u.username
ORDER BY total_activities DESC NULLS LAST;

-- Trading activity summary view
CREATE MATERIALIZED VIEW trading_activity_summary AS
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
AND trading_mode IS NOT NULL
AND created_at >= NOW() - INTERVAL '30 days'
GROUP BY trading_mode, DATE(created_at)
ORDER BY activity_date DESC, trading_mode;

-- Create indexes on materialized views
CREATE INDEX idx_error_summary_category ON error_summary (category, error_count DESC);
CREATE INDEX idx_performance_metrics_source ON performance_metrics (source, total_activities DESC);
CREATE INDEX idx_user_activity_overview_user_id ON user_activity_overview (telegram_user_id);
CREATE INDEX idx_trading_activity_summary_date ON trading_activity_summary (activity_date DESC, trading_mode);

-- Permissions and grants
-- Create application user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rlte_user') THEN
        CREATE ROLE rlte_user WITH LOGIN PASSWORD 'secure_password_change_me';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE shyvr_rlte TO rlte_user;
GRANT USAGE ON SCHEMA public TO rlte_user;

-- Table permissions
GRANT SELECT, INSERT ON activity_logs TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON activity_summaries TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON user_activity_sessions TO rlte_user;
GRANT SELECT ON users TO rlte_user;

-- Sequence permissions
GRANT USAGE ON SEQUENCE activity_logs_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE activity_summaries_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE user_activity_sessions_id_seq TO rlte_user;

-- Materialized view permissions
GRANT SELECT ON recent_activity TO rlte_user;
GRANT SELECT ON error_summary TO rlte_user;
GRANT SELECT ON performance_metrics TO rlte_user;
GRANT SELECT ON user_activity_overview TO rlte_user;
GRANT SELECT ON trading_activity_summary TO rlte_user;

-- Function execution permissions
GRANT EXECUTE ON FUNCTION cleanup_old_activity_logs(INTEGER) TO rlte_user;
GRANT EXECUTE ON FUNCTION generate_activity_summaries(VARCHAR, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) TO rlte_user;

-- Comments for documentation
COMMENT ON TABLE activity_logs IS 'Comprehensive activity logging for all system operations';
COMMENT ON TABLE activity_summaries IS 'Pre-aggregated activity metrics for performance';
COMMENT ON TABLE user_activity_sessions IS 'User session tracking and metrics';

COMMENT ON COLUMN activity_logs.activity_id IS 'Unique identifier for each activity';
COMMENT ON COLUMN activity_logs.checksum IS 'Data integrity checksum';
COMMENT ON COLUMN activity_logs.metadata IS 'Flexible JSON metadata storage';
COMMENT ON COLUMN activity_logs.tags IS 'Array of tags for flexible categorization';

-- Refresh materialized views (should be done periodically)
-- This is included as a reminder - implement as scheduled job
-- REFRESH MATERIALIZED VIEW CONCURRENTLY recent_activity;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY error_summary;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY performance_metrics;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY user_activity_overview;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY trading_activity_summary;