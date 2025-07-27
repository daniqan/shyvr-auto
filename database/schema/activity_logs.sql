-- Activity Logs Table Schema
-- Stores all activity logs for the RLTE dashboard and trading system

-- Drop table if exists (for development)
DROP TABLE IF EXISTS activity_logs CASCADE;

-- Create ENUM types for activity categorization
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

-- Create activity_logs table
CREATE TABLE activity_logs (
    -- Primary identification
    activity_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Core activity fields
    category activity_category NOT NULL,
    action activity_action NOT NULL,
    severity activity_severity NOT NULL DEFAULT 'info',
    source VARCHAR(100) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- Context identification
    user_id BIGINT,
    session_id UUID,
    request_id UUID,
    parent_activity_id UUID,
    correlation_id VARCHAR(100),
    
    -- Trading context
    trading_mode trading_mode,
    token_address VARCHAR(100),
    chain chain_type,
    
    -- Financial context
    amount_usd DECIMAL(18, 6),
    fee_usd DECIMAL(18, 6),
    
    -- Technical context
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_pct DECIMAL(5, 2),
    
    -- Metadata and tags
    metadata JSONB,
    tags TEXT[],
    
    -- Error context
    error_code VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    
    -- API context
    api_endpoint VARCHAR(200),
    http_method VARCHAR(10),
    http_status INTEGER,
    user_agent TEXT,
    ip_address INET,
    
    -- Performance metrics
    response_time_ms INTEGER,
    throughput_ops_per_sec DECIMAL(10, 2),
    
    -- Security context
    security_level VARCHAR(50) DEFAULT 'normal',
    risk_score INTEGER,
    
    -- Dashboard context
    dashboard_component VARCHAR(100),
    dashboard_action VARCHAR(100),
    
    -- System fields
    checksum CHAR(64), -- SHA256 hash for data integrity
    version INTEGER DEFAULT 1
);

-- Create indexes for performance
CREATE INDEX idx_activity_logs_created_at ON activity_logs (created_at DESC);
CREATE INDEX idx_activity_logs_category ON activity_logs (category);
CREATE INDEX idx_activity_logs_action ON activity_logs (action);
CREATE INDEX idx_activity_logs_severity ON activity_logs (severity);
CREATE INDEX idx_activity_logs_source ON activity_logs (source);
CREATE INDEX idx_activity_logs_event_type ON activity_logs (event_type);
CREATE INDEX idx_activity_logs_user_id ON activity_logs (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_activity_logs_session_id ON activity_logs (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_activity_logs_trading_mode ON activity_logs (trading_mode) WHERE trading_mode IS NOT NULL;
CREATE INDEX idx_activity_logs_token_address ON activity_logs (token_address) WHERE token_address IS NOT NULL;
CREATE INDEX idx_activity_logs_chain ON activity_logs (chain) WHERE chain IS NOT NULL;
CREATE INDEX idx_activity_logs_error_code ON activity_logs (error_code) WHERE error_code IS NOT NULL;
CREATE INDEX idx_activity_logs_dashboard_component ON activity_logs (dashboard_component) WHERE dashboard_component IS NOT NULL;

-- Create compound indexes for common queries
CREATE INDEX idx_activity_logs_user_category_created ON activity_logs (user_id, category, created_at DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_activity_logs_trading_created ON activity_logs (trading_mode, created_at DESC) WHERE trading_mode IS NOT NULL;
CREATE INDEX idx_activity_logs_severity_created ON activity_logs (severity, created_at DESC) WHERE severity IN ('error', 'critical', 'alert', 'emergency');

-- Create JSONB indexes for metadata queries
CREATE INDEX idx_activity_logs_metadata_gin ON activity_logs USING GIN (metadata) WHERE metadata IS NOT NULL;

-- Create partial indexes for performance
CREATE INDEX idx_activity_logs_recent ON activity_logs (created_at DESC) WHERE created_at > NOW() - INTERVAL '30 days';
CREATE INDEX idx_activity_logs_errors_recent ON activity_logs (created_at DESC) WHERE severity IN ('error', 'critical', 'alert', 'emergency') AND created_at > NOW() - INTERVAL '7 days';

-- Foreign key constraints (if users table exists)
-- ALTER TABLE activity_logs ADD CONSTRAINT fk_activity_logs_user_id FOREIGN KEY (user_id) REFERENCES users(id);
-- ALTER TABLE activity_logs ADD CONSTRAINT fk_activity_logs_parent FOREIGN KEY (parent_activity_id) REFERENCES activity_logs(activity_id);

-- Create views for common queries
CREATE VIEW activity_logs_recent AS
SELECT *
FROM activity_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

CREATE VIEW activity_logs_errors AS
SELECT *
FROM activity_logs
WHERE severity IN ('error', 'critical', 'alert', 'emergency')
ORDER BY created_at DESC;

CREATE VIEW activity_logs_trading AS
SELECT *
FROM activity_logs
WHERE category = 'trading'
ORDER BY created_at DESC;

CREATE VIEW activity_logs_user_actions AS
SELECT *
FROM activity_logs
WHERE category = 'user' AND user_id IS NOT NULL
ORDER BY created_at DESC;

-- Create a function to clean up old logs (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_activity_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM activity_logs
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days
    AND severity NOT IN ('critical', 'alert', 'emergency'); -- Keep critical logs longer
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get activity statistics
CREATE OR REPLACE FUNCTION get_activity_stats(
    start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE(
    category activity_category,
    action activity_action,
    severity activity_severity,
    count BIGINT,
    avg_execution_time_ms NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        al.category,
        al.action,
        al.severity,
        COUNT(*) as count,
        AVG(al.execution_time_ms) as avg_execution_time_ms
    FROM activity_logs al
    WHERE al.created_at BETWEEN start_time AND end_time
    GROUP BY al.category, al.action, al.severity
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;

-- Add table comments
COMMENT ON TABLE activity_logs IS 'Comprehensive activity logging for RLTE dashboard and trading system';
COMMENT ON COLUMN activity_logs.activity_id IS 'Unique identifier for each activity';
COMMENT ON COLUMN activity_logs.category IS 'High-level categorization of the activity';
COMMENT ON COLUMN activity_logs.action IS 'Specific action taken';
COMMENT ON COLUMN activity_logs.severity IS 'Severity level for monitoring and alerting';
COMMENT ON COLUMN activity_logs.source IS 'Component or service that generated the activity';
COMMENT ON COLUMN activity_logs.event_type IS 'Specific type of event within the category';
COMMENT ON COLUMN activity_logs.metadata IS 'Additional structured data in JSON format';
COMMENT ON COLUMN activity_logs.checksum IS 'SHA256 hash for data integrity verification';