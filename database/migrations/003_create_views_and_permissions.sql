-- Migration 003: Create Views and Permissions
-- This migration creates database views for common queries and sets up permissions
-- Run with: psql -d shyvr_rlte -f database/migrations/003_create_views_and_permissions.sql

BEGIN;

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
-- CREATE DATABASE USER AND PERMISSIONS
-- =============================================================================

-- Create application user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rlte_user') THEN
        CREATE ROLE rlte_user WITH LOGIN PASSWORD 'change_me_in_production';
    END IF;
END
$$;

-- Grant appropriate permissions
GRANT SELECT, INSERT ON activity_logs TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON activity_summaries TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON user_activity_sessions TO rlte_user;
GRANT SELECT, INSERT, UPDATE ON users TO rlte_user;

-- Grant permissions on views
GRANT SELECT ON recent_activity TO rlte_user;
GRANT SELECT ON error_summary TO rlte_user; 
GRANT SELECT ON performance_metrics TO rlte_user;
GRANT SELECT ON user_activity_overview TO rlte_user;
GRANT SELECT ON trading_activity_summary TO rlte_user;

-- Grant usage on sequences
GRANT USAGE ON SEQUENCE activity_logs_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE activity_summaries_id_seq TO rlte_user;
GRANT USAGE ON SEQUENCE user_activity_sessions_id_seq TO rlte_user;

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION cleanup_old_activity_logs() TO rlte_user;
GRANT EXECUTE ON FUNCTION generate_activity_summaries(TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE, VARCHAR) TO rlte_user;
GRANT EXECUTE ON FUNCTION update_user_session_activity() TO rlte_user;
GRANT EXECUTE ON FUNCTION set_indexed_timestamp() TO rlte_user;

COMMIT;