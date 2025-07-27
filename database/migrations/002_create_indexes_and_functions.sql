-- Migration 002: Create Indexes, Functions, Triggers, and Views
-- This migration creates performance optimizations and database functions
-- Run with: psql -d shyvr_rlte -f database/migrations/002_create_indexes_and_functions.sql

BEGIN;

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
-- DATA RETENTION FUNCTIONS
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

COMMIT;