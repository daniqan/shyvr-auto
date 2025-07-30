-- Migration 005: Create Model Preservation Schema
-- This migration creates tables for comprehensive model preservation and versioning
-- Run with: psql -d shyvr_rlte -f database/migrations/005_create_model_preservation_schema.sql

BEGIN;

-- =============================================================================
-- MODEL PRESERVATION METADATA TABLE
-- =============================================================================

-- Create enum for model states
CREATE TYPE model_state AS ENUM ('active', 'preserved', 'archived', 'corrupted', 'deleted');

-- Create enum for preservation priority
CREATE TYPE preservation_priority AS ENUM ('critical', 'high', 'normal', 'low');

-- Main model preservation metadata table
CREATE TABLE model_preservation_metadata (
    -- Primary identifiers
    id BIGSERIAL PRIMARY KEY,
    preservation_id VARCHAR(255) UNIQUE NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    
    -- Model identification
    model_type VARCHAR(50) NOT NULL,  -- lstm, dqn, ensemble, custom
    version VARCHAR(50) NOT NULL,      -- v1.0.0, v2.1.3, etc.
    mode VARCHAR(20) NOT NULL,         -- analysis, simulation, live
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    preserved_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Storage information
    checksum VARCHAR(64) NOT NULL,     -- SHA256 hash
    size_bytes BIGINT NOT NULL,
    gcs_path VARCHAR(500),             -- Full GCS path
    
    -- Performance and training info
    performance_metrics JSONB,         -- Model accuracy, loss, etc.
    training_info JSONB,              -- Epochs, learning rate, etc.
    
    -- Preservation metadata
    preservation_reason VARCHAR(255),  -- manual_save, auto_backup, shutdown, etc.
    priority preservation_priority NOT NULL DEFAULT 'normal',
    tags TEXT[],                      -- Array of tags for filtering
    metadata JSONB,                   -- Additional metadata
    
    -- State tracking
    state model_state NOT NULL DEFAULT 'preserved',
    
    -- User association (nullable for system backups)
    user_id BIGINT REFERENCES users(telegram_user_id),
    
    -- Indexes for efficient querying
    CONSTRAINT chk_version_format CHECK (version ~ '^v?\d+\.\d+\.\d+$|^latest$')
);

-- =============================================================================
-- MODEL VERSION HISTORY TABLE
-- =============================================================================

-- Track version history and relationships
CREATE TABLE model_version_history (
    id BIGSERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    mode VARCHAR(20) NOT NULL,
    
    -- Version tracking
    current_version VARCHAR(50) NOT NULL,
    previous_version VARCHAR(50),
    
    -- Version components for auto-increment
    major_version INT NOT NULL DEFAULT 1,
    minor_version INT NOT NULL DEFAULT 0,
    patch_version INT NOT NULL DEFAULT 0,
    
    -- Relationships
    current_preservation_id VARCHAR(255) REFERENCES model_preservation_metadata(preservation_id),
    previous_preservation_id VARCHAR(255) REFERENCES model_preservation_metadata(preservation_id),
    
    -- Change tracking
    change_reason VARCHAR(255),
    changed_by BIGINT REFERENCES users(telegram_user_id),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Unique constraint for model type and mode
    CONSTRAINT uq_model_version_history UNIQUE (model_type, mode)
);

-- =============================================================================
-- MODEL PRESERVATION EVENTS TABLE
-- =============================================================================

-- Track all preservation events for audit trail
CREATE TABLE model_preservation_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,  -- saved, loaded, rolled_back, deleted, etc.
    preservation_id VARCHAR(255) REFERENCES model_preservation_metadata(preservation_id),
    model_type VARCHAR(50) NOT NULL,
    mode VARCHAR(20) NOT NULL,
    
    -- Event metadata
    event_data JSONB,                 -- Event-specific data
    error_message TEXT,               -- Error if event failed
    success BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- User and timing
    user_id BIGINT REFERENCES users(telegram_user_id),
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    duration_ms INTEGER               -- Event duration in milliseconds
);

-- =============================================================================
-- MODEL PERFORMANCE TRACKING TABLE
-- =============================================================================

-- Track model performance over time for rollback decisions
CREATE TABLE model_performance_tracking (
    id BIGSERIAL PRIMARY KEY,
    preservation_id VARCHAR(255) REFERENCES model_preservation_metadata(preservation_id),
    
    -- Performance metrics
    accuracy NUMERIC(5,4),            -- 0.0000 to 1.0000
    loss NUMERIC(10,6),
    prediction_count BIGINT DEFAULT 0,
    successful_predictions BIGINT DEFAULT 0,
    
    -- Trading performance (for RL models)
    total_return NUMERIC(20,8),
    sharpe_ratio NUMERIC(10,4),
    max_drawdown NUMERIC(10,4),
    win_rate NUMERIC(5,4),
    
    -- Resource usage
    inference_time_ms NUMERIC(10,2),
    memory_usage_mb NUMERIC(10,2),
    
    -- Tracking period
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Additional metrics
    custom_metrics JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Model preservation metadata indexes
CREATE INDEX idx_preservation_model_type_mode ON model_preservation_metadata (model_type, mode);
CREATE INDEX idx_preservation_version ON model_preservation_metadata (version);
CREATE INDEX idx_preservation_preserved_at ON model_preservation_metadata (preserved_at DESC);
CREATE INDEX idx_preservation_state ON model_preservation_metadata (state);
CREATE INDEX idx_preservation_priority ON model_preservation_metadata (priority);
CREATE INDEX idx_preservation_tags ON model_preservation_metadata USING gin (tags);
CREATE INDEX idx_preservation_metadata_gin ON model_preservation_metadata USING gin (metadata);

-- Version history indexes
CREATE INDEX idx_version_history_model_type ON model_version_history (model_type);
CREATE INDEX idx_version_history_changed_at ON model_version_history (changed_at DESC);

-- Preservation events indexes
CREATE INDEX idx_preservation_events_type ON model_preservation_events (event_type);
CREATE INDEX idx_preservation_events_preservation_id ON model_preservation_events (preservation_id);
CREATE INDEX idx_preservation_events_occurred_at ON model_preservation_events (occurred_at DESC);

-- Performance tracking indexes
CREATE INDEX idx_performance_preservation_id ON model_performance_tracking (preservation_id);
CREATE INDEX idx_performance_period ON model_performance_tracking (period_start, period_end);

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Trigger to update model_preservation_metadata.updated_at
CREATE OR REPLACE FUNCTION update_model_preservation_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_model_preservation_updated_at
    BEFORE UPDATE ON model_preservation_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_model_preservation_updated_at();

-- Trigger to log preservation events
CREATE OR REPLACE FUNCTION log_preservation_event()
RETURNS TRIGGER AS $$
BEGIN
    -- Log state changes
    IF TG_OP = 'UPDATE' AND OLD.state != NEW.state THEN
        INSERT INTO model_preservation_events (
            event_type, preservation_id, model_type, mode,
            event_data, success
        ) VALUES (
            'state_changed',
            NEW.preservation_id,
            NEW.model_type,
            NEW.mode,
            jsonb_build_object(
                'old_state', OLD.state,
                'new_state', NEW.state,
                'reason', NEW.preservation_reason
            ),
            true
        );
    END IF;
    
    -- Log new preservations
    IF TG_OP = 'INSERT' THEN
        INSERT INTO model_preservation_events (
            event_type, preservation_id, model_type, mode,
            event_data, success
        ) VALUES (
            'model_preserved',
            NEW.preservation_id,
            NEW.model_type,
            NEW.mode,
            jsonb_build_object(
                'version', NEW.version,
                'size_bytes', NEW.size_bytes,
                'priority', NEW.priority
            ),
            true
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_log_preservation_events
    AFTER INSERT OR UPDATE ON model_preservation_metadata
    FOR EACH ROW
    EXECUTE FUNCTION log_preservation_event();

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get latest model version
CREATE OR REPLACE FUNCTION get_latest_model_version(
    p_model_type VARCHAR,
    p_mode VARCHAR
) RETURNS TABLE (
    preservation_id VARCHAR,
    version VARCHAR,
    preserved_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT mpm.preservation_id, mpm.version, mpm.preserved_at
    FROM model_preservation_metadata mpm
    WHERE mpm.model_type = p_model_type
    AND mpm.mode = p_mode
    AND mpm.state = 'preserved'
    ORDER BY mpm.preserved_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old models based on retention policy
CREATE OR REPLACE FUNCTION cleanup_old_models(
    p_retention_days INTEGER,
    p_max_models_per_type INTEGER
) RETURNS INTEGER AS $$
DECLARE
    v_deleted_count INTEGER := 0;
    v_cutoff_date TIMESTAMP WITH TIME ZONE;
BEGIN
    v_cutoff_date := NOW() - INTERVAL '1 day' * p_retention_days;
    
    -- Mark old models as archived
    WITH models_to_archive AS (
        SELECT preservation_id
        FROM (
            SELECT preservation_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_type, mode 
                       ORDER BY preserved_at DESC
                   ) as rn,
                   preserved_at
            FROM model_preservation_metadata
            WHERE state = 'preserved'
        ) ranked
        WHERE (rn > p_max_models_per_type OR preserved_at < v_cutoff_date)
        AND preservation_id NOT IN (
            -- Keep critical priority models longer
            SELECT preservation_id 
            FROM model_preservation_metadata 
            WHERE priority = 'critical'
        )
    )
    UPDATE model_preservation_metadata
    SET state = 'archived'
    WHERE preservation_id IN (SELECT preservation_id FROM models_to_archive);
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA AND CONSTRAINTS
-- =============================================================================

-- Add constraints for data integrity
ALTER TABLE model_preservation_metadata 
    ADD CONSTRAINT chk_preservation_mode 
    CHECK (mode IN ('analysis', 'simulation', 'live'));

ALTER TABLE model_preservation_metadata 
    ADD CONSTRAINT chk_model_type 
    CHECK (model_type IN ('lstm', 'dqn', 'ensemble', 'custom'));

-- Create initial version history entries
INSERT INTO model_version_history (model_type, mode, current_version)
VALUES 
    ('lstm', 'analysis', 'v1.0.0'),
    ('lstm', 'simulation', 'v1.0.0'),
    ('lstm', 'live', 'v1.0.0'),
    ('dqn', 'analysis', 'v1.0.0'),
    ('dqn', 'simulation', 'v1.0.0'),
    ('dqn', 'live', 'v1.0.0'),
    ('ensemble', 'analysis', 'v1.0.0'),
    ('ensemble', 'simulation', 'v1.0.0'),
    ('ensemble', 'live', 'v1.0.0')
ON CONFLICT (model_type, mode) DO NOTHING;

COMMIT;

-- Migration completed successfully
-- Tables created: model_preservation_metadata, model_version_history, 
--                model_preservation_events, model_performance_tracking
-- Indexes optimized for preservation query patterns
-- Triggers for automatic event logging and updates
-- Helper functions for version management and cleanup