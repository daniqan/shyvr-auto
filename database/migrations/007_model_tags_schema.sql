-- Migration 007: Model Tags Schema
-- This migration adds version tagging functionality to the model preservation system
-- Run with: psql -d shyvr_rlte -f database/migrations/007_model_tags_schema.sql

-- =============================================================================
-- UP MIGRATION
-- =============================================================================

BEGIN;

-- =============================================================================
-- MODEL TAGS TABLE
-- =============================================================================

-- Create model_tags table for version tagging
CREATE TABLE IF NOT EXISTS model_tags (
    id BIGSERIAL PRIMARY KEY,
    tag_id VARCHAR(255) UNIQUE NOT NULL, -- {model_type}-{tag_name} format
    
    -- Tag identification
    tag_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Reference to preserved model
    preservation_id VARCHAR(255) REFERENCES model_preservation_metadata(preservation_id),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Optional metadata
    description TEXT,
    metadata JSONB,
    
    -- User association
    created_by BIGINT REFERENCES users(telegram_user_id),
    updated_by BIGINT REFERENCES users(telegram_user_id),
    
    -- Constraints
    CONSTRAINT chk_tag_name_format CHECK (tag_name ~ '^[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?$'),
    CONSTRAINT chk_tag_name_length CHECK (LENGTH(tag_name) BETWEEN 1 AND 50),
    CONSTRAINT chk_tag_model_type CHECK (model_type IN ('lstm', 'dqn', 'ensemble', 'custom')),
    CONSTRAINT chk_tag_version_format CHECK (version ~ '^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?(\+[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)*)?$'),
    
    -- Unique tag per model type
    CONSTRAINT uq_model_tags_name_type UNIQUE (tag_name, model_type)
);

-- =============================================================================
-- TAG HISTORY TABLE
-- =============================================================================

-- Create tag history table to track tag movements
CREATE TABLE IF NOT EXISTS model_tag_history (
    id BIGSERIAL PRIMARY KEY,
    tag_id VARCHAR(255) NOT NULL,
    tag_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    
    -- Version tracking
    old_version VARCHAR(50),
    new_version VARCHAR(50) NOT NULL,
    old_preservation_id VARCHAR(255),
    new_preservation_id VARCHAR(255),
    
    -- Change details
    action VARCHAR(20) NOT NULL, -- 'created', 'moved', 'deleted'
    reason TEXT,
    
    -- User and timing
    changed_by BIGINT REFERENCES users(telegram_user_id),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT chk_tag_history_action CHECK (action IN ('created', 'moved', 'deleted'))
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Model tags indexes
CREATE INDEX idx_model_tags_name_type ON model_tags (tag_name, model_type);
CREATE INDEX idx_model_tags_model_type ON model_tags (model_type);
CREATE INDEX idx_model_tags_version ON model_tags (version);
CREATE INDEX idx_model_tags_preservation_id ON model_tags (preservation_id);
CREATE INDEX idx_model_tags_created_at ON model_tags (created_at DESC);
CREATE INDEX idx_model_tags_metadata_gin ON model_tags USING gin (metadata);

-- Tag history indexes
CREATE INDEX idx_model_tag_history_tag_id ON model_tag_history (tag_id);
CREATE INDEX idx_model_tag_history_changed_at ON model_tag_history (changed_at DESC);
CREATE INDEX idx_model_tag_history_action ON model_tag_history (action);

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Function to update updated_at timestamp for tags
CREATE OR REPLACE FUNCTION update_model_tags_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for tag updates
CREATE TRIGGER trigger_model_tags_updated_at
    BEFORE UPDATE ON model_tags
    FOR EACH ROW
    EXECUTE FUNCTION update_model_tags_updated_at();

-- Function to log tag changes
CREATE OR REPLACE FUNCTION log_tag_changes()
RETURNS TRIGGER AS $$
BEGIN
    -- Log tag creation
    IF TG_OP = 'INSERT' THEN
        INSERT INTO model_tag_history (
            tag_id, tag_name, model_type, new_version, 
            new_preservation_id, action, changed_by
        ) VALUES (
            NEW.tag_id, NEW.tag_name, NEW.model_type, NEW.version,
            NEW.preservation_id, 'created', NEW.created_by
        );
        RETURN NEW;
    END IF;
    
    -- Log tag moves (version changes)
    IF TG_OP = 'UPDATE' AND (OLD.version != NEW.version OR OLD.preservation_id != NEW.preservation_id) THEN
        INSERT INTO model_tag_history (
            tag_id, tag_name, model_type, old_version, new_version,
            old_preservation_id, new_preservation_id, action, changed_by
        ) VALUES (
            NEW.tag_id, NEW.tag_name, NEW.model_type, OLD.version, NEW.version,
            OLD.preservation_id, NEW.preservation_id, 'moved', NEW.updated_by
        );
        RETURN NEW;
    END IF;
    
    -- Log tag deletion
    IF TG_OP = 'DELETE' THEN
        INSERT INTO model_tag_history (
            tag_id, tag_name, model_type, old_version,
            old_preservation_id, action
        ) VALUES (
            OLD.tag_id, OLD.tag_name, OLD.model_type, OLD.version,
            OLD.preservation_id, 'deleted'
        );
        RETURN OLD;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for tag change logging
CREATE TRIGGER trigger_log_tag_changes
    AFTER INSERT OR UPDATE OR DELETE ON model_tags
    FOR EACH ROW
    EXECUTE FUNCTION log_tag_changes();

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to resolve tag to version
CREATE OR REPLACE FUNCTION resolve_tag_to_version(
    p_model_type VARCHAR,
    p_tag_name VARCHAR
) RETURNS VARCHAR AS $$
DECLARE
    v_version VARCHAR;
BEGIN
    SELECT version INTO v_version
    FROM model_tags
    WHERE model_type = p_model_type
    AND tag_name = p_tag_name;
    
    RETURN v_version;
END;
$$ LANGUAGE plpgsql;

-- Function to get all tags for a model type
CREATE OR REPLACE FUNCTION get_model_tags(
    p_model_type VARCHAR
) RETURNS TABLE (
    tag_name VARCHAR,
    version VARCHAR,
    preservation_id VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE,
    description TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT mt.tag_name, mt.version, mt.preservation_id, mt.created_at, mt.description
    FROM model_tags mt
    WHERE mt.model_type = p_model_type
    ORDER BY mt.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest stable version (for auto-tagging)
CREATE OR REPLACE FUNCTION get_latest_stable_version(
    p_model_type VARCHAR,
    p_mode VARCHAR DEFAULT 'analysis'
) RETURNS VARCHAR AS $$
DECLARE
    v_version VARCHAR;
BEGIN
    -- Get latest non-prerelease version
    SELECT mpm.version INTO v_version
    FROM model_preservation_metadata mpm
    WHERE mpm.model_type = p_model_type
    AND mpm.mode = p_mode
    AND mpm.state IN ('active', 'preserved')
    AND mpm.version ~ '^v?\d+\.\d+\.\d+$' -- No prerelease suffix
    ORDER BY mpm.preserved_at DESC
    LIMIT 1;
    
    RETURN v_version;
END;
$$ LANGUAGE plpgsql;

-- Function to update standard tags automatically
CREATE OR REPLACE FUNCTION update_standard_tags(
    p_model_type VARCHAR,
    p_new_version VARCHAR,
    p_preservation_id VARCHAR,
    p_mode VARCHAR DEFAULT 'analysis'
) RETURNS VOID AS $$
DECLARE
    v_is_prerelease BOOLEAN;
    v_latest_stable VARCHAR;
BEGIN
    -- Check if new version is prerelease
    v_is_prerelease := p_new_version ~ '-[a-zA-Z0-9]+';
    
    -- Always update 'latest' tag
    INSERT INTO model_tags (tag_id, tag_name, model_type, version, preservation_id)
    VALUES (p_model_type || '-latest', 'latest', p_model_type, p_new_version, p_preservation_id)
    ON CONFLICT (tag_name, model_type)
    DO UPDATE SET 
        version = EXCLUDED.version,
        preservation_id = EXCLUDED.preservation_id,
        updated_at = NOW();
    
    -- Update 'stable' tag only if new version is not prerelease
    IF NOT v_is_prerelease THEN
        INSERT INTO model_tags (tag_id, tag_name, model_type, version, preservation_id)
        VALUES (p_model_type || '-stable', 'stable', p_model_type, p_new_version, p_preservation_id)
        ON CONFLICT (tag_name, model_type)
        DO UPDATE SET 
            version = EXCLUDED.version,
            preservation_id = EXCLUDED.preservation_id,
            updated_at = NOW();
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL STANDARD TAGS (Optional)
-- =============================================================================

-- This section can be used to create initial standard tags
-- Uncomment and modify as needed for your environment

/*
-- Create initial 'latest' tags pointing to current latest versions
INSERT INTO model_tags (tag_id, tag_name, model_type, version, preservation_id, description)
SELECT 
    mpm.model_type || '-latest',
    'latest',
    mpm.model_type,
    mpm.version,
    mpm.preservation_id,
    'Latest version of ' || mpm.model_type || ' model'
FROM (
    SELECT DISTINCT ON (model_type, mode) 
        model_type, version, preservation_id
    FROM model_preservation_metadata
    WHERE state IN ('active', 'preserved')
    ORDER BY model_type, mode, preserved_at DESC
) mpm
ON CONFLICT (tag_name, model_type) DO NOTHING;
*/

COMMIT;

-- =============================================================================
-- DOWN MIGRATION (ROLLBACK)
-- =============================================================================

-- To rollback this migration, run the following:
/*
BEGIN;

-- Drop triggers
DROP TRIGGER IF EXISTS trigger_log_tag_changes ON model_tags;
DROP TRIGGER IF EXISTS trigger_model_tags_updated_at ON model_tags;

-- Drop functions
DROP FUNCTION IF EXISTS update_standard_tags(VARCHAR, VARCHAR, VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS get_latest_stable_version(VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS get_model_tags(VARCHAR);
DROP FUNCTION IF EXISTS resolve_tag_to_version(VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS log_tag_changes();
DROP FUNCTION IF EXISTS update_model_tags_updated_at();

-- Drop tables in reverse order due to foreign key constraints
DROP TABLE IF EXISTS model_tag_history CASCADE;
DROP TABLE IF EXISTS model_tags CASCADE;

COMMIT;
*/