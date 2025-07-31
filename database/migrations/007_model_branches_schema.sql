-- Migration 008: Model Branches Schema
-- Phase 2.1.4: Add branch support for experimental models
--
-- This migration adds branch support to the model preservation system,
-- allowing parallel development of experimental models in isolated branches.

-- ============================================================================
-- BRANCHES TABLE
-- ============================================================================

-- Create branches table to track all branches
CREATE TABLE IF NOT EXISTS model_branches (
    branch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    branch_name VARCHAR(63) NOT NULL UNIQUE,
    source_branch VARCHAR(63),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    model_count INTEGER DEFAULT 0,
    
    -- Constraints
    CONSTRAINT branch_name_format CHECK (
        branch_name ~ '^[a-z0-9]([a-z0-9_-]*[a-z0-9])?$' AND
        LENGTH(branch_name) <= 63
    ),
    CONSTRAINT source_branch_exists CHECK (
        source_branch IS NULL OR source_branch IN (SELECT branch_name FROM model_branches)
    )
);

-- Create index for fast lookups
CREATE INDEX idx_model_branches_name ON model_branches(branch_name);
CREATE INDEX idx_model_branches_active ON model_branches(is_active);

-- Insert default main branch
INSERT INTO model_branches (branch_name, description, created_by)
VALUES ('main', 'Main branch for stable models', 'system')
ON CONFLICT (branch_name) DO NOTHING;

-- ============================================================================
-- UPDATE EXISTING TABLES
-- ============================================================================

-- Add branch column to model_preservation_metadata
ALTER TABLE model_preservation_metadata 
ADD COLUMN IF NOT EXISTS branch VARCHAR(63) DEFAULT 'main' NOT NULL;

-- Add foreign key constraint
ALTER TABLE model_preservation_metadata
ADD CONSTRAINT fk_model_branch 
FOREIGN KEY (branch) REFERENCES model_branches(branch_name);

-- Update uniqueness constraint to include branch
ALTER TABLE model_preservation_metadata 
DROP CONSTRAINT IF EXISTS unique_model_version_mode;

ALTER TABLE model_preservation_metadata
ADD CONSTRAINT unique_model_version_mode_branch 
UNIQUE (model_type, version, mode, branch);

-- Add branch to model_versions table
ALTER TABLE model_versions
ADD COLUMN IF NOT EXISTS branch VARCHAR(63) DEFAULT 'main' NOT NULL;

-- Add foreign key constraint
ALTER TABLE model_versions
ADD CONSTRAINT fk_version_branch
FOREIGN KEY (branch) REFERENCES model_branches(branch_name);

-- Update indexes to include branch
DROP INDEX IF EXISTS idx_model_versions_lookup;
CREATE INDEX idx_model_versions_lookup 
ON model_versions(model_type, branch, created_at DESC);

-- Add branch to model_tags table
ALTER TABLE model_tags
ADD COLUMN IF NOT EXISTS branch VARCHAR(63) DEFAULT 'main' NOT NULL;

-- Add foreign key constraint
ALTER TABLE model_tags
ADD CONSTRAINT fk_tag_branch
FOREIGN KEY (branch) REFERENCES model_branches(branch_name);

-- Update tag uniqueness to be per branch
ALTER TABLE model_tags
DROP CONSTRAINT IF EXISTS unique_tag_per_model_type;

ALTER TABLE model_tags
ADD CONSTRAINT unique_tag_per_model_type_branch
UNIQUE (tag_name, model_type, branch);

-- ============================================================================
-- BRANCH OPERATIONS FUNCTIONS
-- ============================================================================

-- Function to check if branch exists
CREATE OR REPLACE FUNCTION branch_exists(p_branch_name VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM model_branches 
        WHERE branch_name = p_branch_name AND is_active = true
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get branch model count
CREATE OR REPLACE FUNCTION get_branch_model_count(p_branch_name VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count
    FROM model_preservation_metadata
    WHERE branch = p_branch_name
    AND state != 'deleted';
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Function to create a new branch
CREATE OR REPLACE FUNCTION create_branch(
    p_branch_name VARCHAR,
    p_source_branch VARCHAR DEFAULT 'main',
    p_description TEXT DEFAULT NULL,
    p_created_by VARCHAR DEFAULT 'system'
)
RETURNS UUID AS $$
DECLARE
    v_branch_id UUID;
BEGIN
    -- Validate source branch exists
    IF NOT branch_exists(p_source_branch) THEN
        RAISE EXCEPTION 'Source branch % does not exist', p_source_branch;
    END IF;
    
    -- Insert new branch
    INSERT INTO model_branches (branch_name, source_branch, description, created_by)
    VALUES (p_branch_name, p_source_branch, p_description, p_created_by)
    RETURNING branch_id INTO v_branch_id;
    
    RETURN v_branch_id;
END;
$$ LANGUAGE plpgsql;

-- Function to delete a branch
CREATE OR REPLACE FUNCTION delete_branch(p_branch_name VARCHAR, p_force BOOLEAN DEFAULT FALSE)
RETURNS VOID AS $$
DECLARE
    v_model_count INTEGER;
BEGIN
    -- Cannot delete main branch
    IF p_branch_name = 'main' THEN
        RAISE EXCEPTION 'Cannot delete main branch';
    END IF;
    
    -- Check if branch has models
    v_model_count := get_branch_model_count(p_branch_name);
    
    IF v_model_count > 0 AND NOT p_force THEN
        RAISE EXCEPTION 'Branch % contains % models. Use force=true to delete.', 
            p_branch_name, v_model_count;
    END IF;
    
    -- Soft delete the branch
    UPDATE model_branches 
    SET is_active = false 
    WHERE branch_name = p_branch_name;
    
    -- If forced, also soft delete all models in the branch
    IF p_force THEN
        UPDATE model_preservation_metadata
        SET state = 'deleted'
        WHERE branch = p_branch_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- UPDATE EXISTING FUNCTIONS
-- ============================================================================

-- Update the get_versions function to support branch parameter
CREATE OR REPLACE FUNCTION get_model_versions(
    p_model_type VARCHAR,
    p_branch VARCHAR DEFAULT 'main'
)
RETURNS TABLE (
    version VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE,
    model_id UUID,
    state model_state_enum
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mpm.version,
        mpm.created_at,
        mpm.model_id,
        mpm.state
    FROM model_preservation_metadata mpm
    WHERE mpm.model_type = p_model_type
    AND mpm.branch = p_branch
    AND mpm.state != 'deleted'
    ORDER BY mpm.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger to update model count in branches table
CREATE OR REPLACE FUNCTION update_branch_model_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE model_branches 
        SET model_count = model_count + 1
        WHERE branch_name = NEW.branch;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE model_branches 
        SET model_count = model_count - 1
        WHERE branch_name = OLD.branch;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Handle branch change
        IF OLD.branch != NEW.branch THEN
            UPDATE model_branches 
            SET model_count = model_count - 1
            WHERE branch_name = OLD.branch;
            
            UPDATE model_branches 
            SET model_count = model_count + 1
            WHERE branch_name = NEW.branch;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_branch_model_count
AFTER INSERT OR UPDATE OR DELETE ON model_preservation_metadata
FOR EACH ROW
EXECUTE FUNCTION update_branch_model_count();

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Grant permissions to application role (adjust role name as needed)
GRANT SELECT, INSERT, UPDATE ON model_branches TO rlte_app;
GRANT EXECUTE ON FUNCTION branch_exists TO rlte_app;
GRANT EXECUTE ON FUNCTION get_branch_model_count TO rlte_app;
GRANT EXECUTE ON FUNCTION create_branch TO rlte_app;
GRANT EXECUTE ON FUNCTION delete_branch TO rlte_app;
GRANT EXECUTE ON FUNCTION get_model_versions TO rlte_app;

-- ============================================================================
-- MIGRATION METADATA
-- ============================================================================

-- Record migration
INSERT INTO schema_migrations (version, description, applied_at)
VALUES (8, 'Add branch support for model preservation', CURRENT_TIMESTAMP);