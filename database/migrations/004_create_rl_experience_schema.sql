-- Migration 004: Create RL Experience Storage Schema
-- This migration creates comprehensive RL experience storage tables for database-first approach
-- Run with: psql -d shyvr_rlte -f database/migrations/004_create_rl_experience_schema.sql

BEGIN;

-- =============================================================================
-- RL EXPERIENCE STORAGE TABLES
-- =============================================================================

-- Core RL experiences table for storing individual state-action-reward transitions
CREATE TABLE rl_experiences (
    -- Primary identifiers
    id BIGSERIAL PRIMARY KEY,
    experience_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    session_id UUID NOT NULL,
    
    -- User association (nullable for system-generated experiences)
    user_id BIGINT REFERENCES users(telegram_user_id),
    
    -- Core RL data fields
    state_data JSONB NOT NULL,              -- Current state representation (prices, indicators, portfolio)
    action INTEGER NOT NULL,                -- Action taken (0=hold, 1=buy, 2=sell, etc.)
    reward NUMERIC(15,8) NOT NULL,          -- Reward received for this transition
    next_state_data JSONB,                  -- Next state (NULL if terminal/done)
    done BOOLEAN NOT NULL DEFAULT FALSE,    -- Whether episode terminated
    priority NUMERIC(8,6) NOT NULL DEFAULT 0.5, -- Priority for prioritized replay
    
    -- Trading context
    trading_mode trading_mode,              -- Mode when experience was generated
    token_address VARCHAR(64),              -- Token being traded
    chain chain_type,                       -- Blockchain
    
    -- Market context for analysis
    market_conditions JSONB,                -- Market indicators, volatility, trends
    
    -- Performance tracking
    performance_metrics JSONB,              -- Execution latency, memory usage, etc.
    
    -- Error tracking
    error_data JSONB,                       -- Error information if action failed
    
    -- Additional metadata
    metadata JSONB,                         -- Strategy info, episode, step, etc.
    
    -- Timestamps
    created_at TIMESTAMP(6) WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- RL Training Sessions table for tracking training runs
CREATE TABLE rl_training_sessions (
    -- Primary identifiers
    id BIGSERIAL PRIMARY KEY,
    session_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    
    -- User association
    user_id BIGINT REFERENCES users(telegram_user_id),
    
    -- Session metadata
    session_name VARCHAR(255),              -- Human-readable session name
    trading_mode trading_mode NOT NULL,     -- Training mode (simulation/live/analysis)
    
    -- Configuration
    agent_config JSONB NOT NULL,            -- Agent hyperparameters, architecture
    environment_config JSONB NOT NULL,      -- Environment settings, market params
    
    -- Performance tracking
    total_experiences INTEGER NOT NULL DEFAULT 0,
    successful_experiences INTEGER NOT NULL DEFAULT 0,
    failed_experiences INTEGER NOT NULL DEFAULT 0,
    total_reward NUMERIC(20,8) NOT NULL DEFAULT 0,
    average_reward NUMERIC(15,8),           -- Computed average reward
    
    -- Session lifecycle
    session_status VARCHAR(20) NOT NULL DEFAULT 'running', -- running, paused, completed, failed
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,               -- Total session duration
    
    -- Additional metadata
    metadata JSONB,                         -- Notes, model checkpoints, etc.
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- RL Performance Metrics table for analytics and monitoring
CREATE TABLE rl_performance_metrics (
    -- Primary identifiers
    id BIGSERIAL PRIMARY KEY,
    metric_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    session_id UUID NOT NULL REFERENCES rl_training_sessions(session_id),
    
    -- User association
    user_id BIGINT REFERENCES users(telegram_user_id),
    
    -- Metric identification
    metric_type VARCHAR(50) NOT NULL,       -- reward, loss, accuracy, latency, etc.
    metric_name VARCHAR(100) NOT NULL,      -- Specific metric name
    metric_value NUMERIC(20,8) NOT NULL,    -- Metric value
    metric_unit VARCHAR(20),                -- Units (%, ms, USD, etc.)
    
    -- Context
    trading_mode trading_mode,              -- Mode when metric was recorded
    
    -- Time aggregation
    time_period_start TIMESTAMP WITH TIME ZONE,
    time_period_end TIMESTAMP WITH TIME ZONE,
    aggregation_level VARCHAR(20) NOT NULL DEFAULT 'instant', -- instant, episode, session, hourly, daily
    
    -- Additional data
    additional_data JSONB,                  -- Confidence intervals, sample sizes, etc.
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- =============================================================================
-- STRATEGIC INDEXES FOR RL QUERY PATTERNS
-- =============================================================================

-- RL Experiences table indexes
CREATE INDEX idx_rl_experiences_session_id ON rl_experiences (session_id);
CREATE INDEX idx_rl_experiences_user_created ON rl_experiences (user_id, created_at DESC);
CREATE INDEX idx_rl_experiences_priority_desc ON rl_experiences (priority DESC);
CREATE INDEX idx_rl_experiences_trading_mode_token ON rl_experiences (trading_mode, token_address);
CREATE INDEX idx_rl_experiences_reward_range ON rl_experiences (reward DESC);
CREATE INDEX idx_rl_experiences_done_priority ON rl_experiences (done, priority DESC);

-- JSONB indexes for complex state queries
CREATE INDEX idx_rl_experiences_state_gin ON rl_experiences USING gin (state_data);
CREATE INDEX idx_rl_experiences_market_conditions_gin ON rl_experiences USING gin (market_conditions);

-- Training Sessions table indexes
CREATE INDEX idx_rl_training_sessions_user_started ON rl_training_sessions (user_id, started_at DESC);
CREATE INDEX idx_rl_training_sessions_status_mode ON rl_training_sessions (session_status, trading_mode);
CREATE INDEX idx_rl_training_sessions_performance ON rl_training_sessions (total_reward DESC, total_experiences DESC);

-- Performance Metrics table indexes
CREATE INDEX idx_rl_performance_metrics_session_type ON rl_performance_metrics (session_id, metric_type);
CREATE INDEX idx_rl_performance_metrics_time_range ON rl_performance_metrics (time_period_start, time_period_end);
CREATE INDEX idx_rl_performance_metrics_aggregation ON rl_performance_metrics (aggregation_level, metric_name, created_at DESC);

-- =============================================================================
-- FOREIGN KEY CONSTRAINTS
-- =============================================================================

-- Add foreign key from rl_experiences to rl_training_sessions
ALTER TABLE rl_experiences ADD CONSTRAINT fk_rl_experiences_session_id 
    FOREIGN KEY (session_id) REFERENCES rl_training_sessions(session_id) ON DELETE CASCADE;

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Trigger to update rl_experiences.updated_at on row changes
CREATE OR REPLACE FUNCTION update_rl_experience_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_rl_experiences_updated_at
    BEFORE UPDATE ON rl_experiences
    FOR EACH ROW
    EXECUTE FUNCTION update_rl_experience_updated_at();

-- Trigger to update training session stats when experiences are added
CREATE OR REPLACE FUNCTION update_training_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update session statistics on experience insert
    IF TG_OP = 'INSERT' THEN
        UPDATE rl_training_sessions 
        SET 
            total_experiences = total_experiences + 1,
            successful_experiences = CASE 
                WHEN NEW.reward > 0 THEN successful_experiences + 1 
                ELSE successful_experiences 
            END,
            failed_experiences = CASE 
                WHEN NEW.reward <= 0 THEN failed_experiences + 1 
                ELSE failed_experiences 
            END,
            total_reward = total_reward + NEW.reward,
            updated_at = NOW()
        WHERE session_id = NEW.session_id;
        
        -- Update average reward
        UPDATE rl_training_sessions 
        SET average_reward = CASE 
            WHEN total_experiences > 0 THEN total_reward / total_experiences 
            ELSE 0 
        END
        WHERE session_id = NEW.session_id;
        
        RETURN NEW;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_training_session_stats
    AFTER INSERT ON rl_experiences
    FOR EACH ROW
    EXECUTE FUNCTION update_training_session_stats();

-- Trigger to update rl_training_sessions.updated_at
CREATE OR REPLACE FUNCTION update_rl_training_session_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_rl_training_sessions_updated_at
    BEFORE UPDATE ON rl_training_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_rl_training_session_updated_at();

-- Trigger to update rl_performance_metrics.updated_at
CREATE OR REPLACE FUNCTION update_rl_performance_metrics_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_rl_performance_metrics_updated_at
    BEFORE UPDATE ON rl_performance_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_rl_performance_metrics_updated_at();

-- =============================================================================
-- INITIAL CONSTRAINTS AND VALIDATIONS
-- =============================================================================

-- Add constraints for data integrity
ALTER TABLE rl_experiences ADD CONSTRAINT chk_rl_experiences_priority_range 
    CHECK (priority >= 0 AND priority <= 1);

ALTER TABLE rl_experiences ADD CONSTRAINT chk_rl_experiences_action_positive 
    CHECK (action >= 0);

ALTER TABLE rl_training_sessions ADD CONSTRAINT chk_rl_training_sessions_experience_counts 
    CHECK (total_experiences >= 0 AND successful_experiences >= 0 AND failed_experiences >= 0);

ALTER TABLE rl_training_sessions ADD CONSTRAINT chk_rl_training_sessions_status 
    CHECK (session_status IN ('running', 'paused', 'completed', 'failed', 'cancelled'));

ALTER TABLE rl_performance_metrics ADD CONSTRAINT chk_rl_performance_metrics_aggregation 
    CHECK (aggregation_level IN ('instant', 'episode', 'session', 'hourly', 'daily', 'weekly'));

-- =============================================================================
-- SAMPLE DATA FOR TESTING (Optional - can be removed in production)
-- =============================================================================

-- Insert sample training session
INSERT INTO rl_training_sessions (
    session_id, user_id, session_name, trading_mode, agent_config, environment_config,
    total_experiences, successful_experiences, failed_experiences, total_reward,
    session_status, started_at, metadata
) VALUES (
    '550e8400-e29b-41d4-a716-446655440000'::uuid,
    NULL,
    'Sample DQN Training Session',
    'simulation',
    '{"algorithm": "DQN", "learning_rate": 0.001, "epsilon": 0.1, "batch_size": 32}',
    '{"initial_balance": 1000.0, "max_position_size": 0.5, "transaction_cost": 0.001}',
    0, 0, 0, 0,
    'running',
    NOW(),
    '{"notes": "Sample session for migration testing"}'
);

-- Insert sample RL experience
INSERT INTO rl_experiences (
    experience_id, session_id, state_data, action, reward, next_state_data, done, priority,
    trading_mode, token_address, chain, market_conditions, performance_metrics, metadata
) VALUES (
    '660e8400-e29b-41d4-a716-446655440001'::uuid,
    '550e8400-e29b-41d4-a716-446655440000'::uuid,
    '{"price": 0.00123, "volume_24h": 150000, "rsi": 65.0, "position_size": 0.1}',
    1,
    0.15,
    '{"price": 0.00125, "volume_24h": 160000, "rsi": 70.0, "position_size": 0.2}',
    false,
    0.8,
    'simulation',
    '0x6982508145454Ce325dDbE47a25d4ec3d2311933',
    'ethereum',
    '{"volatility": 0.15, "trend": "bullish", "support_level": 0.00118}',
    '{"execution_latency_ms": 150, "memory_usage_mb": 64}',
    '{"strategy": "dqn", "episode": 1, "step": 1}'
);

-- Insert sample performance metric
INSERT INTO rl_performance_metrics (
    metric_id, session_id, metric_type, metric_name, metric_value, metric_unit,
    trading_mode, aggregation_level, additional_data
) VALUES (
    '770e8400-e29b-41d4-a716-446655440002'::uuid,
    '550e8400-e29b-41d4-a716-446655440000'::uuid,
    'reward',
    'average_episode_reward',
    0.125,
    'reward_units',
    'simulation',
    'session',
    '{"baseline_comparison": 0.85, "confidence_interval": [0.75, 0.95]}'
);

COMMIT;

-- Migration completed successfully
-- Tables created: rl_experiences, rl_training_sessions, rl_performance_metrics
-- Indexes optimized for RL query patterns
-- Foreign keys established with existing user system
-- Triggers for automatic stats updates
-- Sample data inserted for testing