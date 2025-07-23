-- Shyvr RLTE Database Initialization Script
-- Creates database schema with extensions for ML/RL trading bot

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create enum types
CREATE TYPE trading_mode AS ENUM ('analysis', 'simulation', 'live');
CREATE TYPE user_role AS ENUM ('admin', 'user', 'blocked');
CREATE TYPE rule_type AS ENUM ('filter', 'dca', 'risk', 'custom');
CREATE TYPE chain_type AS ENUM ('ethereum', 'solana', 'base');

-- Users table (from Shyvr Bot pattern)
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    telegram_user_id BIGINT UNIQUE NOT NULL,
    username VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    command_count INTEGER DEFAULT 0,
    notes TEXT
);

-- Agent rules table
CREATE TABLE IF NOT EXISTS agent_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT REFERENCES users(telegram_user_id),
    prompt_text TEXT NOT NULL,
    parsed_rules JSONB NOT NULL,
    rule_type rule_type NOT NULL,
    active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading outcomes table for RL learning
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(64) NOT NULL,
    chain chain_type NOT NULL,
    features JSONB NOT NULL,
    action INTEGER NOT NULL,
    position_size DECIMAL(20,8),
    entry_price DECIMAL(20,8),
    exit_price DECIMAL(20,8),
    profit_pct DECIMAL(10,4),
    profit_usd DECIMAL(12,2),
    trade_mode trading_mode NOT NULL,
    agent_rules_applied JSONB,
    ml_predictions JSONB,
    execution_latency_ms INTEGER,
    gas_fee_usd DECIMAL(10,4),
    slippage_pct DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE
);

-- ML model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    token_address VARCHAR(64),
    chain chain_type,
    prediction_type VARCHAR(30) NOT NULL, -- 'price_direction', 'pump_probability', etc.
    prediction_value DECIMAL(10,4),
    actual_outcome DECIMAL(10,4),
    accuracy_score DECIMAL(5,4),
    mae DECIMAL(10,6),           -- Mean Absolute Error
    mse DECIMAL(10,6),           -- Mean Squared Error
    training_data_size INTEGER,
    prediction_confidence DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RL agent training sessions
CREATE TABLE IF NOT EXISTS rl_training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(30) NOT NULL,
    hyperparameters JSONB NOT NULL,
    training_episodes INTEGER,
    final_reward DECIMAL(12,4),
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(5,4),
    model_checkpoint_path TEXT,
    training_data_size INTEGER,
    validation_episodes INTEGER,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Token features for ML training
CREATE TABLE IF NOT EXISTS token_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(64) NOT NULL,
    chain chain_type NOT NULL,
    features JSONB NOT NULL,      -- All computed features
    target_price_change DECIMAL(10,4), -- For supervised learning
    target_timeframe VARCHAR(10), -- '5m', '1h', '24h'
    market_cap_usd DECIMAL(20,2),
    volume_24h_usd DECIMAL(20,2),
    holder_count INTEGER,
    liquidity_usd DECIMAL(20,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Token discovery log
CREATE TABLE IF NOT EXISTS token_discovery (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(64) NOT NULL,
    chain chain_type NOT NULL,
    symbol VARCHAR(20),
    name VARCHAR(100),
    discovery_source VARCHAR(50) NOT NULL, -- 'helius', 'etherscan', 'telegram', etc.
    raw_data JSONB,
    passed_filters BOOLEAN DEFAULT false,
    filter_reasons TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio states for simulation
CREATE TABLE IF NOT EXISTS portfolio_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT REFERENCES users(telegram_user_id),
    mode trading_mode NOT NULL,
    total_value_usd DECIMAL(20,2),
    available_balance_usd DECIMAL(20,2),
    positions JSONB, -- Current positions
    daily_pnl_usd DECIMAL(12,2),
    total_pnl_usd DECIMAL(12,2),
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(5,4),
    trade_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System events and audit log
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    user_id BIGINT,
    details JSONB,
    severity VARCHAR(20) DEFAULT 'info', -- 'debug', 'info', 'warning', 'error', 'critical'
    source VARCHAR(50), -- Module that generated the event
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    rate_limit_remaining INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_agent_rules_user ON agent_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_rules_active ON agent_rules(active);
CREATE INDEX IF NOT EXISTS idx_agent_rules_type ON agent_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_agent_rules_expires ON agent_rules(expires_at);

CREATE INDEX IF NOT EXISTS idx_trade_outcomes_token ON trade_outcomes(token_address, chain);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_mode ON trade_outcomes(trade_mode);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_created ON trade_outcomes(created_at);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_profit ON trade_outcomes(profit_pct);

CREATE INDEX IF NOT EXISTS idx_model_performance_type ON model_performance(model_type, prediction_type);
CREATE INDEX IF NOT EXISTS idx_model_performance_token ON model_performance(token_address, chain);
CREATE INDEX IF NOT EXISTS idx_model_performance_created ON model_performance(created_at);

CREATE INDEX IF NOT EXISTS idx_token_features_token ON token_features(token_address, chain);
CREATE INDEX IF NOT EXISTS idx_token_features_created ON token_features(created_at);
CREATE INDEX IF NOT EXISTS idx_token_features_timeframe ON token_features(target_timeframe);

CREATE INDEX IF NOT EXISTS idx_token_discovery_token ON token_discovery(token_address, chain);
CREATE INDEX IF NOT EXISTS idx_token_discovery_source ON token_discovery(discovery_source);
CREATE INDEX IF NOT EXISTS idx_token_discovery_created ON token_discovery(created_at);
CREATE INDEX IF NOT EXISTS idx_token_discovery_filters ON token_discovery(passed_filters);

CREATE INDEX IF NOT EXISTS idx_portfolio_states_user ON portfolio_states(user_id, mode);
CREATE INDEX IF NOT EXISTS idx_portfolio_states_created ON portfolio_states(created_at);

CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events(severity);
CREATE INDEX IF NOT EXISTS idx_system_events_created ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_system_events_user ON system_events(user_id);

CREATE INDEX IF NOT EXISTS idx_api_usage_api ON api_usage(api_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_created ON api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_status ON api_usage(status_code);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_rules_updated_at BEFORE UPDATE ON agent_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW active_agent_rules AS
SELECT * FROM agent_rules 
WHERE active = true 
AND (expires_at IS NULL OR expires_at > NOW())
ORDER BY priority DESC, created_at ASC;

CREATE OR REPLACE VIEW recent_trades AS
SELECT * FROM trade_outcomes 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

CREATE OR REPLACE VIEW model_accuracy_summary AS
SELECT 
    model_type,
    prediction_type,
    COUNT(*) as prediction_count,
    AVG(accuracy_score) as avg_accuracy,
    AVG(mae) as avg_mae,
    AVG(mse) as avg_mse,
    MAX(created_at) as last_prediction
FROM model_performance 
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY model_type, prediction_type
ORDER BY avg_accuracy DESC;

CREATE OR REPLACE VIEW trading_performance_summary AS
SELECT 
    trade_mode,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN profit_pct > 0 THEN 1 END) as winning_trades,
    ROUND(COUNT(CASE WHEN profit_pct > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate_pct,
    AVG(profit_pct) as avg_profit_pct,
    SUM(profit_usd) as total_profit_usd,
    AVG(execution_latency_ms) as avg_latency_ms
FROM trade_outcomes 
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY trade_mode
ORDER BY total_profit_usd DESC;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rlte_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rlte_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rlte_user;