-- Migration 008: Create Training Data Corpus Schema
-- Purpose: Comprehensive training data infrastructure with lifecycle management
-- Author: RLTE System
-- Date: 2025-01-08

-- Create ENUMs for data lifecycle management
CREATE TYPE data_source_type AS ENUM ('initial', 'simulation', 'live', 'backtest');
CREATE TYPE training_status_type AS ENUM ('untrained', 'in_training', 'trained', 'archived');
CREATE TYPE training_mode_type AS ENUM ('initial', 'incremental', 'fine_tune');
CREATE TYPE processing_status_type AS ENUM ('pending', 'processing', 'completed', 'failed');

-- ==========================================
-- Core Training Data Tables
-- ==========================================

-- 1. OHLCV Data with Source Tracking (Partitioned by data_source and month)
CREATE TABLE crypto_ohlcv (
    id BIGSERIAL,
    token_id VARCHAR(50) NOT NULL,        -- e.g., 'bitcoin', 'ethereum' (from API)
    symbol VARCHAR(20) NOT NULL,          -- e.g., 'BTC', 'ETH' (ticker symbol)
    token_address VARCHAR(255),
    chain VARCHAR(50),
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV data
    open NUMERIC(24, 8) NOT NULL,
    high NUMERIC(24, 8) NOT NULL,
    low NUMERIC(24, 8) NOT NULL,
    close NUMERIC(24, 8) NOT NULL,
    volume NUMERIC(32, 8) NOT NULL,
    
    -- Data lifecycle management
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_status training_status_type NOT NULL DEFAULT 'untrained',
    model_version VARCHAR(50),
    
    -- Metadata
    exchange VARCHAR(50),
    market_cap NUMERIC(32, 2),
    circulating_supply NUMERIC(32, 8),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (id, data_source, timestamp)
) PARTITION BY LIST (data_source);

-- Create partitions for each data source
CREATE TABLE crypto_ohlcv_initial PARTITION OF crypto_ohlcv
    FOR VALUES IN ('initial');
    
CREATE TABLE crypto_ohlcv_simulation PARTITION OF crypto_ohlcv
    FOR VALUES IN ('simulation');
    
CREATE TABLE crypto_ohlcv_live PARTITION OF crypto_ohlcv
    FOR VALUES IN ('live');
    
CREATE TABLE crypto_ohlcv_backtest PARTITION OF crypto_ohlcv
    FOR VALUES IN ('backtest');

-- 2. Technical Features Table
CREATE TABLE crypto_features (
    id BIGSERIAL PRIMARY KEY,
    ohlcv_id BIGINT NOT NULL,
    token_id VARCHAR(50) NOT NULL,        -- Matches crypto_ohlcv.token_id
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Technical indicators (matching code expectations)
    rsi_14 NUMERIC(8, 4),                    -- RSI with 14 period
    macd NUMERIC(24, 8),
    macd_signal NUMERIC(24, 8),
    macd_histogram NUMERIC(24, 8),
    bb_upper NUMERIC(24, 8),                 -- Bollinger bands
    bb_middle NUMERIC(24, 8),
    bb_lower NUMERIC(24, 8),
    volume_sma_20 NUMERIC(32, 8),            -- Volume SMA with 20 period
    
    -- Additional technical indicators
    ema_12 NUMERIC(24, 8),
    ema_26 NUMERIC(24, 8),
    ema_50 NUMERIC(24, 8),
    ema_200 NUMERIC(24, 8),
    sma_20 NUMERIC(24, 8),
    sma_50 NUMERIC(24, 8),
    sma_200 NUMERIC(24, 8),
    volume_ema NUMERIC(32, 8),
    atr NUMERIC(24, 8),
    adx NUMERIC(8, 4),
    cci NUMERIC(12, 4),
    stoch_k NUMERIC(8, 4),
    stoch_d NUMERIC(8, 4),
    williams_r NUMERIC(8, 4),
    obv NUMERIC(32, 8),
    vwap NUMERIC(24, 8),
    
    -- Price action features
    returns_1h NUMERIC(12, 6),               -- Hourly returns
    returns_24h NUMERIC(12, 6),              -- Daily returns
    returns_7d NUMERIC(12, 6),               -- Weekly returns
    volatility_24h NUMERIC(12, 6),           -- Daily volatility
    price_change_1h NUMERIC(12, 6),
    price_change_4h NUMERIC(12, 6),
    price_change_24h NUMERIC(12, 6),
    price_change_7d NUMERIC(12, 6),
    volatility_1h NUMERIC(12, 6),
    high_low_ratio NUMERIC(12, 6),
    close_open_ratio NUMERIC(12, 6),
    
    -- Metadata
    feature_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Data source tracking
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Market Sentiment Table (Fear & Greed Index)
CREATE TABLE market_sentiment (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Fear & Greed data
    fear_greed_index INTEGER CHECK (fear_greed_index >= 0 AND fear_greed_index <= 100),
    fear_greed_classification VARCHAR(20), -- 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    
    -- Additional sentiment metrics
    btc_dominance NUMERIC(6, 2),
    eth_dominance NUMERIC(6, 2),
    total_market_cap NUMERIC(32, 2),
    total_volume_24h NUMERIC(32, 2),
    
    -- Data source tracking
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. DeFi Metrics Table (TVL and DeFi data)
CREATE TABLE defi_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    protocol_name VARCHAR(100) NOT NULL,     -- Name of protocol or 'DeFi Market Wide'
    chain VARCHAR(50) NOT NULL,              -- Chain or 'multi_chain'
    
    -- TVL metrics
    total_value_locked NUMERIC(32, 2),       -- Renamed from 'tvl' to match code
    tvl_change_24h NUMERIC(12, 6),
    tvl_change_7d NUMERIC(12, 6),
    
    -- Optional protocol-specific fields
    token_address VARCHAR(255),              -- Made optional
    protocol_count INTEGER,
    largest_protocol VARCHAR(100),
    protocol_distribution JSONB,
    
    -- Yield metrics
    avg_apy NUMERIC(12, 6),
    max_apy NUMERIC(12, 6),
    staking_ratio NUMERIC(8, 4),
    
    -- Data source tracking
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. On-chain Metrics Table
CREATE TABLE onchain_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    chain VARCHAR(50) NOT NULL,
    
    -- Transaction metrics (matching code expectations)
    transaction_count_24h BIGINT,          -- Changed to BIGINT for large values like Solana
    active_addresses_24h BIGINT,           -- Changed to BIGINT for scalability
    
    -- Optional fields
    token_address VARCHAR(255),            -- Made optional
    unique_addresses BIGINT,               -- Changed to BIGINT
    new_addresses_24h BIGINT,              -- Changed to BIGINT
    
    -- Whale activity
    whale_activity NUMERIC(12, 6),
    whale_transactions BIGINT,             -- Changed to BIGINT
    large_transactions_count BIGINT,       -- Changed to BIGINT
    
    -- Network metrics
    gas_used NUMERIC(32, 8),
    avg_transaction_fee NUMERIC(24, 8),
    network_utilization NUMERIC(8, 4),
    
    -- Token flow
    exchange_inflow NUMERIC(32, 8),
    exchange_outflow NUMERIC(32, 8),
    net_flow NUMERIC(32, 8),
    
    -- Data source tracking
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Social Sentiment Table (LunarCrush data)
CREATE TABLE social_sentiment (
    id BIGSERIAL PRIMARY KEY,
    token_id VARCHAR(50) NOT NULL,        -- e.g., 'bitcoin', 'ethereum' (from API)
    symbol VARCHAR(20),                   -- e.g., 'BTC', 'ETH' (ticker symbol)
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Social metrics (matching code expectations)
    sentiment_score NUMERIC(8, 4),
    twitter_mentions INTEGER,
    reddit_posts INTEGER,
    social_volume_24h INTEGER,
    social_engagement_24h INTEGER,
    bullish_percentage NUMERIC(8, 4),
    bearish_percentage NUMERIC(8, 4),
    
    -- Additional social metrics
    social_volume INTEGER,
    social_engagement INTEGER,
    social_contributors INTEGER,
    social_dominance NUMERIC(8, 4),
    sentiment_absolute NUMERIC(8, 4),
    sentiment_relative NUMERIC(8, 4),
    
    -- LunarCrush specific
    galaxy_score INTEGER,
    alt_rank INTEGER,
    correlation_rank INTEGER,
    
    -- Platform breakdown
    twitter_volume INTEGER,
    reddit_volume INTEGER,
    medium_articles INTEGER,
    youtube_videos INTEGER,
    
    -- Data source tracking
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Token Metadata Table
CREATE TABLE token_metadata (
    id BIGSERIAL PRIMARY KEY,
    token_id VARCHAR(50) NOT NULL,        -- e.g., 'bitcoin', 'ethereum' (from API)
    symbol VARCHAR(20) NOT NULL,          -- e.g., 'BTC', 'ETH' (ticker symbol)
    token_address VARCHAR(255),
    chain VARCHAR(50),
    
    -- Token info
    name VARCHAR(100),
    decimals INTEGER,
    total_supply NUMERIC(32, 8),
    max_supply NUMERIC(32, 8),
    
    -- Categories and tags
    categories TEXT[],
    tags TEXT[],
    sector VARCHAR(50),
    
    -- Links
    website VARCHAR(255),
    whitepaper VARCHAR(255),
    github VARCHAR(255),
    
    -- Update tracking
    last_updated TIMESTAMPTZ,
    update_source VARCHAR(50),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================================
-- Data Management Tables
-- ==========================================

-- 8. Training Corpus Versions
CREATE TABLE training_corpus_versions (
    version_id SERIAL PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL UNIQUE, -- e.g., 'initial_v1.0', 'live_2024_01'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Corpus metadata
    data_source data_source_type NOT NULL,
    sample_count INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    tokens TEXT[] NOT NULL, -- Array of token symbols/addresses
    
    -- Time range
    start_timestamp TIMESTAMPTZ,
    end_timestamp TIMESTAMPTZ,
    
    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    is_immutable BOOLEAN DEFAULT FALSE, -- True for initial corpus
    
    -- Storage location
    storage_path TEXT, -- GCS path
    
    -- Statistics
    statistics JSONB,
    
    created_by VARCHAR(100),
    notes TEXT
);

-- 9. Model Training History
CREATE TABLE model_training_history (
    training_id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'itransformer', 'patchtst', 'timesmixer', 'timesfm'
    corpus_version_id INTEGER REFERENCES training_corpus_versions(version_id),
    
    -- Training details
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_mode training_mode_type NOT NULL,
    
    -- Performance metrics
    performance_metrics JSONB NOT NULL, -- loss, accuracy, etc.
    validation_metrics JSONB,
    test_metrics JSONB,
    
    -- Model storage
    model_checkpoint_path TEXT NOT NULL,
    model_version VARCHAR(50),
    
    -- Training configuration
    training_config JSONB,
    hyperparameters JSONB,
    
    -- Duration and resources
    training_duration_seconds INTEGER,
    gpu_hours NUMERIC(10, 2),
    
    -- Status
    training_status VARCHAR(20),
    error_message TEXT,
    
    created_by VARCHAR(100)
);

-- 10. Continuous Learning Queue
CREATE TABLE continuous_learning_queue (
    queue_id SERIAL PRIMARY KEY,
    data_batch_id VARCHAR(100) NOT NULL UNIQUE,
    
    -- Time range
    collected_from TIMESTAMPTZ NOT NULL,
    collected_to TIMESTAMPTZ NOT NULL,
    
    -- Data source
    data_source data_source_type NOT NULL,
    
    -- Processing status
    processing_status processing_status_type NOT NULL DEFAULT 'pending',
    
    -- Batch metadata
    samples_count INTEGER NOT NULL,
    tokens TEXT[],
    
    -- Processing details
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Results
    training_ids INTEGER[], -- References to model_training_history
    error_message TEXT,
    
    -- Priority
    priority INTEGER DEFAULT 5,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================================
-- Indexes for Performance
-- ==========================================

-- OHLCV indexes
CREATE INDEX idx_crypto_ohlcv_source_status_timestamp 
    ON crypto_ohlcv (data_source, training_status, timestamp DESC);
CREATE INDEX idx_crypto_ohlcv_token_timestamp 
    ON crypto_ohlcv (token_id, timestamp DESC);
CREATE INDEX idx_crypto_ohlcv_symbol_timestamp 
    ON crypto_ohlcv (symbol, timestamp DESC);
CREATE INDEX idx_crypto_ohlcv_training_status 
    ON crypto_ohlcv (training_status);
CREATE INDEX idx_crypto_ohlcv_model_version 
    ON crypto_ohlcv (model_version) WHERE model_version IS NOT NULL;

-- Features indexes
CREATE INDEX idx_crypto_features_ohlcv_id ON crypto_features (ohlcv_id);
CREATE INDEX idx_crypto_features_token_timestamp 
    ON crypto_features (token_id, timestamp DESC);
CREATE INDEX idx_crypto_features_data_source 
    ON crypto_features (data_source);

-- Sentiment indexes
CREATE INDEX idx_market_sentiment_timestamp ON market_sentiment (timestamp DESC);
CREATE INDEX idx_market_sentiment_data_source ON market_sentiment (data_source);

-- DeFi metrics indexes
CREATE INDEX idx_defi_metrics_protocol_timestamp 
    ON defi_metrics (protocol_name, timestamp DESC);
CREATE INDEX idx_defi_metrics_token_timestamp 
    ON defi_metrics (token_address, timestamp DESC) WHERE token_address IS NOT NULL;
CREATE INDEX idx_defi_metrics_data_source ON defi_metrics (data_source);

-- On-chain indexes
CREATE INDEX idx_onchain_metrics_chain_timestamp 
    ON onchain_metrics (chain, timestamp DESC);
CREATE INDEX idx_onchain_metrics_token_timestamp 
    ON onchain_metrics (token_address, timestamp DESC) WHERE token_address IS NOT NULL;
CREATE INDEX idx_onchain_metrics_data_source ON onchain_metrics (data_source);
CREATE INDEX idx_onchain_metrics_whale_activity 
    ON onchain_metrics (whale_activity DESC) WHERE whale_activity > 0;

-- Social sentiment indexes
CREATE INDEX idx_social_sentiment_token_timestamp 
    ON social_sentiment (token_id, timestamp DESC);
CREATE INDEX idx_social_sentiment_symbol_timestamp 
    ON social_sentiment (symbol, timestamp DESC) WHERE symbol IS NOT NULL;
CREATE INDEX idx_social_sentiment_data_source ON social_sentiment (data_source);
CREATE INDEX idx_social_sentiment_galaxy_score 
    ON social_sentiment (galaxy_score DESC) WHERE galaxy_score IS NOT NULL;

-- Training corpus indexes
CREATE INDEX idx_training_corpus_versions_active 
    ON training_corpus_versions (is_active) WHERE is_active = TRUE;
CREATE INDEX idx_training_corpus_versions_source 
    ON training_corpus_versions (data_source);

-- Model training history indexes
CREATE INDEX idx_model_training_history_model_type 
    ON model_training_history (model_type, trained_at DESC);
CREATE INDEX idx_model_training_history_corpus 
    ON model_training_history (corpus_version_id);

-- Queue indexes
CREATE INDEX idx_continuous_learning_queue_status 
    ON continuous_learning_queue (processing_status, priority DESC);
CREATE INDEX idx_continuous_learning_queue_source 
    ON continuous_learning_queue (data_source, collected_from DESC);

-- ==========================================
-- Triggers for Updated Timestamps
-- ==========================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all tables with updated_at
CREATE TRIGGER update_crypto_ohlcv_updated_at BEFORE UPDATE ON crypto_ohlcv 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_crypto_features_updated_at BEFORE UPDATE ON crypto_features 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_market_sentiment_updated_at BEFORE UPDATE ON market_sentiment 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_defi_metrics_updated_at BEFORE UPDATE ON defi_metrics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_onchain_metrics_updated_at BEFORE UPDATE ON onchain_metrics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_social_sentiment_updated_at BEFORE UPDATE ON social_sentiment 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_token_metadata_updated_at BEFORE UPDATE ON token_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_continuous_learning_queue_updated_at BEFORE UPDATE ON continuous_learning_queue 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================
-- Sample Initial Corpus Data
-- ==========================================

-- Insert sample corpus version for initial training
INSERT INTO training_corpus_versions (
    version_name, data_source, sample_count, feature_count,
    tokens, is_active, is_immutable, notes
) VALUES (
    'initial_v1.0', 'initial', 43200, 130,
    ARRAY['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'MATIC', 'AVAX', 'DOT', 'LINK', 'UNI'],
    TRUE, TRUE,
    'Initial training corpus for model initialization - 6 months of data for 10 major tokens'
);

-- ==========================================
-- Permissions
-- ==========================================

-- Grant appropriate permissions (adjust based on your user setup)
-- Note: Uncomment these lines if rlte_app role exists
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO rlte_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rlte_app;

-- ==========================================
-- Comments for Documentation
-- ==========================================

COMMENT ON TABLE crypto_ohlcv IS 'Core OHLCV price data with lifecycle management for training corpus';
COMMENT ON TABLE crypto_features IS 'Technical indicators and features derived from OHLCV data';
COMMENT ON TABLE market_sentiment IS 'Market-wide sentiment indicators including Fear & Greed Index';
COMMENT ON TABLE defi_metrics IS 'DeFi-specific metrics including TVL and protocol data';
COMMENT ON TABLE onchain_metrics IS 'Blockchain on-chain metrics and whale activity';
COMMENT ON TABLE social_sentiment IS 'Social media sentiment and engagement metrics from LunarCrush';
COMMENT ON TABLE training_corpus_versions IS 'Version control for training data corpus';
COMMENT ON TABLE model_training_history IS 'History of model training runs with performance metrics';
COMMENT ON TABLE continuous_learning_queue IS 'Queue for processing new data in continuous learning';

COMMENT ON TYPE data_source_type IS 'Source of data: initial (standardized corpus), simulation (paper trading), live (production), backtest';
COMMENT ON TYPE training_status_type IS 'Training lifecycle status of data samples';
COMMENT ON TYPE training_mode_type IS 'Type of training: initial (from scratch), incremental (online learning), fine_tune';
COMMENT ON TYPE processing_status_type IS 'Processing status for continuous learning queue';