-- Migration: Add training-ready features to crypto_features table
-- These features are needed for ML model training

-- Add time-based features
ALTER TABLE crypto_features 
ADD COLUMN IF NOT EXISTS hour INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS day_of_week INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS month INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS quarter INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS is_weekend INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS trading_session INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS hour_sin NUMERIC(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS hour_cos NUMERIC(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS day_sin NUMERIC(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS day_cos NUMERIC(10,6) DEFAULT 0;

-- Add price-based features
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS price_change NUMERIC(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS high_low_ratio NUMERIC(20,8) DEFAULT 1,
ADD COLUMN IF NOT EXISTS close_to_high NUMERIC(20,8) DEFAULT 1,
ADD COLUMN IF NOT EXISTS close_to_low NUMERIC(20,8) DEFAULT 1,
ADD COLUMN IF NOT EXISTS volume_price_ratio NUMERIC(20,8) DEFAULT 0;

-- Add momentum features
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS momentum_5 NUMERIC(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS momentum_10 NUMERIC(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS momentum_20 NUMERIC(20,8) DEFAULT 0;

-- Add multiple RSI periods
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS rsi_7 NUMERIC(10,4) DEFAULT 50,
ADD COLUMN IF NOT EXISTS rsi_21 NUMERIC(10,4) DEFAULT 50;

-- Add Bollinger Band position
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS bb_position NUMERIC(10,6) DEFAULT 0.5;

-- Add volatility features
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS volatility NUMERIC(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS volatility_ratio NUMERIC(20,8) DEFAULT 1,
ADD COLUMN IF NOT EXISTS realized_volatility NUMERIC(20,8) DEFAULT 0;

-- Add ML-ready scores
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS volatility_score NUMERIC(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS volume_score NUMERIC(10,6) DEFAULT 0.5,
ADD COLUMN IF NOT EXISTS momentum_score NUMERIC(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS price_momentum NUMERIC(20,8) DEFAULT 0;

-- Add market regime
ALTER TABLE crypto_features
ADD COLUMN IF NOT EXISTS market_regime INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS trend_strength NUMERIC(10,6) DEFAULT 0;

-- Create index for faster queries on time features
CREATE INDEX IF NOT EXISTS idx_crypto_features_time 
ON crypto_features(token_id, granularity, hour, day_of_week);

-- Create index for market regime queries
CREATE INDEX IF NOT EXISTS idx_crypto_features_regime 
ON crypto_features(token_id, market_regime, timestamp DESC);