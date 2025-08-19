-- Migration: Dual Storage Strategy for High Precision Token Data
-- Purpose: Handle meme coins and micro-value tokens with extreme precision
-- Date: 2025-08-19

-- Step 1: Alter crypto_ohlcv table for higher precision prices
-- Using DOUBLE PRECISION for price fields to handle any scale
ALTER TABLE crypto_ohlcv 
    ALTER COLUMN open TYPE DOUBLE PRECISION,
    ALTER COLUMN high TYPE DOUBLE PRECISION,
    ALTER COLUMN low TYPE DOUBLE PRECISION,
    ALTER COLUMN close TYPE DOUBLE PRECISION,
    ALTER COLUMN volume TYPE DOUBLE PRECISION;

-- Step 2: Alter crypto_features table for calculated fields
-- All calculated features should use DOUBLE PRECISION to prevent overflow
ALTER TABLE crypto_features
    -- Technical indicators
    ALTER COLUMN rsi_14 TYPE DOUBLE PRECISION,
    ALTER COLUMN rsi_7 TYPE DOUBLE PRECISION,
    ALTER COLUMN rsi_21 TYPE DOUBLE PRECISION,
    ALTER COLUMN macd TYPE DOUBLE PRECISION,
    ALTER COLUMN macd_signal TYPE DOUBLE PRECISION,
    ALTER COLUMN macd_histogram TYPE DOUBLE PRECISION,
    ALTER COLUMN bb_upper TYPE DOUBLE PRECISION,
    ALTER COLUMN bb_middle TYPE DOUBLE PRECISION,
    ALTER COLUMN bb_lower TYPE DOUBLE PRECISION,
    ALTER COLUMN bb_position TYPE DOUBLE PRECISION,
    
    -- Moving averages
    ALTER COLUMN volume_sma_20 TYPE DOUBLE PRECISION,
    ALTER COLUMN ema_12 TYPE DOUBLE PRECISION,
    ALTER COLUMN ema_26 TYPE DOUBLE PRECISION,
    ALTER COLUMN ema_50 TYPE DOUBLE PRECISION,
    ALTER COLUMN ema_200 TYPE DOUBLE PRECISION,
    ALTER COLUMN sma_20 TYPE DOUBLE PRECISION,
    ALTER COLUMN sma_50 TYPE DOUBLE PRECISION,
    ALTER COLUMN sma_200 TYPE DOUBLE PRECISION,
    
    -- Volatility and momentum
    ALTER COLUMN atr TYPE DOUBLE PRECISION,
    ALTER COLUMN adx TYPE DOUBLE PRECISION,
    ALTER COLUMN returns_1h TYPE DOUBLE PRECISION,
    ALTER COLUMN returns_24h TYPE DOUBLE PRECISION,
    ALTER COLUMN returns_7d TYPE DOUBLE PRECISION,
    ALTER COLUMN volatility_24h TYPE DOUBLE PRECISION,
    ALTER COLUMN price_change_1h TYPE DOUBLE PRECISION,
    ALTER COLUMN price_change_24h TYPE DOUBLE PRECISION,
    ALTER COLUMN price_change_7d TYPE DOUBLE PRECISION,
    
    -- Additional features
    ALTER COLUMN price_change TYPE DOUBLE PRECISION,
    ALTER COLUMN high_low_ratio TYPE DOUBLE PRECISION,
    ALTER COLUMN close_to_high TYPE DOUBLE PRECISION,
    ALTER COLUMN close_to_low TYPE DOUBLE PRECISION,
    ALTER COLUMN volume_price_ratio TYPE DOUBLE PRECISION,
    ALTER COLUMN momentum_5 TYPE DOUBLE PRECISION,
    ALTER COLUMN momentum_10 TYPE DOUBLE PRECISION,
    ALTER COLUMN momentum_20 TYPE DOUBLE PRECISION,
    ALTER COLUMN volatility TYPE DOUBLE PRECISION,
    ALTER COLUMN volatility_ratio TYPE DOUBLE PRECISION,
    ALTER COLUMN realized_volatility TYPE DOUBLE PRECISION,
    ALTER COLUMN volatility_score TYPE DOUBLE PRECISION,
    ALTER COLUMN volume_score TYPE DOUBLE PRECISION,
    ALTER COLUMN momentum_score TYPE DOUBLE PRECISION,
    ALTER COLUMN price_momentum TYPE DOUBLE PRECISION,
    ALTER COLUMN trend_strength TYPE DOUBLE PRECISION,
    
    -- Time features
    ALTER COLUMN hour_sin TYPE DOUBLE PRECISION,
    ALTER COLUMN hour_cos TYPE DOUBLE PRECISION,
    ALTER COLUMN day_sin TYPE DOUBLE PRECISION,
    ALTER COLUMN day_cos TYPE DOUBLE PRECISION;

-- Step 3: Add precision metadata column to track token precision requirements
ALTER TABLE crypto_ohlcv 
    ADD COLUMN IF NOT EXISTS price_precision INTEGER DEFAULT 8;

-- Step 4: Create index on precision column for efficient filtering
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_precision 
    ON crypto_ohlcv(symbol, price_precision);

-- Step 5: Add comment to document the schema change
COMMENT ON COLUMN crypto_ohlcv.open IS 'Opening price - DOUBLE PRECISION for micro-value token support';
COMMENT ON COLUMN crypto_ohlcv.price_precision IS 'Number of decimal places needed for accurate price representation';
COMMENT ON TABLE crypto_features IS 'Feature table using DOUBLE PRECISION to prevent overflow with micro-value tokens';

-- Step 6: Create a function to validate price precision
CREATE OR REPLACE FUNCTION validate_price_precision(
    price DOUBLE PRECISION,
    symbol VARCHAR
) RETURNS BOOLEAN AS $$
BEGIN
    -- Check for invalid values
    IF price IS NULL OR price < 0 OR price = 'Infinity' OR price = '-Infinity' OR price = 'NaN' THEN
        RETURN FALSE;
    END IF;
    
    -- All valid positive prices are acceptable with DOUBLE PRECISION
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Step 7: Add check constraint to ensure valid prices
ALTER TABLE crypto_ohlcv 
    ADD CONSTRAINT check_valid_prices 
    CHECK (
        validate_price_precision(open, symbol) AND
        validate_price_precision(high, symbol) AND
        validate_price_precision(low, symbol) AND
        validate_price_precision(close, symbol)
    );

-- Step 8: Create view for easy access to high-precision tokens
CREATE OR REPLACE VIEW high_precision_tokens AS
SELECT DISTINCT symbol, MAX(price_precision) as max_precision
FROM crypto_ohlcv
WHERE close < 0.001  -- Tokens with very small prices
GROUP BY symbol
ORDER BY max_precision DESC;

-- Migration complete
-- Note: This migration preserves all existing data while enabling support for 
-- tokens with extreme precision requirements (meme coins, micro-cap tokens, etc.)