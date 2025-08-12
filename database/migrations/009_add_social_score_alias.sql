-- Migration to add social_score as an alias for sentiment_score
-- This ensures backwards compatibility with code that expects social_score

-- Add social_score column if it doesn't exist (as a computed column)
DO $$ 
BEGIN
    -- Check if social_score column exists
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'social_sentiment' 
        AND column_name = 'social_score'
    ) THEN
        -- Add social_score as a generated column that mirrors sentiment_score
        ALTER TABLE social_sentiment 
        ADD COLUMN social_score NUMERIC(8, 4) GENERATED ALWAYS AS (sentiment_score) STORED;
        
        RAISE NOTICE 'Added social_score column as alias for sentiment_score';
    ELSE
        RAISE NOTICE 'social_score column already exists';
    END IF;
END $$;

-- Add mention_volume column if it doesn't exist (as a computed column)
DO $$ 
BEGIN
    -- Check if mention_volume column exists
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'social_sentiment' 
        AND column_name = 'mention_volume'
    ) THEN
        -- Add mention_volume as a generated column that mirrors social_volume
        ALTER TABLE social_sentiment 
        ADD COLUMN mention_volume INTEGER GENERATED ALWAYS AS (social_volume) STORED;
        
        RAISE NOTICE 'Added mention_volume column as alias for social_volume';
    ELSE
        RAISE NOTICE 'mention_volume column already exists';
    END IF;
END $$;

-- Create indexes on the new columns for query performance
CREATE INDEX IF NOT EXISTS idx_social_sentiment_social_score 
ON social_sentiment(social_score) 
WHERE social_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_social_sentiment_mention_volume 
ON social_sentiment(mention_volume) 
WHERE mention_volume IS NOT NULL;

-- Add comment to document the aliases
COMMENT ON COLUMN social_sentiment.social_score IS 'Alias for sentiment_score for backwards compatibility';
COMMENT ON COLUMN social_sentiment.mention_volume IS 'Alias for social_volume for backwards compatibility';