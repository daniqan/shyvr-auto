"""
Migration 008 Training Data Corpus TDD Test Suite
Following TDD methodology - tests written BEFORE implementation
Tests use real database connections, no mocks
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
import json
import asyncpg
from unittest.mock import patch


class TestMigration008TrainingDataTDD:
    """TDD tests for Migration 008 - Training Data Corpus Schema"""
    
    @pytest.mark.asyncio
    async def test_crypto_ohlcv_table_exists(self, mock_database_pool):
        """Test that crypto_ohlcv table exists with proper schema"""
        async with mock_database_pool.acquire() as conn:
            # This test should fail initially (TDD - Red phase)
            result = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'crypto_ohlcv'
                ORDER BY ordinal_position
            """)
            
            # Verify essential columns exist
            column_names = [row['column_name'] for row in result]
            assert 'id' in column_names
            assert 'token_symbol' in column_names
            assert 'token_address' in column_names
            assert 'chain' in column_names
            assert 'timestamp' in column_names
            assert 'open' in column_names
            assert 'high' in column_names
            assert 'low' in column_names
            assert 'close' in column_names
            assert 'volume' in column_names
            
            # Data lifecycle management columns
            assert 'data_source' in column_names  # ENUM: 'initial', 'simulation', 'live', 'backtest'
            assert 'collection_timestamp' in column_names
            assert 'training_status' in column_names  # ENUM: 'untrained', 'in_training', 'trained', 'archived'
            assert 'model_version' in column_names
            
            # Verify data types
            columns_dict = {row['column_name']: row for row in result}
            assert columns_dict['data_source']['data_type'] == 'USER-DEFINED'  # ENUM type
            assert columns_dict['training_status']['data_type'] == 'USER-DEFINED'  # ENUM type
            assert columns_dict['volume']['data_type'] == 'numeric'
    
    @pytest.mark.asyncio
    async def test_crypto_features_table_exists(self, mock_database_pool):
        """Test that crypto_features table exists for technical indicators"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'crypto_features'
                ORDER BY ordinal_position
            """)
            
            column_names = [row['column_name'] for row in result]
            
            # Technical indicators
            assert 'rsi' in column_names
            assert 'macd' in column_names
            assert 'bollinger_upper' in column_names
            assert 'bollinger_lower' in column_names
            assert 'ema_12' in column_names
            assert 'ema_26' in column_names
            assert 'volume_sma' in column_names
            
            # Data source tracking
            assert 'data_source' in column_names
            assert 'collection_timestamp' in column_names
    
    @pytest.mark.asyncio
    async def test_market_sentiment_table_exists(self, mock_database_pool):
        """Test market_sentiment table for Fear & Greed Index"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'market_sentiment'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'fear_greed_index' in column_names
            assert 'sentiment_classification' in column_names
            assert 'data_source' in column_names
            assert 'timestamp' in column_names
    
    @pytest.mark.asyncio
    async def test_defi_metrics_table_exists(self, mock_database_pool):
        """Test defi_metrics table for TVL and DeFi data"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'defi_metrics'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'token_address' in column_names
            assert 'tvl' in column_names
            assert 'tvl_change_24h' in column_names
            assert 'protocol_count' in column_names
            assert 'data_source' in column_names
    
    @pytest.mark.asyncio
    async def test_onchain_metrics_table_exists(self, mock_database_pool):
        """Test onchain_metrics table for blockchain data"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'onchain_metrics'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'token_address' in column_names
            assert 'transaction_count' in column_names
            assert 'unique_addresses' in column_names
            assert 'whale_activity' in column_names
            assert 'gas_used' in column_names
            assert 'data_source' in column_names
    
    @pytest.mark.asyncio
    async def test_social_sentiment_table_exists(self, mock_database_pool):
        """Test social_sentiment table for LunarCrush data"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'social_sentiment'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'token_symbol' in column_names
            assert 'social_volume' in column_names
            assert 'social_engagement' in column_names
            assert 'sentiment_score' in column_names
            assert 'galaxy_score' in column_names
            assert 'data_source' in column_names
    
    @pytest.mark.asyncio
    async def test_training_corpus_versions_table(self, mock_database_pool):
        """Test training_corpus_versions table for version management"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'training_corpus_versions'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'version_id' in column_names
            assert 'version_name' in column_names  # e.g., 'initial_v1.0', 'live_2024_01'
            assert 'created_at' in column_names
            assert 'data_source' in column_names  # ENUM: 'initial', 'simulation', 'live'
            assert 'sample_count' in column_names
            assert 'feature_count' in column_names
            assert 'tokens' in column_names  # Array of token addresses
            assert 'is_active' in column_names  # Currently used for training
    
    @pytest.mark.asyncio
    async def test_model_training_history_table(self, mock_database_pool):
        """Test model_training_history table for tracking training"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'model_training_history'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'training_id' in column_names
            assert 'model_type' in column_names  # 'lstm', 'itransformer', etc.
            assert 'corpus_version_id' in column_names  # FK to training_corpus_versions
            assert 'trained_at' in column_names
            assert 'training_mode' in column_names  # ENUM: 'initial', 'incremental', 'fine_tune'
            assert 'performance_metrics' in column_names  # JSONB
            assert 'model_checkpoint_path' in column_names
    
    @pytest.mark.asyncio
    async def test_continuous_learning_queue_table(self, mock_database_pool):
        """Test continuous_learning_queue table for incremental learning"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'continuous_learning_queue'
            """)
            
            column_names = [row['column_name'] for row in result]
            assert 'queue_id' in column_names
            assert 'data_batch_id' in column_names
            assert 'collected_from' in column_names
            assert 'collected_to' in column_names
            assert 'data_source' in column_names  # ENUM: 'simulation', 'live'
            assert 'processing_status' in column_names  # ENUM: 'pending', 'processing', 'completed', 'failed'
            assert 'samples_count' in column_names
    
    @pytest.mark.asyncio
    async def test_data_source_enum_values(self, mock_database_pool):
        """Test that data_source ENUM has correct values"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT enumlabel
                FROM pg_enum
                JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
                WHERE pg_type.typname = 'data_source_type'
            """)
            
            enum_values = [row['enumlabel'] for row in result]
            assert 'initial' in enum_values
            assert 'simulation' in enum_values
            assert 'live' in enum_values
            assert 'backtest' in enum_values
    
    @pytest.mark.asyncio
    async def test_training_status_enum_values(self, mock_database_pool):
        """Test that training_status ENUM has correct values"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT enumlabel
                FROM pg_enum
                JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
                WHERE pg_type.typname = 'training_status_type'
            """)
            
            enum_values = [row['enumlabel'] for row in result]
            assert 'untrained' in enum_values
            assert 'in_training' in enum_values
            assert 'trained' in enum_values
            assert 'archived' in enum_values
    
    @pytest.mark.asyncio
    async def test_partitioning_on_crypto_ohlcv(self, mock_database_pool):
        """Test that crypto_ohlcv is partitioned by data_source and month"""
        async with mock_database_pool.acquire() as conn:
            # Check if table is partitioned
            result = await conn.fetchrow("""
                SELECT 
                    c.relname,
                    c.relkind,
                    p.partstrat,
                    p.partattrs
                FROM pg_class c
                LEFT JOIN pg_partitioned_table p ON c.oid = p.partrelid
                WHERE c.relname = 'crypto_ohlcv'
            """)
            
            assert result is not None
            assert result['relkind'] == 'p'  # 'p' indicates partitioned table
            assert result['partstrat'] is not None  # Should have partition strategy
    
    @pytest.mark.asyncio
    async def test_indexes_for_query_performance(self, mock_database_pool):
        """Test that appropriate indexes exist for query performance"""
        async with mock_database_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'crypto_ohlcv'
            """)
            
            index_names = [row['indexname'] for row in result]
            
            # Critical indexes for query patterns
            assert 'idx_crypto_ohlcv_source_status_timestamp' in index_names
            assert 'idx_crypto_ohlcv_token_timestamp' in index_names
            assert 'idx_crypto_ohlcv_training_status' in index_names
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, mock_database_pool):
        """Test foreign key relationships are properly established"""
        async with mock_database_pool.acquire() as conn:
            # Check FK from model_training_history to training_corpus_versions
            result = await conn.fetch("""
                SELECT
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = 'model_training_history'
            """)
            
            fk_constraints = {row['column_name']: row for row in result}
            assert 'corpus_version_id' in fk_constraints
            assert fk_constraints['corpus_version_id']['foreign_table_name'] == 'training_corpus_versions'
    
    @pytest.mark.asyncio
    async def test_insert_initial_corpus_data(self, mock_database_pool):
        """Test inserting initial corpus data with proper flags"""
        async with mock_database_pool.acquire() as conn:
            # Insert sample OHLCV data marked as initial corpus
            await conn.execute("""
                INSERT INTO crypto_ohlcv (
                    token_symbol, token_address, chain, timestamp,
                    open, high, low, close, volume,
                    data_source, collection_timestamp, training_status
                ) VALUES (
                    'BTC', '0xbtc_address', 'bitcoin', $1,
                    45000.0, 45500.0, 44800.0, 45200.0, 1000000000.0,
                    'initial', $2, 'untrained'
                )
            """, datetime.now(timezone.utc), datetime.now(timezone.utc))
            
            # Verify data was inserted with correct source
            result = await conn.fetchrow("""
                SELECT data_source, training_status
                FROM crypto_ohlcv
                WHERE token_symbol = 'BTC'
                AND data_source = 'initial'
            """)
            
            assert result is not None
            assert result['data_source'] == 'initial'
            assert result['training_status'] == 'untrained'
    
    @pytest.mark.asyncio
    async def test_query_data_by_source(self, mock_database_pool):
        """Test querying data filtered by data_source"""
        async with mock_database_pool.acquire() as conn:
            # Query only initial corpus data
            result = await conn.fetch("""
                SELECT DISTINCT data_source
                FROM crypto_ohlcv
                WHERE data_source = 'initial'
            """)
            
            # Query only simulation data
            sim_result = await conn.fetch("""
                SELECT DISTINCT data_source
                FROM crypto_ohlcv
                WHERE data_source = 'simulation'
            """)
            
            # These queries should work without errors
            assert isinstance(result, list)
            assert isinstance(sim_result, list)
    
    @pytest.mark.asyncio
    async def test_corpus_version_creation(self, mock_database_pool):
        """Test creating a training corpus version"""
        async with mock_database_pool.acquire() as conn:
            # Create a corpus version
            await conn.execute("""
                INSERT INTO training_corpus_versions (
                    version_name, created_at, data_source,
                    sample_count, feature_count, tokens, is_active
                ) VALUES (
                    'initial_v1.0', $1, 'initial',
                    43200, 130, ARRAY['BTC', 'ETH', 'SOL'], true
                )
            """, datetime.now(timezone.utc))
            
            # Verify version was created
            result = await conn.fetchrow("""
                SELECT version_name, data_source, is_active
                FROM training_corpus_versions
                WHERE version_name = 'initial_v1.0'
            """)
            
            assert result is not None
            assert result['version_name'] == 'initial_v1.0'
            assert result['data_source'] == 'initial'
            assert result['is_active'] == True
    
    @pytest.mark.asyncio
    async def test_continuous_learning_queue_operations(self, mock_database_pool):
        """Test continuous learning queue operations"""
        async with mock_database_pool.acquire() as conn:
            # Add batch to queue
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await conn.execute("""
                INSERT INTO continuous_learning_queue (
                    data_batch_id, collected_from, collected_to,
                    data_source, processing_status, samples_count
                ) VALUES (
                    $1, $2, $3, 'simulation', 'pending', 240
                )
            """, batch_id, 
                datetime.now(timezone.utc) - timedelta(hours=24),
                datetime.now(timezone.utc))
            
            # Query pending batches
            result = await conn.fetch("""
                SELECT data_batch_id, processing_status
                FROM continuous_learning_queue
                WHERE processing_status = 'pending'
                ORDER BY collected_from ASC
            """)
            
            assert len(result) > 0
            assert result[0]['processing_status'] == 'pending'


class TestDataIntegrity:
    """Test data integrity and constraints"""
    
    @pytest.mark.asyncio
    async def test_no_data_source_mixing(self, mock_database_pool):
        """Test that data sources remain strictly separated"""
        async with mock_database_pool.acquire() as conn:
            # This query should return no results if data sources are properly separated
            result = await conn.fetch("""
                SELECT cv.version_name, cv.data_source
                FROM training_corpus_versions cv
                WHERE cv.version_name LIKE 'initial%'
                AND cv.data_source != 'initial'
            """)
            
            assert len(result) == 0, "Initial corpus versions should only have 'initial' data source"
    
    @pytest.mark.asyncio
    async def test_training_status_transitions(self, mock_database_pool):
        """Test that training_status follows valid transitions"""
        async with mock_database_pool.acquire() as conn:
            # Insert data with 'untrained' status
            token_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO crypto_ohlcv (
                    id, token_symbol, timestamp, open, high, low, close, volume,
                    data_source, training_status
                ) VALUES ($1, 'TEST', $2, 100, 105, 95, 102, 1000, 'initial', 'untrained')
            """, token_id, datetime.now(timezone.utc))
            
            # Update to 'in_training'
            await conn.execute("""
                UPDATE crypto_ohlcv
                SET training_status = 'in_training'
                WHERE id = $1
            """, token_id)
            
            # Update to 'trained'
            await conn.execute("""
                UPDATE crypto_ohlcv
                SET training_status = 'trained'
                WHERE id = $1
            """, token_id)
            
            # Verify final status
            result = await conn.fetchrow("""
                SELECT training_status
                FROM crypto_ohlcv
                WHERE id = $1
            """, token_id)
            
            assert result['training_status'] == 'trained'


class TestRealAPIIntegration:
    """Test with real API calls - NO MOCKS"""
    
    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_coingecko_data_storage(self, mock_database_pool):
        """Test storing real CoinGecko data in training corpus"""
        # This test will use real CoinGecko API
        # Will be implemented after schema is created
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_lunarcrush_social_data(self, mock_database_pool):
        """Test storing real LunarCrush social sentiment data"""
        # This test will use real LunarCrush API
        # Will be implemented after schema is created
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_defi_llama_tvl_data(self, mock_database_pool):
        """Test storing real DeFiLlama TVL data"""
        # This test will use real DeFiLlama API
        # Will be implemented after schema is created
        pass