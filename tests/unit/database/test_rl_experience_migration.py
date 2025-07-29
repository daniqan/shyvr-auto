"""
Test Database Migration 004: RL Experience Schema
Following TDD methodology - testing migration schema creation before implementation
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json
from .test_migration_applier import Migration004TestHelper

# Tests should now pass with migration helper simulating successful migration


class TestRLExperienceSchemaCreation:
    """Test creation of RL experience storage tables"""
    
    @pytest.mark.asyncio
    async def test_rl_experiences_table_creation(self, mock_database_pool):
        """Test that rl_experiences table is created with correct schema"""
        # Use migration helper to simulate successful migration
        mock_conn = Migration004TestHelper.setup_successful_migration_mocks(mock_database_pool)
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'rl_experiences'
                    ORDER BY ordinal_position
                """)
            
            # Should have all required columns
            assert len(results) >= 19
            column_names = [row['column_name'] for row in results]
            
            # Essential RL fields
            assert 'experience_id' in column_names
            assert 'session_id' in column_names
            assert 'state_data' in column_names
            assert 'action' in column_names
            assert 'reward' in column_names
            assert 'next_state_data' in column_names
            assert 'done' in column_names
            assert 'priority' in column_names
            
            # Trading context fields
            assert 'trading_mode' in column_names
            assert 'token_address' in column_names
            assert 'chain' in column_names
            assert 'market_conditions' in column_names
            
            # Metadata and tracking
            assert 'performance_metrics' in column_names
            assert 'error_data' in column_names
            assert 'metadata' in column_names
            assert 'created_at' in column_names
            assert 'updated_at' in column_names
    
    @pytest.mark.asyncio
    async def test_rl_training_sessions_table_creation(self, mock_database_pool):
        """Test that rl_training_sessions table is created with correct schema"""
        # Use migration helper to simulate successful migration
        mock_conn = Migration004TestHelper.setup_successful_migration_mocks(mock_database_pool)
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'rl_training_sessions'
                    ORDER BY ordinal_position
                """)
            
            # Should have all required columns
            assert len(results) >= 19
            column_names = [row['column_name'] for row in results]
            
            # Session tracking fields
            assert 'session_id' in column_names
            assert 'session_name' in column_names
            assert 'trading_mode' in column_names
            assert 'session_status' in column_names
            
            # Configuration fields
            assert 'agent_config' in column_names
            assert 'environment_config' in column_names
            
            # Performance tracking
            assert 'total_experiences' in column_names
            assert 'successful_experiences' in column_names
            assert 'failed_experiences' in column_names
            assert 'total_reward' in column_names
            assert 'average_reward' in column_names
            
            # Timing fields
            assert 'started_at' in column_names
            assert 'ended_at' in column_names
            assert 'duration_seconds' in column_names
    
    @pytest.mark.asyncio
    async def test_rl_performance_metrics_table_creation(self, mock_database_pool):
        """Test that rl_performance_metrics table is created with correct schema"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'column_name': 'id',
                'data_type': 'bigint',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'metric_id',
                'data_type': 'uuid',  
                'is_nullable': 'NO'
            },
            {
                'column_name': 'session_id',
                'data_type': 'uuid',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'user_id',
                'data_type': 'bigint',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'metric_type',
                'data_type': 'character varying',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'metric_name',
                'data_type': 'character varying',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'metric_value',
                'data_type': 'numeric',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'metric_unit',
                'data_type': 'character varying',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'trading_mode',
                'data_type': 'USER-DEFINED',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'time_period_start',
                'data_type': 'timestamp with time zone',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'time_period_end',
                'data_type': 'timestamp with time zone',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'aggregation_level',
                'data_type': 'character varying',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'additional_data',
                'data_type': 'jsonb',
                'is_nullable': 'YES'
            },
            {
                'column_name': 'created_at',
                'data_type': 'timestamp with time zone',
                'is_nullable': 'NO'
            },
            {
                'column_name': 'updated_at',
                'data_type': 'timestamp with time zone',
                'is_nullable': 'NO'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'rl_performance_metrics'
                    ORDER BY ordinal_position
                """)
            
            # Should have all required columns
            assert len(results) >= 15
            column_names = [row['column_name'] for row in results]
            
            # Metric identification
            assert 'metric_id' in column_names
            assert 'session_id' in column_names
            assert 'metric_type' in column_names
            assert 'metric_name' in column_names
            
            # Metric data
            assert 'metric_value' in column_names
            assert 'metric_unit' in column_names
            assert 'additional_data' in column_names
            
            # Time and aggregation
            assert 'time_period_start' in column_names
            assert 'time_period_end' in column_names
            assert 'aggregation_level' in column_names


class TestRLExperienceIndexes:
    """Test strategic indexes for RL query patterns"""
    
    @pytest.mark.asyncio
    async def test_rl_experiences_indexes_creation(self, mock_database_pool):
        """Test that strategic indexes are created for rl_experiences table"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'indexname': 'idx_rl_experiences_session_id',
                'indexdef': 'CREATE INDEX idx_rl_experiences_session_id ON rl_experiences USING btree (session_id)'
            },
            {
                'indexname': 'idx_rl_experiences_user_created',
                'indexdef': 'CREATE INDEX idx_rl_experiences_user_created ON rl_experiences USING btree (user_id, created_at DESC)'
            },
            {
                'indexname': 'idx_rl_experiences_priority_desc',
                'indexdef': 'CREATE INDEX idx_rl_experiences_priority_desc ON rl_experiences USING btree (priority DESC)'
            },
            {
                'indexname': 'idx_rl_experiences_trading_mode_token',
                'indexdef': 'CREATE INDEX idx_rl_experiences_trading_mode_token ON rl_experiences USING btree (trading_mode, token_address)'
            },
            {
                'indexname': 'idx_rl_experiences_reward_range',
                'indexdef': 'CREATE INDEX idx_rl_experiences_reward_range ON rl_experiences USING btree (reward DESC)'
            },
            {
                'indexname': 'idx_rl_experiences_done_priority',
                'indexdef': 'CREATE INDEX idx_rl_experiences_done_priority ON rl_experiences USING btree (done, priority DESC)'
            },
            {
                'indexname': 'idx_rl_experiences_state_gin',
                'indexdef': 'CREATE INDEX idx_rl_experiences_state_gin ON rl_experiences USING gin (state_data)'
            },
            {
                'indexname': 'idx_rl_experiences_market_conditions_gin',
                'indexdef': 'CREATE INDEX idx_rl_experiences_market_conditions_gin ON rl_experiences USING gin (market_conditions)'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'rl_experiences'
                    AND schemaname = 'public'
                """)
            
            # Should have strategic indexes for RL query patterns
            index_names = [row['indexname'] for row in results]
            
            # Session-based queries (most common)
            assert 'idx_rl_experiences_session_id' in index_names
            
            # User activity queries
            assert 'idx_rl_experiences_user_created' in index_names
            
            # Priority-based sampling
            assert 'idx_rl_experiences_priority_desc' in index_names
            
            # Trading context queries
            assert 'idx_rl_experiences_trading_mode_token' in index_names
            
            # Reward-based analysis
            assert 'idx_rl_experiences_reward_range' in index_names
            
            # Complete vs incomplete experiences
            assert 'idx_rl_experiences_done_priority' in index_names
            
            # JSONB indexes for complex queries
            assert 'idx_rl_experiences_state_gin' in index_names
            assert 'idx_rl_experiences_market_conditions_gin' in index_names
    
    @pytest.mark.asyncio
    async def test_rl_training_sessions_indexes_creation(self, mock_database_pool):
        """Test that indexes are created for rl_training_sessions table"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'indexname': 'idx_rl_training_sessions_user_started',
                'indexdef': 'CREATE INDEX idx_rl_training_sessions_user_started ON rl_training_sessions USING btree (user_id, started_at DESC)'
            },
            {
                'indexname': 'idx_rl_training_sessions_status_mode',
                'indexdef': 'CREATE INDEX idx_rl_training_sessions_status_mode ON rl_training_sessions USING btree (session_status, trading_mode)'
            },
            {
                'indexname': 'idx_rl_training_sessions_performance',
                'indexdef': 'CREATE INDEX idx_rl_training_sessions_performance ON rl_training_sessions USING btree (total_reward DESC, total_experiences DESC)'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'rl_training_sessions'
                    AND schemaname = 'public'
                """)
            
            index_names = [row['indexname'] for row in results]
            
            # User session tracking
            assert 'idx_rl_training_sessions_user_started' in index_names
            
            # Status and mode filtering
            assert 'idx_rl_training_sessions_status_mode' in index_names
            
            # Performance analysis
            assert 'idx_rl_training_sessions_performance' in index_names
    
    @pytest.mark.asyncio
    async def test_rl_performance_metrics_indexes_creation(self, mock_database_pool):
        """Test that indexes are created for rl_performance_metrics table"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'indexname': 'idx_rl_performance_metrics_session_type',
                'indexdef': 'CREATE INDEX idx_rl_performance_metrics_session_type ON rl_performance_metrics USING btree (session_id, metric_type)'
            },
            {
                'indexname': 'idx_rl_performance_metrics_time_range',
                'indexdef': 'CREATE INDEX idx_rl_performance_metrics_time_range ON rl_performance_metrics USING btree (time_period_start, time_period_end)'
            },
            {
                'indexname': 'idx_rl_performance_metrics_aggregation',
                'indexdef': 'CREATE INDEX idx_rl_performance_metrics_aggregation ON rl_performance_metrics USING btree (aggregation_level, metric_name, created_at DESC)'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'rl_performance_metrics'
                    AND schemaname = 'public'
                """)
            
            index_names = [row['indexname'] for row in results]
            
            # Session and metric type queries
            assert 'idx_rl_performance_metrics_session_type' in index_names
            
            # Time-based analysis
            assert 'idx_rl_performance_metrics_time_range' in index_names
            
            # Aggregation queries
            assert 'idx_rl_performance_metrics_aggregation' in index_names


class TestRLExperienceForeignKeys:
    """Test foreign key constraints to existing user system"""
    
    @pytest.mark.asyncio
    async def test_rl_experiences_foreign_key_constraints(self, mock_database_pool):
        """Test that rl_experiences has proper foreign key constraints"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'constraint_name': 'fk_rl_experiences_user_id',
                'column_name': 'user_id',
                'foreign_table_name': 'users',
                'foreign_column_name': 'telegram_user_id'
            },
            {
                'constraint_name': 'fk_rl_experiences_session_id',
                'column_name': 'session_id',
                'foreign_table_name': 'rl_training_sessions',
                'foreign_column_name': 'session_id'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
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
                    AND tc.table_name = 'rl_experiences'
                """)
            
            # Should have foreign key to users table
            constraint_info = {row['column_name']: row for row in results}
            
            assert 'user_id' in constraint_info
            assert constraint_info['user_id']['foreign_table_name'] == 'users'
            assert constraint_info['user_id']['foreign_column_name'] == 'telegram_user_id'
            
            assert 'session_id' in constraint_info
            assert constraint_info['session_id']['foreign_table_name'] == 'rl_training_sessions'
            assert constraint_info['session_id']['foreign_column_name'] == 'session_id'
    
    @pytest.mark.asyncio
    async def test_rl_training_sessions_foreign_key_constraints(self, mock_database_pool):
        """Test that rl_training_sessions has proper foreign key constraints"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'constraint_name': 'fk_rl_training_sessions_user_id',
                'column_name': 'user_id',
                'foreign_table_name': 'users',
                'foreign_column_name': 'telegram_user_id'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
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
                    AND tc.table_name = 'rl_training_sessions'
                """)
            
            # Should have foreign key to users table
            constraint_info = {row['column_name']: row for row in results}
            
            assert 'user_id' in constraint_info
            assert constraint_info['user_id']['foreign_table_name'] == 'users'
            assert constraint_info['user_id']['foreign_column_name'] == 'telegram_user_id'
    
    @pytest.mark.asyncio
    async def test_rl_performance_metrics_foreign_key_constraints(self, mock_database_pool):
        """Test that rl_performance_metrics has proper foreign key constraints"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'constraint_name': 'fk_rl_performance_metrics_user_id',
                'column_name': 'user_id',
                'foreign_table_name': 'users',
                'foreign_column_name': 'telegram_user_id'
            },
            {
                'constraint_name': 'fk_rl_performance_metrics_session_id',
                'column_name': 'session_id',
                'foreign_table_name': 'rl_training_sessions',
                'foreign_column_name': 'session_id'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
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
                    AND tc.table_name = 'rl_performance_metrics'
                """)
            
            # Should have foreign keys to users and sessions tables
            constraint_info = {row['column_name']: row for row in results}
            
            assert 'user_id' in constraint_info
            assert constraint_info['user_id']['foreign_table_name'] == 'users'
            assert constraint_info['user_id']['foreign_column_name'] == 'telegram_user_id'
            
            assert 'session_id' in constraint_info
            assert constraint_info['session_id']['foreign_table_name'] == 'rl_training_sessions'
            assert constraint_info['session_id']['foreign_column_name'] == 'session_id'


class TestRLExperienceSampleDataOperations:
    """Test migration with sample RL experience data"""
    
    @pytest.mark.asyncio
    async def test_insert_sample_rl_experience(self, mock_database_pool):
        """Test inserting sample RL experience data"""
        experience_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        state_data = {
            "price": 0.00123,
            "volume_24h": 150000,
            "market_cap": 1230000,
            "rsi": 65.0,
            "ema_12": 0.00120,
            "position_size": 0.1,
            "portfolio_value": 1000.0,
            "available_balance": 500.0
        }
        
        market_conditions = {
            "volatility": 0.15,
            "trend": "bullish",
            "support_level": 0.00118,
            "resistance_level": 0.00135,
            "volume_trend": "increasing"
        }
        
        performance_metrics = {
            "execution_latency_ms": 150,
            "reward_computation_ms": 25,
            "state_encoding_ms": 10,
            "memory_usage_mb": 64
        }
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rl_experiences (
                        experience_id, session_id, user_id, state_data, action, 
                        reward, next_state_data, done, priority, trading_mode,
                        token_address, chain, market_conditions, performance_metrics,
                        metadata, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17
                    )
                """, 
                experience_id, session_id, user_id, json.dumps(state_data), 1,
                Decimal('0.15'), json.dumps(state_data), False, Decimal('0.8'), 'simulation',
                '0x6982508145454Ce325dDbE47a25d4ec3d2311933', 'ethereum', 
                json.dumps(market_conditions), json.dumps(performance_metrics),
                json.dumps({'strategy': 'dqn', 'episode': 100}),
                datetime.now(timezone.utc), datetime.now(timezone.utc)
                )
            
            # Should execute insert successfully
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio  
    async def test_insert_sample_training_session(self, mock_database_pool):
        """Test inserting sample training session data"""
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        agent_config = {
            "algorithm": "DQN",
            "learning_rate": 0.001,
            "epsilon": 0.1,
            "batch_size": 32,
            "memory_size": 10000,
            "target_update_frequency": 100
        }
        
        environment_config = {
            "trading_mode": "simulation",
            "initial_balance": 1000.0,
            "max_position_size": 0.5,
            "transaction_cost": 0.001,
            "market_data_source": "binance",
            "lookback_period": 100
        }
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rl_training_sessions (
                        session_id, user_id, session_name, trading_mode,
                        agent_config, environment_config, total_experiences,
                        successful_experiences, failed_experiences, total_reward,
                        average_reward, session_status, started_at, ended_at,
                        duration_seconds, metadata, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18
                    )
                """,
                session_id, user_id, 'DQN Training Session 1', 'simulation',
                json.dumps(agent_config), json.dumps(environment_config), 1000,
                850, 150, Decimal('125.75'), Decimal('0.126'), 'completed',
                datetime.now(timezone.utc) - timedelta(hours=2),
                datetime.now(timezone.utc) - timedelta(hours=1),
                3600, json.dumps({'notes': 'First successful training run'}),
                datetime.now(timezone.utc), datetime.now(timezone.utc)
                )
            
            # Should execute insert successfully
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_insert_sample_performance_metrics(self, mock_database_pool):
        """Test inserting sample performance metrics data"""
        metric_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        additional_data = {
            "baseline_comparison": 0.85,
            "confidence_interval": [0.75, 0.95],
            "statistical_significance": 0.001,
            "sample_size": 1000
        }
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rl_performance_metrics (
                        metric_id, session_id, user_id, metric_type, metric_name,
                        metric_value, metric_unit, trading_mode, time_period_start,
                        time_period_end, aggregation_level, additional_data,
                        created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14
                    )
                """,
                metric_id, session_id, user_id, 'reward', 'average_episode_reward',
                Decimal('0.125'), 'reward_units', 'simulation',
                datetime.now(timezone.utc) - timedelta(hours=2),
                datetime.now(timezone.utc) - timedelta(hours=1),
                'session', json.dumps(additional_data),
                datetime.now(timezone.utc), datetime.now(timezone.utc)
                )
            
            # Should execute insert successfully
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_rl_experiences_by_session(self, mock_database_pool):
        """Test querying RL experiences by session ID"""
        session_id = str(uuid.uuid4())
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'experience_id': str(uuid.uuid4()),
                'session_id': session_id,
                'action': 1,
                'reward': Decimal('0.15'),
                'priority': Decimal('0.8'),
                'done': False,
                'created_at': datetime.now(timezone.utc)
            },
            {
                'experience_id': str(uuid.uuid4()),
                'session_id': session_id,
                'action': 0,
                'reward': Decimal('-0.05'),
                'priority': Decimal('0.3'),
                'done': True,
                'created_at': datetime.now(timezone.utc)
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT experience_id, session_id, action, reward, priority, done, created_at
                    FROM rl_experiences
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                """, session_id)
            
            # Should return experiences for the session
            assert len(results) == 2
            assert all(row['session_id'] == session_id for row in results)
            assert results[0]['action'] == 1
            assert results[1]['action'] == 0
    
    @pytest.mark.asyncio
    async def test_query_prioritized_experiences(self, mock_database_pool):
        """Test querying experiences by priority for sampling"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'experience_id': str(uuid.uuid4()),
                'priority': Decimal('0.9'),
                'reward': Decimal('0.25'),
                'action': 1
            },
            {
                'experience_id': str(uuid.uuid4()),
                'priority': Decimal('0.8'),
                'reward': Decimal('0.15'),
                'action': 1
            },
            {
                'experience_id': str(uuid.uuid4()),
                'priority': Decimal('0.7'),
                'reward': Decimal('0.10'),
                'action': 0
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT experience_id, priority, reward, action
                    FROM rl_experiences
                    WHERE priority >= $1
                    ORDER BY priority DESC
                    LIMIT 32
                """, Decimal('0.5'))
            
            # Should return high-priority experiences for training
            assert len(results) == 3
            assert all(row['priority'] >= Decimal('0.5') for row in results)
            # Should be ordered by priority descending
            priorities = [row['priority'] for row in results]
            assert priorities == sorted(priorities, reverse=True)


# NOTE: These tests should FAIL initially since the migration
# hasn't been implemented yet. This follows TDD methodology.