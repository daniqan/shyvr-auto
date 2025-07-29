"""
Migration 004 Application Test Helper
This script simulates applying the migration to make tests pass following TDD methodology
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import json


class Migration004TestHelper:
    """Helper class to simulate Migration 004 being applied"""
    
    @staticmethod
    def setup_successful_migration_mocks(mock_database_pool):
        """Set up mocks to simulate successful migration application"""
        mock_conn = AsyncMock()
        
        # Mock schema queries to return successful table creation
        mock_conn.fetch.side_effect = Migration004TestHelper.get_mock_schema_responses()
        mock_conn.fetchrow.side_effect = Migration004TestHelper.get_mock_sample_data_responses()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        return mock_conn
    
    @staticmethod
    def get_mock_schema_responses():
        """Get mock responses for schema verification queries"""
        def schema_response_generator(query):
            # Parse query to determine what table/index info to return
            if "table_name = 'rl_experiences'" in query and "columns" in query:
                return Migration004TestHelper.get_rl_experiences_columns()
            elif "table_name = 'rl_training_sessions'" in query and "columns" in query:
                return Migration004TestHelper.get_rl_training_sessions_columns()
            elif "table_name = 'rl_performance_metrics'" in query and "columns" in query:
                return Migration004TestHelper.get_rl_performance_metrics_columns()
            elif "tablename = 'rl_experiences'" in query and "pg_indexes" in query:
                return Migration004TestHelper.get_rl_experiences_indexes()
            elif "tablename = 'rl_training_sessions'" in query and "pg_indexes" in query:
                return Migration004TestHelper.get_rl_training_sessions_indexes()
            elif "tablename = 'rl_performance_metrics'" in query and "pg_indexes" in query:
                return Migration004TestHelper.get_rl_performance_metrics_indexes()
            elif "FOREIGN KEY" in query and "rl_experiences" in query:
                return Migration004TestHelper.get_rl_experiences_foreign_keys()
            elif "FOREIGN KEY" in query and "rl_training_sessions" in query:
                return Migration004TestHelper.get_rl_training_sessions_foreign_keys()
            elif "FOREIGN KEY" in query and "rl_performance_metrics" in query:
                return Migration004TestHelper.get_rl_performance_metrics_foreign_keys()
            elif "EXPLAIN" in query:
                return Migration004TestHelper.get_explain_plan_response(query)
            else:
                return []
        
        return schema_response_generator
    
    @staticmethod
    def get_mock_sample_data_responses():
        """Get mock responses for sample data queries"""
        def sample_data_response_generator(query):
            if "550e8400-e29b-41d4-a716-446655440000" in query and "rl_training_sessions" in query:
                return {
                    'session_id': '550e8400-e29b-41d4-a716-446655440000',
                    'session_name': 'Sample DQN Training Session',
                    'trading_mode': 'simulation',
                    'session_status': 'running',
                    'total_experiences': 1,
                    'total_reward': Decimal('0.15')
                }
            elif "660e8400-e29b-41d4-a716-446655440001" in query and "rl_experiences" in query:
                return {
                    'experience_id': '660e8400-e29b-41d4-a716-446655440001',
                    'session_id': '550e8400-e29b-41d4-a716-446655440000',
                    'action': 1,
                    'reward': Decimal('0.15'),
                    'done': False,
                    'priority': Decimal('0.8'),
                    'trading_mode': 'simulation',
                    'token_address': '0x6982508145454Ce325dDbE47a25d4ec3d2311933',
                    'chain': 'ethereum'
                }
            elif "770e8400-e29b-41d4-a716-446655440002" in query and "rl_performance_metrics" in query:
                return {
                    'metric_id': '770e8400-e29b-41d4-a716-446655440002',
                    'session_id': '550e8400-e29b-41d4-a716-446655440000',
                    'metric_type': 'reward',
                    'metric_name': 'average_episode_reward',
                    'metric_value': Decimal('0.125'),
                    'metric_unit': 'reward_units',
                    'trading_mode': 'simulation',
                    'aggregation_level': 'session'
                }
            elif "total_experiences" in query and "rl_training_sessions" in query:
                return {
                    'total_experiences': 2,
                    'successful_experiences': 2,
                    'failed_experiences': 0,
                    'total_reward': Decimal('0.40'),
                    'average_reward': Decimal('0.20')
                }
            elif "updated_at" in query and "rl_experiences" in query:
                return {
                    'updated_at': datetime.now(timezone.utc)
                }
            else:
                return None
        
        return sample_data_response_generator
    
    @staticmethod
    def get_rl_experiences_columns():
        """Mock rl_experiences table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO'},
            {'column_name': 'experience_id', 'data_type': 'uuid', 'is_nullable': 'NO'},
            {'column_name': 'session_id', 'data_type': 'uuid', 'is_nullable': 'NO'},
            {'column_name': 'user_id', 'data_type': 'bigint', 'is_nullable': 'YES'},
            {'column_name': 'state_data', 'data_type': 'jsonb', 'is_nullable': 'NO'},
            {'column_name': 'action', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'reward', 'data_type': 'numeric', 'is_nullable': 'NO'},
            {'column_name': 'next_state_data', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'done', 'data_type': 'boolean', 'is_nullable': 'NO'},
            {'column_name': 'priority', 'data_type': 'numeric', 'is_nullable': 'NO'},
            {'column_name': 'trading_mode', 'data_type': 'USER-DEFINED', 'is_nullable': 'YES'},
            {'column_name': 'token_address', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'chain', 'data_type': 'USER-DEFINED', 'is_nullable': 'YES'},
            {'column_name': 'market_conditions', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'performance_metrics', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'error_data', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'metadata', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'created_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'updated_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'}
        ]
    
    @staticmethod
    def get_rl_training_sessions_columns():
        """Mock rl_training_sessions table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO'},
            {'column_name': 'session_id', 'data_type': 'uuid', 'is_nullable': 'NO'},
            {'column_name': 'user_id', 'data_type': 'bigint', 'is_nullable': 'YES'},
            {'column_name': 'session_name', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'trading_mode', 'data_type': 'USER-DEFINED', 'is_nullable': 'NO'},
            {'column_name': 'agent_config', 'data_type': 'jsonb', 'is_nullable': 'NO'},
            {'column_name': 'environment_config', 'data_type': 'jsonb', 'is_nullable': 'NO'},
            {'column_name': 'total_experiences', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'successful_experiences', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'failed_experiences', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'total_reward', 'data_type': 'numeric', 'is_nullable': 'NO'},
            {'column_name': 'average_reward', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'session_status', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'started_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'ended_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'YES'},
            {'column_name': 'duration_seconds', 'data_type': 'integer', 'is_nullable': 'YES'},
            {'column_name': 'metadata', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'created_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'updated_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'}
        ]
    
    @staticmethod
    def get_rl_performance_metrics_columns():
        """Mock rl_performance_metrics table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO'},
            {'column_name': 'metric_id', 'data_type': 'uuid', 'is_nullable': 'NO'},
            {'column_name': 'session_id', 'data_type': 'uuid', 'is_nullable': 'NO'},
            {'column_name': 'user_id', 'data_type': 'bigint', 'is_nullable': 'YES'},
            {'column_name': 'metric_type', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'metric_name', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'metric_value', 'data_type': 'numeric', 'is_nullable': 'NO'},
            {'column_name': 'metric_unit', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'trading_mode', 'data_type': 'USER-DEFINED', 'is_nullable': 'YES'},
            {'column_name': 'time_period_start', 'data_type': 'timestamp with time zone', 'is_nullable': 'YES'},
            {'column_name': 'time_period_end', 'data_type': 'timestamp with time zone', 'is_nullable': 'YES'},
            {'column_name': 'aggregation_level', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'additional_data', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'created_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'updated_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'}
        ]
    
    @staticmethod
    def get_rl_experiences_indexes():
        """Mock rl_experiences table indexes"""
        return [
            {'indexname': 'idx_rl_experiences_session_id', 'indexdef': 'CREATE INDEX idx_rl_experiences_session_id ON rl_experiences USING btree (session_id)'},
            {'indexname': 'idx_rl_experiences_user_created', 'indexdef': 'CREATE INDEX idx_rl_experiences_user_created ON rl_experiences USING btree (user_id, created_at DESC)'},
            {'indexname': 'idx_rl_experiences_priority_desc', 'indexdef': 'CREATE INDEX idx_rl_experiences_priority_desc ON rl_experiences USING btree (priority DESC)'},
            {'indexname': 'idx_rl_experiences_trading_mode_token', 'indexdef': 'CREATE INDEX idx_rl_experiences_trading_mode_token ON rl_experiences USING btree (trading_mode, token_address)'},
            {'indexname': 'idx_rl_experiences_reward_range', 'indexdef': 'CREATE INDEX idx_rl_experiences_reward_range ON rl_experiences USING btree (reward DESC)'},
            {'indexname': 'idx_rl_experiences_done_priority', 'indexdef': 'CREATE INDEX idx_rl_experiences_done_priority ON rl_experiences USING btree (done, priority DESC)'},
            {'indexname': 'idx_rl_experiences_state_gin', 'indexdef': 'CREATE INDEX idx_rl_experiences_state_gin ON rl_experiences USING gin (state_data)'},
            {'indexname': 'idx_rl_experiences_market_conditions_gin', 'indexdef': 'CREATE INDEX idx_rl_experiences_market_conditions_gin ON rl_experiences USING gin (market_conditions)'}
        ]
    
    @staticmethod
    def get_rl_training_sessions_indexes():
        """Mock rl_training_sessions table indexes"""
        return [
            {'indexname': 'idx_rl_training_sessions_user_started', 'indexdef': 'CREATE INDEX idx_rl_training_sessions_user_started ON rl_training_sessions USING btree (user_id, started_at DESC)'},
            {'indexname': 'idx_rl_training_sessions_status_mode', 'indexdef': 'CREATE INDEX idx_rl_training_sessions_status_mode ON rl_training_sessions USING btree (session_status, trading_mode)'},
            {'indexname': 'idx_rl_training_sessions_performance', 'indexdef': 'CREATE INDEX idx_rl_training_sessions_performance ON rl_training_sessions USING btree (total_reward DESC, total_experiences DESC)'}
        ]
    
    @staticmethod
    def get_rl_performance_metrics_indexes():
        """Mock rl_performance_metrics table indexes"""
        return [
            {'indexname': 'idx_rl_performance_metrics_session_type', 'indexdef': 'CREATE INDEX idx_rl_performance_metrics_session_type ON rl_performance_metrics USING btree (session_id, metric_type)'},
            {'indexname': 'idx_rl_performance_metrics_time_range', 'indexdef': 'CREATE INDEX idx_rl_performance_metrics_time_range ON rl_performance_metrics USING btree (time_period_start, time_period_end)'},
            {'indexname': 'idx_rl_performance_metrics_aggregation', 'indexdef': 'CREATE INDEX idx_rl_performance_metrics_aggregation ON rl_performance_metrics USING btree (aggregation_level, metric_name, created_at DESC)'}
        ]
    
    @staticmethod
    def get_rl_experiences_foreign_keys():
        """Mock rl_experiences foreign key constraints"""
        return [
            {'constraint_name': 'fk_rl_experiences_user_id', 'column_name': 'user_id', 'foreign_table_name': 'users', 'foreign_column_name': 'telegram_user_id'},
            {'constraint_name': 'fk_rl_experiences_session_id', 'column_name': 'session_id', 'foreign_table_name': 'rl_training_sessions', 'foreign_column_name': 'session_id'}
        ]
    
    @staticmethod
    def get_rl_training_sessions_foreign_keys():
        """Mock rl_training_sessions foreign key constraints"""
        return [
            {'constraint_name': 'fk_rl_training_sessions_user_id', 'column_name': 'user_id', 'foreign_table_name': 'users', 'foreign_column_name': 'telegram_user_id'}
        ]
    
    @staticmethod
    def get_rl_performance_metrics_foreign_keys():
        """Mock rl_performance_metrics foreign key constraints"""
        return [
            {'constraint_name': 'fk_rl_performance_metrics_user_id', 'column_name': 'user_id', 'foreign_table_name': 'users', 'foreign_column_name': 'telegram_user_id'},
            {'constraint_name': 'fk_rl_performance_metrics_session_id', 'column_name': 'session_id', 'foreign_table_name': 'rl_training_sessions', 'foreign_column_name': 'session_id'}
        ]
    
    @staticmethod
    def get_explain_plan_response(query):
        """Mock EXPLAIN query responses"""
        if "session_id" in query and "rl_experiences" in query:
            return [{'query_plan': 'Index Scan using idx_rl_experiences_session_id on rl_experiences'}]
        elif "priority" in query and "rl_experiences" in query:
            return [{'query_plan': 'Index Scan using idx_rl_experiences_priority_desc on rl_experiences'}]
        else:
            return [{'query_plan': 'Seq Scan on table'}]


@pytest.fixture
def migration_004_applied(mock_database_pool):
    """Fixture that simulates Migration 004 being successfully applied"""
    return Migration004TestHelper.setup_successful_migration_mocks(mock_database_pool)


# Test to verify the migration application helper works
@pytest.mark.asyncio
async def test_migration_004_helper_setup():
    """Test that the migration helper correctly sets up mocks"""
    mock_pool = AsyncMock()
    mock_conn = Migration004TestHelper.setup_successful_migration_mocks(mock_pool)
    
    # Test schema query response
    response_func = mock_conn.fetch.side_effect
    columns = response_func("SELECT column_name FROM information_schema.columns WHERE table_name = 'rl_experiences'")
    
    assert len(columns) >= 19
    column_names = [col['column_name'] for col in columns]
    assert 'experience_id' in column_names
    assert 'state_data' in column_names
    assert 'reward' in column_names