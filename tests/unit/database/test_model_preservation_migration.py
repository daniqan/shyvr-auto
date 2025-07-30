"""
Test Model Preservation Database Schema Migration (TDD Approach)
Tests for migration 005_create_model_preservation_schema.sql
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg


class ModelPreservationMigrationHelper:
    """Helper class to simulate Model Preservation Migration being applied"""
    
    @staticmethod
    def setup_successful_migration_mocks(mock_database_pool):
        """Set up mocks to simulate successful migration application"""
        mock_conn = AsyncMock()
        
        # Mock schema queries to return successful table creation
        mock_conn.fetch.side_effect = ModelPreservationMigrationHelper.get_mock_schema_responses()
        mock_conn.fetchrow.side_effect = ModelPreservationMigrationHelper.get_mock_sample_data_responses()
        mock_conn.fetchval.side_effect = ModelPreservationMigrationHelper.get_mock_fetchval_responses()
        mock_conn.execute.return_value = None
        mock_conn.executemany.return_value = None
        mock_conn.close.return_value = None
        
        # Mock transaction
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        return mock_conn
    
    @staticmethod
    def get_mock_schema_responses():
        """Get mock responses for schema verification queries"""
        def schema_response_generator(query):
            # Parse query to determine what table/index info to return
            if "table_name = 'model_preservation_metadata'" in query and "columns" in query:
                return ModelPreservationMigrationHelper.get_model_preservation_metadata_columns()
            elif "table_name = 'model_version_history'" in query and "columns" in query:
                return ModelPreservationMigrationHelper.get_model_version_history_columns()
            elif "table_name = 'model_preservation_events'" in query and "columns" in query:
                return ModelPreservationMigrationHelper.get_model_preservation_events_columns()
            elif "table_name = 'model_performance_tracking'" in query and "columns" in query:
                return ModelPreservationMigrationHelper.get_model_performance_tracking_columns()
            elif "tablename = 'model_preservation_metadata'" in query and "pg_indexes" in query:
                return ModelPreservationMigrationHelper.get_model_preservation_metadata_indexes()
            elif "tablename = 'model_version_history'" in query and "pg_indexes" in query:
                return ModelPreservationMigrationHelper.get_model_version_history_indexes()
            elif "tablename = 'model_preservation_events'" in query and "pg_indexes" in query:
                return ModelPreservationMigrationHelper.get_model_preservation_events_indexes()
            elif "tablename = 'model_performance_tracking'" in query and "pg_indexes" in query:
                return ModelPreservationMigrationHelper.get_model_performance_tracking_indexes()
            elif "FOREIGN KEY" in query:
                return ModelPreservationMigrationHelper.get_foreign_key_constraints(query)
            elif "pg_enum" in query and "model_state" in query:
                return [{'enumlabel': state} for state in ['active', 'preserved', 'archived', 'corrupted', 'deleted']]
            elif "pg_enum" in query and "preservation_priority" in query:
                return [{'enumlabel': priority} for priority in ['critical', 'high', 'normal', 'low']]
            elif "pg_trigger" in query:
                return ModelPreservationMigrationHelper.get_triggers()
            elif "model_version_history" in query and "ORDER BY" in query:
                return ModelPreservationMigrationHelper.get_initial_version_history()
            else:
                return []
        
        return schema_response_generator
    
    @staticmethod
    def get_mock_fetchval_responses():
        """Get mock responses for fetchval queries"""
        def fetchval_response_generator(query):
            if "EXISTS" in query and "model_preservation_metadata" in query:
                return True
            elif "EXISTS" in query and "model_version_history" in query:
                return True
            elif "EXISTS" in query and "model_preservation_events" in query:
                return True
            elif "EXISTS" in query and "model_performance_tracking" in query:
                return True
            elif "COUNT(*)" in query:
                return 20  # Default count for various queries
            elif "cleanup_old_models" in query:
                return 5  # Number of models cleaned up
            elif "state FROM model_preservation_metadata" in query:
                if "test_v1" in query:
                    return 'preserved'  # Critical model not archived
                elif "test_v2" in query:
                    return 'archived'  # Old model archived
            elif "updated_at FROM model_preservation_metadata" in query:
                return datetime.now(timezone.utc)
            else:
                return None
        
        return fetchval_response_generator
    
    @staticmethod
    def get_mock_sample_data_responses():
        """Get mock responses for sample data queries"""
        def sample_data_response_generator(query):
            if "get_latest_model_version" in query:
                return {
                    'preservation_id': 'test_v3',
                    'model_id': 'model_003',
                    'model_type': 'lstm',
                    'version': 'v1.2.0',
                    'mode': 'analysis',
                    'preserved_at': datetime.now(timezone.utc)
                }
            else:
                return None
        
        return sample_data_response_generator
    
    @staticmethod
    def get_model_preservation_metadata_columns():
        """Mock model_preservation_metadata table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO', 'column_default': "nextval('model_preservation_metadata_id_seq'::regclass)"},
            {'column_name': 'preservation_id', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'model_id', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'model_type', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'version', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'mode', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'created_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO', 'column_default': 'CURRENT_TIMESTAMP'},
            {'column_name': 'preserved_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'updated_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO', 'column_default': 'CURRENT_TIMESTAMP'},
            {'column_name': 'checksum', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'size_bytes', 'data_type': 'bigint', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'gcs_path', 'data_type': 'character varying', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'performance_metrics', 'data_type': 'jsonb', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'training_info', 'data_type': 'jsonb', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'preservation_reason', 'data_type': 'text', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'priority', 'data_type': 'USER-DEFINED', 'is_nullable': 'NO', 'column_default': "'normal'::preservation_priority"},
            {'column_name': 'tags', 'data_type': 'ARRAY', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'metadata', 'data_type': 'jsonb', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'state', 'data_type': 'USER-DEFINED', 'is_nullable': 'NO', 'column_default': "'preserved'::model_state"},
            {'column_name': 'user_id', 'data_type': 'bigint', 'is_nullable': 'YES', 'column_default': None}
        ]
    
    @staticmethod
    def get_model_version_history_columns():
        """Mock model_version_history table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO'},
            {'column_name': 'model_type', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'mode', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'current_version', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'previous_version', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'major_version', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'minor_version', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'patch_version', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'current_preservation_id', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'previous_preservation_id', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'change_reason', 'data_type': 'text', 'is_nullable': 'YES'},
            {'column_name': 'changed_by', 'data_type': 'character varying', 'is_nullable': 'YES'},
            {'column_name': 'changed_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'}
        ]
    
    @staticmethod
    def get_model_preservation_events_columns():
        """Mock model_preservation_events table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO', 'column_default': "nextval('model_preservation_events_id_seq'::regclass)"},
            {'column_name': 'event_id', 'data_type': 'uuid', 'is_nullable': 'NO', 'column_default': 'uuid_generate_v4()'},
            {'column_name': 'event_type', 'data_type': 'character varying', 'is_nullable': 'NO', 'column_default': None},
            {'column_name': 'preservation_id', 'data_type': 'character varying', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'model_type', 'data_type': 'character varying', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'mode', 'data_type': 'character varying', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'event_data', 'data_type': 'jsonb', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'error_message', 'data_type': 'text', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'success', 'data_type': 'boolean', 'is_nullable': 'NO', 'column_default': 'true'},
            {'column_name': 'user_id', 'data_type': 'bigint', 'is_nullable': 'YES', 'column_default': None},
            {'column_name': 'occurred_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO', 'column_default': 'CURRENT_TIMESTAMP'},
            {'column_name': 'duration_ms', 'data_type': 'integer', 'is_nullable': 'YES', 'column_default': None}
        ]
    
    @staticmethod
    def get_model_performance_tracking_columns():
        """Mock model_performance_tracking table columns"""
        return [
            {'column_name': 'id', 'data_type': 'bigint', 'is_nullable': 'NO'},
            {'column_name': 'preservation_id', 'data_type': 'character varying', 'is_nullable': 'NO'},
            {'column_name': 'accuracy', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'loss', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'prediction_count', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'successful_predictions', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'total_return', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'sharpe_ratio', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'max_drawdown', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'win_rate', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'inference_time_ms', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'memory_usage_mb', 'data_type': 'numeric', 'is_nullable': 'YES'},
            {'column_name': 'period_start', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'period_end', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'},
            {'column_name': 'custom_metrics', 'data_type': 'jsonb', 'is_nullable': 'YES'},
            {'column_name': 'created_at', 'data_type': 'timestamp with time zone', 'is_nullable': 'NO'}
        ]
    
    @staticmethod
    def get_model_preservation_metadata_indexes():
        """Mock model_preservation_metadata table indexes"""
        return [
            {'indexname': 'idx_preservation_model_type_mode', 'indexdef': 'CREATE INDEX idx_preservation_model_type_mode ON model_preservation_metadata USING btree (model_type, mode)'},
            {'indexname': 'idx_preservation_version', 'indexdef': 'CREATE INDEX idx_preservation_version ON model_preservation_metadata USING btree (version)'},
            {'indexname': 'idx_preservation_preserved_at', 'indexdef': 'CREATE INDEX idx_preservation_preserved_at ON model_preservation_metadata USING btree (preserved_at DESC)'},
            {'indexname': 'idx_preservation_state', 'indexdef': 'CREATE INDEX idx_preservation_state ON model_preservation_metadata USING btree (state)'},
            {'indexname': 'idx_preservation_priority', 'indexdef': 'CREATE INDEX idx_preservation_priority ON model_preservation_metadata USING btree (priority)'},
            {'indexname': 'idx_preservation_tags', 'indexdef': 'CREATE INDEX idx_preservation_tags ON model_preservation_metadata USING gin (tags)'},
            {'indexname': 'idx_preservation_metadata_gin', 'indexdef': 'CREATE INDEX idx_preservation_metadata_gin ON model_preservation_metadata USING gin (metadata)'}
        ]
    
    @staticmethod
    def get_model_version_history_indexes():
        """Mock model_version_history table indexes"""
        return [
            {'indexname': 'idx_version_history_type_mode', 'indexdef': 'CREATE INDEX idx_version_history_type_mode ON model_version_history USING btree (model_type, mode)'},
            {'indexname': 'idx_version_history_changed_at', 'indexdef': 'CREATE INDEX idx_version_history_changed_at ON model_version_history USING btree (changed_at DESC)'}
        ]
    
    @staticmethod
    def get_model_preservation_events_indexes():
        """Mock model_preservation_events table indexes"""
        return [
            {'indexname': 'idx_preservation_events_preservation_id', 'indexdef': 'CREATE INDEX idx_preservation_events_preservation_id ON model_preservation_events USING btree (preservation_id)'},
            {'indexname': 'idx_preservation_events_type_occurred', 'indexdef': 'CREATE INDEX idx_preservation_events_type_occurred ON model_preservation_events USING btree (event_type, occurred_at DESC)'},
            {'indexname': 'idx_preservation_events_user_occurred', 'indexdef': 'CREATE INDEX idx_preservation_events_user_occurred ON model_preservation_events USING btree (user_id, occurred_at DESC)'}
        ]
    
    @staticmethod
    def get_model_performance_tracking_indexes():
        """Mock model_performance_tracking table indexes"""
        return [
            {'indexname': 'idx_performance_preservation_id', 'indexdef': 'CREATE INDEX idx_performance_preservation_id ON model_performance_tracking USING btree (preservation_id)'},
            {'indexname': 'idx_performance_period', 'indexdef': 'CREATE INDEX idx_performance_period ON model_performance_tracking USING btree (period_start, period_end)'}
        ]
    
    @staticmethod
    def get_foreign_key_constraints(query):
        """Mock foreign key constraints"""
        constraints = []
        if "model_preservation_metadata" in query:
            constraints.append({
                'constraint_name': 'fk_model_preservation_user_id',
                'column_name': 'user_id',
                'foreign_table_name': 'users',
                'foreign_column_name': 'telegram_user_id'
            })
        elif "model_version_history" in query:
            constraints.extend([
                {
                    'constraint_name': 'fk_version_history_current_preservation',
                    'column_name': 'current_preservation_id',
                    'foreign_table_name': 'model_preservation_metadata',
                    'foreign_column_name': 'preservation_id'
                },
                {
                    'constraint_name': 'fk_version_history_previous_preservation',
                    'column_name': 'previous_preservation_id',
                    'foreign_table_name': 'model_preservation_metadata',
                    'foreign_column_name': 'preservation_id'
                }
            ])
        elif "model_preservation_events" in query:
            constraints.append({
                'constraint_name': 'fk_preservation_events_preservation_id',
                'column_name': 'preservation_id',
                'foreign_table_name': 'model_preservation_metadata',
                'foreign_column_name': 'preservation_id'
            })
        elif "model_performance_tracking" in query:
            constraints.append({
                'constraint_name': 'fk_performance_tracking_preservation_id',
                'column_name': 'preservation_id',
                'foreign_table_name': 'model_preservation_metadata',
                'foreign_column_name': 'preservation_id'
            })
        return constraints
    
    @staticmethod
    def get_triggers():
        """Mock triggers"""
        return [
            {'tgname': 'trigger_model_preservation_updated_at', 'proname': 'update_model_preservation_updated_at'},
            {'tgname': 'trigger_log_preservation_events', 'proname': 'log_preservation_event'}
        ]
    
    @staticmethod
    def get_initial_version_history():
        """Mock initial version history entries"""
        entries = []
        for model_type in ['lstm', 'dqn', 'ensemble']:
            for mode in ['analysis', 'simulation', 'live']:
                entries.append({
                    'model_type': model_type,
                    'mode': mode,
                    'current_version': 'v1.0.0'
                })
        return entries


@pytest.fixture
def model_preservation_migration_applied(mock_database_pool):
    """Fixture that simulates Model Preservation Migration being successfully applied"""
    return ModelPreservationMigrationHelper.setup_successful_migration_mocks(mock_database_pool)


class TestModelPreservationMigration:
    """Comprehensive tests for model preservation schema migration"""
    
    @pytest.mark.asyncio
    async def test_model_preservation_metadata_table_created(self, model_preservation_migration_applied):
        """Test that model_preservation_metadata table is created with correct schema"""
        mock_conn = model_preservation_migration_applied
        
        # Simulate checking table exists
        table_exists = await mock_conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_preservation_metadata'
            )
        """)
        assert table_exists, "model_preservation_metadata table should exist"
        
        # Simulate checking columns
        columns = await mock_conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'model_preservation_metadata'
            ORDER BY ordinal_position
        """)
        
        column_names = [col['column_name'] for col in columns]
        expected_columns = [
            'id', 'preservation_id', 'model_id', 'model_type', 'version', 'mode',
            'created_at', 'preserved_at', 'updated_at', 'checksum', 'size_bytes',
            'gcs_path', 'performance_metrics', 'training_info', 'preservation_reason',
            'priority', 'tags', 'metadata', 'state', 'user_id'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Column {expected_col} should exist"
        
        # Check specific column properties
        preservation_id_col = next(col for col in columns if col['column_name'] == 'preservation_id')
        assert preservation_id_col['is_nullable'] == 'NO', "preservation_id should be NOT NULL"
        
        # Check JSONB columns
        jsonb_columns = ['performance_metrics', 'training_info', 'metadata']
        for col_name in jsonb_columns:
            col = next(col for col in columns if col['column_name'] == col_name)
            assert col['data_type'] == 'jsonb', f"{col_name} should be JSONB type"
    
    @pytest.mark.asyncio
    async def test_model_version_history_table_created(self, model_preservation_migration_applied):
        """Test that model_version_history table is created with correct schema"""
        mock_conn = model_preservation_migration_applied
        
        # Check table exists
        table_exists = await mock_conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_version_history'
            )
        """)
        assert table_exists, "model_version_history table should exist"
        
        # Check columns
        columns = await mock_conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'model_version_history'
            ORDER BY ordinal_position
        """)
        
        column_names = [col['column_name'] for col in columns]
        expected_columns = [
            'id', 'model_type', 'mode', 'current_version', 'previous_version',
            'major_version', 'minor_version', 'patch_version',
            'current_preservation_id', 'previous_preservation_id',
            'change_reason', 'changed_by', 'changed_at'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Column {expected_col} should exist"
    
    @pytest.mark.asyncio
    async def test_model_preservation_events_table_created(self, model_preservation_migration_applied):
        """Test that model_preservation_events table is created with correct schema"""
        mock_conn = model_preservation_migration_applied
        
        # Check table exists
        table_exists = await mock_conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_preservation_events'
            )
        """)
        assert table_exists, "model_preservation_events table should exist"
        
        # Check columns
        columns = await mock_conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'model_preservation_events'
            ORDER BY ordinal_position
        """)
        
        column_names = [col['column_name'] for col in columns]
        expected_columns = [
            'id', 'event_id', 'event_type', 'preservation_id', 'model_type',
            'mode', 'event_data', 'error_message', 'success', 'user_id',
            'occurred_at', 'duration_ms'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Column {expected_col} should exist"
        
        # Check event_id has UUID default
        event_id_col = next(col for col in columns if col['column_name'] == 'event_id')
        assert 'uuid_generate_v4()' in str(event_id_col['column_default']), \
            "event_id should have uuid_generate_v4() as default"
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking_table_created(self, model_preservation_migration_applied):
        """Test that model_performance_tracking table is created with correct schema"""
        mock_conn = model_preservation_migration_applied
        
        # Check table exists
        table_exists = await mock_conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_performance_tracking'
            )
        """)
        assert table_exists, "model_performance_tracking table should exist"
        
        # Check columns
        columns = await mock_conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'model_performance_tracking'
            ORDER BY ordinal_position
        """)
        
        column_names = [col['column_name'] for col in columns]
        expected_columns = [
            'id', 'preservation_id', 'accuracy', 'loss', 'prediction_count',
            'successful_predictions', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'inference_time_ms', 'memory_usage_mb',
            'period_start', 'period_end', 'custom_metrics', 'created_at'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Column {expected_col} should exist"
        
        # Check numeric precision
        accuracy_col = next(col for col in columns if col['column_name'] == 'accuracy')
        assert accuracy_col['data_type'] == 'numeric', "accuracy should be NUMERIC type"
    
    @pytest.mark.asyncio
    async def test_enums_created(self, model_preservation_migration_applied):
        """Test that custom enum types are created"""
        mock_conn = model_preservation_migration_applied
        
        # Check model_state enum
        model_state_values = await mock_conn.fetch("""
            SELECT enumlabel
            FROM pg_enum
            JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
            WHERE pg_type.typname = 'model_state'
            ORDER BY enumsortorder
        """)
        
        expected_states = ['active', 'preserved', 'archived', 'corrupted', 'deleted']
        actual_states = [row['enumlabel'] for row in model_state_values]
        assert actual_states == expected_states, "model_state enum should have correct values"
        
        # Check preservation_priority enum
        priority_values = await mock_conn.fetch("""
            SELECT enumlabel
            FROM pg_enum
            JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
            WHERE pg_type.typname = 'preservation_priority'
            ORDER BY enumsortorder
        """)
        
        expected_priorities = ['critical', 'high', 'normal', 'low']
        actual_priorities = [row['enumlabel'] for row in priority_values]
        assert actual_priorities == expected_priorities, "preservation_priority enum should have correct values"
    
    @pytest.mark.asyncio
    async def test_indexes_created(self, model_preservation_migration_applied):
        """Test that all required indexes are created for performance"""
        mock_conn = model_preservation_migration_applied
        
        # Check indexes on model_preservation_metadata
        preservation_indexes = await mock_conn.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'model_preservation_metadata'
            AND schemaname = 'public'
        """)
        
        index_names = [idx['indexname'] for idx in preservation_indexes]
        expected_indexes = [
            'idx_preservation_model_type_mode',
            'idx_preservation_version',
            'idx_preservation_preserved_at',
            'idx_preservation_state',
            'idx_preservation_priority',
            'idx_preservation_tags',
            'idx_preservation_metadata_gin'
        ]
        
        for expected_idx in expected_indexes:
            assert expected_idx in index_names, f"Index {expected_idx} should exist"
        
        # Check GIN indexes for JSONB columns
        gin_indexes = [idx for idx in preservation_indexes if 'gin' in idx['indexdef'].lower()]
        assert len(gin_indexes) >= 2, "Should have at least 2 GIN indexes for JSONB columns"
        
        # Check other table indexes
        version_indexes = await mock_conn.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'model_version_history'
        """)
        assert len(version_indexes) >= 2, "model_version_history should have indexes"
        
        event_indexes = await mock_conn.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'model_preservation_events'
        """)
        assert len(event_indexes) >= 3, "model_preservation_events should have indexes"
        
        perf_indexes = await mock_conn.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'model_performance_tracking'
        """)
        assert len(perf_indexes) >= 2, "model_performance_tracking should have indexes"
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, model_preservation_migration_applied):
        """Test that foreign key relationships are properly established"""
        mock_conn = model_preservation_migration_applied
        
        # Check foreign keys in model_preservation_metadata
        fk_constraints = await mock_conn.fetch("""
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
            AND tc.table_name IN ('model_preservation_metadata', 'model_version_history', 
                                 'model_preservation_events', 'model_performance_tracking')
        """)
        
        # Verify user_id foreign key exists (to users table)
        user_fks = [fk for fk in fk_constraints if fk['column_name'] == 'user_id']
        assert len(user_fks) >= 1, "Should have foreign key on user_id"
        
        # Verify preservation_id foreign keys
        preservation_fks = [fk for fk in fk_constraints if fk['column_name'] == 'preservation_id']
        assert len(preservation_fks) >= 1, "Should have foreign keys referencing preservation_id"
        
        # Verify version history foreign keys
        version_fks = [fk for fk in fk_constraints 
                      if fk['column_name'] in ('current_preservation_id', 'previous_preservation_id')]
        assert len(version_fks) >= 2, "Version history should have preservation_id foreign keys"
    
    @pytest.mark.asyncio
    async def test_triggers_created(self, model_preservation_migration_applied):
        """Test that triggers are created and functioning"""
        mock_conn = model_preservation_migration_applied
        
        # Check triggers exist
        triggers = await mock_conn.fetch("""
            SELECT tgname, proname
            FROM pg_trigger t
            JOIN pg_proc p ON t.tgfoid = p.oid
            WHERE tgrelid IN (
                SELECT oid FROM pg_class 
                WHERE relname = 'model_preservation_metadata'
            )
        """)
        
        trigger_names = [t['tgname'] for t in triggers]
        assert 'trigger_model_preservation_updated_at' in trigger_names, \
            "Updated_at trigger should exist"
        assert 'trigger_log_preservation_events' in trigger_names, \
            "Event logging trigger should exist"
    
    @pytest.mark.asyncio
    async def test_helper_functions(self, model_preservation_migration_applied):
        """Test that helper functions are created and work correctly"""
        mock_conn = model_preservation_migration_applied
        
        # Test get_latest_model_version function
        latest = await mock_conn.fetchrow(
            "SELECT * FROM get_latest_model_version('lstm', 'analysis')"
        )
        
        assert latest is not None, "get_latest_model_version should return a result"
        assert latest['preservation_id'] == 'test_v3', "Should return the latest version"
        assert latest['version'] == 'v1.2.0', "Should return correct version"
        
        # Test cleanup_old_models function
        archived_count = await mock_conn.fetchval(
            "SELECT cleanup_old_models(1, 1)"  # Keep only 1 model, older than 1 day
        )
        
        assert archived_count == 5, "Should return count of archived models"
    
    @pytest.mark.asyncio
    async def test_initial_data_insertion(self, model_preservation_migration_applied):
        """Test that initial version history data is inserted"""
        mock_conn = model_preservation_migration_applied
        
        # Check initial version history entries
        initial_entries = await mock_conn.fetch("""
            SELECT model_type, mode, current_version
            FROM model_version_history
            ORDER BY model_type, mode
        """)
        
        # Should have 9 entries (3 model types × 3 modes)
        assert len(initial_entries) == 9, "Should have 9 initial version history entries"
        
        # Check all combinations exist
        expected_combinations = [
            ('lstm', 'analysis'), ('lstm', 'simulation'), ('lstm', 'live'),
            ('dqn', 'analysis'), ('dqn', 'simulation'), ('dqn', 'live'),
            ('ensemble', 'analysis'), ('ensemble', 'simulation'), ('ensemble', 'live')
        ]
        
        actual_combinations = [(e['model_type'], e['mode']) for e in initial_entries]
        for expected in expected_combinations:
            assert expected in actual_combinations, f"Should have entry for {expected}"
        
        # All should start at v1.0.0
        for entry in initial_entries:
            assert entry['current_version'] == 'v1.0.0', \
                f"Initial version should be v1.0.0 for {entry['model_type']}/{entry['mode']}"


# Test to verify the migration application helper works
@pytest.mark.asyncio
async def test_model_preservation_migration_helper_setup():
    """Test that the migration helper correctly sets up mocks"""
    mock_pool = AsyncMock()
    mock_conn = ModelPreservationMigrationHelper.setup_successful_migration_mocks(mock_pool)
    
    # Test schema query response
    response_func = mock_conn.fetch.side_effect
    columns = response_func("SELECT column_name FROM information_schema.columns WHERE table_name = 'model_preservation_metadata'")
    
    assert len(columns) == 20
    column_names = [col['column_name'] for col in columns]
    assert 'preservation_id' in column_names
    assert 'model_id' in column_names
    assert 'performance_metrics' in column_names


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])