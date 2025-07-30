"""
Test Model Preservation Database Schema Migration (TDD Approach)
Tests for migration 005_create_model_preservation_schema.sql
"""

import asyncio
import pytest
import asyncpg
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import os
import json
from typing import Dict, List, Any, Optional


class TestModelPreservationMigration:
    """Comprehensive tests for model preservation schema migration"""
    
    @classmethod
    def setup_class(cls):
        """Set up test database connection parameters"""
        cls.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'shyvr_rlte_test'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
    
    async def get_test_connection(self):
        """Get a test database connection"""
        return await asyncpg.connect(**self.db_config)
    
    async def apply_migration(self, conn: asyncpg.Connection):
        """Apply the migration to the test database"""
        migration_path = '/Users/kendo/daniqan/shyvrai-rlte/database/migrations/005_create_model_preservation_schema.sql'
        
        # Read migration file
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        # Execute migration
        await conn.execute(migration_sql)
    
    async def rollback_migration(self, conn: asyncpg.Connection):
        """Rollback the migration for clean test state"""
        # Drop tables in reverse order due to foreign key constraints
        tables = [
            'model_performance_tracking',
            'model_preservation_events', 
            'model_version_history',
            'model_preservation_metadata'
        ]
        
        for table in tables:
            await conn.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
        
        # Drop types
        await conn.execute('DROP TYPE IF EXISTS model_state CASCADE')
        await conn.execute('DROP TYPE IF EXISTS preservation_priority CASCADE')
        
        # Drop functions
        await conn.execute('DROP FUNCTION IF EXISTS update_model_preservation_updated_at() CASCADE')
        await conn.execute('DROP FUNCTION IF EXISTS log_preservation_event() CASCADE')
        await conn.execute('DROP FUNCTION IF EXISTS get_latest_model_version(VARCHAR, VARCHAR) CASCADE')
        await conn.execute('DROP FUNCTION IF EXISTS cleanup_old_models(INTEGER, INTEGER) CASCADE')
    
    @pytest.mark.asyncio
    async def test_model_preservation_metadata_table_created(self):
        """Test that model_preservation_metadata table is created with correct schema"""
        conn = await self.get_test_connection()
        
        try:
            # Rollback any existing migration
            await self.rollback_migration(conn)
            
            # Apply migration
            await self.apply_migration(conn)
            
            # Check table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_preservation_metadata'
                )
            """)
            assert table_exists, "model_preservation_metadata table should exist"
            
            # Check columns
            columns = await conn.fetch("""
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
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_model_version_history_table_created(self):
        """Test that model_version_history table is created with correct schema"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_version_history'
                )
            """)
            assert table_exists, "model_version_history table should exist"
            
            # Check columns
            columns = await conn.fetch("""
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
            
            # Check unique constraint
            constraints = await conn.fetch("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_name = 'model_version_history'
                AND constraint_type = 'UNIQUE'
            """)
            
            assert any(c['constraint_name'] == 'uq_model_version_history' for c in constraints), \
                "Unique constraint on (model_type, mode) should exist"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_model_preservation_events_table_created(self):
        """Test that model_preservation_events table is created with correct schema"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_preservation_events'
                )
            """)
            assert table_exists, "model_preservation_events table should exist"
            
            # Check columns
            columns = await conn.fetch("""
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
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking_table_created(self):
        """Test that model_performance_tracking table is created with correct schema"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_performance_tracking'
                )
            """)
            assert table_exists, "model_performance_tracking table should exist"
            
            # Check columns
            columns = await conn.fetch("""
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
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_enums_created(self):
        """Test that custom enum types are created"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check model_state enum
            model_state_values = await conn.fetch("""
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
            priority_values = await conn.fetch("""
                SELECT enumlabel
                FROM pg_enum
                JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
                WHERE pg_type.typname = 'preservation_priority'
                ORDER BY enumsortorder
            """)
            
            expected_priorities = ['critical', 'high', 'normal', 'low']
            actual_priorities = [row['enumlabel'] for row in priority_values]
            assert actual_priorities == expected_priorities, "preservation_priority enum should have correct values"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_indexes_created(self):
        """Test that all required indexes are created for performance"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check indexes on model_preservation_metadata
            preservation_indexes = await conn.fetch("""
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
            version_indexes = await conn.fetch("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'model_version_history'
            """)
            assert len(version_indexes) >= 2, "model_version_history should have indexes"
            
            event_indexes = await conn.fetch("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'model_preservation_events'
            """)
            assert len(event_indexes) >= 3, "model_preservation_events should have indexes"
            
            perf_indexes = await conn.fetch("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'model_performance_tracking'
            """)
            assert len(perf_indexes) >= 2, "model_performance_tracking should have indexes"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self):
        """Test that foreign key relationships are properly established"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check foreign keys in model_preservation_metadata
            fk_constraints = await conn.fetch("""
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
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_triggers_created(self):
        """Test that triggers are created and functioning"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check triggers exist
            triggers = await conn.fetch("""
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
            
            # Test updated_at trigger functionality
            preservation_id = f'test_{uuid.uuid4()}'
            await conn.execute("""
                INSERT INTO model_preservation_metadata (
                    preservation_id, model_id, model_type, version, mode,
                    created_at, preserved_at, checksum, size_bytes, state
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, preservation_id, 'model_001', 'lstm', 'v1.0.0', 'analysis',
                datetime.now(timezone.utc), datetime.now(timezone.utc),
                'abc123', 1024, 'preserved')
            
            # Update and check updated_at changed
            original_updated = await conn.fetchval(
                "SELECT updated_at FROM model_preservation_metadata WHERE preservation_id = $1",
                preservation_id
            )
            
            await asyncio.sleep(0.1)  # Small delay to ensure timestamp difference
            
            await conn.execute(
                "UPDATE model_preservation_metadata SET state = 'archived' WHERE preservation_id = $1",
                preservation_id
            )
            
            new_updated = await conn.fetchval(
                "SELECT updated_at FROM model_preservation_metadata WHERE preservation_id = $1",
                preservation_id
            )
            
            assert new_updated > original_updated, "updated_at should be updated by trigger"
            
            # Check event was logged by trigger
            event_logged = await conn.fetchval("""
                SELECT COUNT(*) FROM model_preservation_events
                WHERE preservation_id = $1 AND event_type = 'state_changed'
            """, preservation_id)
            
            assert event_logged > 0, "State change event should be logged by trigger"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_helper_functions(self):
        """Test that helper functions are created and work correctly"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Test get_latest_model_version function
            # Insert test data
            test_preservations = [
                ('test_v1', 'model_001', 'lstm', 'v1.0.0', 'analysis', 
                 datetime.now(timezone.utc) - timedelta(days=2)),
                ('test_v2', 'model_002', 'lstm', 'v1.1.0', 'analysis',
                 datetime.now(timezone.utc) - timedelta(days=1)),
                ('test_v3', 'model_003', 'lstm', 'v1.2.0', 'analysis',
                 datetime.now(timezone.utc))
            ]
            
            for pres_id, model_id, model_type, version, mode, preserved_at in test_preservations:
                await conn.execute("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, pres_id, model_id, model_type, version, mode,
                    datetime.now(timezone.utc), preserved_at, 'checksum', 1024, 'preserved')
            
            # Test get_latest_model_version function
            latest = await conn.fetchrow(
                "SELECT * FROM get_latest_model_version('lstm', 'analysis')"
            )
            
            assert latest is not None, "get_latest_model_version should return a result"
            assert latest['preservation_id'] == 'test_v3', "Should return the latest version"
            assert latest['version'] == 'v1.2.0', "Should return correct version"
            
            # Test cleanup_old_models function
            # Mark one as critical priority
            await conn.execute("""
                UPDATE model_preservation_metadata 
                SET priority = 'critical' 
                WHERE preservation_id = 'test_v1'
            """)
            
            # Run cleanup (should archive old models except critical ones)
            archived_count = await conn.fetchval(
                "SELECT cleanup_old_models(1, 1)"  # Keep only 1 model, older than 1 day
            )
            
            # Check that non-critical old model was archived
            v2_state = await conn.fetchval(
                "SELECT state FROM model_preservation_metadata WHERE preservation_id = 'test_v2'",
            )
            assert v2_state == 'archived', "Old non-critical model should be archived"
            
            # Check that critical model was not archived
            v1_state = await conn.fetchval(
                "SELECT state FROM model_preservation_metadata WHERE preservation_id = 'test_v1'",
            )
            assert v1_state == 'preserved', "Critical model should not be archived"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_constraints_validation(self):
        """Test that constraints are properly enforced"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Test version format constraint
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 'test_bad_version', 'model_001', 'lstm', 'bad_version_format', 'analysis',
                    datetime.now(timezone.utc), datetime.now(timezone.utc),
                    'checksum', 1024, 'preserved')
            
            # Test mode constraint
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 'test_bad_mode', 'model_001', 'lstm', 'v1.0.0', 'invalid_mode',
                    datetime.now(timezone.utc), datetime.now(timezone.utc),
                    'checksum', 1024, 'preserved')
            
            # Test model_type constraint
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 'test_bad_type', 'model_001', 'invalid_type', 'v1.0.0', 'analysis',
                    datetime.now(timezone.utc), datetime.now(timezone.utc),
                    'checksum', 1024, 'preserved')
            
            # Test unique constraint on preservation_id
            await conn.execute("""
                INSERT INTO model_preservation_metadata (
                    preservation_id, model_id, model_type, version, mode,
                    created_at, preserved_at, checksum, size_bytes, state
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 'unique_test', 'model_001', 'lstm', 'v1.0.0', 'analysis',
                datetime.now(timezone.utc), datetime.now(timezone.utc),
                'checksum', 1024, 'preserved')
            
            with pytest.raises(asyncpg.UniqueViolationError):
                await conn.execute("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 'unique_test', 'model_002', 'lstm', 'v1.0.0', 'analysis',
                    datetime.now(timezone.utc), datetime.now(timezone.utc),
                    'checksum', 1024, 'preserved')
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_initial_data_insertion(self):
        """Test that initial version history data is inserted"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Check initial version history entries
            initial_entries = await conn.fetch("""
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
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_migration_rollback(self):
        """Test that migration can be rolled back cleanly"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration
            await self.apply_migration(conn)
            
            # Verify tables exist
            tables_exist = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name IN (
                    'model_preservation_metadata', 'model_version_history',
                    'model_preservation_events', 'model_performance_tracking'
                )
            """)
            assert len(tables_exist) == 4, "All 4 tables should exist after migration"
            
            # Rollback migration
            await self.rollback_migration(conn)
            
            # Verify tables don't exist
            tables_after_rollback = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name IN (
                    'model_preservation_metadata', 'model_version_history',
                    'model_preservation_events', 'model_performance_tracking'
                )
            """)
            assert len(tables_after_rollback) == 0, "No tables should exist after rollback"
            
            # Verify types don't exist
            types_after_rollback = await conn.fetch("""
                SELECT typname 
                FROM pg_type 
                WHERE typname IN ('model_state', 'preservation_priority')
            """)
            assert len(types_after_rollback) == 0, "No custom types should exist after rollback"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with a large number of preserved models"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            # Insert a large number of test records
            batch_size = 1000
            num_batches = 5
            
            import time
            start_time = time.time()
            
            for batch in range(num_batches):
                data = []
                for i in range(batch_size):
                    preservation_id = f'perf_test_{batch}_{i}'
                    model_id = f'model_{batch}_{i}'
                    version = f'v1.{batch}.{i}'
                    
                    data.append((
                        preservation_id, model_id, 'lstm', version, 'analysis',
                        datetime.now(timezone.utc), datetime.now(timezone.utc),
                        'checksum', 1024 * (i + 1), 'preserved',
                        json.dumps({'accuracy': 0.9 + (i * 0.0001)}),
                        json.dumps({'epochs': 100, 'batch_size': 32})
                    ))
                
                await conn.executemany("""
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, state,
                        performance_metrics, training_info
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, data)
            
            insert_time = time.time() - start_time
            
            # Test query performance with indexes
            query_start = time.time()
            
            # Query by model_type and mode (should use index)
            results = await conn.fetch("""
                SELECT preservation_id, version, performance_metrics
                FROM model_preservation_metadata
                WHERE model_type = 'lstm' AND mode = 'analysis'
                ORDER BY preserved_at DESC
                LIMIT 100
            """)
            
            query_time = time.time() - query_start
            
            # Performance assertions
            assert len(results) == 100, "Should return 100 results"
            assert insert_time < 10, f"Inserting {batch_size * num_batches} records should take < 10s, took {insert_time}s"
            assert query_time < 0.5, f"Indexed query should take < 0.5s, took {query_time}s"
            
            # Test JSONB query performance
            jsonb_start = time.time()
            
            high_accuracy_models = await conn.fetch("""
                SELECT preservation_id, performance_metrics->>'accuracy' as accuracy
                FROM model_preservation_metadata
                WHERE (performance_metrics->>'accuracy')::float > 0.95
                LIMIT 50
            """)
            
            jsonb_time = time.time() - jsonb_start
            assert jsonb_time < 1.0, f"JSONB query should take < 1s, took {jsonb_time}s"
            
        finally:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent insert and update operations"""
        conn = await self.get_test_connection()
        
        try:
            # Apply migration if not already applied
            await self.apply_migration(conn)
            
            async def insert_preservation(conn_pool, preservation_id):
                async with conn_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO model_preservation_metadata (
                            preservation_id, model_id, model_type, version, mode,
                            created_at, preserved_at, checksum, size_bytes, state
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, preservation_id, f'model_{preservation_id}', 'dqn', 'v1.0.0', 'simulation',
                        datetime.now(timezone.utc), datetime.now(timezone.utc),
                        'checksum', 2048, 'preserved')
            
            async def log_event(conn_pool, preservation_id):
                async with conn_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO model_preservation_events (
                            event_type, preservation_id, model_type, mode, success
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, 'model_loaded', preservation_id, 'dqn', 'simulation', True)
            
            # Create connection pool for concurrent operations
            pool = await asyncpg.create_pool(**self.db_config, min_size=5, max_size=10)
            
            try:
                # Prepare preservation IDs
                preservation_ids = [f'concurrent_{i}' for i in range(20)]
                
                # Insert preservations concurrently
                insert_tasks = [insert_preservation(pool, pid) for pid in preservation_ids]
                await asyncio.gather(*insert_tasks)
                
                # Log events concurrently
                event_tasks = [log_event(pool, pid) for pid in preservation_ids]
                await asyncio.gather(*event_tasks)
                
                # Verify all operations succeeded
                preservation_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM model_preservation_metadata WHERE preservation_id LIKE 'concurrent_%'"
                )
                assert preservation_count == 20, "All concurrent preservations should be inserted"
                
                event_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM model_preservation_events WHERE preservation_id LIKE 'concurrent_%'"
                )
                assert event_count == 20, "All concurrent events should be logged"
                
            finally:
                await pool.close()
                
        finally:
            await conn.close()


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])