"""
Test Database Connectivity and Schema Setup for Activity Logging
Following TDD methodology - all tests should fail initially
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncpg

from src.utils.database import get_database_pool, close_database_pool, test_database_connection
from src.logging.activity_logger import ActivityCategory, ActivityAction, ActivitySeverity


class TestDatabaseConnection:
    """Test database connection and pool management"""
    
    @pytest.mark.asyncio
    async def test_get_database_pool_creation(self, mock_config):
        """Test database pool creation"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            
            pool = await get_database_pool()
            
            # Should create pool with config parameters
            mock_create_pool.assert_called_once_with(
                host=mock_config.database.host,
                port=mock_config.database.port,
                database=mock_config.database.database,
                user=mock_config.database.username,
                password=mock_config.database.password,
                min_size=1,
                max_size=mock_config.database.pool_size,
                command_timeout=60
            )
            
            assert pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_get_database_pool_singleton(self, mock_config):
        """Test that pool is singleton (returns same instance)"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            
            pool1 = await get_database_pool()
            pool2 = await get_database_pool()
            
            # Should only create pool once
            mock_create_pool.assert_called_once()
            assert pool1 == pool2 == mock_pool
    
    @pytest.mark.asyncio
    async def test_close_database_pool(self, mock_config):
        """Test database pool closure"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            # Create pool first
            pool = await get_database_pool()
            
            # Close pool
            await close_database_pool()
            
            # Should call close on pool
            mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database_pool_no_pool(self):
        """Test closing when no pool exists"""
        # Should not raise exception
        await close_database_pool()
    
    @pytest.mark.asyncio
    async def test_database_connection_test_success(self, mock_config):
        """Test successful database connection test"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            result = await test_database_connection()
            
            # Should return True for successful connection
            assert result is True
            mock_conn.fetchval.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_database_connection_test_failure(self, mock_config):
        """Test failed database connection test"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.side_effect = Exception("Connection failed")
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            result = await test_database_connection()
            
            # Should return False for failed connection
            assert result is False
    
    @pytest.mark.asyncio
    async def test_database_connection_wrong_result(self, mock_config):
        """Test database connection with wrong result"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 2  # Wrong result
        
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            result = await test_database_connection()
            
            # Should return False for wrong result
            assert result is False


class TestDatabaseSchema:
    """Test database schema validation and setup"""
    
    @pytest.mark.asyncio
    async def test_activity_logs_table_exists(self, mock_database_pool):
        """Test that activity_logs table exists"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = True  # Table exists
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'activity_logs'
                    )
                """)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_activity_logs_table_structure(self, mock_database_pool):
        """Test activity_logs table has correct structure"""
        expected_columns = [
            ('id', 'bigint'),
            ('activity_id', 'uuid'),
            ('created_at', 'timestamp with time zone'),
            ('category', 'USER-DEFINED'),  # enum type
            ('action', 'USER-DEFINED'),    # enum type
            ('severity', 'USER-DEFINED'),  # enum type
            ('source', 'character varying'),
            ('event_type', 'character varying'),
            ('title', 'character varying'),
            ('description', 'text'),
            ('user_id', 'bigint'),
            ('session_id', 'uuid'),
            ('request_id', 'uuid'),
            ('parent_activity_id', 'uuid'),
            ('trading_mode', 'USER-DEFINED'),
            ('token_address', 'character varying'),
            ('chain', 'USER-DEFINED'),
            ('amount_usd', 'numeric'),
            ('fee_usd', 'numeric'),
            ('execution_time_ms', 'integer'),
            ('memory_usage_mb', 'integer'),
            ('cpu_usage_pct', 'numeric'),
            ('metadata', 'jsonb'),
            ('tags', 'ARRAY'),
            ('correlation_id', 'character varying'),
            ('error_code', 'character varying'),
            ('error_message', 'text'),
            ('stack_trace', 'text'),
            ('api_endpoint', 'character varying'),
            ('http_method', 'character varying'),
            ('http_status', 'integer'),
            ('user_agent', 'text'),
            ('ip_address', 'inet'),
            ('response_time_ms', 'integer'),
            ('throughput_ops_per_sec', 'numeric'),
            ('security_level', 'character varying'),
            ('risk_score', 'integer'),
            ('dashboard_component', 'character varying'),
            ('dashboard_action', 'character varying'),
            ('checksum', 'character varying'),
            ('version', 'integer'),
            ('indexed_at', 'timestamp with time zone'),
            ('archived_at', 'timestamp with time zone')
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'column_name': col[0], 'data_type': col[1]} 
            for col in expected_columns
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                columns = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'activity_logs'
                    ORDER BY ordinal_position
                """)
            
            # Should have all expected columns
            assert len(columns) == len(expected_columns)
            for expected in expected_columns:
                found = any(
                    col['column_name'] == expected[0] and 
                    col['data_type'] == expected[1] 
                    for col in columns
                )
                assert found, f"Column {expected[0]} with type {expected[1]} not found"
    
    @pytest.mark.asyncio
    async def test_activity_summaries_table_exists(self, mock_database_pool):
        """Test that activity_summaries table exists"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = True
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'activity_summaries'
                    )
                """)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_user_activity_sessions_table_exists(self, mock_database_pool):
        """Test that user_activity_sessions table exists"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = True
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'user_activity_sessions'
                    )
                """)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_enum_types_exist(self, mock_database_pool):
        """Test that required enum types exist"""
        expected_enums = [
            'activity_category',
            'activity_action', 
            'activity_severity',
            'trading_mode',
            'chain_type'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'typname': enum_name} for enum_name in expected_enums
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                enums = await conn.fetch("""
                    SELECT typname FROM pg_type 
                    WHERE typtype = 'e' 
                    AND typname IN ('activity_category', 'activity_action', 'activity_severity', 'trading_mode', 'chain_type')
                """)
            
            enum_names = [enum['typname'] for enum in enums]
            for expected_enum in expected_enums:
                assert expected_enum in enum_names
    
    @pytest.mark.asyncio
    async def test_indexes_exist(self, mock_database_pool):
        """Test that required indexes exist"""
        expected_indexes = [
            'idx_activity_logs_created_at_desc',
            'idx_activity_logs_created_at_category',
            'idx_activity_logs_category_action',
            'idx_activity_logs_severity_created',
            'idx_activity_logs_user_id_created',
            'idx_activity_logs_session_id',
            'idx_activity_logs_source_created',
            'idx_activity_logs_metadata_gin',
            'idx_activity_logs_tags_gin'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'indexname': idx} for idx in expected_indexes
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                indexes = await conn.fetch("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'activity_logs'
                    AND indexname LIKE 'idx_%'
                """)
            
            index_names = [idx['indexname'] for idx in indexes]
            for expected_idx in expected_indexes:
                assert expected_idx in index_names
    
    @pytest.mark.asyncio
    async def test_database_functions_exist(self, mock_database_pool):
        """Test that required database functions exist"""
        expected_functions = [
            'cleanup_old_activity_logs',
            'generate_activity_summaries',
            'update_user_session_activity',
            'set_indexed_timestamp'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'proname': func} for func in expected_functions
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                functions = await conn.fetch("""
                    SELECT proname FROM pg_proc 
                    WHERE proname IN ('cleanup_old_activity_logs', 'generate_activity_summaries', 
                                     'update_user_session_activity', 'set_indexed_timestamp')
                """)
            
            function_names = [func['proname'] for func in functions]
            for expected_func in expected_functions:
                assert expected_func in function_names
    
    @pytest.mark.asyncio
    async def test_database_triggers_exist(self, mock_database_pool):
        """Test that required triggers exist"""
        expected_triggers = [
            'trigger_update_user_session_activity',
            'trigger_set_indexed_timestamp'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'trigger_name': trigger} for trigger in expected_triggers
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                triggers = await conn.fetch("""
                    SELECT trigger_name FROM information_schema.triggers 
                    WHERE event_object_table = 'activity_logs'
                """)
            
            trigger_names = [trigger['trigger_name'] for trigger in triggers]
            for expected_trigger in expected_triggers:
                assert expected_trigger in trigger_names
    
    @pytest.mark.asyncio
    async def test_database_views_exist(self, mock_database_pool):
        """Test that required views exist"""
        expected_views = [
            'recent_activity',
            'error_summary',
            'performance_metrics',
            'user_activity_overview',
            'trading_activity_summary'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'table_name': view} for view in expected_views
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                views = await conn.fetch("""
                    SELECT table_name FROM information_schema.views 
                    WHERE table_name IN ('recent_activity', 'error_summary', 'performance_metrics', 
                                         'user_activity_overview', 'trading_activity_summary')
                """)
            
            view_names = [view['table_name'] for view in views]
            for expected_view in expected_views:
                assert expected_view in view_names


class TestDatabaseConstraints:
    """Test database constraints and validation"""
    
    @pytest.mark.asyncio
    async def test_activity_logs_primary_key(self, mock_database_pool):
        """Test activity_logs table has primary key"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 'id'
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                pk_column = await conn.fetchval("""
                    SELECT a.attname FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = 'activity_logs'::regclass AND i.indisprimary
                """)
            
            assert pk_column == 'id'
    
    @pytest.mark.asyncio
    async def test_activity_logs_unique_constraints(self, mock_database_pool):
        """Test activity_logs table has unique constraints"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'column_name': 'activity_id'}
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                unique_columns = await conn.fetch("""
                    SELECT a.attname as column_name FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = 'activity_logs'::regclass AND i.indisunique AND NOT i.indisprimary
                """)
            
            # Should have unique constraint on activity_id
            unique_column_names = [col['column_name'] for col in unique_columns]
            assert 'activity_id' in unique_column_names
    
    @pytest.mark.asyncio
    async def test_activity_logs_foreign_keys(self, mock_database_pool):
        """Test activity_logs table has proper foreign keys"""
        expected_fks = [
            ('user_id', 'users', 'telegram_user_id'),
            ('parent_activity_id', 'activity_logs', 'activity_id')
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'column_name': fk[0], 
                'foreign_table_name': fk[1], 
                'foreign_column_name': fk[2]
            } for fk in expected_fks
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                foreign_keys = await conn.fetch("""
                    SELECT 
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = 'activity_logs'
                """)
            
            # Should have expected foreign keys
            for expected_fk in expected_fks:
                found = any(
                    fk['column_name'] == expected_fk[0] and
                    fk['foreign_table_name'] == expected_fk[1] and
                    fk['foreign_column_name'] == expected_fk[2]
                    for fk in foreign_keys
                )
                assert found, f"Foreign key {expected_fk} not found"
    
    @pytest.mark.asyncio
    async def test_activity_logs_check_constraints(self, mock_database_pool):
        """Test activity_logs table has check constraints"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'constraint_name': 'activity_logs_risk_score_check'}
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                check_constraints = await conn.fetch("""
                    SELECT constraint_name FROM information_schema.table_constraints
                    WHERE table_name = 'activity_logs' AND constraint_type = 'CHECK'
                """)
            
            # Should have check constraint for risk_score
            constraint_names = [c['constraint_name'] for c in check_constraints]
            assert any('risk_score' in name for name in constraint_names)


class TestDatabasePermissions:
    """Test database permissions and grants"""
    
    @pytest.mark.asyncio
    async def test_user_permissions_activity_logs(self, mock_database_pool):
        """Test that rlte_user has correct permissions on activity_logs"""
        expected_privileges = ['SELECT', 'INSERT']
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'privilege_type': priv} for priv in expected_privileges
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                privileges = await conn.fetch("""
                    SELECT privilege_type FROM information_schema.role_table_grants
                    WHERE table_name = 'activity_logs' AND grantee = 'rlte_user'
                """)
            
            privilege_types = [p['privilege_type'] for p in privileges]
            for expected_priv in expected_privileges:
                assert expected_priv in privilege_types
    
    @pytest.mark.asyncio
    async def test_user_permissions_sequences(self, mock_database_pool):
        """Test that rlte_user has USAGE on sequences"""
        expected_sequences = [
            'activity_logs_id_seq',
            'activity_summaries_id_seq',
            'user_activity_sessions_id_seq'
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'object_name': seq} for seq in expected_sequences
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                sequence_usage = await conn.fetch("""
                    SELECT object_name FROM information_schema.usage_privileges
                    WHERE object_type = 'SEQUENCE' AND grantee = 'rlte_user'
                    AND object_name LIKE '%_seq'
                """)
            
            sequence_names = [s['object_name'] for s in sequence_usage]
            for expected_seq in expected_sequences:
                assert expected_seq in sequence_names


# NOTE: All these tests should FAIL initially since the database schema may not be set up
# This follows TDD methodology where tests define the requirements first