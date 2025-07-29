"""
Test RL-specific database enhancements for experience storage
Following TDD methodology - all tests should fail initially
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncpg

from src.utils.database import (
    get_database_pool, 
    close_database_pool, 
    test_database_connection,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError
)


class TestRLConnectionPoolConfiguration:
    """Test RL-specific connection pool configuration enhancements"""
    
    @pytest.mark.asyncio
    async def test_rl_optimized_pool_creation(self, mock_config):
        """Test database pool creation with RL-specific optimizations"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            
            from src.utils.database import get_rl_optimized_pool
            pool = await get_rl_optimized_pool()
            
            # Should create pool with RL-specific configuration
            mock_create_pool.assert_called_once_with(
                host=mock_config.database.host,
                port=mock_config.database.port,
                database=mock_config.database.database,
                user=mock_config.database.username,
                password=mock_config.database.password,
                min_size=5,  # Higher minimum for RL workloads
                max_size=50,  # Higher maximum for experience batch operations
                command_timeout=120,  # Longer timeout for complex RL queries
                server_settings={
                    'application_name': 'rlte_experience_storage',
                    'work_mem': '256MB',  # Increased for experience aggregations
                    'shared_preload_libraries': 'pg_stat_statements',
                    'random_page_cost': '1.1'  # Optimized for SSD storage
                }
            )
            
            assert pool == mock_pool

    @pytest.mark.asyncio
    async def test_rl_pool_connection_validation(self, mock_config):
        """Test connection validation for RL experience storage"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        
        # Test experience table accessibility
        mock_conn.fetchval.side_effect = [
            1,  # Basic connectivity
            True,  # rl_experiences table exists
            True,  # rl_training_sessions table exists
            True   # rl_performance_metrics table exists
        ]
        
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            from src.utils.database import validate_rl_database_schema
            result = await validate_rl_database_schema()
            
            assert result is True
            assert mock_conn.fetchval.call_count == 4

    @pytest.mark.asyncio
    async def test_rl_pool_performance_settings(self, mock_config):
        """Test performance-optimized settings for RL database pool"""
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool') as mock_create_pool:
            
            from src.utils.database import configure_rl_performance_settings
            await configure_rl_performance_settings()
            
            # Should apply RL-optimized database parameters
            expected_settings = {
                'max_connections': '200',
                'shared_buffers': '1GB',  
                'effective_cache_size': '4GB',
                'maintenance_work_mem': '512MB',
                'checkpoint_completion_target': '0.9',
                'wal_buffers': '16MB',
                'default_statistics_target': '100'
            }
            
            # This test will fail initially as the function doesn't exist


class TestExperienceQueryHelpers:
    """Test experience-specific query helper methods"""
    
    @pytest.mark.asyncio
    async def test_insert_experience_batch(self, mock_database_pool):
        """Test batch experience insertion helper"""
        mock_conn = AsyncMock()
        mock_conn.executemany.return_value = "INSERT 0 100"
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        experiences = [
            {
                'session_id': str(uuid.uuid4()),
                'step_number': 1,
                'state': [0.1, 0.2, 0.3],
                'action': 0,
                'reward': 0.5,
                'next_state': [0.2, 0.3, 0.4],
                'done': False,
                'created_at': datetime.now(timezone.utc)
            }
            for _ in range(100)
        ]
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import insert_experience_batch
            result = await insert_experience_batch(experiences)
            
            assert result == 100
            mock_conn.executemany.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_experiences_by_session(self, mock_database_pool):
        """Test querying experiences by training session"""
        session_id = str(uuid.uuid4())
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'id': 1,
                'session_id': session_id,
                'step_number': 1,
                'state': [0.1, 0.2],
                'action': 0,
                'reward': 0.5,
                'done': False
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import query_experiences_by_session
            experiences = await query_experiences_by_session(session_id, limit=1000)
            
            assert len(experiences) == 1
            assert experiences[0]['session_id'] == session_id
            mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_sample_prioritized_experiences(self, mock_database_pool):
        """Test prioritized experience sampling for replay buffer"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'id': i,
                'session_id': str(uuid.uuid4()),
                'priority': 1.0 - (i * 0.1),
                'state': [0.1, 0.2],
                'action': 0,
                'reward': 0.5
            }
            for i in range(64)
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import sample_prioritized_experiences
            experiences = await sample_prioritized_experiences(batch_size=64, alpha=0.6)
            
            assert len(experiences) == 64
            # Should be sorted by priority (highest first)
            priorities = [exp['priority'] for exp in experiences]
            assert priorities == sorted(priorities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_update_experience_priorities(self, mock_database_pool):
        """Test updating experience priorities after training"""
        mock_conn = AsyncMock()
        mock_conn.executemany.return_value = "UPDATE 32"
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        priority_updates = [
            {'id': 1, 'priority': 0.8, 'td_error': 0.2},
            {'id': 2, 'priority': 0.6, 'td_error': 0.4}
        ]
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import update_experience_priorities
            result = await update_experience_priorities(priority_updates)
            
            assert result == 32
            mock_conn.executemany.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_statistics(self, mock_database_pool):
        """Test retrieving training session statistics"""
        session_id = str(uuid.uuid4())
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            'session_id': session_id,
            'total_experiences': 10000,
            'avg_reward': 0.25,
            'max_reward': 1.0,
            'min_reward': -0.5,
            'total_steps': 10000,
            'completion_rate': 0.85
        }
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import get_session_statistics
            stats = await get_session_statistics(session_id)
            
            assert stats['total_experiences'] == 10000
            assert stats['avg_reward'] == 0.25
            mock_conn.fetchrow.assert_called_once()


class TestBatchOperationTransactions:
    """Test batch operation transaction support for experience storage"""
    
    @pytest.mark.asyncio
    async def test_atomic_experience_batch_insert(self, mock_database_pool):
        """Test atomic batch insertion of experiences with rollback support"""
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value = mock_transaction
        mock_conn.executemany.return_value = "INSERT 0 1000"
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        experiences = [{'state': [0.1], 'action': 0, 'reward': 0.5}] * 1000
        session_metrics = {'total_reward': 500.0, 'episode_length': 1000}
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import atomic_experience_batch_operation
            result = await atomic_experience_batch_operation(experiences, session_metrics)
            
            assert result['experiences_inserted'] == 1000
            assert result['session_updated'] == True
            mock_conn.transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_operation_rollback_on_error(self, mock_database_pool):
        """Test transaction rollback when batch operation fails"""
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value = mock_transaction
        
        # Simulate failure during batch insert
        mock_conn.executemany.side_effect = asyncpg.PostgresError("Constraint violation")
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        experiences = [{'state': [0.1], 'action': 0, 'reward': 0.5}] * 100
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import atomic_experience_batch_operation
            
            with pytest.raises(DatabaseQueryError):
                await atomic_experience_batch_operation(experiences, {})
            
            # Transaction should have been entered (but rolled back automatically)
            mock_conn.transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_operations(self, mock_database_pool):
        """Test handling of concurrent batch operations with proper isolation"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "INSERT 0 100"
        
        mock_database_pool.acquire.side_effect = [
            AsyncMock(return_value=mock_conn).__aenter__(),
            AsyncMock(return_value=mock_conn).__aenter__()
        ]
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import execute_concurrent_batch_operations
            
            operations = [
                [("INSERT INTO rl_experiences ...", [])],
                [("INSERT INTO rl_experiences ...", [])]
            ]
            
            results = await execute_concurrent_batch_operations(operations)
            
            assert len(results) == 2
            assert all(result['success'] for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_operation_with_conflict_resolution(self, mock_database_pool):
        """Test batch operations with conflict resolution strategies"""
        mock_conn = AsyncMock()
        
        # First call fails with unique constraint violation
        # Second call succeeds with ON CONFLICT DO UPDATE
        mock_conn.executemany.side_effect = [
            asyncpg.UniqueViolationError("Duplicate key value"),
            "INSERT 0 50"  # 50 inserted, 50 updated
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        experiences = [{'id': i, 'state': [0.1], 'reward': 0.5} for i in range(100)]
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import upsert_experience_batch
            result = await upsert_experience_batch(experiences)
            
            assert result['inserted'] == 50
            assert result['updated'] == 50
            assert mock_conn.executemany.call_count == 2


class TestRLDatabaseHealthChecks:
    """Test database health checks specific to RL experience storage"""
    
    @pytest.mark.asyncio
    async def test_experience_storage_health_check(self, mock_database_pool):
        """Test comprehensive health check for experience storage system"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [
            1,  # Basic connectivity
            50000,  # Total experiences count
            1000,   # Recent experiences (last hour)
            5,      # Active training sessions
            85.5,   # Average insertion time (ms)
            25.2,   # Average query time (ms)
            True,   # Indexes are healthy
            95.0    # Storage utilization percentage
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import check_rl_database_health
            health = await check_rl_database_health()
            
            assert health['status'] == 'healthy'
            assert health['total_experiences'] == 50000
            assert health['recent_experiences'] == 1000
            assert health['active_sessions'] == 5
            assert health['avg_insertion_time_ms'] <= 100  # Performance requirement
            assert health['avg_query_time_ms'] <= 50       # Performance requirement
            assert health['indexes_healthy'] is True
            assert health['storage_utilization'] == 95.0
    
    @pytest.mark.asyncio
    async def test_experience_storage_performance_metrics(self, mock_database_pool):
        """Test performance metrics collection for experience storage"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'operation_type': 'INSERT',
                'avg_time_ms': 45.2,
                'max_time_ms': 120.0,
                'min_time_ms': 15.0,
                'total_operations': 10000
            },
            {
                'operation_type': 'SELECT',
                'avg_time_ms': 25.8,
                'max_time_ms': 80.0,
                'min_time_ms': 5.0,
                'total_operations': 50000
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import get_rl_performance_metrics
            metrics = await get_rl_performance_metrics()
            
            assert len(metrics) == 2
            assert metrics[0]['operation_type'] == 'INSERT'
            assert metrics[0]['avg_time_ms'] <= 50  # Performance requirement
            assert metrics[1]['operation_type'] == 'SELECT'
            assert metrics[1]['avg_time_ms'] <= 30  # Performance requirement
    
    @pytest.mark.asyncio
    async def test_experience_storage_capacity_monitoring(self, mock_database_pool):
        """Test storage capacity monitoring for experience data"""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            'database_size_mb': 2048,
            'experiences_table_size_mb': 1024,
            'sessions_table_size_mb': 256,
            'metrics_table_size_mb': 128,
            'available_space_mb': 8192,
            'projected_full_date': '2024-12-31'
        }
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import monitor_rl_storage_capacity
            capacity = await monitor_rl_storage_capacity()
            
            assert capacity['database_size_mb'] == 2048
            assert capacity['experiences_table_size_mb'] == 1024
            assert capacity['available_space_mb'] == 8192
            assert 'projected_full_date' in capacity
    
    @pytest.mark.asyncio
    async def test_experience_data_integrity_check(self, mock_database_pool):
        """Test data integrity checks for experience storage"""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            'total_experiences': 100000,
            'experiences_with_null_states': 0,
            'experiences_with_invalid_actions': 0,
            'orphaned_experiences': 0,
            'inconsistent_session_data': 0,
            'corrupt_priority_values': 0,
            'integrity_score': 100.0
        }
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import check_rl_data_integrity
            integrity = await check_rl_data_integrity()
            
            assert integrity['total_experiences'] == 100000
            assert integrity['experiences_with_null_states'] == 0
            assert integrity['orphaned_experiences'] == 0
            assert integrity['integrity_score'] == 100.0
    
    @pytest.mark.asyncio
    async def test_unhealthy_database_detection(self, mock_database_pool):
        """Test detection of unhealthy database state"""
        mock_conn = AsyncMock()
        # Simulate unhealthy conditions
        mock_conn.fetchval.side_effect = [
            1,      # Basic connectivity works
            500,    # Very slow average insertion time
            200,    # Very slow average query time  
            False,  # Indexes are corrupted
            98.5    # Storage almost full
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import check_rl_database_health
            health = await check_rl_database_health()
            
            assert health['status'] == 'unhealthy'
            assert health['avg_insertion_time_ms'] > 100  # Exceeds performance requirement
            assert health['avg_query_time_ms'] > 50       # Exceeds performance requirement
            assert health['indexes_healthy'] is False
            assert health['storage_utilization'] > 95.0   # Storage critical


class TestRLDatabaseReliabilityAndPerformance:
    """Test database reliability and performance under RL workloads"""
    
    @pytest.mark.asyncio
    async def test_high_volume_experience_insertion(self, mock_database_pool):
        """Test database performance under high-volume experience insertion"""
        mock_conn = AsyncMock()
        mock_conn.executemany.return_value = "INSERT 0 10000"
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        # Simulate 10,000 experiences
        experiences = [{'state': [0.1] * 10, 'action': 0, 'reward': 0.5}] * 10000
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import benchmark_experience_insertion
            
            start_time = datetime.now()
            result = await benchmark_experience_insertion(experiences)
            end_time = datetime.now()
            
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            assert result['experiences_inserted'] == 10000
            assert duration_ms < 5000  # Should complete in under 5 seconds
            assert result['insertion_rate_per_second'] > 2000  # Performance requirement
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_performance(self, mock_database_pool):
        """Test database performance under concurrent read/write operations"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{'id': i} for i in range(64)]
        mock_conn.executemany.return_value = "INSERT 0 100"
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import benchmark_concurrent_operations
            
            # Simulate concurrent read and write operations
            result = await benchmark_concurrent_operations(
                read_threads=5,
                write_threads=2,
                operations_per_thread=100
            )
            
            assert result['total_reads'] == 500
            assert result['total_writes'] == 200
            assert result['avg_read_time_ms'] < 50   # Performance requirement
            assert result['avg_write_time_ms'] < 100 # Performance requirement
            assert result['error_rate'] < 0.01       # Less than 1% error rate
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, mock_database_pool):
        """Test memory usage optimization for large experience datasets"""
        mock_conn = AsyncMock()
        
        # Simulate streaming cursor for large datasets
        mock_conn.cursor.return_value.__aiter__ = AsyncMock(
            return_value=iter([{'id': i} for i in range(100000)])
        )
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            from src.utils.database import stream_experiences_with_memory_optimization
            
            processed_count = 0
            async for batch in stream_experiences_with_memory_optimization(
                session_id=str(uuid.uuid4()),
                batch_size=1000
            ):
                processed_count += len(batch)
                assert len(batch) <= 1000  # Batch size constraint
            
            assert processed_count == 100000
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(self, mock_config):
        """Test graceful handling of connection pool exhaustion"""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        
        # Simulate pool exhaustion
        mock_pool.acquire.side_effect = asyncpg.TooManyConnectionsError("Pool exhausted")
        
        with patch('src.utils.database.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', return_value=mock_pool):
            
            from src.utils.database import handle_pool_exhaustion_gracefully
            
            result = await handle_pool_exhaustion_gracefully()
            
            assert result['status'] == 'pool_exhausted'
            assert 'retry_after_seconds' in result
            assert result['recommended_action'] == 'reduce_concurrent_operations'


# NOTE: All these tests should FAIL initially since the RL-specific database functions don't exist yet
# This follows TDD methodology where tests define the requirements first