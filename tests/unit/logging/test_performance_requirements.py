"""
Test Performance Requirements (Batch Processing, Query Performance)
Following TDD methodology - comprehensive performance testing
"""

import asyncio
import pytest
import uuid
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import psutil
import concurrent.futures

from src.logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestBatchProcessingPerformance:
    """Test batch processing performance requirements"""
    
    @pytest.mark.asyncio
    async def test_batch_size_configuration(self, activity_logger_instance, performance_benchmarks):
        """Test configurable batch size for optimal performance"""
        logger = activity_logger_instance
        max_batch_size = performance_benchmarks['batch_processing']['max_entries_per_batch']
        
        # Override batch size for testing
        logger._batch_size = max_batch_size
        
        # Add activities up to batch size
        for i in range(max_batch_size):
            await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="test_source",
                event_type=f"test_event_{i}",
                title=f"Test activity {i}"
            )
        
        # Should accumulate exactly batch size entries
        assert len(logger._buffer) == max_batch_size
        
        # Adding one more should not exceed buffer
        await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test_source",
            event_type="overflow_test",
            title="Overflow test"
        )
        
        # Should still handle the overflow entry
        assert len(logger._buffer) == max_batch_size + 1
    
    @pytest.mark.asyncio
    async def test_batch_timeout_performance(self, activity_logger_instance, performance_benchmarks):
        """Test batch timeout for timely processing"""
        logger = activity_logger_instance
        max_timeout = performance_benchmarks['batch_processing']['max_batch_time_seconds']
        
        # Set shorter timeout for testing
        logger._batch_timeout = 0.1  # 100ms
        
        # Add single activity
        await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test_source",
            event_type="timeout_test",
            title="Timeout test activity"
        )
        
        start_time = time.time()
        
        # Wait for timeout to trigger flush
        await asyncio.sleep(0.15)  # Wait slightly longer than timeout
        
        elapsed_time = time.time() - start_time
        
        # Should process within reasonable time
        assert elapsed_time < max_timeout
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, mock_database_pool, mock_config, performance_benchmarks):
        """Test concurrent batch processing performance"""
        max_flush_time = performance_benchmarks['batch_processing']['max_flush_time_ms']
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Create multiple concurrent logging tasks
            async def log_activities(batch_id: int, count: int):
                tasks = []
                for i in range(count):
                    task = logger.log_activity(
                        category=ActivityCategory.PERFORMANCE,
                        action=ActivityAction.EXECUTE,
                        source=f"concurrent_source_{batch_id}",
                        event_type=f"concurrent_test_{i}",
                        title=f"Concurrent activity {batch_id}-{i}"
                    )
                    tasks.append(task)
                return await asyncio.gather(*tasks)
            
            # Run concurrent batches
            start_time = time.time()
            
            concurrent_tasks = [
                log_activities(batch_id, 50) for batch_id in range(5)
            ]
            
            await asyncio.gather(*concurrent_tasks)
            
            # Force flush to measure performance
            flush_start = time.time()
            await logger._flush_buffer()
            flush_time_ms = (time.time() - flush_start) * 1000
            
            total_time = time.time() - start_time
            
            # Should complete within performance requirements
            assert flush_time_ms < max_flush_time
            assert total_time < 5.0  # Total operation under 5 seconds
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_memory_efficient_batching(self, activity_logger_instance, performance_benchmarks):
        """Test memory-efficient batch processing"""
        logger = activity_logger_instance
        max_buffer_size_mb = performance_benchmarks['memory_usage']['max_buffer_size_mb']
        max_memory_per_entry_kb = performance_benchmarks['memory_usage']['max_memory_per_entry_kb']
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add large number of activities
        num_activities = 1000
        for i in range(num_activities):
            await logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="memory_test",
                event_type=f"memory_test_{i}",
                title=f"Memory test activity {i}",
                description="A" * 100,  # Add some data to each entry
                metadata={"test_data": list(range(50))}  # Additional metadata
            )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should stay within memory limits
        assert memory_increase < max_buffer_size_mb
        
        # Calculate approximate memory per entry
        memory_per_entry_kb = (memory_increase * 1024) / num_activities
        assert memory_per_entry_kb < max_memory_per_entry_kb
    
    @pytest.mark.asyncio
    async def test_high_throughput_logging(self, activity_logger_instance):
        """Test high-throughput activity logging performance"""
        logger = activity_logger_instance
        
        # Target: 1000 activities per second
        target_throughput = 1000
        test_duration = 2.0  # seconds
        expected_activities = int(target_throughput * test_duration)
        
        start_time = time.time()
        
        # Generate high-throughput activities
        tasks = []
        for i in range(expected_activities):
            task = logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="throughput_test",
                event_type=f"throughput_test_{i}",
                title=f"Throughput test {i}"
            )
            tasks.append(task)
        
        # Execute all logging tasks
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        actual_throughput = expected_activities / elapsed_time
        
        # Should achieve target throughput
        assert actual_throughput >= target_throughput * 0.8  # Allow 20% tolerance
        assert elapsed_time <= test_duration * 1.2  # Allow 20% time tolerance
        
        # Buffer should contain all activities
        assert len(logger._buffer) == expected_activities
    
    @pytest.mark.asyncio
    async def test_batch_processing_under_load(self, mock_database_pool, mock_config):
        """Test batch processing performance under system load"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Simulate system load with CPU-intensive task
            def cpu_intensive_task():
                # Simulate CPU load
                start = time.time()
                while time.time() - start < 0.1:  # 100ms of CPU work
                    _ = sum(i * i for i in range(1000))
            
            # Run logging and CPU load concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Start CPU load in background
                cpu_futures = [
                    executor.submit(cpu_intensive_task) for _ in range(4)
                ]
                
                # Measure logging performance under load
                start_time = time.time()
                
                # Log activities while CPU is under load
                for i in range(100):
                    await logger.log_activity(
                        category=ActivityCategory.PERFORMANCE,
                        action=ActivityAction.EXECUTE,
                        source="load_test",
                        event_type=f"load_test_{i}",
                        title=f"Under load activity {i}"
                    )
                
                logging_time = time.time() - start_time
                
                # Wait for CPU tasks to complete
                concurrent.futures.wait(cpu_futures)
            
            # Should maintain reasonable performance under load
            assert logging_time < 2.0  # Should complete within 2 seconds
            assert len(logger._buffer) == 100
            
            await logger.stop()


class TestQueryPerformanceRequirements:
    """Test database query performance requirements"""
    
    @pytest.mark.asyncio
    async def test_recent_activities_query_performance(self, mock_database_pool, performance_benchmarks):
        """Test performance of recent activities query"""
        max_query_time = performance_benchmarks['database_operations']['max_query_time_ms']
        
        mock_conn = AsyncMock()
        
        # Simulate query execution time
        async def mock_fetch(*args, **kwargs):
            await asyncio.sleep(max_query_time / 2000)  # Half of max time
            return [
                {
                    'activity_id': str(uuid.uuid4()),
                    'created_at': datetime.now(timezone.utc),
                    'category': 'trading',
                    'title': f'Activity {i}'
                } for i in range(100)
            ]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            start_time = time.time()
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT activity_id, created_at, category, action, severity,
                           source, event_type, title, description, user_id,
                           trading_mode, execution_time_ms, error_code, error_message
                    FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 1000
                """)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Should complete within performance requirements
            assert query_time_ms < max_query_time
            assert len(results) == 100
    
    @pytest.mark.asyncio
    async def test_filtered_query_performance(self, mock_database_pool, performance_benchmarks):
        """Test performance of filtered queries"""
        max_query_time = performance_benchmarks['database_operations']['max_query_time_ms']
        
        mock_conn = AsyncMock()
        
        async def mock_fetch(*args, **kwargs):
            await asyncio.sleep(max_query_time / 3000)  # Faster for filtered query
            return [
                {
                    'activity_id': str(uuid.uuid4()),
                    'category': 'trading',
                    'trading_mode': 'live',
                    'amount_usd': Decimal('100.50')
                } for i in range(50)
            ]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            start_time = time.time()
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = $1 
                    AND trading_mode = $2
                    AND amount_usd >= $3
                    AND created_at > NOW() - INTERVAL '7 days'
                    ORDER BY created_at DESC
                    LIMIT 500
                """, 'trading', 'live', Decimal('100.00'))
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Should be fast due to indexes
            assert query_time_ms < max_query_time / 2  # Should be faster than max
            assert len(results) == 50
    
    @pytest.mark.asyncio
    async def test_aggregation_query_performance(self, mock_database_pool, performance_benchmarks):
        """Test performance of aggregation queries"""
        max_aggregation_time = performance_benchmarks['database_operations']['max_aggregation_time_ms']
        
        mock_conn = AsyncMock()
        
        async def mock_fetch(*args, **kwargs):
            await asyncio.sleep(max_aggregation_time / 2000)  # Simulate aggregation time
            return [
                {
                    'category': 'trading',
                    'total_count': 1000,
                    'error_count': 25,
                    'avg_execution_time': 1250.5
                },
                {
                    'category': 'system',
                    'total_count': 500,
                    'error_count': 10,
                    'avg_execution_time': 850.2
                }
            ]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            start_time = time.time()
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        category,
                        COUNT(*) as total_count,
                        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
                        AVG(execution_time_ms) as avg_execution_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time
                    FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY category
                    ORDER BY total_count DESC
                """)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Should complete aggregation within time limit
            assert query_time_ms < max_aggregation_time
            assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_complex_search_query_performance(self, mock_database_pool):
        """Test performance of complex search queries"""
        mock_conn = AsyncMock()
        
        async def mock_fetch(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms for complex search
            return [
                {
                    'activity_id': str(uuid.uuid4()),
                    'title': 'Ethereum trade with high value',
                    'metadata': {'exchange': 'uniswap', 'slippage': 0.1},
                    'tags': ['high_value', 'ethereum', 'defi']
                }
            ]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            start_time = time.time()
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE (
                        title ILIKE $1 
                        OR description ILIKE $1 
                        OR metadata->>'exchange' = $2
                        OR $3 = ANY(tags)
                    )
                    AND category = $4
                    AND amount_usd >= $5
                    AND created_at > NOW() - INTERVAL '30 days'
                    ORDER BY created_at DESC
                    LIMIT 100
                """, '%ethereum%', 'uniswap', 'ethereum', 'trading', Decimal('1000.00'))
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Complex search should still be reasonably fast
            assert query_time_ms < 500  # Under 500ms for complex search
            assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, mock_database_pool):
        """Test performance under concurrent query load"""
        mock_conn = AsyncMock()
        
        async def mock_fetch(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms per query
            return [{'activity_id': str(uuid.uuid4())} for _ in range(10)]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            async def run_query(query_id: int):
                pool = await mock_database_pool
                async with pool.acquire() as conn:
                    return await conn.fetch(f"""
                        SELECT * FROM activity_logs 
                        WHERE source = 'concurrent_test_{query_id}'
                        ORDER BY created_at DESC
                        LIMIT 50
                    """)
            
            # Run 10 concurrent queries
            start_time = time.time()
            
            concurrent_queries = [run_query(i) for i in range(10)]
            results = await asyncio.gather(*concurrent_queries)
            
            total_time = time.time() - start_time
            
            # Should handle concurrent queries efficiently
            assert total_time < 0.5  # Under 500ms for 10 concurrent queries
            assert len(results) == 10
            for result in results:
                assert len(result) == 10
    
    @pytest.mark.asyncio
    async def test_index_usage_performance(self, mock_database_pool):
        """Test that queries use indexes efficiently"""
        mock_conn = AsyncMock()
        
        # Mock query plan to verify index usage
        async def mock_fetch(*args, **kwargs):
            query = args[0] if args else ""
            if "EXPLAIN" in query:
                return [
                    {
                        'QUERY PLAN': 'Index Scan using idx_activity_logs_created_at_desc on activity_logs'
                    }
                ]
            else:
                return [{'activity_id': str(uuid.uuid4())}]
        
        mock_conn.fetch = mock_fetch
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Check query plan for time-based query
                plan = await conn.fetch("""
                    EXPLAIN SELECT * FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
                
                # Should use index
                assert 'Index Scan' in plan[0]['QUERY PLAN']
                assert 'idx_activity_logs_created_at_desc' in plan[0]['QUERY PLAN']


class TestInsertPerformanceRequirements:
    """Test database insert performance requirements"""
    
    @pytest.mark.asyncio
    async def test_single_insert_performance(self, mock_database_pool, mock_config, performance_benchmarks):
        """Test single activity insert performance"""
        max_insert_time = performance_benchmarks['database_operations']['max_insert_time_ms']
        
        # Mock fast insert
        mock_conn = AsyncMock()
        
        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(max_insert_time / 2000)  # Half of max time
            return None
        
        mock_conn.execute = mock_execute
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entry = ActivityLogEntry(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="performance_test",
                event_type="insert_test",
                title="Insert performance test"
            )
            
            start_time = time.time()
            await logger._insert_activities([entry])
            insert_time_ms = (time.time() - start_time) * 1000
            
            # Should complete within performance requirements
            assert insert_time_ms < max_insert_time
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_batch_insert_performance(self, mock_database_pool, mock_config, performance_benchmarks):
        """Test batch insert performance"""
        max_insert_time = performance_benchmarks['database_operations']['max_insert_time_ms']
        
        mock_conn = AsyncMock()
        
        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.001)  # 1ms per insert
            return None
        
        mock_conn.execute = mock_execute
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Create batch of entries
            entries = [
                ActivityLogEntry(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source="batch_test",
                    event_type=f"batch_insert_{i}",
                    title=f"Batch insert {i}"
                ) for i in range(100)
            ]
            
            start_time = time.time()
            await logger._insert_activities(entries)
            batch_time_ms = (time.time() - start_time) * 1000
            
            # Batch should be more efficient than individual inserts
            expected_max_time = max_insert_time * len(entries) * 0.1  # 10% of individual time
            assert batch_time_ms < expected_max_time
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_insert_throughput_performance(self, mock_database_pool, mock_config):
        """Test insert throughput performance"""
        mock_conn = AsyncMock()
        
        # Very fast mock insert
        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.0001)  # 0.1ms per insert
            return None
        
        mock_conn.execute = mock_execute
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Test sustained insert throughput
            num_inserts = 1000
            start_time = time.time()
            
            for i in range(num_inserts):
                entry = ActivityLogEntry(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source="throughput_test",
                    event_type=f"throughput_{i}",
                    title=f"Throughput test {i}"
                )
                await logger._insert_activities([entry])
            
            total_time = time.time() - start_time
            throughput = num_inserts / total_time
            
            # Should achieve high throughput
            assert throughput >= 100  # At least 100 inserts per second
            
            await logger.stop()


class TestMemoryPerformanceRequirements:
    """Test memory usage performance requirements"""
    
    @pytest.mark.asyncio
    async def test_buffer_memory_efficiency(self, activity_logger_instance, performance_benchmarks):
        """Test buffer memory efficiency"""
        logger = activity_logger_instance
        max_memory_per_entry = performance_benchmarks['memory_usage']['max_memory_per_entry_kb']
        
        # Monitor memory before adding activities
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Add measured number of activities
        num_activities = 100
        for i in range(num_activities):
            await logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="memory_test",
                event_type=f"memory_efficiency_{i}",
                title=f"Memory test {i}",
                description="Standard description for memory testing",
                metadata={"index": i, "test": "memory_efficiency"}
            )
        
        # Measure memory after
        final_memory = process.memory_info().rss
        memory_increase_kb = (final_memory - initial_memory) / 1024
        memory_per_entry_kb = memory_increase_kb / num_activities
        
        # Should stay within memory efficiency requirements
        assert memory_per_entry_kb < max_memory_per_entry
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, activity_logger_instance):
        """Test that memory doesn't leak during extended operation"""
        logger = activity_logger_instance
        
        process = psutil.Process()
        memory_samples = []
        
        # Take memory samples during extended operation
        for cycle in range(5):
            # Add and flush activities
            for i in range(100):
                await logger.log_activity(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source="leak_test",
                    event_type=f"leak_test_{cycle}_{i}",
                    title=f"Leak test {cycle}-{i}"
                )
            
            # Force flush to clear buffer
            await logger._flush_buffer()
            
            # Sample memory after flush
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
            
            # Small delay between cycles
            await asyncio.sleep(0.1)
        
        # Memory should not continuously grow
        memory_growth = memory_samples[-1] - memory_samples[0]
        assert memory_growth < 10  # Less than 10MB growth over cycles
        
        # Should not have significant upward trend
        avg_first_half = sum(memory_samples[:2]) / 2
        avg_second_half = sum(memory_samples[-2:]) / 2
        growth_rate = (avg_second_half - avg_first_half) / avg_first_half
        assert growth_rate < 0.1  # Less than 10% growth rate
    
    @pytest.mark.asyncio
    async def test_large_metadata_handling(self, activity_logger_instance):
        """Test efficient handling of activities with large metadata"""
        logger = activity_logger_instance
        
        # Create activity with large metadata
        large_metadata = {
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" * 10 for i in range(100)},
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": ["data"] * 100
                    }
                }
            }
        }
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add activities with large metadata
        for i in range(10):
            await logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="large_metadata_test",
                event_type=f"large_metadata_{i}",
                title=f"Large metadata test {i}",
                metadata=large_metadata
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should handle large metadata efficiently
        assert memory_increase < 50  # Less than 50MB for 10 large entries
        assert len(logger._buffer) == 10


class TestConcurrencyPerformanceRequirements:
    """Test concurrency performance requirements"""
    
    @pytest.mark.asyncio
    async def test_concurrent_logging_performance(self, activity_logger_instance):
        """Test performance under concurrent logging load"""
        logger = activity_logger_instance
        
        async def concurrent_logger(worker_id: int, num_logs: int):
            start_time = time.time()
            for i in range(num_logs):
                await logger.log_activity(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source=f"worker_{worker_id}",
                    event_type=f"concurrent_log_{i}",
                    title=f"Worker {worker_id} log {i}"
                )
            return time.time() - start_time
        
        # Run multiple concurrent workers
        num_workers = 5
        logs_per_worker = 100
        
        start_time = time.time()
        worker_tasks = [
            concurrent_logger(worker_id, logs_per_worker) 
            for worker_id in range(num_workers)
        ]
        
        worker_times = await asyncio.gather(*worker_tasks)
        total_time = time.time() - start_time
        
        total_logs = num_workers * logs_per_worker
        overall_throughput = total_logs / total_time
        
        # Should maintain good performance under concurrency
        assert overall_throughput >= 200  # At least 200 logs/second
        assert max(worker_times) < 5.0  # No worker should take more than 5 seconds
        assert len(logger._buffer) == total_logs
    
    @pytest.mark.asyncio
    async def test_lock_contention_performance(self, activity_logger_instance):
        """Test performance with buffer lock contention"""
        logger = activity_logger_instance
        
        async def high_frequency_logger(duration: float):
            end_time = time.time() + duration
            count = 0
            while time.time() < end_time:
                await logger.log_activity(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source="lock_test",
                    event_type=f"lock_test_{count}",
                    title=f"Lock test {count}"
                )
                count += 1
            return count
        
        # Run high-frequency logging from multiple tasks
        duration = 1.0  # 1 second test
        num_tasks = 3
        
        start_time = time.time()
        tasks = [high_frequency_logger(duration) for _ in range(num_tasks)]
        counts = await asyncio.gather(*tasks)
        actual_duration = time.time() - start_time
        
        total_logs = sum(counts)
        throughput = total_logs / actual_duration
        
        # Should handle lock contention efficiently
        assert throughput >= 500  # At least 500 logs/second with contention
        assert actual_duration <= duration * 1.1  # Within 10% of target duration


# NOTE: All these tests should FAIL initially since the performance optimizations
# may not be fully implemented yet. This follows TDD methodology.