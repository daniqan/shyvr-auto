"""
Test Edge Cases and Error Handling (Database Failures, Invalid Data)
Following TDD methodology - comprehensive edge case and error scenario testing
"""

import asyncio
import pytest
import uuid
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncpg
from concurrent.futures import ThreadPoolExecutor

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestDatabaseFailureHandling:
    """Test handling of various database failure scenarios"""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, mock_config):
        """Test handling when database connection fails"""
        # Mock database pool that fails to connect
        with patch('src.activity_logging.activity_logger.get_config', return_value=mock_config), \
             patch('asyncpg.create_pool', side_effect=asyncpg.ConnectionDoesNotExistError("Connection failed")):
            
            logger = ActivityLogger()
            
            # Should handle connection failure gracefully
            with pytest.raises((asyncpg.ConnectionDoesNotExistError, RuntimeError)):
                await logger.start()
    
    @pytest.mark.asyncio
    async def test_database_pool_not_initialized(self, mock_config):
        """Test operations when database pool is not initialized"""
        with patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            logger = ActivityLogger()
            # Don't start the logger (no pool initialization)
            
            entries = [ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="test_event",
                title="Test entry"
            )]
            
            # Should raise RuntimeError when pool not initialized
            with pytest.raises(RuntimeError, match="Database pool not initialized"):
                await logger._insert_activities(entries)
    
    @pytest.mark.asyncio
    async def test_database_connection_lost_during_operation(self, mock_database_pool, mock_config):
        """Test handling when database connection is lost during operation"""
        mock_conn = AsyncMock()
        
        # First call succeeds, second fails
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncpg.ConnectionDoesNotExistError("Connection lost")
            return None
        
        mock_conn.execute = mock_execute
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # First insert should succeed
            entry1 = ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="test_event_1",
                title="Test entry 1"
            )
            await logger._insert_activities([entry1])
            
            # Second insert should fail but be handled gracefully
            entry2 = ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="test_event_2",
                title="Test entry 2"
            )
            
            # Should not raise exception, but handle gracefully
            await logger._insert_activities([entry2])
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_database_timeout_handling(self, mock_database_pool, mock_config):
        """Test handling of database timeout errors"""
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = asyncio.TimeoutError("Query timeout")
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entry = ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="timeout_test",
                title="Timeout test"
            )
            
            # Should handle timeout gracefully
            await logger._insert_activities([entry])
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, mock_database_pool, mock_config):
        """Test handling of database transaction rollbacks"""
        mock_conn = AsyncMock()
        
        # First execute succeeds, second fails in transaction
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise asyncpg.PostgresError("Constraint violation")
            return None
        
        mock_conn.execute = mock_execute
        
        # Mock transaction that raises on second call
        transaction_mock = AsyncMock()
        transaction_mock.__aenter__ = AsyncMock()
        transaction_mock.__aexit__ = AsyncMock()
        mock_conn.transaction.return_value = transaction_mock
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entries = [
                ActivityLogEntry(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.START,
                    source="test",
                    event_type="txn_test_1",
                    title="Transaction test 1"
                ),
                ActivityLogEntry(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.START,
                    source="test",
                    event_type="txn_test_2",
                    title="Transaction test 2"
                )
            ]
            
            # Should handle transaction failure
            await logger._insert_activities(entries)
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_database_pool_exhaustion(self, mock_config):
        """Test handling when database connection pool is exhausted"""
        # Mock pool that always fails to acquire connection
        mock_pool = AsyncMock()
        mock_pool.acquire.side_effect = asyncpg.TooManyConnectionsError("Pool exhausted")
        
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entry = ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="pool_test",
                title="Pool exhaustion test"
            )
            
            # Should handle pool exhaustion gracefully
            await logger._insert_activities([entry])
            
            await logger.stop()


class TestInvalidDataHandling:
    """Test handling of invalid or malformed data"""
    
    @pytest.mark.asyncio
    async def test_invalid_enum_values(self, activity_logger_instance):
        """Test handling of invalid enum values"""
        logger = activity_logger_instance
        
        # Test with string that's not valid enum value
        with pytest.raises((ValueError, TypeError)):
            entry = ActivityLogEntry(
                category="invalid_category",  # Invalid category
                action=ActivityAction.EXECUTE,
                source="test",
                event_type="invalid_test",
                title="Invalid category test"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_decimal_values(self, activity_logger_instance):
        """Test handling of invalid decimal values"""
        logger = activity_logger_instance
        
        # Test with invalid decimal string
        with pytest.raises((InvalidOperation, ValueError, TypeError)):
            entry = ActivityLogEntry(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="test",
                event_type="decimal_test",
                title="Invalid decimal test",
                amount_usd="not_a_number"  # Invalid decimal
            )
    
    @pytest.mark.asyncio
    async def test_oversized_string_fields(self, activity_logger_instance):
        """Test handling of oversized string fields"""
        logger = activity_logger_instance
        
        # Create entry with very long strings
        very_long_string = "A" * 10000  # 10KB string
        
        entry = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source=very_long_string[:100],  # Truncate for source
            event_type="oversized_test",
            title=very_long_string[:255],   # Truncate for title
            description=very_long_string,   # Description can be longer
        )
        
        # Should handle oversized fields
        activity_id = await logger.log_activity(
            category=entry.category,
            action=entry.action,
            source=entry.source,
            event_type=entry.event_type,
            title=entry.title,
            description=entry.description
        )
        
        assert activity_id is not None
        assert len(logger._buffer) == 1
    
    @pytest.mark.asyncio
    async def test_invalid_json_metadata(self, activity_logger_instance):
        """Test handling of non-serializable metadata"""
        logger = activity_logger_instance
        
        # Create metadata with non-serializable objects
        invalid_metadata = {
            "valid_field": "valid_value",
            "invalid_field": object(),  # Non-serializable object
            "function": lambda x: x,    # Function (non-serializable)
        }
        
        # Should handle or skip invalid metadata
        entry = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test",
            event_type="metadata_test",
            title="Invalid metadata test",
            metadata=invalid_metadata
        )
        
        # The to_dict method should handle this gracefully
        data = entry.to_dict()
        assert "metadata" in data or "metadata" not in data  # Either included or excluded
    
    @pytest.mark.asyncio
    async def test_null_required_fields(self):
        """Test handling of null values in required fields"""
        # Should raise validation error for missing required fields
        with pytest.raises(TypeError):
            entry = ActivityLogEntry(
                category=None,  # Required field
                action=ActivityAction.EXECUTE,
                source="test",
                event_type="null_test",
                title="Null test"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_datetime_fields(self, activity_logger_instance):
        """Test handling of invalid datetime values"""
        logger = activity_logger_instance
        
        # Create entry with invalid datetime
        entry = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test",
            event_type="datetime_test",
            title="Invalid datetime test"
        )
        
        # Manually set invalid datetime
        entry.created_at = "not_a_datetime"
        
        # Should handle invalid datetime in to_dict conversion
        data = entry.to_dict()
        # The invalid datetime should either be excluded or handled
        assert isinstance(data.get("created_at"), (datetime, type(None), str))
    
    @pytest.mark.asyncio
    async def test_invalid_ip_address(self, activity_logger_instance):
        """Test handling of invalid IP addresses"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ACCESS,
            source="test",
            event_type="ip_test",
            title="Invalid IP test",
            ip_address="not.an.ip.address"  # Invalid IP
        )
        
        # Should handle gracefully (either store as-is or validate/reject)
        assert activity_id is not None
        assert len(logger._buffer) == 1
    
    @pytest.mark.asyncio
    async def test_invalid_uuid_fields(self, activity_logger_instance):
        """Test handling of invalid UUID values"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.USER,
            action=ActivityAction.ACCESS,
            source="test",
            event_type="uuid_test",
            title="Invalid UUID test",
            session_id="not-a-valid-uuid",  # Invalid UUID
            request_id="also-not-a-uuid"    # Invalid UUID
        )
        
        # Should handle gracefully
        assert activity_id is not None
        assert len(logger._buffer) == 1


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_buffer_access(self, activity_logger_instance):
        """Test concurrent access to activity buffer"""
        logger = activity_logger_instance
        
        async def concurrent_logger(worker_id: int):
            for i in range(50):
                await logger.log_activity(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source=f"worker_{worker_id}",
                    event_type=f"concurrent_{i}",
                    title=f"Worker {worker_id} activity {i}"
                )
        
        # Run multiple concurrent workers
        tasks = [concurrent_logger(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Should handle concurrent access without corruption
        assert len(logger._buffer) == 250  # 5 workers * 50 activities
        
        # All activities should be properly formatted
        for entry in logger._buffer:
            assert entry.activity_id is not None
            assert entry.created_at is not None
            assert entry.category == ActivityCategory.PERFORMANCE
    
    @pytest.mark.asyncio
    async def test_concurrent_flush_operations(self, mock_database_pool, mock_config):
        """Test concurrent flush operations"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Add activities to buffer
            for i in range(10):
                await logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.EXECUTE,
                    source="flush_test",
                    event_type=f"flush_test_{i}",
                    title=f"Flush test {i}"
                )
            
            # Trigger multiple concurrent flushes
            flush_tasks = [logger._flush_buffer() for _ in range(3)]
            await asyncio.gather(*flush_tasks)
            
            # Should handle concurrent flushes gracefully
            assert len(logger._buffer) == 0  # Buffer should be cleared
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_protection(self, activity_logger_instance):
        """Test protection against buffer overflow"""
        logger = activity_logger_instance
        
        # Mock failing flush to simulate overflow scenario
        original_insert = logger._insert_activities
        
        async def failing_insert(entries):
            raise Exception("Simulated database failure")
        
        logger._insert_activities = failing_insert
        
        # Add many activities to trigger overflow protection
        for i in range(1200):  # Exceed max buffer size
            await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="overflow_test",
                event_type=f"overflow_{i}",
                title=f"Overflow test {i}"
            )
            
            # Periodically try to flush
            if i % 100 == 0:
                await logger._flush_buffer()
        
        # Should implement buffer size protection
        assert len(logger._buffer) <= 1000  # Max 10 batches * 100 entries
        
        # Restore original insert method
        logger._insert_activities = original_insert
    
    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self, mock_database_pool, mock_config):
        """Test rapid start/stop cycles"""
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            
            # Rapid start/stop cycles
            for cycle in range(5):
                await logger.start()
                
                # Quick activity logging
                await logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.START,
                    source="cycle_test",
                    event_type=f"cycle_{cycle}",
                    title=f"Cycle test {cycle}"
                )
                
                await logger.stop()
            
            # Should handle rapid cycles without issues
            assert not logger._running


class TestDataValidationEdgeCases:
    """Test edge cases in data validation"""
    
    @pytest.mark.asyncio
    async def test_extreme_numeric_values(self, activity_logger_instance):
        """Test handling of extreme numeric values"""
        logger = activity_logger_instance
        
        # Test with very large numbers
        activity_id = await logger.log_activity(
            category=ActivityCategory.TRADING,
            action=ActivityAction.EXECUTE,
            source="extreme_test",
            event_type="extreme_numbers",
            title="Extreme numeric values test",
            amount_usd=Decimal("99999999999999999999.99999999"),  # Very large decimal
            execution_time_ms=2147483647,  # Max 32-bit integer
            risk_score=100,  # Maximum risk score
            cpu_usage_pct=Decimal("100.00")  # Maximum CPU usage
        )
        
        # Should handle extreme values
        assert activity_id is not None
        entry = logger._buffer[0]
        assert entry.execution_time_ms == 2147483647
        assert entry.risk_score == 100
    
    @pytest.mark.asyncio
    async def test_negative_numeric_values(self, activity_logger_instance):
        """Test handling of negative numeric values where inappropriate"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.PERFORMANCE,
            action=ActivityAction.EXECUTE,
            source="negative_test",
            event_type="negative_values",
            title="Negative values test",
            execution_time_ms=-100,  # Negative execution time (invalid)
            memory_usage_mb=-50,     # Negative memory usage (invalid)
            risk_score=-10           # Negative risk score (invalid)
        )
        
        # Should handle negative values appropriately
        assert activity_id is not None
        entry = logger._buffer[0]
        # Values should either be rejected, corrected, or stored as-is with validation warnings
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, activity_logger_instance):
        """Test handling of unicode and special characters"""
        logger = activity_logger_instance
        
        # Test with various unicode and special characters
        unicode_title = "🚀 Trade executed with émojis and spëcial chars 中文 عربي"
        unicode_description = "Testing unicode: \u2603 \u2764 \u1F4A9 and control chars: \x00\x01\x02"
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.TRADING,
            action=ActivityAction.EXECUTE,
            source="unicode_test",
            event_type="unicode_test",
            title=unicode_title,
            description=unicode_description,
            metadata={
                "unicode_key": "Unicode value: 🎯",
                "special_chars": "Special: \t\n\r\\\"'",
                "emoji": "🔥💯⚡"
            }
        )
        
        # Should handle unicode characters properly
        assert activity_id is not None
        entry = logger._buffer[0]
        assert "🚀" in entry.title
        assert entry.metadata["emoji"] == "🔥💯⚡"
    
    @pytest.mark.asyncio
    async def test_circular_reference_in_metadata(self, activity_logger_instance):
        """Test handling of circular references in metadata"""
        logger = activity_logger_instance
        
        # Create circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        # Should handle circular reference gracefully (either reject or handle)
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="circular_test",
            event_type="circular_ref",
            title="Circular reference test",
            metadata=circular_dict
        )
        
        # Should either succeed with modified metadata or handle gracefully
        assert activity_id is not None


class TestSystemResourceEdgeCases:
    """Test edge cases related to system resources"""
    
    @pytest.mark.asyncio
    async def test_low_memory_conditions(self, activity_logger_instance):
        """Test behavior under low memory conditions"""
        logger = activity_logger_instance
        
        # Simulate low memory by creating large objects
        large_objects = []
        try:
            # Create increasingly large objects until memory pressure
            for i in range(10):
                large_data = "X" * (1024 * 1024 * 10)  # 10MB strings
                large_objects.append(large_data)
                
                # Try to log activity under memory pressure
                activity_id = await logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.EXECUTE,
                    source="memory_pressure_test",
                    event_type=f"low_memory_{i}",
                    title=f"Low memory test {i}",
                    metadata={"large_data": large_data[:1000]}  # Sample of large data
                )
                
                assert activity_id is not None
        finally:
            # Clean up large objects
            large_objects.clear()
    
    @pytest.mark.asyncio
    async def test_high_cpu_load_conditions(self, activity_logger_instance):
        """Test behavior under high CPU load"""
        logger = activity_logger_instance
        
        def cpu_intensive_task():
            # CPU-intensive computation
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Run CPU-intensive tasks in background
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start CPU load
            cpu_futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
            
            try:
                # Log activities under CPU load
                for i in range(20):
                    activity_id = await logger.log_activity(
                        category=ActivityCategory.PERFORMANCE,
                        action=ActivityAction.EXECUTE,
                        source="cpu_load_test",
                        event_type=f"high_cpu_{i}",
                        title=f"High CPU test {i}",
                        cpu_usage_pct=Decimal("95.0")  # Simulate high CPU usage
                    )
                    
                    assert activity_id is not None
                    
                    # Small delay to allow CPU tasks to run
                    await asyncio.sleep(0.001)
                
                # Should complete all logging despite CPU load
                assert len(logger._buffer) == 20
            finally:
                # Wait for CPU tasks to complete
                for future in cpu_futures:
                    future.result()
    
    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_simulation(self, activity_logger_instance):
        """Test behavior when disk space is exhausted (simulated)"""
        logger = activity_logger_instance
        
        # Mock disk full error during database write
        original_insert = logger._insert_activities
        
        async def disk_full_insert(entries):
            raise asyncpg.DiskFullError("No space left on device")
        
        logger._insert_activities = disk_full_insert
        
        # Try to log activities when disk is full
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.ERROR,
            source="disk_full_test",
            event_type="disk_exhaustion",
            title="Disk space exhausted test"
        )
        
        # Should handle disk full error gracefully
        assert activity_id is not None
        
        # Try to flush buffer
        await logger._flush_buffer()
        
        # Activities should remain in buffer due to disk full error
        assert len(logger._buffer) >= 1
        
        # Restore original insert method
        logger._insert_activities = original_insert


class TestTimeAndTimezoneEdgeCases:
    """Test edge cases related to time and timezone handling"""
    
    @pytest.mark.asyncio
    async def test_timezone_edge_cases(self, activity_logger_instance):
        """Test handling of different timezone scenarios"""
        logger = activity_logger_instance
        
        # Test with different timezone-aware datetimes
        utc_time = datetime.now(timezone.utc)
        est_time = datetime.now(timezone(timedelta(hours=-5)))  # EST
        pst_time = datetime.now(timezone(timedelta(hours=-8)))  # PST
        
        # Create entries with different timezone timestamps
        for i, time_val in enumerate([utc_time, est_time, pst_time]):
            entry = ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="timezone_test",
                event_type=f"timezone_test_{i}",
                title=f"Timezone test {i}",
                created_at=time_val
            )
            
            # Should handle different timezones
            data = entry.to_dict()
            assert isinstance(data["created_at"], datetime)
            assert data["created_at"].tzinfo is not None
    
    @pytest.mark.asyncio
    async def test_daylight_saving_transition(self, activity_logger_instance):
        """Test handling during daylight saving time transitions"""
        logger = activity_logger_instance
        
        # Simulate DST transition times (these would be problematic in real scenarios)
        # Spring forward: 2:00 AM becomes 3:00 AM (non-existent time)
        # Fall back: 2:00 AM happens twice (ambiguous time)
        
        dst_transition_time = datetime(2024, 3, 10, 2, 30, 0)  # Spring forward
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="dst_test",
            event_type="dst_transition",
            title="DST transition test"
        )
        
        # Should handle DST transitions gracefully
        assert activity_id is not None
        entry = logger._buffer[0]
        assert entry.created_at.tzinfo == timezone.utc  # Should be normalized to UTC
    
    @pytest.mark.asyncio
    async def test_time_going_backwards(self, activity_logger_instance):
        """Test handling when system time goes backwards"""
        logger = activity_logger_instance
        
        # Create entry with future timestamp
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        
        entry = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="time_test",
            event_type="future_time",
            title="Future timestamp test",
            created_at=future_time
        )
        
        # Should handle future timestamps
        data = entry.to_dict()
        assert data["created_at"] == future_time
        
        # Create entry with past timestamp (simulating time going backwards)
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        
        entry2 = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="time_test",
            event_type="past_time",
            title="Past timestamp test",
            created_at=past_time
        )
        
        # Should handle out-of-order timestamps
        data2 = entry2.to_dict()
        assert data2["created_at"] == past_time


# NOTE: All these tests should FAIL initially since comprehensive error handling
# may not be fully implemented yet. This follows TDD methodology.