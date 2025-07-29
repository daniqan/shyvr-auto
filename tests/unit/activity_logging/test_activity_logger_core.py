"""
Test ActivityLogger Core Functionality
Following TDD methodology - all tests should fail initially since implementation is incomplete
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import json

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType, activity_logger
)


class TestActivityLogEntry:
    """Test ActivityLogEntry data class functionality"""
    
    def test_activity_log_entry_creation(self, sample_activity_entry):
        """Test basic ActivityLogEntry creation and auto-generation"""
        entry = sample_activity_entry
        
        # Should auto-generate required fields
        assert entry.activity_id is not None
        assert entry.created_at is not None
        assert isinstance(entry.activity_id, str)
        assert isinstance(entry.created_at, datetime)
        
        # Should preserve provided fields
        assert entry.category == ActivityCategory.TRADING
        assert entry.action == ActivityAction.EXECUTE
        assert entry.source == "trading_engine"
        assert entry.title == "Buy order executed"
        
    def test_activity_log_entry_auto_fields(self):
        """Test auto-generation of activity_id and timestamps"""
        entry = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test",
            event_type="test_event",
            title="Test entry"
        )
        
        # Should auto-generate activity_id
        assert entry.activity_id is not None
        assert len(entry.activity_id) == 36  # UUID4 format
        
        # Should auto-generate created_at
        assert entry.created_at is not None
        assert isinstance(entry.created_at, datetime)
        assert entry.created_at.tzinfo == timezone.utc
        
        # Should auto-generate request_id when session_id provided
        entry_with_session = ActivityLogEntry(
            category=ActivityCategory.USER,
            action=ActivityAction.LOGIN,
            source="dashboard",
            event_type="user_login",
            title="User login",
            session_id=str(uuid.uuid4())
        )
        assert entry_with_session.request_id is not None
        
    def test_activity_log_entry_to_dict(self, sample_activity_entry):
        """Test conversion to dictionary for database insertion"""
        entry = sample_activity_entry
        data = entry.to_dict()
        
        # Should include all non-None fields
        assert "activity_id" in data
        assert "category" in data
        assert "action" in data
        assert "source" in data
        assert "event_type" in data
        assert "title" in data
        
        # Enums should be converted to values
        assert data["category"] == "trading"
        assert data["action"] == "execute"
        assert data["severity"] == "info"  # default value
        assert data["trading_mode"] == "simulation"
        assert data["chain"] == "ethereum"
        
        # Decimal should be preserved
        assert isinstance(data["amount_usd"], Decimal)
        assert data["amount_usd"] == Decimal("100.50")
        
        # Lists and dicts should be preserved
        assert data["metadata"] == {"slippage": 0.1, "gas_fee": 2.5}
        
        # None values should be excluded
        assert "tags" not in data  # was None
        
    def test_activity_log_entry_checksum_generation(self, sample_activity_entry):
        """Test checksum generation for data integrity"""
        entry = sample_activity_entry
        checksum = entry.generate_checksum()
        
        # Should generate valid checksum
        assert checksum is not None
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex
        
        # Same entry should generate same checksum
        checksum2 = entry.generate_checksum()
        assert checksum == checksum2
        
        # Different entry should generate different checksum
        entry2 = ActivityLogEntry(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="different_source",
            event_type="different_event",
            title="Different title"
        )
        checksum3 = entry2.generate_checksum()
        assert checksum != checksum3


class TestActivityLoggerInitialization:
    """Test ActivityLogger initialization and lifecycle"""
    
    @pytest.mark.asyncio
    async def test_activity_logger_initialization(self, mock_database_pool, mock_config):
        """Test ActivityLogger initialization"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            
            # Should initialize with default values
            assert logger._batch_size == 100
            assert logger._batch_timeout == 5.0
            assert logger._buffer == []
            assert logger._running is False
            assert logger._pool is None
            assert logger._flush_task is None
    
    @pytest.mark.asyncio
    async def test_activity_logger_start(self, mock_database_pool, mock_config):
        """Test ActivityLogger start functionality"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Should be marked as running
            assert logger._running is True
            assert logger._pool == mock_database_pool
            assert logger._flush_task is not None
            
            # Clean up
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_activity_logger_stop(self, mock_database_pool, mock_config):
        """Test ActivityLogger stop functionality"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            await logger.stop()
            
            # Should be marked as stopped
            assert logger._running is False
    
    @pytest.mark.asyncio
    async def test_activity_logger_double_start(self, mock_database_pool, mock_config):
        """Test that double start doesn't cause issues"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            await logger.start()  # Should not raise exception
            
            assert logger._running is True
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_activity_logger_stop_without_start(self, mock_database_pool, mock_config):
        """Test that stop without start doesn't cause issues"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.stop()  # Should not raise exception
            
            assert logger._running is False


class TestActivityLoggerCoreLogging:
    """Test core logging functionality"""
    
    @pytest.mark.asyncio
    async def test_log_activity_basic(self, activity_logger_instance):
        """Test basic activity logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test_source",
            event_type="test_event",
            title="Test activity"
        )
        
        # Should return activity ID
        assert activity_id is not None
        assert isinstance(activity_id, str)
        
        # Should add to buffer
        assert len(logger._buffer) == 1
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.SYSTEM
        assert entry.action == ActivityAction.START
        assert entry.source == "test_source"
        assert entry.title == "Test activity"
    
    @pytest.mark.asyncio
    async def test_log_activity_with_context(self, activity_logger_instance):
        """Test activity logging with full context"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.TRADING,
            action=ActivityAction.EXECUTE,
            source="trading_engine",
            event_type="trade_execution",
            title="Trade executed",
            severity=ActivitySeverity.INFO,
            user_id=123456789,
            session_id=str(uuid.uuid4()),
            trading_mode=TradingMode.SIMULATION,
            token_address="0x123",
            chain=ChainType.ETHEREUM,
            amount_usd=Decimal("100.50"),
            execution_time_ms=1500,
            metadata={"test": "data"}
        )
        
        # Should create entry with all context
        assert len(logger._buffer) == 1
        entry = logger._buffer[0]
        assert entry.user_id == 123456789
        assert entry.trading_mode == TradingMode.SIMULATION
        assert entry.token_address == "0x123"
        assert entry.amount_usd == Decimal("100.50")
        assert entry.execution_time_ms == 1500
        assert entry.metadata == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_log_activity_checksum_generation(self, activity_logger_instance):
        """Test that checksum is automatically generated"""
        logger = activity_logger_instance
        
        await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test",
            event_type="test_event",
            title="Test"
        )
        
        entry = logger._buffer[0]
        assert entry.checksum is not None
        assert len(entry.checksum) == 64  # SHA256 hex
    
    @pytest.mark.asyncio
    async def test_critical_activity_immediate_flush(self, activity_logger_instance):
        """Test that critical activities trigger immediate flush"""
        logger = activity_logger_instance
        
        with patch.object(logger, '_flush_buffer', new_callable=AsyncMock) as mock_flush:
            await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.ERROR,
                source="test",
                event_type="critical_error",
                title="Critical error",
                severity=ActivitySeverity.CRITICAL
            )
            
            # Should trigger immediate flush for critical severity
            mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_error_with_exception(self, activity_logger_instance):
        """Test error logging with exception handling"""
        logger = activity_logger_instance
        
        try:
            raise ValueError("Test error message")
        except Exception as e:
            activity_id = await logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="test_source",
                event_type="test_error",
                title="Test error occurred",
                error_code="TEST_ERROR",
                exception=e
            )
        
        # Should create error entry with stack trace
        assert len(logger._buffer) == 1
        entry = logger._buffer[0]
        assert entry.action == ActivityAction.ERROR
        assert entry.severity == ActivitySeverity.ERROR
        assert entry.error_code == "TEST_ERROR"
        assert entry.error_message == "Test error message"
        assert "ValueError: Test error message" in entry.stack_trace
    
    @pytest.mark.asyncio
    async def test_log_error_without_exception(self, activity_logger_instance):
        """Test error logging without exception"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_error(
            category=ActivityCategory.API,
            source="api_client",
            event_type="api_error",
            title="API call failed",
            error_code="API_TIMEOUT",
            error_message="Request timed out after 30 seconds"
        )
        
        # Should create error entry
        entry = logger._buffer[0]
        assert entry.error_code == "API_TIMEOUT"
        assert entry.error_message == "Request timed out after 30 seconds"
        assert entry.stack_trace is not None  # Should capture current stack
    
    @pytest.mark.asyncio
    async def test_log_performance(self, activity_logger_instance):
        """Test performance logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_performance(
            source="test_component",
            operation="test_operation",
            execution_time_ms=1250,
            success=True,
            metadata={"test": "data"}
        )
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.PERFORMANCE
        assert entry.action == ActivityAction.SUCCESS
        assert entry.source == "test_component"
        assert entry.event_type == "performance_test_operation"
        assert entry.execution_time_ms == 1250
        assert entry.severity == ActivitySeverity.INFO
    
    @pytest.mark.asyncio
    async def test_log_performance_failure(self, activity_logger_instance):
        """Test performance logging for failed operations"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_performance(
            source="test_component",
            operation="test_operation",
            execution_time_ms=30000,
            success=False
        )
        
        entry = logger._buffer[0]
        assert entry.action == ActivityAction.FAILURE
        assert entry.severity == ActivitySeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_log_user_action(self, activity_logger_instance):
        """Test user action logging"""
        logger = activity_logger_instance
        
        session_id = str(uuid.uuid4())
        activity_id = await logger.log_user_action(
            user_id=123456789,
            action=ActivityAction.ACCESS,
            component="trading",
            title="Accessed trading page",
            session_id=session_id,
            ip_address="192.168.1.100"
        )
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.USER
        assert entry.action == ActivityAction.ACCESS
        assert entry.source == "dashboard"
        assert entry.user_id == 123456789
        assert entry.session_id == session_id
        assert entry.dashboard_component == "trading"
        assert entry.ip_address == "192.168.1.100"
    
    @pytest.mark.asyncio
    async def test_log_trading_activity(self, activity_logger_instance):
        """Test trading activity logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_trading_activity(
            action=ActivityAction.EXECUTE,
            title="Buy order executed",
            trading_mode=TradingMode.LIVE,
            token_address="0x123",
            chain=ChainType.ETHEREUM,
            amount_usd=Decimal("500.25"),
            execution_time_ms=2000
        )
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.TRADING
        assert entry.action == ActivityAction.EXECUTE
        assert entry.source == "trading_engine"
        assert entry.trading_mode == TradingMode.LIVE
        assert entry.token_address == "0x123"
        assert entry.chain == ChainType.ETHEREUM
        assert entry.amount_usd == Decimal("500.25")
    
    @pytest.mark.asyncio
    async def test_log_api_call(self, activity_logger_instance):
        """Test API call logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_api_call(
            api_name="jupiter_client",
            endpoint="/quote",
            method="GET",
            status_code=200,
            response_time_ms=450,
            success=True,
            user_agent="Test Agent"
        )
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.API
        assert entry.action == ActivityAction.SUCCESS
        assert entry.source == "jupiter_client"
        assert entry.event_type == "api_call"
        assert entry.api_endpoint == "/quote"
        assert entry.http_method == "GET"
        assert entry.http_status == 200
        assert entry.response_time_ms == 450
        assert entry.user_agent == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_log_api_call_failure(self, activity_logger_instance):
        """Test API call failure logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_api_call(
            api_name="birdeye_client",
            endpoint="/token-info",
            method="POST",
            status_code=500,
            response_time_ms=5000,
            success=False,
            error_code="INTERNAL_ERROR"
        )
        
        entry = logger._buffer[0]
        assert entry.action == ActivityAction.FAILURE
        assert entry.severity == ActivitySeverity.WARNING
        assert entry.http_status == 500
        assert entry.error_code == "INTERNAL_ERROR"


class TestActivityLoggerBatchProcessing:
    """Test batch processing functionality"""
    
    @pytest.mark.asyncio
    async def test_buffer_accumulation(self, activity_logger_instance):
        """Test that activities accumulate in buffer"""
        logger = activity_logger_instance
        
        # Add multiple activities
        for i in range(5):
            await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="test",
                event_type=f"test_event_{i}",
                title=f"Test activity {i}"
            )
        
        # Should accumulate in buffer
        assert len(logger._buffer) == 5
    
    @pytest.mark.asyncio
    async def test_manual_flush_buffer(self, activity_logger_instance):
        """Test manual buffer flushing"""
        logger = activity_logger_instance
        
        # Add activities to buffer
        for i in range(3):
            await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="test",
                event_type=f"test_event_{i}",
                title=f"Test activity {i}"
            )
        
        assert len(logger._buffer) == 3
        
        # Mock the insert method to verify it's called
        with patch.object(logger, '_insert_activities', new_callable=AsyncMock) as mock_insert:
            await logger._flush_buffer()
            
            # Should call insert with buffered entries
            mock_insert.assert_called_once()
            call_args = mock_insert.call_args[0][0]
            assert len(call_args) == 3
            
            # Buffer should be cleared
            assert len(logger._buffer) == 0
    
    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, activity_logger_instance):
        """Test flushing empty buffer"""
        logger = activity_logger_instance
        
        with patch.object(logger, '_insert_activities', new_callable=AsyncMock) as mock_insert:
            await logger._flush_buffer()
            
            # Should not call insert for empty buffer
            mock_insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_flush_buffer_error_handling(self, activity_logger_instance):
        """Test error handling during buffer flush"""
        logger = activity_logger_instance
        
        # Add activity to buffer
        await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test",
            event_type="test_event",
            title="Test activity"
        )
        
        # Mock insert to raise exception
        with patch.object(logger, '_insert_activities', new_callable=AsyncMock) as mock_insert:
            mock_insert.side_effect = Exception("Database error")
            
            await logger._flush_buffer()
            
            # Buffer should be re-added on error (for retry)
            assert len(logger._buffer) == 1
    
    @pytest.mark.asyncio
    async def test_buffer_size_limit(self, activity_logger_instance):
        """Test buffer size limits to prevent memory issues"""
        logger = activity_logger_instance
        
        # Mock insert to always fail
        with patch.object(logger, '_insert_activities', new_callable=AsyncMock) as mock_insert:
            mock_insert.side_effect = Exception("Database error")
            
            # Fill buffer beyond limit (10 batches * 100 = 1000 entries)
            for i in range(1050):
                await logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.EXECUTE,
                    source="test",
                    event_type=f"test_event_{i}",
                    title=f"Test activity {i}"
                )
                
                # Trigger flush periodically to simulate error scenario
                if i % 100 == 0:
                    await logger._flush_buffer()
            
            # Buffer should not exceed limit (max 10 batches)
            assert len(logger._buffer) <= 1000


class TestActivityLoggerDatabaseOperations:
    """Test database operations"""
    
    @pytest.mark.asyncio
    async def test_insert_activities_success(self, mock_database_pool, mock_config):
        """Test successful database insertion"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Create test entries
            entries = [
                ActivityLogEntry(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.START,
                    source="test",
                    event_type="test_event",
                    title="Test entry 1"
                ),
                ActivityLogEntry(
                    category=ActivityCategory.TRADING,
                    action=ActivityAction.EXECUTE,
                    source="trading",
                    event_type="trade",
                    title="Test entry 2"
                )
            ]
            
            # Should execute INSERT statements
            await logger._insert_activities(entries)
            
            # Verify database calls
            conn = mock_database_pool.acquire.return_value.__aenter__.return_value
            assert conn.execute.call_count == 2  # One INSERT per entry
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_insert_activities_no_pool(self, mock_config):
        """Test insert when pool is not initialized"""
        with patch('src.logging.activity_logger.get_config', return_value=mock_config):
            logger = ActivityLogger()
            # Don't start logger (no pool)
            
            entries = [ActivityLogEntry(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="test",
                event_type="test_event",
                title="Test entry"
            )]
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Database pool not initialized"):
                await logger._insert_activities(entries)
    
    @pytest.mark.asyncio
    async def test_insert_activities_field_mapping(self, mock_database_pool, mock_config):
        """Test that all fields are properly mapped for database insertion"""
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Create entry with all possible fields
            entry = ActivityLogEntry(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="trading_engine",
                event_type="trade_execution",
                title="Complete trade entry",
                description="Full featured trade",
                severity=ActivitySeverity.INFO,
                user_id=123456789,
                session_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                trading_mode=TradingMode.SIMULATION,
                token_address="0x123",
                chain=ChainType.ETHEREUM,
                amount_usd=Decimal("100.50"),
                fee_usd=Decimal("2.50"),
                execution_time_ms=1500,
                memory_usage_mb=45,
                cpu_usage_pct=Decimal("12.5"),
                metadata={"test": "data"},
                tags=["test", "trading"],
                correlation_id="test-correlation",
                error_code="TEST_ERROR",
                error_message="Test error",
                stack_trace="Test stack trace",
                api_endpoint="/test",
                http_method="POST",
                http_status=200,
                user_agent="Test Agent",
                ip_address="192.168.1.100",
                response_time_ms=500,
                throughput_ops_per_sec=Decimal("100.5"),
                security_level="high",
                risk_score=25,
                dashboard_component="trading",
                dashboard_action="execute_trade"
            )
            entry.checksum = entry.generate_checksum()
            
            await logger._insert_activities([entry])
            
            # Verify the execute call includes all expected fields
            conn = mock_database_pool.acquire.return_value.__aenter__.return_value
            conn.execute.assert_called()
            
            # Check that the query includes all expected fields
            call_args = conn.execute.call_args
            query = call_args[0][0]
            values = call_args[0][1:]
            
            # Should have INSERT statement with all fields
            assert "INSERT INTO activity_logs" in query
            assert "activity_id" in query
            assert "trading_mode" in query
            assert "metadata" in query
            
            await logger.stop()


class TestGlobalConvenienceFunctions:
    """Test global convenience functions"""
    
    @pytest.mark.asyncio
    async def test_log_system_event(self):
        """Test log_system_event convenience function"""
        from src.activity_logging.activity_logger import log_system_event
        
        with patch.object(activity_logger, 'log_activity', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = "test-activity-id"
            
            activity_id = await log_system_event(
                event_type="system_startup",
                title="System started",
                metadata={"version": "1.0.0"}
            )
            
            # Should call logger with correct parameters
            mock_log.assert_called_once_with(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="system",
                event_type="system_startup",
                title="System started",
                metadata={"version": "1.0.0"}
            )
            
            assert activity_id == "test-activity-id"
    
    @pytest.mark.asyncio
    async def test_log_dashboard_action(self):
        """Test log_dashboard_action convenience function"""
        from src.activity_logging.activity_logger import log_dashboard_action
        
        with patch.object(activity_logger, 'log_user_action', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = "test-activity-id"
            
            activity_id = await log_dashboard_action(
                user_id=123456789,
                component="trading",
                action="view_portfolio"
            )
            
            # Should call logger with correct parameters
            mock_log.assert_called_once_with(
                user_id=123456789,
                action=ActivityAction.ACCESS,
                component="trading",
                title="User accessed trading: view_portfolio",
                dashboard_component="trading",
                dashboard_action="view_portfolio"
            )
    
    @pytest.mark.asyncio
    async def test_log_trade_execution(self):
        """Test log_trade_execution convenience function"""
        from src.activity_logging.activity_logger import log_trade_execution
        
        with patch.object(activity_logger, 'log_trading_activity', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = "test-activity-id"
            
            activity_id = await log_trade_execution(
                trading_mode=TradingMode.LIVE,
                token_address="0x123",
                chain=ChainType.ETHEREUM,
                amount_usd=Decimal("100.50"),
                success=True,
                execution_time_ms=1500
            )
            
            # Should call logger with correct parameters
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]["action"] == ActivityAction.SUCCESS
            assert call_args[1]["trading_mode"] == TradingMode.LIVE
            assert call_args[1]["severity"] == ActivitySeverity.INFO
    
    @pytest.mark.asyncio
    async def test_log_ml_prediction(self):
        """Test log_ml_prediction convenience function"""
        from src.activity_logging.activity_logger import log_ml_prediction
        
        with patch.object(activity_logger, 'log_activity', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = "test-activity-id"
            
            activity_id = await log_ml_prediction(
                model_name="lstm",
                prediction_type="price_movement",
                execution_time_ms=850,
                confidence=0.87
            )
            
            # Should call logger with correct parameters
            mock_log.assert_called_once_with(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.EXECUTE,
                source="ml_model_lstm",
                event_type="prediction_price_movement",
                title="ML prediction: price_movement",
                execution_time_ms=850,
                metadata={'confidence': 0.87}
            )


class TestPerformanceTracker:
    """Test performance tracking context manager"""
    
    @pytest.mark.asyncio
    async def test_performance_tracker_success(self):
        """Test performance tracker for successful operation"""
        from src.activity_logging.activity_logger import performance_tracker
        
        with patch.object(activity_logger, 'log_performance', new_callable=AsyncMock) as mock_log_perf:
            mock_log_perf.return_value = "test-activity-id"
            
            async with performance_tracker(
                source="test_component",
                operation="test_operation",
                metadata={"test": "data"}
            ) as tracker:
                # Simulate some work
                await asyncio.sleep(0.01)
            
            # Should log performance
            mock_log_perf.assert_called_once()
            call_args = mock_log_perf.call_args
            assert call_args[1]["source"] == "test_component"
            assert call_args[1]["operation"] == "test_operation"
            assert call_args[1]["success"] is True
            assert call_args[1]["execution_time_ms"] > 0
            assert call_args[1]["metadata"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_performance_tracker_failure(self):
        """Test performance tracker for failed operation"""
        from src.activity_logging.activity_logger import performance_tracker
        
        with patch.object(activity_logger, 'log_performance', new_callable=AsyncMock) as mock_log_perf, \
             patch.object(activity_logger, 'log_error', new_callable=AsyncMock) as mock_log_error:
            
            mock_log_perf.return_value = "perf-activity-id"
            mock_log_error.return_value = "error-activity-id"
            
            with pytest.raises(ValueError):
                async with performance_tracker(
                    source="test_component",
                    operation="test_operation"
                ) as tracker:
                    raise ValueError("Test error")
            
            # Should log both performance and error
            mock_log_perf.assert_called_once()
            perf_args = mock_log_perf.call_args
            assert perf_args[1]["success"] is False
            
            mock_log_error.assert_called_once()
            error_args = mock_log_error.call_args
            assert error_args[1]["parent_activity_id"] == "perf-activity-id"


# NOTE: All these tests should FAIL initially since the implementation is incomplete
# This follows TDD methodology where tests are written first, then implementation follows