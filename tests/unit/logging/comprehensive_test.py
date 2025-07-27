#!/usr/bin/env python3
"""
Comprehensive test to validate that our ActivityLogger implementation 
meets all the requirements from the original test file
"""

import asyncio
import sys
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add paths - adjust for new location in tests/unit/logging/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction,
    ActivitySeverity, TradingMode, ChainType, activity_logger,
    log_system_event, log_dashboard_action, log_trade_execution, log_ml_prediction,
    performance_tracker
)


def create_mock_setup():
    """Create proper mock setup for tests"""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction)
    
    mock_pool = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire = MagicMock(return_value=mock_acquire)
    
    mock_config = MagicMock()
    mock_config.database.host = "localhost"
    mock_config.database.port = 5432
    mock_config.database.database = "test_db"
    mock_config.database.username = "test_user"
    mock_config.database.password = "test_password"
    mock_config.database.pool_size = 5
    
    return mock_pool, mock_conn, mock_config


def sample_activity_entry():
    """Sample activity log entry for testing"""
    return ActivityLogEntry(
        category=ActivityCategory.TRADING,
        action=ActivityAction.EXECUTE,
        source="trading_engine",
        event_type="trade_execution",
        title="Buy order executed",
        description="Successfully executed buy order for PEPE token",
        user_id=123456789,
        session_id=str(uuid.uuid4()),
        trading_mode=TradingMode.SIMULATION,
        token_address="0x6982508145454Ce325dDbE47a25d4ec3d2311933",
        chain=ChainType.ETHEREUM,
        amount_usd=Decimal("100.50"),
        execution_time_ms=1250,
        metadata={"slippage": 0.1, "gas_fee": 2.5}
    )


class TestActivityLogEntry:
    """Test ActivityLogEntry data class functionality"""
    
    def test_activity_log_entry_creation(self):
        """Test basic ActivityLogEntry creation and auto-generation"""
        entry = sample_activity_entry()
        
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
        print("✓ test_activity_log_entry_creation")
        
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
        print("✓ test_activity_log_entry_auto_fields")
        
    def test_activity_log_entry_to_dict(self):
        """Test conversion to dictionary for database insertion"""
        entry = sample_activity_entry()
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
        print("✓ test_activity_log_entry_to_dict")
        
    def test_activity_log_entry_checksum_generation(self):
        """Test checksum generation for data integrity"""
        entry = sample_activity_entry()
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
        print("✓ test_activity_log_entry_checksum_generation")


class TestActivityLoggerInitialization:
    """Test ActivityLogger initialization and lifecycle"""
    
    async def test_activity_logger_initialization(self):
        """Test ActivityLogger initialization"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            
            # Should initialize with default values
            assert logger._batch_size == 100
            assert logger._batch_timeout == 5.0
            assert logger._buffer == []
            assert logger._running is False
            assert logger._pool is None
            assert logger._flush_task is None
            print("✓ test_activity_logger_initialization")
    
    async def test_activity_logger_start(self):
        """Test ActivityLogger start functionality"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Should be marked as running
            assert logger._running is True
            assert logger._pool == mock_pool
            assert logger._flush_task is not None
            
            # Clean up
            await logger.stop()
            print("✓ test_activity_logger_start")
    
    async def test_activity_logger_stop(self):
        """Test ActivityLogger stop functionality"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            await logger.stop()
            
            # Should be marked as stopped
            assert logger._running is False
            print("✓ test_activity_logger_stop")


class TestActivityLoggerCoreLogging:
    """Test core logging functionality"""
    
    async def test_log_activity_basic(self):
        """Test basic activity logging"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_activity_basic")
    
    async def test_log_activity_with_context(self):
        """Test activity logging with full context"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_activity_with_context")
    
    async def test_log_error_with_exception(self):
        """Test error logging with exception handling"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_error_with_exception")
    
    async def test_log_performance(self):
        """Test performance logging"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_performance")
    
    async def test_log_user_action(self):
        """Test user action logging"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_user_action")
    
    async def test_log_api_call(self):
        """Test API call logging"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
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
            
            await logger.stop()
            print("✓ test_log_api_call")


class TestGlobalConvenienceFunctions:
    """Test global convenience functions"""
    
    async def test_log_system_event(self):
        """Test log_system_event convenience function"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            await activity_logger.start()
            
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
            
            await activity_logger.stop()
            print("✓ test_log_system_event")
    
    async def test_log_trade_execution(self):
        """Test log_trade_execution convenience function"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            await activity_logger.start()
            
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
            
            await activity_logger.stop()
            print("✓ test_log_trade_execution")


class TestPerformanceTracker:
    """Test performance tracking context manager"""
    
    async def test_performance_tracker_success(self):
        """Test performance tracker for successful operation"""
        mock_pool, mock_conn, mock_config = create_mock_setup()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            await activity_logger.start()
            
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
            
            await activity_logger.stop()
            print("✓ test_performance_tracker_success")


async def run_all_tests():
    """Run all test classes"""
    print("Running comprehensive ActivityLogger tests...\n")
    
    # Test ActivityLogEntry
    print("Testing ActivityLogEntry...")
    entry_tests = TestActivityLogEntry()
    entry_tests.test_activity_log_entry_creation()
    entry_tests.test_activity_log_entry_auto_fields()
    entry_tests.test_activity_log_entry_to_dict()
    entry_tests.test_activity_log_entry_checksum_generation()
    print("ActivityLogEntry tests completed!\n")
    
    # Test ActivityLogger Initialization
    print("Testing ActivityLogger Initialization...")
    init_tests = TestActivityLoggerInitialization()
    await init_tests.test_activity_logger_initialization()
    await init_tests.test_activity_logger_start()
    await init_tests.test_activity_logger_stop()
    print("ActivityLogger initialization tests completed!\n")
    
    # Test Core Logging
    print("Testing ActivityLogger Core Logging...")
    core_tests = TestActivityLoggerCoreLogging()
    await core_tests.test_log_activity_basic()
    await core_tests.test_log_activity_with_context()
    await core_tests.test_log_error_with_exception()
    await core_tests.test_log_performance()
    await core_tests.test_log_user_action()
    await core_tests.test_log_api_call()
    print("Core logging tests completed!\n")
    
    # Test Convenience Functions
    print("Testing Global Convenience Functions...")
    convenience_tests = TestGlobalConvenienceFunctions()
    await convenience_tests.test_log_system_event()
    await convenience_tests.test_log_trade_execution()
    print("Convenience function tests completed!\n")
    
    # Test Performance Tracker
    print("Testing Performance Tracker...")
    tracker_tests = TestPerformanceTracker()
    await tracker_tests.test_performance_tracker_success()
    print("Performance tracker tests completed!\n")


async def main():
    try:
        await run_all_tests()
        print("🎉 All comprehensive tests passed! ActivityLogger implementation is complete and robust.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())