#!/usr/bin/env python3
"""
Simple test runner to verify ActivityLogger implementation
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path - adjust for new location in tests/utils/
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction,
    ActivitySeverity, TradingMode, ChainType
)


async def test_activity_log_entry():
    """Test ActivityLogEntry basic functionality"""
    print("Testing ActivityLogEntry...")
    
    # Test basic creation
    entry = ActivityLogEntry(
        category=ActivityCategory.TRADING,
        action=ActivityAction.EXECUTE,
        source="trading_engine",
        event_type="trade_execution",
        title="Buy order executed"
    )
    
    # Check auto-generated fields
    assert entry.activity_id is not None
    assert entry.created_at is not None
    assert isinstance(entry.activity_id, str)
    assert isinstance(entry.created_at, datetime)
    print("✓ Auto-generation of activity_id and created_at works")
    
    # Test to_dict conversion
    data = entry.to_dict()
    assert "activity_id" in data
    assert "category" in data
    assert data["category"] == "trading"
    assert data["action"] == "execute"
    print("✓ to_dict() conversion works")
    
    # Test checksum generation
    checksum = entry.generate_checksum()
    assert checksum is not None
    assert isinstance(checksum, str)
    assert len(checksum) == 64  # SHA256 hex
    print("✓ Checksum generation works")
    
    print("ActivityLogEntry tests passed!\n")


async def test_activity_logger_basic():
    """Test basic ActivityLogger functionality"""
    print("Testing ActivityLogger...")
    
    # Create proper async context manager mocks
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    # Mock the transaction context manager
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction)
    
    # Mock the pool acquire context manager  
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
    
    # Test logger initialization
    with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        await logger.start()
        
        assert logger._running is True
        assert logger._pool == mock_pool
        print("✓ Logger initialization works")
        
        # Test basic logging
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test_source",
            event_type="test_event",
            title="Test activity"
        )
        
        assert activity_id is not None
        assert isinstance(activity_id, str)
        assert len(logger._buffer) == 1
        print("✓ Basic activity logging works")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_id = await logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="test_source",
                event_type="test_error",
                title="Test error",
                exception=e
            )
            
            assert error_id is not None
            error_entry = logger._buffer[-1]  # Last entry
            assert error_entry.action == ActivityAction.ERROR
            assert error_entry.error_message == "Test error"
            assert "ValueError: Test error" in error_entry.stack_trace
            print("✓ Error logging with exception works")
        
        # Test performance logging
        perf_id = await logger.log_performance(
            source="test_component",
            operation="test_operation",
            execution_time_ms=1500,
            success=True
        )
        
        assert perf_id is not None
        perf_entry = logger._buffer[-1]
        assert perf_entry.category == ActivityCategory.PERFORMANCE
        assert perf_entry.execution_time_ms == 1500
        print("✓ Performance logging works")
        
        # Test manual flush
        await logger._flush_buffer()
        mock_conn.execute.assert_called()
        print("✓ Buffer flushing works")
        
        await logger.stop()
        assert logger._running is False
        print("✓ Logger shutdown works")
    
    print("ActivityLogger tests passed!\n")


async def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")
    
    from src.logging.activity_logger import (
        log_system_event, log_dashboard_action, log_trade_execution, log_ml_prediction
    )
    
    # Create proper async context manager mocks
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    # Mock the transaction context manager
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction)
    
    # Mock the pool acquire context manager  
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
    
    with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.logging.activity_logger.get_config', return_value=mock_config):
        
        from src.logging.activity_logger import activity_logger
        await activity_logger.start()
        
        # Test system event logging
        system_id = await log_system_event(
            event_type="system_startup",
            title="System started"
        )
        assert system_id is not None
        print("✓ log_system_event works")
        
        # Test dashboard action logging
        dashboard_id = await log_dashboard_action(
            user_id=123456,
            component="trading",
            action="view_portfolio"
        )
        assert dashboard_id is not None
        print("✓ log_dashboard_action works")
        
        # Test trade execution logging
        trade_id = await log_trade_execution(
            trading_mode=TradingMode.SIMULATION,
            token_address="0x123",
            chain=ChainType.ETHEREUM,
            amount_usd=Decimal("100.50"),
            success=True
        )
        assert trade_id is not None
        print("✓ log_trade_execution works")
        
        # Test ML prediction logging
        ml_id = await log_ml_prediction(
            model_name="lstm",
            prediction_type="price_movement",
            execution_time_ms=850,
            confidence=0.87
        )
        assert ml_id is not None
        print("✓ log_ml_prediction works")
        
        await activity_logger.stop()
    
    print("Convenience functions tests passed!\n")


async def test_performance_tracker():
    """Test performance tracking context manager"""
    print("Testing performance tracker...")
    
    from src.logging.activity_logger import performance_tracker
    
    # Create proper async context manager mocks
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    # Mock the transaction context manager
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction)
    
    # Mock the pool acquire context manager  
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
    
    with patch('src.logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.logging.activity_logger.get_config', return_value=mock_config):
        
        from src.logging.activity_logger import activity_logger
        await activity_logger.start()
        
        # Test successful operation tracking
        async with performance_tracker(
            source="test_component",
            operation="test_operation"
        ) as tracker:
            await asyncio.sleep(0.01)  # Simulate work
        
        # Should have logged performance
        assert len(activity_logger._buffer) > 0
        perf_entry = activity_logger._buffer[-1]
        assert perf_entry.category == ActivityCategory.PERFORMANCE
        assert perf_entry.execution_time_ms > 0
        print("✓ Performance tracker context manager works")
        
        await activity_logger.stop()
    
    print("Performance tracker tests passed!\n")


async def main():
    """Run all tests"""
    print("Running ActivityLogger implementation tests...\n")
    
    try:
        await test_activity_log_entry()
        await test_activity_logger_basic()
        await test_convenience_functions()
        await test_performance_tracker()
        
        print("🎉 All tests passed! ActivityLogger implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())