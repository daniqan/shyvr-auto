#!/usr/bin/env python3
"""
Direct test runner for ActivityLogger without pytest
"""

import sys
import asyncio
import uuid
from decimal import Decimal
from pathlib import Path

# Add paths - adjust for new location in tests/utils/
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test dependencies
from unittest.mock import AsyncMock, MagicMock, patch

# Import ActivityLogger components
from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction,
    ActivitySeverity, TradingMode, ChainType
)

# Sample activity entry helper function
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


def create_mock_setup():
    """Create proper mock setup for tests"""
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
    
    return mock_pool, mock_conn, mock_config


def test_activity_log_entry_creation():
    """Test ActivityLogEntry creation and auto-generation"""
    print("Testing ActivityLogEntry creation...")
    
    entry = sample_activity_entry()
    
    # Should auto-generate required fields
    assert entry.activity_id is not None
    assert entry.created_at is not None
    assert isinstance(entry.activity_id, str)
    
    # Should preserve provided fields
    assert entry.category == ActivityCategory.TRADING
    assert entry.action == ActivityAction.EXECUTE
    assert entry.source == "trading_engine"
    assert entry.title == "Buy order executed"
    
    print("✓ ActivityLogEntry creation test passed")


def test_activity_log_entry_auto_fields():
    """Test auto-generation of activity_id and timestamps"""
    print("Testing auto-generation of fields...")
    
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
    
    print("✓ Auto-generation test passed")


def test_activity_log_entry_to_dict():
    """Test conversion to dictionary for database insertion"""
    print("Testing to_dict conversion...")
    
    entry = sample_activity_entry()
    data = entry.to_dict()
    
    # Should include all non-None fields
    assert "activity_id" in data
    assert "category" in data
    assert "action" in data
    assert "source" in data
    
    # Enums should be converted to values
    assert data["category"] == "trading"
    assert data["action"] == "execute"
    assert data["severity"] == "info"  # default value
    assert data["trading_mode"] == "simulation"
    assert data["chain"] == "ethereum"
    
    # Decimal should be preserved
    assert isinstance(data["amount_usd"], Decimal)
    assert data["amount_usd"] == Decimal("100.50")
    
    print("✓ to_dict conversion test passed")


def test_activity_log_entry_checksum_generation():
    """Test checksum generation for data integrity"""
    print("Testing checksum generation...")
    
    entry = sample_activity_entry()
    checksum = entry.generate_checksum()
    
    # Should generate valid checksum
    assert checksum is not None
    assert isinstance(checksum, str)
    assert len(checksum) == 64  # SHA256 hex
    
    # Same entry should generate same checksum
    checksum2 = entry.generate_checksum()
    assert checksum == checksum2
    
    print("✓ Checksum generation test passed")


async def test_activity_logger_initialization():
    """Test ActivityLogger initialization"""
    print("Testing ActivityLogger initialization...")
    
    mock_pool, mock_conn, mock_config = create_mock_setup()
    
    with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        
        # Should initialize with default values
        assert logger._batch_size == 100
        assert logger._batch_timeout == 5.0
        assert logger._buffer == []
        assert logger._running is False
        assert logger._pool is None
        assert logger._flush_task is None
        
        print("✓ ActivityLogger initialization test passed")


async def test_activity_logger_start():
    """Test ActivityLogger start functionality"""
    print("Testing ActivityLogger start...")
    
    mock_pool, mock_conn, mock_config = create_mock_setup()
    
    with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        await logger.start()
        
        # Should be marked as running
        assert logger._running is True
        assert logger._pool == mock_pool
        assert logger._flush_task is not None
        
        # Clean up
        await logger.stop()
        
        print("✓ ActivityLogger start test passed")


async def test_log_activity_basic():
    """Test basic activity logging"""
    print("Testing basic activity logging...")
    
    mock_pool, mock_conn, mock_config = create_mock_setup()
    
    with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
        
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
        
        print("✓ Basic activity logging test passed")


async def test_critical_activity_immediate_flush():
    """Test that critical activities trigger immediate flush"""
    print("Testing critical activity immediate flush...")
    
    mock_pool, mock_conn, mock_config = create_mock_setup()
    
    with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        await logger.start()
        
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
        
        await logger.stop()
        
        print("✓ Critical activity immediate flush test passed")


async def main():
    """Run all tests"""
    print("Running ActivityLogger core tests directly...\n")
    
    try:
        # Test ActivityLogEntry
        test_activity_log_entry_creation()
        test_activity_log_entry_auto_fields()
        test_activity_log_entry_to_dict()
        test_activity_log_entry_checksum_generation()
        
        # Test ActivityLogger
        await test_activity_logger_initialization()
        await test_activity_logger_start()
        await test_log_activity_basic()
        await test_critical_activity_immediate_flush()
        
        print("\n🎉 All core ActivityLogger tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())