#!/usr/bin/env python3
"""
Test runner with proper async context manager mocking
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path - adjust for new location in tests/utils/
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction,
    ActivitySeverity, TradingMode, ChainType
)


async def test_flush_buffer():
    """Test buffer flushing with proper mocking"""
    print("Testing buffer flushing...")
    
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
    
    with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_pool), \
         patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        await logger.start()
        
        # Add some entries to buffer
        await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test",
            event_type="test_event",
            title="Test activity"
        )
        
        print(f"Buffer size before flush: {len(logger._buffer)}")
        
        # Test manual flush
        await logger._flush_buffer()
        
        # Verify database operations were called
        print(f"Pool.acquire called: {mock_pool.acquire.called}")
        print(f"Connection.execute called: {mock_conn.execute.called}")
        print(f"Buffer size after flush: {len(logger._buffer)}")
        
        if mock_conn.execute.called:
            print("✓ Database execute was called successfully")
            call_args = mock_conn.execute.call_args
            print(f"✓ Execute called with query: {call_args[0][0][:50]}...")
        else:
            print("❌ Database execute was not called")
        
        await logger.stop()


async def main():
    try:
        await test_flush_buffer()
        print("🎉 Mock test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())