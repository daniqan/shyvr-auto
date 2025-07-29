import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg

from src.activity_logging.activity_logger import (
    ActivityCategory, ActivityAction, ActivitySeverity, 
    TradingMode, ChainType, ActivityLogEntry, ActivityLogger
)


@pytest.fixture
def sample_activity_entry() -> ActivityLogEntry:
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


@pytest.fixture
def mock_database_pool():
    """Mock asyncpg database pool"""
    pool = AsyncMock(spec=asyncpg.Pool)
    
    # Mock connection
    conn = AsyncMock()
    conn.execute.return_value = None
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = None
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock()
    
    # Mock pool acquire
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock()
    
    return pool


@pytest.fixture
def mock_config():
    """Mock configuration for activity logger"""
    config = MagicMock()
    config.database.host = "localhost"
    config.database.port = 5432
    config.database.database = "test_rlte"
    config.database.username = "test_user"
    config.database.password = "test_password"
    config.database.pool_size = 5
    return config


@pytest.fixture
async def activity_logger_instance(mock_database_pool, mock_config):
    """Activity logger instance with mocked dependencies"""
    with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
         patch('src.logging.activity_logger.get_config', return_value=mock_config):
        
        logger = ActivityLogger()
        await logger.start()
        yield logger
        await logger.stop()