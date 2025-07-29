"""
Pytest fixtures for database migration tests
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg


@pytest.fixture
def mock_database_pool():
    """Mock asyncpg database pool for migration tests"""
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
def sample_rl_experience_data():
    """Sample RL experience data for testing"""
    return {
        'experience_id': str(uuid.uuid4()),
        'session_id': str(uuid.uuid4()),
        'user_id': 123456789,
        'state_data': {
            "price": 0.00123,
            "volume_24h": 150000,
            "market_cap": 1230000,
            "rsi": 65.0,
            "ema_12": 0.00120,
            "position_size": 0.1,
            "portfolio_value": 1000.0,
            "available_balance": 500.0
        },
        'action': 1,  # Buy action
        'reward': Decimal('0.15'),
        'next_state_data': {
            "price": 0.00125,
            "volume_24h": 160000,
            "market_cap": 1250000,
            "rsi": 70.0,
            "ema_12": 0.00122,
            "position_size": 0.2,
            "portfolio_value": 1015.0,
            "available_balance": 485.0
        },
        'done': False,
        'priority': Decimal('0.8'),
        'trading_mode': 'simulation',
        'token_address': '0x6982508145454Ce325dDbE47a25d4ec3d2311933',
        'chain': 'ethereum',
        'market_conditions': {
            "volatility": 0.15,
            "trend": "bullish",
            "support_level": 0.00118,
            "resistance_level": 0.00135,
            "volume_trend": "increasing"
        },
        'performance_metrics': {
            "execution_latency_ms": 150,
            "reward_computation_ms": 25,
            "state_encoding_ms": 10,
            "memory_usage_mb": 64
        },
        'metadata': {
            'strategy': 'dqn',
            'episode': 100,
            'step': 250
        }
    }


@pytest.fixture
def sample_training_session_data():
    """Sample training session data for testing"""
    return {
        'session_id': str(uuid.uuid4()),
        'user_id': 123456789,
        'session_name': 'DQN Training Session 1',
        'trading_mode': 'simulation',
        'agent_config': {
            "algorithm": "DQN",
            "learning_rate": 0.001,
            "epsilon": 0.1,
            "batch_size": 32,
            "memory_size": 10000,
            "target_update_frequency": 100
        },
        'environment_config': {
            "trading_mode": "simulation",
            "initial_balance": 1000.0,
            "max_position_size": 0.5,
            "transaction_cost": 0.001,
            "market_data_source": "binance",
            "lookback_period": 100
        },
        'total_experiences': 1000,
        'successful_experiences': 850,
        'failed_experiences': 150,
        'total_reward': Decimal('125.75'),
        'average_reward': Decimal('0.126'),
        'session_status': 'completed',
        'started_at': datetime.now(timezone.utc) - timedelta(hours=2),
        'ended_at': datetime.now(timezone.utc) - timedelta(hours=1),
        'duration_seconds': 3600,
        'metadata': {
            'notes': 'First successful training run',
            'model_checkpoints': ['checkpoint_100.pth', 'checkpoint_500.pth', 'final.pth']
        }
    }


@pytest.fixture
def sample_performance_metric_data():
    """Sample performance metric data for testing"""
    return {
        'metric_id': str(uuid.uuid4()),
        'session_id': str(uuid.uuid4()),
        'user_id': 123456789,
        'metric_type': 'reward',
        'metric_name': 'average_episode_reward',
        'metric_value': Decimal('0.125'),
        'metric_unit': 'reward_units',
        'trading_mode': 'simulation',
        'time_period_start': datetime.now(timezone.utc) - timedelta(hours=2),
        'time_period_end': datetime.now(timezone.utc) - timedelta(hours=1),
        'aggregation_level': 'session',
        'additional_data': {
            "baseline_comparison": 0.85,
            "confidence_interval": [0.75, 0.95],
            "statistical_significance": 0.001,
            "sample_size": 1000
        }
    }


@pytest.fixture
def batch_rl_experiences():
    """Batch of RL experiences for testing bulk operations"""
    experiences = []
    session_id = str(uuid.uuid4())
    user_id = 123456789
    
    for i in range(10):
        experiences.append({
            'experience_id': str(uuid.uuid4()),
            'session_id': session_id,
            'user_id': user_id,
            'state_data': {
                "price": 0.001 + (i * 0.0001),
                "volume_24h": 150000 + (i * 10000),
                "rsi": 50.0 + (i * 2),
                "position_size": i * 0.01
            },
            'action': i % 3,  # Rotate between actions 0, 1, 2
            'reward': Decimal(str(0.1 + (i * 0.01))),
            'done': i == 9,  # Last experience is terminal
            'priority': Decimal(str(0.5 + (i * 0.05))),
            'trading_mode': 'simulation',
            'token_address': '0x6982508145454Ce325dDbE47a25d4ec3d2311933',
            'chain': 'ethereum'
        })
    
    return experiences


@pytest.fixture
def mock_migration_runner():
    """Mock migration runner for testing schema deployment"""
    runner = MagicMock()
    runner.run_migration.return_value = True
    runner.rollback_migration.return_value = True
    runner.get_applied_migrations.return_value = ['001', '002', '003']
    return runner