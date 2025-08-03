"""
Integration test configuration and fixtures for comprehensive Phase 7.1 testing.
This module provides fixtures and configuration specifically for integration tests
that require real system components and GCP-like environments.
"""
import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any, Generator
from decimal import Decimal

# Test-specific configuration that won't interfere with production
TEST_CONFIG = {
    'environment': 'test',
    'security': {
        'secret_key': 'test_secret_key_12345',  # Valid test key
        'jwt_algorithm': 'HS256',
        'session_timeout': 3600
    },
    'database': {
        'type': 'sqlite',
        'url': 'sqlite:///:memory:',  # In-memory database for tests
        'pool_size': 5,
        'max_overflow': 10
    },
    'trading': {
        'max_position_size': Decimal('0.1'),
        'max_daily_loss': Decimal('0.05'),
        'risk_limits': {
            'max_leverage': Decimal('2.0'),
            'position_size_limit': Decimal('0.1'),
            'daily_loss_limit': Decimal('0.05')
        }
    },
    'ml_analysis': {
        'model_cache_size': 10,
        'inference_timeout': 5.0,
        'batch_size': 32
    },
    'monitoring': {
        'metrics_interval': 1,
        'alert_thresholds': {
            'error_rate': 0.1,
            'latency_ms': 1000
        }
    },
    'model_preservation': {
        'storage_type': 'memory',
        'cache_size': 100,
        'retention_days': 30
    },
    'safety': {
        'emergency_stop_enabled': True,
        'circuit_breaker_enabled': True,
        'max_drawdown': Decimal('0.15')
    }
}

@pytest.fixture(scope="session")
def integration_test_config():
    """Provide test configuration for integration tests."""
    return TEST_CONFIG.copy()

@pytest.fixture(scope="session") 
def mock_environment_variables():
    """Mock environment variables required for integration tests."""
    test_env = {
        'ENVIRONMENT': 'test',
        'SECRET_KEY': 'test_secret_key_12345',
        'DATABASE_URL': 'sqlite:///:memory:',
        'DATABASE_PASSWORD': 'test_password',
        'JWT_ALGORITHM': 'HS256',
        'TELEGRAM_TOKEN': 'test_token_12345',
        'TELEGRAM_WEBHOOK_SECRET': 'test_webhook_secret',
        'OPENAI_API_KEY': 'test_api_key',
        'BIRDEYE_API_KEY': 'test_birdeye_key',
        'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/test_credentials.json',
        'GCP_PROJECT_ID': 'test-project',
        'CLOUD_SQL_INSTANCE': 'test-instance',
        'GCS_BUCKET': 'test-bucket'
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        yield test_env

@pytest.fixture(scope="function")
def temp_directory():
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="integration_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def test_database():
    """Create test database instance."""
    # This will be implemented when we create the database integration tests
    pass

@pytest.fixture(scope="function")
async def async_test_client():
    """Create async test client for API testing."""
    # This will be implemented when we create the API integration tests
    pass

@pytest.fixture(scope="function")
def mock_trading_environment():
    """Mock trading environment for safe testing."""
    # This will be implemented when we create the trading integration tests
    pass

@pytest.fixture(scope="function")
def mock_ml_models():
    """Mock ML models for testing model integration."""
    # This will be implemented when we create the ML integration tests
    pass

@pytest.fixture(scope="function")
def mock_safety_systems():
    """Mock safety systems for testing safety integration."""
    # This will be implemented when we create the safety integration tests
    pass

class IntegrationTestHelper:
    """Helper class for integration test utilities."""
    
    @staticmethod
    def create_test_market_data():
        """Create realistic test market data."""
        return {
            'symbol': 'ETH/USD',
            'price': Decimal('2000.50'),
            'volume': Decimal('1000.0'),
            'timestamp': '2025-08-03T12:00:00Z',
            'bid': Decimal('2000.25'),
            'ask': Decimal('2000.75')
        }
    
    @staticmethod
    def create_test_trading_signal():
        """Create test trading signal."""
        return {
            'action': 'buy',
            'symbol': 'ETH/USD',
            'quantity': Decimal('0.1'),
            'confidence': 0.85,
            'reasoning': 'ML model prediction',
            'timestamp': '2025-08-03T12:00:00Z'
        }
    
    @staticmethod
    def create_test_portfolio():
        """Create test portfolio state."""
        return {
            'total_value': Decimal('10000.00'),
            'available_balance': Decimal('5000.00'),
            'positions': {
                'ETH/USD': {
                    'quantity': Decimal('2.5'),
                    'average_price': Decimal('2000.00'),
                    'current_value': Decimal('5000.00')
                }
            }
        }

@pytest.fixture(scope="function")
def integration_test_helper():
    """Provide integration test helper utilities."""
    return IntegrationTestHelper()

# Event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_tracker():
    """Track performance metrics during tests."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            
        def start_timing(self, operation: str):
            import time
            self.metrics[operation] = {'start': time.time()}
            
        def end_timing(self, operation: str):
            import time
            if operation in self.metrics:
                self.metrics[operation]['end'] = time.time()
                self.metrics[operation]['duration'] = (
                    self.metrics[operation]['end'] - 
                    self.metrics[operation]['start']
                )
                
        def get_duration(self, operation: str) -> float:
            return self.metrics.get(operation, {}).get('duration', 0.0)
            
        def assert_performance(self, operation: str, max_duration: float):
            duration = self.get_duration(operation)
            assert duration <= max_duration, (
                f"Operation '{operation}' took {duration:.3f}s, "
                f"exceeding limit of {max_duration:.3f}s"
            )
    
    return PerformanceTracker()

# Load testing fixtures
@pytest.fixture(scope="function")
def load_test_config():
    """Configuration for load testing."""
    return {
        'concurrent_users': 10,
        'requests_per_second': 100,
        'test_duration_seconds': 30,
        'ramp_up_seconds': 5
    }