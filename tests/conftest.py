"""
Pytest configuration and shared fixtures for RLTE tests
"""

import asyncio
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.utils.base import AgentRule, Chain, TokenInfo
from src.utils.config import ConfigManager, RLTEConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config_file() -> Generator[Path, None, None]:
    """Create a temporary configuration file for testing"""
    config_content = """
app:
  name: "test-rlte"
  version: "0.1.0"
  environment: "test"
  log_level: "DEBUG"

database:
  host: "localhost"
  port: 5432
  database: "test_rlte"
  username: "test_user"
  password: "test_password"
  pool_size: 5
  max_overflow: 10

redis:
  host: "localhost"
  port: 6379
  db: 1

telegram:
  token: "test_telegram_token"
  webhook_secret: "test_webhook_secret"
  admin_users: [123456789]

agent:
  model_type: "mock"
  model_name: "test-model"
  max_tokens: 100
  temperature: 0.0
  max_active_rules: 5

trading:
  modes:
    analysis: true
    simulation: true
    live: false
  risk_management:
    max_position_size_pct: 0.5
    max_daily_loss_pct: 2.0
    max_drawdown_pct: 5.0

ml:
  training:
    batch_size: 16
    learning_rate: 0.01
    epochs: 10
    validation_split: 0.2

rl:
  algorithm: "DQN"
  training_episodes: 100
  epsilon_start: 1.0
  epsilon_end: 0.1
  batch_size: 32

apis:
  test_api:
    api_key: "test_key"
    base_url: "https://api.test.com"
    rate_limit: 10
    timeout: 5
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_config(temp_config_file: Path) -> RLTEConfig:
    """Load test configuration"""
    config_manager = ConfigManager(temp_config_file)
    return config_manager.load()


@pytest.fixture
def mock_database():
    """Mock database connection"""
    mock_db = AsyncMock()
    mock_db.execute.return_value = None
    mock_db.fetch.return_value = []
    mock_db.fetchrow.return_value = None
    mock_db.fetchval.return_value = None
    return mock_db


@pytest.fixture
def mock_redis():
    """Mock Redis connection"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    return mock_redis


@pytest.fixture
def sample_token_info() -> TokenInfo:
    """Sample token information for testing"""
    return TokenInfo(
        address="0x6982508145454Ce325dDbE47a25d4ec3d2311933",
        symbol="PEPE",
        name="Pepe",
        chain=Chain.ETHEREUM,
        decimals=18,
    )


@pytest.fixture
def sample_agent_rule() -> AgentRule:
    """Sample agent rule for testing"""
    return AgentRule(
        id="test-rule-1",
        prompt_text="If price drops below $0.50, buy 0.1 ETH",
        rule_type="dca",
        parsed_conditions={
            "condition": "price_drop",
            "threshold": 0.50,
            "action": "buy",
            "amount": 0.1,
            "asset": "ETH",
        },
        active=True,
        priority=1,
    )


@pytest.fixture
def mock_api_responses() -> dict[str, Any]:
    """Mock API responses for external services"""
    return {
        "helius_token_metadata": {
            "symbol": "TEST",
            "name": "Test Token",
            "decimals": 9,
            "supply": 1000000000,
        },
        "etherscan_token_info": {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "contractAddress": "0x123",
                    "tokenName": "Test Token",
                    "tokenSymbol": "TEST",
                    "tokenDecimal": "18",
                }
            ],
        },
        "birdeye_token_overview": {
            "data": {
                "address": "0x123",
                "symbol": "TEST",
                "name": "Test Token",
                "liquidity": 50000,
                "holder": 150,
                "v24hChangePercent": 5.5,
                "priceUsd": 0.00123,
            }
        },
        "x_api_tweets": {
            "data": [
                {
                    "id": "1234567890",
                    "text": "Just bought some TEST token! Going to the moon! 🚀",
                    "created_at": "2025-01-01T12:00:00.000Z",
                    "public_metrics": {"retweet_count": 5, "like_count": 25, "reply_count": 3},
                }
            ],
            "meta": {"result_count": 1},
        },
    }


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.75]  # 75% pump probability
    mock_model.fit.return_value = None
    mock_model.score.return_value = 0.85  # 85% accuracy
    return mock_model


@pytest.fixture
def mock_rl_agent():
    """Mock RL agent for testing"""
    mock_agent = AsyncMock()
    mock_agent.act.return_value = 1  # Buy action
    mock_agent.update.return_value = None
    mock_agent.save.return_value = None
    mock_agent.load.return_value = None
    return mock_agent


@pytest.fixture
def sample_token_features() -> dict[str, Any]:
    """Sample token features for ML testing"""
    return {
        "price_usd": 0.00123,
        "market_cap": 1230000,
        "volume_24h": 150000,
        "holder_count": 150,
        "liquidity_usd": 50000,
        "price_change_1h": 2.5,
        "price_change_24h": 5.5,
        "volume_change_24h": 15.0,
        "rsi_14": 65.0,
        "ema_12": 0.00120,
        "ema_26": 0.00118,
        "whale_transactions": 3,
        "social_mentions": 25,
        "sentiment_score": 0.65,
    }


@pytest.fixture
def sample_trade_outcome() -> dict[str, Any]:
    """Sample trade outcome for RL testing"""
    return {
        "token_address": "0x123",
        "chain": "ethereum",
        "action": 1,  # Buy
        "entry_price": 0.00123,
        "exit_price": 0.00145,
        "profit_pct": 17.9,
        "profit_usd": 25.50,
        "execution_latency_ms": 1500,
        "gas_fee_usd": 2.50,
        "slippage_pct": 0.1,
    }


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for testing"""
    mock_bot = AsyncMock()
    mock_bot.send_message.return_value = None
    mock_bot.edit_message_text.return_value = None
    mock_bot.delete_message.return_value = None
    mock_bot.get_me.return_value = MagicMock(id=123456, username="test_bot")
    return mock_bot


@pytest.fixture
def mock_wallet():
    """Mock cryptocurrency wallet for testing"""
    mock_wallet = AsyncMock()
    mock_wallet.get_balance.return_value = 1000.0  # $1000 USD
    mock_wallet.execute_trade.return_value = {
        "tx_hash": "0xabc123",
        "status": "success",
        "gas_used": 21000,
        "gas_price": 20,
    }
    mock_wallet.sign_transaction.return_value = "0xsignature"
    return mock_wallet


@pytest.fixture
def clean_environment():
    """Clean environment variables for testing"""
    # Store original env vars
    original_env = os.environ.copy()

    # Clear relevant env vars
    test_env_vars = ["DB_HOST", "DB_PASSWORD", "TELEGRAM_TOKEN", "HELIUS_API_KEY", "X_BEARER_TOKEN"]

    for var in test_env_vars:
        os.environ.pop(var, None)

    yield

    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


# Async test utilities
@pytest_asyncio.fixture
async def async_mock_response():
    """Async mock HTTP response"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text.return_value = "OK"
    return mock_response


# Performance test fixtures
@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing"""
    return {
        "response_time_ms": 2000,  # 2 seconds max
        "memory_usage_mb": 500,  # 500MB max
        "cpu_usage_pct": 80,  # 80% max
        "database_query_ms": 100,  # 100ms max per query
        "ml_prediction_ms": 1000,  # 1 second max for ML predictions
        "api_call_ms": 5000,  # 5 seconds max for API calls
    }


# Test data generation utilities
class TestDataGenerator:
    """Generate test data for various scenarios"""

    @staticmethod
    def generate_market_data(volatility: float = 0.1) -> dict[str, Any]:
        """Generate realistic market data"""
        import random

        base_price = random.uniform(0.0001, 1.0)
        return {
            "price_usd": base_price,
            "market_cap": base_price * random.randint(1000000, 100000000),
            "volume_24h": random.randint(10000, 1000000),
            "price_change_24h": random.uniform(-volatility, volatility) * 100,
            "volume_change_24h": random.uniform(-50, 200),
            "holder_count": random.randint(50, 5000),
            "liquidity_usd": random.randint(5000, 500000),
        }

    @staticmethod
    def generate_social_data() -> dict[str, Any]:
        """Generate social media data"""
        import random

        return {
            "x_mentions": random.randint(0, 100),
            "telegram_members": random.randint(100, 10000),
            "sentiment_score": random.uniform(-1.0, 1.0),
            "influence_score": random.uniform(0.0, 1.0),
        }


@pytest.fixture
def test_data_generator():
    """Test data generator instance"""
    return TestDataGenerator()


# Mark tests that require network access
slow = pytest.mark.slow
network = pytest.mark.network
integration = pytest.mark.integration
unit = pytest.mark.unit
performance = pytest.mark.performance
