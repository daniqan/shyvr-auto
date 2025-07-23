"""
Tests for RL agent base classes and data structures
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from typing import Tuple

from src.rl_agent.base import (
    TradeAction, ModelType, MarketState, TradingResult, RewardMetrics, 
    AgentConfig, RLAgentBase, RLTrainingError, RLPredictionError
)
from src.discovery.base import DiscoveredToken
from src.ml_analysis.base import PredictionResult, PredictionDirection


class TestTradeAction:
    """Test TradeAction enum"""
    
    def test_trade_action_values(self):
        """Test that all trade actions have expected values"""
        assert TradeAction.HOLD.value == "hold"
        assert TradeAction.BUY.value == "buy" 
        assert TradeAction.SELL.value == "sell"
        assert TradeAction.STRONG_BUY.value == "strong_buy"
        assert TradeAction.STRONG_SELL.value == "strong_sell"
    
    def test_trade_action_count(self):
        """Test that we have expected number of actions"""
        assert len(TradeAction) == 5


class TestModelType:
    """Test ModelType enum"""
    
    def test_model_type_values(self):
        """Test that all model types have expected values"""
        assert ModelType.DQN.value == "dqn"
        assert ModelType.DDQN.value == "double_dqn"
        assert ModelType.DUELING_DQN.value == "dueling_dqn"
        assert ModelType.RAINBOW.value == "rainbow"


class TestMarketState:
    """Test MarketState data structure"""
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        from src.utils.base import Chain
        return DiscoveredToken(
            address="0x123...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000,
            market_cap=1500000
        )
    
    @pytest.fixture
    def sample_ml_prediction(self, sample_token):
        """Create sample ML prediction"""
        from src.ml_analysis.base import ModelType
        return PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.BUY,
            confidence=0.75,
            price_prediction_1h=1.55,
            price_prediction_4h=1.60,
            price_prediction_24h=1.65
        )
    
    def test_market_state_creation(self, sample_token):
        """Test basic market state creation"""
        state = MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000,
            market_cap=1500000
        )
        
        assert state.token == sample_token
        assert state.price_usd == 1.50
        assert state.price_change_24h == 5.2
        assert state.volume_24h == 100000
        assert state.market_cap == 1500000
        assert state.current_position == 0.0
        assert state.portfolio_value == 10000.0
    
    def test_market_state_to_vector(self, sample_token):
        """Test conversion to feature vector"""
        state = MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000,
            market_cap=1500000,
            rsi=65.0,
            macd=0.05,
            current_position=0.5,
            market_volatility=0.3
        )
        
        vector = state.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == MarketState.get_feature_size()
        assert all(np.isfinite(vector))  # No NaN or infinite values
    
    def test_market_state_to_vector_with_ml_prediction(self, sample_token, sample_ml_prediction):
        """Test feature vector with ML prediction"""
        state = MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000,
            ml_prediction=sample_ml_prediction,
            prediction_confidence=0.75
        )
        
        vector = state.to_vector()
        
        assert len(vector) == MarketState.get_feature_size()
        assert vector[-2] == 0.75  # prediction_confidence
        assert vector[-1] == 0.5   # ML direction encoding for BUY
    
    def test_bollinger_position_calculation(self, sample_token):
        """Test Bollinger Bands position calculation"""
        state = MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000,
            bollinger_upper=1.60,
            bollinger_lower=1.40
        )
        
        # Price at 1.50, bands at 1.40-1.60, should be 0.5 (middle)
        position = state._bollinger_position()
        assert abs(position - 0.5) < 0.01
        
        # Test edge cases
        state.bollinger_upper = None
        assert state._bollinger_position() == 0.5
        
        state.bollinger_upper = 1.40
        state.bollinger_lower = 1.60  # Invalid bands
        assert state._bollinger_position() == 0.5
    
    def test_ml_direction_encoding(self, sample_token):
        """Test ML direction encoding"""
        # Test with no prediction
        state = MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=5.2,
            volume_24h=100000
        )
        assert state._ml_direction_encoding() == 0.0
        
        # Test with different predictions
        directions_expected = [
            (PredictionDirection.STRONG_SELL, -1.0),
            (PredictionDirection.SELL, -0.5),
            (PredictionDirection.HOLD, 0.0),
            (PredictionDirection.BUY, 0.5),
            (PredictionDirection.STRONG_BUY, 1.0)
        ]
        
        for direction, expected in directions_expected:
            from src.ml_analysis.base import ModelType
            prediction = PredictionResult(
                token=sample_token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                direction=direction, 
                confidence=0.8
            )
            state.ml_prediction = prediction
            assert state._ml_direction_encoding() == expected
    
    def test_feature_size_consistency(self):
        """Test that feature size matches actual vector length"""
        # This ensures we update get_feature_size() when adding features
        assert MarketState.get_feature_size() == 19


class TestTradingResult:
    """Test TradingResult data structure"""
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        from src.utils.base import Chain
        return DiscoveredToken(
            address="0x123...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            volume_24h=100000
        )
    
    def test_trading_result_creation(self, sample_token):
        """Test basic trading result creation"""
        result = TradingResult(
            action=TradeAction.BUY,
            token=sample_token,
            executed_at=datetime.now(),
            price=1.50,
            quantity=100,
            value_usd=150.0
        )
        
        assert result.action == TradeAction.BUY
        assert result.token == sample_token
        assert result.price == 1.50
        assert result.quantity == 100
        assert result.value_usd == 150.0
        assert result.success is True
        assert result.slippage == 0.0
        assert result.fees == 0.0
    
    def test_trading_result_to_dict(self, sample_token):
        """Test conversion to dictionary"""
        timestamp = datetime.now()
        result = TradingResult(
            action=TradeAction.SELL,
            token=sample_token,
            executed_at=timestamp,
            price=1.60,
            quantity=50,
            value_usd=80.0,
            slippage=0.02,
            fees=1.0,
            realized_pnl=5.0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['action'] == 'sell'
        assert result_dict['token_address'] == '0x123...'
        assert result_dict['token_symbol'] == 'TEST'
        assert result_dict['executed_at'] == timestamp.isoformat()
        assert result_dict['price'] == 1.60
        assert result_dict['quantity'] == 50
        assert result_dict['value_usd'] == 80.0
        assert result_dict['slippage'] == 0.02
        assert result_dict['fees'] == 1.0
        assert result_dict['realized_pnl'] == 5.0


class TestRewardMetrics:
    """Test RewardMetrics calculations"""
    
    def test_reward_metrics_creation(self):
        """Test basic reward metrics creation"""
        metrics = RewardMetrics(
            absolute_return=100.0,
            relative_return=0.05,
            sharpe_ratio=1.2,
            win_rate=0.65
        )
        
        assert metrics.absolute_return == 100.0
        assert metrics.relative_return == 0.05
        assert metrics.sharpe_ratio == 1.2
        assert metrics.win_rate == 0.65
        assert metrics.total_reward == 0.0  # Not calculated yet
    
    def test_calculate_total_reward_default_config(self):
        """Test total reward calculation with default config"""
        metrics = RewardMetrics(
            risk_adjusted_return=0.08,
            sharpe_ratio=1.5,
            max_drawdown=0.12,
            win_rate=0.62
        )
        
        reward = metrics.calculate_total_reward()
        
        # Expected: 0.08*0.4 + 1.5*0.3 + 0.12*(-0.2) + 0.62*0.1
        expected = 0.08 * 0.4 + 1.5 * 0.3 + 0.12 * (-0.2) + 0.62 * 0.1
        assert abs(reward - expected) < 0.001
        assert metrics.total_reward == reward
    
    def test_calculate_total_reward_custom_config(self):
        """Test total reward calculation with custom config"""
        metrics = RewardMetrics(
            risk_adjusted_return=0.10,
            sharpe_ratio=2.0,
            max_drawdown=0.08,
            win_rate=0.70
        )
        
        custom_config = {
            'return_weight': 0.5,
            'sharpe_weight': 0.3,
            'drawdown_weight': -0.15,
            'win_rate_weight': 0.05
        }
        
        reward = metrics.calculate_total_reward(custom_config)
        
        expected = 0.10 * 0.5 + 2.0 * 0.3 + 0.08 * (-0.15) + 0.70 * 0.05
        assert abs(reward - expected) < 0.001


class TestAgentConfig:
    """Test AgentConfig data structure"""
    
    def test_agent_config_defaults(self):
        """Test default configuration values"""
        config = AgentConfig()
        
        assert config.model_type == ModelType.DQN
        assert config.hidden_size == 256
        assert config.num_layers == 3
        assert config.dropout == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.replay_buffer_size == 10000
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.05
        assert config.max_position_size == 0.1
        assert config.target_sharpe_ratio == 1.5
        assert config.target_win_rate == 0.6
        
        # Check reward config
        assert 'return_weight' in config.reward_config
        assert config.reward_config['return_weight'] == 0.4
    
    def test_agent_config_custom_values(self):
        """Test configuration with custom values"""
        custom_reward_config = {
            'return_weight': 0.5,
            'sharpe_weight': 0.25,
            'drawdown_weight': -0.25
        }
        
        config = AgentConfig(
            model_type=ModelType.DDQN,
            hidden_size=512,
            learning_rate=5e-4,
            max_position_size=0.15,
            reward_config=custom_reward_config
        )
        
        assert config.model_type == ModelType.DDQN
        assert config.hidden_size == 512
        assert config.learning_rate == 5e-4
        assert config.max_position_size == 0.15
        assert config.reward_config == custom_reward_config


class MockRLAgent(RLAgentBase):
    """Mock RL agent for testing abstract base class"""
    
    async def predict_action(self, state: MarketState) -> Tuple[TradeAction, float]:
        """Mock prediction"""
        return TradeAction.HOLD, 0.5
    
    async def train_step(self, batch_experiences):
        """Mock training step"""
        return {"loss": 0.1, "q_value": 0.5}
    
    def save_model(self, filepath: str) -> bool:
        """Mock save model"""
        return True
    
    def load_model(self, filepath: str) -> bool:
        """Mock load model"""
        self.is_trained = True
        return True


class TestRLAgentBase:
    """Test RLAgentBase abstract class"""
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration"""
        return AgentConfig(target_sharpe_ratio=1.2, target_win_rate=0.6, target_max_drawdown=0.1)
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create mock agent"""
        return MockRLAgent(agent_config)
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token"""
        from src.utils.base import Chain
        return DiscoveredToken(
            address="0x123...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            volume_24h=100000
        )
    
    def test_agent_initialization(self, mock_agent, agent_config):
        """Test agent initialization"""
        assert mock_agent.config == agent_config
        assert mock_agent.is_trained is False
        assert mock_agent.training_episodes == 0
        assert isinstance(mock_agent.performance_metrics, RewardMetrics)
    
    @pytest.mark.asyncio
    async def test_predict_action(self, mock_agent):
        """Test action prediction"""
        from src.utils.base import Chain
        state = MarketState(
            token=DiscoveredToken(
                address="0x123...", 
                symbol="TEST", 
                name="Test", 
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test"
            ),
            price_usd=1.50,
            price_change_24h=5.0,
            volume_24h=100000
        )
        
        action, confidence = await mock_agent.predict_action(state)
        
        assert isinstance(action, TradeAction)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_train_step(self, mock_agent):
        """Test training step"""
        batch_experiences = [
            {"state": [1, 2, 3], "action": 0, "reward": 0.1, "next_state": [1, 2, 4], "done": False}
        ]
        
        metrics = await mock_agent.train_step(batch_experiences)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "q_value" in metrics
    
    def test_update_performance_metrics_empty(self, mock_agent):
        """Test performance metrics update with empty results"""
        metrics = mock_agent.update_performance_metrics([])
        
        assert isinstance(metrics, RewardMetrics)
        assert metrics.absolute_return == 0.0
        assert metrics.win_rate == 0.0
    
    def test_update_performance_metrics_with_results(self, mock_agent, sample_token):
        """Test performance metrics update with trading results"""
        results = [
            TradingResult(
                action=TradeAction.BUY,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=100,
                value_usd=150.0,
                success=True,
                realized_pnl=10.0,
                portfolio_value_after=10100.0
            ),
            TradingResult(
                action=TradeAction.SELL,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.60,
                quantity=100,
                value_usd=160.0,
                success=True,
                realized_pnl=-5.0,
                portfolio_value_after=10095.0
            )
        ]
        
        metrics = mock_agent.update_performance_metrics(results)
        
        assert metrics.absolute_return == 5.0  # 10.0 - 5.0
        assert metrics.win_rate == 0.5  # 1 win out of 2 trades
    
    def test_meets_performance_targets_false(self, mock_agent):
        """Test performance targets check when not met"""
        # Default metrics don't meet targets
        assert mock_agent.meets_performance_targets() is False
    
    def test_meets_performance_targets_true(self, mock_agent):
        """Test performance targets check when met"""
        # Set metrics to meet targets
        mock_agent.performance_metrics.sharpe_ratio = 1.5
        mock_agent.performance_metrics.win_rate = 0.65
        mock_agent.performance_metrics.max_drawdown = 0.08
        
        assert mock_agent.meets_performance_targets() is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_agent):
        """Test health check"""
        health = await mock_agent.health_check()
        
        assert isinstance(health, dict)
        assert "is_trained" in health
        assert "training_episodes" in health
        assert "meets_targets" in health
        assert "performance_metrics" in health
        assert "config" in health
        
        assert health["is_trained"] is False
        assert health["training_episodes"] == 0
        assert health["meets_targets"] is False
    
    def test_save_load_model(self, mock_agent):
        """Test model save and load"""
        filepath = "/tmp/test_model.pth"
        
        # Test save
        assert mock_agent.save_model(filepath) is True
        
        # Test load
        assert mock_agent.load_model(filepath) is True
        assert mock_agent.is_trained is True


class TestRLExceptions:
    """Test RL-specific exceptions"""
    
    def test_rl_training_error(self):
        """Test RLTrainingError exception"""
        with pytest.raises(RLTrainingError):
            raise RLTrainingError("Training failed")
    
    def test_rl_prediction_error(self):
        """Test RLPredictionError exception"""
        with pytest.raises(RLPredictionError):
            raise RLPredictionError("Prediction failed")
    
    def test_rl_model_error(self):
        """Test RLModelError exception"""
        from src.rl_agent.base import RLModelError
        
        with pytest.raises(RLModelError):
            raise RLModelError("Model operation failed")