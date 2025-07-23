"""
Tests for ML-RL Integration Pipeline

Following TDD methodology - these tests define the expected behavior
of the integrated ML-RL trading system before implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Any, Optional

from src.rl_agent.base import TradeAction, MarketState, TradingResult, AgentConfig
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.training_pipeline import TrainingConfig, DQNTrainingPipeline
from src.ml_analysis.base import PredictionResult, ModelType, PredictionDirection, TechnicalIndicators
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestMLRLIntegration:
    """Test ML-RL integration components"""
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        return DiscoveredToken(
            address="0x123...",
            symbol="TESTCOIN",
            name="Test Coin",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            volume_24h=250000
        )
    
    @pytest.fixture
    def sample_ml_prediction(self, sample_token):
        """Create sample ML prediction"""
        return PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=1.55,
            price_prediction_4h=1.60,
            price_prediction_24h=1.65,
            direction=PredictionDirection.BUY,
            confidence=0.75,
            probability_up=0.68,
            volatility_forecast=0.15,
            stop_loss_level=1.35,
            take_profit_level=1.80
        )
    
    def test_ml_enhanced_market_state_creation(self, sample_token, sample_ml_prediction):
        """Test creating ML-enhanced market state from predictions"""
        # This test will fail initially - we need to implement MLEnhancedMarketState
        from src.integration.ml_rl_bridge import MLEnhancedMarketState
        
        state = MLEnhancedMarketState.from_prediction(
            prediction=sample_ml_prediction,
            current_portfolio_value=10000,
            position_size=0.0
        )
        
        # Should include both RL state features and ML predictions
        assert state.token == sample_token
        assert state.price_usd == sample_token.price_usd
        assert state.ml_prediction_1h == 1.55
        assert state.ml_prediction_24h == 1.65
        assert state.ml_confidence == 0.75
        assert state.volatility_forecast == 0.15
    
    def test_ml_enhanced_market_state_feature_vector(self, sample_token, sample_ml_prediction):
        """Test ML-enhanced market state converts to feature vector"""
        from src.integration.ml_rl_bridge import MLEnhancedMarketState
        
        state = MLEnhancedMarketState.from_prediction(
            prediction=sample_ml_prediction,
            current_portfolio_value=10000,
            position_size=0.1
        )
        
        feature_vector = state.to_feature_vector()
        
        # Should be larger than standard MarketState (19 -> 25+ features)
        assert len(feature_vector) >= 25
        assert isinstance(feature_vector, np.ndarray)
        assert feature_vector.dtype == np.float32
        
        # Should include ML prediction features
        ml_features = feature_vector[19:]  # After standard RL features
        assert len(ml_features) >= 6  # ML predictions + confidence + volatility


class TestMLRLBridge:
    """Test ML-RL bridge component that connects predictions to actions"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        return [
            DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            for i in range(3)
        ]
    
    @pytest.fixture
    def mock_ml_predictions(self, sample_tokens):
        """Create mock ML predictions"""
        predictions = []
        for i, token in enumerate(sample_tokens):
            prediction = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_1h=token.price_usd * (1.0 + 0.02 * (i + 1)),
                price_prediction_24h=token.price_usd * (1.0 + 0.05 * (i + 1)),
                direction=PredictionDirection.BUY if i % 2 == 0 else PredictionDirection.SELL,
                confidence=0.6 + i * 0.1,
                probability_up=0.5 + i * 0.05,
                volatility_forecast=0.1 + i * 0.02
            )
            predictions.append(prediction)
        return predictions
    
    def test_ml_rl_bridge_initialization(self, sample_tokens):
        """Test ML-RL bridge should initialize with required components"""
        from src.integration.ml_rl_bridge import MLRLBridge
        
        # Mock ML analyzer and RL agent
        mock_ml_analyzer = MagicMock()
        mock_rl_agent = MagicMock()
        
        bridge = MLRLBridge(
            ml_analyzer=mock_ml_analyzer,
            rl_agent=mock_rl_agent,
            tokens=sample_tokens
        )
        
        assert bridge.ml_analyzer == mock_ml_analyzer
        assert bridge.rl_agent == mock_rl_agent
        assert len(bridge.tokens) == 3
        assert hasattr(bridge, 'prediction_cache')
    
    def test_ml_rl_bridge_predict_and_act(self, sample_tokens, mock_ml_predictions):
        """Test integrated prediction and action workflow"""
        from src.integration.ml_rl_bridge import MLRLBridge
        
        # Mock components
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = mock_ml_predictions
        
        mock_rl_agent = MagicMock()
        mock_rl_agent.predict_action.return_value = TradeAction.BUY
        
        bridge = MLRLBridge(
            ml_analyzer=mock_ml_analyzer,
            rl_agent=mock_rl_agent,
            tokens=sample_tokens
        )
        
        # Should be able to get integrated predictions and actions
        results = bridge.predict_and_act(portfolio_value=10000, positions={})
        
        assert len(results) == 3
        for result in results:
            assert 'token' in result
            assert 'ml_prediction' in result
            assert 'rl_action' in result
            assert 'confidence' in result
    
    def test_ml_rl_bridge_caching(self, sample_tokens):
        """Test ML prediction caching for performance"""
        from src.integration.ml_rl_bridge import MLRLBridge
        
        mock_ml_analyzer = MagicMock()
        mock_rl_agent = MagicMock()
        
        bridge = MLRLBridge(
            ml_analyzer=mock_ml_analyzer,
            rl_agent=mock_rl_agent,
            tokens=sample_tokens,
            cache_ttl_minutes=5
        )
        
        # First call should hit ML analyzer
        bridge.get_ml_predictions()
        assert mock_ml_analyzer.analyze_batch.call_count == 1
        
        # Second call within TTL should use cache
        bridge.get_ml_predictions()
        assert mock_ml_analyzer.analyze_batch.call_count == 1  # No additional call


class TestMLRLTrainingPipeline:
    """Test integrated ML-RL training pipeline"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for training"""
        return [
            DiscoveredToken(
                address=f"0x{i:04d}...",
                symbol=f"TKN{i}",
                name=f"Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now() - timedelta(days=i),
                discovery_source="test",
                price_usd=1.0 + i * 0.25,
                volume_24h=100000 * (i + 1)
            )
            for i in range(2)
        ]
    
    @pytest.fixture
    def mock_price_data(self, sample_tokens):
        """Create mock price data"""
        price_data = {}
        np.random.seed(42)  # For reproducible tests
        
        for token in sample_tokens:
            prices = []
            base_price = token.price_usd
            
            for j in range(100):
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                    price = max(price, 0.01)
                
                prices.append(price)
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [
                    datetime.now() - timedelta(hours=100-i) 
                    for i in range(100)
                ]
            }
        
        return price_data
    
    def test_ml_rl_training_pipeline_initialization(self, sample_tokens, mock_price_data):
        """Test ML-RL integrated training pipeline initialization"""
        from src.integration.ml_rl_bridge import MLRLTrainingPipeline
        
        config = TrainingConfig(
            num_episodes=10,
            max_steps_per_episode=20,
            use_ml_features=True,
            ml_weight=0.3  # 30% ML, 70% RL
        )
        
        pipeline = MLRLTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Should initialize both ML and RL components
        assert pipeline.config == config
        assert pipeline.ml_analyzer is not None
        assert pipeline.rl_agent is not None
        assert pipeline.bridge is not None
        assert len(pipeline.tokens) == 2
    
    def test_ml_rl_training_episode(self, sample_tokens, mock_price_data):
        """Test running training episode with ML-RL integration"""
        from src.integration.ml_rl_bridge import MLRLTrainingPipeline
        
        config = TrainingConfig(
            num_episodes=1,
            max_steps_per_episode=10,
            use_ml_features=True
        )
        
        pipeline = MLRLTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Should be able to run episode with ML-RL integration
        episode_result = pipeline.run_ml_rl_episode(episode_num=0)
        
        assert 'total_reward' in episode_result
        assert 'ml_predictions_used' in episode_result
        assert 'rl_actions_taken' in episode_result
        assert 'integrated_decisions' in episode_result
        assert isinstance(episode_result['ml_predictions_used'], int)
        assert episode_result['ml_predictions_used'] > 0
    
    def test_ml_rl_training_full_pipeline(self, sample_tokens, mock_price_data):
        """Test complete ML-RL training process"""
        from src.integration.ml_rl_bridge import MLRLTrainingPipeline
        
        config = TrainingConfig(
            num_episodes=3,
            max_steps_per_episode=5,
            use_ml_features=True,
            ml_weight=0.4
        )
        
        pipeline = MLRLTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Run complete integrated training
        results = pipeline.train_integrated()
        
        assert 'episodes_completed' in results
        assert 'ml_rl_metrics' in results
        assert 'integration_performance' in results
        assert results['episodes_completed'] == 3
        
        # Should track both ML and RL performance
        ml_rl_metrics = results['ml_rl_metrics']
        assert 'ml_accuracy' in ml_rl_metrics
        assert 'rl_rewards' in ml_rl_metrics
        assert 'integration_efficiency' in ml_rl_metrics


class TestMLRLConfiguration:
    """Test ML-RL integration configuration"""
    
    def test_ml_rl_config_creation(self):
        """Test ML-RL configuration with default values"""
        from src.integration.ml_rl_bridge import MLRLConfig
        
        config = MLRLConfig()
        
        # Should have sensible defaults for ML-RL integration
        assert isinstance(config.ml_weight, float)
        assert 0.0 <= config.ml_weight <= 1.0
        assert isinstance(config.rl_weight, float)
        assert config.ml_weight + config.rl_weight == 1.0
        assert isinstance(config.prediction_horizon, str)
        assert config.cache_ttl_minutes > 0
    
    def test_ml_rl_config_custom_values(self):
        """Test ML-RL configuration with custom values"""
        from src.integration.ml_rl_bridge import MLRLConfig
        
        config = MLRLConfig(
            ml_weight=0.3,
            rl_weight=0.7,
            prediction_horizon="4h",
            cache_ttl_minutes=10,
            use_ensemble_predictions=True
        )
        
        assert config.ml_weight == 0.3
        assert config.rl_weight == 0.7
        assert config.prediction_horizon == "4h"
        assert config.cache_ttl_minutes == 10
        assert config.use_ensemble_predictions is True


class TestMLRLIntegrationError:
    """Test ML-RL integration error handling"""
    
    def test_ml_rl_integration_error_creation(self):
        """Test ML-RL integration error exception"""
        from src.integration.ml_rl_bridge import MLRLIntegrationError
        
        error = MLRLIntegrationError("Integration failed")
        assert isinstance(error, Exception)
        assert str(error) == "Integration failed"
    
    def test_ml_rl_integration_error_inheritance(self):
        """Test error inheritance from base errors"""
        from src.integration.ml_rl_bridge import MLRLIntegrationError
        from src.rl_agent.base import RLTrainingError
        
        error = MLRLIntegrationError("Test error")
        assert isinstance(error, RLTrainingError)


class TestMLRLPerformanceMetrics:
    """Test ML-RL integration performance tracking"""
    
    def test_integration_metrics_creation(self):
        """Test integration performance metrics tracking"""
        from src.integration.ml_rl_bridge import MLRLPerformanceMetrics
        
        metrics = MLRLPerformanceMetrics()
        
        # Should track both ML and RL performance
        assert hasattr(metrics, 'ml_accuracy_history')
        assert hasattr(metrics, 'rl_reward_history')
        assert hasattr(metrics, 'integration_scores')
        assert hasattr(metrics, 'decision_alignment')
        assert len(metrics.ml_accuracy_history) == 0
    
    def test_integration_metrics_update(self):
        """Test updating integration performance metrics"""
        from src.integration.ml_rl_bridge import MLRLPerformanceMetrics
        
        metrics = MLRLPerformanceMetrics()
        
        # Should be able to add episode data
        metrics.add_episode_data(
            episode=1,
            ml_accuracy=0.75,
            rl_reward=150.0,
            decisions_aligned=8,
            total_decisions=10,
            integration_latency=0.05
        )
        
        assert len(metrics.ml_accuracy_history) == 1
        assert metrics.ml_accuracy_history[0] == 0.75
        assert len(metrics.rl_reward_history) == 1
        assert metrics.rl_reward_history[0] == 150.0
        assert len(metrics.decision_alignment) == 1
        assert metrics.decision_alignment[0] == 0.8  # 8/10
    
    def test_integration_metrics_statistics(self):
        """Test computing integration statistics"""
        from src.integration.ml_rl_bridge import MLRLPerformanceMetrics
        
        metrics = MLRLPerformanceMetrics()
        
        # Add sample data
        for i in range(5):
            metrics.add_episode_data(
                episode=i,
                ml_accuracy=0.7 + i * 0.02,
                rl_reward=100 + i * 20,
                decisions_aligned=7 + i,
                total_decisions=10,
                integration_latency=0.05 - i * 0.005
            )
        
        stats = metrics.get_statistics()
        
        assert 'mean_ml_accuracy' in stats
        assert 'mean_rl_reward' in stats
        assert 'mean_decision_alignment' in stats
        assert 'mean_integration_latency' in stats
        assert stats['episodes_completed'] == 5
        assert stats['mean_ml_accuracy'] > 0.7
        assert stats['mean_decision_alignment'] > 0.7


class TestMLRLIntegrationEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def full_integration_setup(self):
        """Create complete integration setup"""
        # Create tokens
        tokens = [
            DiscoveredToken(
                address="0x0001...",
                symbol="INTTOKEN",
                name="Integration Token",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=2.0,
                volume_24h=500000
            )
        ]
        
        # Create price data
        np.random.seed(123)
        price_data = {}
        
        for token in tokens:
            prices = []
            base_price = token.price_usd
            
            for j in range(50):
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.002, 0.03)
                    price = prices[-1] * (1 + change)
                    price = max(price, 0.1)
                
                prices.append(price)
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [
                    datetime.now() - timedelta(hours=50-i) 
                    for i in range(50)
                ]
            }
        
        return tokens, price_data
    
    def test_complete_ml_rl_integration_workflow(self, full_integration_setup):
        """Test complete ML-RL integration workflow"""
        tokens, price_data = full_integration_setup
        
        from src.integration.ml_rl_bridge import MLRLTrainingPipeline, MLRLConfig
        
        # Configuration for integrated training
        training_config = TrainingConfig(
            num_episodes=2,
            max_steps_per_episode=10,
            use_ml_features=True
        )
        
        ml_rl_config = MLRLConfig(
            ml_weight=0.4,
            rl_weight=0.6,
            prediction_horizon="1h"
        )
        
        pipeline = MLRLTrainingPipeline(
            training_config=training_config,
            ml_rl_config=ml_rl_config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Run complete integrated training
        results = pipeline.train_integrated()
        
        # Validate comprehensive results
        assert results['episodes_completed'] == 2
        assert 'training_time' in results
        assert 'ml_rl_metrics' in results
        assert 'integration_performance' in results
        
        # Check integration-specific metrics
        integration_perf = results['integration_performance']
        assert 'ml_rl_correlation' in integration_perf
        assert 'decision_consistency' in integration_perf
        assert 'prediction_accuracy' in integration_perf
        
        # Verify pipeline components worked together
        assert pipeline.bridge is not None
        assert hasattr(pipeline, 'integration_metrics')
        assert len(pipeline.integration_metrics.ml_accuracy_history) == 2