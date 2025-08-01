"""
Test that mock implementations have been successfully removed and replaced with real ones
"""

import pytest
import asyncio
from datetime import datetime
import numpy as np
import pandas as pd

from src.integration.ml_rl_bridge import MLRLTrainingPipeline
from src.rl_agent.training_pipeline import TrainingConfig
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestMockRemovalIntegration:
    """Test that all mocks have been removed and replaced with real implementations"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        return [
            DiscoveredToken(
                address="0x1234567890123456789012345678901234567890",
                symbol="TEST1",
                name="Test Token 1",
                chain=Chain.ETHEREUM,
                price_usd=100.0,
                volume_24h=1000000.0,
                market_cap=10000000.0,
                discovered_at=datetime.now(),
                discovery_source="test"
            ),
            DiscoveredToken(
                address="0x2345678901234567890123456789012345678901",
                symbol="TEST2", 
                name="Test Token 2",
                chain=Chain.ETHEREUM,
                price_usd=200.0,
                volume_24h=2000000.0,
                market_cap=20000000.0,
                discovered_at=datetime.now(),
                discovery_source="test"
            )
        ]
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        return {
            "0x1234567890123456789012345678901234567890": pd.DataFrame({
                'timestamp': dates,
                'price': np.random.normal(100, 10, len(dates)),
                'volume': np.random.exponential(1000, len(dates)),
                'market_cap': np.random.normal(10000000, 100000, len(dates))
            }),
            "0x2345678901234567890123456789012345678901": pd.DataFrame({
                'timestamp': dates,
                'price': np.random.normal(200, 20, len(dates)),
                'volume': np.random.exponential(2000, len(dates)), 
                'market_cap': np.random.normal(20000000, 200000, len(dates))
            })
        }
    
    @pytest.fixture
    def training_config(self):
        """Create training configuration"""
        return TrainingConfig(
            num_episodes=10,
            max_steps_per_episode=50,
            learning_rate=1e-4,
            batch_size=16
        )
    
    def test_ml_rl_pipeline_uses_real_implementations(self, sample_tokens, sample_historical_data, training_config):
        """Test that MLRLTrainingPipeline uses real ML and RL implementations"""
        pipeline = MLRLTrainingPipeline(
            config=training_config,
            tokens=sample_tokens,
            historical_data=sample_historical_data
        )
        
        # Verify that real implementations are used, not mocks
        assert hasattr(pipeline.ml_analyzer, '_model'), "ML analyzer should have real model attribute"
        assert hasattr(pipeline.rl_agent, 'is_trained'), "RL agent should have real training state"
        
        # Verify the components are proper instances, not mocks
        from src.ml_analysis.lstm_model import LSTMPricePredictor
        from src.rl_agent.dqn_agent import DQNTradingAgent
        
        assert isinstance(pipeline.ml_analyzer, LSTMPricePredictor), "Should use real LSTM analyzer"
        assert isinstance(pipeline.rl_agent, DQNTradingAgent), "Should use real DQN agent"
        
        # Verify no mock objects are present
        import unittest.mock
        assert not isinstance(pipeline.ml_analyzer, unittest.mock.MagicMock), "ML analyzer should not be a mock"
        assert not isinstance(pipeline.rl_agent, unittest.mock.MagicMock), "RL agent should not be a mock"
    
    def test_ml_analyzer_has_real_methods(self, sample_tokens, sample_historical_data, training_config):
        """Test that ML analyzer has real method implementations"""
        pipeline = MLRLTrainingPipeline(
            config=training_config,
            tokens=sample_tokens,
            historical_data=sample_historical_data
        )
        
        analyzer = pipeline.ml_analyzer
        
        # Check that methods exist and are not mock objects
        assert callable(analyzer.get_required_features), "Should have get_required_features method"
        assert callable(analyzer.is_model_trained), "Should have is_model_trained method"
        assert callable(analyzer.train_model), "Should have train_model method"
        assert callable(analyzer.analyze_token), "Should have analyze_token method"
        assert callable(analyzer.batch_analyze), "Should have batch_analyze method"
        
        # Verify method returns real values
        required_features = analyzer.get_required_features()
        assert isinstance(required_features, list), "Should return list of feature names"
        assert len(required_features) > 0, "Should have required features"
        assert 'price' in required_features, "Should require price feature"
        
        # Verify training state
        assert analyzer.is_model_trained() == False, "Model should not be trained initially"
    
    def test_rl_agent_has_real_methods(self, sample_tokens, sample_historical_data, training_config):
        """Test that RL agent has real method implementations"""
        pipeline = MLRLTrainingPipeline(
            config=training_config,
            tokens=sample_tokens, 
            historical_data=sample_historical_data
        )
        
        agent = pipeline.rl_agent
        
        # Check that methods exist and are not mock objects
        assert callable(agent.predict_action), "Should have predict_action method"
        assert callable(agent.train_step), "Should have train_step method"
        assert callable(agent.save_model), "Should have save_model method" 
        assert callable(agent.load_model), "Should have load_model method"
        assert callable(agent.health_check), "Should have health_check method"
        
        # Verify agent configuration
        assert hasattr(agent, 'config'), "Should have configuration"
        assert hasattr(agent, 'is_trained'), "Should have training state"
        assert agent.is_trained == False, "Agent should not be trained initially"
    
    @pytest.mark.asyncio
    async def test_ml_rl_bridge_integration(self, sample_tokens, sample_historical_data, training_config):
        """Test that ML-RL bridge works with real implementations"""
        pipeline = MLRLTrainingPipeline(
            config=training_config,
            tokens=sample_tokens,
            historical_data=sample_historical_data
        )
        
        bridge = pipeline.bridge
        
        # Verify bridge components are real implementations
        assert not hasattr(bridge.ml_analyzer, '_mock_name'), "ML analyzer should not be a mock"
        assert not hasattr(bridge.rl_agent, '_mock_name'), "RL agent should not be a mock"
        
        # Test that bridge can handle real predictions (may fail due to untrained models, which is expected)
        try:
            predictions = await bridge.get_ml_predictions()
            # If we get here, predictions were returned (possibly low-confidence fallback predictions)
            assert isinstance(predictions, list), "Should return list of predictions"
        except Exception as e:
            # Expected for untrained models - verify it's the right kind of error
            assert "trained" in str(e).lower() or "model" in str(e).lower(), f"Should fail due to untrained model, got: {e}"
    
    def test_no_mock_imports_in_production_code(self):
        """Test that production code doesn't import mock modules"""
        # Read the ml_rl_bridge file and verify no mock imports
        import ast
        import inspect
        from src.integration import ml_rl_bridge
        
        source = inspect.getsource(ml_rl_bridge)
        tree = ast.parse(source)
        
        # Check for unittest.mock imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert 'mock' not in alias.name.lower(), f"Found mock import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'mock' in node.module.lower():
                    # Only allow mock imports in test-specific functions that are clearly marked
                    assert False, f"Found mock import: from {node.module}"
    
    def test_performance_metrics_still_work(self, sample_tokens, sample_historical_data, training_config):
        """Test that performance metrics still work with real implementations"""
        pipeline = MLRLTrainingPipeline(
            config=training_config,
            tokens=sample_tokens,
            historical_data=sample_historical_data
        )
        
        # Verify metrics tracking is in place
        assert hasattr(pipeline, 'integration_metrics'), "Should have integration metrics"
        assert hasattr(pipeline.integration_metrics, 'ml_accuracy_history'), "Should track ML accuracy"
        assert hasattr(pipeline.integration_metrics, 'rl_reward_history'), "Should track RL rewards"
        
        # Initial state should be empty
        stats = pipeline.integration_metrics.get_statistics()
        assert stats['episodes_completed'] == 0, "Should start with 0 episodes"
        assert stats['mean_ml_accuracy'] == 0.0, "Should start with 0 accuracy"