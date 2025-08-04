"""
Integration tests for Phase 1.4 - Full Model Manager Integration
Tests the complete pipeline from Transformer models through to RL agent features
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import ModelType, PredictionResult, PredictionDirection
from src.integration.ml_rl_bridge import MLEnhancedMarketState, MLRLBridge
from src.rl_agent.base import MarketState
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestPhase14Integration:
    """Comprehensive integration tests for Phase 1.4 Transformer integration"""
    
    @pytest.fixture
    def sample_token(self):
        """Create a sample token for testing"""
        return DiscoveredToken(
            address="0x123456789abcdef",
            chain=Chain.ETHEREUM,
            symbol="TESTCOIN",
            name="Test Coin",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            market_cap=10000000.0,
            volume_24h=1000000.0
        )
    
    @pytest.fixture
    def historical_data(self):
        """Create sample historical data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'timestamp': dates,
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
    
    @pytest.fixture
    def model_manager_config(self):
        """Configuration for ModelManager with all Transformers"""
        return {
            'cache_ttl_minutes': 1,
            'model_dir': 'test_models',
            'lstm': {'sequence_length': 20, 'hidden_size': 32},
            'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2},
            'itransformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2},
            'patchtst': {'d_model': 64, 'patch_len': 8, 'stride': 4},
            'timesmixer': {'d_model': 64, 'seq_len': 48, 'pred_len': 12}
        }
    
    def test_model_manager_initializes_all_transformers(self, model_manager_config):
        """Test that ModelManager initializes all five models (LSTM + 4 Transformers)"""
        model_manager = ModelManager(model_manager_config)
        
        # Verify all expected models are initialized
        expected_models = {
            ModelType.LSTM, ModelType.TRANSFORMER, ModelType.ITRANSFORMER,
            ModelType.PATCHTST, ModelType.TIMESMIXER
        }
        
        assert set(model_manager._models.keys()) == expected_models
        
        # Verify weights are properly distributed
        assert len(model_manager._model_weights) == 5
        assert abs(sum(model_manager._model_weights.values()) - 1.0) < 0.001  # Sum to 1.0
        
        # Verify performance tracking is set up
        for model_type in expected_models:
            assert model_type in model_manager._model_performance
            perf = model_manager._model_performance[model_type]
            assert 'accuracy' in perf
            assert 'predictions_made' in perf
            assert 'memory_usage_mb' in perf
            assert 'avg_inference_time_ms' in perf
    
    def test_regime_based_weighting(self, model_manager_config):
        """Test market regime-aware model weighting"""
        model_manager = ModelManager(model_manager_config)
        
        # Test different market regimes
        regimes = ['bull', 'bear', 'volatile', 'sideways', 'normal']
        
        for regime in regimes:
            for model_type in [ModelType.LSTM, ModelType.ITRANSFORMER, ModelType.TIMESMIXER]:
                factor = model_manager._get_regime_factor(model_type, regime)
                assert 0.0 < factor <= 2.0  # Reasonable factor range
        
        # Test specific regime preferences
        volatile_lstm = model_manager._get_regime_factor(ModelType.LSTM, 'volatile')
        volatile_timesmixer = model_manager._get_regime_factor(ModelType.TIMESMIXER, 'volatile')
        
        # TimesMixer should have higher factor in volatile markets
        assert volatile_timesmixer > volatile_lstm
    
    def test_enhanced_market_state_feature_extraction(self, sample_token):
        """Test MLEnhancedMarketState extracts Transformer features correctly"""
        # Create a prediction result with ensemble information
        prediction = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_1h=105.0,
            price_prediction_4h=110.0,
            price_prediction_24h=120.0,
            direction=PredictionDirection.BUY,
            confidence=0.85,
            probability_up=0.8,
            prediction_uncertainty=0.12,
            features_used=['ensemble_5_models', 'itransformer_weight_0.25', 'patchtst_weight_0.20']
        )
        
        # Mock model weights
        model_weights = {
            ModelType.LSTM: 0.25,
            ModelType.TRANSFORMER: 0.15,
            ModelType.ITRANSFORMER: 0.25,
            ModelType.PATCHTST: 0.20,
            ModelType.TIMESMIXER: 0.15
        }
        
        # Create enhanced market state
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.1,
            model_weights=model_weights
        )
        
        # Verify Transformer features are extracted
        assert enhanced_state.prediction_uncertainty == 0.12
        assert enhanced_state.model_consensus == 1.0  # 5 models / 5 max
        assert enhanced_state.transformer_weight == 0.75  # Sum of Transformer weights
        assert enhanced_state.lstm_weight == 0.25
        assert enhanced_state.ensemble_diversity is not None
        assert enhanced_state.ensemble_diversity > 0  # Should have some diversity
    
    def test_feature_vector_size_compatibility(self, sample_token):
        """Test that feature vector size matches DQN agent expectations"""
        # Create a basic prediction
        prediction = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.TRANSFORMER,
            direction=PredictionDirection.HOLD,
            confidence=0.7
        )
        
        # Create enhanced market state
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Get feature vector
        feature_vector = enhanced_state.to_feature_vector()
        
        # Verify size matches expected DQN input size
        expected_size = MarketState.get_enhanced_feature_size()
        assert len(feature_vector) == expected_size == 34
        
        # Verify feature vector contains reasonable values
        assert not any(pd.isna(feature_vector))  # No NaN values
        assert not any(np.isinf(feature_vector))  # No infinite values
        
        # Verify feature ranges are reasonable
        for i, value in enumerate(feature_vector):
            assert -1000 <= value <= 1000, f"Feature {i} has unreasonable value: {value}"
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self, model_manager_config):
        """Test that memory usage is tracked for Transformer models"""
        model_manager = ModelManager(model_manager_config)
        
        # Test memory usage calculation for different model types
        transformer_types = [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, 
                           ModelType.PATCHTST, ModelType.TIMESMIXER]
        
        for model_type in transformer_types:
            memory_usage = await model_manager._get_model_memory_usage(model_type)
            assert memory_usage >= 0.0  # Memory usage should be non-negative
            assert memory_usage < 10000.0  # Should be reasonable (< 10GB)
    
    def test_ml_rl_bridge_integration(self, sample_token, model_manager_config):
        """Test complete ML-RL Bridge integration with Transformers"""
        # Create mock ML analyzer and RL agent
        mock_ml_analyzer = Mock()
        mock_rl_agent = Mock()
        
        # Mock prediction with Transformer features
        mock_prediction = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_24h=115.0,
            direction=PredictionDirection.BUY,
            confidence=0.82,
            prediction_uncertainty=0.08,
            features_used=['ensemble_4_models']
        )
        
        # Mock ML analyzer methods
        mock_ml_analyzer.batch_analyze = AsyncMock(return_value=[mock_prediction])
        mock_ml_analyzer.get_model_performance = Mock(return_value={
            'weights': {
                ModelType.LSTM: 0.2,
                ModelType.ITRANSFORMER: 0.3,
                ModelType.PATCHTST: 0.3,
                ModelType.TIMESMIXER: 0.2
            }
        })
        
        # Mock RL agent
        mock_rl_agent.predict_action = Mock(return_value='BUY')
        
        # Create bridge
        bridge = MLRLBridge(
            ml_analyzer=mock_ml_analyzer,
            rl_agent=mock_rl_agent,
            tokens=[sample_token]
        )
        
        # Test predict and act
        results = bridge.predict_and_act(
            portfolio_value=10000.0,
            positions={}
        )
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        
        assert result['token'] == sample_token
        assert result['ml_prediction'] == mock_prediction
        assert result['rl_action'] == 'BUY'
        assert 'enhanced_state' in result
        assert 'transformer_confidence' in result
        assert 'model_consensus' in result
        assert 'prediction_uncertainty' in result
        
        # Verify enhanced state has Transformer features
        enhanced_state = result['enhanced_state']
        assert enhanced_state.transformer_weight == 0.8  # 0.3 + 0.3 + 0.2
        assert enhanced_state.lstm_weight == 0.2
        assert enhanced_state.prediction_uncertainty == 0.08
    
    def test_model_weight_normalization(self, model_manager_config):
        """Test that model weights are properly normalized"""
        model_manager = ModelManager(model_manager_config)
        
        # Test weight normalization
        total_weight = sum(model_manager._model_weights.values())
        assert abs(total_weight - 1.0) < 0.001  # Should sum to 1.0
        
        # Test individual weight ranges
        for model_type, weight in model_manager._model_weights.items():
            assert 0.02 <= weight <= 1.0  # Between minimum and maximum
    
    @pytest.mark.asyncio
    async def test_dynamic_weight_updates(self, model_manager_config):
        """Test dynamic model weight updates with performance data"""
        model_manager = ModelManager(model_manager_config)
        
        # Simulate different performance for different models
        model_manager._model_performance[ModelType.ITRANSFORMER]['accuracy'] = 0.9
        model_manager._model_performance[ModelType.ITRANSFORMER]['predictions_made'] = 100
        
        model_manager._model_performance[ModelType.LSTM]['accuracy'] = 0.7
        model_manager._model_performance[ModelType.LSTM]['predictions_made'] = 150
        
        # Update weights with volatile market regime (favors TimesMixer and iTransformer)
        await model_manager._update_model_weights('volatile')
        
        # Verify that iTransformer gets higher weight due to better accuracy + regime bonus
        itransformer_weight = model_manager._model_weights[ModelType.ITRANSFORMER]
        lstm_weight = model_manager._model_weights[ModelType.LSTM]
        
        assert itransformer_weight > lstm_weight  # Better performing model should have higher weight
        
        # Verify total weight still sums to 1.0
        total_weight = sum(model_manager._model_weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_transformer_specific_features_bounds(self, sample_token):
        """Test that Transformer-specific features are within expected bounds"""
        prediction = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            direction=PredictionDirection.BUY,
            confidence=0.8,
            prediction_uncertainty=0.05
        )
        
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.5
        )
        
        # Test bounds for Transformer features
        assert 0.0 <= enhanced_state.prediction_uncertainty <= 1.0
        assert 0.0 <= enhanced_state.model_consensus <= 1.0
        assert 0.0 <= enhanced_state.attention_focus <= 1.0
        assert 0.0 <= enhanced_state.temporal_importance <= 1.0
        assert -1.0 <= enhanced_state.cross_asset_correlation <= 1.0
        assert 0.0 <= enhanced_state.regime_confidence <= 1.0
        assert 0.0 <= enhanced_state.transformer_weight <= 1.0
        assert 0.0 <= enhanced_state.lstm_weight <= 1.0
        assert 0.0 <= enhanced_state.ensemble_diversity <= 1.0
    
    def test_phase_1_4_complete_pipeline(self, sample_token, model_manager_config):
        """Test the complete Phase 1.4 pipeline end-to-end"""
        # Initialize ModelManager with all Transformers
        model_manager = ModelManager(model_manager_config)
        
        # Verify initialization
        assert len(model_manager._models) == 5  # LSTM + 4 Transformers
        assert len(model_manager._model_weights) == 5
        assert len(model_manager._model_performance) == 5
        
        # Verify regime factors work for all models
        for model_type in model_manager._models.keys():
            factor = model_manager._get_regime_factor(model_type, 'volatile')
            assert 0.5 <= factor <= 2.0  # Reasonable range
        
        # Create a mock prediction that would come from ensemble
        prediction = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_1h=102.0,
            price_prediction_4h=108.0,
            price_prediction_24h=118.0,
            direction=PredictionDirection.BUY,
            confidence=0.87,
            prediction_uncertainty=0.06,
            features_used=['ensemble_5_models', 'transformer_total_weight_0.70']
        )
        
        # Create enhanced market state with model weights
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.15,
            model_weights=model_manager._model_weights
        )
        
        # Verify enhanced state has all required features
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 34  # 19 + 6 + 9
        
        # Verify Transformer-specific features are populated
        assert enhanced_state.transformer_weight > 0
        assert enhanced_state.prediction_uncertainty == 0.06
        assert enhanced_state.model_consensus > 0
        
        # Verify feature vector is compatible with DQN
        expected_size = MarketState.get_enhanced_feature_size()
        assert len(feature_vector) == expected_size
        
        print(f"✅ Phase 1.4 integration test passed!")
        print(f"   - Models initialized: {len(model_manager._models)}")
        print(f"   - Feature vector size: {len(feature_vector)}")
        print(f"   - Transformer weight: {enhanced_state.transformer_weight:.3f}")
        print(f"   - Model consensus: {enhanced_state.model_consensus:.3f}")
        print(f"   - Prediction uncertainty: {enhanced_state.prediction_uncertainty:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])