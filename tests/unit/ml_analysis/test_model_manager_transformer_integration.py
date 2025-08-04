"""
Test Transformer integration with ModelManager
Following TDD principles - these tests should FAIL initially
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timedelta

from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import ModelType, PredictionResult, PredictionDirection
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestModelManagerTransformerIntegration:
    """Test Transformer model integration with ModelManager"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration with Transformer models enabled"""
        return {
            'cache_ttl_minutes': 5,
            'model_dir': 'test_models',
            'transformer': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'sequence_length': 60,
                'prediction_horizons': [1, 4, 24]
            },
            'itransformer': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'sequence_length': 60,
                'invert_time_attention': True
            },
            'patchtst': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'patch_len': 16,
                'stride': 8
            }
        }
    
    @pytest.fixture
    def sample_token(self):
        """Sample token for testing"""
        return DiscoveredToken(
            address="0x123...",
            name="Test Token",
            symbol="TEST",
            chain=Chain.ETHEREUM,
            price_usd=1.0,
            market_cap=1000000,
            volume_24h=100000,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
    
    def test_transformer_models_initialization(self, mock_config):
        """Test that Transformer models are properly initialized in ModelManager"""
        # This should FAIL initially - Transformer models not yet integrated
        model_manager = ModelManager(mock_config)
        
        # Should contain TRANSFORMER, ITRANSFORMER, PATCHTST model types
        expected_types = {ModelType.LSTM, ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST}
        actual_types = set(model_manager._models.keys())
        
        assert expected_types.issubset(actual_types), f"Missing transformer models. Expected: {expected_types}, Got: {actual_types}"
    
    def test_transformer_model_weights_initialization(self, mock_config):
        """Test that Transformer models have proper weights assigned"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        transformer_types = [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]
        for model_type in transformer_types:
            assert model_type in model_manager._model_weights, f"Missing weight for {model_type}"
            assert 0 < model_manager._model_weights[model_type] <= 1.0, f"Invalid weight for {model_type}"
    
    @pytest.mark.asyncio
    async def test_transformer_analyze_token(self, mock_config, sample_token):
        """Test token analysis with Transformer models"""
        # This should FAIL initially - Transformer models not implemented
        model_manager = ModelManager(mock_config)
        
        # Mock historical data
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': [1.0] * 100,
            'high': [1.1] * 100, 
            'low': [0.9] * 100,
            'close': [1.0] * 100,
            'volume': [1000] * 100
        })
        
        # Should be able to analyze with Transformer models
        result = await model_manager.analyze_token(sample_token, historical_data, use_ensemble=False)
        
        assert isinstance(result, PredictionResult)
        assert result.model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]
        assert result.confidence > 0.0
        assert result.price_prediction_24h is not None
    
    @pytest.mark.asyncio 
    async def test_transformer_ensemble_prediction(self, mock_config, sample_token):
        """Test ensemble prediction including Transformer models"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': [1.0] * 100,
            'high': [1.1] * 100,
            'low': [0.9] * 100, 
            'close': [1.0] * 100,
            'volume': [1000] * 100
        })
        
        result = await model_manager.analyze_token(sample_token, historical_data, use_ensemble=True)
        
        assert result.model_type == ModelType.ENSEMBLE
        # Should include contributions from Transformer models
        assert any("transformer" in feature.lower() for feature in result.features_used)
    
    @pytest.mark.asyncio
    async def test_transformer_training_integration(self, mock_config):
        """Test training Transformer models through ModelManager"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        # Mock training data
        training_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1H'),
            'open': [1.0] * 1000,
            'high': [1.1] * 1000,
            'low': [0.9] * 1000,
            'close': [1.0] * 1000,
            'volume': [1000] * 1000,
            'target_1h': [1.01] * 1000,
            'target_4h': [1.02] * 1000,
            'target_24h': [1.05] * 1000
        })
        
        transformer_types = [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]
        results = await model_manager.train_models(training_data, transformer_types)
        
        for model_type in transformer_types:
            assert model_type in results, f"Missing training result for {model_type}"
            assert isinstance(results[model_type], bool), f"Training result should be boolean for {model_type}"
    
    def test_transformer_health_check(self, mock_config):
        """Test health check includes Transformer models"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        status = asyncio.run(model_manager.health_check())
        
        transformer_types = ['transformer', 'itransformer', 'patchtst']
        for model_type in transformer_types:
            assert model_type in status['models'], f"Missing {model_type} in health status"
            assert 'healthy' in status['models'][model_type]
            assert 'trained' in status['models'][model_type]
    
    def test_transformer_config_validation(self, mock_config):
        """Test that Transformer configurations are properly validated"""
        # This should FAIL initially
        # Invalid config should raise error
        invalid_config = mock_config.copy()
        invalid_config['transformer']['d_model'] = 0  # Invalid dimension
        
        with pytest.raises(Exception):  # Should raise configuration error
            ModelManager(invalid_config)
    
    @pytest.mark.asyncio
    async def test_transformer_model_preservation(self, mock_config):
        """Test that Transformer models can be saved and loaded"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        # Mock that model is trained
        for model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]:
            if model_type in model_manager._models:
                model_manager._models[model_type]._is_trained = True
        
        # Save models
        save_results = {}
        for model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]:
            if model_type in model_manager._models:
                await model_manager._save_model(model_type)
                save_results[model_type] = True
        
        # Load models
        load_results = await model_manager.load_models([ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST])
        
        for model_type in save_results:
            assert load_results.get(model_type, False), f"Failed to load {model_type}"
    
    def test_transformer_feature_requirements(self, mock_config):
        """Test that Transformer models specify their required features"""
        # This should FAIL initially
        model_manager = ModelManager(mock_config)
        
        transformer_types = [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]
        for model_type in transformer_types:
            if model_type in model_manager._models:
                model = model_manager._models[model_type]
                features = model.get_required_features()
                
                assert isinstance(features, list), f"{model_type} should return list of required features"
                assert len(features) > 0, f"{model_type} should specify required features"
                # Should include sequence-based features
                assert any("sequence" in feature.lower() or "temporal" in feature.lower() for feature in features)


class TestTransformerSpecificFeatures:
    """Test Transformer-specific features in ModelManager"""
    
    @pytest.fixture
    def model_manager_with_transformers(self, mock_config):
        """ModelManager instance with Transformer support"""
        return ModelManager(mock_config)
    
    def test_attention_weight_extraction(self, model_manager_with_transformers, sample_token):
        """Test that attention weights can be extracted from Transformer models"""
        # This should FAIL initially
        # Mock a trained transformer
        if ModelType.TRANSFORMER in model_manager_with_transformers._models:
            transformer_model = model_manager_with_transformers._models[ModelType.TRANSFORMER]
            
            # Should have method to extract attention weights
            assert hasattr(transformer_model, 'get_attention_weights'), "Transformer should provide attention weights"
    
    def test_sequence_length_adaptation(self, model_manager_with_transformers):
        """Test that Transformer models can handle variable sequence lengths"""
        # This should FAIL initially
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='1H'),
            'close': [1.0] * 30,
            'volume': [1000] * 30
        })
        
        long_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='1H'),
            'close': [1.0] * 200,
            'volume': [1000] * 200
        })
        
        # Both should work (models should adapt to sequence length)
        for model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST]:
            if model_type in model_manager_with_transformers._models:
                model = model_manager_with_transformers._models[model_type]
                # Should handle both short and long sequences
                assert hasattr(model, 'prepare_sequences'), f"{model_type} should handle variable sequences"
    
    @pytest.mark.asyncio
    async def test_multi_horizon_predictions(self, model_manager_with_transformers, sample_token):
        """Test that Transformer models provide multi-horizon predictions"""
        # This should FAIL initially
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': [1.0] * 100,
            'high': [1.1] * 100,
            'low': [0.9] * 100,
            'close': [1.0] * 100,
            'volume': [1000] * 100
        })
        
        result = await model_manager_with_transformers.analyze_token(sample_token, historical_data)
        
        # Should provide predictions for multiple horizons
        assert result.price_prediction_1h is not None, "Should provide 1h prediction"
        assert result.price_prediction_4h is not None, "Should provide 4h prediction"
        assert result.price_prediction_24h is not None, "Should provide 24h prediction"


if __name__ == "__main__":
    pytest.main([__file__])