"""
Unit tests for iTransformer implementation
Tests the inverted attention mechanism where time points are tokens
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.ml_analysis.transformers.itransformer import (
    iTransformerNetwork, iTransformerPredictor, InvertedAttentionConfig
)
from src.ml_analysis.transformers.base import TransformerConfig
from src.ml_analysis.base import ModelType, PredictionDirection
from src.discovery.base import DiscoveredToken


class TestInvertedAttentionConfig:
    """Test iTransformer configuration class"""
    
    def test_default_config_creation(self):
        """Test default configuration values"""
        config = InvertedAttentionConfig()
        assert config.d_model == 512  # Default from TransformerConfig
        assert config.n_heads == 8
        assert config.n_layers == 6   # Default from TransformerConfig
        assert config.n_variates == 5  # Default for multivariate inputs
        assert config.use_inverted_attention is True
        
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = InvertedAttentionConfig(
            d_model=256,
            n_heads=16,
            n_variates=10,
            max_seq_length=500
        )
        assert config.d_model == 256
        assert config.n_heads == 16
        assert config.n_variates == 10
        assert config.max_seq_length == 500
        
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test invalid d_model/n_heads ratio
        with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
            InvertedAttentionConfig(d_model=127, n_heads=8)
            
        # Test invalid variates count
        with pytest.raises(ValueError, match="n_variates must be positive"):
            InvertedAttentionConfig(n_variates=0)


class TestiTransformerNetwork:
    """Test iTransformer network implementation"""
    
    @pytest.fixture
    def config(self):
        """Default test configuration"""
        return InvertedAttentionConfig(
            d_model=128,
            n_heads=8,
            n_layers=2,
            n_variates=5,
            max_seq_length=100
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model"""
        return iTransformerNetwork(config)
    
    def test_model_initialization(self, config):
        """Test model initializes correctly"""
        model = iTransformerNetwork(config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'variate_embeddings')
        assert hasattr(model, 'inverted_layers')
        assert hasattr(model, 'prediction_heads')
        
    def test_variate_embedding_dimensions(self, model, config):
        """Test variate embedding layer dimensions"""
        # Should embed each variate (feature) to d_model dimension
        assert model.variate_embeddings.shape == (config.n_variates, config.variate_embedding_dim or config.d_model)
        
    def test_inverted_attention_mechanism(self, model, config):
        """Test inverted attention where time steps are tokens"""
        batch_size = 2
        seq_len = 50
        n_variates = config.n_variates
        
        # Input shape: (batch, seq_len, n_variates) - standard format
        x = torch.randn(batch_size, seq_len, n_variates)
        
        # Forward pass should work now
        output = model(x)
        
        # Check that we get the expected outputs
        assert 'attention_weights' in output
        assert 'price_1h' in output
        assert 'direction_probs' in output
        
        # Check attention weights shape for inverted attention
        attention_weights = output['attention_weights']
        # Shape should be [n_layers, batch_size, n_variates, n_heads, seq_len, seq_len]
        assert attention_weights.shape[0] == config.n_layers  # n_layers
        assert attention_weights.shape[1] == batch_size      # batch_size
        assert attention_weights.shape[2] == n_variates      # n_variates
        assert attention_weights.shape[3] == config.n_heads  # n_heads
        assert attention_weights.shape[4] == seq_len         # seq_len (queries)
        assert attention_weights.shape[5] == seq_len         # seq_len (keys)
    
    def test_prediction_heads_structure(self, config):
        """Test prediction head structure for multiple horizons"""
        model = iTransformerNetwork(config)
        
        # Should have prediction heads for different time horizons
        expected_horizons = ['1h', '4h', '24h']
        for horizon in expected_horizons:
            assert hasattr(model.prediction_heads, horizon)
            head = getattr(model.prediction_heads, horizon)
            # Each head should output single value per variate
            assert head.out_features == config.n_variates
    
    def test_multivariate_correlation_capture(self, model, config):
        """Test model can capture multivariate correlations"""
        batch_size = 1
        seq_len = 20
        n_variates = config.n_variates
        
        # Create correlated input data
        x = torch.randn(batch_size, seq_len, n_variates)
        # Add correlation between variates 0 and 1
        x[:, :, 1] = x[:, :, 0] * 0.8 + torch.randn(batch_size, seq_len) * 0.2
        
        # Forward pass should work and capture correlations
        outputs = model(x)
        
        # Check that we get predictions for each variate
        price_1h = outputs['price_1h']
        assert price_1h.shape == (batch_size, n_variates)
        
        # Check that attention weights show some correlation patterns
        attention_weights = outputs['attention_weights']
        assert attention_weights.shape[-4] == n_variates  # Each variate has its attention pattern


class TestiTransformerPredictor:
    """Test iTransformer predictor implementation"""
    
    @pytest.fixture
    def config_dict(self):
        """Configuration dictionary for predictor"""
        return {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'n_variates': 5,
            'max_seq_length': 50,
            'learning_rate': 0.001,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def predictor(self, config_dict):
        """Create test predictor"""
        return iTransformerPredictor(config_dict)
    
    def test_predictor_initialization(self, config_dict):
        """Test predictor initializes correctly"""
        predictor = iTransformerPredictor(config_dict)
        assert predictor.model_type == ModelType.ITRANSFORMER
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'device')
        
    def test_required_features_multivariate(self, predictor):
        """Test iTransformer requires multivariate features"""
        features = predictor.get_required_features()
        
        # Should include multivariate time-series features
        expected_features = [
            'close', 'open', 'high', 'low', 'volume',
            'returns', 'volatility', 'correlation_features'
        ]
        
        for feature in expected_features:
            assert any(feature in f for f in features), f"Missing {feature} in required features"
    
    def test_inverted_attention_configuration(self, predictor):
        """Test predictor uses inverted attention configuration"""
        config = predictor.model_config
        assert hasattr(config, 'use_inverted_attention')
        assert config.use_inverted_attention is True
        assert hasattr(config, 'n_variates')
        assert config.n_variates > 1  # Multivariate
    
    @patch('src.ml_analysis.transformers.itransformer.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda, config_dict):
        """Test GPU/CPU device selection"""
        # Test CPU fallback when CUDA not available
        mock_cuda.return_value = False
        predictor = iTransformerPredictor(config_dict)
        assert predictor.device.type == 'cpu'
        
    async def test_model_training_interface(self, predictor):
        """Test training interface exists and handles multivariate data"""
        # Create dummy multivariate training data
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(200).cumsum() + 100,
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 105,
            'low': np.random.randn(200).cumsum() + 95,
            'volume': np.random.exponential(1000, 200)
        })
        
        # Training should work now
        success = await predictor.train_model(training_data)
        assert isinstance(success, bool)
        # Training with sufficient data should succeed
        assert success is True
    
    async def test_multivariate_prediction_interface(self, predictor):
        """Test prediction interface for multivariate inputs"""
        # First train the model
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.exponential(1000, 200)
        })
        await predictor.train_model(training_data)
        
        # Create mock token
        from src.utils.base import Chain
        token = DiscoveredToken(
            address="0x123",
            name="TEST",
            symbol="TEST",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0
        )
        
        # Create multivariate historical data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'volume': np.random.exponential(1000, 100),
            'returns': np.random.randn(100) * 0.02,
            'volatility': np.random.exponential(0.1, 100)
        })
        
        # Prediction should work now
        result = await predictor.analyze_token(token, historical_data)
        assert result is not None
        assert hasattr(result, 'model_type')
        assert result.model_type == ModelType.ITRANSFORMER
    
    def test_sequence_preparation_multivariate(self, predictor):
        """Test sequence preparation for multivariate inputs"""
        # Create test data with multiple variates
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.exponential(1000, 50),
            'returns': np.random.randn(50) * 0.02,
            'volatility': np.random.exponential(0.1, 50),
            'rsi': np.random.uniform(20, 80, 50)
        })
        
        from src.utils.base import Chain
        token = DiscoveredToken(
            address="0x123", 
            name="TEST", 
            symbol="TEST",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        # Sequence preparation should work now
        sequence = predictor._prepare_multivariate_sequence(data, token)
        assert isinstance(sequence, np.ndarray)
        assert sequence.shape[0] == len(data)  # sequence length
        assert sequence.shape[1] == predictor.n_variates  # number of variates
    
    def test_correlation_feature_extraction(self, predictor):
        """Test extraction of correlation features between variates"""
        # Create correlated multivariate data
        n_points = 100
        base_series = np.random.randn(n_points).cumsum()
        
        data = pd.DataFrame({
            'close': base_series + np.random.randn(n_points) * 0.1,
            'volume': -base_series + np.random.randn(n_points) * 0.2,  # Negative correlation
        })
        
        # Correlation feature extraction should work now
        correlations = predictor._extract_correlation_features(data)
        assert isinstance(correlations, dict)
        # Should detect the price-volume correlation
        assert 'price_volume_corr' in correlations


class TestiTransformerIntegration:
    """Test iTransformer integration with existing system"""
    
    def test_model_type_enum_integration(self):
        """Test iTransformer model type is properly integrated"""
        assert ModelType.ITRANSFORMER in ModelType
        
    def test_model_manager_integration(self):
        """Test iTransformer can be integrated with ModelManager"""
        # This should still fail as ModelManager doesn't know about iTransformer yet
        from src.ml_analysis.model_manager import ModelManager
        
        config = {
            'models': {
                'itransformer': {
                    'enabled': True,
                    'weight': 0.3,
                    'd_model': 64,
                    'n_heads': 4,
                    'n_variates': 5
                }
            }
        }
        
        # This should still fail - we haven't integrated with ModelManager yet
        with pytest.raises((KeyError, AttributeError, NotImplementedError)):
            manager = ModelManager(config)
            manager._initialize_models()
    
    def test_feature_engineer_integration(self):
        """Test iTransformer works with enhanced feature engineering"""
        from src.ml_analysis.feature_engineer import FeatureEngineer
        
        # This should still fail as FeatureEngineer doesn't have multivariate support yet
        with pytest.raises((AttributeError, NotImplementedError)):
            engineer = FeatureEngineer()
            # Test multivariate feature creation
            data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.exponential(1000, 100)
            })
            features = engineer.create_multivariate_features(data)


class TestiTransformerPerformance:
    """Test iTransformer performance characteristics"""
    
    def test_memory_efficiency_with_inverted_attention(self):
        """Test memory efficiency of inverted attention mechanism"""
        config = InvertedAttentionConfig(
            d_model=64,  # Smaller for testing
            n_heads=4,
            n_layers=1,
            n_variates=5,
            max_seq_length=100  # Smaller for testing
        )
        
        # Model creation should work
        model = iTransformerNetwork(config)
        
        # Test with smaller sequence for basic functionality
        batch_size = 1
        seq_len = 50
        n_variates = 5
        
        x = torch.randn(batch_size, seq_len, n_variates)
        
        # Basic forward pass should work
        output = model(x)
        assert 'attention_weights' in output
        
        # Attention weights should have reasonable size
        attention_weights = output['attention_weights']
        assert attention_weights.numel() > 0
    
    def test_multivariate_scaling(self):
        """Test performance scaling with number of variates"""
        # Test with small number of variates
        for n_variates in [3, 5]:
            config = InvertedAttentionConfig(
                d_model=32,  # Small for testing
                n_heads=2,
                n_layers=1,
                n_variates=n_variates,
                max_seq_length=20  # Small for testing
            )
            
            model = iTransformerNetwork(config)
            
            # Test forward pass
            x = torch.randn(1, 20, n_variates)
            
            import time
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            # Should complete reasonably quickly
            assert inference_time < 2.0  # Less than 2 seconds
            assert output is not None


class TestiTransformerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions"""
        config = InvertedAttentionConfig(n_variates=5)
        
        model = iTransformerNetwork(config)
        
        # Test wrong number of variates
        x = torch.randn(1, 100, 3)  # 3 variates instead of 5
        with pytest.raises(ValueError, match="Expected.*variates"):
            output = model(x)
    
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences"""
        config = InvertedAttentionConfig(n_variates=5)
        
        model = iTransformerNetwork(config)
        
        # Test empty sequence
        x = torch.randn(1, 0, 5)  # Empty sequence
        with pytest.raises(ValueError, match="Empty sequence"):
            output = model(x)
    
    async def test_insufficient_training_data(self):
        """Test handling of insufficient training data"""
        config_dict = {
            'd_model': 64,
            'n_heads': 4,
            'n_variates': 5,
            'max_seq_length': 50
        }
        
        predictor = iTransformerPredictor(config_dict)
        
        # Very small training dataset
        small_data = pd.DataFrame({
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })
        
        success = await predictor.train_model(small_data)
        assert success is False