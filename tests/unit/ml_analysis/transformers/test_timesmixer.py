"""
Unit tests for TimesMixer implementation
Tests the decomposable mixing approach for time-series forecasting
Following TDD methodology - these tests should FAIL initially
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch

# Import only the specific components we need to avoid config issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

# Import directly to avoid config loading issues - THESE WILL FAIL INITIALLY
from src.ml_analysis.transformers.timesmixer import (
    TimesMixerNetwork, TimesMixerPredictor, TimesMixerConfig,
    PastDecomposableMixing, FutureMultipredictorMixing
)
from src.ml_analysis.base import ModelType, PredictionDirection
from src.discovery.base import DiscoveredToken


class TestTimesMixerConfig:
    """Test TimesMixer configuration class"""
    
    def test_default_config_creation(self):
        """Test default configuration values"""
        config = TimesMixerConfig()
        assert config.d_model == 128  # Smaller default for efficiency
        assert config.n_heads == 8
        assert config.n_layers == 3   # Fewer layers for mixing approach
        assert config.seq_len == 96    # Standard lookback window
        assert config.pred_len == 24   # Standard prediction horizon
        assert config.d_ff == 256      # Smaller feed-forward dimension
        assert config.top_k == 5       # Top-k mixing components
        assert config.num_kernels == 6 # Number of decomposition kernels
        assert config.use_time_mixing is True
        assert config.use_feature_mixing is True
        
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = TimesMixerConfig(
            d_model=256,
            n_heads=16,
            seq_len=168,
            pred_len=48,
            top_k=10,
            num_kernels=12
        )
        assert config.d_model == 256
        assert config.n_heads == 16
        assert config.seq_len == 168
        assert config.pred_len == 48
        assert config.top_k == 10
        assert config.num_kernels == 12
        
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test invalid d_model/n_heads ratio
        with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
            TimesMixerConfig(d_model=127, n_heads=8)
            
        # Test invalid sequence lengths
        with pytest.raises(ValueError, match="seq_len must be positive"):
            TimesMixerConfig(seq_len=0)
            
        # Test invalid prediction length
        with pytest.raises(ValueError, match="pred_len must be positive"):
            TimesMixerConfig(pred_len=0)
            
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            TimesMixerConfig(top_k=0)
    
    def test_mixing_configuration(self):
        """Test mixing-specific configuration options"""
        config = TimesMixerConfig(
            use_time_mixing=False,
            use_feature_mixing=True,
            decomposition_kernel='moving_avg'
        )
        assert config.use_time_mixing is False
        assert config.use_feature_mixing is True
        assert config.decomposition_kernel == 'moving_avg'
        
    def test_memory_estimation(self):
        """Test memory usage estimation for mixing approach"""
        config = TimesMixerConfig(
            seq_len=168,
            pred_len=24,
            d_model=128,
            top_k=5
        )
        memory_mb = config.estimate_memory_usage_mb()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
        # TimesMixer should be more memory efficient than full attention
        assert memory_mb < 500  # Should be reasonable for production


class TestPastDecomposableMixing:
    """Test Past Decomposable Mixing (PDM) component"""
    
    @pytest.fixture
    def pdm_config(self):
        """PDM test configuration"""
        return {
            'seq_len': 96,
            'd_model': 128,
            'top_k': 5,
            'num_kernels': 6
        }
    
    @pytest.fixture
    def pdm_layer(self, pdm_config):
        """Create PDM test layer"""
        return PastDecomposableMixing(**pdm_config)
    
    def test_pdm_initialization(self, pdm_config):
        """Test PDM layer initializes correctly"""
        pdm = PastDecomposableMixing(**pdm_config)
        assert isinstance(pdm, nn.Module)
        assert hasattr(pdm, 'decomposition')
        assert hasattr(pdm, 'time_mixing')
        assert hasattr(pdm, 'feature_mixing')
        
    def test_seasonal_trend_decomposition(self, pdm_layer, pdm_config):
        """Test seasonal-trend decomposition functionality"""
        batch_size = 2
        seq_len = pdm_config['seq_len']
        d_model = pdm_config['d_model']
        
        # Input with clear seasonal pattern
        x = torch.randn(batch_size, seq_len, d_model)
        # Add seasonal component
        seasonal = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
        seasonal = seasonal.expand(batch_size, seq_len, d_model)
        x = x + seasonal
        
        # PDM should decompose into seasonal and trend components
        seasonal_out, trend_out = pdm_layer.decompose(x)
        
        assert seasonal_out.shape == (batch_size, seq_len, d_model)
        assert trend_out.shape == (batch_size, seq_len, d_model)
        
        # Seasonal component should capture periodicity
        assert not torch.allclose(seasonal_out, trend_out, atol=1e-3)
    
    def test_time_mixing_mechanism(self, pdm_layer, pdm_config):
        """Test time domain mixing in PDM"""
        batch_size = 2
        seq_len = pdm_config['seq_len']
        d_model = pdm_config['d_model']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Time mixing should preserve temporal structure
        mixed_output = pdm_layer.time_mixing(x)
        
        assert mixed_output.shape == (batch_size, seq_len, d_model)
        # Output should be different from input (mixing occurred)
        assert not torch.allclose(mixed_output, x, atol=1e-3)
    
    def test_feature_mixing_mechanism(self, pdm_layer, pdm_config):
        """Test feature domain mixing in PDM"""
        batch_size = 2
        seq_len = pdm_config['seq_len']
        d_model = pdm_config['d_model']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Feature mixing should mix across feature dimension
        mixed_output = pdm_layer.feature_mixing(x)
        
        assert mixed_output.shape == (batch_size, seq_len, d_model)
        # Output should be different from input (mixing occurred)
        assert not torch.allclose(mixed_output, x, atol=1e-3)
    
    def test_pdm_forward_pass(self, pdm_layer, pdm_config):
        """Test complete PDM forward pass"""
        batch_size = 2
        seq_len = pdm_config['seq_len']
        d_model = pdm_config['d_model']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pdm_layer(x)
        
        # Should output processed representations
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output, x, atol=1e-3)


class TestFutureMultipredictorMixing:
    """Test Future Multipredictor Mixing (FMM) component"""
    
    @pytest.fixture
    def fmm_config(self):
        """FMM test configuration"""
        return {
            'seq_len': 96,
            'pred_len': 24,
            'd_model': 128,
            'top_k': 5
        }
    
    @pytest.fixture
    def fmm_layer(self, fmm_config):
        """Create FMM test layer"""
        return FutureMultipredictorMixing(**fmm_config)
    
    def test_fmm_initialization(self, fmm_config):
        """Test FMM layer initializes correctly"""
        fmm = FutureMultipredictorMixing(**fmm_config)
        assert isinstance(fmm, nn.Module)
        assert hasattr(fmm, 'multi_predictors')
        assert hasattr(fmm, 'mixing_weights')
        assert hasattr(fmm, 'output_projection')
        
    def test_multi_predictor_generation(self, fmm_layer, fmm_config):
        """Test multiple predictor generation"""
        batch_size = 2
        seq_len = fmm_config['seq_len']
        pred_len = fmm_config['pred_len']
        d_model = fmm_config['d_model']
        top_k = fmm_config['top_k']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Should generate multiple predictions
        multi_preds = fmm_layer.generate_multi_predictions(x)
        
        assert multi_preds.shape == (batch_size, top_k, pred_len, d_model)
        
        # Different predictors should generate different outputs
        for i in range(top_k - 1):
            assert not torch.allclose(multi_preds[:, i], multi_preds[:, i+1], atol=1e-3)
    
    def test_adaptive_mixing_weights(self, fmm_layer, fmm_config):
        """Test adaptive mixing weight computation"""
        batch_size = 2
        seq_len = fmm_config['seq_len']
        d_model = fmm_config['d_model']
        top_k = fmm_config['top_k']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Should compute adaptive weights
        weights = fmm_layer.compute_mixing_weights(x)
        
        assert weights.shape == (batch_size, top_k)
        # Weights should sum to 1 (softmax normalization)
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        # All weights should be positive
        assert (weights >= 0).all()
    
    def test_fmm_forward_pass(self, fmm_layer, fmm_config):
        """Test complete FMM forward pass"""
        batch_size = 2
        seq_len = fmm_config['seq_len']
        pred_len = fmm_config['pred_len']
        d_model = fmm_config['d_model']
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = fmm_layer(x)
        
        # Should output future predictions
        assert output.shape == (batch_size, pred_len, d_model)


class TestTimesMixerNetwork:
    """Test TimesMixer network implementation"""
    
    @pytest.fixture
    def config(self):
        """Default test configuration"""
        return TimesMixerConfig(
            d_model=128,
            n_heads=8,
            n_layers=2,
            seq_len=96,
            pred_len=24,
            top_k=5,
            num_kernels=6
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model"""
        return TimesMixerNetwork(config)
    
    def test_model_initialization(self, config):
        """Test model initializes correctly"""
        model = TimesMixerNetwork(config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'pdm_layers')
        assert hasattr(model, 'fmm_layer')
        assert hasattr(model, 'prediction_heads')
        assert len(model.pdm_layers) == config.n_layers
        
    def test_input_embedding_and_projection(self, model, config):
        """Test input embedding and projection layers"""
        batch_size = 2
        seq_len = config.seq_len
        n_features = 7  # Standard financial features
        
        # Input shape: (batch, seq_len, n_features)
        x = torch.randn(batch_size, seq_len, n_features)
        
        # Model should handle input projection to d_model
        embedded = model.input_embedding(x)
        assert embedded.shape == (batch_size, seq_len, config.d_model)
        
    def test_decomposable_mixing_forward(self, model, config):
        """Test decomposable mixing in forward pass"""
        batch_size = 2
        seq_len = config.seq_len
        n_features = 7
        
        x = torch.randn(batch_size, seq_len, n_features)
        
        # Forward pass should work
        output = model(x)
        
        # Check that we get the expected outputs
        assert 'predictions' in output
        assert 'mixing_weights' in output
        assert 'seasonal_components' in output
        assert 'trend_components' in output
        
        # Check output shapes
        predictions = output['predictions']
        assert predictions.shape == (batch_size, config.pred_len, n_features)
        
    def test_multi_scale_pattern_recognition(self, model, config):
        """Test multi-scale pattern recognition capability"""
        batch_size = 1
        seq_len = config.seq_len
        n_features = 7
        
        # Create input with multiple time scales
        x = torch.zeros(batch_size, seq_len, n_features)
        # Short-term pattern (high frequency)
        short_term = torch.sin(torch.linspace(0, 8*np.pi, seq_len))
        # Long-term pattern (low frequency)
        long_term = torch.sin(torch.linspace(0, 2*np.pi, seq_len))
        
        x[0, :, 0] = short_term + long_term
        x[0, :, 1] = long_term  # Only long-term
        x[0, :, 2] = short_term  # Only short-term
        
        output = model(x)
        
        # Model should capture different scale patterns
        seasonal = output['seasonal_components']
        trend = output['trend_components']
        
        assert seasonal.shape == (batch_size, seq_len, n_features)
        assert trend.shape == (batch_size, seq_len, n_features)
        
        # Seasonal and trend should be different
        assert not torch.allclose(seasonal, trend, atol=1e-3)
    
    def test_time_and_feature_mixing_separation(self, model, config):
        """Test separation of time and feature mixing"""
        batch_size = 2
        seq_len = config.seq_len
        n_features = 7
        
        x = torch.randn(batch_size, seq_len, n_features)
        
        # Test with only time mixing enabled
        model.config.use_feature_mixing = False
        output_time_only = model(x)
        
        # Test with only feature mixing enabled
        model.config.use_time_mixing = False
        model.config.use_feature_mixing = True
        output_feature_only = model(x)
        
        # Outputs should be different when different mixing is used
        assert not torch.allclose(
            output_time_only['predictions'], 
            output_feature_only['predictions'], 
            atol=1e-3
        )
    
    def test_multi_horizon_predictions(self, model, config):
        """Test multi-horizon prediction capability"""
        batch_size = 2
        seq_len = config.seq_len
        n_features = 7
        
        x = torch.randn(batch_size, seq_len, n_features)
        output = model(x)
        
        # Should have predictions for different horizons
        predictions = output['predictions']
        assert predictions.shape == (batch_size, config.pred_len, n_features)
        
        # Should have mixing weights showing contribution of different predictors
        mixing_weights = output['mixing_weights']
        assert mixing_weights.shape == (batch_size, config.top_k)
        
        # Weights should sum to 1
        assert torch.allclose(mixing_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)


class TestTimesMixerPredictor:
    """Test TimesMixer predictor implementation"""
    
    @pytest.fixture
    def config_dict(self):
        """Configuration dictionary for predictor"""
        return {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': 96,
            'pred_len': 24,
            'top_k': 5,
            'num_kernels': 6,
            'learning_rate': 0.001,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def predictor(self, config_dict):
        """Create test predictor"""
        return TimesMixerPredictor(config_dict)
    
    def test_predictor_initialization(self, config_dict):
        """Test predictor initializes correctly"""
        predictor = TimesMixerPredictor(config_dict)
        assert predictor.model_type == ModelType.TIMESMIXER
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'device')
        assert hasattr(predictor, 'seq_len')
        assert hasattr(predictor, 'pred_len')
        
    def test_required_features_for_mixing(self, predictor):
        """Test required features for decomposable mixing"""
        features = predictor.get_required_features()
        
        # Should include features suitable for decomposable mixing
        expected_features = [
            'close', 'open', 'high', 'low', 'volume',
            'returns', 'volatility', 'seasonal_features'
        ]
        
        for feature in expected_features:
            assert any(feature in f for f in features), f"Missing {feature} in required features"
    
    def test_mixing_based_configuration(self, predictor):
        """Test predictor uses mixing-based configuration"""
        config = predictor.model_config
        assert hasattr(config, 'top_k')
        assert hasattr(config, 'num_kernels')
        assert hasattr(config, 'use_time_mixing')
        assert hasattr(config, 'use_feature_mixing')
        assert config.top_k > 0
        assert config.num_kernels > 0
    
    @patch('src.ml_analysis.transformers.timesmixer.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda, config_dict):
        """Test GPU/CPU device selection"""
        mock_cuda.return_value = False
        predictor = TimesMixerPredictor(config_dict)
        assert predictor.device.type == 'cpu'
    
    async def test_model_training_with_mixing(self, predictor):
        """Test training interface handles decomposable mixing"""
        # Create dummy training data suitable for mixing
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(200).cumsum() + 100,
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 105,
            'low': np.random.randn(200).cumsum() + 95,
            'volume': np.random.exponential(1000, 200),
            'returns': np.random.randn(200) * 0.02,
            'volatility': np.random.exponential(0.1, 200)
        })
        
        success = await predictor.train_model(training_data)
        assert isinstance(success, bool)
        assert success is True  # Should succeed with sufficient data
    
    async def test_decomposable_mixing_prediction(self, predictor):
        """Test prediction interface for decomposable mixing"""
        # First train the model
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.exponential(1000, 200),
            'returns': np.random.randn(200) * 0.02,
            'volatility': np.random.exponential(0.1, 200)
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
        
        # Create historical data suitable for mixing
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
        
        result = await predictor.analyze_token(token, historical_data)
        assert result is not None
        assert hasattr(result, 'model_type')
        assert result.model_type == ModelType.TIMESMIXER
    
    def test_sequence_to_mixing_preparation(self, predictor):
        """Test conversion of sequences for mixing approach"""
        # Create test data with seasonal patterns
        dates = pd.date_range('2024-01-01', periods=168, freq='H')  # 1 week
        data = pd.DataFrame({
            'close': np.sin(np.linspace(0, 4*np.pi, 168)) * 10 + 100,  # Seasonal
            'volume': np.random.exponential(1000, 168),
            'returns': np.random.randn(168) * 0.02
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
        
        # Test mixing sequence preparation
        sequence = predictor._prepare_mixing_sequence(data, token)
        assert isinstance(sequence, np.ndarray)
        
        # Should have dimensions compatible with mixing processing
        seq_len, n_features = sequence.shape
        assert seq_len == predictor.model_config.seq_len
        assert n_features >= 3  # At least close, volume, returns
    
    def test_seasonal_trend_decomposition_processing(self, predictor):
        """Test seasonal-trend decomposition in sequence processing"""
        # Create data with clear seasonal and trend components
        n_points = 168
        trend = np.linspace(100, 120, n_points)  # Linear trend
        seasonal = np.sin(np.linspace(0, 4*np.pi, n_points)) * 5  # Seasonal pattern
        noise = np.random.randn(n_points) * 1
        
        data = pd.DataFrame({
            'close': trend + seasonal + noise,
            'volume': np.random.exponential(1000, n_points),
            'returns': np.random.randn(n_points) * 0.02
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
        
        sequence = predictor._prepare_mixing_sequence(data, token)
        
        # Should preserve both trend and seasonal patterns
        assert sequence.shape[0] == predictor.model_config.seq_len
        
        # Check that the sequence contains meaningful patterns
        close_values = sequence[:, 0] if sequence.shape[1] > 0 else sequence[:, 0]
        assert np.std(close_values) > 0  # Should have variation


class TestTimesMixerIntegration:
    """Test TimesMixer integration with existing system"""
    
    def test_model_type_enum_integration(self):
        """Test TimesMixer model type is properly integrated"""
        assert ModelType.TIMESMIXER in ModelType
        
    def test_model_manager_integration(self):
        """Test TimesMixer can be integrated with ModelManager"""
        # Skip this test to avoid config issues for now
        # This will be implemented when ModelManager is updated
        pytest.skip("ModelManager integration test skipped due to config requirements")


class TestTimesMixerPerformance:
    """Test TimesMixer performance characteristics"""
    
    def test_memory_efficiency_with_mixing(self):
        """Test memory efficiency of decomposable mixing approach"""
        config = TimesMixerConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=96,
            pred_len=24,
            top_k=5
        )
        
        model = TimesMixerNetwork(config)
        
        batch_size = 1
        seq_len = config.seq_len
        n_features = 7
        
        x = torch.randn(batch_size, seq_len, n_features)
        
        # Forward pass should be memory efficient
        output = model(x)
        assert output is not None
        
        # Should handle longer sequences efficiently
        long_config = TimesMixerConfig(seq_len=168, pred_len=48)
        long_model = TimesMixerNetwork(long_config)
        long_x = torch.randn(batch_size, 168, n_features)
        long_output = long_model(long_x)
        assert long_output is not None
    
    def test_mixing_component_scaling(self):
        """Test performance scaling with different mixing parameters"""
        for top_k in [3, 5, 7]:
            config = TimesMixerConfig(
                d_model=32,
                n_heads=2,
                n_layers=1,
                seq_len=48,
                pred_len=12,
                top_k=top_k
            )
            
            model = TimesMixerNetwork(config)
            
            x = torch.randn(1, 48, 5)
            
            import time
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            inference_time = end_time - start_time
            assert inference_time < 2.0  # Should be reasonable
            assert output is not None


class TestTimesMixerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_sequence_length_compatibility(self):
        """Test handling of sequence lengths for decomposition"""
        config = TimesMixerConfig(
            seq_len=24,
            pred_len=6,
            num_kernels=6
        )
        
        model = TimesMixerNetwork(config)
        
        # Sequence too short for effective decomposition
        x_short = torch.randn(1, 12, 5)  # Shorter than seq_len
        with pytest.raises(ValueError, match="Sequence length.*too short"):
            output = model(x_short)
    
    def test_prediction_length_validation(self):
        """Test handling of prediction length validation"""
        config = TimesMixerConfig(seq_len=96, pred_len=0)
        
        with pytest.raises(ValueError, match="pred_len must be positive"):
            model = TimesMixerNetwork(config)
    
    async def test_insufficient_training_data_for_decomposition(self):
        """Test handling of insufficient training data for decomposition"""
        config_dict = {
            'd_model': 64,
            'n_heads': 4,
            'seq_len': 96,
            'pred_len': 24,
            'top_k': 5
        }
        
        predictor = TimesMixerPredictor(config_dict)
        
        # Very small training dataset - insufficient for decomposition
        small_data = pd.DataFrame({
            'close': [1, 2, 3, 4, 5],  # Only 5 points, need at least seq_len
            'volume': [100, 200, 300, 400, 500]
        })
        
        success = await predictor.train_model(small_data)
        assert success is False  # Should fail due to insufficient data


class TestTimesMixerDecomposition:
    """Test TimesMixer's decomposition capabilities"""
    
    def test_seasonal_trend_separation(self):
        """Test seasonal-trend decomposition accuracy"""
        config = TimesMixerConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=96,
            pred_len=24
        )
        
        model = TimesMixerNetwork(config)
        
        # Create synthetic data with known seasonal and trend components
        seq_len = config.seq_len
        trend = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
        seasonal = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
        combined = trend + seasonal
        
        x = combined.expand(1, seq_len, 5)  # Expand to 5 features
        output = model(x)
        
        # Should decompose into seasonal and trend components
        seasonal_out = output['seasonal_components']
        trend_out = output['trend_components']
        
        assert seasonal_out.shape == (1, seq_len, 5)
        assert trend_out.shape == (1, seq_len, 5)
        
        # Components should be different
        assert not torch.allclose(seasonal_out, trend_out, atol=1e-3)
    
    def test_multi_predictor_diversity(self):
        """Test diversity of multiple predictors in FMM"""
        config = TimesMixerConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            seq_len=48,
            pred_len=12,
            top_k=5
        )
        
        model = TimesMixerNetwork(config)
        
        x = torch.randn(1, 48, 3)
        output = model(x)
        
        # Should have mixing weights that show different predictor contributions
        mixing_weights = output['mixing_weights']
        assert mixing_weights.shape == (1, config.top_k)
        
        # Not all predictors should have equal weight (diversity)
        weights = mixing_weights[0].detach().numpy()
        assert not np.allclose(weights, weights[0], atol=1e-2)