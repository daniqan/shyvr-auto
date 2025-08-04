"""
Unit tests for PatchTST implementation
Tests the patch-based tokenization approach for long time-series sequences
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

# Import directly to avoid config loading issues
from src.ml_analysis.transformers.patchtst import (
    PatchTSTNetwork, PatchTSTPredictor, PatchTSTConfig
)
from src.ml_analysis.base import ModelType, PredictionDirection
from src.discovery.base import DiscoveredToken


class TestPatchTSTConfig:
    """Test PatchTST configuration class"""
    
    def test_default_config_creation(self):
        """Test default configuration values"""
        config = PatchTSTConfig()
        assert config.d_model == 128  # Smaller default for efficiency
        assert config.n_heads == 8
        assert config.n_layers == 3   # Fewer layers than standard transformer
        assert config.patch_length == 16  # Patch size for tokenization
        assert config.stride == 8     # Overlap between patches
        assert config.n_channels == 1  # Univariate by default
        assert config.channel_independence is True
        
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = PatchTSTConfig(
            d_model=256,
            n_heads=16,
            patch_length=32,
            stride=16,
            n_channels=5,
            max_seq_length=1000
        )
        assert config.d_model == 256
        assert config.n_heads == 16
        assert config.patch_length == 32
        assert config.stride == 16
        assert config.n_channels == 5
        assert config.max_seq_length == 1000
        
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test patch_length > stride constraint
        with pytest.raises(ValueError, match="patch_length must be greater than stride"):
            PatchTSTConfig(patch_length=8, stride=16)
            
        # Test d_model divisibility by n_heads
        with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
            PatchTSTConfig(d_model=127, n_heads=8)
            
        # Test positive channels
        with pytest.raises(ValueError, match="n_channels must be positive"):
            PatchTSTConfig(n_channels=0)
    
    def test_patch_calculation(self):
        """Test patch count calculation"""
        config = PatchTSTConfig(
            max_seq_length=100,
            patch_length=16,
            stride=8
        )
        expected_patches = (100 - 16) // 8 + 1  # Standard sliding window calculation
        assert config.get_n_patches() == expected_patches
        
    def test_memory_estimation(self):
        """Test memory usage estimation for patch-based approach"""
        config = PatchTSTConfig(
            max_seq_length=1000,
            patch_length=16,
            n_channels=5
        )
        memory_mb = config.estimate_memory_usage_mb()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
        # PatchTST should be more memory efficient than full attention
        assert memory_mb < 1000  # Should be reasonable for production


class TestPatchTSTNetwork:
    """Test PatchTST network implementation"""
    
    @pytest.fixture
    def config(self):
        """Default test configuration"""
        return PatchTSTConfig(
            d_model=128,
            n_heads=8,
            n_layers=2,
            patch_length=16,
            stride=8,
            n_channels=3,
            max_seq_length=100
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model"""
        return PatchTSTNetwork(config)
    
    def test_model_initialization(self, config):
        """Test model initializes correctly"""
        model = PatchTSTNetwork(config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'patch_embedding')
        assert hasattr(model, 'channel_embeddings')
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'prediction_heads')
        
    def test_patch_embedding_dimensions(self, model, config):
        """Test patch embedding layer dimensions"""
        # Should embed patches of length patch_length to d_model dimension
        assert model.patch_embedding.in_features == config.patch_length
        assert model.patch_embedding.out_features == config.d_model
        
    def test_channel_independence_structure(self, model, config):
        """Test channel independence implementation"""
        # Should have separate processing for each channel
        if config.channel_independence:
            assert hasattr(model, 'channel_embeddings')
            assert model.channel_embeddings.num_embeddings == config.n_channels
            
    def test_patch_tokenization_forward(self, model, config):
        """Test patch-based tokenization in forward pass"""
        batch_size = 2
        seq_len = 80  # Must be compatible with patch_length and stride
        n_channels = config.n_channels
        
        # Input shape: (batch, seq_len, n_channels)
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Forward pass should work
        output = model(x)
        
        # Check that we get the expected outputs
        assert 'price_1h' in output
        assert 'price_4h' in output
        assert 'price_24h' in output
        assert 'confidence' in output
        assert 'direction_probs' in output
        
        # Check output shapes
        for horizon in ['1h', '4h', '24h']:
            pred = output[f'price_{horizon}']
            if config.channel_independence:
                # Independent predictions per channel
                assert pred.shape == (batch_size, n_channels)
            else:
                # Single prediction aggregated across channels
                assert pred.shape == (batch_size, 1)
    
    def test_patch_creation_and_processing(self, model, config):
        """Test patch creation from time series"""
        batch_size = 1
        seq_len = 64  # Evenly divisible by patch parameters
        n_channels = config.n_channels
        
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Calculate expected number of patches
        expected_n_patches = (seq_len - config.patch_length) // config.stride + 1
        
        # Model should handle patch creation internally
        output = model(x)
        
        # Check that model processes correct number of patches
        # This is implicit - the model should handle patch dimension correctly
        assert output is not None
        assert 'price_1h' in output
        
    def test_channel_independence_mechanism(self, config):
        """Test that channels are processed independently"""
        config_independent = PatchTSTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            patch_length=8,
            stride=4,
            n_channels=3,
            channel_independence=True
        )
        
        model = PatchTSTNetwork(config_independent)
        
        batch_size = 1
        seq_len = 32
        n_channels = 3
        
        # Create input with distinct patterns per channel
        x = torch.zeros(batch_size, seq_len, n_channels)
        x[:, :, 0] = torch.linspace(0, 1, seq_len)  # Linear trend
        x[:, :, 1] = torch.sin(torch.linspace(0, 4*np.pi, seq_len))  # Sinusoidal
        x[:, :, 2] = torch.randn(seq_len)  # Random noise
        
        output = model(x)
        
        # With channel independence, should get different predictions per channel
        price_1h = output['price_1h']
        assert price_1h.shape == (batch_size, n_channels)
        
        # Predictions should be different for different channels
        channel_preds = price_1h[0, :].detach().numpy()
        assert not np.allclose(channel_preds, channel_preds[0], atol=1e-3)
    
    def test_long_sequence_efficiency(self, config):
        """Test efficiency with long sequences via patching"""
        # Test with longer sequence that would be problematic for full attention
        long_config = PatchTSTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            patch_length=16,
            stride=8,
            n_channels=2,
            max_seq_length=512  # Long sequence
        )
        
        model = PatchTSTNetwork(long_config)
        
        batch_size = 1
        seq_len = 256  # Reasonable test length
        n_channels = 2
        
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Should handle long sequences efficiently
        import time
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 1.0  # Should be fast due to patching
        assert output is not None


class TestPatchTSTPredictor:
    """Test PatchTST predictor implementation"""
    
    @pytest.fixture
    def config_dict(self):
        """Configuration dictionary for predictor"""
        return {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'patch_length': 16,
            'stride': 8,
            'n_channels': 3,
            'max_seq_length': 100,
            'learning_rate': 0.001,
            'dropout': 0.1,
            'channel_independence': True
        }
    
    @pytest.fixture
    def predictor(self, config_dict):
        """Create test predictor"""
        return PatchTSTPredictor(config_dict)
    
    def test_predictor_initialization(self, config_dict):
        """Test predictor initializes correctly"""
        predictor = PatchTSTPredictor(config_dict)
        assert predictor.model_type == ModelType.PATCHTST
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'device')
        assert hasattr(predictor, 'patch_length')
        assert hasattr(predictor, 'stride')
        
    def test_required_features_for_patching(self, predictor):
        """Test required features for patch-based processing"""
        features = predictor.get_required_features()
        
        # Should include features suitable for patching
        expected_features = [
            'close', 'open', 'high', 'low', 'volume',
            'returns', 'patch_sequence_data'
        ]
        
        for feature in expected_features:
            assert any(feature in f for f in features), f"Missing {feature} in required features"
    
    def test_patch_based_configuration(self, predictor):
        """Test predictor uses patch-based configuration"""
        config = predictor.model_config
        assert hasattr(config, 'patch_length')
        assert hasattr(config, 'stride')
        assert hasattr(config, 'channel_independence')
        assert config.patch_length > 0
        assert config.stride > 0
        assert config.stride <= config.patch_length
    
    @patch('src.ml_analysis.transformers.patchtst.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda, config_dict):
        """Test GPU/CPU device selection"""
        mock_cuda.return_value = False
        predictor = PatchTSTPredictor(config_dict)
        assert predictor.device.type == 'cpu'
    
    async def test_model_training_with_patches(self, predictor):
        """Test training interface handles patch-based processing"""
        # Create dummy training data suitable for patching
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(200).cumsum() + 100,
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 105,
            'low': np.random.randn(200).cumsum() + 95,
            'volume': np.random.exponential(1000, 200),
            'returns': np.random.randn(200) * 0.02
        })
        
        success = await predictor.train_model(training_data)
        assert isinstance(success, bool)
        assert success is True  # Should succeed with sufficient data
    
    async def test_patch_based_prediction(self, predictor):
        """Test prediction interface for patch-based processing"""
        # First train the model
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        training_data = pd.DataFrame({
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.exponential(1000, 200),
            'returns': np.random.randn(200) * 0.02
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
        
        # Create historical data suitable for patching
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'volume': np.random.exponential(1000, 100),
            'returns': np.random.randn(100) * 0.02
        })
        
        result = await predictor.analyze_token(token, historical_data)
        assert result is not None
        assert hasattr(result, 'model_type')
        assert result.model_type == ModelType.PATCHTST
    
    def test_sequence_to_patches_conversion(self, predictor):
        """Test conversion of sequences to patches"""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=64, freq='H')
        data = pd.DataFrame({
            'close': np.random.randn(64).cumsum() + 100,
            'volume': np.random.exponential(1000, 64),
            'returns': np.random.randn(64) * 0.02
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
        
        # Test patch sequence preparation
        sequence = predictor._prepare_patch_sequence(data, token)
        assert isinstance(sequence, np.ndarray)
        
        # Should have dimensions compatible with patch processing
        seq_len, n_channels = sequence.shape
        assert seq_len == len(data)
        assert n_channels == predictor.model_config.n_channels
    
    def test_channel_independence_processing(self, predictor):
        """Test channel independence in sequence processing"""
        # Create data with distinct channel patterns
        n_points = 80
        data = pd.DataFrame({
            'close': np.linspace(100, 200, n_points),  # Linear trend
            'volume': np.sin(np.linspace(0, 4*np.pi, n_points)) * 1000 + 1000,  # Sinusoidal
            'returns': np.random.randn(n_points) * 0.02  # Random
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
        
        sequence = predictor._prepare_patch_sequence(data, token)
        
        # Should preserve distinct patterns in different channels
        assert sequence.shape[1] >= 3  # At least 3 channels
        
        # Check that channels have different patterns
        channel_stds = np.std(sequence, axis=0)
        assert len(np.unique(channel_stds.round(3))) > 1  # Different patterns


class TestPatchTSTIntegration:
    """Test PatchTST integration with existing system"""
    
    def test_model_type_enum_integration(self):
        """Test PatchTST model type is properly integrated"""
        assert ModelType.PATCHTST in ModelType
        
    def test_model_manager_integration(self):
        """Test PatchTST can be integrated with ModelManager"""
        # Skip this test to avoid config issues for now
        # This will be implemented when ModelManager is updated
        pytest.skip("ModelManager integration test skipped due to config requirements")


class TestPatchTSTPerformance:
    """Test PatchTST performance characteristics"""
    
    def test_memory_efficiency_with_patching(self):
        """Test memory efficiency of patch-based approach"""
        config = PatchTSTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            patch_length=16,
            stride=8,
            n_channels=3,
            max_seq_length=256
        )
        
        model = PatchTSTNetwork(config)
        
        batch_size = 1
        seq_len = 128  # Reasonable sequence length
        n_channels = 3
        
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Forward pass should be memory efficient
        output = model(x)
        assert output is not None
        
        # Should handle longer sequences efficiently
        long_x = torch.randn(batch_size, 256, n_channels)
        long_output = model(long_x)
        assert long_output is not None
    
    def test_patch_length_scaling(self):
        """Test performance scaling with different patch lengths"""
        for patch_length in [8, 16, 32]:
            config = PatchTSTConfig(
                d_model=32,
                n_heads=2,
                n_layers=1,
                patch_length=patch_length,
                stride=patch_length // 2,
                n_channels=2,
                max_seq_length=128
            )
            
            model = PatchTSTNetwork(config)
            
            x = torch.randn(1, 64, 2)
            
            import time
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            inference_time = end_time - start_time
            assert inference_time < 2.0  # Should be reasonable
            assert output is not None


class TestPatchTSTEdgeCases:
    """Test edge cases and error handling"""
    
    def test_sequence_length_compatibility(self):
        """Test handling of sequence lengths incompatible with patch parameters"""
        config = PatchTSTConfig(
            patch_length=16,
            stride=8,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        
        # Sequence too short for even one patch
        x_short = torch.randn(1, 10, 2)  # Shorter than patch_length
        with pytest.raises(ValueError, match="Sequence length.*too short"):
            output = model(x_short)
    
    def test_channel_mismatch_handling(self):
        """Test handling of channel count mismatches"""
        config = PatchTSTConfig(n_channels=3)
        model = PatchTSTNetwork(config)
        
        # Wrong number of channels
        x_wrong_channels = torch.randn(1, 64, 5)  # 5 channels instead of 3
        with pytest.raises(ValueError, match="Expected.*channels"):
            output = model(x_wrong_channels)
    
    async def test_insufficient_training_data_for_patches(self):
        """Test handling of insufficient training data for patch creation"""
        config_dict = {
            'd_model': 64,
            'n_heads': 4,
            'patch_length': 16,
            'stride': 8,
            'n_channels': 2,
            'max_seq_length': 50
        }
        
        predictor = PatchTSTPredictor(config_dict)
        
        # Very small training dataset - insufficient for patches
        small_data = pd.DataFrame({
            'close': [1, 2, 3, 4, 5],  # Only 5 points, need at least patch_length
            'volume': [100, 200, 300, 400, 500]
        })
        
        success = await predictor.train_model(small_data)
        assert success is False  # Should fail due to insufficient data


class TestPatchTSTLongHorizonForecasting:
    """Test PatchTST's effectiveness for long-horizon forecasting"""
    
    def test_multi_horizon_predictions(self):
        """Test multi-horizon prediction capability"""
        config = PatchTSTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            patch_length=16,
            stride=8,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        
        x = torch.randn(1, 64, 2)
        output = model(x)
        
        # Should have predictions for multiple horizons
        horizons = ['1h', '4h', '24h']
        for horizon in horizons:
            assert f'price_{horizon}' in output
            pred = output[f'price_{horizon}']
            if config.channel_independence:
                assert pred.shape == (1, config.n_channels)
            else:
                assert pred.shape == (1, 1)
    
    def test_long_sequence_forecasting(self):
        """Test forecasting capability with long input sequences"""
        config = PatchTSTConfig(
            d_model=32,
            n_heads=2,
            n_layers=2,
            patch_length=32,
            stride=16,
            n_channels=1,
            max_seq_length=512
        )
        
        model = PatchTSTNetwork(config)
        
        # Long sequence input
        x = torch.randn(1, 256, 1)
        output = model(x)
        
        # Should still produce valid predictions
        assert 'price_24h' in output
        assert output['price_24h'].shape == (1, 1)  # Single channel output