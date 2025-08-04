"""
Standalone unit tests for PatchTST implementation
These tests work without complex configuration dependencies
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import unittest.mock as mock
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

# Mock configuration before any imports
def mock_config():
    config = mock.MagicMock()
    config.security.secret_key = "test_secret_key_1234567890"
    config.ml = mock.MagicMock()
    return config

# Apply comprehensive mocks
mock_patches = [
    mock.patch('src.utils.config.get_config', return_value=mock_config()),
    mock.patch('src.activity_logging.activity_logger.get_config', return_value=mock_config()),
    mock.patch('src.activity_logging.activity_logger.ActivityLogger'),
]

# Start all patches
for patch in mock_patches:
    patch.start()

try:
    # Now safe to import
    from src.ml_analysis.transformers.patchtst import PatchTSTConfig, PatchTSTNetwork, PatchTSTPredictor
    from src.ml_analysis.base import ModelType, PredictionDirection
    from src.discovery.base import DiscoveredToken
    from src.utils.base import Chain
finally:
    # Stop patches
    for patch in mock_patches:
        patch.stop()


class TestPatchTSTStandalone:
    """Standalone tests for PatchTST that don't require complex configuration"""
    
    def test_config_creation_and_validation(self):
        """Test PatchTST configuration creation and validation"""
        # Test default config
        config = PatchTSTConfig()
        assert config.d_model == 128
        assert config.patch_length == 16
        assert config.stride == 8
        assert config.n_channels == 1
        assert config.channel_independence is True
        
        # Test custom config
        custom_config = PatchTSTConfig(
            d_model=256,
            patch_length=32,
            stride=16,
            n_channels=3
        )
        assert custom_config.d_model == 256
        assert custom_config.patch_length == 32
        assert custom_config.n_channels == 3
        
        # Test validation - patch_length > stride
        with pytest.raises(ValueError, match="patch_length must be greater than stride"):
            PatchTSTConfig(patch_length=8, stride=16)
        
        # Test validation - positive channels
        with pytest.raises(ValueError, match="n_channels must be positive"):
            PatchTSTConfig(n_channels=0)
    
    def test_patch_calculation(self):
        """Test patch count calculation"""
        config = PatchTSTConfig(
            max_seq_length=100,
            patch_length=16,
            stride=8
        )
        expected_patches = (100 - 16) // 8 + 1
        assert config.get_n_patches() == expected_patches
        
        # Test with custom sequence length
        assert config.get_n_patches(64) == (64 - 16) // 8 + 1
    
    def test_network_initialization(self):
        """Test PatchTST network initialization"""
        config = PatchTSTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            patch_length=16,
            stride=8,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        
        # Check required components
        assert hasattr(model, 'patch_embedding')
        assert hasattr(model, 'channel_embeddings')
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'prediction_heads')
        
        # Check patch embedding dimensions
        assert model.patch_embedding.projection.in_features == config.patch_length
        assert model.patch_embedding.projection.out_features == config.d_model
        
        # Check prediction heads
        for horizon in ['1h', '4h', '24h']:
            assert hasattr(model.prediction_heads, horizon)
    
    def test_patch_creation(self):
        """Test patch creation from time series"""
        config = PatchTSTConfig(
            d_model=32,
            patch_length=8,
            stride=4,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        
        # Test input
        batch_size = 1
        seq_len = 32
        n_channels = 2
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Create patches
        patches = model._create_patches(x)
        
        # Check dimensions
        expected_n_patches = (seq_len - config.patch_length) // config.stride + 1
        assert patches.shape == (batch_size, n_channels, expected_n_patches, config.patch_length)
        
        # Test error case - sequence too short
        x_short = torch.randn(1, 5, 2)  # Shorter than patch_length
        with pytest.raises(ValueError, match="Sequence length.*too short"):
            model._create_patches(x_short)
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        config = PatchTSTConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            patch_length=8,
            stride=4,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        
        # Test input
        batch_size = 2
        seq_len = 32
        n_channels = 2
        x = torch.randn(batch_size, seq_len, n_channels)
        
        # Forward pass
        output = model(x)
        
        # Check required outputs
        assert 'price_1h' in output
        assert 'price_4h' in output
        assert 'price_24h' in output
        assert 'confidence' in output
        assert 'direction_probs' in output
        
        # Check output shapes with channel independence
        assert output['price_1h'].shape == (batch_size, n_channels)
        assert output['confidence'].shape == (batch_size, 1)
        
        # Test wrong number of channels
        x_wrong = torch.randn(1, 32, 5)  # 5 channels instead of 2
        with pytest.raises(ValueError, match="Expected.*channels"):
            model(x_wrong)
    
    def test_channel_independence(self):
        """Test channel independence functionality"""
        config = PatchTSTConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            patch_length=8,
            stride=4,
            n_channels=3,
            channel_independence=True
        )
        
        model = PatchTSTNetwork(config)
        
        # Create input with distinct patterns per channel
        batch_size = 1
        seq_len = 32
        n_channels = 3
        
        x = torch.zeros(batch_size, seq_len, n_channels)
        x[:, :, 0] = torch.linspace(0, 1, seq_len)  # Linear trend
        x[:, :, 1] = torch.sin(torch.linspace(0, 4*np.pi, seq_len))  # Sinusoidal
        x[:, :, 2] = torch.randn(seq_len)  # Random noise
        
        output = model(x)
        
        # Should get different predictions per channel
        price_1h = output['price_1h']
        assert price_1h.shape == (batch_size, n_channels)
        
        # Predictions should be different for different channels
        channel_preds = price_1h[0, :].detach().numpy()
        assert not np.allclose(channel_preds, channel_preds[0], atol=1e-3)
    
    def test_predictor_initialization(self):
        """Test PatchTST predictor initialization"""
        config_dict = {
            'd_model': 32,
            'n_heads': 2,
            'n_layers': 1,
            'patch_length': 8,
            'stride': 4,
            'n_channels': 2,
            'max_seq_length': 50
        }
        
        # Mock the config during predictor creation
        with mock.patch('src.utils.config.get_config', return_value=mock_config()):
            with mock.patch('src.activity_logging.activity_logger.get_config', return_value=mock_config()):
                predictor = PatchTSTPredictor(config_dict)
        
        assert predictor.model_type == ModelType.PATCHTST
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'patch_length')
        assert hasattr(predictor, 'stride')
        assert predictor.patch_length == 8
        assert predictor.stride == 4
        assert predictor.n_channels == 2
    
    def test_sequence_preparation(self):
        """Test sequence preparation for patch processing"""
        config_dict = {
            'd_model': 32,
            'patch_length': 8,
            'stride': 4,
            'n_channels': 2
        }
        
        with mock.patch('src.utils.config.get_config', return_value=mock_config()):
            with mock.patch('src.activity_logging.activity_logger.get_config', return_value=mock_config()):
                predictor = PatchTSTPredictor(config_dict)
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(40).cumsum() + 100,
            'volume': np.random.exponential(1000, 40),
            'returns': np.random.randn(40) * 0.02
        })
        
        token = DiscoveredToken(
            address="0x123",
            name="TEST",
            symbol="TEST",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        sequence = predictor._prepare_patch_sequence(data, token)
        
        # Check sequence properties
        assert isinstance(sequence, np.ndarray)
        assert sequence.shape[0] == len(data)
        assert sequence.shape[1] == predictor.n_channels
        assert sequence.dtype == np.float32
    
    def test_memory_efficiency(self):
        """Test memory efficiency estimation"""
        config = PatchTSTConfig(
            max_seq_length=1000,
            patch_length=16,
            n_channels=5
        )
        
        memory_mb = config.estimate_memory_usage_mb()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
        
        # Should be more efficient than full attention
        assert memory_mb < 1000  # Reasonable for production
    
    def test_model_type_integration(self):
        """Test model type enum integration"""
        assert ModelType.PATCHTST in ModelType
        assert ModelType.PATCHTST.value == "patchtst"


class TestPatchTSTPerformanceStandalone:
    """Performance tests for PatchTST"""
    
    def test_inference_speed(self):
        """Test inference speed with patches"""
        config = PatchTSTConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            patch_length=16,
            stride=8,
            n_channels=2
        )
        
        model = PatchTSTNetwork(config)
        model.eval()
        
        # Test with reasonable sequence length
        x = torch.randn(1, 128, 2)
        
        import time
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 2.0  # Should be fast
        assert output is not None
    
    def test_patch_scaling(self):
        """Test scaling with different patch parameters"""
        patch_lengths = [8, 16, 32]
        
        for patch_length in patch_lengths:
            config = PatchTSTConfig(
                d_model=32,
                n_heads=2,
                n_layers=1,
                patch_length=patch_length,
                stride=patch_length // 2,
                n_channels=2
            )
            
            model = PatchTSTNetwork(config)
            x = torch.randn(1, 64, 2)
            
            # Should work with different patch sizes
            output = model(x)
            assert output is not None
            assert 'price_1h' in output


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])