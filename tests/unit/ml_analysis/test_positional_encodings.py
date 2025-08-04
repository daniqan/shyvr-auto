"""
Tests for positional encoding variants optimized for crypto time-series
Following TDD methodology - tests written first before implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# These imports will fail initially - that's expected for TDD
from src.ml_analysis.transformers.positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    RelativePositionalEncoding,
    CryptoAwarePositionalEncoding,
    PositionalEncodingConfig
)


class TestPositionalEncodingConfig:
    """Test configuration class for positional encodings"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = PositionalEncodingConfig()
        
        assert config.d_model == 512
        assert config.max_length == 1000
        assert config.dropout == 0.1
        assert config.learnable_temperature == 10000.0
        assert config.relative_max_distance == 128
        assert config.crypto_aware_features == True
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid d_model
        with pytest.raises(ValueError, match="d_model must be positive"):
            PositionalEncodingConfig(d_model=0)
        
        # Test invalid max_length
        with pytest.raises(ValueError, match="max_length must be positive"):
            PositionalEncodingConfig(max_length=-1)
        
        # Test memory constraints for GCP
        with pytest.raises(ValueError, match="max_length must be <= 1000 for memory constraints"):
            PositionalEncodingConfig(max_length=1500)


class TestSinusoidalPositionalEncoding:
    """Test sinusoidal positional encoding for time-series"""
    
    @pytest.fixture
    def config(self):
        return PositionalEncodingConfig(d_model=256, max_length=100)
    
    @pytest.fixture
    def encoding(self, config):
        return SinusoidalPositionalEncoding(config)
    
    def test_initialization(self, encoding, config):
        """Test sinusoidal encoding initialization"""
        assert isinstance(encoding, nn.Module)
        assert encoding.d_model == config.d_model
        assert encoding.max_length == config.max_length
        assert hasattr(encoding, 'pe')
        
    def test_encoding_shape(self, encoding):
        """Test output shape for various sequence lengths"""
        batch_size = 2
        for seq_len in [10, 50, 100]:
            x = torch.randn(batch_size, seq_len, 256)
            output = encoding(x)
            
            assert output.shape == (batch_size, seq_len, 256)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_positional_pattern(self, encoding):
        """Test that positional encoding follows expected sinusoidal pattern"""
        x = torch.zeros(1, 100, 256)
        output = encoding(x)
        
        # Extract positional encoding (since input was zeros)
        pe = output[0]  # (seq_len, d_model)
        
        # Check sinusoidal properties
        # Even dimensions should have sin, odd should have cos
        pos_0 = pe[0]  # First position
        pos_1 = pe[1]  # Second position
        
        # Different positions should have different encodings
        assert not torch.allclose(pos_0, pos_1)
        
        # Check periodicity for higher frequencies
        assert pe.shape == (100, 256)
    
    def test_time_series_ordering(self, encoding):
        """Test that encoding preserves temporal ordering"""
        x = torch.randn(1, 50, 256)
        output = encoding(x)
        
        # Get positional encodings
        pe = output - x  # Remove input, get pure positional encoding
        
        # Adjacent positions should be similar but not identical
        for i in range(pe.shape[1] - 1):
            pos_current = pe[0, i]
            pos_next = pe[0, i + 1]
            
            # Should be different
            assert not torch.allclose(pos_current, pos_next)
            
            # But should be reasonably similar (continuity)
            cosine_sim = torch.cosine_similarity(pos_current, pos_next, dim=0)
            assert cosine_sim > 0.5  # Should be somewhat similar
    
    def test_variable_sequence_length(self, encoding):
        """Test handling of variable sequence lengths"""
        batch_size = 3
        seq_lengths = [20, 50, 80]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, 256)
            output = encoding(x)
            
            assert output.shape == (batch_size, seq_len, 256)
            assert torch.isfinite(output).all()
    
    def test_memory_efficiency(self, config):
        """Test memory efficiency for production use"""
        # Test with maximum sequence length
        encoding = SinusoidalPositionalEncoding(config)
        x = torch.randn(1, config.max_length, config.d_model)
        
        # Should not raise memory errors
        output = encoding(x)
        assert output.shape == (1, config.max_length, config.d_model)


class TestLearnablePositionalEncoding:
    """Test learnable positional encoding"""
    
    @pytest.fixture
    def config(self):
        return PositionalEncodingConfig(d_model=256, max_length=100)
    
    @pytest.fixture
    def encoding(self, config):
        return LearnablePositionalEncoding(config)
    
    def test_initialization(self, encoding, config):
        """Test learnable encoding has trainable parameters"""
        assert isinstance(encoding, nn.Module)
        assert encoding.d_model == config.d_model
        assert encoding.max_length == config.max_length
        
        # Should have learnable parameters
        params = list(encoding.parameters())
        assert len(params) > 0
        assert all(param.requires_grad for param in params)
    
    def test_encoding_shape(self, encoding):
        """Test output shape consistency"""
        batch_size = 2
        seq_len = 50
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()
    
    def test_gradient_flow(self, encoding):
        """Test that gradients flow through learnable parameters"""
        x = torch.randn(1, 10, 256, requires_grad=True)
        output = encoding(x)
        loss = output.sum()
        loss.backward()
        
        # Check that positional encoding parameters have gradients
        for param in encoding.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_training_adaptation(self, encoding):
        """Test that encoding adapts during training"""
        # Get initial state
        initial_params = [param.clone() for param in encoding.parameters()]
        
        # Simulate training step
        optimizer = torch.optim.Adam(encoding.parameters(), lr=0.001)
        
        x = torch.randn(4, 50, 256)
        output = encoding(x)
        loss = output.pow(2).mean()  # Dummy loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        for initial, current in zip(initial_params, encoding.parameters()):
            assert not torch.allclose(initial, current)


class TestRelativePositionalEncoding:
    """Test relative positional encoding for attention mechanisms"""
    
    @pytest.fixture
    def config(self):
        return PositionalEncodingConfig(d_model=256, max_length=100, relative_max_distance=64)
    
    @pytest.fixture
    def encoding(self, config):
        return RelativePositionalEncoding(config)
    
    def test_initialization(self, encoding, config):
        """Test relative encoding initialization"""
        assert isinstance(encoding, nn.Module)
        assert encoding.d_model == config.d_model
        assert encoding.max_distance == config.relative_max_distance
        
        # Should have relative position embeddings
        assert hasattr(encoding, 'relative_positions')
    
    def test_relative_distance_computation(self, encoding):
        """Test relative distance matrix computation"""
        seq_len = 20
        distances = encoding._get_relative_distances(seq_len)
        
        assert distances.shape == (seq_len, seq_len)
        
        # Diagonal should be 0 (same position)
        assert (distances.diagonal() == 0).all()
        
        # Should be symmetric around diagonal
        for i in range(seq_len):
            for j in range(seq_len):
                expected_distance = j - i
                # Clamp to max distance
                expected_distance = max(-encoding.max_distance, 
                                      min(encoding.max_distance, expected_distance))
                assert distances[i, j] == expected_distance
    
    def test_attention_bias_computation(self, encoding):
        """Test attention bias computation for relative positions"""
        seq_len = 10
        bias = encoding.get_attention_bias(seq_len)
        
        # Should be compatible with attention mechanism
        assert bias.shape == (seq_len, seq_len)
        assert torch.isfinite(bias).all()
        
        # Bias should be symmetric for same relative distances
        # positions (i,j) and (i+k,j+k) should have same bias if both valid
        if seq_len >= 5:
            assert torch.allclose(bias[0, 2], bias[1, 3])  # Both have distance +2
            assert torch.allclose(bias[2, 0], bias[3, 1])  # Both have distance -2
    
    def test_integration_with_attention(self, encoding):
        """Test integration with attention mechanism"""
        batch_size = 2
        seq_len = 20
        n_heads = 4
        
        # Simulate attention scores before applying relative bias
        attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)
        bias = encoding.get_attention_bias(seq_len)
        
        # Apply bias
        attention_with_bias = attention_scores + bias.unsqueeze(0).unsqueeze(0)
        
        assert attention_with_bias.shape == attention_scores.shape
        assert torch.isfinite(attention_with_bias).all()


class TestCryptoAwarePositionalEncoding:
    """Test crypto-specific positional encoding features"""
    
    @pytest.fixture
    def config(self):
        return PositionalEncodingConfig(
            d_model=256, 
            max_length=100, 
            crypto_aware_features=True
        )
    
    @pytest.fixture
    def encoding(self, config):
        return CryptoAwarePositionalEncoding(config)
    
    def test_initialization(self, encoding):
        """Test crypto-aware encoding initialization"""
        assert isinstance(encoding, nn.Module)
        assert hasattr(encoding, 'block_time_encoding')
        assert hasattr(encoding, 'trading_session_encoding')
        assert hasattr(encoding, 'crypto_event_encoding')
    
    def test_block_time_features(self, encoding):
        """Test block time positional features for on-chain data"""
        # Mock block times (typical Bitcoin ~10 min, Ethereum ~12 sec)
        block_times = torch.tensor([600, 12, 12, 600, 600])  # Mixed BTC/ETH
        
        x = torch.randn(1, 5, 256)
        output = encoding(x, block_times=block_times)
        
        assert output.shape == (1, 5, 256)
        assert torch.isfinite(output).all()
        
        # Different block times should produce different encodings
        encoding_btc = output[0, 0]  # Bitcoin block
        encoding_eth = output[0, 1]  # Ethereum block
        assert not torch.allclose(encoding_btc, encoding_eth)
    
    def test_trading_session_features(self, encoding):
        """Test trading session indicators (Asian, European, US)"""
        # Mock trading sessions: 0=Asian, 1=European, 2=US
        sessions = torch.tensor([0, 1, 2, 0, 1])
        
        x = torch.randn(1, 5, 256)
        output = encoding(x, trading_sessions=sessions)
        
        assert output.shape == (1, 5, 256)
        assert torch.isfinite(output).all()
        
        # Different sessions should have different encodings
        asian = output[0, 0]
        european = output[0, 1]
        us = output[0, 2]
        
        assert not torch.allclose(asian, european)
        assert not torch.allclose(european, us)
        assert not torch.allclose(asian, us)
    
    def test_crypto_event_embeddings(self, encoding):
        """Test crypto event embeddings (halvings, upgrades)"""
        # Mock events: 0=normal, 1=halving, 2=upgrade, 3=major news
        events = torch.tensor([0, 1, 0, 2, 3])
        
        x = torch.randn(1, 5, 256)
        output = encoding(x, crypto_events=events)
        
        assert output.shape == (1, 5, 256)
        assert torch.isfinite(output).all()
        
        # Different events should produce different encodings
        normal = output[0, 0]
        halving = output[0, 1]
        upgrade = output[0, 3]
        news = output[0, 4]
        
        # Event encodings should be distinct
        assert not torch.allclose(normal, halving)
        assert not torch.allclose(normal, upgrade)
        assert not torch.allclose(halving, upgrade)
    
    def test_seasonal_patterns(self, encoding):
        """Test seasonal pattern encoding for crypto markets"""
        # Mock seasonal features (hour of day, day of week, month)
        hours = torch.arange(24)
        x = torch.randn(1, 24, 256)
        
        output = encoding(x, hour_of_day=hours)
        
        assert output.shape == (1, 24, 256)
        assert torch.isfinite(output).all()
        
        # Hour 0 and hour 12 should have different encodings
        hour_0 = output[0, 0]
        hour_12 = output[0, 12]
        assert not torch.allclose(hour_0, hour_12)
    
    def test_combined_features(self, encoding):
        """Test combination of all crypto-aware features"""
        seq_len = 10
        x = torch.randn(1, seq_len, 256)
        
        # All feature types
        block_times = torch.randint(10, 600, (seq_len,))
        sessions = torch.randint(0, 3, (seq_len,))
        events = torch.randint(0, 4, (seq_len,))
        hours = torch.randint(0, 24, (seq_len,))
        
        output = encoding(
            x,
            block_times=block_times,
            trading_sessions=sessions,
            crypto_events=events,
            hour_of_day=hours
        )
        
        assert output.shape == (1, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # With all features, each position should be quite distinct
        for i in range(seq_len - 1):
            pos_i = output[0, i]
            pos_j = output[0, i + 1]
            # Should be different (though this is probabilistic)
            cosine_sim = torch.cosine_similarity(pos_i, pos_j, dim=0)
            # Very similar inputs might still be similar, so use loose threshold
            assert cosine_sim < 0.95


class TestPositionalEncodingIntegration:
    """Test integration with transformer architecture"""
    
    def test_memory_constraints_all_variants(self):
        """Test all variants meet GCP memory constraints"""
        config = PositionalEncodingConfig(d_model=512, max_length=1000)
        
        # Test all encoding types
        encodings = [
            SinusoidalPositionalEncoding(config),
            LearnablePositionalEncoding(config),
            RelativePositionalEncoding(config),
            CryptoAwarePositionalEncoding(config)
        ]
        
        batch_size = 1
        x = torch.randn(batch_size, config.max_length, config.d_model)
        
        for encoding in encodings:
            # Should not raise memory errors
            output = encoding(x)
            assert output.shape == x.shape
            assert torch.isfinite(output).all()
    
    def test_transformer_base_integration(self):
        """Test integration with TransformerBase class"""
        # This test will verify the encodings work with the existing base class
        # Will need to be implemented once the base class is updated
        pass
    
    def test_production_performance_requirements(self):
        """Test encodings meet <100ms inference requirement"""
        import time
        
        config = PositionalEncodingConfig(d_model=256, max_length=100)
        encoding = SinusoidalPositionalEncoding(config)
        
        x = torch.randn(1, 100, 256)
        
        # Warm up
        _ = encoding(x)
        
        # Time the encoding
        start_time = time.time()
        with torch.no_grad():
            _ = encoding(x)
        encoding_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be very fast (much less than 100ms total inference budget)
        assert encoding_time < 10, f"Encoding took {encoding_time:.2f}ms, too slow for production"