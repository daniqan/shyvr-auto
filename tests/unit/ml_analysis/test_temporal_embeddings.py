"""
Tests for temporal embeddings optimized for financial time-series
Following TDD methodology - tests written first before implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# These imports will fail initially - that's expected for TDD
from src.ml_analysis.transformers.temporal_embeddings import (
    TemporalEmbedding,
    FinancialTemporalEmbedding,
    CryptoMarketEmbedding,
    TemporalEmbeddingConfig,
    TradingSessionEncoder,
    MarketRegimeEncoder,
    VolatilityRegimeEncoder
)


class TestTemporalEmbeddingConfig:
    """Test configuration for temporal embeddings"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = TemporalEmbeddingConfig()
        
        assert config.d_model == 512
        assert config.max_sequence_length == 1000
        assert config.enable_trading_sessions == True
        assert config.enable_market_regimes == True
        assert config.enable_volatility_regimes == True
        assert config.enable_crypto_events == True
        assert config.dropout == 0.1
        
    def test_config_validation(self):
        """Test configuration validation"""
        with pytest.raises(ValueError, match="d_model must be positive"):
            TemporalEmbeddingConfig(d_model=0)
        
        with pytest.raises(ValueError, match="max_sequence_length must be positive"):
            TemporalEmbeddingConfig(max_sequence_length=-1)
        
        # GCP memory constraints
        with pytest.raises(ValueError, match="max_sequence_length must be <= 1000"):
            TemporalEmbeddingConfig(max_sequence_length=1500)


class TestBasicTemporalEmbedding:
    """Test basic temporal embedding functionality"""
    
    @pytest.fixture
    def config(self):
        return TemporalEmbeddingConfig(d_model=256, max_sequence_length=100)
    
    @pytest.fixture
    def embedding(self, config):
        return TemporalEmbedding(config)
    
    def test_initialization(self, embedding, config):
        """Test temporal embedding initialization"""
        assert isinstance(embedding, nn.Module)
        assert embedding.d_model == config.d_model
        assert embedding.max_length == config.max_sequence_length
        
        # Should have embeddings for basic temporal features
        assert hasattr(embedding, 'hour_embedding')
        assert hasattr(embedding, 'day_embedding')
        assert hasattr(embedding, 'month_embedding')
        assert hasattr(embedding, 'year_embedding')
    
    def test_forward_pass_shape(self, embedding):
        """Test forward pass output shape"""
        batch_size = 2
        seq_len = 50
        
        # Mock temporal features
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))  # Unix timestamps
        
        output = embedding(timestamps)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
    
    def test_timestamp_parsing(self, embedding):
        """Test timestamp parsing into temporal components"""
        # Create known timestamps
        timestamp = datetime(2024, 6, 15, 14, 30, 0).timestamp()  # June 15, 2024, 2:30 PM
        timestamps = torch.tensor([[timestamp]], dtype=torch.long)
        
        output = embedding(timestamps)
        
        assert output.shape == (1, 1, 256)
        assert torch.isfinite(output).all()
    
    def test_variable_sequence_lengths(self, embedding):
        """Test handling of variable sequence lengths"""
        batch_size = 3
        
        for seq_len in [10, 50, 100]:
            timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
            output = embedding(timestamps)
            
            assert output.shape == (batch_size, seq_len, 256)
            assert torch.isfinite(output).all()
    
    def test_temporal_consistency(self, embedding):
        """Test that similar timestamps produce similar embeddings"""
        # Timestamps 1 hour apart
        base_time = datetime(2024, 1, 1, 12, 0, 0).timestamp()
        time_1 = base_time
        time_2 = base_time + 3600  # +1 hour
        
        timestamps_1 = torch.tensor([[time_1]], dtype=torch.long)
        timestamps_2 = torch.tensor([[time_2]], dtype=torch.long)
        
        embed_1 = embedding(timestamps_1)
        embed_2 = embedding(timestamps_2)
        
        # Should be similar but not identical
        cosine_sim = torch.cosine_similarity(
            embed_1.flatten(), embed_2.flatten(), dim=0
        )
        assert 0.7 < cosine_sim < 0.99  # Similar but distinguishable


class TestFinancialTemporalEmbedding:
    """Test financial market specific temporal embeddings"""
    
    @pytest.fixture
    def config(self):
        return TemporalEmbeddingConfig(
            d_model=256,
            enable_trading_sessions=True,
            enable_market_regimes=True
        )
    
    @pytest.fixture
    def embedding(self, config):
        return FinancialTemporalEmbedding(config)
    
    def test_trading_session_encoding(self, embedding):
        """Test trading session encoding (Asian, European, US)"""
        batch_size = 1
        seq_len = 24  # 24 hours
        
        # Create timestamps spanning different trading sessions
        base_time = datetime(2024, 6, 15, 0, 0, 0).timestamp()  # Midnight UTC
        timestamps = torch.tensor([
            [base_time + i * 3600 for i in range(seq_len)]  # Hourly for 24 hours
        ], dtype=torch.long)
        
        output = embedding(timestamps)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different trading sessions should produce different embeddings
        # Asian session (roughly hours 0-8 UTC)
        # European session (roughly hours 8-16 UTC)  
        # US session (roughly hours 16-24 UTC)
        asian_embed = output[0, 2]    # 2 AM UTC
        european_embed = output[0, 10] # 10 AM UTC
        us_embed = output[0, 18]      # 6 PM UTC
        
        # Should be distinguishable
        asian_european_sim = torch.cosine_similarity(asian_embed, european_embed, dim=0)
        european_us_sim = torch.cosine_similarity(european_embed, us_embed, dim=0)
        
        assert asian_european_sim < 0.9
        assert european_us_sim < 0.9
    
    def test_market_regime_encoding(self, embedding):
        """Test market regime encoding (bull, bear, sideways)"""
        batch_size = 1
        seq_len = 10
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock market regimes: 0=bull, 1=bear, 2=sideways
        market_regimes = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
        
        output = embedding(timestamps, market_regimes=market_regimes)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different regimes should produce different embeddings
        bull_embed = output[0, 0]
        bear_embed = output[0, 1] 
        sideways_embed = output[0, 2]
        
        bull_bear_sim = torch.cosine_similarity(bull_embed, bear_embed, dim=0)
        bull_sideways_sim = torch.cosine_similarity(bull_embed, sideways_embed, dim=0)
        
        assert bull_bear_sim < 0.85
        assert bull_sideways_sim < 0.85
    
    def test_volatility_regime_encoding(self, embedding):
        """Test volatility regime encoding (low, medium, high)"""
        batch_size = 1
        seq_len = 9
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock volatility regimes: 0=low, 1=medium, 2=high
        volatility_regimes = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2]])
        
        output = embedding(timestamps, volatility_regimes=volatility_regimes)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different volatility regimes should be distinguishable
        low_vol = output[0, 0]
        med_vol = output[0, 1]
        high_vol = output[0, 2]
        
        low_high_sim = torch.cosine_similarity(low_vol, high_vol, dim=0)
        assert low_high_sim < 0.8  # Should be quite different


class TestCryptoMarketEmbedding:
    """Test crypto-specific market embeddings"""
    
    @pytest.fixture
    def config(self):
        return TemporalEmbeddingConfig(
            d_model=256,
            enable_crypto_events=True,
            enable_trading_sessions=True
        )
    
    @pytest.fixture
    def embedding(self, config):
        return CryptoMarketEmbedding(config)
    
    def test_crypto_event_embedding(self, embedding):
        """Test crypto event embeddings (halvings, upgrades, etc.)"""
        batch_size = 1
        seq_len = 8
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock crypto events: 0=normal, 1=halving, 2=upgrade, 3=major_news, 4=regulation
        crypto_events = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
        
        output = embedding(timestamps, crypto_events=crypto_events)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different events should produce different embeddings
        normal = output[0, 0]
        halving = output[0, 1]
        upgrade = output[0, 2]
        news = output[0, 3]
        regulation = output[0, 4]
        
        # Major events should be quite different from normal
        normal_halving_sim = torch.cosine_similarity(normal, halving, dim=0)
        normal_regulation_sim = torch.cosine_similarity(normal, regulation, dim=0)
        
        assert normal_halving_sim < 0.8
        assert normal_regulation_sim < 0.8
    
    def test_block_time_encoding(self, embedding):
        """Test block time encoding for different cryptocurrencies"""
        batch_size = 1
        seq_len = 6
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock block times: Bitcoin ~600s, Ethereum ~12s, Solana ~0.4s
        block_times = torch.tensor([[600, 12, 0.4, 600, 12, 0.4]])
        
        output = embedding(timestamps, block_times=block_times)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different block times should be encoded differently
        btc_embed = output[0, 0]  # Bitcoin
        eth_embed = output[0, 1]  # Ethereum
        sol_embed = output[0, 2]  # Solana
        
        btc_eth_sim = torch.cosine_similarity(btc_embed, eth_embed, dim=0)
        eth_sol_sim = torch.cosine_similarity(eth_embed, sol_embed, dim=0)
        
        assert btc_eth_sim < 0.9
        assert eth_sol_sim < 0.9
    
    def test_defi_protocol_events(self, embedding):
        """Test DeFi protocol event embeddings"""
        batch_size = 1
        seq_len = 5
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock DeFi events: 0=normal, 1=liquidity_event, 2=governance, 3=hack, 4=launch
        defi_events = torch.tensor([[0, 1, 2, 3, 4]])
        
        output = embedding(timestamps, defi_events=defi_events)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Different DeFi events should be distinct
        normal = output[0, 0]
        hack = output[0, 3]  # Major negative event
        launch = output[0, 4]  # Major positive event
        
        normal_hack_sim = torch.cosine_similarity(normal, hack, dim=0)
        hack_launch_sim = torch.cosine_similarity(hack, launch, dim=0)
        
        assert normal_hack_sim < 0.7  # Very different
        assert hack_launch_sim < 0.6   # Opposite events
    
    def test_network_congestion_encoding(self, embedding):
        """Test network congestion level encoding"""
        batch_size = 1
        seq_len = 4
        
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Mock congestion levels: 0=low, 1=medium, 2=high, 3=extreme
        congestion_levels = torch.tensor([[0, 1, 2, 3]])
        
        output = embedding(timestamps, network_congestion=congestion_levels)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert torch.isfinite(output).all()
        
        # Higher congestion should be progressively different
        low = output[0, 0]
        high = output[0, 2]
        extreme = output[0, 3]
        
        low_high_sim = torch.cosine_similarity(low, high, dim=0)
        low_extreme_sim = torch.cosine_similarity(low, extreme, dim=0)
        
        assert low_high_sim < 0.8
        assert low_extreme_sim < 0.7  # Should be even more different


class TestTradingSessionEncoder:
    """Test trading session specific encoding"""
    
    @pytest.fixture
    def encoder(self):
        return TradingSessionEncoder(d_model=128)
    
    def test_session_detection(self, encoder):
        """Test automatic trading session detection from timestamps"""
        # Test timestamps in different sessions
        timestamps = [
            datetime(2024, 6, 15, 2, 0, 0),   # Asian session
            datetime(2024, 6, 15, 10, 0, 0),  # European session  
            datetime(2024, 6, 15, 18, 0, 0),  # US session
        ]
        
        sessions = encoder.detect_trading_sessions(timestamps)
        
        assert len(sessions) == 3
        assert sessions[0] == 0  # Asian
        assert sessions[1] == 1  # European
        assert sessions[2] == 2  # US
    
    def test_session_overlap_encoding(self, encoder):
        """Test encoding of session overlaps (higher volatility periods)"""
        # Times during session overlaps
        overlap_times = [
            datetime(2024, 6, 15, 8, 0, 0),   # Asian-European overlap
            datetime(2024, 6, 15, 13, 0, 0),  # European-US overlap
        ]
        
        sessions = encoder.detect_trading_sessions(overlap_times)
        overlaps = encoder.detect_session_overlaps(overlap_times)
        
        assert len(overlaps) == 2
        assert overlaps[0] == True   # Overlap detected
        assert overlaps[1] == True   # Overlap detected


class TestMarketRegimeEncoder:
    """Test market regime detection and encoding"""
    
    @pytest.fixture  
    def encoder(self):
        return MarketRegimeEncoder(d_model=128, lookback_window=20)
    
    def test_regime_detection_from_prices(self, encoder):
        """Test market regime detection from price data"""
        # Mock price data for different regimes
        bull_prices = torch.tensor([100, 105, 110, 115, 120, 125, 130])  # Uptrend
        bear_prices = torch.tensor([130, 125, 120, 115, 110, 105, 100])  # Downtrend  
        sideways_prices = torch.tensor([100, 102, 98, 101, 99, 103, 100])  # Sideways
        
        bull_regime = encoder.detect_regime(bull_prices)
        bear_regime = encoder.detect_regime(bear_prices)
        sideways_regime = encoder.detect_regime(sideways_prices)
        
        assert bull_regime == 0    # Bull market
        assert bear_regime == 1    # Bear market
        assert sideways_regime == 2  # Sideways market
    
    def test_regime_transition_detection(self, encoder):
        """Test detection of regime transitions"""
        # Price data with regime change (bull to bear)
        transition_prices = torch.tensor([
            100, 105, 110, 115, 120,  # Bull phase
            118, 115, 110, 105, 100   # Bear phase
        ])
        
        regimes = encoder.detect_regime_sequence(transition_prices)
        transitions = encoder.detect_transitions(regimes)
        
        assert len(transitions) > 0  # Should detect transition
        # Transition should occur around the middle of the sequence


class TestVolatilityRegimeEncoder:
    """Test volatility regime detection and encoding"""
    
    @pytest.fixture
    def encoder(self):
        return VolatilityRegimeEncoder(d_model=128, vol_window=10)
    
    def test_volatility_calculation(self, encoder):
        """Test volatility calculation from returns"""
        # Mock return data with different volatilities
        low_vol_returns = torch.tensor([0.001, -0.002, 0.0015, -0.001, 0.002])
        high_vol_returns = torch.tensor([0.05, -0.08, 0.06, -0.04, 0.07])
        
        low_vol = encoder.calculate_volatility(low_vol_returns)
        high_vol = encoder.calculate_volatility(high_vol_returns)
        
        assert high_vol > low_vol
        assert low_vol > 0
        assert high_vol > 0
    
    def test_volatility_regime_classification(self, encoder):
        """Test volatility regime classification"""
        low_vol = 0.01   # 1% volatility
        med_vol = 0.05   # 5% volatility  
        high_vol = 0.15  # 15% volatility
        
        low_regime = encoder.classify_volatility_regime(low_vol)
        med_regime = encoder.classify_volatility_regime(med_vol)
        high_regime = encoder.classify_volatility_regime(high_vol)
        
        assert low_regime == 0   # Low volatility
        assert med_regime == 1   # Medium volatility
        assert high_regime == 2  # High volatility


class TestTemporalEmbeddingIntegration:
    """Test integration with existing transformer architecture"""
    
    def test_memory_efficiency_all_embeddings(self):
        """Test all embeddings meet memory constraints"""
        config = TemporalEmbeddingConfig(d_model=512, max_sequence_length=1000)
        
        embeddings = [
            TemporalEmbedding(config),
            FinancialTemporalEmbedding(config),
            CryptoMarketEmbedding(config)
        ]
        
        batch_size = 1
        timestamps = torch.randint(0, 1000000, (batch_size, config.max_sequence_length))
        
        for embedding in embeddings:
            output = embedding(timestamps)
            assert output.shape == (batch_size, config.max_sequence_length, config.d_model)
            assert torch.isfinite(output).all()
    
    def test_transformer_base_compatibility(self):
        """Test compatibility with TransformerBase"""
        # This will test integration once the base class is updated
        config = TemporalEmbeddingConfig(d_model=256)
        embedding = TemporalEmbedding(config)
        
        # Should be compatible with transformer input requirements
        batch_size, seq_len, d_model = 2, 50, 256
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        output = embedding(timestamps)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_production_performance_requirements(self):
        """Test embeddings meet <100ms inference requirement"""
        import time
        
        config = TemporalEmbeddingConfig(d_model=256, max_sequence_length=100)
        embedding = CryptoMarketEmbedding(config)  # Most complex embedding
        
        timestamps = torch.randint(0, 1000000, (1, 100))
        
        # Warm up
        _ = embedding(timestamps)
        
        # Time the embedding
        start_time = time.time()
        with torch.no_grad():
            _ = embedding(timestamps)
        embedding_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be very fast
        assert embedding_time < 20, f"Embedding took {embedding_time:.2f}ms, too slow for production"
    
    def test_gradient_flow_all_embeddings(self):
        """Test gradient flow through all embedding types"""
        config = TemporalEmbeddingConfig(d_model=128, max_sequence_length=20)
        
        embeddings = [
            TemporalEmbedding(config),
            FinancialTemporalEmbedding(config),
            CryptoMarketEmbedding(config)
        ]
        
        for embedding in embeddings:
            timestamps = torch.randint(0, 1000000, (1, 20))
            output = embedding(timestamps)
            loss = output.sum()
            loss.backward()
            
            # Check gradients exist and are finite
            for param in embedding.parameters():
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_feature_combination_consistency(self):
        """Test that combining different temporal features is consistent"""
        config = TemporalEmbeddingConfig(d_model=256)
        embedding = CryptoMarketEmbedding(config)
        
        batch_size, seq_len = 1, 10
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len))
        
        # Test with different feature combinations
        basic_output = embedding(timestamps)
        
        with_events = embedding(
            timestamps, 
            crypto_events=torch.randint(0, 5, (batch_size, seq_len))
        )
        
        with_block_times = embedding(
            timestamps,
            block_times=torch.rand(batch_size, seq_len) * 600
        )
        
        # All should have same shape but different values
        assert basic_output.shape == with_events.shape == with_block_times.shape
        assert not torch.allclose(basic_output, with_events)
        assert not torch.allclose(basic_output, with_block_times)
        assert not torch.allclose(with_events, with_block_times)