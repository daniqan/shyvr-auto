"""
Tests for TransformerBase abstract class and core transformer components
Following TDD methodology - tests written first
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.ml_analysis.transformers.base import TransformerBase, TransformerConfig
from src.ml_analysis.base import ModelType, PredictionResult
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestTransformerConfig:
    """Test TransformerConfig dataclass"""
    
    def test_transformer_config_defaults(self):
        """Test default values for TransformerConfig"""
        config = TransformerConfig()
        
        # Test default values according to GCP production requirements
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 2048
        assert config.dropout == 0.1
        assert config.max_seq_length == 1000  # Memory constraint <2GB
        assert config.activation == "gelu"
        assert config.norm_type == "layer"
        assert config.use_flash_attention == True  # For memory efficiency
        assert config.use_gradient_checkpointing == True
        assert config.vocab_size == 10000
        
    def test_transformer_config_validation(self):
        """Test TransformerConfig validation for production constraints"""
        # Test memory constraint validation
        with pytest.raises(ValueError, match="max_seq_length must be <= 1000"):
            TransformerConfig(max_seq_length=1500)
        
        # Test reasonable head size for attention
        with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
            TransformerConfig(d_model=512, n_heads=7)
    
    def test_transformer_config_memory_estimation(self):
        """Test memory usage estimation for GCP production"""
        config = TransformerConfig(max_seq_length=1000, d_model=512, n_heads=8)
        memory_mb = config.estimate_memory_usage_mb()
        
        # Should be under 2GB limit
        assert memory_mb < 2048
        assert isinstance(memory_mb, float)


class MockTransformer(TransformerBase):
    """Mock implementation for testing abstract base class"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(ModelType.TRANSFORMER, config.to_dict())
        # Store the original config object separately  
        self._transformer_config = config
        self.embedding = nn.Linear(config.d_model, config.d_model)
        self.output_projection = nn.Linear(config.d_model, 1)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        embedded = self.embedding(x)
        return self.output_projection(embedded.mean(dim=1))  # Simple mean pooling
    
    def get_attention_weights(self) -> torch.Tensor:
        # Mock attention weights for testing
        return torch.rand(1, 8, 100, 100)  # (batch, heads, seq_len, seq_len)


class TestTransformerBase:
    """Test TransformerBase abstract class"""
    
    @pytest.fixture
    def transformer_config(self):
        """Fixture for transformer configuration"""
        return TransformerConfig(
            d_model=256,  # Smaller for testing
            n_heads=4,
            n_layers=2,
            max_seq_length=100,
            dropout=0.1
        )
    
    @pytest.fixture
    def mock_transformer(self, transformer_config):
        """Fixture for mock transformer implementation"""
        return MockTransformer(transformer_config)
    
    @pytest.fixture
    def sample_token(self):
        """Fixture for sample discovered token"""
        return DiscoveredToken(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            price_usd=100.0
        )
    
    def test_transformer_base_initialization(self, mock_transformer):
        """Test TransformerBase initialization"""
        assert mock_transformer.model_type == ModelType.TRANSFORMER
        assert isinstance(mock_transformer.config, dict)
        assert not mock_transformer.is_model_trained()
        assert mock_transformer._model is None
    
    def test_transformer_base_abstract_methods(self):
        """Test that TransformerBase cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TransformerBase(ModelType.TRANSFORMER, {})
    
    def test_forward_method_signature(self, mock_transformer):
        """Test forward method accepts correct input shapes"""
        batch_size, seq_len, d_model = 2, 50, 256
        x = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = mock_transformer.forward(x, attention_mask)
        
        assert output.shape == (batch_size, 1)  # Price prediction output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_weights_method(self, mock_transformer):
        """Test attention weights extraction"""
        weights = mock_transformer.get_attention_weights()
        
        assert isinstance(weights, torch.Tensor)
        assert len(weights.shape) == 4  # (batch, heads, seq_len, seq_len)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0
    
    def test_variable_sequence_length_handling(self, mock_transformer):
        """Test handling of variable sequence lengths - critical for time-series"""
        # Test different sequence lengths
        for seq_len in [10, 50, 100]:
            x = torch.randn(1, seq_len, 256)
            attention_mask = torch.ones(1, seq_len)
            
            output = mock_transformer.forward(x, attention_mask)
            assert output.shape == (1, 1)
            assert not torch.isnan(output).any()
    
    def test_memory_efficiency_constraints(self, mock_transformer):
        """Test memory usage stays within GCP production limits"""
        # Test with maximum allowed sequence length
        batch_size = 4  # Small batch for memory testing
        seq_len = 1000  # Maximum allowed
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # This should not raise memory errors
        with torch.no_grad():
            output = mock_transformer.forward(x)
            assert output.shape == (batch_size, 1)
    
    @pytest.mark.asyncio
    async def test_analyze_token_method(self, mock_transformer, sample_token):
        """Test analyze_token implementation requirement"""
        # Mock the training state
        mock_transformer._is_trained = True
        mock_transformer._model = mock_transformer
        
        # Mock feature engineering
        with patch.object(mock_transformer, '_extract_features') as mock_extract:
            mock_extract.return_value = torch.randn(1, 100, 256)
            
            result = await mock_transformer.analyze_token(sample_token)
            
            assert isinstance(result, PredictionResult)
            assert result.token == sample_token
            assert result.model_type == ModelType.TRANSFORMER
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.analyzed_at, datetime)
    
    def test_get_required_features_method(self, mock_transformer):
        """Test required features specification"""
        features = mock_transformer.get_required_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(feature, str) for feature in features)
        
        # Should include transformer-specific features
        expected_features = [
            'price_sequence', 'volume_sequence', 'temporal_position',
            'attention_mask', 'sequence_length'
        ]
        for feature in expected_features:
            assert feature in features
    
    @pytest.mark.asyncio
    async def test_train_model_signature(self, mock_transformer):
        """Test train_model method signature"""
        import pandas as pd
        
        # Create mock training data
        training_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
            'price': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.exponential(1000, 1000),
        })
        
        # Mock the actual training implementation
        with patch.object(mock_transformer, '_train_transformer') as mock_train:
            mock_train.return_value = True
            
            success = await mock_transformer.train_model(training_data)
            assert isinstance(success, bool)
            mock_train.assert_called_once_with(training_data)
    
    def test_health_check_method(self, mock_transformer):
        """Test health check functionality"""
        # Initially unhealthy (not trained)
        health_future = mock_transformer.health_check()
        assert hasattr(health_future, '__await__')  # Should be async
        
        # Set as trained
        mock_transformer._is_trained = True
        mock_transformer._model = mock_transformer
        
        # Should now be healthy
        health_future = mock_transformer.health_check()
        assert hasattr(health_future, '__await__')
    
    def test_edge_case_empty_sequence(self, mock_transformer):
        """Test handling of edge case: empty sequence"""
        # Empty sequence should be handled gracefully
        x = torch.empty(1, 0, 256)  # Empty sequence
        
        with pytest.raises(ValueError, match="Empty sequence not supported"):
            mock_transformer.forward(x)
    
    def test_edge_case_single_timestep(self, mock_transformer):
        """Test handling of single timestep sequence"""
        x = torch.randn(1, 1, 256)  # Single timestep
        
        output = mock_transformer.forward(x)
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()
    
    def test_batch_processing_capability(self, mock_transformer):
        """Test batch processing for multiple assets (BTC, ETH, SOL)"""
        # Simulate processing multiple assets simultaneously
        batch_size = 3  # BTC, ETH, SOL
        seq_len = 100
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mock_transformer.forward(x)
        assert output.shape == (batch_size, 1)
        
        # Each asset should get independent prediction
        assert not torch.allclose(output[0], output[1])  # Different predictions
    
    def test_inference_time_constraint(self, mock_transformer):
        """Test inference time meets <100ms requirement for real-time trading"""
        import time
        
        x = torch.randn(1, 100, 256)
        
        # Warm up
        _ = mock_transformer.forward(x)
        
        # Time the inference
        start_time = time.time()
        with torch.no_grad():
            _ = mock_transformer.forward(x)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be well under 100ms for production use
        assert inference_time < 100, f"Inference took {inference_time:.2f}ms, exceeds 100ms limit"


class TestTransformerConfigIntegration:
    """Test TransformerConfig integration requirements"""
    
    def test_config_serialization(self):
        """Test configuration can be serialized for model preservation"""
        config = TransformerConfig(d_model=512, n_heads=8)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['d_model'] == 512
        assert config_dict['n_heads'] == 8
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(config_dict)
        assert isinstance(json_str, str)
    
    def test_config_from_dict(self):
        """Test configuration can be restored from dictionary"""
        original_config = TransformerConfig(d_model=512, n_heads=8, dropout=0.2)
        config_dict = original_config.to_dict()
        
        restored_config = TransformerConfig.from_dict(config_dict)
        
        assert restored_config.d_model == original_config.d_model
        assert restored_config.n_heads == original_config.n_heads
        assert restored_config.dropout == original_config.dropout
    
    def test_production_ready_configs(self):
        """Test production-ready configurations for different model types"""
        # iTransformer config
        itransformer_config = TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=4,
            max_seq_length=1000,
            use_flash_attention=True
        )
        assert itransformer_config.estimate_memory_usage_mb() < 2048
        
        # PatchTST config
        patchtst_config = TransformerConfig(
            d_model=256,
            n_heads=4,
            n_layers=3,
            max_seq_length=1000,
            use_gradient_checkpointing=True
        )
        assert patchtst_config.estimate_memory_usage_mb() < 2048
        
        # TimesMixer config
        timesmixer_config = TransformerConfig(
            d_model=384,
            n_heads=6,
            n_layers=4,
            max_seq_length=1000
        )
        assert timesmixer_config.estimate_memory_usage_mb() < 2048