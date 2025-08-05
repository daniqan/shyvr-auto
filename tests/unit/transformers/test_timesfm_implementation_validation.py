"""
Implementation validation tests for TimesFM Phase 2.2.2
Tests the actual working implementation of real model loading and inference
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Import the enhanced TimesFM wrapper
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.ml_analysis.transformers.timesfm_wrapper import (
    TimesFMWrapper, TimesFMConfig, TimesFMRealModel, KVCache,
    CryptoTimesFMTokenizer, GCPTimesFMOptimizer
)


class TestTimesFMImplementationValidation:
    """Validate the actual TimesFM implementation works"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config_dict = {
            'model_name': 'google/timesfm-1.0-200m',
            'prediction_length': 24,
            'context_length': 128,  # Smaller for testing
            'batch_size': 4,
            'use_zero_shot': True,
            'memory_efficient': True,
            'gcp_optimized': True,
            'streaming_enabled': True,
            'enable_fallback': True
        }
        
        # Sample time series data
        np.random.seed(42)  # For reproducibility
        self.time_series_1d = np.random.randn(100).cumsum() + 50000
        self.time_series_2d = np.random.randn(100, 4).cumsum(axis=0) + np.array([50000, 1000, 50100, 49900])
        
        self.multivariate_data = {
            'BTC': self.time_series_2d[:, 0],
            'ETH': self.time_series_2d[:, 1] * 0.05,
            'SOL': self.time_series_2d[:, 2] * 0.001
        }
    
    def test_timesfm_config_validation(self):
        """Test TimesFM configuration validation"""
        config = TimesFMConfig.from_dict(self.config_dict)
        
        assert config.model_name == 'google/timesfm-1.0-200m'
        assert config.prediction_length == 24
        assert config.context_length == 128
        assert config.use_zero_shot == True
        assert config.memory_efficient == True
        
        # Test memory estimation
        memory_usage = config.estimate_memory_usage()
        assert memory_usage > 0
        assert isinstance(memory_usage, float)
    
    def test_timesfm_real_model_creation(self):
        """Test TimesFMRealModel can be created"""
        from transformers import PretrainedConfig
        
        config = PretrainedConfig(
            vocab_size=10000,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,  # Smaller for testing
            intermediate_size=3072,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        model = TimesFMRealModel(config)
        
        assert model is not None
        assert hasattr(model, 'prediction_heads')
        assert 'h1' in model.prediction_heads
        assert 'h24' in model.prediction_heads
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (1, 10))
        output = model(dummy_input)
        
        assert 'predictions' in output
        assert 'confidence' in output
        assert output['predictions'].shape[0] == 1  # batch size
    
    def test_kv_cache_functionality(self):
        """Test KV cache implementation"""
        cache = KVCache(max_length=100, hidden_size=64)
        
        # Test cache operations
        key = torch.randn(1, 64)
        value = torch.randn(1, 64)
        
        cache.update(key, value)
        assert cache.current_length == 1
        
        # Test retrieval
        cached_keys, cached_values = cache.get_cached_kv()
        assert cached_keys is not None
        assert cached_values is not None
        assert cached_keys.shape == (1, 64)
        
        # Test cache clearing
        cache.clear()
        assert cache.current_length == 0
        assert len(cache.cache) == 0
    
    def test_timesfm_wrapper_initialization(self):
        """Test TimesFM wrapper initializes correctly"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        assert wrapper is not None
        assert wrapper.timesfm_config is not None
        assert wrapper.tokenizer is not None
        assert wrapper.gcp_optimizer is not None
        
        # Test basic properties
        assert hasattr(wrapper, '_kv_cache')
        assert hasattr(wrapper, '_real_model')
        assert hasattr(wrapper, '_compiled_model')
        assert hasattr(wrapper, '_performance_metrics')
    
    def test_zero_shot_prediction_basic(self):
        """Test basic zero-shot prediction functionality"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Test with 1D time series
        predictions = wrapper.zero_shot_predict(self.time_series_1d[:50], horizon=5)
        
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_batch_inference_functionality(self):
        """Test batch inference works"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Create batch data
        batch_data = [
            self.time_series_1d[:30],
            self.time_series_1d[10:40],
            self.time_series_1d[20:50]
        ]
        
        predictions = wrapper.batch_inference(batch_data)
        
        assert predictions is not None
        assert len(predictions) == 3
        assert all(isinstance(pred, np.ndarray) for pred in predictions)
        assert all(len(pred) == wrapper.timesfm_config.prediction_length for pred in predictions)
    
    def test_multivariate_prediction(self):
        """Test multivariate prediction functionality"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        predictions = wrapper.predict_multivariate(self.multivariate_data, horizon=12)
        
        assert predictions is not None
        assert isinstance(predictions, dict)
        assert 'BTC' in predictions
        assert 'ETH' in predictions
        assert 'SOL' in predictions
        
        for asset, pred in predictions.items():
            assert isinstance(pred, np.ndarray)
            assert len(pred) == 12
    
    def test_streaming_functionality(self):
        """Test streaming prediction functionality"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Initialize streaming
        context_data = self.time_series_1d[:50]
        wrapper.init_streaming(context_data)
        
        assert wrapper._streaming_initialized == True
        assert wrapper._streaming_context is not None
        
        # Update stream with new point
        new_point = np.array([50500.0])
        prediction = wrapper.update_stream(new_point)
        
        assert prediction is not None
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) > 0
    
    def test_ensemble_compatibility(self):
        """Test ensemble system compatibility"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        ensemble_pred = wrapper.get_ensemble_predictions(self.time_series_1d[:50])
        
        assert isinstance(ensemble_pred, dict)
        assert 'price_1h' in ensemble_pred
        assert 'price_4h' in ensemble_pred
        assert 'price_24h' in ensemble_pred
        
        # All predictions should be numeric
        for key, value in ensemble_pred.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_confidence_estimation(self):
        """Test prediction confidence estimation"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        confidence = wrapper.get_prediction_confidence(self.time_series_1d[:50])
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_attention_weights_extraction(self):
        """Test attention weights can be extracted"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        attention_weights = wrapper.get_attention_weights()
        
        assert attention_weights is not None
        assert isinstance(attention_weights, torch.Tensor)
        assert len(attention_weights.shape) >= 2  # Should have at least batch and sequence dimensions
    
    def test_model_info_comprehensive(self):
        """Test comprehensive model information"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        model_info = wrapper.get_model_info()
        
        assert isinstance(model_info, dict)
        
        # Phase 2.2.2 enhanced info
        expected_keys = [
            'model_type', 'timesfm_version', 'parameter_count',
            'real_model_available', 'compiled_model_available',
            'kv_cache_enabled', 'streaming_enabled', 'gcp_optimized',
            'multi_horizon_heads', 'available_horizons', 'version_info'
        ]
        
        for key in expected_keys:
            assert key in model_info, f"Missing key: {key}"
    
    def test_health_check_functionality(self):
        """Test health check works"""
        import asyncio
        
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Run health check
        health_status = asyncio.run(wrapper.health_check())
        
        # Should return boolean for basic health check
        assert isinstance(health_status, bool)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        memory_usage = wrapper.get_memory_usage()
        
        assert isinstance(memory_usage, float)
        assert memory_usage > 0
    
    def test_compilation_status(self):
        """Test model compilation status tracking"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        status = wrapper.get_compilation_status()
        
        assert isinstance(status, dict)
        
        expected_keys = [
            'compiled_model_available', 'quantized_model_available',
            'kv_cache_enabled', 'autocast_enabled', 'model_warm_up_completed',
            'batch_processor_ready', 'prediction_cache_enabled'
        ]
        
        for key in expected_keys:
            assert key in status
    
    def test_required_features(self):
        """Test required features list"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        features = wrapper.get_required_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Phase 2.2.2 enhanced features
        expected_features = ['close', 'volume', 'timestamp', 'market_regime', 'volatility']
        
        for feature in expected_features:
            assert feature in features
    
    def test_data_cleaning_functionality(self):
        """Test data cleaning handles NaN/Inf values"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Create data with NaN and Inf values
        dirty_data = self.time_series_1d[:50].copy()
        dirty_data[10] = np.nan
        dirty_data[20] = np.inf
        dirty_data[30] = -np.inf
        
        # Should not crash and should return valid predictions
        predictions = wrapper.zero_shot_predict(dirty_data, horizon=5)
        
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_performance_tracking(self):
        """Test performance metrics are tracked"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Make some predictions to generate performance data
        wrapper.zero_shot_predict(self.time_series_1d[:30], horizon=5)
        wrapper.zero_shot_predict(self.time_series_1d[10:40], horizon=5)
        
        # Check performance metrics
        assert hasattr(wrapper, '_performance_metrics')
        assert 'inference_times' in wrapper._performance_metrics
        assert len(wrapper._performance_metrics['inference_times']) >= 2
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        # Test with empty data (should trigger error recovery)
        empty_data = np.array([])
        
        try:
            predictions = wrapper.zero_shot_predict(empty_data, horizon=5)
            # Should either work with fallback or raise a proper error
            if predictions is not None:
                assert isinstance(predictions, np.ndarray)
        except (ValueError, RuntimeError) as e:
            # Expected error for invalid input
            assert "Empty" in str(e) or "Invalid" in str(e)
    
    def test_market_regime_adaptation(self):
        """Test market regime adaptive functionality"""
        wrapper = TimesFMWrapper(self.config_dict)
        
        market_conditions = {
            'regime': 'volatile',
            'volatility': 0.8,
            'trend_strength': 0.3
        }
        
        model_weight = wrapper.get_model_weight(market_conditions)
        
        assert isinstance(model_weight, float)
        assert 0.0 <= model_weight <= 1.0
    
    def test_tokenizer_functionality(self):
        """Test crypto-specific tokenizer"""
        tokenizer = CryptoTimesFMTokenizer()
        
        # Test price tokenization
        prices = [50000, 50100, 49950, 50200]
        tokens = tokenizer.tokenize_prices(prices)
        
        assert isinstance(tokens, torch.Tensor)
        assert tokens.dtype == torch.long
        assert len(tokens) == len(prices)
    
    def test_gcp_optimizer_functionality(self):
        """Test GCP optimizer"""
        optimizer = GCPTimesFMOptimizer(batch_size=4, memory_efficient=True)
        
        # Test batch creation
        data_list = [np.random.randn(50) for _ in range(10)]
        batches = optimizer.create_batches(data_list)
        
        assert len(batches) > 0
        assert all(len(batch) <= 4 for batch in batches)
        
        # Test memory estimation
        memory_usage = optimizer.estimate_memory_usage(128, 4)
        assert isinstance(memory_usage, float)
        assert memory_usage > 0


if __name__ == "__main__":
    # Run the validation tests
    pytest.main([__file__, "-v", "--tb=short"])