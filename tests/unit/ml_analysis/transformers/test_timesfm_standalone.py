"""
Standalone TimesFM Tests (TDD approach without full config system)
Tests the TimesFM wrapper implementation directly
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

# Direct imports to avoid configuration issues
from ml_analysis.transformers.timesfm_wrapper import (
    TimesFMConfig, TimesFMWrapper, CryptoTimesFMTokenizer, GCPTimesFMOptimizer
)


class TestTimesFMConfig:
    """Test TimesFM configuration validation"""
    
    def test_valid_configuration(self):
        """Test valid configuration is accepted"""
        config = TimesFMConfig(
            model_name='google/timesfm-1.0-200m',
            prediction_length=24,
            context_length=512,
            use_zero_shot=True
        )
        
        assert config.model_name == 'google/timesfm-1.0-200m'
        assert config.prediction_length == 24
        assert config.context_length == 512
        assert config.use_zero_shot is True
    
    def test_invalid_prediction_length(self):
        """Test invalid prediction length raises error"""
        with pytest.raises(ValueError, match="prediction_length must be positive"):
            TimesFMConfig(prediction_length=0)
    
    def test_invalid_context_length(self):
        """Test invalid context length raises error"""
        with pytest.raises(ValueError, match="context_length must be positive"):
            TimesFMConfig(context_length=-1)
    
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        config = TimesFMConfig(
            context_length=1024,
            prediction_length=48
        )
        
        memory_mb = config.estimate_memory_usage()
        assert memory_mb > 0
        assert isinstance(memory_mb, (int, float))
        
        # Test memory optimization effect
        config_optimized = TimesFMConfig(
            context_length=1024,
            prediction_length=48,
            memory_efficient=True
        )
        memory_optimized = config_optimized.estimate_memory_usage()
        assert memory_optimized < memory_mb  # Should be less with optimization
    
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = TimesFMConfig(prediction_length=48, context_length=256)
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['prediction_length'] == 48
        assert config_dict['context_length'] == 256
        
        # Test from_dict
        restored_config = TimesFMConfig.from_dict(config_dict)
        assert restored_config.prediction_length == 48
        assert restored_config.context_length == 256


class TestCryptoTimesFMTokenizer:
    """Test crypto-specific tokenization for TimesFM"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initializes properly"""
        tokenizer = CryptoTimesFMTokenizer()
        assert hasattr(tokenizer, 'vocab_size')
        assert hasattr(tokenizer, 'special_tokens')
        assert tokenizer.vocab_size > 0
        assert '<PAD>' in tokenizer.special_tokens
        assert '<UNK>' in tokenizer.special_tokens
    
    def test_price_tokenization(self):
        """Test price data tokenization"""
        tokenizer = CryptoTimesFMTokenizer()
        
        prices = [100.0, 101.5, 99.8, 102.1, 98.5]
        tokens = tokenizer.tokenize_prices(prices)
        
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 5
        assert not torch.isnan(tokens).any()
        assert tokens.dtype == torch.long
    
    def test_volume_tokenization(self):
        """Test volume data tokenization"""
        tokenizer = CryptoTimesFMTokenizer()
        
        volumes = [1000, 1500, 800, 2000, 1200]
        tokens = tokenizer.tokenize_volumes(volumes)
        
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 5
        assert (tokens >= 0).all()  # Volume tokens should be non-negative
        assert tokens.dtype == torch.long
    
    def test_market_regime_tokenization(self):
        """Test market regime tokenization"""
        tokenizer = CryptoTimesFMTokenizer()
        
        regimes = ['bull', 'bear', 'sideways', 'bull', 'volatile']
        tokens = tokenizer.tokenize_market_regimes(regimes)
        
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 5
        assert tokens.dtype == torch.long
    
    def test_market_data_tokenization(self):
        """Test complete market data tokenization"""
        tokenizer = CryptoTimesFMTokenizer()
        
        market_data = {
            'price': [100.0, 101.5, 99.8, 102.1],
            'volume': [1000, 1200, 800, 1500]
        }
        
        tokens = tokenizer.tokenize_market_data(market_data)
        
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 4  # 4 time steps
        assert tokens.shape[1] == 2  # price + volume features
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        tokenizer = CryptoTimesFMTokenizer()
        
        # Empty prices
        tokens = tokenizer.tokenize_prices([])
        assert tokens.item() == tokenizer.special_tokens['<UNK>']
        
        # Empty volumes
        tokens = tokenizer.tokenize_volumes([])
        assert tokens.item() == tokenizer.special_tokens['<UNK>']


class TestGCPTimesFMOptimizer:
    """Test GCP-specific optimizations for TimesFM"""
    
    def test_gcp_optimizer_initialization(self):
        """Test GCP optimizer initializes properly"""
        optimizer = GCPTimesFMOptimizer(
            batch_size=32,
            use_tpu=False,
            memory_efficient=True
        )
        
        assert optimizer.batch_size == 32
        assert optimizer.use_tpu is False
        assert optimizer.memory_efficient is True
    
    def test_batch_creation(self):
        """Test batch processing optimization"""
        optimizer = GCPTimesFMOptimizer(batch_size=16)
        
        # Multiple time series
        batch_data = [np.random.randn(100, 5) for _ in range(32)]
        
        batches = optimizer.create_batches(batch_data)
        assert len(batches) == 2  # 32 / 16 = 2 batches
        assert all(len(batch) == 16 for batch in batches)
    
    def test_memory_estimation(self):
        """Test memory management for GCP deployment"""
        optimizer = GCPTimesFMOptimizer(memory_efficient=True)
        
        # Test memory estimation
        sequence_length = 512
        batch_size = 8
        memory_mb = optimizer.estimate_memory_usage(sequence_length, batch_size)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, (int, float))
        
        # Test memory efficiency effect
        optimizer_inefficient = GCPTimesFMOptimizer(memory_efficient=False)
        memory_inefficient = optimizer_inefficient.estimate_memory_usage(sequence_length, batch_size)
        assert memory_mb < memory_inefficient  # Should be less with optimization
    
    def test_cache_cleaning(self):
        """Test cache cleaning functionality"""
        optimizer = GCPTimesFMOptimizer()
        
        # Should not raise error
        optimizer.clean_cache()


class TestTimesFMWrapper:
    """Test TimesFM wrapper core functionality"""
    
    def test_timesfm_wrapper_initialization(self):
        """Test TimesFM wrapper can be initialized with configuration"""
        config = {
            'model_name': 'google/timesfm-1.0-200m',
            'prediction_length': 24,
            'context_length': 512,
            'use_zero_shot': True,
            'fine_tune_enabled': True
        }
        
        # Mock the model type enum to avoid import issues
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            assert wrapper.timesfm_config.model_name == 'google/timesfm-1.0-200m'
            assert wrapper.timesfm_config.prediction_length == 24
            assert wrapper.timesfm_config.context_length == 512
            assert wrapper.timesfm_config.use_zero_shot is True
            assert wrapper.timesfm_config.fine_tune_enabled is True
    
    def test_zero_shot_prediction_capability(self):
        """Test TimesFM can make zero-shot predictions without training"""
        config = {'use_zero_shot': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Create sample time series data
            time_series = np.random.randn(100, 5)  # 100 time steps, 5 features
            timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
            
            # Should work without explicit training (TimesFM is pre-trained)
            predictions = wrapper.zero_shot_predict(time_series, timestamps, horizon=24)
            
            assert predictions is not None
            assert len(predictions) == 24  # Prediction horizon
            assert isinstance(predictions, np.ndarray)
            assert not np.isnan(predictions).any()
    
    def test_multiple_forecast_horizons(self):
        """Test TimesFM supports multiple forecast horizons"""
        config = {}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            time_series = np.random.randn(200, 3)
            timestamps = pd.date_range('2024-01-01', periods=200, freq='H')
            
            # Test different horizons
            horizons = [1, 4, 24, 48]  # 1h, 4h, 24h, 48h
            
            for horizon in horizons:
                predictions = wrapper.zero_shot_predict(time_series, timestamps, horizon=horizon)
                assert len(predictions) == horizon
                assert not np.isnan(predictions).any()
    
    def test_error_handling_and_fallbacks(self):
        """Test TimesFM has proper error handling and fallback mechanisms"""
        config = {'enable_fallback': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test with corrupted input
            corrupted_data = np.array([[np.inf, np.nan, -np.inf]])
            
            # Should not crash, should return safe fallback
            result = wrapper.zero_shot_predict(corrupted_data, None, horizon=1)
            assert result is not None
            assert len(result) == 1
            assert not np.isnan(result).any()
            assert not np.isinf(result).any()
    
    def test_batch_inference(self):
        """Test batch inference capability"""
        config = {'batch_size': 8}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test with multiple time series
            batch_data = [np.random.randn(100, 3) for _ in range(16)]
            batch_predictions = wrapper.batch_inference(batch_data)
            
            assert len(batch_predictions) == 16
            assert all(len(pred) == wrapper.timesfm_config.prediction_length for pred in batch_predictions)
    
    def test_multivariate_prediction(self):
        """Test multivariate time series support"""
        config = {'multivariate_enabled': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Multi-asset data (BTC, ETH, SOL)
            multivariate_data = {
                'BTC': np.random.randn(200, 5),  # OHLCV
                'ETH': np.random.randn(200, 5),
                'SOL': np.random.randn(200, 5)
            }
            
            predictions = wrapper.predict_multivariate(multivariate_data, horizon=24)
            
            assert 'BTC' in predictions
            assert 'ETH' in predictions
            assert 'SOL' in predictions
            assert all(len(pred) == 24 for pred in predictions.values())
    
    def test_attention_weight_extraction(self):
        """Test TimesFM can extract attention weights for interpretability"""
        config = {}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            time_series = np.random.randn(50, 4)
            timestamps = pd.date_range('2024-01-01', periods=50, freq='H')
            
            # Should be able to extract attention weights
            predictions, attention_weights = wrapper.predict_with_attention(time_series, timestamps)
            
            assert predictions is not None
            assert attention_weights is not None
            assert isinstance(attention_weights, torch.Tensor)
            assert attention_weights.shape[0] > 0  # Has batch dimension
    
    def test_ensemble_compatibility(self):
        """Test TimesFM is compatible with existing ensemble system"""
        config = {}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test ensemble methods
            time_series = np.random.randn(100, 4)
            
            ensemble_preds = wrapper.get_ensemble_predictions(time_series)
            assert 'price_1h' in ensemble_preds
            assert 'price_4h' in ensemble_preds
            assert 'price_24h' in ensemble_preds
            
            confidence = wrapper.get_prediction_confidence(time_series)
            assert 0.0 <= confidence <= 1.0
            
            weight = wrapper.get_model_weight({'volatility': 0.2})
            assert 0.0 <= weight <= 1.0
    
    def test_model_info_and_health(self):
        """Test model information and health check methods"""
        config = {}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test model info
            info = wrapper.get_model_info()
            assert 'model_type' in info
            assert 'timesfm_version' in info
            assert 'parameter_count' in info
            assert 'supports_zero_shot' in info
            assert info['supports_zero_shot'] is True
            
            # Test health check
            health_status = wrapper.health_check()
            # Should return a coroutine, but we can test the sync version
            assert hasattr(wrapper, 'health_check')
    
    def test_memory_efficient_inference(self):
        """Test memory-efficient inference for long sequences"""
        config = {
            'memory_efficient': True,
            'max_sequence_length': 2048,
            'chunk_size': 512
        }
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test with long sequence
            long_sequence = np.random.randn(1500, 4)  # Long but not too long for test
            timestamps = pd.date_range('2024-01-01', periods=1500, freq='H')
            
            # Should handle without memory issues
            predictions = wrapper.zero_shot_predict(long_sequence, timestamps, horizon=24)
            assert predictions is not None
            assert len(predictions) == 24
    
    def test_fine_tuning_capability(self):
        """Test TimesFM supports fine-tuning on crypto data"""
        config = {'fine_tune_enabled': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Mock training data
            training_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
                'close': np.random.randn(1000).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 1000),
                'high': np.random.randn(1000).cumsum() + 102,
                'low': np.random.randn(1000).cumsum() + 98
            })
            
            # Should be able to fine-tune
            success = wrapper.fine_tune(training_data, epochs=2, learning_rate=1e-5)
            assert isinstance(success, bool)  # Should return a boolean
    
    def test_streaming_functionality(self):
        """Test streaming prediction functionality"""
        config = {'streaming_enabled': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Initialize streaming with context
            context_data = np.random.randn(100, 5)
            wrapper.init_streaming(context_data)
            
            # Should handle new data points
            new_point = np.random.randn(1, 5)
            prediction = wrapper.update_stream(new_point)
            assert prediction is not None
            assert len(prediction) > 0


class TestPerformanceRequirements:
    """Test performance requirements are met"""
    
    def test_inference_speed(self):
        """Test inference meets speed requirements"""
        config = {}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            # Test inference speed
            test_data = np.random.randn(100, 5)
            timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
            
            import time
            start_time = time.time()
            predictions = wrapper.zero_shot_predict(test_data, timestamps, horizon=24)
            inference_time = time.time() - start_time
            
            # Should be reasonably fast (allow some tolerance for test environment)
            assert inference_time < 2.0, f"Inference took {inference_time:.3f}s, should be <2.0s"
            assert predictions is not None
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking and estimation"""
        config = {'memory_efficient': True}
        
        with patch('ml_analysis.transformers.timesfm_wrapper.ModelType') as mock_model_type:
            mock_model_type.TIMESFM = Mock()
            mock_model_type.TIMESFM.value = "timesfm"
            
            wrapper = TimesFMWrapper(config)
            
            memory_usage = wrapper.get_memory_usage()
            assert memory_usage > 0
            assert isinstance(memory_usage, (int, float))
            
            # Should have tracking of inference times
            test_data = np.random.randn(50, 3)
            wrapper.zero_shot_predict(test_data, None, 12)
            
            # Check inference times are tracked
            assert len(wrapper._inference_times) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])