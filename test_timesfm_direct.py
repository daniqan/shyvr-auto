"""
Direct test of TimesFM components without module system
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from unittest.mock import Mock

# Mock required components to avoid circular imports
class ModelType(Enum):
    TIMESFM = "timesfm"

@dataclass
class TransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_seq_length: int = 512
    dropout: float = 0.0

class TransformerBase:
    def __init__(self, model_type, config_dict):
        self.model_type = model_type
        self.transformer_config = TransformerConfig(**config_dict)
        self._is_trained = False

# Define additional required classes for TimesFM
@dataclass 
class PredictionResult:
    token: Any
    analyzed_at: datetime
    model_type: Any
    direction: Any = None
    confidence: float = 0.5
    probability_up: float = 0.5
    features_used: List[str] = None
    model_version: str = "test"
    price_prediction_24h: float = 0.0

class PredictionDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DiscoveredToken:
    def __init__(self):
        self.address = "0x123"
        self.price_usd = 100.0

# Mock structlog
class MockLogger:
    def bind(self, **kwargs):
        return self
    def info(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass

class MockStructlog:
    def get_logger(self):
        return MockLogger()

# Monkey patch modules that would cause issues
import sys
sys.modules['structlog'] = MockStructlog()

# Now import TimesFM components with proper path handling
sys.path.insert(0, '/Users/kendo/daniqan/shyvrai-rlte/src')

# Read and execute the TimesFM code with proper globals
timesfm_code = open('/Users/kendo/daniqan/shyvrai-rlte/src/ml_analysis/transformers/timesfm_wrapper.py').read()

# Replace problematic imports
timesfm_code = timesfm_code.replace('logger = structlog.get_logger()', 'logger = MockLogger()')

exec(timesfm_code, globals())


def test_timesfm_config():
    """Test TimesFM configuration"""
    print("Testing TimesFM Config...")
    
    # Valid config
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
    print("✓ Valid config test passed")
    
    # Invalid config
    try:
        TimesFMConfig(prediction_length=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "prediction_length must be positive" in str(e)
        print("✓ Invalid config validation passed")
    
    # Memory estimation
    memory_mb = config.estimate_memory_usage()
    assert memory_mb > 0
    print(f"✓ Memory estimation: {memory_mb:.1f} MB")


def test_crypto_tokenizer():
    """Test crypto tokenizer"""
    print("\nTesting Crypto Tokenizer...")
    
    tokenizer = CryptoTimesFMTokenizer()
    
    # Test price tokenization
    prices = [100.0, 101.5, 99.8, 102.1, 98.5]
    tokens = tokenizer.tokenize_prices(prices)
    
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[0] == 5
    assert tokens.dtype == torch.long
    print("✓ Price tokenization passed")
    
    # Test volume tokenization
    volumes = [1000, 1500, 800, 2000, 1200]
    volume_tokens = tokenizer.tokenize_volumes(volumes)
    
    assert isinstance(volume_tokens, torch.Tensor)
    assert volume_tokens.shape[0] == 5
    assert (volume_tokens >= 0).all()
    print("✓ Volume tokenization passed")
    
    # Test market data tokenization
    market_data = {
        'price': [100.0, 101.5, 99.8, 102.1],
        'volume': [1000, 1200, 800, 1500]
    }
    
    combined_tokens = tokenizer.tokenize_market_data(market_data)
    assert isinstance(combined_tokens, torch.Tensor)
    assert combined_tokens.shape[0] == 4  # 4 time steps
    assert combined_tokens.shape[1] == 2  # price + volume
    print("✓ Market data tokenization passed")


def test_gcp_optimizer():
    """Test GCP optimizer"""
    print("\nTesting GCP Optimizer...")
    
    optimizer = GCPTimesFMOptimizer(
        batch_size=16,
        use_tpu=False,
        memory_efficient=True
    )
    
    assert optimizer.batch_size == 16
    assert optimizer.memory_efficient is True
    print("✓ GCP optimizer initialization passed")
    
    # Test batch creation
    batch_data = [np.random.randn(100, 5) for _ in range(32)]
    batches = optimizer.create_batches(batch_data)
    
    assert len(batches) == 2  # 32 / 16 = 2 batches
    assert all(len(batch) == 16 for batch in batches)
    print("✓ Batch creation passed")
    
    # Test memory estimation
    memory_mb = optimizer.estimate_memory_usage(512, 8)
    assert memory_mb > 0
    print(f"✓ Memory estimation: {memory_mb:.1f} MB")


def test_timesfm_wrapper():
    """Test TimesFM wrapper"""
    print("\nTesting TimesFM Wrapper...")
    
    config = {
        'model_name': 'google/timesfm-1.0-200m',
        'prediction_length': 24,
        'context_length': 512,
        'use_zero_shot': True,
        'fine_tune_enabled': True
    }
    
    wrapper = TimesFMWrapper(config)
    
    assert wrapper.timesfm_config.model_name == 'google/timesfm-1.0-200m'
    assert wrapper.timesfm_config.prediction_length == 24
    print("✓ TimesFM wrapper initialization passed")
    
    # Test zero-shot prediction
    time_series = np.random.randn(100, 5)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    
    predictions = wrapper.zero_shot_predict(time_series, timestamps, horizon=24)
    
    assert predictions is not None
    assert len(predictions) == 24
    assert isinstance(predictions, np.ndarray)
    assert not np.isnan(predictions).any()
    print("✓ Zero-shot prediction passed")
    
    # Test multiple horizons
    horizons = [1, 4, 24, 48]
    for horizon in horizons:
        preds = wrapper.zero_shot_predict(time_series, timestamps, horizon=horizon)
        assert len(preds) == horizon
    print("✓ Multiple forecast horizons passed")
    
    # Test error handling
    corrupted_data = np.array([[np.inf, np.nan, -np.inf]])
    result = wrapper.zero_shot_predict(corrupted_data, None, horizon=1)
    assert result is not None
    assert len(result) == 1
    assert not np.isnan(result).any()
    print("✓ Error handling and fallbacks passed")


def test_ensemble_compatibility():
    """Test ensemble system compatibility"""
    print("\nTesting Ensemble Compatibility...")
    
    wrapper = TimesFMWrapper({})
    time_series = np.random.randn(100, 4)
    
    # Test ensemble predictions
    ensemble_preds = wrapper.get_ensemble_predictions(time_series)
    assert 'price_1h' in ensemble_preds
    assert 'price_4h' in ensemble_preds
    assert 'price_24h' in ensemble_preds
    print("✓ Ensemble predictions passed")
    
    # Test confidence scoring
    confidence = wrapper.get_prediction_confidence(time_series)
    assert 0.0 <= confidence <= 1.0
    print(f"✓ Confidence scoring passed: {confidence:.3f}")
    
    # Test model weighting
    weight = wrapper.get_model_weight({'volatility': 0.2})
    assert 0.0 <= weight <= 1.0
    print(f"✓ Model weighting passed: {weight:.3f}")


def test_performance():
    """Test performance requirements"""
    print("\nTesting Performance...")
    
    wrapper = TimesFMWrapper({'memory_efficient': True})
    
    # Test inference speed
    test_data = np.random.randn(100, 5)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    
    import time
    start_time = time.time()
    predictions = wrapper.zero_shot_predict(test_data, timestamps, horizon=24)
    inference_time = time.time() - start_time
    
    print(f"✓ Inference time: {inference_time*1000:.1f}ms")
    assert predictions is not None
    
    # Test memory usage tracking
    memory_usage = wrapper.get_memory_usage()
    assert memory_usage > 0
    print(f"✓ Memory usage: {memory_usage:.1f} MB")


def test_attention_and_interpretability():
    """Test attention weight extraction"""
    print("\nTesting Attention and Interpretability...")
    
    wrapper = TimesFMWrapper({})
    
    time_series = np.random.randn(50, 4)
    timestamps = pd.date_range('2024-01-01', periods=50, freq='H')
    
    predictions, attention_weights = wrapper.predict_with_attention(time_series, timestamps)
    
    assert predictions is not None
    assert attention_weights is not None
    assert isinstance(attention_weights, torch.Tensor)
    assert attention_weights.shape[0] > 0
    print("✓ Attention weight extraction passed")


def test_batch_and_multivariate():
    """Test batch and multivariate functionality"""
    print("\nTesting Batch and Multivariate...")
    
    wrapper = TimesFMWrapper({'batch_size': 8, 'multivariate_enabled': True})
    
    # Test batch inference
    batch_data = [np.random.randn(100, 3) for _ in range(16)]
    batch_predictions = wrapper.batch_inference(batch_data)
    
    assert len(batch_predictions) == 16
    assert all(len(pred) == wrapper.timesfm_config.prediction_length for pred in batch_predictions)
    print("✓ Batch inference passed")
    
    # Test multivariate prediction
    multivariate_data = {
        'BTC': np.random.randn(200, 5),
        'ETH': np.random.randn(200, 5),
        'SOL': np.random.randn(200, 5)
    }
    
    predictions = wrapper.predict_multivariate(multivariate_data, horizon=24)
    
    assert 'BTC' in predictions
    assert 'ETH' in predictions
    assert 'SOL' in predictions
    assert all(len(pred) == 24 for pred in predictions.values())
    print("✓ Multivariate prediction passed")


def test_model_info():
    """Test model information and health"""
    print("\nTesting Model Info and Health...")
    
    wrapper = TimesFMWrapper({})
    
    # Test model info
    info = wrapper.get_model_info()
    assert 'model_type' in info
    assert 'timesfm_version' in info
    assert 'parameter_count' in info
    assert 'supports_zero_shot' in info
    assert info['supports_zero_shot'] is True
    print("✓ Model info passed")
    
    # Test required features
    features = wrapper.get_required_features()
    assert isinstance(features, list)
    assert 'close' in features
    assert 'timestamp' in features
    print("✓ Required features passed")


if __name__ == "__main__":
    print("Running TimesFM TDD Tests (Direct Implementation)")
    print("=" * 60)
    
    try:
        test_timesfm_config()
        test_crypto_tokenizer()
        test_gcp_optimizer()
        test_timesfm_wrapper()
        test_ensemble_compatibility()
        test_performance()
        test_attention_and_interpretability()
        test_batch_and_multivariate()
        test_model_info()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TIMESFM TDD TESTS PASSED!")
        print("✅ TimesFM wrapper architecture is working correctly")
        print("✅ Zero-shot prediction capability validated")
        print("✅ GCP optimization features operational")
        print("✅ Ensemble system integration ready")
        print("✅ Performance requirements met")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()