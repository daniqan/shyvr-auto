"""
Minimal standalone test for TimesFM Zero-Shot Pipeline - Phase 2.2.3
Avoiding configuration dependencies for clean TDD testing
"""

import pytest
import numpy as np
import os
import sys

# Test standalone import without configuration dependencies
def test_timesfm_zeroshot_pipeline_module_missing():
    """Test that the zero-shot pipeline module doesn't exist yet (TDD principle)"""
    with pytest.raises(ImportError):
        # This should fail until we create the module
        import src.ml_analysis.transformers.timesfm_zeroshot_pipeline


def test_timesfm_classes_missing():
    """Test that required zero-shot classes don't exist yet"""
    
    # Should fail - TimesFMZeroShotPipeline doesn't exist
    with pytest.raises((ImportError, AttributeError)):
        from src.ml_analysis.transformers.timesfm_zeroshot_pipeline import TimesFMZeroShotPipeline
    
    # Should fail - StreamingZeroShotPredictor doesn't exist  
    with pytest.raises((ImportError, AttributeError)):
        from src.ml_analysis.transformers.timesfm_zeroshot_pipeline import StreamingZeroShotPredictor
    
    # Should fail - RLTEZeroShotIntegrator doesn't exist
    with pytest.raises((ImportError, AttributeError)):
        from src.ml_analysis.transformers.timesfm_zeroshot_pipeline import RLTEZeroShotIntegrator


def test_sample_data_generation():
    """Test that our test data generation works (this should pass)"""
    def generate_sample_price_data(symbol: str, length: int = 100) -> np.ndarray:
        """Generate sample price data"""
        np.random.seed(42)
        base_price = 50000 if symbol == 'BTC' else 3000
        returns = np.random.normal(0, 0.02, length)
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        return prices.astype(np.float32)
    
    btc_data = generate_sample_price_data('BTC', 100)
    assert len(btc_data) == 100
    assert btc_data.dtype == np.float32
    assert btc_data[0] > 0
    assert not np.any(np.isnan(btc_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])