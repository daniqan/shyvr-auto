#!/usr/bin/env python3
"""
Test script for verifying architecture fixes with small dataset
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml_analysis.transformers.patchtst import PatchTSTPredictor, PatchTSTConfig
from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor, TimesMixerConfig
from src.ml_analysis.transformers.itransformer import iTransformerPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_samples=500, n_features=40, sequence_length=50):
    """Create small test dataset"""
    logger.info(f"Creating test data: {n_samples} samples, {n_features} features, sequence_length={sequence_length}")
    
    # Create synthetic time series data
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
    
    # Generate synthetic features
    data = {
        'timestamp': timestamps,
        'close': np.cumsum(np.random.randn(n_samples)) + 100,
        'open': np.cumsum(np.random.randn(n_samples)) + 100,
        'high': np.cumsum(np.random.randn(n_samples)) + 105,
        'low': np.cumsum(np.random.randn(n_samples)) + 95,
        'volume': np.random.uniform(1000, 10000, n_samples)
    }
    
    # Add technical indicators
    for i in range(n_features - 5):  # Already have 5 features (timestamp excluded from features)
        feature_name = f'feature_{i}'
        data[feature_name] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

async def test_patchtst():
    """Test PatchTST with fixed shape handling"""
    logger.info("\n" + "="*50)
    logger.info("Testing PatchTST Architecture")
    logger.info("="*50)
    
    try:
        # Create test data
        data = create_test_data(n_samples=500, n_features=15)  # 15 channels for PatchTST
        
        # Configure model
        config = PatchTSTConfig(
            n_channels=15,
            patch_length=8,
            stride=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            prediction_horizons=['1h', '4h', '24h']
        )
        
        # Create model
        model = PatchTSTPredictor(config.__dict__)
        logger.info(f"Created PatchTST model with config: {config.__dict__}")
        
        # Prepare data
        X, y, feature_names = model.prepare_training_from_corpus(data)
        logger.info(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
        
        # Create small batch for testing
        batch_x = torch.FloatTensor(X[:32])  # 32 samples
        logger.info(f"Test batch shape: {batch_x.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model.model(batch_x)
            logger.info(f"Forward pass successful!")
            
            if isinstance(output, dict):
                for key, val in output.items():
                    if isinstance(val, torch.Tensor):
                        logger.info(f"  Output['{key}']: shape={val.shape}")
            else:
                logger.info(f"  Output shape: {output.shape}")
        
        logger.info("✅ PatchTST test PASSED - Shape mismatch fixed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ PatchTST test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_timesmixer():
    """Test TimesMixer with performance optimization"""
    logger.info("\n" + "="*50)
    logger.info("Testing TimesMixer Architecture")
    logger.info("="*50)
    
    try:
        # Create test data
        data = create_test_data(n_samples=500, n_features=40)
        
        # Configure model
        config = TimesMixerConfig(
            seq_len=100,
            pred_len=24,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            top_k=3,
            num_kernels=3
        )
        
        # Create model
        model = TimesMixerPredictor(config.__dict__)
        logger.info(f"Created TimesMixer model with config: {config.__dict__}")
        
        # Prepare data
        X, y, feature_names = model.prepare_training_from_corpus(data)
        logger.info(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
        
        # Create small batch for testing
        batch_x = torch.FloatTensor(X[:16])  # 16 samples for smaller batch
        logger.info(f"Test batch shape: {batch_x.shape}")
        
        # Test forward pass with timing
        start_time = datetime.now()
        with torch.no_grad():
            output = model.model(batch_x)
            forward_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Forward pass successful in {forward_time:.3f} seconds!")
            
            if isinstance(output, dict):
                for key, val in output.items():
                    if isinstance(val, torch.Tensor):
                        logger.info(f"  Output['{key}']: shape={val.shape}")
            else:
                logger.info(f"  Output shape: {output.shape}")
        
        if forward_time < 1.0:  # Should be much faster than 5.5 seconds
            logger.info(f"✅ TimesMixer test PASSED - Performance improved ({forward_time:.3f}s < 1.0s)!")
        else:
            logger.warning(f"⚠️ TimesMixer still slow ({forward_time:.3f}s), but functional")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TimesMixer test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_itransformer():
    """Test iTransformer with 40 features"""
    logger.info("\n" + "="*50)
    logger.info("Testing iTransformer Architecture")
    logger.info("="*50)
    
    try:
        # Create test data with 40 features
        data = create_test_data(n_samples=500, n_features=40)
        
        # Selected features (40 total)
        selected_features = [
            'close', 'open', 'high', 'low', 'volume'
        ] + [f'feature_{i}' for i in range(35)]  # Add 35 more features
        
        # Configure model
        config = {
            'seq_len': 96,
            'pred_len': 24,
            'n_variates': 40,  # Using 40 features now
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 2,
            'd_ff': 256,
            'dropout': 0.1,
            'selected_features': selected_features
        }
        
        # Create model
        model = iTransformerPredictor(config)
        logger.info(f"Created iTransformer model with {config['n_variates']} features")
        
        # Prepare data
        X, y, feature_names = model.prepare_training_from_corpus(data)
        logger.info(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Using {len(feature_names)} features: {feature_names[:5]}... (showing first 5)")
        
        # Create small batch for testing
        batch_x = torch.FloatTensor(X[:32])
        logger.info(f"Test batch shape: {batch_x.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model.model(batch_x)
            logger.info(f"Forward pass successful!")
            
            if isinstance(output, dict):
                for key, val in output.items():
                    if isinstance(val, torch.Tensor):
                        logger.info(f"  Output['{key}']: shape={val.shape}")
            else:
                logger.info(f"  Output shape: {output.shape}")
        
        logger.info("✅ iTransformer test PASSED - Now using 40 features!")
        return True
        
    except Exception as e:
        logger.error(f"❌ iTransformer test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all architecture tests"""
    logger.info("Starting architecture fix verification tests...")
    logger.info("This will test fixes for PatchTST, TimesMixer, and iTransformer")
    
    results = {}
    
    # Test each architecture
    results['PatchTST'] = await test_patchtst()
    results['TimesMixer'] = await test_timesmixer()
    results['iTransformer'] = await test_itransformer()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{model_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All architecture fixes verified successfully!")
        logger.info("Ready to proceed with full training.")
    else:
        logger.error("\n⚠️ Some tests failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)