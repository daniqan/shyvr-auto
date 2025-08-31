"""
Debug script to isolate and fix the iTransformer tensor dimension mismatch issue
"""
import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import required modules
from src.ml_analysis.transformers.itransformer import iTransformerPredictor, InvertedAttentionConfig

def create_dummy_corpus_data(n_samples: int = 500, n_features: int = 25) -> pd.DataFrame:
    """Create dummy corpus data that mimics the real structure"""
    
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    # Core OHLCV features
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)  # Random walk prices
    data = {
        'close': prices,
        'open': prices + np.random.randn(n_samples) * 0.5,
        'high': prices + abs(np.random.randn(n_samples)) * 2,
        'low': prices - abs(np.random.randn(n_samples)) * 2,
        'volume': np.random.lognormal(10, 1, n_samples),
    }
    
    # Add technical indicators (to match corpus structure)
    feature_names = [
        'rsi_14', 'rsi_7', 'rsi_21',
        'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_position',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_score', 'volume_score',
        'market_regime', 'trend_strength',
        'sma_20', 'ema_12', 'atr_normalized',
        'funding_rate', 'correlation_score'
    ]
    
    # Generate random feature data
    for feature in feature_names:
        if 'rsi' in feature:
            data[feature] = np.random.uniform(0, 100, n_samples)
        elif 'bb_position' in feature:
            data[feature] = np.random.uniform(-1, 1, n_samples)
        elif 'regime' in feature or 'score' in feature:
            data[feature] = np.random.uniform(0, 1, n_samples)
        else:
            data[feature] = np.random.randn(n_samples)
    
    # Add metadata columns that would be filtered out
    data.update({
        'timestamp': timestamps,
        'token_id': 'WBTC',
        'feature_version': '1.0',
        'data_source': 'debug'
    })
    
    df = pd.DataFrame(data)
    print(f"Created dummy corpus with {len(df)} samples and {len(df.columns)} columns")
    print(f"Feature columns: {[c for c in df.columns if c not in ['timestamp', 'token_id', 'feature_version', 'data_source']][:10]}...")
    
    return df

def test_itransformer_tensor_mismatch():
    """Test the specific tensor mismatch issue"""
    
    print("=" * 60)
    print("TESTING iTransformer TENSOR DIMENSION MISMATCH")
    print("=" * 60)
    
    # Create dummy data
    corpus_data = create_dummy_corpus_data()
    
    # Test different n_variates configurations
    test_configs = [
        {'n_variates': 20, 'batch_size': 32},  # Current failing config
        {'n_variates': 25, 'batch_size': 32},  # Match actual features
        {'n_variates': 32, 'batch_size': 32},  # Match batch size
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n--- TEST {i+1}: n_variates={test_config['n_variates']}, batch_size={test_config['batch_size']} ---")
        
        try:
            # Initialize iTransformer with test configuration
            config = {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.1,
                'max_seq_length': 96,
                'n_variates': test_config['n_variates'],
                'prediction_horizons': ['1h', '4h', '24h'],
                'cross_variate_attention': True,
                'temporal_fusion_layers': 1
            }
            
            print(f"Initializing iTransformer with config: {config}")
            model = iTransformerPredictor(config)
            
            # Prepare data from corpus
            print(f"Preparing training data from corpus...")
            X, y, feature_names = model.prepare_training_from_corpus(
                corpus_data,
                sequence_length=96,
                prediction_horizons=[1, 4, 24]
            )
            
            print(f"Prepared data shapes: X={X.shape}, y={y.shape}")
            print(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")
            print(f"Model n_variates after preparation: {model.model_config.n_variates}")
            
            # Create a small batch for testing
            batch_size = test_config['batch_size']
            if len(X) < batch_size:
                batch_size = len(X)
                
            batch_X = torch.FloatTensor(X[:batch_size])
            batch_y = torch.FloatTensor(y[:batch_size])
            
            print(f"Testing with batch: X={batch_X.shape}, y={batch_y.shape}")
            
            # Test forward pass
            model.model.eval()
            with torch.no_grad():
                print("Starting forward pass...")
                outputs = model.model(batch_X)
                print(f"Forward pass successful! Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'tensor'}")
                
                # Check output shapes
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}")
                            
            print(f"✅ TEST {i+1} PASSED")
            
        except Exception as e:
            print(f"❌ TEST {i+1} FAILED: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # If this is the tensor mismatch we're looking for, analyze it
            if "must match the size" in str(e) and ("20" in str(e) and "32" in str(e)):
                print("\n🔍 FOUND THE TENSOR MISMATCH!")
                print(f"Error details: {e}")
                print("This is the issue we need to fix.")
                return False
    
    return True

if __name__ == "__main__":
    success = test_itransformer_tensor_mismatch()
    if success:
        print("\n🎉 All tests passed! The tensor mismatch issue appears to be resolved.")
    else:
        print("\n🔧 Tensor mismatch issue identified. Ready to fix.")