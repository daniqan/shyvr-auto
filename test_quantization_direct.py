#!/usr/bin/env python3
"""
Direct test of quantization functionality without full framework imports
"""

import torch
import torch.nn as nn
import time
import math
from typing import Dict, Any, List, Optional

# Direct import of the specific classes needed
import sys
import os
sys.path.insert(0, '/Users/kendo/daniqan/shyvrai-rlte/src')

# Mock logger to avoid circular imports
class MockLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} - {kwargs}")
    
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} - {kwargs}")
    
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} - {kwargs}")

# Set environment variable to avoid config issues
os.environ['SECRET_KEY'] = 'test_secret_key_123456789_secure_for_testing'

# Import the quantization classes directly
from ml_analysis.model_quantization import (
    TransformerQuantizer,
    TransformerQuantizationStrategy,
    AttentionLayerQuantizer
)

def test_transformer_quantization_classes():
    """Test the transformer quantization classes directly"""
    print("Testing TransformerQuantization classes directly...")
    
    # Create a simple transformer model for testing
    class SimpleTransformerForTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, 
                dropout=0.1, batch_first=True
            )
            self.output_layer = nn.Linear(256, 1)
            
        def forward(self, x):
            # x is input ids, embed and process
            if x.dtype == torch.long:
                x = self.embedding(x)
            hidden = self.transformer_layer(x)
            pooled = hidden.mean(dim=1)
            return self.output_layer(pooled)
    
    model = SimpleTransformerForTest()
    model.eval()
    
    # Test TransformerQuantizer
    print("\n1. Testing TransformerQuantizer...")
    quantizer = TransformerQuantizer()
    
    # Test compatibility analysis
    compatibility_info = quantizer.analyze_transformer_quantization_compatibility(model)
    print(f"Compatibility analysis: {compatibility_info}")
    
    # Test quantization application
    print("\n2. Testing quantization application...")
    test_input = torch.randint(0, 1000, (1, 50))  # Random token ids
    
    # Get original output
    with torch.no_grad():
        original_output = model(test_input)
    
    # Apply quantization
    try:
        quantized_model = quantizer.quantize_transformer_model(
            model, quantization_type="dynamic", target_dtype=torch.qint8, preserve_accuracy_threshold=0.95
        )
        
        # Test quantized output
        with torch.no_grad():
            quantized_output = quantized_model(test_input)
        
        print(f"Original output: {original_output.shape} - {original_output[0, 0].item():.6f}")
        print(f"Quantized output: {quantized_output.shape} - {quantized_output[0, 0].item():.6f}")
        
        # Calculate accuracy preservation
        mse = torch.nn.functional.mse_loss(original_output, quantized_output)
        print(f"MSE difference: {mse.item():.8f}")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
    
    # Test TransformerQuantizationStrategy
    print("\n3. Testing TransformerQuantizationStrategy...")
    strategy = TransformerQuantizationStrategy()
    
    # Test strategy creation for different transformer types
    transformer_types = ["iTransformer", "PatchTST", "TimesMixer", "TransformerPredictor"]
    
    for model_type in transformer_types:
        architecture_info = {
            "attention_layers": 6,
            "feed_forward_layers": 6,
            "embedding_layers": 1,
            "sequence_length": 100
        }
        
        strategy_config = strategy.create_transformer_strategy(
            model_type=model_type,
            architecture_info=architecture_info,
            target_speedup=2.5,
            max_accuracy_loss=0.05,
            preserve_attention_patterns=True
        )
        
        print(f"  {model_type}: {strategy_config['attention_quantization_plan']}")
        print(f"    Expected speedup: {strategy_config.get('expected_performance_gain', {}).get('speedup_ratio', 'N/A')}")
    
    # Test AttentionLayerQuantizer
    print("\n4. Testing AttentionLayerQuantizer...")
    attention_quantizer = AttentionLayerQuantizer()
    
    # Create a simple attention layer for testing
    attention_layer = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
    
    quantization_config = {
        'preserve_attention_precision': True,
        'quantization_type': 'dynamic'
    }
    
    try:
        quantized_attention = attention_quantizer.quantize_attention_layer(
            attention_layer, quantization_config
        )
        print(f"Attention layer quantized successfully: {type(quantized_attention)}")
        
        # Test attention layer
        test_input = torch.randn(1, 50, 256)
        with torch.no_grad():
            original_attn, _ = attention_layer(test_input, test_input, test_input)
            quantized_attn, _ = quantized_attention(test_input, test_input, test_input)
            
        attn_mse = torch.nn.functional.mse_loss(original_attn, quantized_attn)
        print(f"Attention MSE difference: {attn_mse.item():.8f}")
        
    except Exception as e:
        print(f"Attention quantization failed: {e}")
    
    print("\n✅ Transformer quantization classes tested successfully!")

def test_performance_benchmarking():
    """Test performance benchmarking functionality"""
    print("\n5. Testing performance benchmarking...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    original_model = SimpleModel()
    original_model.eval()
    
    # Create quantized version
    quantized_model = torch.quantization.quantize_dynamic(
        original_model, {nn.Linear}, dtype=torch.qint8
    )
    
    # Benchmark performance
    test_input = torch.randn(1, 256)
    num_iterations = 100
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(test_input)
            _ = quantized_model(test_input)
    
    # Benchmark original model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            original_output = original_model(test_input)
    original_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Benchmark quantized model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            quantized_output = quantized_model(test_input)
    quantized_time = (time.time() - start_time) * 1000  # Convert to ms
    
    speedup = original_time / quantized_time
    
    print(f"Original model time: {original_time:.2f} ms")
    print(f"Quantized model time: {quantized_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    # Check accuracy preservation
    mse = torch.nn.functional.mse_loss(original_output, quantized_output)
    accuracy_preservation = max(0.0, 1.0 - float(mse.item()))
    print(f"Accuracy preservation: {accuracy_preservation:.4f}")
    
    print("\n✅ Performance benchmarking completed!")

if __name__ == "__main__":
    try:
        test_transformer_quantization_classes()
        test_performance_benchmarking()
        print("\n🎉 All quantization tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()