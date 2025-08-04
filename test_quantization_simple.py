#!/usr/bin/env python3
"""
Simple test to verify transformer quantization functionality without full app context
"""

import torch
import torch.nn as nn
from src.ml_analysis.transformers.base import TransformerBase, TransformerConfig
from src.ml_analysis.base import ModelType

class SimpleTransformer(TransformerBase):
    """Simple transformer for testing quantization"""
    
    def __init__(self):
        config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 1024,
            'dropout': 0.1,
            'max_seq_length': 100,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'mixed_precision': False,
            'vocab_size': 1000,
            'pad_token_id': 0,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100
        }
        super().__init__(ModelType.TRANSFORMER, config)
        
        # Create simple transformer layers
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['d_model'],
                nhead=config['n_heads'],
                dim_feedforward=config['d_ff'],
                dropout=config['dropout'],
                batch_first=True
            )
            for _ in range(config['n_layers'])
        ])
        self.output_projection = nn.Linear(config['d_model'], 1)
        
        # Mark as trained for testing
        self._is_trained = True
        self._model = self
    
    def forward(self, x: torch.Tensor, attention_mask=None):
        """Simple forward pass"""
        # x is already embedded features, so just pass through transformer
        hidden = x
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Global average pooling and output projection
        pooled = hidden.mean(dim=1)  # (batch_size, d_model)
        output = self.output_projection(pooled)  # (batch_size, 1)
        
        return output
    
    def get_attention_weights(self):
        """Return dummy attention weights for testing"""
        batch_size, seq_len = 1, 10
        n_heads = self.transformer_config.n_heads
        return torch.randn(batch_size, n_heads, seq_len, seq_len)


def test_quantization_compatibility():
    """Test quantization compatibility analysis"""
    print("Testing quantization compatibility...")
    
    model = SimpleTransformer()
    
    # Test compatibility check
    is_compatible = model.is_quantization_compatible()
    print(f"Model is quantization compatible: {is_compatible}")
    
    # Test quantization preparation
    print("\nPreparing model for quantization...")
    prepared_model = model.prepare_for_quantization(
        quantization_type="dynamic",
        preserve_attention_precision=True
    )
    
    # Test quantization application
    print("Applying quantization...")
    quantized_model = prepared_model.apply_quantization(torch.qint8)
    
    # Test quantization info
    quant_info = quantized_model.get_quantization_info()
    print(f"\nQuantization info:")
    print(f"  Is quantized: {quant_info['is_quantized']}")
    print(f"  Quantization type: {quant_info['quantization_type']}")
    print(f"  Estimated speedup: {quant_info['estimated_speedup']:.2f}x")
    print(f"  Estimated memory reduction: {quant_info['estimated_memory_reduction']:.1%}")
    
    # Test forward pass with quantized model
    print("\nTesting forward pass...")
    test_input = torch.randn(1, 10, 256)
    
    with torch.no_grad():
        original_output = model.forward(test_input)
        quantized_output = quantized_model.forward(test_input)
    
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    print(f"Output difference (MSE): {torch.nn.functional.mse_loss(original_output, quantized_output):.6f}")
    
    # Test performance benchmarking
    print("\nBenchmarking performance...")
    test_inputs = [torch.randn(1, 10, 256) for _ in range(5)]
    benchmark_results = quantized_model.benchmark_quantization_performance(
        test_inputs, num_warmup=2, num_iterations=5
    )
    
    print(f"Benchmark results:")
    print(f"  Original latency: {benchmark_results['original_latency_ms']:.2f} ms")
    print(f"  Quantized latency: {benchmark_results['quantized_latency_ms']:.2f} ms")
    print(f"  Speedup ratio: {benchmark_results['speedup_ratio']:.2f}x")
    print(f"  Memory usage: {benchmark_results['memory_usage_mb']:.2f} MB")
    print(f"  Accuracy preservation: {benchmark_results['accuracy_preservation']:.4f}")
    
    print("\n✅ All quantization tests passed!")


if __name__ == "__main__":
    test_quantization_compatibility()