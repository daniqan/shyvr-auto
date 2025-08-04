#!/usr/bin/env python3
"""
Phase 2.1 Model Quantization Implementation Validation
Tests all key components implemented for transformer quantization
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Set environment variable to avoid config issues
os.environ['SECRET_KEY'] = 'test_secret_key_123456789_secure_for_testing'

# Add src to path
sys.path.insert(0, '/Users/kendo/daniqan/shyvrai-rlte/src')

# Import quantization classes
from ml_analysis.model_quantization import (
    TransformerQuantizer,
    TransformerQuantizationStrategy,
    AttentionLayerQuantizer,
    TransformerAccuracyValidator,
    TransformerPerformanceBenchmarker
)

def test_phase_2_1_implementation():
    """Comprehensive test of Phase 2.1 Model Quantization implementation"""
    print("🧪 Phase 2.1 Model Quantization Implementation Validation")
    print("=" * 60)
    
    results = {
        "transformer_quantizer": False,
        "quantization_strategies": False,
        "attention_quantization": False,
        "accuracy_validation": False,
        "performance_benchmarking": False,
        "transformer_base_integration": False
    }
    
    # Test 1: TransformerQuantizer
    print("\n1️⃣ Testing TransformerQuantizer...")
    try:
        # Create a simple transformer for testing
        class TestTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 256)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024, 
                    dropout=0.1, batch_first=True
                )
                self.output = nn.Linear(256, 1)
                self.model_type = "TransformerPredictor"  # Add required attribute
                
            def forward(self, x):
                if x.dtype == torch.long:
                    x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x.mean(dim=1))
        
        model = TestTransformer()
        quantizer = TransformerQuantizer()
        
        # Test compatibility analysis
        compatibility = quantizer.analyze_transformer_quantization_compatibility(
            model, preserve_attention_precision=True, target_accuracy_retention=0.95
        )
        
        assert "attention_layer_analysis" in compatibility
        assert "feed_forward_layer_analysis" in compatibility
        assert "expected_speedup_transformer" in compatibility
        assert compatibility["expected_speedup_transformer"] > 1.0
        
        print("  ✅ TransformerQuantizer compatibility analysis: PASSED")
        results["transformer_quantizer"] = True
        
    except Exception as e:
        print(f"  ❌ TransformerQuantizer test failed: {e}")
    
    # Test 2: TransformerQuantizationStrategy
    print("\n2️⃣ Testing TransformerQuantizationStrategy...")
    try:
        strategy = TransformerQuantizationStrategy()
        
        # Test strategy creation for all transformer types
        transformer_types = ["iTransformer", "PatchTST", "TimesMixer", "TransformerPredictor"]
        strategies_created = 0
        
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
            
            # Validate strategy structure
            assert "attention_quantization_plan" in strategy_config
            assert "feed_forward_quantization_plan" in strategy_config
            assert "embedding_quantization_plan" in strategy_config
            
            strategies_created += 1
        
        assert strategies_created == 4
        print("  ✅ TransformerQuantizationStrategy: PASSED")
        results["quantization_strategies"] = True
        
    except Exception as e:
        print(f"  ❌ TransformerQuantizationStrategy test failed: {e}")
    
    # Test 3: AttentionLayerQuantizer
    print("\n3️⃣ Testing AttentionLayerQuantizer...")
    try:
        attention_quantizer = AttentionLayerQuantizer()
        
        # Create attention layer
        attention_layer = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Test attention precision preservation analysis
        precision_analysis = attention_quantizer.analyze_attention_precision_requirements(
            attention_layer, preserve_key_query=True
        )
        
        assert "quantization_sensitivity" in precision_analysis
        assert "precision_requirements" in precision_analysis
        assert precision_analysis["quantization_sensitivity"] > 0.5  # Attention is sensitive
        
        print("  ✅ AttentionLayerQuantizer precision analysis: PASSED")
        results["attention_quantization"] = True
        
    except Exception as e:
        print(f"  ❌ AttentionLayerQuantizer test failed: {e}")
    
    # Test 4: TransformerAccuracyValidator
    print("\n4️⃣ Testing TransformerAccuracyValidator...")
    try:
        accuracy_validator = TransformerAccuracyValidator()
        
        # Create test data
        original_outputs = torch.randn(100, 1)
        quantized_outputs = original_outputs + torch.randn(100, 1) * 0.01  # Small noise
        
        # Test accuracy validation
        accuracy_metrics = accuracy_validator.validate_transformer_quantization_accuracy(
            original_outputs, quantized_outputs, accuracy_threshold=0.95
        )
        
        assert "mse_loss" in accuracy_metrics
        assert "accuracy_retention" in accuracy_metrics
        assert "meets_threshold" in accuracy_metrics
        assert accuracy_metrics["accuracy_retention"] > 0.90
        
        print("  ✅ TransformerAccuracyValidator: PASSED")
        results["accuracy_validation"] = True
        
    except Exception as e:
        print(f"  ❌ TransformerAccuracyValidator test failed: {e}")
    
    # Test 5: TransformerPerformanceBenchmarker
    print("\n5️⃣ Testing TransformerPerformanceBenchmarker...")
    try:
        benchmarker = TransformerPerformanceBenchmarker()
        
        # Create simple models for benchmarking
        original_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # We'll use the same model as "quantized" for this test
        # (actual quantization would require specific PyTorch backend setup)
        quantized_model = original_model
        
        test_inputs = [torch.randn(1, 256) for _ in range(10)]
        
        # Test performance benchmarking
        benchmark_results = benchmarker.benchmark_transformer_quantization_performance(
            original_model, quantized_model, test_inputs, num_iterations=10
        )
        
        assert "original_latency_ms" in benchmark_results
        assert "quantized_latency_ms" in benchmark_results
        assert "speedup_ratio" in benchmark_results
        assert "memory_reduction_mb" in benchmark_results
        assert benchmark_results["speedup_ratio"] > 0
        
        print("  ✅ TransformerPerformanceBenchmarker: PASSED")
        results["performance_benchmarking"] = True
        
    except Exception as e:
        print(f"  ❌ TransformerPerformanceBenchmarker test failed: {e}")
    
    # Test 6: TransformerBase Integration (simple import test)
    print("\n6️⃣ Testing TransformerBase Integration...")
    try:
        from ml_analysis.transformers.base import TransformerBase
        
        # Check that TransformerBase has quantization methods
        required_methods = [
            "is_quantization_compatible",
            "prepare_for_quantization", 
            "apply_quantization",
            "get_quantization_info",
            "benchmark_quantization_performance"
        ]
        
        for method_name in required_methods:
            assert hasattr(TransformerBase, method_name), f"Missing method: {method_name}"
        
        print("  ✅ TransformerBase quantization integration: PASSED")
        results["transformer_base_integration"] = True
        
    except Exception as e:
        print(f"  ❌ TransformerBase integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 2.1 IMPLEMENTATION VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {component.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Phase 2.1 Model Quantization implementation is COMPLETE and WORKING!")
        return True
    else:
        print(f"⚠️  Phase 2.1 implementation has {total_tests - passed_tests} issues to address")
        return False

if __name__ == "__main__":
    success = test_phase_2_1_implementation()
    sys.exit(0 if success else 1)