"""
Test cases for Transformer Inference Optimization functionality - Phase 2.1

This module contains comprehensive test cases for transformer-specific inference optimizations,
following TDD methodology with failing tests first.

Tests cover:
- torch.compile integration
- ONNX export capabilities  
- Model compilation optimizations
- Inference pipeline integration
- Performance benchmarking
"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import the modules we'll be testing (these will fail initially)
try:
    from src.ml_analysis.transformers.optimization import (
        TorchCompileConfig,
        TorchCompileOptimizer,
        ONNXExportConfig,
        ONNXExporter,
        ModelCompilationManager,
        InferenceCompiler
    )
    from src.ml_analysis.inference_optimizer import (
        CompiledModelOptimizer,
        CompilationCacheManager
    )
except ImportError:
    # Expected to fail initially - we'll implement these classes
    pass


@dataclass
class CompilationTestConfig:
    """Test configuration for compilation tests"""
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = True
    enable_profiling: bool = True


class TestTorchCompileConfig:
    """Test torch.compile configuration"""
    
    def test_compile_config_initialization(self):
        """Test TorchCompileConfig initialization with default values"""
        # This test will fail until we implement TorchCompileConfig
        config = TorchCompileConfig()
        
        assert config.backend in ["inductor", "aot_eager", "onnxrt", "tensorrt"]
        assert config.mode in ["default", "reduce-overhead", "max-autotune"]
        assert config.fullgraph is not None
        assert config.dynamic is not None
        assert config.enable_profiling is not None
        
    def test_compile_config_custom_values(self):
        """Test TorchCompileConfig with custom values"""
        config = TorchCompileConfig(
            backend="tensorrt",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
            enable_profiling=True
        )
        
        assert config.backend == "tensorrt"
        assert config.mode == "max-autotune"
        assert config.fullgraph is True
        assert config.dynamic is False
        assert config.enable_profiling is True
        
    def test_compile_config_validation(self):
        """Test TorchCompileConfig parameter validation"""
        # Test invalid backend
        with pytest.raises(ValueError):
            TorchCompileConfig(backend="invalid_backend")
            
        # Test invalid mode
        with pytest.raises(ValueError):
            TorchCompileConfig(mode="invalid_mode")
            
    def test_compile_config_hardware_compatibility(self):
        """Test hardware compatibility checks"""
        config = TorchCompileConfig()
        
        compatibility = config.check_hardware_compatibility()
        
        assert "cuda_available" in compatibility
        assert "tensorrt_available" in compatibility
        assert "supported_backends" in compatibility
        assert isinstance(compatibility["supported_backends"], list)


class TestTorchCompileOptimizer:
    """Test torch.compile optimizer"""
    
    @pytest.fixture
    def compile_config(self):
        """Fixture providing compile configuration"""
        return TorchCompileConfig(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=True
        )
    
    @pytest.fixture
    def compile_optimizer(self, compile_config):
        """Fixture providing torch compile optimizer"""
        return TorchCompileOptimizer(compile_config)
    
    @pytest.fixture
    def simple_model(self):
        """Fixture providing simple model for testing"""
        class SimpleTransformer(nn.Module):
            def __init__(self, d_model=256, n_heads=4, n_layers=2):
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(1000, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, n_heads), 
                    n_layers
                )
                self.output = nn.Linear(d_model, 1000)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        return SimpleTransformer()
    
    def test_optimizer_initialization(self, compile_optimizer):
        """Test TorchCompileOptimizer initialization"""
        assert compile_optimizer.config is not None
        assert hasattr(compile_optimizer, 'compilation_cache')
        assert hasattr(compile_optimizer, 'performance_tracker')
        
    def test_model_compilation_basic(self, compile_optimizer, simple_model):
        """Test basic model compilation"""
        compiled_model = compile_optimizer.compile_model(simple_model)
        
        # Compiled model should be callable
        assert callable(compiled_model)
        
        # Test with sample input
        sample_input = torch.randint(0, 1000, (2, 10))
        output = compiled_model(sample_input)
        
        assert output.shape == (2, 10, 1000)
        
    def test_compilation_with_different_backends(self, simple_model):
        """Test compilation with different backends"""
        backends = ["inductor", "aot_eager"]
        
        for backend in backends:
            if compile_optimizer._is_backend_available(backend):
                config = TorchCompileConfig(backend=backend)
                optimizer = TorchCompileOptimizer(config)
                
                compiled_model = optimizer.compile_model(simple_model)
                
                # Test inference
                sample_input = torch.randint(0, 1000, (1, 8))
                output = compiled_model(sample_input)
                assert output.shape == (1, 8, 1000)
                
    def test_compilation_with_dynamic_shapes(self, compile_optimizer, simple_model):
        """Test compilation with dynamic input shapes"""
        # Enable dynamic shapes
        compile_optimizer.config.dynamic = True
        
        compiled_model = compile_optimizer.compile_model(simple_model)
        
        # Test with different input shapes
        input_shapes = [(1, 5), (2, 8), (1, 12)]
        
        for batch_size, seq_len in input_shapes:
            sample_input = torch.randint(0, 1000, (batch_size, seq_len))
            output = compiled_model(sample_input)
            assert output.shape == (batch_size, seq_len, 1000)
            
    def test_compilation_performance_tracking(self, compile_optimizer, simple_model):
        """Test compilation performance tracking"""
        # Compile model
        compiled_model = compile_optimizer.compile_model(simple_model)
        
        # Run inference to collect performance data
        sample_input = torch.randint(0, 1000, (4, 16))
        
        # Warmup
        for _ in range(5):
            _ = compiled_model(sample_input)
            
        # Measure performance
        start_time = time.time()
        for _ in range(10):
            _ = compiled_model(sample_input)
        compilation_time = time.time() - start_time
        
        performance_stats = compile_optimizer.get_performance_statistics()
        
        assert "compilation_time_ms" in performance_stats
        assert "inference_speedup" in performance_stats
        assert performance_stats["compilation_time_ms"] > 0
        
    def test_compilation_cache_management(self, compile_optimizer, simple_model):
        """Test compilation cache management"""
        # First compilation - should cache
        compiled_model1 = compile_optimizer.compile_model(simple_model, cache_key="model1")
        
        # Second compilation with same key - should use cache
        compiled_model2 = compile_optimizer.compile_model(simple_model, cache_key="model1")
        
        # Should return cached version
        cache_stats = compile_optimizer.get_cache_statistics()
        assert cache_stats["cache_hits"] > 0
        
    def test_compilation_error_handling(self, compile_optimizer):
        """Test compilation error handling"""
        # Create problematic model
        class ProblematicModel(nn.Module):
            def forward(self, x):
                # Intentionally problematic operation
                return x / 0  # Division by zero
        
        problematic_model = ProblematicModel()
        
        # Should handle compilation errors gracefully
        try:
            compiled_model = compile_optimizer.compile_model(problematic_model)
            # If compilation succeeds, test should still work
            assert compiled_model is not None
        except Exception as e:
            # Should provide meaningful error information
            assert "compilation" in str(e).lower() or "torch" in str(e).lower()
            
    def test_fullgraph_compilation(self, simple_model):
        """Test fullgraph compilation mode"""
        config = TorchCompileConfig(fullgraph=True)
        optimizer = TorchCompileOptimizer(config)
        
        compiled_model = optimizer.compile_model(simple_model)
        
        # Test inference
        sample_input = torch.randint(0, 1000, (2, 8))
        output = compiled_model(sample_input)
        
        assert output.shape == (2, 8, 1000)


class TestONNXExportConfig:
    """Test ONNX export configuration"""
    
    def test_onnx_config_initialization(self):
        """Test ONNXExportConfig initialization"""
        config = ONNXExportConfig()
        
        assert config.opset_version > 0
        assert config.export_params is not None
        assert config.do_constant_folding is not None
        assert config.input_names is not None
        assert config.output_names is not None
        assert config.dynamic_axes is not None
        
    def test_onnx_config_custom_values(self):
        """Test ONNXExportConfig with custom values"""
        config = ONNXExportConfig(
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}}
        )
        
        assert config.opset_version == 14
        assert config.export_params is True
        assert config.do_constant_folding is True
        assert config.input_names == ["input"]
        assert config.output_names == ["output"]
        assert "input" in config.dynamic_axes
        
    def test_onnx_config_validation(self):
        """Test ONNXExportConfig validation"""
        # Test invalid opset version
        with pytest.raises(ValueError):
            ONNXExportConfig(opset_version=5)  # Too old
            
        # Test invalid dynamic axes format
        with pytest.raises(ValueError):
            ONNXExportConfig(dynamic_axes="invalid")


class TestONNXExporter:
    """Test ONNX export functionality"""
    
    @pytest.fixture
    def onnx_config(self):
        """Fixture providing ONNX configuration"""
        return ONNXExportConfig(
            opset_version=11,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
    
    @pytest.fixture
    def onnx_exporter(self, onnx_config):
        """Fixture providing ONNX exporter"""
        return ONNXExporter(onnx_config)
    
    @pytest.fixture
    def export_model(self):
        """Fixture providing model for export"""
        class ExportModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                self.linear = nn.Linear(128, 1000)
                
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                return self.linear(x)
        
        return ExportModel()
    
    def test_onnx_exporter_initialization(self, onnx_exporter):
        """Test ONNXExporter initialization"""
        assert onnx_exporter.config is not None
        assert hasattr(onnx_exporter, 'export_statistics')
        
    def test_basic_onnx_export(self, onnx_exporter, export_model):
        """Test basic ONNX export"""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Export model
            sample_input = torch.randint(0, 1000, (2, 10))
            
            export_success = onnx_exporter.export_model(
                export_model, sample_input, temp_path
            )
            
            assert export_success is True
            assert os.path.exists(temp_path)
            
            # Check file size (should be non-empty)
            assert os.path.getsize(temp_path) > 0
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_onnx_export_with_dynamic_axes(self, onnx_exporter, export_model):
        """Test ONNX export with dynamic axes"""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Export with dynamic axes
            sample_input = torch.randint(0, 1000, (1, 5))
            
            export_success = onnx_exporter.export_model(
                export_model, sample_input, temp_path
            )
            
            assert export_success is True
            
            # Verify dynamic axes in export info
            export_info = onnx_exporter.get_export_info(temp_path)
            assert "dynamic_axes" in export_info
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_onnx_model_validation(self, onnx_exporter, export_model):
        """Test ONNX model validation after export"""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            sample_input = torch.randint(0, 1000, (2, 8))
            
            # Export model
            onnx_exporter.export_model(export_model, sample_input, temp_path)
            
            # Validate exported model
            validation_result = onnx_exporter.validate_exported_model(temp_path, sample_input)
            
            assert validation_result["is_valid"] is True
            assert "pytorch_output" in validation_result
            assert "onnx_output" in validation_result
            
            # Outputs should be close
            pytorch_output = validation_result["pytorch_output"]
            onnx_output = validation_result["onnx_output"]
            
            assert np.allclose(pytorch_output, onnx_output, atol=1e-5)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_onnx_export_optimization(self, onnx_exporter, export_model):
        """Test ONNX export with optimization"""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            sample_input = torch.randint(0, 1000, (2, 8))
            
            # Export with optimization
            export_success = onnx_exporter.export_model_optimized(
                export_model, sample_input, temp_path, optimization_level=2
            )
            
            assert export_success is True
            
            # Check optimization statistics
            stats = onnx_exporter.get_export_statistics()
            assert "optimization_level" in stats
            assert "export_time_ms" in stats
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_batch_onnx_export(self, onnx_exporter):
        """Test batch export of multiple models"""
        models = []
        for i in range(3):
            model = nn.Sequential(
                nn.Embedding(100, 32),
                nn.Linear(32, 10)
            )
            models.append((f"model_{i}", model))
            
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_input = torch.randint(0, 100, (1, 5))
            
            export_results = onnx_exporter.batch_export_models(
                models, sample_input, temp_dir
            )
            
            assert len(export_results) == 3
            
            for result in export_results:
                assert result["success"] is True
                assert os.path.exists(result["file_path"])


class TestModelCompilationManager:
    """Test model compilation manager"""
    
    @pytest.fixture
    def compilation_manager(self):
        """Fixture providing compilation manager"""
        return ModelCompilationManager(
            enable_torch_compile=True,
            enable_onnx_export=True,
            cache_compiled_models=True
        )
    
    @pytest.fixture
    def test_transformer(self):
        """Fixture providing test transformer model"""
        class TestTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(5000, 256)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(256, 8, batch_first=True),
                    num_layers=3
                )
                self.output = nn.Linear(256, 5000)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        return TestTransformer()
    
    def test_compilation_manager_initialization(self, compilation_manager):
        """Test ModelCompilationManager initialization"""
        assert compilation_manager.enable_torch_compile is not None
        assert compilation_manager.enable_onnx_export is not None
        assert compilation_manager.cache_compiled_models is not None
        assert hasattr(compilation_manager, 'torch_compiler')
        assert hasattr(compilation_manager, 'onnx_exporter')
        
    def test_end_to_end_compilation(self, compilation_manager, test_transformer):
        """Test end-to-end model compilation"""
        sample_input = torch.randint(0, 5000, (2, 16))
        
        compilation_result = compilation_manager.compile_model_complete(
            test_transformer, sample_input, model_name="test_transformer"
        )
        
        assert "torch_compiled_model" in compilation_result
        assert "compilation_success" in compilation_result
        assert compilation_result["compilation_success"] is True
        
        # Test compiled model
        compiled_model = compilation_result["torch_compiled_model"]
        output = compiled_model(sample_input)
        assert output.shape == (2, 16, 5000)
        
    def test_compilation_with_onnx_export(self, compilation_manager, test_transformer):
        """Test compilation with ONNX export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_input = torch.randint(0, 5000, (1, 8))
            
            compilation_result = compilation_manager.compile_and_export(
                test_transformer, sample_input, 
                model_name="test_transformer",
                export_dir=temp_dir
            )
            
            assert compilation_result["torch_compilation_success"] is True
            assert compilation_result["onnx_export_success"] is True
            assert os.path.exists(compilation_result["onnx_file_path"])
            
    def test_compilation_performance_comparison(self, compilation_manager, test_transformer):
        """Test performance comparison between original and compiled models"""
        sample_input = torch.randint(0, 5000, (4, 12))
        
        # Benchmark original model
        original_times = []
        test_transformer.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = test_transformer(sample_input)
                
            # Measure
            for _ in range(10):
                start_time = time.time()
                _ = test_transformer(sample_input)
                original_times.append(time.time() - start_time)
        
        # Compile and benchmark compiled model
        compilation_result = compilation_manager.compile_model_complete(
            test_transformer, sample_input
        )
        
        compiled_model = compilation_result["torch_compiled_model"]
        compiled_times = []
        
        with torch.no_grad():
            # Warmup compiled model
            for _ in range(5):
                _ = compiled_model(sample_input)
                
            # Measure compiled model
            for _ in range(10):
                start_time = time.time()
                _ = compiled_model(sample_input)
                compiled_times.append(time.time() - start_time)
        
        # Calculate performance metrics
        avg_original_time = np.mean(original_times)
        avg_compiled_time = np.mean(compiled_times)
        
        performance_metrics = compilation_manager.calculate_performance_improvement(
            avg_original_time, avg_compiled_time
        )
        
        assert "speedup_ratio" in performance_metrics
        assert "latency_reduction_ms" in performance_metrics
        assert performance_metrics["speedup_ratio"] > 0
        
    def test_compilation_cache_persistence(self, compilation_manager, test_transformer):
        """Test compilation cache persistence"""
        sample_input = torch.randint(0, 5000, (1, 10))
        model_name = "persistent_test_model"
        
        # First compilation
        result1 = compilation_manager.compile_model_complete(
            test_transformer, sample_input, model_name=model_name
        )
        
        # Second compilation with same name - should use cache
        result2 = compilation_manager.compile_model_complete(
            test_transformer, sample_input, model_name=model_name
        )
        
        cache_stats = compilation_manager.get_cache_statistics()
        assert cache_stats["cache_hits"] > 0
        
    def test_compilation_error_recovery(self, compilation_manager):
        """Test compilation error recovery"""
        # Create model that might fail compilation
        class ProblematicModel(nn.Module):
            def forward(self, x):
                # Dynamic operation that might cause compilation issues
                if x.sum() > 0:
                    return x * 2
                else:
                    return x * 3
        
        problematic_model = ProblematicModel()
        sample_input = torch.randn(2, 4)
        
        compilation_result = compilation_manager.compile_model_complete(
            problematic_model, sample_input, 
            fallback_to_original=True
        )
        
        # Should either succeed or fallback gracefully
        assert "compilation_success" in compilation_result
        if not compilation_result["compilation_success"]:
            assert "fallback_model" in compilation_result
            assert compilation_result["fallback_model"] is not None


class TestInferenceCompiler:
    """Test inference compiler integration"""
    
    @pytest.fixture
    def inference_compiler(self):
        """Fixture providing inference compiler"""
        return InferenceCompiler(
            enable_torch_compile=True,
            compile_backend="inductor",
            optimization_level=2
        )
    
    def test_inference_compiler_initialization(self, inference_compiler):
        """Test InferenceCompiler initialization"""
        assert inference_compiler.enable_torch_compile is not None
        assert inference_compiler.compile_backend is not None
        assert inference_compiler.optimization_level > 0
        
    def test_inference_pipeline_compilation(self, inference_compiler):
        """Test compilation of complete inference pipeline"""
        # Create inference pipeline components
        tokenizer_mock = Mock()
        model_mock = Mock()
        post_processor_mock = Mock()
        
        # Set up mock returns
        tokenizer_mock.return_value = torch.randint(0, 1000, (2, 10))
        model_mock.return_value = torch.randn(2, 10, 1000)
        post_processor_mock.return_value = {"predictions": [0.8, 0.7]}
        
        pipeline_components = {
            "tokenizer": tokenizer_mock,
            "model": model_mock,
            "post_processor": post_processor_mock
        }
        
        compiled_pipeline = inference_compiler.compile_inference_pipeline(
            pipeline_components
        )
        
        assert "compiled_model" in compiled_pipeline
        assert "compilation_info" in compiled_pipeline
        
        # Test compiled pipeline
        test_input = ["test sentence", "another test"]
        result = compiled_pipeline["compiled_model"](test_input)
        assert result is not None
        
    def test_streaming_inference_compilation(self, inference_compiler):
        """Test compilation for streaming inference"""
        class StreamingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(128, 256, batch_first=True)
                self.output = nn.Linear(256, 1000)
                
            def forward(self, x, hidden_state=None):
                output, new_hidden = self.rnn(x, hidden_state)
                return self.output(output), new_hidden
        
        streaming_model = StreamingModel()
        
        compiled_streaming = inference_compiler.compile_for_streaming(
            streaming_model, input_shape=(1, 1, 128)
        )
        
        assert "compiled_model" in compiled_streaming
        assert "supports_streaming" in compiled_streaming
        assert compiled_streaming["supports_streaming"] is True
        
        # Test streaming functionality
        compiled_model = compiled_streaming["compiled_model"]
        sample_input = torch.randn(1, 1, 128)
        
        output, hidden = compiled_model(sample_input)
        assert output.shape == (1, 1, 1000)
        assert hidden is not None


class TestCompiledModelOptimizer:
    """Test compiled model optimizer from inference_optimizer.py"""
    
    @pytest.fixture
    def compiled_optimizer(self):
        """Fixture providing compiled model optimizer"""
        return CompiledModelOptimizer(
            enable_compilation=True,
            compilation_backend="inductor",
            cache_compiled_models=True
        )
    
    def test_compiled_optimizer_initialization(self, compiled_optimizer):
        """Test CompiledModelOptimizer initialization"""
        assert compiled_optimizer.enable_compilation is not None
        assert compiled_optimizer.compilation_backend is not None
        assert compiled_optimizer.cache_compiled_models is not None
        
    def test_model_compilation_for_inference(self, compiled_optimizer):
        """Test model compilation specifically for inference"""
        # Create test model
        test_model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        sample_input = torch.randn(4, 100)
        
        compiled_result = compiled_optimizer.compile_for_inference(
            test_model, sample_input
        )
        
        assert "compiled_model" in compiled_result
        assert "optimization_info" in compiled_result
        
        # Test inference
        compiled_model = compiled_result["compiled_model"]
        output = compiled_model(sample_input)
        assert output.shape == (4, 10)
        
    def test_batch_inference_optimization(self, compiled_optimizer):
        """Test batch inference optimization"""
        model = nn.Linear(50, 20)
        
        # Different batch sizes
        batch_inputs = [
            torch.randn(2, 50),
            torch.randn(4, 50),
            torch.randn(8, 50)
        ]
        
        optimized_results = compiled_optimizer.optimize_batch_inference(
            model, batch_inputs
        )
        
        assert len(optimized_results) == 3
        for i, result in enumerate(optimized_results):
            expected_batch_size = [2, 4, 8][i]
            assert result.shape == (expected_batch_size, 20)
            
    def test_inference_latency_optimization(self, compiled_optimizer):
        """Test inference latency optimization"""
        # Create model with different complexity levels
        complex_model = nn.Sequential(
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        
        sample_input = torch.randn(1, 200)
        
        # Measure original latency
        original_times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = complex_model(sample_input)
            original_times.append(time.time() - start_time)
        
        # Compile and measure optimized latency
        compiled_result = compiled_optimizer.compile_for_inference(
            complex_model, sample_input
        )
        compiled_model = compiled_result["compiled_model"]
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = compiled_model(sample_input)
        
        compiled_times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = compiled_model(sample_input)
            compiled_times.append(time.time() - start_time)
        
        avg_original = np.mean(original_times)
        avg_compiled = np.mean(compiled_times)
        
        latency_improvement = compiled_optimizer.calculate_latency_improvement(
            avg_original, avg_compiled
        )
        
        assert "improvement_ratio" in latency_improvement
        assert "latency_reduction_ms" in latency_improvement


class TestCompilationCacheManager:
    """Test compilation cache manager"""
    
    @pytest.fixture
    def cache_manager(self):
        """Fixture providing compilation cache manager"""
        return CompilationCacheManager(
            cache_size_limit=100,
            enable_persistent_cache=False,
            eviction_strategy="lru"
        )
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test CompilationCacheManager initialization"""
        assert cache_manager.cache_size_limit > 0
        assert cache_manager.enable_persistent_cache is not None
        assert cache_manager.eviction_strategy in ["lru", "fifo", "lfu"]
        
    def test_compilation_caching(self, cache_manager):
        """Test compilation result caching"""
        # Mock compiled model
        mock_compiled_model = Mock()
        model_key = "test_model_v1"
        compilation_info = {
            "backend": "inductor",
            "compile_time_ms": 1500,
            "optimization_level": 2
        }
        
        # Cache compiled model
        success = cache_manager.cache_compiled_model(
            model_key, mock_compiled_model, compilation_info
        )
        assert success is True
        
        # Retrieve from cache
        cached_result = cache_manager.get_cached_model(model_key)
        assert cached_result is not None
        assert cached_result["compiled_model"] == mock_compiled_model
        assert cached_result["compilation_info"] == compilation_info
        
    def test_cache_eviction_lru(self, cache_manager):
        """Test LRU cache eviction"""
        cache_manager.cache_size_limit = 3
        
        # Fill cache beyond limit
        for i in range(5):
            mock_model = Mock()
            cache_manager.cache_compiled_model(
                f"model_{i}", mock_model, {"compile_time_ms": 100}
            )
        
        # First models should be evicted
        assert cache_manager.get_cached_model("model_0") is None
        assert cache_manager.get_cached_model("model_1") is None
        
        # Recent models should be retained
        assert cache_manager.get_cached_model("model_4") is not None
        
    def test_cache_statistics_collection(self, cache_manager):
        """Test cache statistics collection"""
        # Perform cache operations
        mock_model = Mock()
        cache_manager.cache_compiled_model("test_model", mock_model, {})
        
        cache_manager.get_cached_model("test_model")  # Hit
        cache_manager.get_cached_model("nonexistent")  # Miss
        
        stats = cache_manager.get_cache_statistics()
        
        assert "cache_size" in stats
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert stats["hit_count"] > 0
        assert stats["miss_count"] > 0


class TestIntegrationTransformerOptimization:
    """Integration tests for transformer optimization functionality"""
    
    def test_end_to_end_transformer_optimization(self):
        """Test end-to-end transformer optimization pipeline"""
        # Create transformer model
        class OptimizationTestTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(2000, 128)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(128, 4, batch_first=True),
                    num_layers=2
                )
                self.output = nn.Linear(128, 2000)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        model = OptimizationTestTransformer()
        sample_input = torch.randint(0, 2000, (2, 20))
        
        # Test torch compilation
        compile_config = TorchCompileConfig(backend="inductor")
        torch_optimizer = TorchCompileOptimizer(compile_config)
        compiled_model = torch_optimizer.compile_model(model)
        
        # Test ONNX export
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            onnx_config = ONNXExportConfig(
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
            )
            onnx_exporter = ONNXExporter(onnx_config)
            
            export_success = onnx_exporter.export_model(model, sample_input, temp_path)
            assert export_success is True
            
            # Test compiled model inference
            compiled_output = compiled_model(sample_input)
            assert compiled_output.shape == (2, 20, 2000)
            
            # Test ONNX model validation
            validation_result = onnx_exporter.validate_exported_model(temp_path, sample_input)
            assert validation_result["is_valid"] is True
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    @pytest.mark.slow
    def test_performance_benchmark_compilation(self):
        """Benchmark compilation performance improvements"""
        # Create substantial model for meaningful benchmarking
        class BenchmarkTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10000, 512)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, batch_first=True),
                    num_layers=6
                )
                self.output = nn.Linear(512, 10000)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        model = BenchmarkTransformer()
        sample_input = torch.randint(0, 10000, (4, 32))
        
        # Benchmark original model
        model.eval()
        original_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(sample_input)
                
            # Measure
            for _ in range(20):
                start_time = time.time()
                _ = model(sample_input)
                original_times.append(time.time() - start_time)
        
        # Compile and benchmark
        compile_config = TorchCompileConfig(backend="inductor", mode="default")
        optimizer = TorchCompileOptimizer(compile_config)
        
        compiled_model = optimizer.compile_model(model)
        
        compiled_times = []
        with torch.no_grad():
            # Warmup compiled model
            for _ in range(5):
                _ = compiled_model(sample_input)
                
            # Measure compiled model
            for _ in range(20):
                start_time = time.time()
                _ = compiled_model(sample_input)
                compiled_times.append(time.time() - start_time)
        
        # Calculate performance improvement
        avg_original = np.mean(original_times) * 1000  # ms
        avg_compiled = np.mean(compiled_times) * 1000  # ms
        speedup = avg_original / avg_compiled
        
        print(f"Original average time: {avg_original:.2f}ms")
        print(f"Compiled average time: {avg_compiled:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Expect some performance improvement (even small improvements are valuable)
        assert speedup > 0.8  # At least not slower
        
    def test_memory_efficiency_compilation(self):
        """Test memory efficiency of compiled models"""
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
        
        sample_input = torch.randn(8, 1000)
        
        if torch.cuda.is_available():
            model = model.cuda()
            sample_input = sample_input.cuda()
            
            # Measure original model memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(sample_input)
            
            original_memory = torch.cuda.max_memory_allocated()
            
            # Compile and measure memory
            compile_config = TorchCompileConfig(backend="inductor")
            optimizer = TorchCompileOptimizer(compile_config)
            compiled_model = optimizer.compile_model(model)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = compiled_model(sample_input)
            
            compiled_memory = torch.cuda.max_memory_allocated()
            
            # Memory usage should be comparable or better
            memory_ratio = compiled_memory / original_memory
            assert memory_ratio <= 1.2  # Allow up to 20% more memory for compilation overhead
            
            print(f"Original memory: {original_memory / 1024 / 1024:.2f} MB")
            print(f"Compiled memory: {compiled_memory / 1024 / 1024:.2f} MB")
            print(f"Memory ratio: {memory_ratio:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])