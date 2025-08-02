"""
Test-Driven Development for Inference Optimization - Phase 5.2

This module contains comprehensive tests for model inference optimization including:
- Inference pipeline optimization
- GPU acceleration with CPU fallback
- Model quantization support
- Inference caching systems
- Batch prediction capabilities

Following TDD methodology - these tests will fail initially and drive the implementation.
"""

import asyncio
import time
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Test fixtures and utilities
@dataclass
class MockMarketState:
    """Mock market state for testing"""
    price: float = 100.0
    volume: float = 1000.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector"""
        return np.array([self.price, self.volume, time.time()])
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to enhanced feature vector"""
        return np.array([
            self.price, self.volume, time.time(),
            self.price * 0.1, self.volume * 0.1, 1.0  # Additional features
        ])

@dataclass 
class MockDiscoveredToken:
    """Mock token for testing"""
    address: str = "0x123"
    price_usd: float = 100.0
    symbol: str = "TEST"
    name: str = "Test Token"

class TestInferenceOptimizer:
    """Test suite for inference optimization capabilities"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.parameters = Mock(return_value=[torch.randn(10, 10)])
        return model
    
    @pytest.fixture  
    def mock_market_states(self):
        """Create batch of mock market states"""
        return [MockMarketState(price=100.0 + i*10, volume=1000.0 + i*100) 
                for i in range(10)]
    
    def test_inference_optimizer_initialization(self):
        """Test InferenceOptimizer can be initialized with proper configuration"""
        # This test will fail initially - InferenceOptimizer doesn't exist yet
        from src.ml_analysis.inference_optimizer import InferenceOptimizer, InferenceConfig
        
        config = InferenceConfig(
            batch_size=32,
            max_batch_wait_ms=50,
            enable_gpu=True,
            enable_quantization=False,
            cache_size_mb=256,
            optimization_level=2
        )
        
        optimizer = InferenceOptimizer(config)
        
        assert optimizer.config.batch_size == 32
        assert optimizer.config.max_batch_wait_ms == 50
        assert optimizer.config.enable_gpu is True
        assert optimizer.device_manager is not None
        assert optimizer.batch_processor is not None
        assert optimizer.cache_manager is not None
    
    def test_device_detection_and_selection(self):
        """Test automatic device detection with GPU/CPU fallback"""
        from src.ml_analysis.inference_optimizer import DeviceManager
        
        device_manager = DeviceManager()
        
        # Should detect available devices
        assert hasattr(device_manager, 'available_devices')
        assert len(device_manager.available_devices) > 0
        
        # Should have primary device (CPU at minimum)
        assert device_manager.primary_device is not None
        assert str(device_manager.primary_device) in ['cuda:0', 'cpu']
        
        # Should provide fallback mechanism
        fallback_device = device_manager.get_fallback_device()
        assert fallback_device is not None
        
        # Test device health check
        is_healthy = device_manager.check_device_health(device_manager.primary_device)
        assert isinstance(is_healthy, bool)
    
    def test_batch_processing_configuration(self):
        """Test batch processing with configurable parameters"""
        from src.ml_analysis.inference_optimizer import BatchProcessor
        
        processor = BatchProcessor(
            batch_size=16,
            max_wait_ms=100,
            auto_sizing=True
        )
        
        assert processor.batch_size == 16
        assert processor.max_wait_ms == 100
        assert processor.auto_sizing is True
        assert processor.current_batch == []
        assert processor.batch_ready is False
    
    def test_inference_caching_system(self):
        """Test inference result caching with TTL and size limits"""
        from src.ml_analysis.inference_optimizer import InferenceCache
        
        cache = InferenceCache(
            max_size_mb=64,
            ttl_seconds=300,
            enable_compression=True
        )
        
        # Test cache operations
        test_key = "test_model_prediction_hash123"
        test_result = {"prediction": [0.7, 0.2, 0.1], "confidence": 0.85}
        
        # Should be able to store and retrieve results
        success = cache.put(test_key, test_result)
        assert success is True
        
        cached_result = cache.get(test_key)
        assert cached_result is not None
        assert cached_result["prediction"] == [0.7, 0.2, 0.1]
        assert cached_result["confidence"] == 0.85
        
        # Test cache miss
        missing_result = cache.get("nonexistent_key")
        assert missing_result is None
        
        # Test cache metrics
        metrics = cache.get_metrics()
        assert "hit_rate" in metrics
        assert "size_mb" in metrics
        assert "entries_count" in metrics
    
    def test_model_quantization_support(self):
        """Test model quantization for inference acceleration"""
        from src.ml_analysis.inference_optimizer import ModelQuantizer
        
        # Create mock PyTorch model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        quantizer = ModelQuantizer()
        
        # Test INT8 quantization
        quantized_model = quantizer.quantize_int8(mock_model)
        assert quantized_model is not None
        
        # Test quantization validation
        is_quantized = quantizer.is_quantized(quantized_model)
        assert isinstance(is_quantized, bool)
        
        # Test quantization benchmarking
        benchmark_results = quantizer.benchmark_quantization(
            original_model=mock_model,
            quantized_model=quantized_model,
            test_input=torch.randn(1, 10)
        )
        
        assert "latency_original" in benchmark_results
        assert "latency_quantized" in benchmark_results
        assert "accuracy_loss" in benchmark_results
        assert "memory_reduction" in benchmark_results
    
    @pytest.mark.asyncio
    async def test_batch_inference_pipeline(self, mock_model, mock_market_states):
        """Test optimized batch inference pipeline"""
        from src.ml_analysis.inference_optimizer import InferenceOptimizer, InferenceConfig
        
        config = InferenceConfig(batch_size=8, max_batch_wait_ms=50)
        optimizer = InferenceOptimizer(config)
        
        # Test batch prediction
        predictions = await optimizer.predict_batch(
            model=mock_model,
            inputs=mock_market_states[:8]
        )
        
        assert len(predictions) == 8
        assert all("prediction" in pred for pred in predictions)
        assert all("latency_ms" in pred for pred in predictions)
        
        # Test auto-batching with dynamic sizing
        optimizer.config.auto_batch_sizing = True
        dynamic_predictions = await optimizer.predict_batch(
            model=mock_model,
            inputs=mock_market_states
        )
        
        assert len(dynamic_predictions) == len(mock_market_states)
    
    @pytest.mark.asyncio
    async def test_gpu_acceleration_with_fallback(self, mock_model):
        """Test GPU acceleration with automatic CPU fallback"""
        from src.ml_analysis.inference_optimizer import InferenceOptimizer, InferenceConfig
        
        config = InferenceConfig(enable_gpu=True, gpu_fallback_enabled=True)
        optimizer = InferenceOptimizer(config)
        
        test_input = MockMarketState()
        
        # Mock GPU failure to test fallback
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch.object(mock_model, 'to', side_effect=RuntimeError("CUDA error")):
            
            prediction = await optimizer.predict_single(mock_model, test_input)
            
            # Should fallback to CPU and still work 
            assert prediction is not None
            assert "device_used" in prediction
            assert prediction["device_used"] == "cpu"
            assert "fallback_reason" in prediction
    
    def test_inference_performance_monitoring(self):
        """Test inference performance monitoring and metrics"""
        from src.ml_analysis.inference_optimizer import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Record inference timing
        monitor.record_inference(
            model_type="lstm",
            latency_ms=45.2,
            batch_size=16,
            device="cuda:0"
        )
        
        monitor.record_inference(
            model_type="dqn", 
            latency_ms=23.1,
            batch_size=8,
            device="cpu"
        )
        
        # Get performance metrics
        metrics = monitor.get_metrics()
        
        assert "average_latency_ms" in metrics
        assert "throughput_predictions_per_sec" in metrics
        assert "device_usage" in metrics
        assert "model_performance" in metrics
        
        # Test performance alerting
        alerts = monitor.check_performance_thresholds(
            max_latency_ms=100,
            min_throughput=10
        )
        
        assert isinstance(alerts, list)
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self):
        """Test memory usage optimization during inference"""
        from src.ml_analysis.inference_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Test memory profiling
        memory_profile = optimizer.profile_memory_usage()
        assert "total_memory_mb" in memory_profile
        assert "available_memory_mb" in memory_profile
        assert "gpu_memory_mb" in memory_profile
        
        # Test automatic memory cleanup
        initial_memory = optimizer.get_memory_usage()
        
        # Simulate memory pressure
        optimizer.cleanup_memory()
        
        final_memory = optimizer.get_memory_usage()
        
        # Should have same or less memory usage after cleanup
        assert final_memory["allocated_mb"] <= initial_memory["allocated_mb"]
    
    def test_inference_result_serialization(self):
        """Test serialization of inference results for caching"""
        from src.ml_analysis.inference_optimizer import ResultSerializer
        
        serializer = ResultSerializer()
        
        # Test complex result serialization
        complex_result = {
            "predictions": np.array([0.1, 0.7, 0.2]),
            "confidence": 0.85,
            "model_metadata": {
                "version": "2.0.1",
                "timestamp": datetime.now(),
                "device": "cuda:0"
            },
            "feature_importance": np.array([0.3, 0.4, 0.3])
        }
        
        # Should serialize to bytes
        serialized = serializer.serialize(complex_result)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Should deserialize back to original
        deserialized = serializer.deserialize(serialized)
        assert deserialized is not None
        np.testing.assert_array_equal(
            deserialized["predictions"], 
            complex_result["predictions"]
        )
        assert deserialized["confidence"] == complex_result["confidence"]
    
    @pytest.mark.asyncio
    async def test_lstm_inference_optimization(self):
        """Test LSTM-specific inference optimization"""
        from src.ml_analysis.lstm_model import LSTMPricePredictor
        from src.ml_analysis.inference_optimizer import LSTMInferenceOptimizer
        
        # Create LSTM predictor
        config = {"sequence_length": 20, "hidden_size": 64}
        lstm_predictor = LSTMPricePredictor(config)
        
        # Create LSTM-specific optimizer
        optimizer = LSTMInferenceOptimizer(
            sequence_cache_size=100,
            enable_sequence_batching=True
        )
        
        # Test sequence-aware batching
        test_token = MockDiscoveredToken()
        
        # This should optimize sequence processing
        optimized_result = await optimizer.optimize_lstm_inference(
            predictor=lstm_predictor,
            token=test_token,
            enable_caching=True
        )
        
        assert "prediction_result" in optimized_result
        assert "optimization_metrics" in optimized_result
        assert "cache_hit" in optimized_result
    
    @pytest.mark.asyncio
    async def test_dqn_inference_optimization(self):
        """Test DQN-specific inference optimization"""
        from src.rl_agent.dqn_agent import DQNTradingAgent
        from src.ml_analysis.inference_optimizer import DQNInferenceOptimizer
        
        # Create mock DQN agent
        mock_config = Mock()
        mock_config.hidden_size = 128
        mock_config.batch_size = 32
        
        # Create DQN-specific optimizer
        optimizer = DQNInferenceOptimizer(
            action_cache_size=1000,
            enable_q_value_caching=True
        )
        
        # Test action prediction optimization
        test_state = MockMarketState()
        
        optimized_action = await optimizer.optimize_action_prediction(
            agent=Mock(),  # Mock agent
            state=test_state,
            enable_caching=True
        )
        
        assert "action" in optimized_action
        assert "confidence" in optimized_action
        assert "optimization_metrics" in optimized_action
        assert "cache_performance" in optimized_action
    
    def test_multi_model_inference_coordination(self):
        """Test coordinated inference across multiple models"""
        from src.ml_analysis.inference_optimizer import MultiModelCoordinator
        
        coordinator = MultiModelCoordinator()
        
        # Register models
        coordinator.register_model("lstm", Mock(), priority=1)
        coordinator.register_model("dqn", Mock(), priority=2)
        
        # Test model selection
        selected_model = coordinator.select_optimal_model(
            task_type="price_prediction",
            performance_requirements={"max_latency_ms": 100}
        )
        
        assert selected_model is not None
        assert "model_name" in selected_model
        assert "estimated_latency" in selected_model
        
        # Test load balancing
        load_metrics = coordinator.get_load_metrics()
        assert "active_models" in load_metrics
        assert "total_requests" in load_metrics
        assert "average_latency" in load_metrics
    
    @pytest.mark.asyncio
    async def test_inference_pipeline_end_to_end(self):
        """Test complete optimized inference pipeline end-to-end"""
        from src.ml_analysis.inference_optimizer import OptimizedInferencePipeline
        
        pipeline = OptimizedInferencePipeline(
            enable_gpu=True,
            enable_batching=True,
            enable_caching=True,
            enable_quantization=False,
            batch_size=16,
            cache_size_mb=128
        )
        
        # Test pipeline initialization
        await pipeline.initialize()
        assert pipeline.is_initialized is True
        
        # Test inference request processing
        test_requests = [
            {"model_type": "lstm", "input": MockMarketState()},
            {"model_type": "dqn", "input": MockMarketState()},
        ]
        
        results = await pipeline.process_requests(test_requests)
        
        assert len(results) == len(test_requests)
        for result in results:
            assert "prediction" in result
            assert "latency_ms" in result
            assert "cache_hit" in result
            assert "device_used" in result
        
        # Test pipeline metrics
        pipeline_metrics = await pipeline.get_metrics()
        assert "total_requests" in pipeline_metrics
        assert "average_latency" in pipeline_metrics
        assert "cache_hit_rate" in pipeline_metrics
        assert "gpu_utilization" in pipeline_metrics
        
        # Cleanup
        await pipeline.shutdown()

class TestGPUAcceleration:
    """Test suite for GPU acceleration capabilities"""
    
    def test_gpu_device_detection(self):
        """Test GPU device detection and capability assessment"""
        from src.ml_analysis.inference_optimizer import GPUManager
        
        gpu_manager = GPUManager()
        
        # Should detect GPU availability
        has_gpu = gpu_manager.has_gpu()
        assert isinstance(has_gpu, bool)
        
        if has_gpu:
            # Should provide GPU information
            gpu_info = gpu_manager.get_gpu_info()
            assert "device_count" in gpu_info
            assert "memory_total" in gpu_info
            assert "cuda_version" in gpu_info
            
            # Should allow GPU selection
            selected_gpu = gpu_manager.select_optimal_gpu()
            assert selected_gpu is not None
    
    def test_gpu_memory_management(self):
        """Test GPU memory management and optimization"""
        from src.ml_analysis.inference_optimizer import GPUMemoryManager
        
        memory_manager = GPUMemoryManager()
        
        # Test memory monitoring
        memory_info = memory_manager.get_memory_info()
        assert "allocated" in memory_info
        assert "cached" in memory_info
        assert "reserved" in memory_info
        
        # Test memory cleanup
        initial_memory = memory_manager.get_allocated_memory()
        memory_manager.clear_cache()
        final_memory = memory_manager.get_allocated_memory()
        
        # Memory should be same or less after cleanup
        assert final_memory <= initial_memory
    
    def test_cuda_error_handling(self):
        """Test CUDA error handling and recovery"""
        from src.ml_analysis.inference_optimizer import CUDAErrorHandler
        
        error_handler = CUDAErrorHandler()
        
        # Test error detection
        is_cuda_error = error_handler.is_cuda_error(RuntimeError("CUDA out of memory"))
        assert is_cuda_error is True
        
        is_not_cuda_error = error_handler.is_cuda_error(ValueError("Invalid input"))
        assert is_not_cuda_error is False
        
        # Test recovery strategies
        recovery_action = error_handler.get_recovery_action(
            RuntimeError("CUDA out of memory")
        )
        assert recovery_action in ["clear_cache", "reduce_batch_size", "fallback_cpu"]

class TestModelQuantization:
    """Test suite for model quantization capabilities"""
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization for inference speed up"""
        from src.ml_analysis.inference_optimizer import DynamicQuantizer
        
        quantizer = DynamicQuantizer()
        
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        # Test dynamic quantization
        quantized_model = quantizer.apply_dynamic_quantization(
            model=mock_model,
            dtype=torch.qint8
        )
        
        assert quantized_model is not None
        
        # Test quantization verification
        is_quantized = quantizer.verify_quantization(quantized_model)
        assert isinstance(is_quantized, bool)
    
    def test_static_quantization(self):
        """Test static quantization with calibration"""
        from src.ml_analysis.inference_optimizer import StaticQuantizer
        
        quantizer = StaticQuantizer()
        
        # Mock calibration data
        calibration_data = [torch.randn(1, 10) for _ in range(100)]
        
        # Test calibration
        quantizer.calibrate(
            model=Mock(),
            calibration_data=calibration_data
        )
        
        assert quantizer.is_calibrated is True
        
        # Test static quantization
        quantized_model = quantizer.apply_static_quantization(Mock())
        assert quantized_model is not None
    
    def test_quantization_aware_training_preparation(self):
        """Test preparation for quantization-aware training"""
        from src.ml_analysis.inference_optimizer import QATPreparator
        
        preparator = QATPreparator()
        
        # Test QAT model preparation
        qat_model = preparator.prepare_qat_model(
            model=Mock(),
            backend="fbgemm"
        )
        
        assert qat_model is not None
        
        # Test fake quantization insertion
        fake_quant_model = preparator.insert_fake_quantization(Mock())
        assert fake_quant_model is not None

class TestBatchProcessing:
    """Test suite for batch processing capabilities"""
    
    @pytest.mark.asyncio
    async def test_dynamic_batch_sizing(self):
        """Test dynamic batch sizing based on model capacity"""
        from src.ml_analysis.inference_optimizer import DynamicBatcher
        
        batcher = DynamicBatcher(
            min_batch_size=1,
            max_batch_size=64,
            target_latency_ms=100
        )
        
        # Test batch size adaptation
        inputs = [MockMarketState() for _ in range(20)]
        
        optimal_batch_size = await batcher.find_optimal_batch_size(
            model=Mock(),
            sample_inputs=inputs[:5],
            target_latency_ms=50
        )
        
        assert isinstance(optimal_batch_size, int)
        assert 1 <= optimal_batch_size <= 64
    
    @pytest.mark.asyncio  
    async def test_request_batching_with_timeout(self):
        """Test request batching with configurable timeout"""
        from src.ml_analysis.inference_optimizer import RequestBatcher
        
        batcher = RequestBatcher(
            batch_size=8,
            timeout_ms=100
        )
        
        # Add requests gradually
        request_futures = []
        for i in range(5):
            future = batcher.add_request(f"request_{i}", MockMarketState())
            request_futures.append(future)
        
        # Should batch and process requests
        results = await asyncio.gather(*request_futures, timeout=1.0)
        
        assert len(results) == 5
        for result in results:
            assert result is not None
    
    def test_batch_memory_optimization(self):
        """Test memory-efficient batch processing"""
        from src.ml_analysis.inference_optimizer import MemoryEfficientBatcher
        
        batcher = MemoryEfficientBatcher(
            max_memory_mb=512,
            memory_growth_factor=1.5
        )
        
        # Test memory estimation
        estimated_memory = batcher.estimate_batch_memory(
            batch_size=32,
            input_size=1000
        )
        
        assert isinstance(estimated_memory, float)
        assert estimated_memory > 0
        
        # Test batch size limiting based on memory
        safe_batch_size = batcher.get_safe_batch_size(
            input_size=2000,
            available_memory_mb=256
        )
        
        assert isinstance(safe_batch_size, int)
        assert safe_batch_size > 0