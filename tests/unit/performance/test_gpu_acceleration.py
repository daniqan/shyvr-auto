"""
Test-Driven Development for GPU Acceleration - Phase 5.2

This module contains comprehensive tests for GPU acceleration with CPU fallback including:
- GPU device detection and management
- CUDA memory optimization
- Automatic CPU fallback mechanisms
- GPU performance monitoring
- Multi-GPU support and load balancing

Following TDD methodology - these tests will fail initially and drive the implementation.
"""

import pytest
import torch
import time
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class MockTensor:
    """Mock tensor for testing"""
    shape: tuple
    device: str = "cpu"
    dtype: str = "float32"
    
    def to(self, device: str):
        """Mock tensor device transfer"""
        return MockTensor(self.shape, device, self.dtype)
    
    def cuda(self):
        """Mock CUDA transfer"""
        return MockTensor(self.shape, "cuda:0", self.dtype)
    
    def cpu(self):
        """Mock CPU transfer"""
        return MockTensor(self.shape, "cpu", self.dtype)

class TestGPUDeviceManager:
    """Test suite for GPU device detection and management"""
    
    def test_gpu_device_manager_initialization(self):
        """Test GPUDeviceManager initialization and device detection"""
        from src.ml_analysis.gpu_acceleration import GPUDeviceManager
        
        device_manager = GPUDeviceManager()
        
        # Should detect available devices
        assert hasattr(device_manager, 'available_devices')
        assert isinstance(device_manager.available_devices, list)
        
        # Should have device capabilities info
        assert hasattr(device_manager, 'device_capabilities')
        assert isinstance(device_manager.device_capabilities, dict)
        
        # Should determine if GPU is available
        has_gpu = device_manager.has_gpu()
        assert isinstance(has_gpu, bool)
        
        # Should provide primary device
        primary_device = device_manager.get_primary_device()
        assert primary_device is not None
        assert isinstance(primary_device, str)
    
    def test_gpu_capability_detection(self):
        """Test GPU capability detection and assessment"""
        from src.ml_analysis.gpu_acceleration import GPUDeviceManager
        
        device_manager = GPUDeviceManager()
        
        if device_manager.has_gpu():
            # Should provide detailed GPU information
            gpu_info = device_manager.get_gpu_info()
            
            required_fields = [
                "device_count", "cuda_version", "driver_version",
                "total_memory", "compute_capability"
            ]
            
            for field in required_fields:
                assert field in gpu_info
            
            # Should assess GPU suitability for ML workloads
            is_suitable = device_manager.is_gpu_suitable_for_ml()
            assert isinstance(is_suitable, bool)
            
            # Should provide memory information per device
            for device_id in range(gpu_info["device_count"]):
                memory_info = device_manager.get_device_memory_info(device_id)
                assert "total" in memory_info
                assert "free" in memory_info
                assert "used" in memory_info
    
    def test_device_selection_strategy(self):
        """Test intelligent device selection based on workload"""
        from src.ml_analysis.gpu_acceleration import GPUDeviceManager
        
        device_manager = GPUDeviceManager()
        
        # Test device selection for inference workload
        inference_device = device_manager.select_device_for_inference(
            model_size_mb=150,
            batch_size=32,
            sequence_length=50
        )
        
        assert inference_device is not None
        assert isinstance(inference_device, str)
        
        # Test device selection for training workload
        training_device = device_manager.select_device_for_training(
            model_size_mb=500,
            batch_size=16,
            gradient_accumulation_steps=4
        )
        
        assert training_device is not None
        assert isinstance(training_device, str)
        
        # Test fallback to CPU when GPU insufficient
        large_model_device = device_manager.select_device_for_inference(
            model_size_mb=32000,  # Very large model
            batch_size=128
        )
        
        # Should fallback to CPU or provide null if no suitable device
        assert large_model_device is not None
    
    @pytest.mark.asyncio
    async def test_device_health_monitoring(self):
        """Test GPU device health monitoring and diagnostics"""
        from src.ml_analysis.gpu_acceleration import GPUHealthMonitor
        
        health_monitor = GPUHealthMonitor()
        
        # Test device health check
        health_status = await health_monitor.check_device_health("cuda:0")
        
        required_health_fields = [
            "is_healthy", "temperature", "utilization", "memory_usage",
            "error_count", "last_error"
        ]
        
        for field in required_health_fields:
            assert field in health_status
        
        # Test health monitoring over time
        monitoring_results = []
        for _ in range(3):
            result = await health_monitor.monitor_device("cuda:0", duration_seconds=1)
            monitoring_results.append(result)
            await asyncio.sleep(0.1)
        
        assert len(monitoring_results) == 3
        for result in monitoring_results:
            assert "timestamp" in result
            assert "metrics" in result
    
    def test_multi_gpu_detection_and_management(self):
        """Test multi-GPU detection and load balancing"""
        from src.ml_analysis.gpu_acceleration import MultiGPUManager
        
        multi_gpu_manager = MultiGPUManager()
        
        # Should detect multiple GPUs if available
        gpu_count = multi_gpu_manager.get_gpu_count()
        assert isinstance(gpu_count, int)
        assert gpu_count >= 0
        
        if gpu_count > 1:
            # Test load balancing across GPUs
            load_balancer = multi_gpu_manager.get_load_balancer()
            assert load_balancer is not None
            
            # Test device assignment for multiple requests
            device_assignments = []
            for i in range(10):
                device = load_balancer.assign_device(request_id=f"req_{i}")
                device_assignments.append(device)
            
            # Should distribute across available GPUs
            unique_devices = set(device_assignments)
            assert len(unique_devices) <= gpu_count
            
            # Test load metrics
            load_metrics = load_balancer.get_load_metrics()
            assert "device_loads" in load_metrics
            assert "total_requests" in load_metrics

class TestGPUMemoryManager:
    """Test suite for GPU memory management and optimization"""
    
    def test_gpu_memory_manager_initialization(self):
        """Test GPU memory manager initialization"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryManager
        
        memory_manager = GPUMemoryManager(
            device="cuda:0",
            cache_size_mb=512,
            enable_memory_pooling=True
        )
        
        assert memory_manager.device == "cuda:0"
        assert memory_manager.cache_size_mb == 512
        assert memory_manager.enable_memory_pooling is True
        
        # Should track memory usage
        assert hasattr(memory_manager, 'allocated_memory')
        assert hasattr(memory_manager, 'cached_memory')
    
    def test_memory_allocation_and_deallocation(self):
        """Test GPU memory allocation and deallocation tracking"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryManager
        
        memory_manager = GPUMemoryManager()
        
        # Test memory allocation tracking
        initial_memory = memory_manager.get_allocated_memory()
        assert isinstance(initial_memory, float)
        
        # Simulate tensor allocation
        tensor_id = memory_manager.allocate_tensor(
            size=(1000, 1000),
            dtype=torch.float32,
            device="cuda:0"
        )
        
        post_allocation_memory = memory_manager.get_allocated_memory()
        assert post_allocation_memory >= initial_memory
        
        # Test memory deallocation
        memory_manager.deallocate_tensor(tensor_id)
        
        final_memory = memory_manager.get_allocated_memory()
        # Memory should be same or less after deallocation
        assert final_memory <= post_allocation_memory
    
    def test_memory_cache_management(self):
        """Test GPU memory cache management and optimization"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryManager
        
        memory_manager = GPUMemoryManager(cache_size_mb=256)
        
        # Test cache operations
        cache_info = memory_manager.get_cache_info()
        assert "size_mb" in cache_info
        assert "utilization" in cache_info
        assert "hit_rate" in cache_info
        
        # Test cache cleanup
        initial_cache_size = cache_info["size_mb"]
        
        memory_manager.clear_cache()
        
        post_cleanup_cache = memory_manager.get_cache_info()
        assert post_cleanup_cache["size_mb"] <= initial_cache_size
        
        # Test automatic cache management
        memory_manager.enable_auto_cache_management(
            max_cache_size_mb=128,
            cleanup_threshold=0.8
        )
        
        auto_managed = memory_manager.is_auto_managed()
        assert auto_managed is True
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling of GPU memory pressure situations"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryManager
        
        memory_manager = GPUMemoryManager()
        
        # Test memory pressure detection
        is_under_pressure = memory_manager.is_memory_under_pressure()
        assert isinstance(is_under_pressure, bool)
        
        # Test memory pressure response
        if is_under_pressure:
            response_actions = await memory_manager.handle_memory_pressure()
            
            expected_actions = [
                "clear_cache", "reduce_batch_size", "enable_gradient_checkpointing",
                "offload_to_cpu", "fragment_defragmentation"
            ]
            
            assert isinstance(response_actions, list)
            for action in response_actions:
                assert action in expected_actions
        
        # Test memory fragmentation detection and handling
        fragmentation_info = memory_manager.get_fragmentation_info()
        assert "fragmentation_ratio" in fragmentation_info
        assert "largest_free_block" in fragmentation_info
        
        if fragmentation_info["fragmentation_ratio"] > 0.3:
            await memory_manager.defragment_memory()
            
            post_defrag_info = memory_manager.get_fragmentation_info()
            # Fragmentation should improve after defragmentation
            assert post_defrag_info["fragmentation_ratio"] <= fragmentation_info["fragmentation_ratio"]
    
    def test_memory_pool_optimization(self):
        """Test memory pool optimization for repeated allocations"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryPool
        
        memory_pool = GPUMemoryPool(
            device="cuda:0",
            pool_size_mb=1024,
            block_sizes=[1, 4, 16, 64, 256]  # MB
        )
        
        # Test pool initialization
        pool_info = memory_pool.get_pool_info()
        assert "total_size_mb" in pool_info
        assert "available_blocks" in pool_info
        assert "allocation_stats" in pool_info
        
        # Test efficient allocation from pool
        allocations = []
        for size_mb in [1, 4, 16, 4, 1]:  # Mixed sizes
            allocation = memory_pool.allocate(size_mb * 1024 * 1024)  # Convert to bytes
            allocations.append(allocation)
            assert allocation is not None
        
        # Test pool statistics
        stats = memory_pool.get_allocation_stats()
        assert "total_allocations" in stats
        assert "pool_hits" in stats
        assert "external_allocations" in stats
        
        # Pool hits should be > 0 for repeated allocations
        assert stats["pool_hits"] > 0
        
        # Test memory release back to pool
        for allocation in allocations:
            memory_pool.deallocate(allocation)
        
        final_pool_info = memory_pool.get_pool_info()
        # Available blocks should increase after deallocation
        assert len(final_pool_info["available_blocks"]) >= len(pool_info["available_blocks"])

class TestCPUFallback:
    """Test suite for automatic CPU fallback mechanisms"""
    
    def test_fallback_trigger_detection(self):
        """Test detection of scenarios requiring CPU fallback"""
        from src.ml_analysis.gpu_acceleration import CPUFallbackManager
        
        fallback_manager = CPUFallbackManager()
        
        # Test CUDA error detection
        cuda_error = RuntimeError("CUDA out of memory")
        should_fallback = fallback_manager.should_fallback_to_cpu(cuda_error)
        assert should_fallback is True
        
        # Test non-CUDA error
        regular_error = ValueError("Invalid input shape")
        should_not_fallback = fallback_manager.should_fallback_to_cpu(regular_error)
        assert should_not_fallback is False
        
        # Test memory pressure fallback
        memory_fallback = fallback_manager.should_fallback_due_to_memory(
            required_memory_mb=8000,
            available_memory_mb=1000
        )
        assert memory_fallback is True
        
        # Test device unavailability fallback
        device_fallback = fallback_manager.should_fallback_due_to_device(
            device="cuda:5",  # Non-existent device
            max_devices=2
        )
        assert device_fallback is True
    
    @pytest.mark.asyncio
    async def test_automatic_fallback_execution(self):
        """Test automatic execution of CPU fallback"""
        from src.ml_analysis.gpu_acceleration import AutoFallbackExecutor
        
        executor = AutoFallbackExecutor(
            enable_fallback=True,
            fallback_timeout_seconds=5.0
        )
        
        # Mock function that fails on GPU but works on CPU
        async def mock_gpu_inference(device: str):
            if "cuda" in device:
                raise RuntimeError("CUDA out of memory")
            return {"prediction": [0.5, 0.3, 0.2], "device_used": device}
        
        # Test automatic fallback
        result = await executor.execute_with_fallback(
            func=mock_gpu_inference,
            primary_device="cuda:0",
            fallback_device="cpu"
        )
        
        assert result is not None
        assert result["device_used"] == "cpu"
        assert "fallback_reason" in result
        assert result["fallback_reason"] == "CUDA out of memory"
    
    def test_fallback_performance_tracking(self):
        """Test tracking of fallback performance and patterns"""
        from src.ml_analysis.gpu_acceleration import FallbackTracker
        
        tracker = FallbackTracker()
        
        # Record fallback events
        tracker.record_fallback(
            original_device="cuda:0",
            fallback_device="cpu",
            reason="CUDA out of memory",
            latency_impact_ms=150.5
        )
        
        tracker.record_fallback(
            original_device="cuda:1",
            fallback_device="cpu", 
            reason="Device not available",
            latency_impact_ms=75.2
        )
        
        # Get fallback statistics
        stats = tracker.get_fallback_stats()
        
        assert "total_fallbacks" in stats
        assert "fallback_rate" in stats
        assert "average_latency_impact" in stats
        assert "common_reasons" in stats
        
        assert stats["total_fallbacks"] == 2
        assert "CUDA out of memory" in stats["common_reasons"]
        
        # Test fallback pattern analysis
        patterns = tracker.analyze_fallback_patterns()
        assert "time_based_patterns" in patterns
        assert "device_failure_rates" in patterns
        assert "reason_frequency" in patterns
    
    def test_smart_device_selection_with_fallback(self):
        """Test smart device selection considering fallback history"""
        from src.ml_analysis.gpu_acceleration import SmartDeviceSelector
        
        selector = SmartDeviceSelector()
        
        # Record device reliability history
        selector.record_device_performance("cuda:0", success=True, latency_ms=25.5)
        selector.record_device_performance("cuda:0", success=False, error="CUDA error")
        selector.record_device_performance("cuda:1", success=True, latency_ms=30.2)
        selector.record_device_performance("cpu", success=True, latency_ms=120.0)
        
        # Test device selection based on reliability
        selected_device = selector.select_best_device(
            available_devices=["cuda:0", "cuda:1", "cpu"],
            workload_type="inference",
            reliability_weight=0.7,
            performance_weight=0.3
        )
        
        assert selected_device is not None
        assert selected_device in ["cuda:0", "cuda:1", "cpu"]
        
        # Test device ranking
        device_rankings = selector.rank_devices(
            available_devices=["cuda:0", "cuda:1", "cpu"],
            criteria=["reliability", "performance", "availability"]
        )
        
        assert len(device_rankings) == 3
        for ranking in device_rankings:
            assert "device" in ranking
            assert "score" in ranking
            assert "reliability" in ranking

class TestGPUPerformanceMonitoring:
    """Test suite for GPU performance monitoring and optimization"""
    
    def test_gpu_utilization_monitoring(self):
        """Test real-time GPU utilization monitoring"""
        from src.ml_analysis.gpu_acceleration import GPUUtilizationMonitor
        
        monitor = GPUUtilizationMonitor()
        
        # Test current utilization reading
        utilization = monitor.get_current_utilization("cuda:0")
        
        required_metrics = [
            "gpu_utilization", "memory_utilization", "temperature",
            "power_draw", "clock_speed"
        ]
        
        for metric in required_metrics:
            assert metric in utilization
            assert isinstance(utilization[metric], (int, float))
            assert utilization[metric] >= 0
        
        # Test utilization history tracking
        history = monitor.get_utilization_history("cuda:0", duration_minutes=5)
        
        assert "timestamps" in history
        assert "gpu_utilization" in history
        assert "memory_utilization" in history
        assert len(history["timestamps"]) > 0
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self):
        """Test GPU performance benchmarking for model workloads"""
        from src.ml_analysis.gpu_acceleration import GPUBenchmark
        
        benchmark = GPUBenchmark()
        
        # Test inference benchmark
        inference_results = await benchmark.benchmark_inference(
            model_type="lstm",
            input_shapes=[(1, 50, 10), (8, 50, 10), (32, 50, 10)],
            device="cuda:0"
        )
        
        assert "throughput_samples_per_sec" in inference_results
        assert "latency_ms" in inference_results
        assert "memory_usage_mb" in inference_results
        assert "batch_performance" in inference_results
        
        # Test memory bandwidth benchmark
        memory_benchmark = await benchmark.benchmark_memory_bandwidth(
            device="cuda:0",
            test_sizes=[1, 10, 100, 1000]  # MB
        )
        
        assert "read_bandwidth_gbps" in memory_benchmark
        assert "write_bandwidth_gbps" in memory_benchmark
        assert "copy_bandwidth_gbps" in memory_benchmark
        
        # Test compute benchmark
        compute_benchmark = await benchmark.benchmark_compute_performance(
            device="cuda:0",
            operations=["matrix_multiply", "convolution", "attention"]
        )
        
        for operation in ["matrix_multiply", "convolution", "attention"]:
            assert operation in compute_benchmark
            assert "gflops" in compute_benchmark[operation]
            assert "latency_ms" in compute_benchmark[operation]
    
    def test_thermal_monitoring_and_throttling(self):
        """Test GPU thermal monitoring and throttling detection"""
        from src.ml_analysis.gpu_acceleration import ThermalMonitor
        
        thermal_monitor = ThermalMonitor()
        
        # Test temperature monitoring
        temperature_info = thermal_monitor.get_temperature_info("cuda:0")
        
        assert "current_temp" in temperature_info
        assert "max_temp" in temperature_info
        assert "critical_temp" in temperature_info
        assert "fan_speed" in temperature_info
        
        # Test throttling detection
        throttling_status = thermal_monitor.check_throttling("cuda:0")
        
        assert "is_throttling" in throttling_status
        assert "throttle_reasons" in throttling_status
        assert "performance_impact" in throttling_status
        
        # Test thermal alerts
        alerts = thermal_monitor.check_thermal_alerts(
            device="cuda:0",
            warning_temp=80,
            critical_temp=90
        )
        
        assert isinstance(alerts, list)
        for alert in alerts:
            assert "severity" in alert
            assert "message" in alert
            assert "temperature" in alert
    
    def test_power_consumption_monitoring(self):
        """Test GPU power consumption monitoring and optimization"""
        from src.ml_analysis.gpu_acceleration import PowerMonitor
        
        power_monitor = PowerMonitor()
        
        # Test power consumption reading
        power_info = power_monitor.get_power_info("cuda:0")
        
        assert "current_power_watts" in power_info
        assert "max_power_watts" in power_info
        assert "power_limit_watts" in power_info
        assert "efficiency_gflops_per_watt" in power_info
        
        # Test power optimization recommendations
        optimization_suggestions = power_monitor.get_optimization_suggestions(
            device="cuda:0",
            target_efficiency=10.0  # GFLOPS/Watt
        )
        
        expected_suggestions = [
            "reduce_clock_speed", "lower_power_limit", "optimize_batch_size",
            "enable_mixed_precision", "use_tensor_cores"
        ]
        
        assert isinstance(optimization_suggestions, list)
        for suggestion in optimization_suggestions:
            assert suggestion in expected_suggestions or isinstance(suggestion, dict)

class TestGPUInferenceOptimization:
    """Test suite for GPU-specific inference optimizations"""
    
    @pytest.mark.asyncio
    async def test_tensor_core_utilization(self):
        """Test automatic Tensor Core utilization for supported operations"""
        from src.ml_analysis.gpu_acceleration import TensorCoreOptimizer
        
        optimizer = TensorCoreOptimizer()
        
        # Test Tensor Core availability detection
        has_tensor_cores = optimizer.has_tensor_cores("cuda:0")
        assert isinstance(has_tensor_cores, bool)
        
        if has_tensor_cores:
            # Test operation optimization for Tensor Cores
            mock_model = Mock()
            mock_model.half = Mock(return_value=mock_model)  # FP16 conversion
            
            optimized_model = await optimizer.optimize_for_tensor_cores(
                model=mock_model,
                enable_mixed_precision=True
            )
            
            assert optimized_model is not None
            
            # Verify mixed precision was enabled
            mock_model.half.assert_called_once()
            
            # Test Tensor Core performance validation
            performance_gain = await optimizer.validate_tensor_core_performance(
                original_model=Mock(),
                optimized_model=optimized_model,
                test_input=MockTensor((1, 512, 512))
            )
            
            assert "speedup_ratio" in performance_gain
            assert "tensor_core_utilization" in performance_gain
    
    def test_cuda_kernel_optimization(self):
        """Test custom CUDA kernel optimization for specific operations"""
        from src.ml_analysis.gpu_acceleration import CUDAKernelOptimizer
        
        optimizer = CUDAKernelOptimizer()
        
        # Test kernel compilation and caching
        kernel_id = optimizer.compile_kernel(
            kernel_name="optimized_attention",
            source_code="__global__ void optimized_attention() { /* mock kernel */ }",
            compile_options=["-O3", "--use_fast_math"]
        )
        
        assert kernel_id is not None
        assert isinstance(kernel_id, str)
        
        # Test kernel execution
        execution_result = optimizer.execute_kernel(
            kernel_id=kernel_id,
            grid_size=(1, 1, 1),
            block_size=(256, 1, 1),
            args=[]
        )
        
        assert "execution_time_ms" in execution_result
        assert "success" in execution_result
        
        # Test kernel performance profiling
        profile_results = optimizer.profile_kernel(
            kernel_id=kernel_id,
            iterations=100
        )
        
        assert "average_time_ms" in profile_results
        assert "peak_memory_usage" in profile_results
        assert "occupancy" in profile_results
    
    def test_mixed_precision_optimization(self):
        """Test automatic mixed precision optimization"""
        from src.ml_analysis.gpu_acceleration import MixedPrecisionOptimizer
        
        optimizer = MixedPrecisionOptimizer()
        
        # Test model analysis for mixed precision compatibility
        mock_model = Mock()
        compatibility_report = optimizer.analyze_model_compatibility(mock_model)
        
        assert "compatible_operations" in compatibility_report
        assert "incompatible_operations" in compatibility_report
        assert "expected_speedup" in compatibility_report
        assert "memory_savings" in compatibility_report
        
        # Test automatic mixed precision application
        optimized_model = optimizer.apply_mixed_precision(
            model=mock_model,
            optimization_level="aggressive"
        )
        
        assert optimized_model is not None
        
        # Test mixed precision training preparation
        training_config = optimizer.prepare_for_mixed_precision_training(
            model=optimized_model,
            loss_scaling="dynamic"
        )
        
        assert "scaler" in training_config
        assert "loss_scale" in training_config
        assert "growth_factor" in training_config
    
    @pytest.mark.asyncio
    async def test_cuda_graph_optimization(self):
        """Test CUDA graph optimization for repetitive inference patterns"""
        from src.ml_analysis.gpu_acceleration import CUDAGraphOptimizer
        
        optimizer = CUDAGraphOptimizer()
        
        # Test CUDA graph creation from model
        mock_model = Mock()
        mock_input = MockTensor((1, 100))
        
        graph_info = await optimizer.create_cuda_graph(
            model=mock_model,
            sample_input=mock_input,
            warmup_iterations=10
        )
        
        assert "graph_id" in graph_info
        assert "capture_success" in graph_info
        assert "memory_usage" in graph_info
        
        if graph_info["capture_success"]:
            # Test graph execution
            execution_result = await optimizer.execute_cuda_graph(
                graph_id=graph_info["graph_id"],
                input_data=mock_input
            )
            
            assert "output" in execution_result
            assert "execution_time_ms" in execution_result
            
            # Test performance comparison
            perf_comparison = await optimizer.compare_graph_performance(
                graph_id=graph_info["graph_id"],
                regular_model=mock_model,
                test_inputs=[mock_input] * 100
            )
            
            assert "graph_avg_time" in perf_comparison
            assert "regular_avg_time" in perf_comparison
            assert "speedup_ratio" in perf_comparison
    
    def test_gpu_memory_optimization(self):
        """Test GPU memory optimization techniques"""
        from src.ml_analysis.gpu_acceleration import GPUMemoryOptimizer
        
        optimizer = GPUMemoryOptimizer()
        
        # Test gradient checkpointing setup
        mock_model = Mock()
        checkpointed_model = optimizer.enable_gradient_checkpointing(
            model=mock_model,
            checkpoint_ratio=0.5
        )
        
        assert checkpointed_model is not None
        
        # Test activation offloading
        offload_config = optimizer.setup_activation_offloading(
            model=mock_model,
            offload_device="cpu",
            offload_threshold_mb=100
        )
        
        assert "offload_layers" in offload_config
        assert "memory_savings_mb" in offload_config
        assert "latency_overhead_ms" in offload_config
        
        # Test memory fragmentation reduction
        fragmentation_result = optimizer.reduce_memory_fragmentation(
            device="cuda:0",
            defragment_threshold=0.3
        )
        
        assert "fragmentation_before" in fragmentation_result
        assert "fragmentation_after" in fragmentation_result
        assert "memory_recovered_mb" in fragmentation_result

class TestAsyncGPUOperations:
    """Test suite for asynchronous GPU operations and streaming"""
    
    @pytest.mark.asyncio
    async def test_cuda_stream_management(self):
        """Test CUDA stream management for concurrent operations"""
        from src.ml_analysis.gpu_acceleration import CUDAStreamManager
        
        stream_manager = CUDAStreamManager(max_streams=4)
        
        # Test stream creation and allocation
        stream_id = stream_manager.allocate_stream("inference_stream")
        assert stream_id is not None
        assert isinstance(stream_id, str)
        
        # Test concurrent stream operations
        async def mock_gpu_operation(stream_id: str, duration_ms: int):
            await asyncio.sleep(duration_ms / 1000)
            return f"operation_completed_on_{stream_id}"
        
        # Launch concurrent operations on different streams
        tasks = []
        for i in range(3):
            stream_id = stream_manager.allocate_stream(f"stream_{i}")
            task = mock_gpu_operation(stream_id, 100)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        
        # Test stream synchronization
        await stream_manager.synchronize_all_streams()
        
        # Test stream cleanup
        cleanup_result = stream_manager.cleanup_unused_streams()
        assert "cleaned_streams" in cleanup_result
        assert "active_streams" in cleanup_result
    
    @pytest.mark.asyncio
    async def test_asynchronous_inference_pipeline(self):
        """Test asynchronous inference pipeline with GPU acceleration"""
        from src.ml_analysis.gpu_acceleration import AsyncGPUInferencePipeline
        
        pipeline = AsyncGPUInferencePipeline(
            max_concurrent_requests=8,
            enable_batching=True,
            batch_timeout_ms=50
        )
        
        await pipeline.initialize()
        
        # Test concurrent inference requests
        test_inputs = [MockTensor((1, 100)) for _ in range(10)]
        
        inference_tasks = []
        for i, input_tensor in enumerate(test_inputs):
            task = pipeline.submit_inference_request(
                request_id=f"req_{i}",
                input_data=input_tensor,
                model_id="test_model"
            )
            inference_tasks.append(task)
        
        results = await asyncio.gather(*inference_tasks, timeout=5.0)
        
        assert len(results) == 10
        for result in results:
            assert "prediction" in result
            assert "latency_ms" in result
            assert "device_used" in result
        
        # Test pipeline metrics
        metrics = await pipeline.get_performance_metrics()
        assert "total_requests" in metrics
        assert "average_latency" in metrics
        assert "throughput_per_second" in metrics
        assert "gpu_utilization" in metrics
        
        await pipeline.shutdown()
    
    def test_memory_transfer_optimization(self):
        """Test optimized CPU-GPU memory transfers"""
        from src.ml_analysis.gpu_acceleration import MemoryTransferOptimizer
        
        optimizer = MemoryTransferOptimizer()
        
        # Test pinned memory allocation for faster transfers
        pinned_buffer = optimizer.allocate_pinned_memory(
            size_bytes=1024 * 1024,  # 1MB
            dtype="float32"
        )
        
        assert pinned_buffer is not None
        assert hasattr(pinned_buffer, 'is_pinned')
        
        # Test asynchronous transfer
        mock_data = np.random.randn(1000, 100).astype(np.float32)
        
        transfer_result = optimizer.async_transfer_to_device(
            data=mock_data,
            device="cuda:0",
            stream_id="transfer_stream"
        )
        
        assert "transfer_id" in transfer_result
        assert "estimated_time_ms" in transfer_result
        
        # Test transfer completion checking
        is_complete = optimizer.is_transfer_complete(transfer_result["transfer_id"])
        assert isinstance(is_complete, bool)
        
        # Test transfer performance optimization
        optimization_report = optimizer.optimize_transfer_performance(
            data_size_mb=100,
            transfer_frequency=10,  # transfers per second
            source_device="cpu",
            target_device="cuda:0"
        )
        
        assert "recommended_batch_size" in optimization_report
        assert "optimal_pinned_memory_size" in optimization_report
        assert "expected_bandwidth_gbps" in optimization_report