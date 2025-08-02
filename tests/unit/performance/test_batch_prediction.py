"""
Test-Driven Development for Batch Prediction Capabilities - Phase 5.2

This module contains comprehensive tests for batch prediction optimization including:
- Dynamic batch sizing and optimization
- Request batching with timeout handling
- Memory-efficient batch processing
- Batch result aggregation and distribution
- Load balancing across batch processing workers

Following TDD methodology - these tests will fail initially and drive the implementation.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

@dataclass
class BatchRequest:
    """Batch request for testing"""
    request_id: str
    model_id: str
    input_data: Any
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    timeout_ms: Optional[int] = None
    
    def __hash__(self):
        return hash(self.request_id)

@dataclass
class BatchResult:
    """Batch result for testing"""
    request_id: str
    prediction: Any
    confidence: float
    processing_time_ms: float
    batch_info: Dict[str, Any] = field(default_factory=dict)

class TestBatchProcessor:
    """Test suite for core batch processing functionality"""
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization with configuration"""
        from src.ml_analysis.batch_prediction import BatchProcessor, BatchConfig
        
        config = BatchConfig(
            max_batch_size=32,
            min_batch_size=1,
            batch_timeout_ms=100,
            enable_dynamic_batching=True,
            memory_limit_mb=512,
            max_queue_size=1000
        )
        
        processor = BatchProcessor(config)
        
        assert processor.config.max_batch_size == 32
        assert processor.config.min_batch_size == 1
        assert processor.config.batch_timeout_ms == 100
        assert processor.config.enable_dynamic_batching is True
        assert processor.request_queue is not None
        assert processor.batch_stats is not None
        assert processor.is_running is False
    
    @pytest.mark.asyncio
    async def test_request_queuing_and_batching(self):
        """Test request queuing and automatic batching"""
        from src.ml_analysis.batch_prediction import BatchProcessor
        
        processor = BatchProcessor()
        await processor.start()
        
        # Create test requests
        test_requests = []
        for i in range(10):
            request = BatchRequest(
                request_id=f"req_{i}",
                model_id="test_model",
                input_data=np.random.randn(1, 50),
                priority=1 if i < 5 else 2  # Mixed priorities
            )
            test_requests.append(request)
        
        # Submit requests
        submission_futures = []
        for request in test_requests:
            future = processor.submit_request(request)
            submission_futures.append(future)
        
        # Wait for batch processing
        results = await asyncio.gather(*submission_futures, timeout=5.0)
        
        assert len(results) == 10
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.request_id.startswith("req_")
            assert "batch_size" in result.batch_info
            assert "position_in_batch" in result.batch_info
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_timeout_based_batching(self):
        """Test batching based on timeout when batch size not reached"""
        from src.ml_analysis.batch_prediction import BatchProcessor, BatchConfig
        
        # Configure short timeout for testing
        config = BatchConfig(
            max_batch_size=10,
            batch_timeout_ms=50,  # Very short timeout
            enable_dynamic_batching=True
        )
        
        processor = BatchProcessor(config)
        await processor.start()
        
        # Submit only 3 requests (less than max batch size)
        requests = []
        futures = []
        
        for i in range(3):
            request = BatchRequest(
                request_id=f"timeout_req_{i}",
                model_id="timeout_model",
                input_data=np.random.randn(1, 20)
            )
            requests.append(request)
            
            future = processor.submit_request(request)
            futures.append(future)
        
        # Should complete due to timeout, not batch size
        start_time = time.time()
        results = await asyncio.gather(*futures, timeout=1.0)
        elapsed_time = (time.time() - start_time) * 1000
        
        # Should process within timeout window
        assert elapsed_time < 200  # Some margin for processing
        assert len(results) == 3
        
        # All results should be from same batch
        batch_ids = {result.batch_info.get("batch_id") for result in results}
        assert len(batch_ids) == 1  # Single batch
        
        await processor.stop()
    
    def test_priority_based_batching(self):
        """Test priority-based request ordering in batches"""
        from src.ml_analysis.batch_prediction import PriorityBatchScheduler
        
        scheduler = PriorityBatchScheduler()
        
        # Create requests with different priorities
        high_priority_requests = [
            BatchRequest(f"high_{i}", "model", f"data_{i}", priority=3)
            for i in range(5)
        ]
        
        low_priority_requests = [
            BatchRequest(f"low_{i}", "model", f"data_{i}", priority=1)
            for i in range(5)
        ]
        
        medium_priority_requests = [
            BatchRequest(f"med_{i}", "model", f"data_{i}", priority=2)
            for i in range(5)
        ]
        
        # Add requests in mixed order
        all_requests = low_priority_requests + high_priority_requests + medium_priority_requests
        np.random.shuffle(all_requests)
        
        for request in all_requests:
            scheduler.add_request(request)
        
        # Get next batch
        next_batch = scheduler.get_next_batch(batch_size=8)
        
        assert len(next_batch) == 8
        
        # Verify priority ordering (higher priority first)
        priorities = [req.priority for req in next_batch]
        assert priorities == sorted(priorities, reverse=True)
        
        # High priority requests should be first
        high_priority_count = sum(1 for req in next_batch if req.priority == 3)
        assert high_priority_count == 5  # All high priority requests included
    
    def test_batch_size_optimization(self):
        """Test dynamic batch size optimization based on performance"""
        from src.ml_analysis.batch_prediction import BatchSizeOptimizer
        
        optimizer = BatchSizeOptimizer()
        
        # Simulate performance data for different batch sizes
        performance_data = [
            {"batch_size": 1, "latency_ms": 50, "throughput": 20, "memory_mb": 100},
            {"batch_size": 4, "latency_ms": 80, "throughput": 50, "memory_mb": 200},
            {"batch_size": 8, "latency_ms": 120, "throughput": 80, "memory_mb": 350},
            {"batch_size": 16, "latency_ms": 200, "throughput": 120, "memory_mb": 600},
            {"batch_size": 32, "latency_ms": 350, "throughput": 150, "memory_mb": 1100},
        ]
        
        # Test optimal batch size calculation
        optimal_batch_size = optimizer.find_optimal_batch_size(
            performance_data=performance_data,
            latency_constraint_ms=250,
            memory_constraint_mb=800,
            optimization_target="throughput"
        )
        
        assert optimal_batch_size is not None
        assert 1 <= optimal_batch_size <= 32
        
        # Should respect memory constraint
        optimal_perf = next(p for p in performance_data if p["batch_size"] == optimal_batch_size)
        assert optimal_perf["memory_mb"] <= 800
        assert optimal_perf["latency_ms"] <= 250
        
        # Test adaptive batch sizing based on system load
        current_load = {
            "cpu_usage": 0.6,
            "memory_usage": 0.4,
            "queue_length": 50,
            "average_latency_ms": 100
        }
        
        adaptive_size = optimizer.calculate_adaptive_batch_size(
            current_load=current_load,
            base_batch_size=16,
            target_latency_ms=150,
            max_queue_length=100
        )
        
        assert isinstance(adaptive_size, int)
        assert 1 <= adaptive_size <= 64

class TestDynamicBatching:
    """Test suite for dynamic batching capabilities"""
    
    @pytest.mark.asyncio
    async def test_dynamic_batch_formation(self):
        """Test dynamic batch formation based on request patterns"""
        from src.ml_analysis.batch_prediction import DynamicBatcher
        
        batcher = DynamicBatcher(
            min_batch_size=2,
            max_batch_size=16,
            adaptive_timeout=True
        )
        
        # Test batch formation with varying request rates
        async def simulate_request_stream(request_rate_per_second, duration_seconds):
            requests = []
            request_interval = 1.0 / request_rate_per_second
            
            for i in range(int(duration_seconds * request_rate_per_second)):
                request = BatchRequest(
                    request_id=f"stream_req_{i}",
                    model_id="stream_model",
                    input_data=np.random.randn(1, 30)
                )
                requests.append(request)
                
                if i < int(duration_seconds * request_rate_per_second) - 1:
                    await asyncio.sleep(request_interval)
            
            return requests
        
        # Simulate high request rate
        high_rate_requests = await simulate_request_stream(20, 1.0)  # 20 req/sec
        
        # Process high rate requests
        batches = await batcher.form_batches_from_stream(high_rate_requests)
        
        assert len(batches) > 0
        
        # High rate should produce larger batches
        avg_batch_size = sum(len(batch) for batch in batches) / len(batches)
        assert avg_batch_size >= 8  # Should form reasonably large batches
        
        # Test low request rate
        low_rate_requests = await simulate_request_stream(2, 1.0)  # 2 req/sec
        low_rate_batches = await batcher.form_batches_from_stream(low_rate_requests)
        
        # Low rate should produce smaller batches due to timeout
        low_avg_batch_size = sum(len(batch) for batch in low_rate_batches) / len(low_rate_batches)
        assert low_avg_batch_size < avg_batch_size
    
    def test_batch_composition_optimization(self):
        """Test optimization of batch composition for similar requests"""
        from src.ml_analysis.batch_prediction import BatchCompositionOptimizer
        
        optimizer = BatchCompositionOptimizer()
        
        # Create requests with different characteristics
        lstm_requests = [
            BatchRequest(f"lstm_{i}", "lstm_model", np.random.randn(1, 100))
            for i in range(10)
        ]
        
        dqn_requests = [
            BatchRequest(f"dqn_{i}", "dqn_model", np.random.randn(1, 50))
            for i in range(8)
        ]
        
        cnn_requests = [
            BatchRequest(f"cnn_{i}", "cnn_model", np.random.randn(1, 3, 224, 224))
            for i in range(6)
        ]
        
        all_requests = lstm_requests + dqn_requests + cnn_requests
        np.random.shuffle(all_requests)
        
        # Test homogeneous batch formation
        homogeneous_batches = optimizer.create_homogeneous_batches(
            requests=all_requests,
            max_batch_size=8,
            similarity_threshold=0.8
        )
        
        assert len(homogeneous_batches) >= 3  # At least one per model type
        
        # Verify batch homogeneity
        for batch in homogeneous_batches:
            model_ids = {req.model_id for req in batch}
            assert len(model_ids) == 1  # All requests in batch have same model
        
        # Test heterogeneous batch optimization
        heterogeneous_batches = optimizer.create_optimized_heterogeneous_batches(
            requests=all_requests,
            max_batch_size=12,
            balance_factor=0.3
        )
        
        # Should create fewer batches by mixing compatible requests
        assert len(heterogeneous_batches) <= len(homogeneous_batches)
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_calculation(self):
        """Test adaptive timeout calculation based on system performance"""
        from src.ml_analysis.batch_prediction import AdaptiveTimeoutCalculator
        
        calculator = AdaptiveTimeoutCalculator()
        
        # Test timeout adaptation based on system metrics
        system_metrics = {
            "average_processing_time_ms": 80,
            "queue_length": 25,
            "cpu_utilization": 0.7,
            "memory_pressure": 0.4
        }
        
        adaptive_timeout = calculator.calculate_adaptive_timeout(
            base_timeout_ms=100,
            system_metrics=system_metrics,
            target_latency_ms=200,
            min_timeout_ms=50,
            max_timeout_ms=500
        )
        
        assert 50 <= adaptive_timeout <= 500
        
        # Test timeout under high load
        high_load_metrics = {
            "average_processing_time_ms": 150,
            "queue_length": 100,
            "cpu_utilization": 0.9,
            "memory_pressure": 0.8
        }
        
        high_load_timeout = calculator.calculate_adaptive_timeout(
            base_timeout_ms=100,
            system_metrics=high_load_metrics,
            target_latency_ms=200,
            min_timeout_ms=50,
            max_timeout_ms=500
        )
        
        # High load should result in shorter timeout to process smaller batches faster
        assert high_load_timeout <= adaptive_timeout
        
        # Test timeout history tracking
        timeout_history = calculator.get_timeout_adaptation_history()
        
        assert "adaptations" in timeout_history
        assert "average_timeout" in timeout_history
        assert "adaptation_frequency" in timeout_history

class TestMemoryEfficientBatching:
    """Test suite for memory-efficient batch processing"""
    
    def test_memory_aware_batch_sizing(self):
        """Test batch sizing based on memory constraints"""
        from src.ml_analysis.batch_prediction import MemoryAwareBatcher
        
        batcher = MemoryAwareBatcher(
            max_memory_mb=512,
            memory_safety_margin=0.2
        )
        
        # Test memory estimation for different input sizes
        small_requests = [
            BatchRequest(f"small_{i}", "model", np.random.randn(1, 10))
            for i in range(20)
        ]
        
        large_requests = [
            BatchRequest(f"large_{i}", "model", np.random.randn(1, 1000))
            for i in range(10)
        ]
        
        # Test batch sizing for small requests
        small_batch_size = batcher.calculate_memory_safe_batch_size(
            sample_requests=small_requests[:5],
            available_memory_mb=400
        )
        
        # Test batch sizing for large requests
        large_batch_size = batcher.calculate_memory_safe_batch_size(
            sample_requests=large_requests[:3],
            available_memory_mb=400
        )
        
        # Large requests should result in smaller batch size
        assert large_batch_size < small_batch_size
        assert large_batch_size >= 1  # Should allow at least one request
        
        # Test memory usage prediction
        memory_prediction = batcher.predict_batch_memory_usage(
            requests=small_requests[:small_batch_size],
            include_model_memory=True,
            include_intermediate_tensors=True
        )
        
        assert "total_memory_mb" in memory_prediction
        assert "input_memory_mb" in memory_prediction
        assert "model_memory_mb" in memory_prediction
        assert "intermediate_memory_mb" in memory_prediction
        
        # Predicted memory should be within limits
        assert memory_prediction["total_memory_mb"] <= 512 * 0.8  # With safety margin
    
    def test_streaming_batch_processing(self):
        """Test streaming batch processing for large datasets"""
        from src.ml_analysis.batch_prediction import StreamingBatchProcessor
        
        processor = StreamingBatchProcessor(
            batch_size=8,
            buffer_size=32,
            enable_prefetching=True
        )
        
        # Create large dataset simulation
        def large_dataset_generator():
            for i in range(1000):
                yield BatchRequest(
                    request_id=f"stream_{i}",
                    model_id="streaming_model", 
                    input_data=np.random.randn(1, 64)
                )
        
        # Test streaming processing
        processed_count = 0
        batch_count = 0
        
        async def process_batch(batch):
            nonlocal processed_count, batch_count
            # Simulate batch processing
            await asyncio.sleep(0.01)  # Simulate processing time
            processed_count += len(batch)
            batch_count += 1
            return [
                BatchResult(
                    request_id=req.request_id,
                    prediction=np.random.randn(3),
                    confidence=0.85,
                    processing_time_ms=10.0
                ) for req in batch
            ]
        
        # Process stream
        results = []
        async for batch in processor.process_stream(
            data_generator=large_dataset_generator(),
            batch_processor=process_batch,
            max_concurrent_batches=4
        ):
            results.extend(batch)
        
        assert processed_count == 1000
        assert len(results) == 1000
        assert batch_count > 100  # Should create many batches
    
    def test_memory_pool_optimization(self):
        """Test memory pool optimization for batch processing"""
        from src.ml_analysis.batch_prediction import BatchMemoryPool
        
        memory_pool = BatchMemoryPool(
            pool_size_mb=256,
            block_sizes=[1, 4, 16, 64],  # MB
            enable_garbage_collection=True
        )
        
        # Test memory allocation from pool
        allocation_results = []
        
        for size_mb in [1, 4, 16, 4, 1, 64, 16]:
            allocation = memory_pool.allocate(size_mb * 1024 * 1024)  # Convert to bytes
            allocation_results.append(allocation)
            assert allocation is not None
        
        # Test pool statistics
        pool_stats = memory_pool.get_pool_statistics()
        
        assert "total_allocations" in pool_stats
        assert "pool_hits" in pool_stats
        assert "external_allocations" in pool_stats
        assert "fragmentation_ratio" in pool_stats
        
        # Should have some pool hits for repeated sizes
        assert pool_stats["pool_hits"] > 0
        
        # Test memory defragmentation
        initial_fragmentation = pool_stats["fragmentation_ratio"]
        
        memory_pool.defragment()
        
        post_defrag_stats = memory_pool.get_pool_statistics()
        final_fragmentation = post_defrag_stats["fragmentation_ratio"]
        
        # Defragmentation should improve or maintain fragmentation ratio
        assert final_fragmentation <= initial_fragmentation
        
        # Test automatic garbage collection
        freed_memory = memory_pool.garbage_collect(
            aggressive=True,
            age_threshold_seconds=0
        )
        
        assert "freed_bytes" in freed_memory
        assert "freed_blocks" in freed_memory

class TestBatchResultAggregation:
    """Test suite for batch result aggregation and distribution"""
    
    @pytest.mark.asyncio
    async def test_result_aggregation_strategies(self):
        """Test different result aggregation strategies"""
        from src.ml_analysis.batch_prediction import ResultAggregator
        
        aggregator = ResultAggregator()
        
        # Create batch results with different characteristics
        batch_results = []
        for i in range(8):
            result = BatchResult(
                request_id=f"batch_req_{i}",
                prediction=np.random.randn(3),
                confidence=0.7 + i * 0.03,
                processing_time_ms=50.0 + i * 5.0,
                batch_info={"batch_id": "test_batch_1", "position": i}
            )
            batch_results.append(result)
        
        # Test ensemble aggregation
        ensemble_result = await aggregator.aggregate_ensemble_predictions(
            batch_results=batch_results,
            aggregation_method="weighted_average",
            confidence_weighting=True
        )
        
        assert "aggregated_prediction" in ensemble_result
        assert "confidence_score" in ensemble_result
        assert "individual_contributions" in ensemble_result
        
        # Test statistical aggregation
        stats_aggregation = aggregator.compute_batch_statistics(batch_results)
        
        assert "mean_confidence" in stats_aggregation
        assert "std_confidence" in stats_aggregation
        assert "mean_processing_time" in stats_aggregation
        assert "prediction_variance" in stats_aggregation
        
        # Test outlier detection in batch results
        outlier_analysis = aggregator.detect_batch_outliers(
            batch_results=batch_results,
            outlier_threshold=2.0  # Standard deviations
        )
        
        assert "outlier_indices" in outlier_analysis
        assert "outlier_scores" in outlier_analysis
        assert "outlier_reasons" in outlier_analysis
    
    def test_result_distribution_strategies(self):
        """Test strategies for distributing results back to requesters"""
        from src.ml_analysis.batch_prediction import ResultDistributor
        
        distributor = ResultDistributor()
        
        # Create mock result distribution scenario
        batch_results = [
            BatchResult(f"req_{i}", np.random.randn(2), 0.8, 25.0)
            for i in range(12)
        ]
        
        # Mock request callbacks/futures
        result_callbacks = {}
        for i in range(12):
            callback_mock = Mock()
            result_callbacks[f"req_{i}"] = callback_mock
        
        # Test synchronous distribution
        distribution_result = distributor.distribute_results_sync(
            batch_results=batch_results,
            result_callbacks=result_callbacks
        )
        
        assert "distributed_count" in distribution_result
        assert "failed_distributions" in distribution_result
        assert distribution_result["distributed_count"] == 12
        
        # Verify all callbacks were called
        for callback in result_callbacks.values():
            callback.assert_called_once()
        
        # Test priority-based distribution
        priority_distribution = distributor.distribute_with_priority(
            batch_results=batch_results,
            priority_mapping={f"req_{i}": 3-i//4 for i in range(12)},  # Decreasing priority
            max_concurrent_distributions=4
        )
        
        assert "priority_order" in priority_distribution
        assert len(priority_distribution["priority_order"]) == 12
        
        # High priority items should be first
        first_priorities = [priority_distribution["priority_order"][i] for i in range(4)]
        assert all(p >= 2 for p in first_priorities)
    
    @pytest.mark.asyncio
    async def test_streaming_result_distribution(self):
        """Test streaming distribution of results as they become available"""
        from src.ml_analysis.batch_prediction import StreamingResultDistributor
        
        distributor = StreamingResultDistributor(
            buffer_size=16,
            distribution_batch_size=4
        )
        
        # Simulate streaming results
        async def result_stream():
            for i in range(20):
                await asyncio.sleep(0.01)  # Simulate processing delay
                yield BatchResult(
                    request_id=f"stream_result_{i}",
                    prediction=[i * 0.1, 1 - i * 0.1],
                    confidence=0.8 + i * 0.01,
                    processing_time_ms=30.0 + i
                )
        
        # Collect distributed results
        distributed_results = []
        
        async def result_handler(result):
            distributed_results.append(result)
        
        # Test streaming distribution
        await distributor.distribute_streaming_results(
            result_stream=result_stream(),
            result_handler=result_handler,
            max_distribution_latency_ms=100
        )
        
        assert len(distributed_results) == 20
        
        # Results should be distributed in order
        for i, result in enumerate(distributed_results):
            assert result.request_id == f"stream_result_{i}"
        
        # Test distribution metrics
        distribution_metrics = distributor.get_distribution_metrics()
        
        assert "total_distributed" in distribution_metrics
        assert "average_distribution_latency" in distribution_metrics
        assert "distribution_throughput" in distribution_metrics

class TestBatchLoadBalancing:
    """Test suite for load balancing across batch processing workers"""
    
    def test_worker_pool_management(self):
        """Test management of batch processing worker pool"""
        from src.ml_analysis.batch_prediction import BatchWorkerPool
        
        worker_pool = BatchWorkerPool(
            min_workers=2,
            max_workers=8,
            auto_scaling=True,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3
        )
        
        # Test worker pool initialization
        assert worker_pool.min_workers == 2
        assert worker_pool.max_workers == 8
        assert worker_pool.current_worker_count >= 2
        
        # Test worker health monitoring
        worker_health = worker_pool.check_worker_health()
        
        assert "healthy_workers" in worker_health
        assert "unhealthy_workers" in worker_health
        assert "total_workers" in worker_health
        
        # All initial workers should be healthy
        assert worker_health["healthy_workers"] >= 2
        assert worker_health["unhealthy_workers"] == 0
        
        # Test worker scaling based on load
        high_load_metrics = {
            "queue_length": 100,
            "average_worker_utilization": 0.85,
            "average_response_time": 150
        }
        
        scaling_decision = worker_pool.evaluate_scaling_decision(high_load_metrics)
        
        assert "scale_action" in scaling_decision
        assert "target_worker_count" in scaling_decision
        assert "reasoning" in scaling_decision
        
        # High load should trigger scale up
        assert scaling_decision["scale_action"] in ["scale_up", "maintain"]
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self):
        """Test different load balancing strategies for batch distribution"""
        from src.ml_analysis.batch_prediction import BatchLoadBalancer
        
        load_balancer = BatchLoadBalancer(num_workers=4)
        
        # Create mock workers with different capabilities
        workers = []
        for i in range(4):
            worker_mock = Mock()
            worker_mock.id = f"worker_{i}"
            worker_mock.capacity = 8 + i * 2  # Varying capacities
            worker_mock.current_load = i * 2  # Varying current loads
            worker_mock.average_processing_time = 50 + i * 10
            workers.append(worker_mock)
        
        load_balancer.register_workers(workers)
        
        # Test round-robin distribution
        batches_to_distribute = [
            [BatchRequest(f"rr_req_{i}", "model", "data") for _ in range(4)]
            for i in range(8)
        ]
        
        round_robin_assignments = load_balancer.distribute_batches_round_robin(
            batches=batches_to_distribute
        )
        
        assert len(round_robin_assignments) == 8
        
        # Each worker should get 2 batches (8 batches / 4 workers)
        worker_assignments = {}
        for assignment in round_robin_assignments:
            worker_id = assignment["worker_id"]
            worker_assignments[worker_id] = worker_assignments.get(worker_id, 0) + 1
        
        assert len(worker_assignments) == 4
        assert all(count == 2 for count in worker_assignments.values())
        
        # Test least-loaded distribution
        least_loaded_assignments = load_balancer.distribute_batches_least_loaded(
            batches=batches_to_distribute
        )
        
        # Worker with lowest initial load should get more assignments
        worker_loads = {assignment["worker_id"]: assignment["estimated_load"] 
                       for assignment in least_loaded_assignments}
        
        # Verify load balancing effectiveness
        load_variance = np.var(list(worker_loads.values()))
        assert load_variance < 10  # Should be relatively balanced
    
    def test_adaptive_load_balancing(self):
        """Test adaptive load balancing based on worker performance"""
        from src.ml_analysis.batch_prediction import AdaptiveLoadBalancer
        
        load_balancer = AdaptiveLoadBalancer()
        
        # Simulate worker performance history
        performance_history = {
            "worker_1": {
                "average_latency": 45.0,
                "success_rate": 0.98,
                "throughput": 120,
                "error_count": 2
            },
            "worker_2": {
                "average_latency": 65.0,
                "success_rate": 0.95,
                "throughput": 100,
                "error_count": 5
            },
            "worker_3": {
                "average_latency": 40.0,
                "success_rate": 0.99,
                "throughput": 140,
                "error_count": 1
            }
        }
        
        # Test performance-based weight calculation
        worker_weights = load_balancer.calculate_performance_weights(
            performance_history=performance_history,
            weight_factors={
                "latency": 0.3,
                "success_rate": 0.4,
                "throughput": 0.3
            }
        )
        
        assert len(worker_weights) == 3
        assert all(0 <= weight <= 1 for weight in worker_weights.values())
        
        # Best performing worker (worker_3) should have highest weight
        best_worker = max(worker_weights.items(), key=lambda x: x[1])
        assert best_worker[0] == "worker_3"
        
        # Test adaptive batch assignment
        test_batches = [
            [BatchRequest(f"adaptive_req_{i}", "model", "data")]
            for i in range(12)
        ]
        
        adaptive_assignments = load_balancer.assign_batches_adaptively(
            batches=test_batches,
            worker_weights=worker_weights,
            load_balancing_strategy="weighted_random"
        )
        
        # Higher weight workers should get more assignments
        assignment_counts = {}
        for assignment in adaptive_assignments:
            worker_id = assignment["worker_id"]
            assignment_counts[worker_id] = assignment_counts.get(worker_id, 0) + 1
        
        # Verify adaptive distribution
        worker_3_assignments = assignment_counts.get("worker_3", 0)
        worker_2_assignments = assignment_counts.get("worker_2", 0)
        
        # Best worker should get more assignments than worst worker
        assert worker_3_assignments >= worker_2_assignments

class TestBatchPerformanceOptimization:
    """Test suite for batch processing performance optimization"""
    
    def test_batch_performance_profiling(self):
        """Test profiling of batch processing performance"""
        from src.ml_analysis.batch_prediction import BatchPerformanceProfiler
        
        profiler = BatchPerformanceProfiler()
        
        # Simulate batch processing with different configurations
        batch_configs = [
            {"batch_size": 4, "timeout_ms": 50},
            {"batch_size": 8, "timeout_ms": 100},
            {"batch_size": 16, "timeout_ms": 200},
            {"batch_size": 32, "timeout_ms": 400}
        ]
        
        performance_results = []
        
        for config in batch_configs:
            # Simulate performance data
            perf_data = {
                "batch_size": config["batch_size"],
                "timeout_ms": config["timeout_ms"],
                "average_latency_ms": config["batch_size"] * 8 + config["timeout_ms"] * 0.1,
                "throughput_batches_per_sec": 1000 / (config["batch_size"] * 8 + config["timeout_ms"] * 0.1),
                "memory_usage_mb": config["batch_size"] * 12,
                "cpu_utilization": min(0.95, config["batch_size"] * 0.03)
            }
            performance_results.append(perf_data)
        
        # Test performance analysis
        analysis_result = profiler.analyze_batch_performance(performance_results)
        
        assert "optimal_batch_size" in analysis_result
        assert "performance_bottlenecks" in analysis_result
        assert "scaling_characteristics" in analysis_result
        assert "resource_efficiency" in analysis_result
        
        # Test bottleneck identification
        bottleneck_analysis = profiler.identify_performance_bottlenecks(
            performance_data=performance_results,
            target_metrics={
                "latency_ms": 200,
                "throughput_batches_per_sec": 5.0,
                "memory_mb": 300,
                "cpu_utilization": 0.8
            }
        )
        
        assert "bottleneck_type" in bottleneck_analysis
        assert "severity" in bottleneck_analysis
        assert "recommendations" in bottleneck_analysis
    
    def test_batch_optimization_recommendations(self):
        """Test generation of batch optimization recommendations"""
        from src.ml_analysis.batch_prediction import BatchOptimizationAdvisor
        
        advisor = BatchOptimizationAdvisor()
        
        # Current system state
        current_state = {
            "average_batch_size": 12,
            "average_latency_ms": 180,
            "queue_length": 45,
            "throughput_requests_per_sec": 85,
            "memory_usage_mb": 420,
            "cpu_utilization": 0.65,
            "cache_hit_rate": 0.72
        }
        
        # Performance targets
        targets = {
            "max_latency_ms": 150,
            "min_throughput_requests_per_sec": 100,
            "max_memory_mb": 512,
            "target_cpu_utilization": 0.75,
            "target_cache_hit_rate": 0.85
        }
        
        # Test optimization recommendations
        recommendations = advisor.generate_optimization_recommendations(
            current_state=current_state,
            performance_targets=targets,
            system_constraints={
                "max_batch_size": 32,
                "available_memory_mb": 1024,
                "max_workers": 8
            }
        )
        
        assert "priority_recommendations" in recommendations
        assert "expected_improvements" in recommendations
        assert "implementation_complexity" in recommendations
        
        # Should recommend improvements for latency and throughput
        priority_recs = recommendations["priority_recommendations"]
        latency_improvements = [r for r in priority_recs if "latency" in r["target_metric"]]
        throughput_improvements = [r for r in priority_recs if "throughput" in r["target_metric"]]
        
        assert len(latency_improvements) > 0
        assert len(throughput_improvements) > 0
        
        # Test implementation planning
        implementation_plan = advisor.create_implementation_plan(
            recommendations=recommendations,
            available_resources={
                "development_time_hours": 40,
                "testing_time_hours": 16,
                "deployment_complexity": "medium"
            }
        )
        
        assert "implementation_phases" in implementation_plan
        assert "estimated_timeline" in implementation_plan
        assert "risk_assessment" in implementation_plan
        assert "rollback_strategy" in implementation_plan
    
    @pytest.mark.asyncio
    async def test_auto_tuning_system(self):
        """Test automatic tuning system for batch processing parameters"""
        from src.ml_analysis.batch_prediction import BatchAutoTuner
        
        auto_tuner = BatchAutoTuner(
            tuning_interval_minutes=5,
            performance_history_length=100,
            enable_conservative_tuning=True
        )
        
        # Simulate performance feedback
        performance_feedback = []
        for i in range(20):
            feedback = {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "batch_size": 8 + (i % 4) * 2,  # Varying batch sizes
                "latency_ms": 100 + (i % 4) * 20,
                "throughput": 80 + (i % 4) * 10,
                "success_rate": 0.95 + (i % 4) * 0.01
            }
            performance_feedback.append(feedback)
        
        # Test parameter tuning
        tuning_result = await auto_tuner.tune_parameters(
            performance_history=performance_feedback,
            current_parameters={
                "batch_size": 12,
                "timeout_ms": 150,
                "max_queue_size": 100
            },
            optimization_objectives=["latency", "throughput"]
        )
        
        assert "new_parameters" in tuning_result
        assert "expected_improvement" in tuning_result
        assert "confidence_score" in tuning_result
        assert "tuning_rationale" in tuning_result
        
        new_params = tuning_result["new_parameters"]
        assert "batch_size" in new_params
        assert "timeout_ms" in new_params
        
        # Test conservative tuning (small parameter changes)
        param_changes = {
            key: abs(new_params[key] - current_val) / current_val
            for key, current_val in {"batch_size": 12, "timeout_ms": 150}.items()
            if key in new_params
        }
        
        # Conservative tuning should make small adjustments
        assert all(change < 0.5 for change in param_changes.values())
        
        # Test A/B testing framework for parameter validation
        ab_test_plan = auto_tuner.create_ab_test_plan(
            current_parameters={"batch_size": 12, "timeout_ms": 150},
            candidate_parameters=new_params,
            test_duration_minutes=30,
            traffic_split=0.1  # 10% to new parameters
        )
        
        assert "test_configuration" in ab_test_plan
        assert "success_criteria" in ab_test_plan
        assert "monitoring_metrics" in ab_test_plan
        assert "rollback_triggers" in ab_test_plan