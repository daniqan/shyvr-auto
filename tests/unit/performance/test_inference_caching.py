"""
Test-Driven Development for Inference Caching System - Phase 5.2

This module contains comprehensive tests for inference result caching including:
- Multi-level caching (memory, disk, distributed)
- Cache invalidation and TTL management
- Performance-aware cache optimization
- Cache warming and prefetching strategies
- Cache consistency and concurrency handling

Following TDD methodology - these tests will fail initially and drive the implementation.
"""

import pytest
import asyncio
import time
import hashlib
import pickle
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MockInferenceRequest:
    """Mock inference request for testing"""
    model_id: str
    input_data: Any
    request_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_cache_key(self) -> str:
        """Generate cache key from request"""
        data_hash = hashlib.md5(str(self.input_data).encode()).hexdigest()
        return f"{self.model_id}:{data_hash}"

@dataclass
class MockInferenceResult:
    """Mock inference result for testing"""
    prediction: Any
    confidence: float
    latency_ms: float
    model_version: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TestInferenceCacheCore:
    """Test suite for core inference caching functionality"""
    
    def test_inference_cache_initialization(self):
        """Test InferenceCache initialization with configuration"""
        from src.ml_analysis.inference_caching import InferenceCache, CacheConfig
        
        config = CacheConfig(
            max_memory_mb=256,
            max_disk_mb=1024,
            default_ttl_seconds=3600,
            enable_compression=True,
            compression_threshold=1024,
            eviction_policy="lru"
        )
        
        cache = InferenceCache(config)
        
        assert cache.config.max_memory_mb == 256
        assert cache.config.max_disk_mb == 1024
        assert cache.config.default_ttl_seconds == 3600
        assert cache.config.enable_compression is True
        assert cache.memory_cache is not None
        assert cache.disk_cache is not None
        assert cache.cache_stats is not None
    
    def test_cache_key_generation(self):
        """Test cache key generation and collision handling"""
        from src.ml_analysis.inference_caching import CacheKeyGenerator
        
        key_generator = CacheKeyGenerator()
        
        # Test deterministic key generation
        request1 = MockInferenceRequest(
            model_id="lstm_v1", 
            input_data=[1, 2, 3, 4, 5],
            request_id="req1"
        )
        
        request2 = MockInferenceRequest(
            model_id="lstm_v1",
            input_data=[1, 2, 3, 4, 5],
            request_id="req2"  # Different request ID, same data
        )
        
        key1 = key_generator.generate_key(request1)
        key2 = key_generator.generate_key(request2)
        
        # Same input data should produce same key regardless of request ID
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0
        
        # Test collision detection
        collision_info = key_generator.check_collision_probability(
            num_models=100,
            avg_requests_per_day=10000
        )
        
        assert "collision_probability" in collision_info
        assert "recommended_key_length" in collision_info
        assert collision_info["collision_probability"] < 0.001  # Very low collision rate
    
    @pytest.mark.asyncio
    async def test_cache_put_and_get_operations(self):
        """Test basic cache put and get operations"""
        from src.ml_analysis.inference_caching import InferenceCache
        
        cache = InferenceCache()
        
        # Test cache put
        request = MockInferenceRequest(
            model_id="dqn_v2",
            input_data=np.array([1.0, 2.0, 3.0]),
            request_id="test_req"
        )
        
        result = MockInferenceResult(
            prediction=[0.7, 0.2, 0.1],
            confidence=0.85,
            latency_ms=45.2,
            model_version="v2.1.0"
        )
        
        cache_key = request.to_cache_key()
        
        success = await cache.put(cache_key, result, ttl_seconds=3600)
        assert success is True
        
        # Test cache get
        cached_result = await cache.get(cache_key)
        assert cached_result is not None
        assert cached_result.prediction == [0.7, 0.2, 0.1]
        assert cached_result.confidence == 0.85
        assert cached_result.model_version == "v2.1.0"
        
        # Test cache miss
        missing_result = await cache.get("nonexistent_key")
        assert missing_result is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_and_expiration(self):
        """Test cache TTL and automatic expiration"""
        from src.ml_analysis.inference_caching import InferenceCache
        
        cache = InferenceCache()
        
        # Test short TTL
        request = MockInferenceRequest(
            model_id="test_model",
            input_data="test_data",
            request_id="ttl_test"
        )
        
        result = MockInferenceResult(
            prediction="test_prediction",
            confidence=0.9,
            latency_ms=10.0,
            model_version="v1.0"
        )
        
        cache_key = request.to_cache_key()
        
        # Cache with 1 second TTL
        await cache.put(cache_key, result, ttl_seconds=1)
        
        # Should be available immediately
        cached_result = await cache.get(cache_key)
        assert cached_result is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        expired_result = await cache.get(cache_key)
        assert expired_result is None
        
        # Test TTL extension
        await cache.put(cache_key, result, ttl_seconds=3600)
        extended_success = await cache.extend_ttl(cache_key, additional_seconds=1800)
        assert extended_success is True
    
    def test_cache_compression_and_serialization(self):
        """Test cache compression and serialization of complex objects"""
        from src.ml_analysis.inference_caching import CacheSerializer
        
        serializer = CacheSerializer(enable_compression=True)
        
        # Test complex inference result serialization
        complex_result = MockInferenceResult(
            prediction=np.random.randn(1000, 100),  # Large numpy array
            confidence=0.92,
            latency_ms=78.5,
            model_version="v3.2.1",
            metadata={
                "feature_importance": np.random.randn(100),
                "attention_weights": np.random.randn(50, 50),
                "layer_activations": [np.random.randn(64) for _ in range(10)]
            }
        )
        
        # Test serialization
        serialized_data = serializer.serialize(complex_result)
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        
        # Test deserialization
        deserialized_result = serializer.deserialize(serialized_data)
        assert deserialized_result is not None
        assert deserialized_result.confidence == 0.92
        assert deserialized_result.model_version == "v3.2.1"
        np.testing.assert_array_equal(
            deserialized_result.prediction, 
            complex_result.prediction
        )
        
        # Test compression effectiveness
        compression_stats = serializer.get_compression_stats()
        assert "compression_ratio" in compression_stats
        assert "total_compressed_bytes" in compression_stats
        assert "total_uncompressed_bytes" in compression_stats
        
        # Should achieve reasonable compression on large arrays
        assert compression_stats["compression_ratio"] > 0.1

class TestMultiLevelCaching:
    """Test suite for multi-level caching hierarchy"""
    
    def test_memory_cache_implementation(self):
        """Test in-memory cache implementation with LRU eviction"""
        from src.ml_analysis.inference_caching import MemoryCache
        
        # Create small memory cache for testing eviction
        memory_cache = MemoryCache(max_size_mb=1, eviction_policy="lru")
        
        # Test basic operations
        test_data = MockInferenceResult(
            prediction=[0.5, 0.3, 0.2],
            confidence=0.8,
            latency_ms=20.0,
            model_version="v1.0"
        )
        
        success = memory_cache.put("key1", test_data)
        assert success is True
        
        retrieved_data = memory_cache.get("key1")
        assert retrieved_data is not None
        assert retrieved_data.confidence == 0.8
        
        # Test LRU eviction by adding multiple items
        large_data_items = []
        for i in range(10):
            large_result = MockInferenceResult(
                prediction=np.random.randn(100, 100),  # ~80KB each
                confidence=0.9,
                latency_ms=50.0,
                model_version=f"v{i}"
            )
            large_data_items.append(large_result)
            memory_cache.put(f"large_key_{i}", large_result)
        
        # Early items should be evicted due to size limit
        first_item = memory_cache.get("large_key_0")
        last_item = memory_cache.get("large_key_9")
        
        # Last item should still be there, first might be evicted
        assert last_item is not None
        
        # Test cache statistics
        stats = memory_cache.get_stats()
        assert "hit_rate" in stats
        assert "miss_rate" in stats
        assert "eviction_count" in stats
        assert "current_size_mb" in stats
    
    @pytest.mark.asyncio
    async def test_disk_cache_implementation(self):
        """Test disk-based cache implementation with persistence"""
        from src.ml_analysis.inference_caching import DiskCache
        
        # Use temporary directory for testing
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = DiskCache(
                cache_dir=temp_dir,
                max_size_mb=10,
                enable_compression=True
            )
            
            # Test disk cache operations
            test_result = MockInferenceResult(
                prediction=np.random.randn(500, 200),
                confidence=0.88,
                latency_ms=120.0,
                model_version="v2.0"
            )
            
            success = await disk_cache.put("disk_key_1", test_result)
            assert success is True
            
            # Verify file was created
            cache_files = os.listdir(temp_dir)
            assert len(cache_files) > 0
            
            # Test retrieval
            retrieved_result = await disk_cache.get("disk_key_1")
            assert retrieved_result is not None
            assert retrieved_result.confidence == 0.88
            assert retrieved_result.model_version == "v2.0"
            
            # Test persistence across cache instances
            disk_cache2 = DiskCache(cache_dir=temp_dir, max_size_mb=10)
            persistent_result = await disk_cache2.get("disk_key_1")
            assert persistent_result is not None
            assert persistent_result.confidence == 0.88
            
            # Test cleanup of expired items
            expired_count = await disk_cache.cleanup_expired(max_age_seconds=0)
            assert isinstance(expired_count, int)
    
    @pytest.mark.asyncio
    async def test_distributed_cache_implementation(self):
        """Test distributed cache implementation with Redis backend"""
        from src.ml_analysis.inference_caching import DistributedCache
        
        # Mock Redis client for testing
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.exists = AsyncMock(return_value=1)
        
        with patch('src.ml_analysis.inference_caching.aioredis.from_url', return_value=mock_redis):
            distributed_cache = DistributedCache(
                redis_url="redis://localhost:6379",
                key_prefix="inference_cache:",
                default_ttl=3600
            )
            
            await distributed_cache.initialize()
            
            # Test distributed cache operations
            test_result = MockInferenceResult(
                prediction=[0.6, 0.4],
                confidence=0.93,
                latency_ms=35.0,
                model_version="v1.5"
            )
            
            success = await distributed_cache.put("dist_key_1", test_result)
            assert success is True
            
            # Verify Redis set was called
            mock_redis.set.assert_called()
            
            # Test cache invalidation patterns
            invalidation_result = await distributed_cache.invalidate_pattern("dist_key_*")
            assert "invalidated_keys" in invalidation_result
            
            await distributed_cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_hierarchy_coordination(self):
        """Test coordination between multiple cache levels"""
        from src.ml_analysis.inference_caching import HierarchicalCache
        
        hierarchical_cache = HierarchicalCache(
            memory_cache_mb=64,
            disk_cache_mb=256,
            enable_distributed=False
        )
        
        await hierarchical_cache.initialize()
        
        # Test cache promotion (disk -> memory)
        test_result = MockInferenceResult(
            prediction=[0.8, 0.1, 0.1],
            confidence=0.95,
            latency_ms=15.0,
            model_version="v2.3"
        )
        
        # Store in disk cache first
        await hierarchical_cache.put("hierarchy_key", test_result, cache_level="disk")
        
        # First get should promote to memory
        retrieved_result = await hierarchical_cache.get("hierarchy_key")
        assert retrieved_result is not None
        
        # Verify it's now in memory cache
        memory_hit = hierarchical_cache.memory_cache.get("hierarchy_key")
        assert memory_hit is not None
        
        # Test cache statistics across levels
        hierarchy_stats = await hierarchical_cache.get_comprehensive_stats()
        
        assert "memory_stats" in hierarchy_stats
        assert "disk_stats" in hierarchy_stats
        assert "overall_hit_rate" in hierarchy_stats
        assert "promotion_rate" in hierarchy_stats

class TestCacheInvalidation:
    """Test suite for cache invalidation strategies"""
    
    def test_model_version_invalidation(self):
        """Test cache invalidation when model versions change"""
        from src.ml_analysis.inference_caching import ModelVersionTracker
        
        version_tracker = ModelVersionTracker()
        
        # Register model versions
        version_tracker.register_model_version("lstm", "v1.0.0")
        version_tracker.register_model_version("dqn", "v2.1.0")
        
        # Test version change detection
        version_changed = version_tracker.check_version_change("lstm", "v1.0.1")
        assert version_changed is True
        
        # Test invalidation key generation
        invalidation_keys = version_tracker.get_invalidation_keys_for_model("lstm")
        
        assert isinstance(invalidation_keys, list)
        assert len(invalidation_keys) > 0
        assert all("lstm" in key for key in invalidation_keys)
        
        # Test version compatibility checking
        is_compatible = version_tracker.is_version_compatible(
            cached_version="v1.0.0",
            current_version="v1.0.1",
            compatibility_mode="minor"
        )
        
        # Minor version bump should be compatible in minor mode
        assert is_compatible is True
        
        # Major version bump should not be compatible
        major_compatible = version_tracker.is_version_compatible(
            cached_version="v1.0.0", 
            current_version="v2.0.0",
            compatibility_mode="minor"
        )
        assert major_compatible is False
    
    @pytest.mark.asyncio
    async def test_time_based_invalidation(self):
        """Test time-based cache invalidation strategies"""
        from src.ml_analysis.inference_caching import TimeBasedInvalidator
        
        invalidator = TimeBasedInvalidator()
        
        # Test sliding window invalidation
        sliding_window_keys = await invalidator.get_sliding_window_invalidation(
            window_size_hours=6,
            model_types=["lstm", "dqn"]
        )
        
        assert isinstance(sliding_window_keys, list)
        
        # Test daily refresh invalidation
        daily_refresh_scheduled = await invalidator.schedule_daily_refresh_invalidation(
            refresh_time="02:00:00",
            model_patterns=["lstm_*", "dqn_*"]
        )
        
        assert "scheduled_time" in daily_refresh_scheduled
        assert "patterns_to_invalidate" in daily_refresh_scheduled
        
        # Test adaptive TTL based on model performance
        adaptive_ttl = invalidator.calculate_adaptive_ttl(
            model_id="lstm_v1",
            recent_accuracy=0.92,
            baseline_accuracy=0.95,
            base_ttl_seconds=3600
        )
        
        # Lower accuracy should result in shorter TTL
        assert adaptive_ttl < 3600
        assert adaptive_ttl > 0
    
    def test_dependency_based_invalidation(self):
        """Test invalidation based on data dependencies"""
        from src.ml_analysis.inference_caching import DependencyTracker
        
        dependency_tracker = DependencyTracker()
        
        # Register data dependencies
        dependency_tracker.register_dependency(
            cache_key="prediction_key_1",
            dependencies=["market_data", "user_preferences", "model_parameters"]
        )
        
        # Test dependency change detection
        changed_dependencies = dependency_tracker.check_dependency_changes(
            dependency_name="market_data",
            last_update_time=datetime.now() - timedelta(hours=2)
        )
        
        assert "affected_cache_keys" in changed_dependencies
        assert "change_severity" in changed_dependencies
        
        # Test cascade invalidation
        cascade_keys = dependency_tracker.get_cascade_invalidation_keys(
            changed_dependency="model_parameters"
        )
        
        assert isinstance(cascade_keys, list)
        # Should include keys that depend on model_parameters
        
        # Test dependency graph analysis
        dependency_graph = dependency_tracker.analyze_dependency_graph()
        
        assert "nodes" in dependency_graph
        assert "edges" in dependency_graph
        assert "critical_dependencies" in dependency_graph

class TestCacheWarmingAndPrefetching:
    """Test suite for cache warming and prefetching strategies"""
    
    @pytest.mark.asyncio
    async def test_cache_warming_on_startup(self):
        """Test cache warming strategies during system startup"""
        from src.ml_analysis.inference_caching import CacheWarmer
        
        cache_warmer = CacheWarmer()
        
        # Mock historical usage patterns
        usage_patterns = [
            {"model_id": "lstm_v1", "input_pattern": "price_data", "frequency": 0.8},
            {"model_id": "dqn_v2", "input_pattern": "market_state", "frequency": 0.6},
            {"model_id": "lstm_v1", "input_pattern": "volume_data", "frequency": 0.4}
        ]
        
        # Test cache warming based on usage patterns
        warmed_entries = await cache_warmer.warm_cache_from_patterns(
            usage_patterns=usage_patterns,
            warmup_threshold=0.5,
            max_warm_entries=100
        )
        
        assert "warmed_count" in warmed_entries
        assert "skipped_count" in warmed_entries
        assert "errors_count" in warmed_entries
        
        # Should warm entries with frequency >= 0.5
        assert warmed_entries["warmed_count"] >= 2
        
        # Test critical path warming
        critical_models = ["lstm_v1", "dqn_v2"]
        critical_warming = await cache_warmer.warm_critical_models(
            model_ids=critical_models,
            sample_inputs_per_model=5
        )
        
        assert "successfully_warmed" in critical_warming
        assert len(critical_warming["successfully_warmed"]) > 0
    
    @pytest.mark.asyncio
    async def test_predictive_prefetching(self):
        """Test predictive prefetching based on usage patterns"""
        from src.ml_analysis.inference_caching import PredictivePrefetcher
        
        prefetcher = PredictivePrefetcher()
        
        # Mock recent request history
        request_history = [
            MockInferenceRequest("lstm_v1", "data_1", f"req_{i}") 
            for i in range(10)
        ] + [
            MockInferenceRequest("dqn_v2", "state_1", f"req_{i+10}")
            for i in range(15)
        ]
        
        # Test pattern-based prefetching
        prefetch_predictions = await prefetcher.predict_next_requests(
            request_history=request_history,
            prediction_window_minutes=30,
            confidence_threshold=0.7
        )
        
        assert "predicted_requests" in prefetch_predictions
        assert "confidence_scores" in prefetch_predictions
        
        predicted_requests = prefetch_predictions["predicted_requests"]
        assert len(predicted_requests) > 0
        
        # Should predict more DQN requests based on higher frequency
        dqn_predictions = [r for r in predicted_requests if "dqn" in r["model_id"]]
        assert len(dqn_predictions) > 0
        
        # Test seasonal prefetching
        seasonal_patterns = await prefetcher.analyze_seasonal_patterns(
            request_history=request_history,
            time_granularity="hourly"
        )
        
        assert "hourly_patterns" in seasonal_patterns
        assert "peak_hours" in seasonal_patterns
        assert "model_preferences" in seasonal_patterns
    
    def test_adaptive_prefetching_strategies(self):
        """Test adaptive prefetching based on system performance"""
        from src.ml_analysis.inference_caching import AdaptivePrefetcher
        
        prefetcher = AdaptivePrefetcher()
        
        # Test resource-aware prefetching
        current_resources = {
            "cpu_usage": 0.4,
            "memory_usage": 0.6,
            "cache_hit_rate": 0.75,
            "average_latency_ms": 85.0
        }
        
        prefetch_strategy = prefetcher.adapt_prefetching_strategy(
            current_resources=current_resources,
            target_hit_rate=0.85,
            max_prefetch_cpu_usage=0.2
        )
        
        assert "prefetch_enabled" in prefetch_strategy
        assert "prefetch_aggressiveness" in prefetch_strategy
        assert "resource_limits" in prefetch_strategy
        
        # Should enable prefetching given current resource availability
        assert prefetch_strategy["prefetch_enabled"] is True
        
        # Test backoff strategy under high load
        high_load_resources = {
            "cpu_usage": 0.9,
            "memory_usage": 0.85,
            "cache_hit_rate": 0.65,
            "average_latency_ms": 200.0
        }
        
        backoff_strategy = prefetcher.adapt_prefetching_strategy(
            current_resources=high_load_resources,
            target_hit_rate=0.85,
            max_prefetch_cpu_usage=0.2
        )
        
        # Should reduce or disable prefetching under high load
        assert backoff_strategy["prefetch_aggressiveness"] < prefetch_strategy["prefetch_aggressiveness"]

class TestCachePerformanceOptimization:
    """Test suite for cache performance optimization"""
    
    def test_cache_performance_monitoring(self):
        """Test cache performance monitoring and metrics collection"""
        from src.ml_analysis.inference_caching import CachePerformanceMonitor
        
        monitor = CachePerformanceMonitor()
        
        # Simulate cache operations
        for i in range(100):
            if i % 3 == 0:  # 33% hit rate
                monitor.record_cache_hit("memory", latency_ms=5.0)
            elif i % 5 == 0:  # Additional disk hits
                monitor.record_cache_hit("disk", latency_ms=15.0)
            else:
                monitor.record_cache_miss(latency_ms=100.0)
        
        # Test performance metrics
        performance_metrics = monitor.get_performance_metrics()
        
        assert "overall_hit_rate" in performance_metrics
        assert "memory_hit_rate" in performance_metrics
        assert "disk_hit_rate" in performance_metrics
        assert "average_hit_latency" in performance_metrics
        assert "average_miss_latency" in performance_metrics
        
        # Verify calculated metrics
        assert 0.0 <= performance_metrics["overall_hit_rate"] <= 1.0
        assert performance_metrics["average_hit_latency"] < performance_metrics["average_miss_latency"]
        
        # Test performance trend analysis
        trend_analysis = monitor.analyze_performance_trends(
            time_window_hours=24,
            trend_granularity="hourly"
        )
        
        assert "hit_rate_trend" in trend_analysis
        assert "latency_trend" in trend_analysis
        assert "performance_regression_alerts" in trend_analysis
    
    def test_cache_size_optimization(self):
        """Test automatic cache size optimization"""
        from src.ml_analysis.inference_caching import CacheSizeOptimizer
        
        optimizer = CacheSizeOptimizer()
        
        # Test optimal size calculation
        current_performance = {
            "hit_rate": 0.75,
            "average_latency_ms": 50.0,
            "cache_size_mb": 256,
            "eviction_rate": 0.1
        }
        
        optimal_size = optimizer.calculate_optimal_cache_size(
            current_performance=current_performance,
            target_hit_rate=0.85,
            available_memory_mb=1024,
            cost_per_mb=0.001  # Cost factor for memory usage
        )
        
        assert "recommended_memory_mb" in optimal_size
        assert "recommended_disk_mb" in optimal_size
        assert "expected_hit_rate" in optimal_size
        assert "cost_benefit_ratio" in optimal_size
        
        # Recommended size should be larger to achieve higher hit rate
        assert optimal_size["recommended_memory_mb"] >= current_performance["cache_size_mb"]
        
        # Test dynamic size adjustment
        adjustment_strategy = optimizer.create_dynamic_adjustment_strategy(
            performance_history=[current_performance],
            memory_pressure=0.6,
            performance_targets={"hit_rate": 0.85, "max_latency_ms": 75}
        )
        
        assert "adjustment_direction" in adjustment_strategy
        assert "adjustment_magnitude" in adjustment_strategy
        assert "adjustment_timeline" in adjustment_strategy
    
    def test_cache_hotspot_detection(self):
        """Test detection and optimization of cache hotspots"""
        from src.ml_analysis.inference_caching import HotspotDetector
        
        detector = HotspotDetector()
        
        # Simulate hotspot access patterns
        access_patterns = []
        
        # Create hotspot: 80% of requests to 20% of keys
        hotspot_keys = [f"hotspot_key_{i}" for i in range(20)]
        normal_keys = [f"normal_key_{i}" for i in range(80)]
        
        for _ in range(800):  # 80% hotspot requests
            key = np.random.choice(hotspot_keys)
            access_patterns.append({
                "cache_key": key,
                "timestamp": datetime.now(),
                "access_frequency": 1
            })
        
        for _ in range(200):  # 20% normal requests
            key = np.random.choice(normal_keys)
            access_patterns.append({
                "cache_key": key,
                "timestamp": datetime.now(),
                "access_frequency": 1
            })
        
        # Test hotspot detection
        hotspot_analysis = detector.detect_hotspots(
            access_patterns=access_patterns,
            hotspot_threshold=0.05  # Keys accessed >5% of the time
        )
        
        assert "hotspot_keys" in hotspot_analysis
        assert "hotspot_intensity" in hotspot_analysis
        assert "access_distribution" in hotspot_analysis
        
        # Should detect the artificial hotspots
        detected_hotspots = hotspot_analysis["hotspot_keys"]
        assert len(detected_hotspots) > 0
        
        # Most detected hotspots should be from our hotspot key set
        hotspot_overlap = len([k for k in detected_hotspots if k in hotspot_keys])
        assert hotspot_overlap > len(detected_hotspots) * 0.8
        
        # Test hotspot optimization strategies
        optimization_strategies = detector.recommend_hotspot_optimizations(
            hotspot_analysis=hotspot_analysis,
            current_cache_config={"memory_mb": 256, "disk_mb": 512}
        )
        
        expected_strategies = [
            "increase_memory_cache", "implement_cache_partitioning",
            "add_dedicated_hotspot_cache", "implement_cache_warming"
        ]
        
        assert "recommended_strategies" in optimization_strategies
        recommended = optimization_strategies["recommended_strategies"]
        
        # Should recommend relevant optimization strategies
        assert any(strategy["name"] in expected_strategies for strategy in recommended)

class TestCacheConcurrencyAndConsistency:
    """Test suite for cache concurrency handling and consistency"""
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self):
        """Test cache behavior under concurrent read/write operations"""
        from src.ml_analysis.inference_caching import ConcurrentCache
        
        cache = ConcurrentCache(max_size_mb=64, enable_locking=True)
        
        await cache.initialize()
        
        # Test concurrent writes to same key
        async def concurrent_write(cache_instance, key, value, writer_id):
            await asyncio.sleep(np.random.uniform(0, 0.1))  # Random delay
            return await cache_instance.put(f"{key}_{writer_id}", value)
        
        # Launch concurrent write operations
        write_tasks = []
        for i in range(10):
            test_result = MockInferenceResult(
                prediction=[i * 0.1, (10-i) * 0.1],
                confidence=0.8 + i * 0.01,
                latency_ms=20.0 + i,
                model_version=f"v1.{i}"
            )
            
            task = concurrent_write(cache, "concurrent_key", test_result, i)
            write_tasks.append(task)
        
        write_results = await asyncio.gather(*write_tasks)
        
        # All writes should succeed
        assert all(result is True for result in write_results)
        
        # Test concurrent reads
        async def concurrent_read(cache_instance, key, reader_id):
            await asyncio.sleep(np.random.uniform(0, 0.05))
            return await cache_instance.get(f"{key}_{reader_id}")
        
        read_tasks = []
        for i in range(10):
            task = concurrent_read(cache, "concurrent_key", i)
            read_tasks.append(task)
        
        read_results = await asyncio.gather(*read_tasks)
        
        # All reads should return valid results
        valid_reads = [r for r in read_results if r is not None]
        assert len(valid_reads) == 10  # All keys should exist
    
    @pytest.mark.asyncio
    async def test_cache_consistency_across_levels(self):
        """Test consistency maintenance across cache hierarchy levels"""
        from src.ml_analysis.inference_caching import ConsistentHierarchicalCache
        
        cache = ConsistentHierarchicalCache(
            enable_consistency_checks=True,
            consistency_mode="eventual"
        )
        
        await cache.initialize()
        
        # Test write propagation across levels
        test_result = MockInferenceResult(
            prediction=[0.9, 0.1],
            confidence=0.95,
            latency_ms=25.0,
            model_version="v2.0"
        )
        
        await cache.put("consistency_key", test_result, propagate_levels=["memory", "disk"])
        
        # Verify consistency across levels
        memory_result = await cache.get_from_level("consistency_key", "memory")
        disk_result = await cache.get_from_level("consistency_key", "disk")
        
        assert memory_result is not None
        assert disk_result is not None
        assert memory_result.confidence == disk_result.confidence
        assert memory_result.model_version == disk_result.model_version
        
        # Test consistency repair
        repair_results = await cache.repair_consistency_violations(
            scan_all_keys=True,
            repair_strategy="use_latest_version"
        )
        
        assert "violations_found" in repair_results
        assert "violations_repaired" in repair_results
        assert "repair_success_rate" in repair_results
    
    def test_cache_locking_mechanisms(self):
        """Test cache locking mechanisms for critical sections"""
        from src.ml_analysis.inference_caching import CacheLockManager
        
        lock_manager = CacheLockManager()
        
        # Test exclusive lock acquisition
        with lock_manager.exclusive_lock("test_key", timeout_seconds=5.0) as lock:
            assert lock.is_acquired() is True
            assert lock.lock_type == "exclusive"
            
            # Test lock metadata
            lock_info = lock.get_lock_info()
            assert "lock_id" in lock_info
            assert "acquired_at" in lock_info
            assert "expires_at" in lock_info
        
        # Lock should be released after context exit
        assert lock.is_acquired() is False
        
        # Test shared lock acquisition
        shared_locks = []
        for i in range(3):
            shared_lock = lock_manager.shared_lock(f"shared_key", timeout_seconds=5.0)
            shared_locks.append(shared_lock)
        
        # Multiple shared locks should be possible
        with shared_locks[0]:
            with shared_locks[1]:
                with shared_locks[2]:
                    assert all(lock.is_acquired() for lock in shared_locks)
        
        # Test deadlock detection
        deadlock_detector = lock_manager.get_deadlock_detector()
        
        potential_deadlocks = deadlock_detector.detect_potential_deadlocks(
            active_locks=[
                {"lock_id": "lock1", "key": "key_a", "waiting_for": "key_b"},
                {"lock_id": "lock2", "key": "key_b", "waiting_for": "key_a"},
            ]
        )
        
        assert "deadlock_cycles" in potential_deadlocks
        assert len(potential_deadlocks["deadlock_cycles"]) > 0