"""
Test suite for model preservation caching system - Phase 2.2

This module tests the multi-level caching system including:
- In-memory LRU cache with 2GB limit
- Local disk cache at /tmp/models
- GCS API with retry logic
- Cache warming on startup
- Prefetching for likely models
- Cache metrics and monitoring
- Model compression optimization

Follows TDD methodology with failing tests first.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Import the modules we're testing (these will initially fail)
from src.model_preservation.base import ModelMetadata, PreservationPriority, ModelState
from src.model_preservation.caching import (
    CacheLevel,
    CacheConfig,
    CacheMetrics,
    CacheEntry,
    LRUCache,
    DiskCache,
    GCSCache,
    CacheManager,
    CacheWarmer,
    ModelPrefetcher,
    CacheMonitor,
    CompressionOptimizer
)


class TestCacheLevel:
    """Test cache level enumeration"""
    
    def test_cache_level_values(self):
        """Test cache level enum values"""
        assert CacheLevel.MEMORY.value == "memory"
        assert CacheLevel.DISK.value == "disk"
        assert CacheLevel.GCS.value == "gcs"
    
    def test_cache_level_priority_order(self):
        """Test cache levels have correct priority ordering"""
        # Memory should be fastest (lowest priority number)
        # Disk should be medium
        # GCS should be slowest (highest priority number)
        assert CacheLevel.MEMORY.priority < CacheLevel.DISK.priority
        assert CacheLevel.DISK.priority < CacheLevel.GCS.priority


class TestCacheConfig:
    """Test cache configuration"""
    
    def test_cache_config_creation(self):
        """Test creating cache configuration with defaults"""
        config = CacheConfig()
        assert config.memory_limit_gb == 2.0
        assert config.disk_cache_dir == "/tmp/models"
        assert config.enable_memory_cache is True
        assert config.enable_disk_cache is True
        assert config.enable_gcs_cache is True
        assert config.cache_warmup_enabled is True
        assert config.prefetch_enabled is True
        assert config.compression_enabled is True
        assert config.metrics_enabled is True
    
    def test_cache_config_validation(self):
        """Test cache configuration validation"""
        # Should raise error for invalid memory limit
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            CacheConfig(memory_limit_gb=-1.0)
        
        # Should raise error for invalid disk cache directory
        with pytest.raises(ValueError, match="Disk cache directory cannot be empty"):
            CacheConfig(disk_cache_dir="")
    
    def test_cache_config_custom_values(self):
        """Test creating cache configuration with custom values"""
        config = CacheConfig(
            memory_limit_gb=4.0,
            disk_cache_dir="/custom/cache",
            ttl_memory_seconds=7200,
            ttl_disk_seconds=86400 * 7,
            max_disk_size_gb=20.0,
            compression_level=9
        )
        assert config.memory_limit_gb == 4.0
        assert config.disk_cache_dir == "/custom/cache"
        assert config.ttl_memory_seconds == 7200
        assert config.ttl_disk_seconds == 86400 * 7
        assert config.max_disk_size_gb == 20.0
        assert config.compression_level == 9


class TestCacheMetrics:
    """Test cache metrics collection"""
    
    def test_cache_metrics_creation(self):
        """Test creating cache metrics with defaults"""
        metrics = CacheMetrics()
        assert metrics.total_requests == 0
        assert metrics.memory_hits == 0
        assert metrics.disk_hits == 0
        assert metrics.gcs_hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.memory_usage_bytes == 0
        assert metrics.disk_usage_bytes == 0
        assert metrics.average_response_time_ms == 0.0
    
    def test_cache_metrics_calculations(self):
        """Test cache metrics calculations"""
        metrics = CacheMetrics(
            total_requests=100,
            memory_hits=40,
            disk_hits=30,
            gcs_hits=20,
            misses=10
        )
        
        assert metrics.hit_rate == 0.90  # (40+30+20)/100
        assert metrics.miss_rate == 0.10  # 10/100
        assert metrics.memory_hit_rate == 0.40  # 40/100
        assert metrics.disk_hit_rate == 0.30  # 30/100
        assert metrics.gcs_hit_rate == 0.20  # 20/100
    
    def test_cache_metrics_update(self):
        """Test updating cache metrics"""
        metrics = CacheMetrics()
        
        # Record memory hit
        metrics.record_hit(CacheLevel.MEMORY, 50.0)
        assert metrics.total_requests == 1
        assert metrics.memory_hits == 1
        assert metrics.average_response_time_ms == 50.0
        
        # Record disk hit
        metrics.record_hit(CacheLevel.DISK, 150.0)
        assert metrics.total_requests == 2
        assert metrics.disk_hits == 1
        assert metrics.average_response_time_ms == 100.0  # (50+150)/2
        
        # Record miss
        metrics.record_miss(500.0)
        assert metrics.total_requests == 3
        assert metrics.misses == 1
        assert metrics.miss_rate == 1/3


class TestCacheEntry:
    """Test cache entry data structure"""
    
    def test_cache_entry_creation(self):
        """Test creating cache entry"""
        model_data = b"test model data"
        metadata = {"model_type": "lstm", "version": "v1.0.0"}
        
        entry = CacheEntry(
            key="lstm-v1.0.0",
            data=model_data,
            metadata=metadata,
            size_bytes=len(model_data),
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        assert entry.key == "lstm-v1.0.0"
        assert entry.data == model_data
        assert entry.metadata == metadata
        assert entry.size_bytes == len(model_data)
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.last_accessed, datetime)
    
    def test_cache_entry_expiry(self):
        """Test cache entry expiry logic"""
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        
        entry = CacheEntry(
            key="test",
            data=b"data",
            metadata={},
            size_bytes=4,
            created_at=old_time,
            last_accessed=old_time
        )
        
        # Should be expired with 1 hour TTL
        assert entry.is_expired(ttl_seconds=3600)
        
        # Should not be expired with 3 hour TTL
        assert not entry.is_expired(ttl_seconds=3600 * 3)
    
    def test_cache_entry_access_update(self):
        """Test updating cache entry access time"""
        entry = CacheEntry(
            key="test",
            data=b"data",
            metadata={},
            size_bytes=4,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        old_access_time = entry.last_accessed
        time.sleep(0.01)  # Small delay to ensure time difference
        
        entry.touch()
        assert entry.last_accessed > old_access_time


class TestLRUCache:
    """Test in-memory LRU cache with 2GB limit"""
    
    def test_lru_cache_creation(self):
        """Test creating LRU cache"""
        cache = LRUCache(max_size_bytes=2 * 1024 * 1024 * 1024)  # 2GB
        assert cache.max_size_bytes == 2 * 1024 * 1024 * 1024
        assert cache.current_size_bytes == 0
        assert len(cache) == 0
    
    def test_lru_cache_put_get(self):
        """Test basic put/get operations"""
        cache = LRUCache(max_size_bytes=1024)
        
        # Put entry
        data = b"test data"
        metadata = {"type": "test"}
        cache.put("key1", data, metadata)
        
        # Get entry
        entry = cache.get("key1")
        assert entry is not None
        assert entry.data == data
        assert entry.metadata == metadata
        assert cache.current_size_bytes > 0
    
    def test_lru_cache_eviction(self):
        """Test LRU eviction when size limit exceeded"""
        cache = LRUCache(max_size_bytes=100)  # Small limit
        
        # Add entries that exceed limit
        cache.put("key1", b"x" * 40, {})
        cache.put("key2", b"x" * 40, {})
        cache.put("key3", b"x" * 40, {})  # Should evict key1
        
        # key1 should be evicted, key2 and key3 should remain
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
    
    def test_lru_cache_access_order(self):
        """Test LRU access order maintenance"""
        cache = LRUCache(max_size_bytes=120)  # Reduced size to force eviction
        
        cache.put("key1", b"x" * 30, {})  # ~32 bytes
        cache.put("key2", b"x" * 30, {})  # ~32 bytes  
        cache.put("key3", b"x" * 30, {})  # ~32 bytes (total ~96 bytes)
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add new entry that should evict key2 (least recently used)
        # key4 needs ~42 bytes, total would be ~138, so key2 should be evicted
        cache.put("key4", b"x" * 40, {})
        
        assert cache.get("key1") is not None  # Still there (was accessed)
        assert cache.get("key2") is None      # Evicted (least recently used)
        assert cache.get("key3") is not None  # Still there  
        assert cache.get("key4") is not None  # New entry
    
    def test_lru_cache_clear(self):
        """Test clearing cache"""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("key1", b"data1", {})
        cache.put("key2", b"data2", {})
        
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.current_size_bytes == 0
    
    def test_lru_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size_bytes=1024)
        
        # Should start with zero stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        
        # Add data and test
        cache.put("key1", b"data", {})
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestDiskCache:
    """Test local disk cache at /tmp/models"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_disk_cache_creation(self, temp_cache_dir):
        """Test creating disk cache"""
        cache = DiskCache(cache_dir=temp_cache_dir, max_size_gb=1.0)
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.max_size_bytes == 1.0 * 1024 * 1024 * 1024
        assert cache.cache_dir.exists()
    
    @pytest.mark.asyncio
    async def test_disk_cache_put_get(self, temp_cache_dir):
        """Test basic put/get operations"""
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        # Put entry
        data = b"test data for disk cache"
        metadata = {"model_type": "lstm", "version": "v1.0.0"}
        await cache.put("lstm-v1.0.0", data, metadata)
        
        # Get entry
        entry = await cache.get("lstm-v1.0.0")
        assert entry is not None
        assert entry.data == data
        assert entry.metadata == metadata
        
        # Check file exists on disk
        cache_file = cache.cache_dir / "lstm-v1.0.0.cache"
        assert cache_file.exists()
    
    @pytest.mark.asyncio
    async def test_disk_cache_compression(self, temp_cache_dir):
        """Test disk cache with compression"""
        cache = DiskCache(cache_dir=temp_cache_dir, compression_enabled=True)
        
        # Create larger data that benefits from compression
        data = b"x" * 1000
        await cache.put("large-model", data, {})
        
        # File should exist and be smaller than original data
        cache_file = cache.cache_dir / "large-model.cache"
        assert cache_file.exists()
        assert cache_file.stat().st_size < len(data)
        
        # Should decompress correctly
        entry = await cache.get("large-model")
        assert entry.data == data
    
    @pytest.mark.asyncio
    async def test_disk_cache_expiry(self, temp_cache_dir):
        """Test disk cache TTL expiry"""
        cache = DiskCache(cache_dir=temp_cache_dir, ttl_seconds=1)
        
        # Put entry
        await cache.put("temp-key", b"temp data", {})
        
        # Should be available immediately
        entry = await cache.get("temp-key")
        assert entry is not None
        
        # Wait for expiry
        await asyncio.sleep(1.1)
        
        # Should be expired and return None
        entry = await cache.get("temp-key")
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_disk_cache_size_limit(self, temp_cache_dir):
        """Test disk cache size limit enforcement"""
        cache = DiskCache(cache_dir=temp_cache_dir, max_size_gb=0.001)  # 1MB limit
        
        # Add data that exceeds limit
        large_data = b"x" * (512 * 1024)  # 512KB
        await cache.put("model1", large_data, {})
        await cache.put("model2", large_data, {})  # Should trigger cleanup
        
        # One of the models should be evicted
        model1_exists = await cache.exists("model1")
        model2_exists = await cache.exists("model2")
        
        # At least one should be evicted to stay under limit
        assert not (model1_exists and model2_exists)
    
    @pytest.mark.asyncio
    async def test_disk_cache_cleanup(self, temp_cache_dir):
        """Test disk cache cleanup of expired entries"""
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        # Put some entries
        await cache.put("keep", b"keep this", {})
        await cache.put("expire", b"expire this", {})
        
        # Manually mark one as expired by modifying file time
        import os
        expire_file = cache.cache_dir / "expire.cache"
        old_time = time.time() - (24 * 3600)  # 24 hours ago
        expire_file.touch()  # Create the file first
        os.utime(expire_file, (old_time, old_time))
        
        # Run cleanup
        await cache.cleanup_expired(ttl_seconds=3600)  # 1 hour TTL
        
        # Keep should exist, expire should be gone
        assert await cache.exists("keep")
        assert not await cache.exists("expire")


class TestGCSCache:
    """Test GCS cache with retry logic"""
    
    @pytest.fixture
    def mock_gcs_client(self):
        """Mock GCS client"""
        client = Mock()
        bucket = Mock()
        client.bucket.return_value = bucket
        return client, bucket
    
    def test_gcs_cache_creation(self, mock_gcs_client):
        """Test creating GCS cache"""
        client, bucket = mock_gcs_client
        cache = GCSCache(
            gcs_client=client,
            bucket_name="test-bucket",
            cache_prefix="cache/"
        )
        assert cache.bucket_name == "test-bucket"
        assert cache.cache_prefix == "cache/"
        assert cache.retry_attempts == 3
    
    @pytest.mark.asyncio
    async def test_gcs_cache_put_get(self, mock_gcs_client):
        """Test basic GCS put/get operations"""
        client, bucket = mock_gcs_client
        cache = GCSCache(gcs_client=client, bucket_name="test-bucket")
        
        # Mock blob operations
        blob = Mock()
        bucket.blob.return_value = blob
        blob.upload_from_string = Mock()
        blob.download_as_bytes = Mock(return_value=b"test data")
        blob.exists = Mock(return_value=True)
        blob.metadata = {"model_type": "lstm"}
        blob.size = 9
        blob.time_created = datetime.now()
        
        # Put operation
        await cache.put("test-key", b"test data", {"model_type": "lstm"})
        
        # Verify upload was called
        blob.upload_from_string.assert_called_once()
        
        # Get operation
        entry = await cache.get("test-key")
        assert entry is not None
        assert entry.data == b"test data"
        assert entry.metadata["model_type"] == "lstm"
    
    @pytest.mark.asyncio
    async def test_gcs_cache_retry_logic(self, mock_gcs_client):
        """Test GCS retry logic on failures"""
        client, bucket = mock_gcs_client
        cache = GCSCache(gcs_client=client, bucket_name="test-bucket", retry_attempts=3)
        
        # Mock blob that fails twice then succeeds
        blob = Mock()
        bucket.blob.return_value = blob
        blob.download_as_bytes = Mock(side_effect=[
            Exception("Network error"),
            Exception("Timeout"),
            b"success data"
        ])
        blob.exists = Mock(return_value=True)
        blob.metadata = {}
        blob.size = 12
        blob.time_created = datetime.now()
        
        # Should succeed after retries
        entry = await cache.get("test-key")
        assert entry is not None
        assert entry.data == b"success data"
        assert blob.download_as_bytes.call_count == 3
    
    @pytest.mark.asyncio
    async def test_gcs_cache_retry_exhaustion(self, mock_gcs_client):
        """Test GCS retry exhaustion"""
        client, bucket = mock_gcs_client
        cache = GCSCache(gcs_client=client, bucket_name="test-bucket", retry_attempts=2)
        
        # Mock blob that always fails
        blob = Mock()
        bucket.blob.return_value = blob
        blob.download_as_bytes = Mock(side_effect=Exception("Persistent error"))
        blob.exists = Mock(return_value=True)
        
        # Should return None after retries exhausted
        entry = await cache.get("test-key")
        assert entry is None
        assert blob.download_as_bytes.call_count == 2


class TestCacheManager:
    """Test multi-level cache manager"""
    
    @pytest.fixture
    def mock_caches(self):
        """Create mock cache layers"""
        memory_cache = Mock(spec=LRUCache)
        disk_cache = Mock(spec=DiskCache)
        gcs_cache = Mock(spec=GCSCache)
        return memory_cache, disk_cache, gcs_cache
    
    def test_cache_manager_creation(self, mock_caches):
        """Test creating cache manager"""
        memory_cache, disk_cache, gcs_cache = mock_caches
        config = CacheConfig()
        
        manager = CacheManager(
            config=config,
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            gcs_cache=gcs_cache
        )
        
        assert manager.config == config
        assert manager.memory_cache == memory_cache
        assert manager.disk_cache == disk_cache
        assert manager.gcs_cache == gcs_cache
    
    @pytest.mark.asyncio
    async def test_cache_manager_get_hierarchy(self, mock_caches):
        """Test cache manager respects hierarchy (memory → disk → GCS)"""
        memory_cache, disk_cache, gcs_cache = mock_caches
        config = CacheConfig()
        manager = CacheManager(config, memory_cache, disk_cache, gcs_cache)
        
        # Mock memory cache miss, disk cache hit
        memory_cache.get.return_value = None
        disk_entry = CacheEntry("test", b"disk data", {}, 9, datetime.now(), datetime.now())
        disk_cache.get = AsyncMock(return_value=disk_entry)
        
        # Should get from disk and promote to memory
        entry = await manager.get("test-key")
        
        assert entry == disk_entry
        memory_cache.get.assert_called_once_with("test-key")
        disk_cache.get.assert_called_once_with("test-key")
        gcs_cache.get.assert_not_called()  # Should not reach GCS
        memory_cache.put.assert_called_once()  # Should promote to memory
    
    @pytest.mark.asyncio
    async def test_cache_manager_put_all_levels(self, mock_caches):
        """Test cache manager puts to all levels"""
        memory_cache, disk_cache, gcs_cache = mock_caches
        config = CacheConfig()
        manager = CacheManager(config, memory_cache, disk_cache, gcs_cache)
        
        # Mock async put methods
        disk_cache.put = AsyncMock()
        gcs_cache.put = AsyncMock()
        
        await manager.put("test-key", b"test data", {"type": "test"})
        
        # Should put to all levels
        memory_cache.put.assert_called_once()
        disk_cache.put.assert_called_once()
        gcs_cache.put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_manager_metrics(self, mock_caches):
        """Test cache manager metrics collection"""
        memory_cache, disk_cache, gcs_cache = mock_caches
        config = CacheConfig()
        manager = CacheManager(config, memory_cache, disk_cache, gcs_cache)
        
        # Mock cache stats
        memory_cache.get_stats.return_value = {"hits": 10, "misses": 5}
        disk_cache.get_stats = AsyncMock(return_value={"hits": 3, "misses": 2})
        gcs_cache.get_stats = AsyncMock(return_value={"hits": 1, "misses": 4})
        
        metrics = await manager.get_metrics()
        
        assert isinstance(metrics, CacheMetrics)
        assert metrics.memory_hits == 10
        assert metrics.disk_hits == 3
        assert metrics.gcs_hits == 1
        assert metrics.total_requests > 0


class TestCacheWarmer:
    """Test cache warming on startup"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager"""
        return Mock(spec=CacheManager)
    
    @pytest.fixture
    def mock_preservation_manager(self):
        """Mock preservation manager"""
        manager = Mock()
        manager.list_models = AsyncMock(return_value=[
            {"model_type": "lstm", "version": "v1.0.0", "mode": "analysis"},
            {"model_type": "dqn", "version": "v2.1.0", "mode": "simulation"}
        ])
        manager.load_model = AsyncMock(return_value=(b"model data", {"type": "test"}))
        return manager
    
    def test_cache_warmer_creation(self, mock_cache_manager, mock_preservation_manager):
        """Test creating cache warmer"""
        warmer = CacheWarmer(
            cache_manager=mock_cache_manager,
            preservation_manager=mock_preservation_manager
        )
        assert warmer.cache_manager == mock_cache_manager
        assert warmer.preservation_manager == mock_preservation_manager
    
    @pytest.mark.asyncio
    async def test_cache_warmer_warm_latest_models(self, mock_cache_manager, mock_preservation_manager):
        """Test warming cache with latest models"""
        warmer = CacheWarmer(mock_cache_manager, mock_preservation_manager)
        mock_cache_manager.put = AsyncMock()
        
        await warmer.warm_latest_models()
        
        # Should load and cache latest models
        assert mock_preservation_manager.list_models.call_count >= 1
        assert mock_preservation_manager.load_model.call_count >= 1
        assert mock_cache_manager.put.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_cache_warmer_warm_by_pattern(self, mock_cache_manager, mock_preservation_manager):
        """Test warming cache by usage patterns"""
        warmer = CacheWarmer(mock_cache_manager, mock_preservation_manager)
        mock_cache_manager.put = AsyncMock()
        
        # Mock usage patterns
        patterns = [
            {"model_type": "lstm", "mode": "analysis", "frequency": 0.8},
            {"model_type": "dqn", "mode": "simulation", "frequency": 0.6}
        ]
        
        await warmer.warm_by_patterns(patterns)
        
        # Should warm models based on patterns
        assert mock_cache_manager.put.call_count >= len(patterns)


class TestModelPrefetcher:
    """Test prefetching for likely models"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager"""
        return Mock(spec=CacheManager)
    
    def test_prefetcher_creation(self, mock_cache_manager):
        """Test creating model prefetcher"""
        prefetcher = ModelPrefetcher(cache_manager=mock_cache_manager)
        assert prefetcher.cache_manager == mock_cache_manager
        assert prefetcher.prediction_models == {}
    
    @pytest.mark.asyncio
    async def test_prefetcher_predict_next_models(self, mock_cache_manager):
        """Test predicting next models to prefetch"""
        prefetcher = ModelPrefetcher(mock_cache_manager)
        
        # Mock historical access patterns
        access_history = [
            {"model_type": "lstm", "version": "v1.0.0", "timestamp": datetime.now()},
            {"model_type": "dqn", "version": "v2.0.0", "timestamp": datetime.now()},
            {"model_type": "lstm", "version": "v1.1.0", "timestamp": datetime.now()}
        ]
        
        predictions = await prefetcher.predict_next_models(access_history)
        
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        # Should predict based on patterns
        assert all("model_type" in pred for pred in predictions)
        assert all("probability" in pred for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_prefetcher_prefetch_models(self, mock_cache_manager):
        """Test prefetching predicted models"""
        prefetcher = ModelPrefetcher(mock_cache_manager)
        mock_cache_manager.get = AsyncMock(return_value=None)  # Not in cache
        mock_cache_manager.put = AsyncMock()
        
        # Mock preservation manager for loading
        preservation_manager = Mock()
        preservation_manager.load_model = AsyncMock(return_value=(b"data", {}))
        prefetcher.preservation_manager = preservation_manager
        
        predictions = [
            {"model_type": "lstm", "version": "v1.0.0", "probability": 0.8},
            {"model_type": "dqn", "version": "v2.0.0", "probability": 0.6}
        ]
        
        await prefetcher.prefetch_models(predictions, threshold=0.5)
        
        # Should prefetch models above threshold
        assert preservation_manager.load_model.call_count == 2
        assert mock_cache_manager.put.call_count == 2


class TestCacheMonitor:
    """Test cache metrics and monitoring"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager with metrics"""
        manager = Mock(spec=CacheManager)
        manager.get_metrics = AsyncMock(return_value=CacheMetrics(
            total_requests=100,
            memory_hits=40,
            disk_hits=30,
            gcs_hits=20,
            misses=10
        ))
        return manager
    
    def test_cache_monitor_creation(self, mock_cache_manager):
        """Test creating cache monitor"""
        monitor = CacheMonitor(cache_manager=mock_cache_manager)
        assert monitor.cache_manager == mock_cache_manager
        assert monitor.alerts == []
    
    @pytest.mark.asyncio
    async def test_cache_monitor_collect_metrics(self, mock_cache_manager):
        """Test collecting cache metrics"""
        monitor = CacheMonitor(mock_cache_manager)
        
        metrics = await monitor.collect_metrics()
        
        assert isinstance(metrics, CacheMetrics)
        assert metrics.total_requests == 100
        assert metrics.hit_rate == 0.90
        mock_cache_manager.get_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_monitor_performance_alerts(self, mock_cache_manager):
        """Test performance alerts"""
        # Mock low hit rate
        low_hit_metrics = CacheMetrics(
            total_requests=100,
            memory_hits=10,
            disk_hits=10,
            gcs_hits=10,
            misses=70  # 70% miss rate
        )
        mock_cache_manager.get_metrics = AsyncMock(return_value=low_hit_metrics)
        
        monitor = CacheMonitor(mock_cache_manager)
        await monitor.check_performance()
        
        # Should generate alert for low hit rate
        assert len(monitor.alerts) > 0
        assert any("hit rate" in alert["message"].lower() for alert in monitor.alerts)
    
    @pytest.mark.asyncio
    async def test_cache_monitor_memory_alerts(self, mock_cache_manager):
        """Test memory usage alerts"""
        # Mock high memory usage
        high_memory_metrics = CacheMetrics(
            memory_usage_bytes=1.8 * 1024 * 1024 * 1024  # 1.8GB out of 2GB
        )
        mock_cache_manager.get_metrics = AsyncMock(return_value=high_memory_metrics)
        
        monitor = CacheMonitor(mock_cache_manager)
        await monitor.check_memory_usage(max_memory_gb=2.0)
        
        # Should generate alert for high memory usage
        assert len(monitor.alerts) > 0
        assert any("memory" in alert["message"].lower() for alert in monitor.alerts)


class TestCompressionOptimizer:
    """Test model compression optimization"""
    
    def test_compression_optimizer_creation(self):
        """Test creating compression optimizer"""
        optimizer = CompressionOptimizer()
        assert optimizer.compression_level == 6  # Default zlib level
        assert optimizer.algorithms == ["zlib", "lz4", "brotli"]
    
    def test_compression_analysis(self):
        """Test analyzing data for optimal compression"""
        optimizer = CompressionOptimizer()
        
        # Test different data types
        text_data = b"This is a text string that should compress well" * 10
        binary_data = bytes(range(256)) * 4
        random_data = b"random binary data that doesn't compress well"
        
        text_analysis = optimizer.analyze_data(text_data)
        binary_analysis = optimizer.analyze_data(binary_data)
        random_analysis = optimizer.analyze_data(random_data)
        
        assert "compression_ratio" in text_analysis
        assert "recommended_algorithm" in text_analysis
        assert "estimated_savings" in text_analysis
        
        # Text should compress better than random data
        assert text_analysis["compression_ratio"] > random_analysis["compression_ratio"]
    
    def test_adaptive_compression(self):
        """Test adaptive compression selection"""
        optimizer = CompressionOptimizer()
        
        # Highly compressible data
        compressible_data = b"AAAA" * 1000
        compressed = optimizer.compress_adaptive(compressible_data)
        
        assert len(compressed) < len(compressible_data)
        assert "algorithm" in compressed
        assert "compressed_data" in compressed
        assert "original_size" in compressed
        assert "compressed_size" in compressed
        
        # Should be able to decompress
        decompressed = optimizer.decompress(compressed)
        assert decompressed == compressible_data
    
    def test_compression_benchmarking(self):
        """Test compression algorithm benchmarking"""
        optimizer = CompressionOptimizer()
        
        test_data = b"Test data for compression benchmarking" * 100
        results = optimizer.benchmark_algorithms(test_data)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for algorithm, metrics in results.items():
            assert "compression_ratio" in metrics
            assert "compression_time" in metrics
            assert "decompression_time" in metrics
            assert "compressed_size" in metrics


# Integration Tests
class TestCacheSystemIntegration:
    """Test complete cache system integration"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_cache_hierarchy(self, temp_cache_dir):
        """Test complete cache hierarchy integration"""
        # This test will initially fail since we haven't implemented the classes yet
        config = CacheConfig(
            memory_limit_gb=1.0,
            disk_cache_dir=temp_cache_dir,
            enable_memory_cache=True,
            enable_disk_cache=True,
            enable_gcs_cache=False  # Skip GCS for integration test
        )
        
        # Create actual cache instances (will fail initially)
        memory_cache = LRUCache(max_size_bytes=config.memory_limit_gb * 1024 * 1024 * 1024)
        disk_cache = DiskCache(cache_dir=config.disk_cache_dir)
        
        manager = CacheManager(
            config=config,
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            gcs_cache=None
        )
        
        # Test put/get through hierarchy
        test_data = b"integration test data"
        test_metadata = {"model_type": "integration", "version": "v1.0.0"}
        
        await manager.put("integration-test", test_data, test_metadata)
        
        # Should be in memory cache
        entry = await manager.get("integration-test")
        assert entry is not None
        assert entry.data == test_data
        
        # Clear memory cache, should fallback to disk
        manager.memory_cache.clear()
        entry = await manager.get("integration-test")
        assert entry is not None
        assert entry.data == test_data
    
    @pytest.mark.asyncio
    async def test_cache_warming_integration(self, temp_cache_dir):
        """Test cache warming integration"""
        # Mock preservation manager
        preservation_manager = Mock()
        preservation_manager.list_models = AsyncMock(return_value=[
            {"model_type": "lstm", "version": "v1.0.0", "mode": "analysis"}
        ])
        preservation_manager.load_model = AsyncMock(return_value=(b"warm data", {}))
        
        # Create cache system
        config = CacheConfig(disk_cache_dir=temp_cache_dir)
        memory_cache = LRUCache(max_size_bytes=1024 * 1024)
        disk_cache = DiskCache(cache_dir=temp_cache_dir)
        
        manager = CacheManager(config, memory_cache, disk_cache, None)
        warmer = CacheWarmer(manager, preservation_manager)
        
        # Warm cache
        await warmer.warm_latest_models()
        
        # Should be able to get warmed models from cache
        entry = await manager.get("lstm-v1.0.0-analysis")
        assert entry is not None


# Performance Tests
class TestCachePerformance:
    """Test cache system performance"""
    
    @pytest.mark.asyncio
    async def test_memory_cache_performance(self):
        """Test memory cache performance with large datasets"""
        cache = LRUCache(max_size_bytes=100 * 1024 * 1024)  # 100MB
        
        # Test with various sizes
        start_time = time.time()
        
        for i in range(100):
            data = b"x" * 10000  # 10KB per entry
            cache.put(f"perf-test-{i}", data, {})
        
        put_time = time.time() - start_time
        
        # Get performance
        start_time = time.time()
        
        for i in range(100):
            entry = cache.get(f"perf-test-{i}")
            assert entry is not None
        
        get_time = time.time() - start_time
        
        # Performance should be reasonable
        assert put_time < 1.0  # Should put 100 entries in < 1 second
        assert get_time < 0.1  # Should get 100 entries in < 0.1 seconds
    
    @pytest.mark.asyncio
    async def test_compression_performance(self):
        """Test compression performance impact"""
        optimizer = CompressionOptimizer()
        
        # Test data of various sizes
        sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        
        for size in sizes:
            data = b"x" * size
            
            start_time = time.time()
            compressed = optimizer.compress_adaptive(data)
            compression_time = time.time() - start_time
            
            start_time = time.time()
            decompressed = optimizer.decompress(compressed)
            decompression_time = time.time() - start_time
            
            assert decompressed == data
            
            # Performance should scale reasonably
            assert compression_time < size / 10000  # Rough benchmark
            assert decompression_time < compression_time