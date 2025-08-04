"""
Test cases for KV-cache optimization functionality - Phase 2.1

This module contains comprehensive test cases for key-value caching in transformers,
following TDD methodology with failing tests first.

Tests cover:
- KV-cache initialization and management
- Autoregressive generation with caching
- Cache eviction and memory management
- Multi-head attention with KV-cache
- Performance benchmarking
"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch
from dataclasses import dataclass

# Import the modules we'll be testing (these will fail initially)
try:
    from src.ml_analysis.transformers.optimization import (
        KVCacheConfig,
        KVCacheManager,
        CachedMultiHeadAttention,
        AutoregressiveGenerator
    )
    from src.ml_analysis.inference_optimizer import (
        TransformerInferenceOptimizer,
        KVCacheOptimizer
    )
except ImportError:
    # Expected to fail initially - we'll implement these classes
    pass


@dataclass
class KVCacheTestConfig:
    """Test configuration for KV-cache tests"""
    batch_size: int = 2
    seq_length: int = 128
    d_model: int = 512
    n_heads: int = 8
    max_cache_size: int = 1000
    eviction_strategy: str = "lru"


class TestKVCacheConfig:
    """Test KV-cache configuration"""
    
    def test_kv_cache_config_initialization(self):
        """Test KVCacheConfig initialization with default values"""
        # This test will fail until we implement KVCacheConfig
        config = KVCacheConfig()
        
        assert config.max_cache_size > 0
        assert config.eviction_strategy in ["lru", "fifo", "lfu"]
        assert config.enable_compression is not None
        assert config.memory_limit_mb > 0
        
    def test_kv_cache_config_custom_values(self):
        """Test KVCacheConfig with custom values"""
        config = KVCacheConfig(
            max_cache_size=500,
            eviction_strategy="fifo",
            enable_compression=True,
            memory_limit_mb=256
        )
        
        assert config.max_cache_size == 500
        assert config.eviction_strategy == "fifo"
        assert config.enable_compression is True
        assert config.memory_limit_mb == 256
        
    def test_kv_cache_config_validation(self):
        """Test KVCacheConfig parameter validation"""
        # Test invalid eviction strategy
        with pytest.raises(ValueError):
            KVCacheConfig(eviction_strategy="invalid")
            
        # Test invalid cache size
        with pytest.raises(ValueError):
            KVCacheConfig(max_cache_size=0)


class TestKVCacheManager:
    """Test KV-cache manager functionality"""
    
    @pytest.fixture
    def cache_config(self):
        """Fixture providing test cache configuration"""
        return KVCacheConfig(
            max_cache_size=100,
            eviction_strategy="lru",
            enable_compression=False,
            memory_limit_mb=128
        )
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        """Fixture providing KV-cache manager"""
        return KVCacheManager(cache_config)
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test KVCacheManager initialization"""
        assert cache_manager.config is not None
        assert cache_manager.cache_size == 0
        assert cache_manager.hit_count == 0
        assert cache_manager.miss_count == 0
        
    def test_cache_put_and_get(self, cache_manager):
        """Test basic cache put and get operations"""
        # Create test key-value tensors
        key = "test_seq_0"
        k_tensor = torch.randn(2, 8, 64, 32)  # batch, n_heads, seq_len, head_dim
        v_tensor = torch.randn(2, 8, 64, 32)
        
        # Put tensors in cache
        success = cache_manager.put(key, k_tensor, v_tensor)
        assert success is True
        assert cache_manager.cache_size == 1
        
        # Get tensors from cache
        cached_k, cached_v = cache_manager.get(key)
        assert cached_k is not None
        assert cached_v is not None
        assert torch.allclose(cached_k, k_tensor)
        assert torch.allclose(cached_v, v_tensor)
        assert cache_manager.hit_count == 1
        
    def test_cache_miss(self, cache_manager):
        """Test cache miss behavior"""
        cached_k, cached_v = cache_manager.get("nonexistent_key")
        assert cached_k is None
        assert cached_v is None
        assert cache_manager.miss_count == 1
        
    def test_cache_eviction_lru(self, cache_config):
        """Test LRU cache eviction strategy"""
        cache_config.max_cache_size = 2
        cache_manager = KVCacheManager(cache_config)
        
        # Fill cache to capacity
        for i in range(3):
            k_tensor = torch.randn(1, 4, 32, 16)
            v_tensor = torch.randn(1, 4, 32, 16)
            cache_manager.put(f"key_{i}", k_tensor, v_tensor)
        
        # First key should be evicted
        cached_k, cached_v = cache_manager.get("key_0")
        assert cached_k is None
        assert cached_v is None
        
        # Last two keys should still be in cache
        cached_k, cached_v = cache_manager.get("key_2")
        assert cached_k is not None
        assert cached_v is not None
        
    def test_cache_memory_limit(self, cache_config):
        """Test cache memory limit enforcement"""
        cache_config.memory_limit_mb = 1  # Very small limit
        cache_manager = KVCacheManager(cache_config)
        
        # Try to cache large tensors
        large_k = torch.randn(10, 32, 1024, 64)  # Large tensor
        large_v = torch.randn(10, 32, 1024, 64)
        
        success = cache_manager.put("large_key", large_k, large_v)
        # Should fail due to memory limit
        assert success is False
        
    def test_cache_compression(self, cache_config):
        """Test cache compression functionality"""
        cache_config.enable_compression = True
        cache_manager = KVCacheManager(cache_config)
        
        k_tensor = torch.randn(2, 8, 128, 32)
        v_tensor = torch.randn(2, 8, 128, 32)
        
        success = cache_manager.put("compressed_key", k_tensor, v_tensor)
        assert success is True
        
        cached_k, cached_v = cache_manager.get("compressed_key")
        assert cached_k is not None
        assert cached_v is not None
        # Should be approximately equal due to compression
        assert torch.allclose(cached_k, k_tensor, atol=1e-3)
        assert torch.allclose(cached_v, v_tensor, atol=1e-3)
        
    def test_cache_clear(self, cache_manager):
        """Test cache clearing functionality"""
        # Add some entries
        k_tensor = torch.randn(1, 4, 32, 16)
        v_tensor = torch.randn(1, 4, 32, 16)
        cache_manager.put("key_1", k_tensor, v_tensor)
        cache_manager.put("key_2", k_tensor, v_tensor)
        
        assert cache_manager.cache_size == 2
        
        # Clear cache
        cache_manager.clear()
        assert cache_manager.cache_size == 0
        
        # Verify entries are gone
        cached_k, cached_v = cache_manager.get("key_1")
        assert cached_k is None
        assert cached_v is None
        
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection"""
        k_tensor = torch.randn(1, 4, 32, 16)
        v_tensor = torch.randn(1, 4, 32, 16)
        
        # Perform cache operations
        cache_manager.put("key_1", k_tensor, v_tensor)
        cache_manager.get("key_1")  # Hit
        cache_manager.get("key_2")  # Miss
        
        stats = cache_manager.get_statistics()
        assert stats["cache_size"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5
        assert "memory_usage_mb" in stats


class TestCachedMultiHeadAttention:
    """Test cached multi-head attention implementation"""
    
    @pytest.fixture
    def attention_config(self):
        """Fixture providing attention configuration"""
        return {
            "d_model": 512,
            "n_heads": 8,
            "dropout": 0.1,
            "cache_config": KVCacheConfig(max_cache_size=50)
        }
    
    @pytest.fixture
    def cached_attention(self, attention_config):
        """Fixture providing cached multi-head attention"""
        return CachedMultiHeadAttention(**attention_config)
    
    def test_cached_attention_initialization(self, cached_attention):
        """Test cached attention initialization"""
        assert cached_attention.d_model > 0
        assert cached_attention.n_heads > 0
        assert cached_attention.head_dim == cached_attention.d_model // cached_attention.n_heads
        assert cached_attention.cache_manager is not None
        
    def test_cached_attention_forward_no_cache(self, cached_attention):
        """Test forward pass without using cache"""
        batch_size, seq_len, d_model = 2, 32, 512
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = cached_attention(
            query, key, value, use_cache=False
        )
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, cached_attention.n_heads, seq_len, seq_len)
        
    def test_cached_attention_forward_with_cache(self, cached_attention):
        """Test forward pass with KV-cache"""
        batch_size, seq_len, d_model = 2, 32, 512
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        cache_key = "seq_0"
        
        # First forward pass - should populate cache
        output1, _ = cached_attention(
            query, key, value, use_cache=True, cache_key=cache_key
        )
        
        # Second forward pass - should use cache
        output2, _ = cached_attention(
            query, key, value, use_cache=True, cache_key=cache_key
        )
        
        assert torch.allclose(output1, output2)
        assert cached_attention.cache_manager.hit_count == 1
        
    def test_cached_attention_incremental_generation(self, cached_attention):
        """Test incremental generation with KV-cache"""
        batch_size, d_model = 1, 512
        
        # Initial sequence
        initial_seq_len = 10
        query = torch.randn(batch_size, initial_seq_len, d_model)
        key = torch.randn(batch_size, initial_seq_len, d_model)
        value = torch.randn(batch_size, initial_seq_len, d_model)
        
        # First step
        output1, _ = cached_attention(
            query, key, value, use_cache=True, cache_key="incremental"
        )
        
        # Add one more token
        new_query = torch.randn(batch_size, 1, d_model)
        new_key = torch.randn(batch_size, 1, d_model)
        new_value = torch.randn(batch_size, 1, d_model)
        
        output2, _ = cached_attention(
            new_query, new_key, new_value, 
            use_cache=True, cache_key="incremental", incremental=True
        )
        
        assert output2.shape == (batch_size, 1, d_model)
        
    def test_cached_attention_performance(self, cached_attention):
        """Test performance improvement with caching"""
        batch_size, seq_len, d_model = 1, 128, 512
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Time without cache
        start_time = time.time()
        for _ in range(10):
            cached_attention(query, key, value, use_cache=False)
        time_without_cache = time.time() - start_time
        
        # Time with cache (after first call)
        cached_attention(query, key, value, use_cache=True, cache_key="perf_test")
        
        start_time = time.time()
        for _ in range(10):
            cached_attention(query, key, value, use_cache=True, cache_key="perf_test")
        time_with_cache = time.time() - start_time
        
        # Cache should provide speedup
        assert time_with_cache < time_without_cache


class TestAutoregressiveGenerator:
    """Test autoregressive generation with KV-cache"""
    
    @pytest.fixture
    def generator_config(self):
        """Fixture providing generator configuration"""
        return {
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 2,
            "vocab_size": 1000,
            "max_seq_length": 128,
            "cache_config": KVCacheConfig(max_cache_size=20)
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        """Fixture providing autoregressive generator"""
        return AutoregressiveGenerator(**generator_config)
    
    def test_generator_initialization(self, generator):
        """Test autoregressive generator initialization"""
        assert generator.d_model > 0
        assert generator.n_heads > 0
        assert generator.n_layers > 0
        assert generator.vocab_size > 0
        assert generator.max_seq_length > 0
        assert hasattr(generator, 'transformer_layers')
        assert hasattr(generator, 'output_projection')
        
    def test_generator_forward_single_step(self, generator):
        """Test single forward step"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, generator.vocab_size, (batch_size, seq_len))
        
        logits = generator.forward(input_ids, use_cache=False)
        
        assert logits.shape == (batch_size, seq_len, generator.vocab_size)
        
    def test_generator_generate_sequence(self, generator):
        """Test sequence generation with KV-cache"""
        batch_size = 1
        prompt_length = 5
        generate_length = 10
        
        prompt = torch.randint(0, generator.vocab_size, (batch_size, prompt_length))
        
        generated_sequence = generator.generate(
            prompt, 
            max_length=generate_length,
            use_cache=True,
            temperature=1.0,
            top_k=50
        )
        
        assert generated_sequence.shape == (batch_size, prompt_length + generate_length)
        
    def test_generator_cache_efficiency(self, generator):
        """Test cache efficiency during generation"""
        batch_size = 1
        prompt_length = 5
        generate_length = 10
        
        prompt = torch.randint(0, generator.vocab_size, (batch_size, prompt_length))
        
        # Generate with cache
        start_time = time.time()
        generated_with_cache = generator.generate(
            prompt, max_length=generate_length, use_cache=True
        )
        time_with_cache = time.time() - start_time
        
        # Generate without cache
        start_time = time.time()
        generated_without_cache = generator.generate(
            prompt, max_length=generate_length, use_cache=False
        )
        time_without_cache = time.time() - start_time
        
        # Cache should provide speedup for longer sequences
        if generate_length > 5:
            assert time_with_cache < time_without_cache
            
    def test_generator_beam_search(self, generator):
        """Test beam search with KV-cache"""
        batch_size = 1
        prompt_length = 5
        generate_length = 8
        beam_size = 3
        
        prompt = torch.randint(0, generator.vocab_size, (batch_size, prompt_length))
        
        generated_sequences = generator.beam_search(
            prompt,
            max_length=generate_length,
            beam_size=beam_size,
            use_cache=True
        )
        
        assert len(generated_sequences) == beam_size
        for seq in generated_sequences:
            assert seq.shape == (batch_size, prompt_length + generate_length)
            
    def test_generator_nucleus_sampling(self, generator):
        """Test nucleus (top-p) sampling"""
        batch_size = 1
        prompt_length = 5
        generate_length = 8
        
        prompt = torch.randint(0, generator.vocab_size, (batch_size, prompt_length))
        
        generated_sequence = generator.generate(
            prompt,
            max_length=generate_length,
            use_cache=True,
            sampling_strategy="nucleus",
            top_p=0.9
        )
        
        assert generated_sequence.shape == (batch_size, prompt_length + generate_length)


class TestTransformerInferenceOptimizer:
    """Test transformer-specific inference optimizer"""
    
    @pytest.fixture
    def optimizer_config(self):
        """Fixture providing optimizer configuration"""
        return {
            "enable_kv_cache": True,
            "enable_dynamic_batching": True,
            "max_batch_size": 8,
            "cache_config": KVCacheConfig(max_cache_size=100)
        }
    
    @pytest.fixture
    def transformer_optimizer(self, optimizer_config):
        """Fixture providing transformer inference optimizer"""
        return TransformerInferenceOptimizer(**optimizer_config)
    
    def test_optimizer_initialization(self, transformer_optimizer):
        """Test transformer optimizer initialization"""
        assert transformer_optimizer.enable_kv_cache is True
        assert transformer_optimizer.enable_dynamic_batching is True
        assert transformer_optimizer.max_batch_size > 0
        assert transformer_optimizer.kv_cache_manager is not None
        
    def test_optimizer_single_inference(self, transformer_optimizer):
        """Test single inference optimization"""
        d_model = 256
        seq_len = 32
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        model = Mock()
        model.return_value = torch.randn(1, seq_len, vocab_size)
        
        result = transformer_optimizer.optimize_inference(
            model, input_ids, use_cache=True
        )
        
        assert "logits" in result
        assert "cache_stats" in result
        assert "latency_ms" in result
        
    def test_optimizer_batch_inference(self, transformer_optimizer):
        """Test batch inference optimization"""
        d_model = 256
        seq_len = 32
        vocab_size = 1000
        batch_size = 4
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        model = Mock()
        model.return_value = torch.randn(batch_size, seq_len, vocab_size)
        
        results = transformer_optimizer.optimize_batch_inference(
            model, input_ids, use_cache=True
        )
        
        assert len(results) == batch_size
        for result in results:
            assert "logits" in result
            assert "cache_stats" in result
            
    def test_optimizer_dynamic_batching(self, transformer_optimizer):
        """Test dynamic batching optimization"""
        vocab_size = 1000
        
        # Different sequence lengths
        sequences = [
            torch.randint(0, vocab_size, (1, 16)),
            torch.randint(0, vocab_size, (1, 32)),
            torch.randint(0, vocab_size, (1, 24)),
            torch.randint(0, vocab_size, (1, 40))
        ]
        
        model = Mock()
        model.side_effect = [
            torch.randn(2, 32, vocab_size),  # Padded batch 1
            torch.randn(2, 40, vocab_size)   # Padded batch 2
        ]
        
        results = transformer_optimizer.dynamic_batch_inference(
            model, sequences, use_cache=True
        )
        
        assert len(results) == 4
        
    def test_optimizer_cache_management(self, transformer_optimizer):
        """Test cache management during inference"""
        d_model = 256
        seq_len = 32
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        model = Mock()
        model.return_value = torch.randn(1, seq_len, vocab_size)
        
        # First inference - should populate cache
        result1 = transformer_optimizer.optimize_inference(
            model, input_ids, use_cache=True, cache_key="test_sequence"
        )
        
        # Second inference - should use cache
        result2 = transformer_optimizer.optimize_inference(
            model, input_ids, use_cache=True, cache_key="test_sequence"
        )
        
        assert result1["cache_stats"]["hit"] is False
        assert result2["cache_stats"]["hit"] is True
        
    def test_optimizer_performance_metrics(self, transformer_optimizer):
        """Test performance metrics collection"""
        d_model = 256
        seq_len = 32
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        model = Mock()
        model.return_value = torch.randn(1, seq_len, vocab_size)
        
        # Run multiple inferences
        for i in range(10):
            transformer_optimizer.optimize_inference(
                model, input_ids, use_cache=True, cache_key=f"seq_{i}"
            )
        
        metrics = transformer_optimizer.get_performance_metrics()
        
        assert "average_latency_ms" in metrics
        assert "cache_hit_rate" in metrics
        assert "total_inferences" in metrics
        assert metrics["total_inferences"] == 10


class TestKVCacheOptimizer:
    """Test specialized KV-cache optimizer"""
    
    @pytest.fixture
    def kv_optimizer(self):
        """Fixture providing KV-cache optimizer"""
        cache_config = KVCacheConfig(max_cache_size=50)
        return KVCacheOptimizer(cache_config)
    
    def test_kv_optimizer_initialization(self, kv_optimizer):
        """Test KV-cache optimizer initialization"""
        assert kv_optimizer.cache_config is not None
        assert kv_optimizer.cache_manager is not None
        assert hasattr(kv_optimizer, 'optimization_stats')
        
    def test_kv_optimizer_attention_optimization(self, kv_optimizer):
        """Test attention computation optimization"""
        batch_size, n_heads, seq_len, head_dim = 2, 8, 64, 32
        
        query = torch.randn(batch_size, n_heads, seq_len, head_dim)
        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        output = kv_optimizer.optimize_attention(
            query, key, value, cache_key="test_attention", use_cache=True
        )
        
        assert output.shape == (batch_size, n_heads, seq_len, head_dim)
        
    def test_kv_optimizer_memory_efficiency(self, kv_optimizer):
        """Test memory efficiency of KV-cache"""
        # Create large sequences to test memory efficiency
        batch_size, n_heads, seq_len, head_dim = 1, 8, 256, 32
        
        query = torch.randn(batch_size, n_heads, seq_len, head_dim)
        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        # Measure memory before optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        output = kv_optimizer.optimize_attention(
            query, key, value, cache_key="memory_test", use_cache=True
        )
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_usage = final_memory - initial_memory
            
            # Cache should not significantly increase memory usage
            # (this is a simplified test - real implementation would be more complex)
            assert memory_usage < seq_len * head_dim * 4 * 2  # Rough estimate
            
    def test_kv_optimizer_incremental_computation(self, kv_optimizer):
        """Test incremental computation optimization"""
        batch_size, n_heads, seq_len, head_dim = 1, 4, 32, 16
        
        # Initial computation
        query = torch.randn(batch_size, n_heads, seq_len, head_dim)
        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        output1 = kv_optimizer.optimize_attention(
            query, key, value, cache_key="incremental", use_cache=True
        )
        
        # Incremental computation (add one more token)
        new_query = torch.randn(batch_size, n_heads, 1, head_dim)
        new_key = torch.randn(batch_size, n_heads, 1, head_dim)
        new_value = torch.randn(batch_size, n_heads, 1, head_dim)
        
        output2 = kv_optimizer.optimize_incremental_attention(
            new_query, new_key, new_value, 
            cache_key="incremental", position=seq_len
        )
        
        assert output2.shape == (batch_size, n_heads, 1, head_dim)
        
    def test_kv_optimizer_statistics(self, kv_optimizer):
        """Test optimization statistics collection"""
        batch_size, n_heads, seq_len, head_dim = 1, 4, 32, 16
        
        query = torch.randn(batch_size, n_heads, seq_len, head_dim)
        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        # Perform multiple optimizations
        for i in range(5):
            kv_optimizer.optimize_attention(
                query, key, value, cache_key=f"stats_test_{i}", use_cache=True
            )
        
        stats = kv_optimizer.get_optimization_statistics()
        
        assert "total_optimizations" in stats
        assert "cache_hit_rate" in stats
        assert "average_speedup" in stats
        assert stats["total_optimizations"] == 5


class TestIntegrationKVCache:
    """Integration tests for KV-cache functionality"""
    
    def test_end_to_end_generation_with_cache(self):
        """Test end-to-end generation with KV-cache"""
        # This test should fail initially and pass after implementation
        d_model = 256
        n_heads = 4
        vocab_size = 1000
        seq_len = 20
        
        # Create generator with cache
        generator_config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": 2,
            "vocab_size": vocab_size,
            "max_seq_length": 64,
            "cache_config": KVCacheConfig(max_cache_size=10)
        }
        
        generator = AutoregressiveGenerator(**generator_config)
        
        # Generate sequence
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = generator.generate(
            prompt, max_length=seq_len, use_cache=True, temperature=0.8
        )
        
        assert generated.shape == (1, seq_len + 5)
        
        # Verify cache was used
        cache_stats = generator.get_cache_statistics()
        assert cache_stats["total_cache_operations"] > 0
        
    def test_multi_sequence_cache_management(self):
        """Test cache management with multiple sequences"""
        cache_config = KVCacheConfig(max_cache_size=5, eviction_strategy="lru")
        cache_manager = KVCacheManager(cache_config)
        
        # Cache multiple sequences
        for i in range(8):  # More than cache capacity
            k_tensor = torch.randn(1, 4, 32, 16)
            v_tensor = torch.randn(1, 4, 32, 16)
            cache_manager.put(f"seq_{i}", k_tensor, v_tensor)
        
        # Verify LRU eviction
        assert cache_manager.cache_size <= 5
        
        # Early sequences should be evicted
        cached_k, cached_v = cache_manager.get("seq_0")
        assert cached_k is None
        
        # Recent sequences should be retained
        cached_k, cached_v = cache_manager.get("seq_7")
        assert cached_k is not None
        
    @pytest.mark.slow
    def test_performance_benchmark_kv_cache(self):
        """Benchmark KV-cache performance improvement"""
        d_model = 512
        n_heads = 8
        seq_len = 128
        batch_size = 2
        
        # Create cached attention
        cache_config = KVCacheConfig(max_cache_size=20)
        cached_attention = CachedMultiHeadAttention(
            d_model=d_model, n_heads=n_heads, cache_config=cache_config
        )
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Benchmark without cache
        start_time = time.time()
        for _ in range(50):
            cached_attention(query, key, value, use_cache=False)
        time_without_cache = time.time() - start_time
        
        # Populate cache
        cached_attention(query, key, value, use_cache=True, cache_key="benchmark")
        
        # Benchmark with cache
        start_time = time.time()
        for _ in range(50):
            cached_attention(query, key, value, use_cache=True, cache_key="benchmark")
        time_with_cache = time.time() - start_time
        
        # Calculate speedup
        speedup = time_without_cache / time_with_cache
        assert speedup > 1.5  # Expect at least 1.5x speedup
        
        print(f"KV-Cache speedup: {speedup:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])