"""
Test cases for Dynamic Batching functionality - Phase 2.1

This module contains comprehensive test cases for dynamic batching in transformers,
following TDD methodology with failing tests first.

Tests cover:
- Sequence length-based batching
- Multi-asset prediction batching  
- Memory-aware batch sizing
- Adaptive batch timeout
- Performance optimization
"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch
from dataclasses import dataclass
from concurrent.futures import Future
import asyncio

# Import the modules we'll be testing (these will fail initially)
try:
    from src.ml_analysis.transformers.optimization import (
        DynamicBatchingConfig,
        SequenceLengthBatcher,
        MultiAssetBatcher,
        AdaptiveBatchProcessor
    )
    from src.ml_analysis.inference_optimizer import (
        DynamicBatchingOptimizer,
        BatchSizeOptimizer
    )
    from src.ml_analysis.batch_prediction import (
        TransformerBatchProcessor,
        SequenceAwareBatcher,
        MemoryAwareBatchSizer
    )
except ImportError:
    # Expected to fail initially - we'll implement these classes
    pass


@dataclass
class AssetPredictionRequest:
    """Request for asset prediction"""
    asset_id: str
    sequence_data: torch.Tensor
    sequence_length: int
    priority: int = 1
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class BatchingTestConfig:
    """Test configuration for batching tests"""
    max_batch_size: int = 8
    min_batch_size: int = 1
    batch_timeout_ms: int = 100
    sequence_padding_strategy: str = "longest"
    memory_limit_mb: int = 512
    enable_adaptive_sizing: bool = True


class TestDynamicBatchingConfig:
    """Test dynamic batching configuration"""
    
    def test_batching_config_initialization(self):
        """Test DynamicBatchingConfig initialization with default values"""
        # This test will fail until we implement DynamicBatchingConfig
        config = DynamicBatchingConfig()
        
        assert config.max_batch_size > 0
        assert config.min_batch_size > 0
        assert config.batch_timeout_ms > 0
        assert config.sequence_padding_strategy in ["longest", "fixed", "bucketing"]
        assert config.memory_limit_mb > 0
        assert config.enable_adaptive_sizing is not None
        
    def test_batching_config_custom_values(self):
        """Test DynamicBatchingConfig with custom values"""
        config = DynamicBatchingConfig(
            max_batch_size=16,
            min_batch_size=2,
            batch_timeout_ms=50,
            sequence_padding_strategy="bucketing",
            memory_limit_mb=1024,
            enable_adaptive_sizing=True
        )
        
        assert config.max_batch_size == 16
        assert config.min_batch_size == 2
        assert config.batch_timeout_ms == 50
        assert config.sequence_padding_strategy == "bucketing"
        assert config.memory_limit_mb == 1024
        assert config.enable_adaptive_sizing is True
        
    def test_batching_config_validation(self):
        """Test DynamicBatchingConfig parameter validation"""
        # Test invalid padding strategy
        with pytest.raises(ValueError):
            DynamicBatchingConfig(sequence_padding_strategy="invalid")
            
        # Test invalid batch sizes
        with pytest.raises(ValueError):
            DynamicBatchingConfig(max_batch_size=0)
            
        with pytest.raises(ValueError):
            DynamicBatchingConfig(min_batch_size=5, max_batch_size=2)


class TestSequenceLengthBatcher:
    """Test sequence length-based batching"""
    
    @pytest.fixture
    def batcher_config(self):
        """Fixture providing batcher configuration"""
        return DynamicBatchingConfig(
            max_batch_size=8,
            sequence_padding_strategy="bucketing",
            batch_timeout_ms=100
        )
    
    @pytest.fixture
    def sequence_batcher(self, batcher_config):
        """Fixture providing sequence length batcher"""
        return SequenceLengthBatcher(batcher_config)
    
    def test_sequence_batcher_initialization(self, sequence_batcher):
        """Test SequenceLengthBatcher initialization"""
        assert sequence_batcher.config is not None
        assert hasattr(sequence_batcher, 'sequence_buckets')
        assert hasattr(sequence_batcher, 'batching_stats')
        
    def test_sequence_bucketing(self, sequence_batcher):
        """Test sequence length bucketing"""
        # Create sequences of different lengths
        sequences = [
            torch.randn(16, 256),   # Short sequence
            torch.randn(32, 256),   # Medium sequence
            torch.randn(64, 256),   # Long sequence
            torch.randn(18, 256),   # Short sequence (different length)
            torch.randn(33, 256),   # Medium sequence (different length)
        ]
        
        # Create asset requests
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        buckets = sequence_batcher.create_sequence_buckets(requests)
        
        # Should have multiple buckets based on sequence lengths
        assert len(buckets) > 1
        
        # Sequences in same bucket should have similar lengths
        for bucket_name, bucket_requests in buckets.items():
            lengths = [req.sequence_length for req in bucket_requests]
            assert max(lengths) - min(lengths) <= sequence_batcher.get_bucket_tolerance()
            
    def test_sequence_padding_longest(self, batcher_config):
        """Test padding strategy 'longest'"""
        batcher_config.sequence_padding_strategy = "longest"
        sequence_batcher = SequenceLengthBatcher(batcher_config)
        
        sequences = [
            torch.randn(10, 128),
            torch.randn(15, 128),
            torch.randn(12, 128)
        ]
        
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        padded_batch = sequence_batcher.create_padded_batch(requests)
        
        # All sequences should be padded to length of longest sequence (15)
        assert padded_batch.shape[1] == 15
        assert padded_batch.shape[0] == 3  # batch size
        
    def test_sequence_padding_fixed(self, batcher_config):
        """Test padding strategy 'fixed'"""
        batcher_config.sequence_padding_strategy = "fixed"
        batcher_config.fixed_sequence_length = 20
        sequence_batcher = SequenceLengthBatcher(batcher_config)
        
        sequences = [
            torch.randn(10, 128),
            torch.randn(15, 128),
            torch.randn(12, 128)
        ]
        
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        padded_batch = sequence_batcher.create_padded_batch(requests)
        
        # All sequences should be padded to fixed length (20)
        assert padded_batch.shape[1] == 20
        assert padded_batch.shape[0] == 3
        
    def test_sequence_attention_mask_generation(self, sequence_batcher):
        """Test attention mask generation for padded sequences"""
        sequences = [
            torch.randn(8, 64),   # sequence_length = 8
            torch.randn(12, 64),  # sequence_length = 12
            torch.randn(6, 64),   # sequence_length = 6
        ]
        
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        padded_batch, attention_mask = sequence_batcher.create_padded_batch_with_mask(requests)
        
        # Check attention mask dimensions
        assert attention_mask.shape == (3, 12)  # batch_size x max_seq_len
        
        # Check attention mask values
        assert attention_mask[0, :8].all()   # First 8 positions should be True
        assert not attention_mask[0, 8:].any()  # Rest should be False
        assert attention_mask[1, :12].all()  # All 12 positions should be True
        assert attention_mask[2, :6].all()   # First 6 positions should be True
        assert not attention_mask[2, 6:].any()  # Rest should be False
        
    def test_dynamic_bucket_adjustment(self, sequence_batcher):
        """Test dynamic bucket size adjustment"""
        # Create many sequences to trigger bucket adjustment
        sequences = [torch.randn(i, 128) for i in range(10, 100, 5)]
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        initial_bucket_count = len(sequence_batcher.get_bucket_boundaries())
        
        # Process requests - should trigger bucket adjustment
        sequence_batcher.process_requests_adaptive(requests)
        
        final_bucket_count = len(sequence_batcher.get_bucket_boundaries())
        
        # Bucket count should adapt to sequence length distribution
        # (specific assertion depends on implementation)
        assert final_bucket_count >= initial_bucket_count
        
    def test_batching_statistics(self, sequence_batcher):
        """Test batching statistics collection"""
        sequences = [torch.randn(i*4, 64) for i in range(1, 6)]
        requests = [
            AssetPredictionRequest(f"asset_{i}", seq, seq.shape[0])
            for i, seq in enumerate(sequences)
        ]
        
        # Process multiple batches
        for _ in range(3):
            sequence_batcher.create_sequence_buckets(requests)
        
        stats = sequence_batcher.get_batching_statistics()
        
        assert "total_requests_processed" in stats
        assert "average_batch_size" in stats
        assert "bucket_utilization" in stats
        assert "padding_efficiency" in stats
        assert stats["total_requests_processed"] > 0


class TestMultiAssetBatcher:
    """Test multi-asset prediction batching"""
    
    @pytest.fixture
    def asset_batcher(self):
        """Fixture providing multi-asset batcher"""
        config = DynamicBatchingConfig(max_batch_size=6, batch_timeout_ms=50)
        return MultiAssetBatcher(config)
    
    def test_asset_batcher_initialization(self, asset_batcher):
        """Test MultiAssetBatcher initialization"""
        assert asset_batcher.config is not None
        assert hasattr(asset_batcher, 'asset_queues')
        assert hasattr(asset_batcher, 'batching_metrics')
        
    def test_multi_asset_batching(self, asset_batcher):
        """Test batching requests from multiple assets"""
        # Create requests for different assets
        btc_requests = [
            AssetPredictionRequest("BTC", torch.randn(20, 10), 20, priority=1)
            for _ in range(3)
        ]
        eth_requests = [
            AssetPredictionRequest("ETH", torch.randn(20, 10), 20, priority=1)
            for _ in range(2)
        ]
        ada_requests = [
            AssetPredictionRequest("ADA", torch.randn(20, 10), 20, priority=2)
            for _ in range(4)
        ]
        
        all_requests = btc_requests + eth_requests + ada_requests
        
        batches = asset_batcher.create_mixed_asset_batches(all_requests)
        
        # Should create multiple batches due to max_batch_size=6
        assert len(batches) >= 2
        
        # Each batch should not exceed max size
        for batch in batches:
            assert len(batch.requests) <= 6
            
    def test_asset_priority_batching(self, asset_batcher):
        """Test priority-based batching"""
        high_priority_requests = [
            AssetPredictionRequest("BTC", torch.randn(15, 8), 15, priority=3)
            for _ in range(2)
        ]
        low_priority_requests = [
            AssetPredictionRequest("ETH", torch.randn(15, 8), 15, priority=1)
            for _ in range(4)
        ]
        
        all_requests = high_priority_requests + low_priority_requests
        
        batches = asset_batcher.create_priority_ordered_batches(all_requests)
        
        # First batch should contain high priority requests
        first_batch_priorities = [req.priority for req in batches[0].requests]
        assert max(first_batch_priorities) >= 3
        
    def test_asset_load_balancing(self, asset_batcher):
        """Test load balancing across assets"""
        # Create uneven distribution of requests
        heavy_asset_requests = [
            AssetPredictionRequest("BTC", torch.randn(10, 5), 10)
            for _ in range(10)
        ]
        light_asset_requests = [
            AssetPredictionRequest("ETH", torch.randn(10, 5), 10)
            for _ in range(2)
        ]
        
        all_requests = heavy_asset_requests + light_asset_requests
        
        balanced_batches = asset_batcher.create_load_balanced_batches(all_requests)
        
        # Batches should have mixed assets when possible
        mixed_batch_count = 0
        for batch in balanced_batches:
            asset_ids = set(req.asset_id for req in batch.requests)
            if len(asset_ids) > 1:
                mixed_batch_count += 1
        
        assert mixed_batch_count > 0
        
    def test_temporal_batching(self, asset_batcher):
        """Test temporal-aware batching (recent requests first)"""
        old_requests = []
        for i in range(3):
            req = AssetPredictionRequest("BTC", torch.randn(12, 6), 12)
            req.timestamp = time.time() - 10  # 10 seconds ago
            old_requests.append(req)
        
        new_requests = []
        for i in range(3):
            req = AssetPredictionRequest("ETH", torch.randn(12, 6), 12)
            req.timestamp = time.time()  # Now
            new_requests.append(req)
        
        all_requests = old_requests + new_requests
        
        temporal_batches = asset_batcher.create_temporal_ordered_batches(all_requests)
        
        # First batch should contain newer requests
        first_batch_timestamps = [req.timestamp for req in temporal_batches[0].requests]
        latest_timestamp = max(first_batch_timestamps)
        
        # Should prioritize recent requests
        assert latest_timestamp > time.time() - 2
        
    def test_asset_similarity_batching(self, asset_batcher):
        """Test batching similar assets together"""
        # Create similar asset types
        crypto_requests = [
            AssetPredictionRequest("BTC", torch.randn(16, 8), 16),
            AssetPredictionRequest("ETH", torch.randn(16, 8), 16),
            AssetPredictionRequest("ADA", torch.randn(16, 8), 16),
        ]
        
        stock_requests = [
            AssetPredictionRequest("AAPL", torch.randn(16, 8), 16),
            AssetPredictionRequest("GOOGL", torch.randn(16, 8), 16),
            AssetPredictionRequest("MSFT", torch.randn(16, 8), 16),
        ]
        
        all_requests = crypto_requests + stock_requests
        
        similarity_batches = asset_batcher.create_similarity_based_batches(all_requests)
        
        # Should group similar asset types
        for batch in similarity_batches:
            asset_ids = [req.asset_id for req in batch.requests]
            # Check if batch contains mostly crypto or mostly stocks
            crypto_count = sum(1 for aid in asset_ids if aid in ["BTC", "ETH", "ADA"])
            stock_count = sum(1 for aid in asset_ids if aid in ["AAPL", "GOOGL", "MSFT"])
            
            # At least 2/3 of batch should be same type
            assert crypto_count >= len(asset_ids) * 0.66 or stock_count >= len(asset_ids) * 0.66


class TestAdaptiveBatchProcessor:
    """Test adaptive batch processing"""
    
    @pytest.fixture
    def adaptive_processor(self):
        """Fixture providing adaptive batch processor"""
        config = DynamicBatchingConfig(
            max_batch_size=8,
            min_batch_size=1,
            batch_timeout_ms=100,
            enable_adaptive_sizing=True
        )
        return AdaptiveBatchProcessor(config)
    
    def test_adaptive_processor_initialization(self, adaptive_processor):
        """Test AdaptiveBatchProcessor initialization"""
        assert adaptive_processor.config is not None
        assert hasattr(adaptive_processor, 'performance_history')
        assert hasattr(adaptive_processor, 'current_batch_size')
        assert hasattr(adaptive_processor, 'adaptation_metrics')
        
    async def test_adaptive_batch_size_adjustment(self, adaptive_processor):
        """Test automatic batch size adjustment based on performance"""
        # Simulate high latency scenario
        performance_data = {
            "average_latency_ms": 150,  # High latency
            "memory_usage_mb": 200,
            "throughput_req_per_sec": 20,
            "error_rate": 0.01
        }
        
        old_batch_size = adaptive_processor.current_batch_size
        
        await adaptive_processor.adapt_batch_size(performance_data)
        
        new_batch_size = adaptive_processor.current_batch_size
        
        # Should reduce batch size to improve latency
        assert new_batch_size <= old_batch_size
        
    async def test_adaptive_timeout_adjustment(self, adaptive_processor):
        """Test automatic timeout adjustment"""
        # Simulate low request rate scenario
        request_rate_data = {
            "requests_per_second": 5,  # Low rate
            "queue_length": 2,
            "average_wait_time_ms": 80
        }
        
        old_timeout = adaptive_processor.config.batch_timeout_ms
        
        await adaptive_processor.adapt_timeout(request_rate_data)
        
        new_timeout = adaptive_processor.config.batch_timeout_ms
        
        # Should increase timeout when request rate is low
        assert new_timeout >= old_timeout
        
    async def test_memory_pressure_adaptation(self, adaptive_processor):
        """Test adaptation under memory pressure"""
        # Simulate high memory usage
        memory_data = {
            "memory_usage_mb": 450,  # High usage
            "memory_limit_mb": 512,
            "memory_pressure": 0.88
        }
        
        await adaptive_processor.adapt_to_memory_pressure(memory_data)
        
        # Should reduce batch size under memory pressure
        assert adaptive_processor.current_batch_size < adaptive_processor.config.max_batch_size
        
    def test_performance_trend_analysis(self, adaptive_processor):
        """Test performance trend analysis"""
        # Add performance history
        for i in range(10):
            performance_data = {
                "latency_ms": 100 + i * 5,  # Increasing latency
                "throughput": 50 - i * 2,   # Decreasing throughput
                "batch_size": 8,
                "timestamp": time.time() - (10 - i)
            }
            adaptive_processor.add_performance_data(performance_data)
        
        trend_analysis = adaptive_processor.analyze_performance_trends()
        
        assert "latency_trend" in trend_analysis
        assert "throughput_trend" in trend_analysis
        assert trend_analysis["latency_trend"] == "increasing"
        assert trend_analysis["throughput_trend"] == "decreasing"
        
    def test_adaptive_strategy_selection(self, adaptive_processor):
        """Test adaptive strategy selection"""
        # Test different scenarios
        scenarios = [
            {"latency_ms": 200, "memory_mb": 100, "error_rate": 0.001},  # High latency
            {"latency_ms": 50, "memory_mb": 400, "error_rate": 0.001},   # High memory
            {"latency_ms": 60, "memory_mb": 150, "error_rate": 0.05},    # High error rate
        ]
        
        for scenario in scenarios:
            strategy = adaptive_processor.select_adaptation_strategy(scenario)
            
            assert strategy in ["reduce_batch_size", "increase_timeout", 
                              "memory_optimization", "error_reduction"]


class TestDynamicBatchingOptimizer:
    """Test dynamic batching optimizer from inference_optimizer.py"""
    
    @pytest.fixture
    def batching_optimizer(self):
        """Fixture providing dynamic batching optimizer"""
        config = DynamicBatchingConfig(max_batch_size=8, batch_timeout_ms=50)
        return DynamicBatchingOptimizer(config)
    
    def test_optimizer_initialization(self, batching_optimizer):
        """Test DynamicBatchingOptimizer initialization"""
        assert batching_optimizer.config is not None
        assert hasattr(batching_optimizer, 'sequence_batcher')
        assert hasattr(batching_optimizer, 'performance_tracker')
        
    async def test_optimizer_request_processing(self, batching_optimizer):
        """Test request processing optimization"""
        # Create mock model
        mock_model = Mock()
        mock_model.return_value = torch.randn(4, 20, 512)  # batch_size=4, seq_len=20, d_model=512
        
        # Create requests with different sequence lengths
        requests = [
            AssetPredictionRequest("BTC", torch.randn(16, 256), 16),
            AssetPredictionRequest("ETH", torch.randn(20, 256), 20),
            AssetPredictionRequest("ADA", torch.randn(18, 256), 18),
            AssetPredictionRequest("DOT", torch.randn(24, 256), 24),
        ]
        
        results = await batching_optimizer.process_requests_optimized(mock_model, requests)
        
        assert len(results) == 4
        for result in results:
            assert "prediction" in result
            assert "latency_ms" in result
            assert "batch_info" in result
            
    def test_optimizer_memory_estimation(self, batching_optimizer):
        """Test memory usage estimation for batching"""
        requests = [
            AssetPredictionRequest("BTC", torch.randn(32, 128), 32),
            AssetPredictionRequest("ETH", torch.randn(28, 128), 28),
            AssetPredictionRequest("ADA", torch.randn(35, 128), 35),
        ]
        
        memory_estimate = batching_optimizer.estimate_batch_memory_usage(requests)
        
        assert "total_memory_mb" in memory_estimate
        assert "per_request_memory_mb" in memory_estimate
        assert "padding_overhead_mb" in memory_estimate
        assert memory_estimate["total_memory_mb"] > 0
        
    def test_optimizer_performance_tracking(self, batching_optimizer):
        """Test performance tracking"""
        # Simulate processing results
        batch_results = [
            {"latency_ms": 45, "batch_size": 4, "memory_mb": 128},
            {"latency_ms": 52, "batch_size": 3, "memory_mb": 96},
            {"latency_ms": 38, "batch_size": 5, "memory_mb": 160},
        ]
        
        for result in batch_results:
            batching_optimizer.record_batch_performance(result)
        
        performance_summary = batching_optimizer.get_performance_summary()
        
        assert "average_latency_ms" in performance_summary
        assert "average_batch_size" in performance_summary
        assert "throughput_batches_per_sec" in performance_summary
        assert performance_summary["average_latency_ms"] > 0


class TestBatchSizeOptimizer:
    """Test batch size optimizer"""
    
    @pytest.fixture
    def batch_size_optimizer(self):
        """Fixture providing batch size optimizer"""
        return BatchSizeOptimizer(
            target_latency_ms=100,
            memory_limit_mb=512,
            optimization_objective="balanced"
        )
    
    def test_batch_size_optimizer_initialization(self, batch_size_optimizer):
        """Test BatchSizeOptimizer initialization"""
        assert batch_size_optimizer.target_latency_ms > 0
        assert batch_size_optimizer.memory_limit_mb > 0
        assert batch_size_optimizer.optimization_objective in ["latency", "throughput", "balanced"]
        
    def test_optimal_batch_size_calculation(self, batch_size_optimizer):
        """Test optimal batch size calculation"""
        # Performance data for different batch sizes
        performance_data = [
            {"batch_size": 1, "latency_ms": 20, "memory_mb": 50, "throughput": 50},
            {"batch_size": 2, "latency_ms": 35, "memory_mb": 90, "throughput": 57},
            {"batch_size": 4, "latency_ms": 65, "memory_mb": 160, "throughput": 61},
            {"batch_size": 8, "latency_ms": 120, "memory_mb": 300, "throughput": 66},
            {"batch_size": 16, "latency_ms": 230, "memory_mb": 580, "throughput": 69},
        ]
        
        optimal_size = batch_size_optimizer.find_optimal_batch_size(performance_data)
        
        # Should find size that meets latency constraint (<=100ms) and memory limit (<=512mb)
        assert 1 <= optimal_size <= 8  # Size 16 exceeds both constraints
        
    def test_adaptive_batch_size_adjustment(self, batch_size_optimizer):
        """Test adaptive batch size adjustment"""
        current_performance = {
            "current_batch_size": 4,
            "latency_ms": 150,  # Exceeds target of 100ms
            "memory_mb": 200,
            "throughput": 45
        }
        
        adjustment = batch_size_optimizer.suggest_batch_size_adjustment(current_performance)
        
        assert adjustment["action"] in ["decrease", "increase", "maintain"]
        assert adjustment["suggested_size"] != current_performance["current_batch_size"]
        # Should suggest decrease due to high latency
        assert adjustment["action"] == "decrease"
        
    def test_sequence_length_aware_optimization(self, batch_size_optimizer):
        """Test sequence length aware batch size optimization"""
        sequence_profiles = [
            {"avg_seq_len": 32, "seq_variance": 4, "frequency": 0.4},
            {"avg_seq_len": 64, "seq_variance": 8, "frequency": 0.3},
            {"avg_seq_len": 128, "seq_variance": 16, "frequency": 0.3},
        ]
        
        optimized_sizes = batch_size_optimizer.optimize_for_sequence_profiles(sequence_profiles)
        
        assert len(optimized_sizes) == 3
        for profile, size in zip(sequence_profiles, optimized_sizes):
            # Longer sequences should generally have smaller batch sizes
            if profile["avg_seq_len"] > 64:
                assert size <= 4


class TestTransformerBatchProcessor:
    """Test transformer-specific batch processor"""
    
    @pytest.fixture
    def transformer_processor(self):
        """Fixture providing transformer batch processor"""
        return TransformerBatchProcessor(
            d_model=256,
            n_heads=8,
            max_seq_length=128,
            batch_config=DynamicBatchingConfig(max_batch_size=6)
        )
    
    def test_processor_initialization(self, transformer_processor):
        """Test TransformerBatchProcessor initialization"""
        assert transformer_processor.d_model > 0
        assert transformer_processor.n_heads > 0
        assert transformer_processor.max_seq_length > 0
        assert transformer_processor.batch_config is not None
        
    async def test_transformer_batch_inference(self, transformer_processor):
        """Test transformer batch inference"""
        # Mock transformer model
        mock_transformer = Mock()
        mock_transformer.return_value = torch.randn(3, 20, 256)  # batch=3, seq=20, d_model=256
        
        # Create input sequences
        input_sequences = [
            torch.randint(0, 1000, (16,)),  # vocab_size=1000
            torch.randint(0, 1000, (20,)),
            torch.randint(0, 1000, (18,)),
        ]
        
        results = await transformer_processor.process_batch(mock_transformer, input_sequences)
        
        assert len(results) == 3
        for result in results:
            assert "logits" in result
            assert "attention_weights" in result
            assert "processing_info" in result
            
    def test_attention_computation_optimization(self, transformer_processor):
        """Test attention computation optimization"""
        batch_size, seq_len = 4, 32
        
        query = torch.randn(batch_size, seq_len, transformer_processor.d_model)
        key = torch.randn(batch_size, seq_len, transformer_processor.d_model)
        value = torch.randn(batch_size, seq_len, transformer_processor.d_model)
        
        # Test optimized attention computation
        attention_output = transformer_processor.compute_optimized_attention(query, key, value)
        
        assert attention_output.shape == (batch_size, seq_len, transformer_processor.d_model)
        
    def test_memory_efficient_processing(self, transformer_processor):
        """Test memory efficient batch processing"""
        # Create large batch to test memory efficiency
        large_sequences = [torch.randint(0, 1000, (64,)) for _ in range(10)]
        
        memory_efficient_batches = transformer_processor.create_memory_efficient_batches(
            large_sequences, memory_limit_mb=256
        )
        
        # Should split into multiple smaller batches
        assert len(memory_efficient_batches) > 1
        
        # Each batch should respect memory limit
        for batch in memory_efficient_batches:
            estimated_memory = transformer_processor.estimate_batch_memory(batch)
            assert estimated_memory <= 256  # MB


class TestSequenceAwareBatcher:
    """Test sequence-aware batching from batch_prediction.py"""
    
    @pytest.fixture
    def sequence_aware_batcher(self):
        """Fixture providing sequence-aware batcher"""
        return SequenceAwareBatcher(
            max_sequence_length=128,
            padding_strategy="longest",
            bucketing_strategy="power_of_two"
        )
    
    def test_sequence_aware_initialization(self, sequence_aware_batcher):
        """Test SequenceAwareBatcher initialization"""
        assert sequence_aware_batcher.max_sequence_length > 0
        assert sequence_aware_batcher.padding_strategy in ["longest", "fixed", "bucket"]
        assert sequence_aware_batcher.bucketing_strategy in ["uniform", "power_of_two", "adaptive"]
        
    def test_power_of_two_bucketing(self, sequence_aware_batcher):
        """Test power of two bucketing strategy"""
        sequences = [
            torch.randn(5, 64),    # Bucket: 8
            torch.randn(12, 64),   # Bucket: 16  
            torch.randn(25, 64),   # Bucket: 32
            torch.randn(45, 64),   # Bucket: 64
            torch.randn(70, 64),   # Bucket: 128
        ]
        
        buckets = sequence_aware_batcher.create_power_of_two_buckets(sequences)
        
        # Should have buckets for powers of 2
        expected_bucket_sizes = [8, 16, 32, 64, 128]
        actual_bucket_sizes = list(buckets.keys())
        
        for size in actual_bucket_sizes:
            assert size in expected_bucket_sizes
            
    def test_adaptive_bucketing(self, sequence_aware_batcher):
        """Test adaptive bucketing based on sequence distribution"""
        # Create bimodal distribution
        short_sequences = [torch.randn(i, 32) for i in range(10, 20)]
        long_sequences = [torch.randn(i, 32) for i in range(80, 90)]
        all_sequences = short_sequences + long_sequences
        
        adaptive_buckets = sequence_aware_batcher.create_adaptive_buckets(all_sequences)
        
        # Should create buckets that match the distribution
        assert len(adaptive_buckets) >= 2  # At least two clusters
        
    def test_sequence_similarity_batching(self, sequence_aware_batcher):
        """Test batching based on sequence similarity"""
        # Create sequences with different patterns
        pattern_a_sequences = [torch.ones(20, 16) * 0.5 for _ in range(3)]
        pattern_b_sequences = [torch.ones(20, 16) * -0.5 for _ in range(3)]
        
        all_sequences = pattern_a_sequences + pattern_b_sequences
        
        similarity_batches = sequence_aware_batcher.create_similarity_based_batches(
            all_sequences, similarity_threshold=0.8
        )
        
        # Should group similar sequences together
        assert len(similarity_batches) >= 2


class TestMemoryAwareBatchSizer:
    """Test memory-aware batch sizing"""
    
    @pytest.fixture
    def memory_aware_sizer(self):
        """Fixture providing memory-aware batch sizer"""
        return MemoryAwareBatchSizer(
            memory_limit_mb=512,
            safety_margin=0.1,
            enable_dynamic_adjustment=True
        )
    
    def test_memory_sizer_initialization(self, memory_aware_sizer):
        """Test MemoryAwareBatchSizer initialization"""
        assert memory_aware_sizer.memory_limit_mb > 0
        assert 0 < memory_aware_sizer.safety_margin < 1
        assert memory_aware_sizer.enable_dynamic_adjustment is not None
        
    def test_batch_size_calculation_from_memory(self, memory_aware_sizer):
        """Test batch size calculation based on memory constraints"""
        # Sample sequence to estimate memory per item
        sample_sequence = torch.randn(64, 256)  # seq_len=64, d_model=256
        
        max_batch_size = memory_aware_sizer.calculate_max_batch_size(
            sample_sequence, model_memory_mb=100
        )
        
        assert max_batch_size > 0
        assert max_batch_size <= 32  # Reasonable upper bound
        
    def test_dynamic_adjustment_under_pressure(self, memory_aware_sizer):
        """Test dynamic adjustment under memory pressure"""
        # Simulate memory pressure
        current_memory_usage = 450  # MB
        current_batch_size = 8
        
        adjusted_size = memory_aware_sizer.adjust_for_memory_pressure(
            current_batch_size, current_memory_usage
        )
        
        # Should reduce batch size under pressure
        assert adjusted_size < current_batch_size
        
    def test_memory_monitoring_integration(self, memory_aware_sizer):
        """Test integration with memory monitoring"""
        if torch.cuda.is_available():
            # Test GPU memory monitoring
            gpu_memory_info = memory_aware_sizer.get_gpu_memory_info()
            
            assert "total_memory_mb" in gpu_memory_info
            assert "allocated_memory_mb" in gpu_memory_info
            assert "free_memory_mb" in gpu_memory_info
            
        # Test system memory monitoring
        system_memory_info = memory_aware_sizer.get_system_memory_info()
        
        assert "total_memory_mb" in system_memory_info
        assert "available_memory_mb" in system_memory_info
        assert "memory_pressure" in system_memory_info


class TestIntegrationDynamicBatching:
    """Integration tests for dynamic batching functionality"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_dynamic_batching(self):
        """Test end-to-end dynamic batching pipeline"""
        # Create complete batching pipeline
        config = DynamicBatchingConfig(
            max_batch_size=4,
            batch_timeout_ms=50,
            sequence_padding_strategy="longest",
            enable_adaptive_sizing=True
        )
        
        sequence_batcher = SequenceLengthBatcher(config)
        multi_asset_batcher = MultiAssetBatcher(config)
        adaptive_processor = AdaptiveBatchProcessor(config)
        
        # Create diverse requests
        requests = [
            AssetPredictionRequest("BTC", torch.randn(16, 32), 16, priority=2),
            AssetPredictionRequest("ETH", torch.randn(20, 32), 20, priority=1),
            AssetPredictionRequest("ADA", torch.randn(18, 32), 18, priority=1),
            AssetPredictionRequest("DOT", torch.randn(22, 32), 22, priority=3),
            AssetPredictionRequest("SOL", torch.randn(14, 32), 14, priority=1),
        ]
        
        # Process through pipeline
        sequence_buckets = sequence_batcher.create_sequence_buckets(requests)
        assert len(sequence_buckets) >= 1
        
        priority_batches = multi_asset_batcher.create_priority_ordered_batches(requests)
        assert len(priority_batches) >= 1
        
        # Test adaptive processing
        performance_data = {
            "average_latency_ms": 75,
            "memory_usage_mb": 200,
            "throughput_req_per_sec": 30
        }
        
        await adaptive_processor.adapt_batch_size(performance_data)
        assert adaptive_processor.current_batch_size > 0
        
    def test_performance_comparison_batching_strategies(self):
        """Compare performance of different batching strategies"""
        config = DynamicBatchingConfig(max_batch_size=8)
        
        # Create test requests
        requests = [
            AssetPredictionRequest(f"asset_{i}", torch.randn(16 + i*4, 64), 16 + i*4)
            for i in range(20)
        ]
        
        # Strategy 1: Simple batching
        simple_batcher = SequenceLengthBatcher(config)
        
        start_time = time.time()
        simple_buckets = simple_batcher.create_sequence_buckets(requests)
        simple_time = time.time() - start_time
        
        # Strategy 2: Multi-asset batching
        multi_asset_batcher = MultiAssetBatcher(config)
        
        start_time = time.time()
        asset_batches = multi_asset_batcher.create_mixed_asset_batches(requests)
        asset_time = time.time() - start_time
        
        # Both strategies should complete successfully
        assert len(simple_buckets) > 0
        assert len(asset_batches) > 0
        
        # Performance comparison (times should be comparable)
        assert simple_time < 1.0  # Should be fast
        assert asset_time < 1.0
        
    @pytest.mark.slow
    def test_memory_efficiency_large_batches(self):
        """Test memory efficiency with large batches"""
        memory_sizer = MemoryAwareBatchSizer(memory_limit_mb=256)
        
        # Create large sequences
        large_sequences = [
            torch.randn(128, 512) for _ in range(50)  # Large sequences
        ]
        
        # Test memory-aware batching
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        batches = memory_sizer.create_memory_constrained_batches(large_sequences)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory increase should be reasonable
            assert memory_increase < 300  # MB
        
        # Should create multiple batches to respect memory constraints
        assert len(batches) > 1
        
    def test_real_time_batching_latency(self):
        """Test batching latency for real-time requirements"""
        config = DynamicBatchingConfig(
            max_batch_size=4,
            batch_timeout_ms=10,  # Very tight timeout
            enable_adaptive_sizing=True
        )
        
        processor = AdaptiveBatchProcessor(config)
        
        # Create time-sensitive requests
        requests = [
            AssetPredictionRequest("BTC", torch.randn(20, 16), 20)
            for _ in range(5)
        ]
        
        # Measure batching latency
        start_time = time.time()
        buckets = processor.create_time_constrained_batches(requests)
        batching_latency = (time.time() - start_time) * 1000  # ms
        
        # Batching should be very fast for real-time requirements
        assert batching_latency < 5.0  # Less than 5ms
        assert len(buckets) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])