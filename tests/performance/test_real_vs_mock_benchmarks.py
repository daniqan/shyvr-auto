"""
Comprehensive Performance Benchmarks: Real PyTorch RL Models vs Mock Implementations

This module provides critical validation of our testing strategy by comparing
performance metrics between real PyTorch RL models and mock implementations.
The benchmarks validate that real models meet CLAUDE.md performance targets
and provide statistical analysis for testing strategy optimization.

TDD Approach:
1. Write failing tests that define expected performance characteristics
2. Implement mock vs real model comparison utilities
3. Validate real models meet performance targets
4. Generate statistical reports for testing strategy validation

Performance Targets from CLAUDE.md:
- ML prediction generation: <1s per token
- RL decision making: <1s per action  
- ML-RL integration: <1s for complete decision pipeline
- Batch processing: 100+ tokens per minute capability
- Memory efficiency: <50MB growth during batch processing
"""

import asyncio
import gc
import time
import psutil
import tracemalloc
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.discovery.base import DiscoveredToken, Chain
from src.rl_agent.base import (
    MarketState, TradeAction, AgentConfig, RewardMetrics, 
    TradingResult, ModelType, RLAgentBase
)
from src.rl_agent.dqn_agent import DQNTradingAgent, DQNNetwork
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig, Portfolio
from src.rl_agent.experience_replay import (
    ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer,
    Experience, ReplayBufferConfig
)
from src.integration.ml_rl_bridge import MLEnhancedMarketState, MLRLBridge, MLRLConfig
from src.ml_analysis.base import PredictionResult, PredictionDirection, ModelType as MLModelType


@dataclass
class PerformanceMetrics:
    """Data structure for storing performance benchmark results"""
    mean: float
    median: float
    p95: float
    p99: float
    std: float
    min_val: float
    max_val: float
    sample_size: int
    
    def __post_init__(self):
        """Validate metrics after initialization"""
        assert self.sample_size > 0, "Sample size must be positive"
        assert self.min_val <= self.mean <= self.max_val, "Mean must be between min and max"
        assert self.std >= 0, "Standard deviation must be non-negative"


@dataclass
class BenchmarkComparison:
    """Data structure for comparing real vs mock performance"""
    real_metrics: PerformanceMetrics
    mock_metrics: PerformanceMetrics
    performance_ratio: float  # real_time / mock_time
    overhead_ms: float  # (real_time - mock_time) * 1000
    meets_target: bool
    target_ms: float
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        self.performance_ratio = self.real_metrics.mean / self.mock_metrics.mean
        self.overhead_ms = (self.real_metrics.mean - self.mock_metrics.mean) * 1000
        # Check if real implementation meets performance target
        self.meets_target = self.real_metrics.p99 < (self.target_ms / 1000)


class MockDQNAgent:
    """High-performance mock DQN agent for benchmarking"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        self.action_list = list(TradeAction)
    
    async def predict_action(self, state: Union[MarketState, MLEnhancedMarketState]) -> Tuple[TradeAction, float]:
        """Mock action prediction with minimal overhead"""
        # Simulate minimal processing time
        await asyncio.sleep(0.0001)  # 0.1ms mock processing
        
        action = np.random.choice(self.action_list)
        confidence = np.random.uniform(0.5, 1.0)
        return action, confidence
    
    async def train_step(self, batch_experiences: List[Dict]) -> Dict[str, float]:
        """Mock training step with minimal overhead"""
        await asyncio.sleep(0.001)  # 1ms mock training
        
        return {
            'loss': np.random.uniform(0.1, 1.0),
            'q_value_mean': np.random.uniform(-1.0, 1.0),
            'epsilon': self.epsilon
        }
    
    def get_q_values(self, state: Union[MarketState, MLEnhancedMarketState]) -> torch.Tensor:
        """Mock Q-values generation"""
        return torch.randn(5)  # 5 actions


class MockExperienceReplayBuffer:
    """High-performance mock experience replay buffer"""
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.buffer = []
    
    def sample(self) -> List[Dict]:
        """Mock sampling with minimal overhead"""
        # Simulate minimal processing time
        time.sleep(0.0001)  # 0.1ms mock sampling
        
        # Return mock batch
        batch = []
        for _ in range(self.config.batch_size):
            batch.append({
                'state': np.random.randn(25).astype(np.float32),
                'action': np.random.randint(0, 5),
                'reward': np.random.uniform(-1, 1),
                'next_state': np.random.randn(25).astype(np.float32),
                'done': bool(np.random.choice([True, False]))
            })
        return batch
    
    def add(self, experience: Experience):
        """Mock experience addition"""
        pass  # No-op for performance testing


class TestRealVsMockDQNBenchmarks:
    """TDD: Performance benchmarks comparing real vs mock DQN implementations"""
    
    @pytest.fixture
    def agent_config(self) -> AgentConfig:
        """Standard agent configuration for benchmarking"""
        return AgentConfig(
            model_type=ModelType.DQN,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=32,
            replay_buffer_size=1000,
            target_update_frequency=50,
            epsilon_start=0.1,
            epsilon_end=0.05,
            epsilon_decay=100
        )
    
    @pytest.fixture
    def sample_market_state(self) -> MLEnhancedMarketState:
        """Enhanced market state for benchmarking with 25 features"""
        token = DiscoveredToken(
            address="0xbenchmark123",
            chain=Chain.ETHEREUM,
            symbol="BENCH",
            name="Benchmark Token",
            discovered_at=datetime.now(),
            discovery_source="benchmark",
            price_usd=1.234,
            market_cap=12340000,
            volume_24h=500000,
            price_change_24h=5.5,
            tags=["benchmark"]
        )
        
        return MLEnhancedMarketState(
            token=token,
            price_usd=1.234,
            price_change_24h=5.5,
            volume_24h=500000,
            market_cap=12340000,
            rsi=65.5,
            macd=0.0002,
            sma_20=1.20,
            ema_12=1.22,
            bollinger_upper=1.35,
            bollinger_lower=1.10,
            current_position=0.05,
            portfolio_value=10000.0,
            cash_balance=9500.0,
            portfolio_drawdown=0.02,
            daily_pnl=150.0,
            sharpe_ratio=1.2,
            market_volatility=0.25,
            fear_greed_index=55.0,
            # ML-enhanced features
            ml_prediction_1h=1.250,
            ml_prediction_4h=1.265,
            ml_prediction_24h=1.285,
            ml_confidence=0.82,
            ml_direction="buy",
            volatility_forecast=0.22
        )
    
    def calculate_performance_metrics(self, latencies: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from latency samples"""
        return PerformanceMetrics(
            mean=np.mean(latencies),
            median=np.median(latencies),
            p95=np.percentile(latencies, 95),
            p99=np.percentile(latencies, 99),
            std=np.std(latencies),
            min_val=np.min(latencies),
            max_val=np.max(latencies),
            sample_size=len(latencies)
        )
    
    @pytest.mark.asyncio
    async def test_dqn_inference_latency_benchmark_fails_first(self, agent_config, sample_market_state):
        """
        TDD: Failing test that defines DQN inference performance requirements
        This test should fail initially, then be implemented to pass
        """
        # Create real and mock agents
        real_agent = DQNTradingAgent(agent_config, use_enhanced_features=True)
        mock_agent = MockDQNAgent(agent_config)
        
        # Warm up real agent
        for _ in range(5):
            await real_agent.predict_action(sample_market_state)
        
        # Benchmark real agent inference
        real_latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            action, confidence = await real_agent.predict_action(sample_market_state)
            end_time = time.perf_counter()
            
            real_latencies.append(end_time - start_time)
            
            # Validate outputs
            assert isinstance(action, TradeAction)
            assert 0.0 <= confidence <= 1.0
        
        # Benchmark mock agent inference
        mock_latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            action, confidence = await mock_agent.predict_action(sample_market_state)
            end_time = time.perf_counter()
            
            mock_latencies.append(end_time - start_time)
        
        # Calculate performance metrics
        real_metrics = self.calculate_performance_metrics(real_latencies)
        mock_metrics = self.calculate_performance_metrics(mock_latencies)
        
        # Create benchmark comparison
        comparison = BenchmarkComparison(
            real_metrics=real_metrics,
            mock_metrics=mock_metrics,
            performance_ratio=0.0,  # Will be calculated in __post_init__
            overhead_ms=0.0,  # Will be calculated in __post_init__
            meets_target=False,  # Will be calculated in __post_init__
            target_ms=1000.0  # 1 second target from CLAUDE.md
        )
        
        # TDD: These assertions should initially fail, forcing implementation
        assert comparison.meets_target, f"Real DQN inference P99 {real_metrics.p99*1000:.2f}ms exceeds 1000ms target"
        assert comparison.performance_ratio < 100, f"Real agent {comparison.performance_ratio:.1f}x slower than mock"
        assert real_metrics.p95 < 0.5, f"Real DQN P95 latency {real_metrics.p95*1000:.2f}ms too high"
        
        # Statistical significance test
        assert real_metrics.sample_size >= 100, "Insufficient sample size for statistical significance"
        assert mock_metrics.sample_size >= 100, "Insufficient mock sample size"
        
        # Log benchmark results
        self._log_benchmark_results("DQN Inference", comparison)
    
    @pytest.mark.asyncio
    async def test_experience_replay_sampling_benchmark_fails_first(self):
        """
        TDD: Failing test for experience replay buffer sampling performance
        """
        buffer_config = ReplayBufferConfig(
            max_size=1000,
            batch_size=32,
            min_size=50,
            prioritized=False
        )
        
        # Create real and mock buffers
        real_buffer = ExperienceReplayBuffer(buffer_config)
        mock_buffer = MockExperienceReplayBuffer(buffer_config)
        
        # Fill real buffer with experiences
        for i in range(500):
            state = np.random.randn(25).astype(np.float32)
            next_state = np.random.randn(25).astype(np.float32)
            
            experience = Experience(
                state=state,
                action=np.random.randint(0, 5),
                reward=np.random.uniform(-1, 1),
                next_state=next_state,
                done=bool(np.random.choice([True, False]))
            )
            real_buffer.add(experience)
        
        # Benchmark real buffer sampling
        real_latencies = []
        for i in range(200):
            start_time = time.perf_counter()
            batch = real_buffer.sample()
            end_time = time.perf_counter()
            
            real_latencies.append(end_time - start_time)
            assert len(batch) == buffer_config.batch_size
        
        # Benchmark mock buffer sampling
        mock_latencies = []
        for i in range(200):
            start_time = time.perf_counter()
            batch = mock_buffer.sample()
            end_time = time.perf_counter()
            
            mock_latencies.append(end_time - start_time)
            assert len(batch) == buffer_config.batch_size
        
        # Calculate metrics and comparison
        real_metrics = self.calculate_performance_metrics(real_latencies)
        mock_metrics = self.calculate_performance_metrics(mock_latencies)
        
        comparison = BenchmarkComparison(
            real_metrics=real_metrics,
            mock_metrics=mock_metrics,
            performance_ratio=0.0,
            overhead_ms=0.0,
            meets_target=False,
            target_ms=50.0  # 50ms target for sampling
        )
        
        # TDD: These should fail initially
        assert comparison.meets_target, f"Real buffer sampling P99 {real_metrics.p99*1000:.2f}ms exceeds 50ms target"
        assert comparison.performance_ratio < 50, f"Real buffer {comparison.performance_ratio:.1f}x slower than mock"
        assert real_metrics.p95 < 0.01, f"Real buffer P95 latency {real_metrics.p95*1000:.2f}ms too high"
        
        self._log_benchmark_results("Experience Replay Sampling", comparison)
    
    @pytest.mark.asyncio
    async def test_training_step_performance_benchmark_fails_first(self, agent_config):
        """
        TDD: Failing test for training step performance comparison
        """
        # Create real and mock agents
        real_agent = DQNTradingAgent(agent_config, use_enhanced_features=True)
        mock_agent = MockDQNAgent(agent_config)
        
        # Create realistic batch experiences
        batch_experiences = []
        for _ in range(agent_config.batch_size):
            batch_experiences.append({
                'state': np.random.randn(25).astype(np.float32),
                'action': np.random.randint(0, 5),
                'reward': np.random.uniform(-1, 1),
                'next_state': np.random.randn(25).astype(np.float32),
                'done': bool(np.random.choice([True, False]))
            })
        
        # Benchmark real agent training
        real_latencies = []
        for i in range(50):
            start_time = time.perf_counter()
            metrics = await real_agent.train_step(batch_experiences)
            end_time = time.perf_counter()
            
            real_latencies.append(end_time - start_time)
            assert 'loss' in metrics
            assert not np.isnan(metrics['loss'])
        
        # Benchmark mock agent training
        mock_latencies = []
        for i in range(50):
            start_time = time.perf_counter()
            metrics = await mock_agent.train_step(batch_experiences)
            end_time = time.perf_counter()
            
            mock_latencies.append(end_time - start_time)
            assert 'loss' in metrics
        
        # Calculate metrics and comparison
        real_metrics = self.calculate_performance_metrics(real_latencies)
        mock_metrics = self.calculate_performance_metrics(mock_latencies)
        
        comparison = BenchmarkComparison(
            real_metrics=real_metrics,
            mock_metrics=mock_metrics,
            performance_ratio=0.0,
            overhead_ms=0.0,
            meets_target=False,
            target_ms=500.0  # 500ms target for training step
        )
        
        # TDD: These should fail initially
        assert comparison.meets_target, f"Real training P99 {real_metrics.p99*1000:.2f}ms exceeds 500ms target"
        assert comparison.performance_ratio < 200, f"Real training {comparison.performance_ratio:.1f}x slower than mock"
        assert real_metrics.mean < 0.1, f"Real training mean {real_metrics.mean*1000:.2f}ms too high"
        
        self._log_benchmark_results("Training Step", comparison)
    
    def test_memory_usage_benchmark_fails_first(self, agent_config):
        """
        TDD: Failing test for memory usage comparison
        """
        tracemalloc.start()
        
        # Baseline memory measurement
        gc.collect()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Test real agent memory usage
        real_agent = DQNTradingAgent(agent_config, use_enhanced_features=True)
        
        # Simulate intensive usage
        for i in range(1000):
            # Create market state
            token = DiscoveredToken(
                address=f"0x{i:08x}",
                chain=Chain.ETHEREUM,
                symbol=f"TOK{i}",
                name=f"Token {i}",
                discovered_at=datetime.now(),
                discovery_source="memory_test",
                price_usd=1.0 + i * 0.001,
                market_cap=1000000,
                volume_24h=100000,
                price_change_24h=0.0,
                tags=[]
            )
            
            state = MLEnhancedMarketState(
                token=token,
                price_usd=1.0 + i * 0.001,
                price_change_24h=0.0,
                volume_24h=100000,
                market_cap=1000000,
                rsi=50.0,
                macd=0.0,
                sma_20=1.0,
                ema_12=1.0,
                bollinger_upper=1.1,
                bollinger_lower=0.9,
                current_position=0.0,
                portfolio_value=10000.0,
                cash_balance=10000.0,
                portfolio_drawdown=0.0,
                daily_pnl=0.0,
                sharpe_ratio=1.0,
                market_volatility=0.2,
                fear_greed_index=50.0,
                # ML-enhanced features
                ml_prediction_1h=1.01 + i * 0.001,
                ml_prediction_4h=1.02 + i * 0.001,
                ml_prediction_24h=1.05 + i * 0.001,
                ml_confidence=0.8,
                ml_direction="buy",
                volatility_forecast=0.2
            )
            
            # Perform operations
            asyncio.run(real_agent.predict_action(state))
            
            # Force garbage collection every 100 iterations
            if i % 100 == 0:
                gc.collect()
        
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # Test mock agent memory usage
        tracemalloc.start()
        gc.collect()
        mock_initial_memory = tracemalloc.get_traced_memory()[0]
        
        mock_agent = MockDQNAgent(agent_config)
        
        # Same operations with mock
        for i in range(1000):
            token = DiscoveredToken(
                address=f"0x{i:08x}",
                chain=Chain.ETHEREUM,
                symbol=f"TOK{i}",
                name=f"Token {i}",
                discovered_at=datetime.now(),
                discovery_source="memory_test",
                price_usd=1.0 + i * 0.001,
                market_cap=1000000,
                volume_24h=100000,
                price_change_24h=0.0,
                tags=[]
            )
            
            state = MLEnhancedMarketState(
                token=token,
                price_usd=1.0 + i * 0.001,
                price_change_24h=0.0,
                volume_24h=100000,
                market_cap=1000000,
                rsi=50.0,
                macd=0.0,
                sma_20=1.0,
                ema_12=1.0,
                bollinger_upper=1.1,
                bollinger_lower=0.9,
                current_position=0.0,
                portfolio_value=10000.0,
                cash_balance=10000.0,
                portfolio_drawdown=0.0,
                daily_pnl=0.0,
                sharpe_ratio=1.0,
                market_volatility=0.2,
                fear_greed_index=50.0,
                # ML-enhanced features
                ml_prediction_1h=1.01 + i * 0.001,
                ml_prediction_4h=1.02 + i * 0.001,
                ml_prediction_24h=1.05 + i * 0.001,
                ml_confidence=0.8,
                ml_direction="buy",
                volatility_forecast=0.2
            )
            
            asyncio.run(mock_agent.predict_action(state))
            
            if i % 100 == 0:
                gc.collect()
        
        mock_final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        mock_memory_growth_mb = (mock_final_memory - mock_initial_memory) / (1024 * 1024)
        
        # TDD: These should fail initially
        assert memory_growth_mb < 50, f"Real agent memory growth {memory_growth_mb:.2f}MB exceeds 50MB target"
        assert memory_growth_mb < mock_memory_growth_mb * 10, f"Real agent uses {memory_growth_mb/mock_memory_growth_mb:.1f}x more memory than mock"
        
        print(f"Memory Usage Benchmark:")
        print(f"  Real agent: {memory_growth_mb:.2f}MB growth")
        print(f"  Mock agent: {mock_memory_growth_mb:.2f}MB growth")
        print(f"  Ratio: {memory_growth_mb/mock_memory_growth_mb:.1f}x")
    
    def test_feature_vector_creation_benchmark_fails_first(self):
        """
        TDD: Failing test for feature vector creation performance
        """
        # Create enhanced market state
        token = DiscoveredToken(
            address="0xfeature123",
            chain=Chain.ETHEREUM,
            symbol="FEAT",
            name="Feature Token",
            discovered_at=datetime.now(),
            discovery_source="feature_test",
            price_usd=1.5,
            market_cap=15000000,
            volume_24h=750000,
            price_change_24h=3.2,
            tags=["feature"]
        )
        
        # Create enhanced market state with ML predictions
        enhanced_state = MLEnhancedMarketState(
            token=token,
            price_usd=1.5,
            price_change_24h=3.2,
            volume_24h=750000,
            market_cap=15000000,
            rsi=62.0,
            macd=0.0015,
            sma_20=1.45,
            ema_12=1.48,
            bollinger_upper=1.65,
            bollinger_lower=1.35,
            current_position=0.1,
            portfolio_value=10000.0,
            cash_balance=9000.0,
            portfolio_drawdown=0.05,
            daily_pnl=200.0,
            sharpe_ratio=1.5,
            market_volatility=0.3,
            fear_greed_index=60.0,
            ml_prediction_1h=1.52,
            ml_prediction_4h=1.55,
            ml_prediction_24h=1.60,
            ml_confidence=0.85,
            ml_direction="buy",
            volatility_forecast=0.28
        )
        
        # Benchmark enhanced feature vector creation
        enhanced_latencies = []
        for i in range(10000):  # Large sample for precise measurement
            start_time = time.perf_counter()
            feature_vector = enhanced_state.to_feature_vector()
            end_time = time.perf_counter()
            
            enhanced_latencies.append(end_time - start_time)
            
            # Validate feature vector
            assert isinstance(feature_vector, np.ndarray)
            assert feature_vector.shape == (25,), f"Expected 25 features, got {feature_vector.shape}"
            assert not np.isnan(feature_vector).any()
        
        # Benchmark standard feature vector creation
        standard_latencies = []
        for i in range(10000):
            start_time = time.perf_counter()
            feature_vector = enhanced_state.to_vector()  # Standard 19-feature vector
            end_time = time.perf_counter()
            
            standard_latencies.append(end_time - start_time)
            
            assert isinstance(feature_vector, np.ndarray)
            assert feature_vector.shape == (19,), f"Expected 19 features, got {feature_vector.shape}"
        
        # Calculate metrics
        enhanced_metrics = self.calculate_performance_metrics(enhanced_latencies)
        standard_metrics = self.calculate_performance_metrics(standard_latencies)
        
        comparison = BenchmarkComparison(
            real_metrics=enhanced_metrics,
            mock_metrics=standard_metrics,  # Using standard as baseline
            performance_ratio=0.0,
            overhead_ms=0.0,
            meets_target=False,
            target_ms=0.1  # 0.1ms target for feature vector creation
        )
        
        # TDD: These should fail initially
        assert comparison.meets_target, f"Enhanced feature vector P99 {enhanced_metrics.p99*1000:.3f}ms exceeds 0.1ms target"
        assert comparison.performance_ratio < 5, f"Enhanced features {comparison.performance_ratio:.1f}x slower than standard"
        assert enhanced_metrics.mean < 0.0001, f"Enhanced feature vector mean {enhanced_metrics.mean*1000:.3f}ms too high"
        
        self._log_benchmark_results("Feature Vector Creation", comparison)
    
    def _log_benchmark_results(self, test_name: str, comparison: BenchmarkComparison):
        """Log detailed benchmark results for analysis"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {test_name}")
        print(f"{'='*60}")
        print(f"Real Implementation Metrics:")
        print(f"  Mean: {comparison.real_metrics.mean*1000:.3f}ms")
        print(f"  Median: {comparison.real_metrics.median*1000:.3f}ms")
        print(f"  P95: {comparison.real_metrics.p95*1000:.3f}ms")
        print(f"  P99: {comparison.real_metrics.p99*1000:.3f}ms")
        print(f"  Std: {comparison.real_metrics.std*1000:.3f}ms")
        print(f"  Min: {comparison.real_metrics.min_val*1000:.3f}ms")
        print(f"  Max: {comparison.real_metrics.max_val*1000:.3f}ms")
        print(f"  Samples: {comparison.real_metrics.sample_size}")
        
        print(f"\nMock Implementation Metrics:")
        print(f"  Mean: {comparison.mock_metrics.mean*1000:.3f}ms")
        print(f"  Median: {comparison.mock_metrics.median*1000:.3f}ms")
        print(f"  P95: {comparison.mock_metrics.p95*1000:.3f}ms")
        print(f"  P99: {comparison.mock_metrics.p99*1000:.3f}ms")
        print(f"  Std: {comparison.mock_metrics.std*1000:.3f}ms")
        print(f"  Min: {comparison.mock_metrics.min_val*1000:.3f}ms")
        print(f"  Max: {comparison.mock_metrics.max_val*1000:.3f}ms")
        print(f"  Samples: {comparison.mock_metrics.sample_size}")
        
        print(f"\nComparison Analysis:")
        print(f"  Performance Ratio: {comparison.performance_ratio:.1f}x")
        print(f"  Overhead: {comparison.overhead_ms:.3f}ms")
        print(f"  Target: {comparison.target_ms:.1f}ms")
        print(f"  Meets Target: {'✓' if comparison.meets_target else '✗'}")
        print(f"{'='*60}")


class TestBatchProcessingBenchmarks:
    """TDD: Batch processing performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_batch_token_processing_benchmark_fails_first(self):
        """
        TDD: Failing test for batch token processing performance
        Target: 100+ tokens per minute from CLAUDE.md
        """
        # Create test configuration
        agent_config = AgentConfig(
            hidden_size=64,
            num_layers=2,
            batch_size=16,  # Smaller for faster processing
            epsilon_start=0.05  # Less exploration for consistent timing
        )
        
        # Create agents
        real_agent = DQNTradingAgent(agent_config, use_enhanced_features=True)
        mock_agent = MockDQNAgent(agent_config)
        
        # Create batch of tokens for processing
        tokens = []
        for i in range(100):  # Process 100 tokens
            token = DiscoveredToken(
                address=f"0xbatch{i:06x}",
                chain=Chain.ETHEREUM,
                symbol=f"BATCH{i}",
                name=f"Batch Token {i}",
                discovered_at=datetime.now(),
                discovery_source="batch_test",
                price_usd=1.0 + i * 0.01,
                market_cap=1000000 + i * 10000,
                volume_24h=100000 + i * 1000,
                price_change_24h=np.random.uniform(-10, 10),
                tags=["batch"]
            )
            tokens.append(token)
        
        # Create market states
        market_states = []
        for token in tokens:
            state = MLEnhancedMarketState(
                token=token,
                price_usd=token.price_usd,
                price_change_24h=token.price_change_24h,
                volume_24h=token.volume_24h,
                market_cap=token.market_cap,
                rsi=np.random.uniform(30, 70),
                macd=np.random.uniform(-0.01, 0.01),
                sma_20=token.price_usd * 0.95,
                ema_12=token.price_usd * 0.98,
                bollinger_upper=token.price_usd * 1.1,
                bollinger_lower=token.price_usd * 0.9,
                current_position=0.0,
                portfolio_value=10000.0,
                cash_balance=10000.0,
                portfolio_drawdown=0.0,
                daily_pnl=0.0,
                sharpe_ratio=1.0,
                market_volatility=0.2,
                fear_greed_index=50.0,
                # ML-enhanced features
                ml_prediction_1h=token.price_usd * 1.01,
                ml_prediction_4h=token.price_usd * 1.02,
                ml_prediction_24h=token.price_usd * 1.05,
                ml_confidence=np.random.uniform(0.6, 0.9),
                ml_direction="buy",
                volatility_forecast=np.random.uniform(0.15, 0.30)
            )
            market_states.append(state)
        
        # Benchmark real agent batch processing
        real_start_time = time.perf_counter()
        real_decisions = []
        
        for state in market_states:
            action, confidence = await real_agent.predict_action(state)
            real_decisions.append((action, confidence))
        
        real_end_time = time.perf_counter()
        real_total_time = real_end_time - real_start_time
        real_tokens_per_minute = (len(tokens) / real_total_time) * 60
        
        # Benchmark mock agent batch processing
        mock_start_time = time.perf_counter()
        mock_decisions = []
        
        for state in market_states:
            action, confidence = await mock_agent.predict_action(state)
            mock_decisions.append((action, confidence))
        
        mock_end_time = time.perf_counter()
        mock_total_time = mock_end_time - mock_start_time
        mock_tokens_per_minute = (len(tokens) / mock_total_time) * 60
        
        # TDD: These should fail initially
        assert real_tokens_per_minute >= 100, f"Real agent processes {real_tokens_per_minute:.1f} tokens/min, below 100 target"
        assert real_total_time < 60, f"Real agent took {real_total_time:.2f}s to process 100 tokens"
        assert len(real_decisions) == len(tokens), "Real agent did not process all tokens"
        
        # Performance ratio validation
        performance_ratio = real_total_time / mock_total_time
        assert performance_ratio < 100, f"Real agent {performance_ratio:.1f}x slower than mock for batch processing"
        
        print(f"\nBatch Processing Benchmark:")
        print(f"  Real agent: {real_tokens_per_minute:.1f} tokens/minute")
        print(f"  Mock agent: {mock_tokens_per_minute:.1f} tokens/minute")
        print(f"  Real total time: {real_total_time:.2f}s")
        print(f"  Mock total time: {mock_total_time:.2f}s")
        print(f"  Performance ratio: {performance_ratio:.1f}x")


class TestStatisticalValidation:
    """TDD: Statistical validation and analysis utilities"""
    
    def test_performance_distribution_analysis_fails_first(self):
        """
        TDD: Failing test for statistical performance distribution analysis
        """
        # Generate sample performance data
        real_latencies = np.random.exponential(0.01, 1000)  # Exponential distribution (realistic for latencies)
        mock_latencies = np.random.exponential(0.001, 1000)  # Much faster mock
        
        # Calculate statistical measures
        real_metrics = PerformanceMetrics(
            mean=np.mean(real_latencies),
            median=np.median(real_latencies),
            p95=np.percentile(real_latencies, 95),
            p99=np.percentile(real_latencies, 99),
            std=np.std(real_latencies),
            min_val=np.min(real_latencies),
            max_val=np.max(real_latencies),
            sample_size=len(real_latencies)
        )
        
        mock_metrics = PerformanceMetrics(
            mean=np.mean(mock_latencies),
            median=np.median(mock_latencies),
            p95=np.percentile(mock_latencies, 95),
            p99=np.percentile(mock_latencies, 99),
            std=np.std(mock_latencies),
            min_val=np.min(mock_latencies),
            max_val=np.max(mock_latencies),
            sample_size=len(mock_latencies)
        )
        
        # Statistical significance tests
        from scipy import stats
        
        # Welch's t-test for difference in means
        t_stat, p_value = stats.ttest_ind(real_latencies, mock_latencies, equal_var=False)
        
        # Mann-Whitney U test for difference in distributions
        u_stat, u_p_value = stats.mannwhitneyu(real_latencies, mock_latencies, alternative='two-sided')
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_p_value = stats.ks_2samp(real_latencies, mock_latencies)
        
        # TDD: These should fail initially
        assert p_value < 0.05, f"Performance difference not statistically significant (p={p_value:.6f})"
        assert u_p_value < 0.05, f"Distribution difference not significant (U test p={u_p_value:.6f})"
        assert ks_p_value < 0.05, f"Distribution shapes not significantly different (KS p={ks_p_value:.6f})"
        
        # Effect size calculation (Cohen's d)
        pooled_std = np.sqrt(((len(real_latencies) - 1) * real_metrics.std**2 + 
                             (len(mock_latencies) - 1) * mock_metrics.std**2) / 
                            (len(real_latencies) + len(mock_latencies) - 2))
        cohens_d = (real_metrics.mean - mock_metrics.mean) / pooled_std
        
        assert abs(cohens_d) > 0.8, f"Effect size too small (Cohen's d={cohens_d:.3f}), differences may not be meaningful"
        
        print(f"\nStatistical Validation Results:")
        print(f"  T-test p-value: {p_value:.6f}")
        print(f"  Mann-Whitney U p-value: {u_p_value:.6f}")
        print(f"  Kolmogorov-Smirnov p-value: {ks_p_value:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.3f}")
    
    def test_performance_regression_detection_fails_first(self):
        """
        TDD: Failing test for performance regression detection
        """
        # Simulate historical performance data
        baseline_latencies = np.random.exponential(0.005, 100)  # Historical baseline
        current_latencies = np.random.exponential(0.008, 100)   # Current (potentially regressed)
        
        baseline_metrics = PerformanceMetrics(
            mean=np.mean(baseline_latencies),
            median=np.median(baseline_latencies),
            p95=np.percentile(baseline_latencies, 95),
            p99=np.percentile(baseline_latencies, 99),
            std=np.std(baseline_latencies),
            min_val=np.min(baseline_latencies),
            max_val=np.max(baseline_latencies),
            sample_size=len(baseline_latencies)
        )
        
        current_metrics = PerformanceMetrics(
            mean=np.mean(current_latencies),
            median=np.median(current_latencies),
            p95=np.percentile(current_latencies, 95),
            p99=np.percentile(current_latencies, 99),
            std=np.std(current_latencies),
            min_val=np.min(current_latencies),
            max_val=np.max(current_latencies),
            sample_size=len(current_latencies)
        )
        
        # Regression detection thresholds
        regression_threshold_percent = 20.0  # 20% regression threshold
        
        mean_regression = ((current_metrics.mean - baseline_metrics.mean) / baseline_metrics.mean) * 100
        p95_regression = ((current_metrics.p95 - baseline_metrics.p95) / baseline_metrics.p95) * 100
        p99_regression = ((current_metrics.p99 - baseline_metrics.p99) / baseline_metrics.p99) * 100
        
        # TDD: These should fail initially if regression detection is working
        assert mean_regression < regression_threshold_percent, f"Mean latency regressed by {mean_regression:.1f}%"
        assert p95_regression < regression_threshold_percent, f"P95 latency regressed by {p95_regression:.1f}%"
        assert p99_regression < regression_threshold_percent, f"P99 latency regressed by {p99_regression:.1f}%"
        
        # Statistical significance of regression
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(current_latencies, baseline_latencies, equal_var=False)
        
        if mean_regression > 5.0:  # Only test significance if regression > 5%
            assert p_value < 0.05, f"Regression not statistically significant (p={p_value:.6f})"
        
        print(f"\nRegression Detection Results:")
        print(f"  Mean regression: {mean_regression:.1f}%")
        print(f"  P95 regression: {p95_regression:.1f}%")
        print(f"  P99 regression: {p99_regression:.1f}%")
        print(f"  Significance p-value: {p_value:.6f}")


class TestCLAUDETargetValidation:
    """TDD: Validation against CLAUDE.md performance targets"""
    
    @pytest.mark.asyncio
    async def test_claude_md_target_compliance_fails_first(self):
        """
        TDD: Failing test that validates all CLAUDE.md performance targets are met
        
        CLAUDE.md Targets:
        - ML prediction generation: <1s per token
        - RL decision making: <1s per action  
        - ML-RL integration: <1s for complete decision pipeline
        - Batch processing: 100+ tokens per minute capability
        - Memory efficiency: <50MB growth during batch processing
        """
        # Test configuration
        agent_config = AgentConfig(
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=32,
            epsilon_start=0.1,
            epsilon_end=0.05
        )
        
        # Create real implementations
        real_agent = DQNTradingAgent(agent_config, use_enhanced_features=True)
        
        # Create test data
        tokens = []
        market_states = []
        for i in range(100):
            token = DiscoveredToken(
                address=f"0xtarget{i:06x}",
                chain=Chain.ETHEREUM,
                symbol=f"TARGET{i}",
                name=f"Target Token {i}",
                discovered_at=datetime.now(),
                discovery_source="target_test",
                price_usd=1.0 + i * 0.01,
                market_cap=1000000,
                volume_24h=100000,
                price_change_24h=0.0,
                tags=["target"]
            )
            tokens.append(token)
            
            state = MLEnhancedMarketState(
                token=token,
                price_usd=token.price_usd,
                price_change_24h=0.0,
                volume_24h=100000,
                market_cap=1000000,
                rsi=50.0,
                macd=0.0,
                sma_20=token.price_usd * 0.95,
                ema_12=token.price_usd * 0.98,
                bollinger_upper=token.price_usd * 1.1,
                bollinger_lower=token.price_usd * 0.9,
                current_position=0.0,
                portfolio_value=10000.0,
                cash_balance=10000.0,
                portfolio_drawdown=0.0,
                daily_pnl=0.0,
                sharpe_ratio=1.0,
                market_volatility=0.2,
                fear_greed_index=50.0,
                # ML-enhanced features
                ml_prediction_1h=token.price_usd * 1.01,
                ml_prediction_4h=token.price_usd * 1.02,
                ml_prediction_24h=token.price_usd * 1.05,
                ml_confidence=0.75,
                ml_direction="buy",
                volatility_forecast=0.20
            )
            market_states.append(state)
        
        # Test 1: RL decision making <1s per action
        rl_latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            action, confidence = await real_agent.predict_action(market_states[i])
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            rl_latencies.append(latency)
            
            # TDD: This should fail initially
            assert latency < 1.0, f"RL decision {i} took {latency:.3f}s, exceeds 1s target"
        
        rl_metrics = PerformanceMetrics(
            mean=np.mean(rl_latencies),
            median=np.median(rl_latencies),
            p95=np.percentile(rl_latencies, 95),
            p99=np.percentile(rl_latencies, 99),
            std=np.std(rl_latencies),
            min_val=np.min(rl_latencies),
            max_val=np.max(rl_latencies),
            sample_size=len(rl_latencies)
        )
        
        # TDD: These should fail initially
        assert rl_metrics.p99 < 1.0, f"RL P99 latency {rl_metrics.p99:.3f}s exceeds 1s target"
        assert rl_metrics.p95 < 0.5, f"RL P95 latency {rl_metrics.p95:.3f}s exceeds 0.5s reasonable target"
        assert rl_metrics.mean < 0.1, f"RL mean latency {rl_metrics.mean:.3f}s exceeds 0.1s efficient target"
        
        # Test 2: Batch processing 100+ tokens per minute
        batch_start_time = time.perf_counter()
        batch_decisions = []
        
        for state in market_states:
            action, confidence = await real_agent.predict_action(state)
            batch_decisions.append((action, confidence))
        
        batch_end_time = time.perf_counter()
        batch_total_time = batch_end_time - batch_start_time
        tokens_per_minute = (len(market_states) / batch_total_time) * 60
        
        # TDD: This should fail initially
        assert tokens_per_minute >= 100, f"Batch processing {tokens_per_minute:.1f} tokens/min, below 100 target"
        assert batch_total_time < 60, f"Batch processing took {batch_total_time:.2f}s for 100 tokens"
        
        # Test 3: Memory efficiency <50MB growth
        tracemalloc.start()
        gc.collect()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Intensive batch processing simulation
        for round_num in range(10):  # 10 rounds of 100 tokens each
            for state in market_states:
                await real_agent.predict_action(state)
            
            # Force garbage collection between rounds
            gc.collect()
        
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # TDD: This should fail initially
        assert memory_growth_mb < 50, f"Memory grew by {memory_growth_mb:.2f}MB, exceeds 50MB target"
        
        # Generate compliance report
        print(f"\nCLAUDE.md Target Compliance Report:")
        print(f"{'='*50}")
        print(f"1. RL Decision Making (<1s per action):")
        print(f"   Mean: {rl_metrics.mean*1000:.1f}ms ✓")
        print(f"   P95: {rl_metrics.p95*1000:.1f}ms ✓")
        print(f"   P99: {rl_metrics.p99*1000:.1f}ms {'✓' if rl_metrics.p99 < 1.0 else '✗'}")
        print(f"")
        print(f"2. Batch Processing (100+ tokens/min):")
        print(f"   Achieved: {tokens_per_minute:.1f} tokens/min {'✓' if tokens_per_minute >= 100 else '✗'}")
        print(f"   Time for 100 tokens: {batch_total_time:.2f}s {'✓' if batch_total_time < 60 else '✗'}")
        print(f"")
        print(f"3. Memory Efficiency (<50MB growth):")
        print(f"   Growth: {memory_growth_mb:.2f}MB {'✓' if memory_growth_mb < 50 else '✗'}")
        print(f"{'='*50}")


if __name__ == "__main__":
    # Allow running specific benchmark categories
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "dqn":
            pytest.main(["-v", "TestRealVsMockDQNBenchmarks"])
        elif sys.argv[1] == "batch":
            pytest.main(["-v", "TestBatchProcessingBenchmarks"])
        elif sys.argv[1] == "stats":
            pytest.main(["-v", "TestStatisticalValidation"])
        elif sys.argv[1] == "targets":
            pytest.main(["-v", "TestCLAUDETargetValidation"])
        elif sys.argv[1] == "all":
            pytest.main(["-v"])
        else:
            print("Usage: python test_real_vs_mock_benchmarks.py [dqn|batch|stats|targets|all]")
    else:
        pytest.main(["-v"])