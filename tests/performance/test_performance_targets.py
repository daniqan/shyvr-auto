"""
Performance Target Validation Tests

These tests specifically validate the performance targets mentioned in CLAUDE.md:
- ML prediction generation: <1s per token
- RL decision making: <1s per action  
- ML-RL integration: <1s for complete decision pipeline
- Batch processing: 100+ tokens per minute capability
- Memory efficiency: No significant memory leaks during batch processing
"""

import asyncio
import gc
import time
import psutil
import pytest
import numpy as np
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

from src.discovery.base import DiscoveredToken, TokenStatus
from src.rl_agent.base import MarketState, TradeAction, AgentConfig
from src.integration.ml_rl_bridge import MLRLBridge, MLEnhancedMarketState
from src.utils.base import Chain
from tests.performance.test_ml_rl_performance import PerformanceMLAnalyzer, PerformanceRLAgent


class TestPerformanceTargetValidation:
    """Validate all performance targets from CLAUDE.md"""
    
    @pytest.fixture
    def sample_tokens(self) -> List[DiscoveredToken]:
        """Generate sample tokens for performance testing"""
        tokens = []
        chains = [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE]
        
        for i in range(120):  # More tokens for thorough batch testing
            token = DiscoveredToken(
                address=f"target_token_{i}",
                chain=chains[i % len(chains)],
                symbol=f"TARGET{i}",
                name=f"Target Token {i}",
                discovered_at=datetime.now(),
                discovery_source="target_validation_test",
                status=TokenStatus.DISCOVERED,
                price_usd=np.random.uniform(0.001, 100.0),
                market_cap=np.random.uniform(100000, 100000000),
                volume_24h=np.random.uniform(10000, 10000000),
                price_change_24h=np.random.uniform(-50, 50),
                decimals=18,
                total_supply=np.random.uniform(1000000, 1000000000)
            )
            tokens.append(token)
        
        return tokens
    
    @pytest.fixture
    def performance_ml_analyzer(self):
        """Create performance ML analyzer"""
        return PerformanceMLAnalyzer()
    
    @pytest.fixture
    def performance_rl_agent(self):
        """Create performance RL agent"""
        config = AgentConfig(
            hidden_size=128,
            num_layers=2,
            batch_size=16,
            learning_rate=1e-4
        )
        return PerformanceRLAgent(config)
    
    @pytest.fixture
    def ml_rl_bridge(self, performance_ml_analyzer, performance_rl_agent, sample_tokens):
        """Create ML-RL bridge for integration testing"""
        return MLRLBridge(
            ml_analyzer=performance_ml_analyzer,
            rl_agent=performance_rl_agent,
            tokens=sample_tokens[:20],  # Use subset for bridge tests
            cache_ttl_minutes=5
        )

    @pytest.mark.performance
    def test_ml_prediction_target_single_token(self, performance_ml_analyzer, sample_tokens):
        """Validate: ML prediction generation <1s per token"""
        single_token = sample_tokens[0]
        
        # Test single token prediction multiple times for consistency
        prediction_times = []
        for _ in range(10):
            start_time = time.time()
            result = asyncio.run(performance_ml_analyzer.analyze_token(single_token))
            end_time = time.time()
            
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            
            # Individual prediction should be <1s
            assert prediction_time < 1.0, f"ML prediction took {prediction_time:.3f}s, target <1.0s"
            assert result.processing_time_ms < 1000, f"Reported processing time {result.processing_time_ms}ms, target <1000ms"
        
        # Average should also be well under 1s
        avg_time = np.mean(prediction_times)
        assert avg_time < 0.5, f"Average ML prediction time {avg_time:.3f}s should be well under 1s target"
        
        print(f"ML Prediction Performance: {avg_time:.3f}s avg, {max(prediction_times):.3f}s max - TARGET MET")

    @pytest.mark.performance
    def test_rl_decision_target_single_action(self, performance_rl_agent, sample_tokens):
        """Validate: RL decision making <1s per action"""
        token = sample_tokens[0]
        market_state = MarketState(
            token=token,
            price_usd=token.price_usd,
            price_change_24h=token.price_change_24h or 0.0,
            volume_24h=token.volume_24h or 0.0,
            market_cap=token.market_cap,
            current_position=0.0,
            portfolio_value=10000.0
        )
        
        # Test RL decision making multiple times
        decision_times = []
        for _ in range(10):
            start_time = time.time()
            action, confidence = asyncio.run(performance_rl_agent.predict_action(market_state))
            end_time = time.time()
            
            decision_time = end_time - start_time
            decision_times.append(decision_time)
            
            # Individual decision should be <1s
            assert decision_time < 1.0, f"RL decision took {decision_time:.3f}s, target <1.0s"
            assert isinstance(action, TradeAction), "Should return valid trade action"
            assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        
        # Average should be well under 1s
        avg_time = np.mean(decision_times)
        assert avg_time < 0.1, f"Average RL decision time {avg_time:.3f}s should be well under 1s target"
        
        print(f"RL Decision Performance: {avg_time:.3f}s avg, {max(decision_times):.3f}s max - TARGET MET")

    @pytest.mark.performance
    def test_ml_rl_integration_target_complete_pipeline(self, ml_rl_bridge):
        """Validate: ML-RL integration <1s for complete decision pipeline"""
        portfolio_value = 10000.0
        positions = {}
        
        # Test complete integration pipeline multiple times
        integration_times = []
        for _ in range(5):
            start_time = time.time()
            results = ml_rl_bridge.predict_and_act(portfolio_value, positions)
            end_time = time.time()
            
            integration_time = end_time - start_time
            integration_times.append(integration_time)
            
            # Complete pipeline should be <1s
            assert integration_time < 1.0, f"ML-RL integration took {integration_time:.3f}s, target <1.0s"
            assert len(results) > 0, "Should generate integrated decisions"
            
            # Validate result structure
            for result in results:
                assert 'token' in result, "Result should contain token"
                assert 'ml_prediction' in result, "Result should contain ML prediction"
                assert 'rl_action' in result, "Result should contain RL action"
        
        # Average should be well under 1s
        avg_time = np.mean(integration_times)
        assert avg_time < 0.5, f"Average integration time {avg_time:.3f}s should be well under 1s target"
        
        print(f"ML-RL Integration Performance: {avg_time:.3f}s avg, {max(integration_times):.3f}s max - TARGET MET")

    @pytest.mark.performance
    def test_batch_processing_target_100_tokens_per_minute(self, performance_ml_analyzer, sample_tokens):
        """Validate: Batch processing 100+ tokens per minute capability"""
        # Test various batch sizes to find optimal performance
        batch_sizes = [10, 20, 30, 50]
        best_throughput = 0
        
        for batch_size in batch_sizes:
            if batch_size > len(sample_tokens):
                continue
            
            test_tokens = sample_tokens[:batch_size]
            
            # Measure batch processing time
            start_time = time.time()
            results = asyncio.run(performance_ml_analyzer.batch_analyze(test_tokens))
            end_time = time.time()
            
            processing_time = end_time - start_time
            tokens_per_second = batch_size / processing_time
            tokens_per_minute = tokens_per_second * 60
            
            best_throughput = max(best_throughput, tokens_per_minute)
            
            # Validate batch processing target
            assert len(results) == batch_size, f"Should process all {batch_size} tokens"
            
            print(f"Batch size {batch_size}: {tokens_per_minute:.1f} tokens/min")
        
        # Overall target: 100+ tokens per minute
        assert best_throughput >= 100, f"Best throughput {best_throughput:.1f} tokens/min, target >=100"
        
        print(f"Batch Processing Performance: {best_throughput:.1f} tokens/min - TARGET MET")

    @pytest.mark.performance  
    def test_memory_efficiency_target_no_significant_leaks(self, performance_ml_analyzer, 
                                                         performance_rl_agent, sample_tokens):
        """Validate: Memory efficiency - no significant memory leaks during batch processing"""
        process = psutil.Process()
        
        # Measure initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        # Process multiple batches to detect leaks
        batch_size = 15
        num_batches = 8
        
        for batch_num in range(num_batches):
            # Select tokens for this batch
            start_idx = (batch_num * batch_size) % len(sample_tokens)
            end_idx = min(start_idx + batch_size, len(sample_tokens))
            if start_idx >= len(sample_tokens):
                start_idx = 0
                end_idx = batch_size
            
            batch_tokens = sample_tokens[start_idx:end_idx]
            
            # Process through ML analysis
            ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(batch_tokens))
            
            # Process through RL decision making
            for prediction in ml_results:
                enhanced_state = MLEnhancedMarketState.from_prediction(
                    prediction=prediction,
                    current_portfolio_value=10000.0,
                    position_size=0.0
                )
                _ = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
            
            # Force garbage collection and measure memory
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"Batch {batch_num + 1}: {current_memory:.1f}MB")
        
        # Analyze memory growth pattern
        final_memory = memory_samples[-1]
        total_memory_increase = final_memory - initial_memory
        
        # Calculate memory trend (slope)
        if len(memory_samples) > 1:
            x = np.arange(len(memory_samples))
            memory_trend = np.polyfit(x, memory_samples, 1)[0]  # MB per batch
            
            # Memory trend should not be strongly positive (indicating leak)
            assert memory_trend < 5.0, f"Memory growing at {memory_trend:.2f}MB per batch, possible leak"
            
            # Total memory increase should be reasonable
            assert total_memory_increase < 100, f"Total memory increase {total_memory_increase:.1f}MB too high"
            
            print(f"Memory Efficiency: {memory_trend:.2f}MB/batch trend, {total_memory_increase:.1f}MB total - TARGET MET")

    @pytest.mark.performance
    def test_comprehensive_performance_validation(self, performance_ml_analyzer, performance_rl_agent, 
                                                 ml_rl_bridge, sample_tokens):
        """Comprehensive test combining all performance targets"""
        # Test data
        test_tokens = sample_tokens[:50]  # Reasonable subset for comprehensive test
        
        print("\\n=== COMPREHENSIVE PERFORMANCE VALIDATION ===")
        
        # 1. ML Prediction Speed Test
        print("\\n1. ML Prediction Speed Test:")
        ml_start = time.time()
        ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(test_tokens[:10]))
        ml_time = time.time() - ml_start
        ml_per_token = ml_time / 10
        
        assert ml_per_token < 1.0, f"ML prediction {ml_per_token:.3f}s per token exceeds 1s target"
        print(f"   ✓ ML Analysis: {ml_per_token:.3f}s per token (target <1.0s)")
        
        # 2. RL Decision Speed Test  
        print("\\n2. RL Decision Speed Test:")
        token = test_tokens[0]
        market_state = MarketState(
            token=token,
            price_usd=token.price_usd,
            volume_24h=token.volume_24h or 0.0,
            current_position=0.0,
            portfolio_value=10000.0
        )
        
        rl_start = time.time()
        action, confidence = asyncio.run(performance_rl_agent.predict_action(market_state))
        rl_time = time.time() - rl_start
        
        assert rl_time < 1.0, f"RL decision {rl_time:.3f}s exceeds 1s target"
        print(f"   ✓ RL Decision: {rl_time:.3f}s per action (target <1.0s)")
        
        # 3. Integration Pipeline Test
        print("\\n3. ML-RL Integration Test:")
        integration_start = time.time()
        integration_results = ml_rl_bridge.predict_and_act(10000.0, {})
        integration_time = time.time() - integration_start
        
        assert integration_time < 1.0, f"Integration {integration_time:.3f}s exceeds 1s target"
        print(f"   ✓ Integration: {integration_time:.3f}s per pipeline (target <1.0s)")
        
        # 4. Throughput Test
        print("\\n4. Batch Processing Throughput Test:")
        throughput_start = time.time()
        throughput_results = asyncio.run(performance_ml_analyzer.batch_analyze(test_tokens[:25]))
        throughput_time = time.time() - throughput_start
        tokens_per_minute = (25 / throughput_time) * 60
        
        assert tokens_per_minute >= 100, f"Throughput {tokens_per_minute:.1f} tokens/min below 100 target"
        print(f"   ✓ Throughput: {tokens_per_minute:.1f} tokens/min (target >=100)")
        
        # 5. Memory Efficiency Check
        print("\\n5. Memory Efficiency Test:")
        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Process multiple batches
        for i in range(3):
            batch = test_tokens[i*5:(i+1)*5]
            _ = asyncio.run(performance_ml_analyzer.batch_analyze(batch))
            gc.collect()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before
        
        assert mem_increase < 50, f"Memory increase {mem_increase:.1f}MB too high"
        print(f"   ✓ Memory: {mem_increase:.1f}MB increase (target <50MB)")
        
        print("\\n=== ALL PERFORMANCE TARGETS VALIDATED ✓ ===")


if __name__ == "__main__":
    """
    Run performance target validation tests.
    
    Usage:
    python -m pytest tests/performance/test_performance_targets.py -v -s
    python -m pytest tests/performance/test_performance_targets.py::TestPerformanceTargetValidation::test_comprehensive_performance_validation -v -s
    """
    pass