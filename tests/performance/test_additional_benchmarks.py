"""
Additional comprehensive performance benchmarks for ML-RL pipeline

These tests extend the core performance tests with additional scenarios
covering market conditions, token types, and scaling characteristics.
"""

import asyncio
import time
import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock

from src.discovery.base import DiscoveredToken, TokenStatus
from src.ml_analysis.base import PredictionResult, ModelType, PredictionDirection
from src.utils.base import Chain
from tests.performance.test_ml_rl_performance import PerformanceMLAnalyzer, PerformanceRLAgent
from src.rl_agent.base import AgentConfig


class TestMarketConditionPerformance:
    """Test performance under different market conditions"""
    
    @pytest.fixture
    def bull_market_tokens(self) -> List[DiscoveredToken]:
        """Generate tokens with bull market characteristics"""
        tokens = []
        for i in range(20):
            token = DiscoveredToken(
                address=f"bull_token_{i}",
                chain=Chain.SOLANA,
                symbol=f"BULL{i}",
                name=f"Bull Token {i}",
                discovered_at=datetime.now(),
                discovery_source="bull_market_test",
                status=TokenStatus.DISCOVERED,
                price_usd=np.random.uniform(1.0, 100.0),
                market_cap=np.random.uniform(1000000, 100000000),
                volume_24h=np.random.uniform(500000, 10000000),  # High volume
                price_change_24h=np.random.uniform(5, 50),  # Positive price changes
                decimals=18,
                total_supply=np.random.uniform(1000000, 100000000)
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def bear_market_tokens(self) -> List[DiscoveredToken]:
        """Generate tokens with bear market characteristics"""
        tokens = []
        for i in range(20):
            token = DiscoveredToken(
                address=f"bear_token_{i}",
                chain=Chain.ETHEREUM,
                symbol=f"BEAR{i}",
                name=f"Bear Token {i}",
                discovered_at=datetime.now(),
                discovery_source="bear_market_test",
                status=TokenStatus.DISCOVERED,
                price_usd=np.random.uniform(0.001, 1.0),
                market_cap=np.random.uniform(10000, 1000000),
                volume_24h=np.random.uniform(1000, 100000),  # Low volume
                price_change_24h=np.random.uniform(-50, -5),  # Negative price changes
                decimals=18,
                total_supply=np.random.uniform(1000000, 1000000000)
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def performance_ml_analyzer(self):
        """Create performance ML analyzer"""
        return PerformanceMLAnalyzer()
    
    @pytest.mark.performance
    def test_bull_market_performance(self, benchmark, performance_ml_analyzer, bull_market_tokens):
        """Test ML-RL performance in bull market conditions"""
        def analyze_bull_market():
            return asyncio.run(performance_ml_analyzer.batch_analyze(bull_market_tokens[:10]))
        
        results = benchmark(analyze_bull_market)
        
        # Validate results
        assert len(results) == 10, "Should analyze all bull market tokens"
        for result in results:
            assert result.confidence > 0.0, "Should have confidence in bull market"
            # Bull market should tend toward buy signals
            assert result.probability_up >= 0.4, "Bull market should have reasonable upside probability"
    
    @pytest.mark.performance
    def test_bear_market_performance(self, benchmark, performance_ml_analyzer, bear_market_tokens):
        """Test ML-RL performance in bear market conditions"""
        def analyze_bear_market():
            return asyncio.run(performance_ml_analyzer.batch_analyze(bear_market_tokens[:10]))
        
        results = benchmark(analyze_bear_market)
        
        # Validate results
        assert len(results) == 10, "Should analyze all bear market tokens"
        for result in results:
            assert result.confidence > 0.0, "Should have confidence even in bear market"
            # Bear market analysis should still provide valid predictions
            assert result.processing_time_ms is not None, "Should track processing time"
    
    @pytest.mark.performance
    def test_mixed_market_conditions_performance(self, benchmark, performance_ml_analyzer, 
                                               bull_market_tokens, bear_market_tokens):
        """Test performance with mixed market conditions"""
        mixed_tokens = bull_market_tokens[:5] + bear_market_tokens[:5]
        
        def analyze_mixed_market():
            return asyncio.run(performance_ml_analyzer.batch_analyze(mixed_tokens))
        
        results = benchmark(analyze_mixed_market)
        
        # Validate mixed market handling
        assert len(results) == 10, "Should handle all mixed market tokens"
        
        # Should have variety in predictions
        confidences = [r.confidence for r in results]
        assert len(set(confidences)) > 1, "Should have varied confidence levels"


class TestTokenTypePerformance:
    """Test performance across different token types and chains"""
    
    @pytest.fixture
    def multi_chain_tokens(self) -> Dict[Chain, List[DiscoveredToken]]:
        """Generate tokens across different chains"""
        tokens_by_chain = {}
        
        for chain in [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE]:
            chain_tokens = []
            for i in range(10):
                token = DiscoveredToken(
                    address=f"{chain.value}_token_{i}",
                    chain=chain,
                    symbol=f"{chain.value[:3].upper()}{i}",
                    name=f"{chain.value} Token {i}",
                    discovered_at=datetime.now(),
                    discovery_source=f"{chain.value}_test",
                    status=TokenStatus.DISCOVERED,
                    price_usd=np.random.uniform(0.01, 10.0),
                    market_cap=np.random.uniform(100000, 10000000),
                    volume_24h=np.random.uniform(10000, 1000000),
                    price_change_24h=np.random.uniform(-10, 10),
                    decimals=18,
                    total_supply=np.random.uniform(1000000, 1000000000)
                )
                chain_tokens.append(token)
            tokens_by_chain[chain] = chain_tokens
        
        return tokens_by_chain
    
    @pytest.fixture
    def performance_ml_analyzer(self):
        """Create performance ML analyzer"""
        return PerformanceMLAnalyzer()
    
    @pytest.fixture
    def sample_tokens(self) -> List[DiscoveredToken]:
        """Generate sample tokens for scaling tests"""
        tokens = []
        chains = [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE]
        
        for i in range(50):  # 50 tokens for scaling tests
            token = DiscoveredToken(
                address=f"scale_token_{i}",
                chain=chains[i % len(chains)],
                symbol=f"SCALE{i}",
                name=f"Scale Token {i}",
                discovered_at=datetime.now(),
                discovery_source="scaling_test",
                status=TokenStatus.DISCOVERED,
                price_usd=np.random.uniform(0.001, 10.0),
                market_cap=np.random.uniform(100000, 10000000),
                volume_24h=np.random.uniform(10000, 1000000),
                price_change_24h=np.random.uniform(-20, 20),
                decimals=18,
                total_supply=np.random.uniform(1000000, 1000000000)
            )
            tokens.append(token)
        
        return tokens
    
    @pytest.mark.performance
    def test_cross_chain_performance(self, benchmark, performance_ml_analyzer, multi_chain_tokens):
        """Test performance across different blockchain networks"""
        # Combine tokens from all chains
        all_tokens = []
        for chain_tokens in multi_chain_tokens.values():
            all_tokens.extend(chain_tokens[:3])  # 3 tokens per chain = 9 total
        
        def analyze_cross_chain():
            return asyncio.run(performance_ml_analyzer.batch_analyze(all_tokens))
        
        results = benchmark(analyze_cross_chain)
        
        # Validate cross-chain handling
        assert len(results) == 9, "Should handle all cross-chain tokens"
        
        # Group results by chain
        results_by_chain = {}
        for result in results:
            chain = result.token.chain
            if chain not in results_by_chain:
                results_by_chain[chain] = []
            results_by_chain[chain].append(result)
        
        # Should have results for all chains
        assert len(results_by_chain) == 3, "Should process tokens from all chains"
        
        # Each chain should have consistent performance
        for chain, chain_results in results_by_chain.items():
            avg_processing_time = np.mean([r.processing_time_ms for r in chain_results if r.processing_time_ms])
            assert avg_processing_time < 1000, f"Chain {chain} processing too slow: {avg_processing_time}ms"
    
    @pytest.mark.performance
    def test_token_size_scaling_performance(self, benchmark, performance_ml_analyzer, sample_tokens):
        """Test performance scaling with different token batch sizes"""
        batch_sizes = [1, 5, 10, 20]
        performance_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(sample_tokens):
                continue
                
            test_tokens = sample_tokens[:batch_size]
            
            def analyze_batch_size():
                return asyncio.run(performance_ml_analyzer.batch_analyze(test_tokens))
            
            # Use benchmark.pedantic for consistent measurement
            result = benchmark.pedantic(analyze_batch_size, rounds=3, iterations=1)
            
            performance_results[batch_size] = {
                'mean_time': benchmark.stats.mean,
                'tokens_per_second': batch_size / benchmark.stats.mean,
                'time_per_token': benchmark.stats.mean / batch_size
            }
        
        # Validate scaling characteristics
        for batch_size, metrics in performance_results.items():
            tokens_per_minute = metrics['tokens_per_second'] * 60
            assert tokens_per_minute >= 50, f"Batch size {batch_size}: {tokens_per_minute:.1f} tokens/min too slow"
            assert metrics['time_per_token'] < 1.0, f"Batch size {batch_size}: {metrics['time_per_token']:.3f}s per token too slow"


class TestRealWorldScenarioPerformance:
    """Test performance under realistic trading scenarios"""
    
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
    def high_volatility_tokens(self) -> List[DiscoveredToken]:
        """Generate high volatility tokens"""
        tokens = []
        for i in range(15):
            token = DiscoveredToken(
                address=f"volatile_token_{i}",
                chain=Chain.SOLANA,
                symbol=f"VOL{i}",
                name=f"Volatile Token {i}",
                discovered_at=datetime.now(),
                discovery_source="volatility_test",
                status=TokenStatus.DISCOVERED,
                price_usd=np.random.uniform(0.01, 1.0),
                market_cap=np.random.uniform(50000, 5000000),
                volume_24h=np.random.uniform(100000, 5000000),
                price_change_24h=np.random.uniform(-80, 100),  # High volatility
                decimals=18,
                total_supply=np.random.uniform(1000000, 1000000000)
            )
            tokens.append(token)
        return tokens
    
    @pytest.mark.performance
    def test_high_volatility_token_performance(self, benchmark, performance_ml_analyzer, high_volatility_tokens):
        """Test performance with high volatility tokens"""
        def analyze_volatile_tokens():
            return asyncio.run(performance_ml_analyzer.batch_analyze(high_volatility_tokens[:10]))
        
        results = benchmark(analyze_volatile_tokens)
        
        # Validate high volatility handling
        assert len(results) == 10, "Should handle all volatile tokens"
        
        # High volatility tokens should still get reasonable predictions
        for result in results:
            assert result.confidence > 0.0, "Should have confidence even for volatile tokens"
            assert result.volatility_forecast is not None, "Should forecast volatility"
            assert result.processing_time_ms < 1000, "Should process volatile tokens quickly"
    
    @pytest.mark.performance
    def test_rapid_fire_analysis_performance(self, benchmark, performance_ml_analyzer, high_volatility_tokens):
        """Test performance under rapid-fire analysis scenarios"""
        def rapid_fire_analysis():
            results = []
            # Simulate rapid successive analyses
            for i in range(5):
                batch_results = asyncio.run(performance_ml_analyzer.batch_analyze(high_volatility_tokens[:3]))
                results.extend(batch_results)
            return results
        
        results = benchmark(rapid_fire_analysis)
        
        # Should handle rapid successive calls efficiently
        assert len(results) == 15, "Should handle all rapid-fire analyses"
        
        # Performance should remain consistent
        processing_times = [r.processing_time_ms for r in results if r.processing_time_ms]
        if processing_times:
            avg_time = np.mean(processing_times)
            assert avg_time < 100, f"Average processing time {avg_time}ms too slow for rapid-fire"
    
    @pytest.mark.performance 
    def test_concurrent_ml_rl_decisions(self, benchmark, performance_ml_analyzer, performance_rl_agent, high_volatility_tokens):
        """Test concurrent ML analysis and RL decision making"""
        from src.integration.ml_rl_bridge import MLEnhancedMarketState
        
        def concurrent_ml_rl():
            # Get ML predictions
            ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(high_volatility_tokens[:5]))
            
            # Make RL decisions concurrently
            rl_decisions = []
            for prediction in ml_results:
                enhanced_state = MLEnhancedMarketState.from_prediction(
                    prediction=prediction,
                    current_portfolio_value=10000.0,
                    position_size=0.0
                )
                action, confidence = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
                rl_decisions.append((action, confidence))
            
            return ml_results, rl_decisions
        
        ml_results, rl_decisions = benchmark(concurrent_ml_rl)
        
        # Validate concurrent processing
        assert len(ml_results) == 5, "Should process all ML predictions"
        assert len(rl_decisions) == 5, "Should make all RL decisions"
        
        # Performance targets should still be met
        for result in ml_results:
            if result.processing_time_ms:
                assert result.processing_time_ms < 1000, "ML processing should remain fast during concurrent ops"


if __name__ == "__main__":
    """
    Run additional performance benchmarks.
    
    Usage:
    python -m pytest tests/performance/test_additional_benchmarks.py -v --benchmark-sort=mean
    python -m pytest tests/performance/test_additional_benchmarks.py::TestMarketConditionPerformance -v
    python -m pytest tests/performance/test_additional_benchmarks.py::TestTokenTypePerformance -v
    python -m pytest tests/performance/test_additional_benchmarks.py::TestRealWorldScenarioPerformance -v
    """
    pass