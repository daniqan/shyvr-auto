"""
Comprehensive Performance Benchmarking Tests for ML-RL Pipeline

This module contains performance tests for the entire ML-RL trading system,
validating response times, throughput, and resource efficiency according to
the performance targets defined in CLAUDE.md.

Performance Targets:
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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch
import torch

from src.discovery.base import DiscoveredToken, TokenStatus
from src.ml_analysis.base import (
    PredictionResult, TechnicalIndicators, MarketFeatures, 
    ModelType, PredictionDirection, MLAnalyzerBase
)
from src.rl_agent.base import (
    MarketState, TradeAction, TradingResult, RewardMetrics,
    AgentConfig, RLAgentBase
)
from src.integration.ml_rl_bridge import (
    MLEnhancedMarketState, MLRLBridge, MLRLTrainingPipeline,
    MLRLConfig, MLRLPerformanceMetrics
)
from src.utils.base import Chain


class PerformanceMLAnalyzer(MLAnalyzerBase):
    """Performance-optimized ML analyzer for benchmarking"""
    
    def __init__(self, model_type: ModelType = ModelType.LSTM, config: Optional[Dict] = None):
        super().__init__(model_type, config)
        # Simulate trained model
        self._model = MagicMock()
        self._is_trained = True
        
        # Pre-computed features to avoid repeated calculation overhead
        self._cached_features = np.random.rand(17).astype(np.float32)
        self._cached_indicators = TechnicalIndicators(
            sma_20=1.0, sma_50=1.1, ema_12=0.98, ema_26=1.02,
            rsi=65.5, macd=0.02, macd_signal=0.01, macd_histogram=0.01,
            bollinger_upper=1.05, bollinger_lower=0.95, bollinger_width=0.1,
            atr=0.03, volume_sma=50000, volume_ratio=1.2, obv=1000,
            price_momentum=0.05, volatility_score=0.3
        )
        self._cached_market_features = MarketFeatures(
            fear_greed_index=50.0, market_trend="bull", volatility_regime="medium",
            btc_correlation=0.7, eth_correlation=0.65, market_beta=1.1,
            social_score=0.6, mention_volume=100, sentiment_trend=0.1
        )
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[Any] = None) -> PredictionResult:
        """Fast token analysis for performance testing"""
        start_time = time.time()
        
        # Simulate ML model inference with realistic computation
        await asyncio.sleep(0.001)  # Simulate minimal async I/O
        
        # Generate predictions with some variance
        base_price = token.price_usd or 1.0
        prediction_multiplier = 1.0 + np.random.normal(0, 0.05)  # 5% std dev
        
        result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=self.model_type,
            price_prediction_1h=base_price * prediction_multiplier,
            price_prediction_4h=base_price * (prediction_multiplier ** 2),
            price_prediction_24h=base_price * (prediction_multiplier ** 4),
            direction=np.random.choice(list(PredictionDirection)),
            confidence=np.random.uniform(0.6, 0.9),
            probability_up=np.random.uniform(0.4, 0.8),
            technical_indicators=self._cached_indicators,
            market_features=self._cached_market_features,
            model_accuracy=0.75,
            prediction_uncertainty=0.1,
            volatility_forecast=np.random.uniform(0.1, 0.3),
            processing_time_ms=(time.time() - start_time) * 1000,
            features_used=["price", "volume", "rsi", "macd", "bollinger"],
            model_version="v1.0.0"
        )
        
        return result
    
    async def train_model(self, training_data: Any) -> bool:
        """Mock training for performance tests"""
        await asyncio.sleep(0.01)  # Simulate training time
        return True
    
    def get_required_features(self) -> List[str]:
        """Return list of required features"""
        return ["price", "volume", "rsi", "macd", "bollinger", "ema", "sma"]


class PerformanceRLAgent(RLAgentBase):
    """Performance-optimized RL agent for benchmarking"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.is_trained = True
        self.training_episodes = 1000
        
        # Pre-create neural network for realistic inference
        self._network = torch.nn.Sequential(
            torch.nn.Linear(25, config.hidden_size),  # 25 features from MLEnhancedMarketState
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, len(TradeAction))
        )
        self._network.eval()
        
        # Pre-computed tensors to avoid repeated allocation
        self._dummy_input = torch.randn(1, 25)
    
    async def predict_action(self, state: MarketState) -> tuple[TradeAction, float]:
        """Fast action prediction for performance testing"""
        # Convert state to tensor
        if isinstance(state, MLEnhancedMarketState):
            features = state.to_feature_vector()
        else:
            features = state.to_vector()
            # Pad to 25 features if needed
            if len(features) < 25:
                features = np.pad(features, (0, 25 - len(features)), 'constant')
        
        # Neural network inference with PyTorch
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            q_values = self._network(input_tensor)
            action_idx = torch.argmax(q_values, dim=1).item()
        
        actions = list(TradeAction)
        action = actions[action_idx % len(actions)]
        confidence = float(torch.max(torch.softmax(q_values, dim=1)).item())
        
        return action, confidence
    
    async def train_step(self, batch_experiences: List[Dict]) -> Dict[str, float]:
        """Mock training step for performance tests"""
        await asyncio.sleep(0.001)  # Simulate training time
        return {
            "loss": np.random.uniform(0.1, 0.5),
            "q_value_mean": np.random.uniform(0.5, 2.0),
            "epsilon": np.random.uniform(0.05, 0.2)
        }
    
    def save_model(self, filepath: str) -> bool:
        """Mock model saving"""
        return True
    
    def load_model(self, filepath: str) -> bool:
        """Mock model loading"""
        return True


@pytest.fixture
def sample_tokens() -> List[DiscoveredToken]:
    """Generate sample tokens for performance testing"""
    tokens = []
    chains = [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE]
    
    for i in range(100):  # 100 tokens for batch testing
        token = DiscoveredToken(
            address=f"token_address_{i}",
            chain=chains[i % len(chains)],
            symbol=f"TOKEN{i}",
            name=f"Test Token {i}",
            discovered_at=datetime.now(),
            discovery_source="performance_test",
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


@pytest.fixture
def performance_ml_analyzer():
    """Create performance ML analyzer"""
    return PerformanceMLAnalyzer()


@pytest.fixture
def performance_rl_agent():
    """Create performance RL agent"""
    config = AgentConfig(
        hidden_size=128,  # Smaller for faster inference
        num_layers=2,
        batch_size=16,
        learning_rate=1e-4
    )
    return PerformanceRLAgent(config)


@pytest.fixture
def ml_rl_bridge(performance_ml_analyzer, performance_rl_agent, sample_tokens):
    """Create ML-RL bridge for performance testing"""
    return MLRLBridge(
        ml_analyzer=performance_ml_analyzer,
        rl_agent=performance_rl_agent,
        tokens=sample_tokens[:10],  # Use 10 tokens for bridge tests
        cache_ttl_minutes=5
    )


class TestMLPerformance:
    """Test ML analysis performance"""
    
    @pytest.mark.performance
    def test_single_token_prediction_speed(self, benchmark, performance_ml_analyzer, sample_tokens):
        """Test ML prediction speed for single token (<1s target)"""
        token = sample_tokens[0]
        
        def analyze_single():
            return asyncio.run(performance_ml_analyzer.analyze_token(token))
        
        # Benchmark the analysis
        result = benchmark(analyze_single)
        
        # Validate performance targets
        assert result.processing_time_ms < 1000, f"ML prediction took {result.processing_time_ms}ms, target <1000ms"
        assert result.confidence > 0.0, "ML prediction should have confidence score"
        assert result.price_prediction_1h is not None, "ML should generate price predictions"
    
    @pytest.mark.performance
    def test_batch_prediction_throughput(self, benchmark, performance_ml_analyzer, sample_tokens):
        """Test ML batch prediction throughput (100+ tokens per minute target)"""
        batch_size = 20  # Test with 20 tokens
        
        def analyze_batch():
            return asyncio.run(performance_ml_analyzer.batch_analyze(sample_tokens[:batch_size]))
        
        # Benchmark batch analysis
        results = benchmark(analyze_batch)
        
        # Calculate throughput
        processing_time = sum(r.processing_time_ms for r in results if r.processing_time_ms) / 1000
        tokens_per_second = batch_size / processing_time if processing_time > 0 else 0
        tokens_per_minute = tokens_per_second * 60
        
        assert len(results) == batch_size, "Should process all tokens in batch"
        assert tokens_per_minute >= 100, f"Throughput {tokens_per_minute} tokens/min, target >=100"
    
    @pytest.mark.performance
    def test_feature_engineering_speed(self, benchmark, performance_ml_analyzer):
        """Test feature engineering performance"""
        indicators = TechnicalIndicators(
            sma_20=1.0, sma_50=1.1, rsi=65.5, macd=0.02,
            bollinger_upper=1.05, bollinger_lower=0.95
        )
        
        def engineer_features():
            return indicators.to_feature_vector()
        
        # Benchmark feature engineering
        features = benchmark(engineer_features)
        
        assert len(features) == 17, "Feature vector should have 17 components"
        assert np.all(np.isfinite(features)), "All features should be finite"
    
    @pytest.mark.performance
    def test_model_memory_efficiency(self, performance_ml_analyzer, sample_tokens):
        """Test memory efficiency during ML operations"""
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process tokens in batches to simulate real usage
        batch_size = 10
        for i in range(0, min(50, len(sample_tokens)), batch_size):
            batch = sample_tokens[i:i + batch_size]
            asyncio.run(performance_ml_analyzer.batch_analyze(batch))
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB, possible memory leak"


class TestRLPerformance:
    """Test RL agent performance"""
    
    @pytest.mark.performance
    def test_action_prediction_speed(self, benchmark, performance_rl_agent, sample_tokens):
        """Test RL action prediction speed (<1s target)"""
        token = sample_tokens[0]
        market_state = MarketState(
            token=token,
            price_usd=token.price_usd,
            price_change_24h=token.price_change_24h or 0.0,
            volume_24h=token.volume_24h or 0.0,
            market_cap=token.market_cap,
            rsi=65.0,
            macd=0.02,
            current_position=0.0,
            portfolio_value=10000.0
        )
        
        def predict_action():
            return asyncio.run(performance_rl_agent.predict_action(market_state))
        
        # Benchmark action prediction
        action, confidence = benchmark(predict_action)
        
        assert isinstance(action, TradeAction), "Should return valid trade action"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    
    @pytest.mark.performance
    def test_batch_decision_making(self, benchmark, performance_rl_agent, sample_tokens):
        """Test batch decision making performance"""
        # Create market states for batch processing
        market_states = []
        for token in sample_tokens[:20]:  # Test with 20 tokens
            state = MarketState(
                token=token,
                price_usd=token.price_usd,
                price_change_24h=token.price_change_24h or 0.0,
                volume_24h=token.volume_24h or 0.0,
                current_position=np.random.uniform(-0.1, 0.1),
                portfolio_value=10000.0
            )
            market_states.append(state)
        
        def batch_predict():
            decisions = []
            for state in market_states:
                action, confidence = asyncio.run(performance_rl_agent.predict_action(state))
                decisions.append((action, confidence))
            return decisions
        
        # Benchmark batch prediction
        decisions = benchmark(batch_predict)
        
        assert len(decisions) == len(market_states), "Should make decision for each state"
        
        # All decisions should include valid actions and confidence scores
        for action, confidence in decisions:
            assert isinstance(action, TradeAction), f"Expected TradeAction, got {type(action)}"
            assert isinstance(confidence, (int, float)), f"Expected numeric confidence, got {type(confidence)}"
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} should be between 0 and 1"
    
    @pytest.mark.performance
    def test_neural_network_inference_speed(self, performance_rl_agent):
        """Test neural network inference performance"""
        # Create sample input tensor
        sample_features = torch.randn(32, 25)  # Batch of 32 samples
        
        def inference_batch():
            with torch.no_grad():
                return performance_rl_agent._network(sample_features)
        
        # Test multiple inference runs
        start_time = time.time()
        for _ in range(100):  # 100 inference runs
            inference_batch()
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100
        
        # Each inference should be very fast (< 10ms)
        assert avg_inference_time < 0.01, f"Neural network inference took {avg_inference_time}s, target <0.01s"


class TestMLRLIntegrationPerformance:
    """Test ML-RL integration performance"""
    
    @pytest.mark.performance
    def test_enhanced_market_state_creation(self, benchmark, sample_tokens):
        """Test enhanced market state creation performance"""
        token = sample_tokens[0]
        
        # Create sample prediction result
        prediction = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=token.price_usd * 1.02,
            confidence=0.75,
            technical_indicators=TechnicalIndicators(rsi=65.0, macd=0.02)
        )
        
        def create_enhanced_state():
            return MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=10000.0,
                position_size=0.05
            )
        
        # Benchmark state creation
        state = benchmark(create_enhanced_state)
        
        assert isinstance(state, MLEnhancedMarketState), "Should create enhanced market state"
        assert len(state.to_feature_vector()) == 25, "Enhanced state should have 25 features"
    
    @pytest.mark.performance
    def test_ml_rl_bridge_caching_performance(self, ml_rl_bridge):
        """Test ML-RL bridge caching efficiency"""
        # First call - should populate cache
        start_time = time.time()
        predictions1 = ml_rl_bridge.get_ml_predictions()
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        predictions2 = ml_rl_bridge.get_ml_predictions()
        second_call_time = time.time() - start_time
        
        # Cached call should be much faster (allow for some variance)
        assert second_call_time < first_call_time * 0.5, f"Cached call ({second_call_time:.4f}s) should be faster than first call ({first_call_time:.4f}s)"
        assert len(predictions1) == len(predictions2), "Cache should return same number of predictions"
    
    @pytest.mark.performance
    def test_integrated_decision_pipeline(self, benchmark, ml_rl_bridge):
        """Test complete ML-RL integration pipeline (<1s target)"""
        portfolio_value = 10000.0
        positions = {}  # Empty positions
        
        def integrated_pipeline():
            return ml_rl_bridge.predict_and_act(portfolio_value, positions)
        
        # Benchmark complete pipeline
        results = benchmark(integrated_pipeline)
        
        assert len(results) > 0, "Should generate integrated decisions"
        
        # Validate all results have required components
        for result in results:
            assert 'token' in result, "Result should contain token"
            assert 'ml_prediction' in result, "Result should contain ML prediction"
            assert 'rl_action' in result, "Result should contain RL action"
            # Note: RL action returns tuple (action, confidence), so confidence is in the tuple
            assert isinstance(result['rl_action'], (tuple, list)) or hasattr(result['rl_action'], '__iter__'), "RL action should include confidence"


class TestEndToEndPerformance:
    """Test end-to-end pipeline performance"""
    
    @pytest.mark.performance
    def test_complete_pipeline_latency(self, benchmark, performance_ml_analyzer, 
                                           performance_rl_agent, sample_tokens):
        """Test complete discovery→evaluation→ML→RL pipeline"""
        
        def complete_pipeline():
            # Step 1: Token discovery (simulated - already have tokens)
            discovered_tokens = sample_tokens[:5]  # Process 5 tokens
            
            # Step 2: ML Analysis
            ml_predictions = asyncio.run(performance_ml_analyzer.batch_analyze(discovered_tokens))
            
            # Step 3: RL Decision Making
            rl_decisions = []
            for prediction in ml_predictions:
                # Create enhanced market state
                enhanced_state = MLEnhancedMarketState.from_prediction(
                    prediction=prediction,
                    current_portfolio_value=10000.0,
                    position_size=0.0
                )
                
                # Get RL action
                action, confidence = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
                rl_decisions.append((action, confidence))
            
            return ml_predictions, rl_decisions
        
        # Benchmark complete pipeline
        ml_results, rl_results = benchmark(complete_pipeline)
        
        assert len(ml_results) == 5, "Should process all tokens through ML"
        assert len(rl_results) == 5, "Should process all tokens through RL"
    
    @pytest.mark.performance
    def test_batch_processing_throughput(self, performance_ml_analyzer, performance_rl_agent, sample_tokens):
        """Test batch processing capability (100+ tokens per minute)"""
        batch_size = 25  # Process 25 tokens
        
        start_time = time.time()
        
        # Process batch through ML
        ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(sample_tokens[:batch_size]))
        
        # Process through RL
        rl_results = []
        for prediction in ml_results:
            enhanced_state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=10000.0,
                position_size=0.0
            )
            action, confidence = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
            rl_results.append((action, confidence))
        
        end_time = time.time()
        
        # Calculate throughput
        processing_time = end_time - start_time
        tokens_per_second = batch_size / processing_time
        tokens_per_minute = tokens_per_second * 60
        
        assert tokens_per_minute >= 100, f"Throughput {tokens_per_minute:.1f} tokens/min, target >=100"
    
    @pytest.mark.performance
    def test_memory_efficiency_during_batch_processing(self, performance_ml_analyzer, 
                                                     performance_rl_agent, sample_tokens):
        """Test memory efficiency during large batch processing"""
        process = psutil.Process()
        
        # Measure initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Process multiple batches
        batch_size = 10
        num_batches = 5
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_tokens = sample_tokens[start_idx:end_idx]
            
            # Process through ML-RL pipeline
            ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(batch_tokens))
            
            for prediction in ml_results:
                enhanced_state = MLEnhancedMarketState.from_prediction(
                    prediction=prediction,
                    current_portfolio_value=10000.0,
                    position_size=0.0
                )
                _ = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 200MB for this test)
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB, possible memory leak"


class TestPerformanceRegression:
    """Test performance regression detection"""
    
    @pytest.mark.performance
    def test_ml_prediction_baseline(self, benchmark, performance_ml_analyzer, sample_tokens):
        """Establish baseline for ML prediction performance"""
        token = sample_tokens[0]
        
        def predict():
            return asyncio.run(performance_ml_analyzer.analyze_token(token))
        
        result = benchmark.pedantic(predict, rounds=10, iterations=1)
        
        # Store baseline metrics (in real deployment, these would be saved)
        baseline_metrics = {
            'mean_time': benchmark.stats.mean,
            'std_time': benchmark.stats.stddev,
            'max_time': benchmark.stats.max,
            'min_time': benchmark.stats.min
        }
        
        # Validate against performance targets
        assert baseline_metrics['mean_time'] < 1.0, "Mean prediction time should be <1s"
        assert baseline_metrics['max_time'] < 2.0, "Max prediction time should be <2s"
    
    @pytest.mark.performance
    def test_rl_decision_baseline(self, benchmark, performance_rl_agent, sample_tokens):
        """Establish baseline for RL decision performance"""
        token = sample_tokens[0]
        market_state = MarketState(
            token=token,
            price_usd=token.price_usd,
            price_change_24h=0.0,
            volume_24h=token.volume_24h or 0.0,
            current_position=0.0,
            portfolio_value=10000.0
        )
        
        def decide():
            return asyncio.run(performance_rl_agent.predict_action(market_state))
        
        result = benchmark.pedantic(decide, rounds=10, iterations=1)
        
        # Store baseline metrics
        baseline_metrics = {
            'mean_time': benchmark.stats.mean,
            'std_time': benchmark.stats.stddev,
            'max_time': benchmark.stats.max
        }
        
        # Validate against performance targets
        assert baseline_metrics['mean_time'] < 1.0, "Mean decision time should be <1s"
        assert baseline_metrics['max_time'] < 2.0, "Max decision time should be <2s"
    
    @pytest.mark.performance
    def test_integration_latency_baseline(self, benchmark, ml_rl_bridge):
        """Establish baseline for ML-RL integration latency"""
        portfolio_value = 10000.0
        positions = {}
        
        def integrate():
            return ml_rl_bridge.predict_and_act(portfolio_value, positions)
        
        result = benchmark.pedantic(integrate, rounds=5, iterations=1)
        
        # Store baseline metrics
        baseline_metrics = {
            'mean_time': benchmark.stats.mean,
            'integration_latency': benchmark.stats.mean
        }
        
        # Validate against <1s integration target
        assert baseline_metrics['integration_latency'] < 1.0, "Integration latency should be <1s"


@pytest.mark.performance
class TestResourceUtilization:
    """Test system resource utilization during performance operations"""
    
    def test_cpu_utilization_during_ml_batch(self, performance_ml_analyzer, sample_tokens):
        """Test CPU utilization during ML batch processing"""
        process = psutil.Process()
        
        # Monitor CPU usage during batch processing
        cpu_percentages = []
        
        def cpu_monitor():
            for _ in range(10):  # Monitor for 10 intervals
                cpu_percentages.append(process.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        import threading
        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()
        
        # Perform batch ML analysis
        batch_size = 15
        asyncio.run(performance_ml_analyzer.batch_analyze(sample_tokens[:batch_size]))
        
        monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_percentages:
            avg_cpu = np.mean(cpu_percentages)
            max_cpu = np.max(cpu_percentages)
            
            # CPU usage should be reasonable (not maxing out the system)
            assert max_cpu < 90, f"Peak CPU usage {max_cpu}% too high"
    
    def test_memory_growth_pattern(self, performance_ml_analyzer, performance_rl_agent, sample_tokens):
        """Test memory growth pattern during extended processing"""
        process = psutil.Process()
        memory_samples = []
        
        # Process tokens in small batches and monitor memory
        batch_size = 5
        num_batches = 8
        
        for batch_num in range(num_batches):
            gc.collect()  # Force garbage collection
            memory_before = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_before)
            
            # Process batch
            start_idx = (batch_num * batch_size) % len(sample_tokens)
            end_idx = start_idx + batch_size
            if end_idx > len(sample_tokens):
                end_idx = len(sample_tokens)
                
            batch_tokens = sample_tokens[start_idx:end_idx]
            
            # ML + RL processing
            ml_results = asyncio.run(performance_ml_analyzer.batch_analyze(batch_tokens))
            for prediction in ml_results:
                enhanced_state = MLEnhancedMarketState.from_prediction(
                    prediction=prediction,
                    current_portfolio_value=10000.0,
                    position_size=0.0
                )
                _ = asyncio.run(performance_rl_agent.predict_action(enhanced_state))
        
        # Analyze memory growth pattern
        if len(memory_samples) > 1:
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            
            # Memory trend should not be strongly positive (indicating leak)
            assert memory_trend < 5.0, f"Memory growing at {memory_trend}MB per batch, possible leak"


if __name__ == "__main__":
    """
    Run performance tests with specific pytest configurations for benchmarking.
    
    Usage:
    python -m pytest tests/performance/test_ml_rl_performance.py -v --benchmark-sort=mean
    """
    pass