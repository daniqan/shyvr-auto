"""
High-Frequency Trading Performance Scenarios - Phase 7.2
Tests specific to high-frequency trading performance requirements.

Validates:
- Sustained high-volume trading (1000+ trades/minute)
- Market volatility performance resilience 
- Concurrent multi-asset trading efficiency
- Emergency stop performance under load
- Order execution latency targets (<100ms)
- ML/RL prediction throughput (>1000/sec)

Following TDD methodology - tests written first, then implementations.
"""
import pytest
import asyncio
import time
import threading
import statistics
import concurrent.futures
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

from tests.conftest_integration import (
    mock_environment_variables,
    integration_test_helper,
    performance_tracker,
    load_test_config
)


@dataclass
class TradingSignal:
    """Trading signal for HFT testing."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: Decimal
    confidence: float
    timestamp: datetime
    market_conditions: Dict[str, Any]


@dataclass
class OrderExecutionResult:
    """Order execution result for performance tracking."""
    signal: TradingSignal
    execution_latency_ms: float
    ml_prediction_latency_ms: float
    rl_decision_latency_ms: float
    safety_validation_latency_ms: float
    total_latency_ms: float
    success: bool
    error_msg: Optional[str] = None


@dataclass
class MarketVolatilityScenario:
    """Market volatility scenario for testing."""
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    price_change_rate: float  # % change per minute
    volume_spike_factor: float  # volume multiplier
    duration_minutes: int


class HFTPerformanceSimulator:
    """High-frequency trading performance simulator."""
    
    def __init__(self):
        self.trading_results: List[OrderExecutionResult] = []
        self.market_data_buffer: List[Dict[str, Any]] = []
        self.emergency_stop_triggered = False
        
    def generate_trading_signals(self, count: int, time_window_minutes: int) -> List[TradingSignal]:
        """Generate realistic trading signals for testing."""
        # This will initially fail - implementation needed
        pass
    
    def simulate_order_execution(self, signal: TradingSignal) -> OrderExecutionResult:
        """Simulate complete order execution pipeline."""
        # This will initially fail - implementation needed
        pass
    
    def simulate_ml_prediction(self, signal: TradingSignal) -> Tuple[Dict[str, Any], float]:
        """Simulate ML prediction with timing."""
        # This will initially fail - implementation needed
        pass
    
    def simulate_rl_decision(self, signal: TradingSignal, ml_prediction: Dict[str, Any]) -> Tuple[str, float, float]:
        """Simulate RL decision making with timing."""
        # This will initially fail - implementation needed
        pass
    
    def simulate_safety_validation(self, signal: TradingSignal) -> Tuple[bool, float]:
        """Simulate safety validation with timing."""
        # This will initially fail - implementation needed
        pass
    
    def simulate_market_volatility(self, scenario: MarketVolatilityScenario) -> List[Dict[str, Any]]:
        """Simulate market volatility conditions."""
        # This will initially fail - implementation needed
        pass
    
    def trigger_emergency_stop(self) -> float:
        """Trigger emergency stop and measure response time."""
        # This will initially fail - implementation needed
        pass


class ConcurrentTradingManager:
    """Manages concurrent trading across multiple assets."""
    
    def __init__(self, max_concurrent_assets: int = 50):
        self.max_concurrent_assets = max_concurrent_assets
        self.active_trading_sessions: Dict[str, bool] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
        
    def start_concurrent_trading(self, assets: List[str], duration_minutes: int) -> List[Dict[str, Any]]:
        """Start concurrent trading across multiple assets."""
        # This will initially fail - implementation needed
        pass
    
    def execute_asset_trading(self, asset: str, duration_minutes: int) -> Dict[str, Any]:
        """Execute trading for a specific asset."""
        # This will initially fail - implementation needed
        pass
    
    def monitor_resource_usage(self) -> Dict[str, Any]:
        """Monitor system resource usage during concurrent trading."""
        # This will initially fail - implementation needed
        pass


class TestSustainedHighVolumeTrading:
    """Test sustained high-volume trading performance."""
    
    @pytest.fixture
    def hft_simulator(self):
        """Provide HFT performance simulator."""
        return HFTPerformanceSimulator()
    
    def test_1000_trades_per_minute_sustained_performance(self, mock_environment_variables, hft_simulator, performance_tracker):
        """Test sustained 1000 trades per minute for 5 minutes."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== SUSTAINED HIGH-VOLUME TRADING TEST ===")
            
            target_trades_per_minute = 1000
            test_duration_minutes = 5
            total_target_trades = target_trades_per_minute * test_duration_minutes
            
            performance_tracker.start_timing('sustained_hft_1000_tpm')
            
            # Generate trading signals for sustained trading
            # This will initially fail - implementation needed
            signals = hft_simulator.generate_trading_signals(
                count=total_target_trades,
                time_window_minutes=test_duration_minutes
            )
            
            assert len(signals) == total_target_trades, f"Expected {total_target_trades} signals, got {len(signals)}"
            
            # Execute sustained trading simulation
            start_time = time.time()
            execution_results = []
            
            for i, signal in enumerate(signals):
                if i % 100 == 0:
                    print(f"   Processing trade {i+1}/{total_target_trades}")
                
                result = hft_simulator.simulate_order_execution(signal)
                execution_results.append(result)
                
                # Validate individual trade latency
                assert result.total_latency_ms < 100, f"Trade {i} latency {result.total_latency_ms}ms exceeds 100ms target"
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            performance_tracker.end_timing('sustained_hft_1000_tpm')
            
            # Validate performance metrics
            successful_trades = [r for r in execution_results if r.success]
            success_rate = len(successful_trades) / len(execution_results)
            
            assert success_rate >= 0.99, f"Success rate {success_rate:.2%} below 99% target"
            
            # Validate throughput
            actual_throughput = len(successful_trades) / (actual_duration / 60)  # trades per minute
            assert actual_throughput >= target_trades_per_minute, f"Throughput {actual_throughput:.1f} below {target_trades_per_minute} target"
            
            # Validate latency distribution
            latencies = [r.total_latency_ms for r in successful_trades]
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            
            assert avg_latency < 50, f"Average latency {avg_latency:.1f}ms exceeds 50ms target"
            assert p95_latency < 80, f"P95 latency {p95_latency:.1f}ms exceeds 80ms target" 
            assert p99_latency < 100, f"P99 latency {p99_latency:.1f}ms exceeds 100ms target"
            
            print(f"   ✓ Processed {len(successful_trades)} trades successfully")
            print(f"   ✓ Throughput: {actual_throughput:.1f} trades/minute")
            print(f"   ✓ Latency - Avg: {avg_latency:.1f}ms, P95: {p95_latency:.1f}ms, P99: {p99_latency:.1f}ms")
    
    def test_burst_trading_performance(self, mock_environment_variables, hft_simulator):
        """Test burst trading performance (2000 trades in 30 seconds)."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== BURST TRADING PERFORMANCE TEST ===")
            
            burst_trades = 2000
            burst_duration_seconds = 30
            
            # This will initially fail - implementation needed
            signals = hft_simulator.generate_trading_signals(
                count=burst_trades,
                time_window_minutes=0.5  # 30 seconds
            )
            
            start_time = time.time()
            
            # Use threading for true concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(hft_simulator.simulate_order_execution, signal) for signal in signals]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            # Validate burst performance
            assert actual_duration <= burst_duration_seconds + 5, f"Burst took {actual_duration:.1f}s, target {burst_duration_seconds}s"
            
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            
            assert success_rate >= 0.95, f"Burst success rate {success_rate:.2%} below 95% target"
            
            # Validate latency under burst conditions
            burst_latencies = [r.total_latency_ms for r in successful_results]
            avg_burst_latency = statistics.mean(burst_latencies)
            
            assert avg_burst_latency < 150, f"Burst average latency {avg_burst_latency:.1f}ms exceeds 150ms target"
            
            print(f"   ✓ Burst completed in {actual_duration:.1f}s")
            print(f"   ✓ Success rate: {success_rate:.2%}")
            print(f"   ✓ Average burst latency: {avg_burst_latency:.1f}ms")


class TestMarketVolatilityPerformance:
    """Test performance during various market volatility conditions."""
    
    @pytest.fixture
    def hft_simulator(self):
        """Provide HFT simulator."""
        return HFTPerformanceSimulator()
    
    def test_low_volatility_performance_baseline(self, mock_environment_variables, hft_simulator):
        """Test performance during low volatility (baseline)."""
        with patch.dict('os.environ', mock_environment_variables):
            scenario = MarketVolatilityScenario(
                volatility_level='low',
                price_change_rate=0.1,  # 0.1% per minute
                volume_spike_factor=1.0,
                duration_minutes=10
            )
            
            # This will initially fail - implementation needed
            market_conditions = hft_simulator.simulate_market_volatility(scenario)
            
            # Test trading under low volatility
            signals = hft_simulator.generate_trading_signals(count=500, time_window_minutes=10)
            results = [hft_simulator.simulate_order_execution(signal) for signal in signals[:100]]
            
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            
            # Should have very high success rate under low volatility
            assert success_rate >= 0.99, f"Low volatility success rate {success_rate:.2%} below 99%"
            
            avg_latency = statistics.mean([r.total_latency_ms for r in successful_results])
            assert avg_latency < 50, f"Low volatility latency {avg_latency:.1f}ms exceeds 50ms"
    
    def test_high_volatility_performance_resilience(self, mock_environment_variables, hft_simulator):
        """Test performance resilience during high volatility."""
        with patch.dict('os.environ', mock_environment_variables):
            scenario = MarketVolatilityScenario(
                volatility_level='high',
                price_change_rate=5.0,  # 5% per minute
                volume_spike_factor=3.0,
                duration_minutes=15
            )
            
            # This will initially fail - implementation needed
            market_conditions = hft_simulator.simulate_market_volatility(scenario)
            
            # Test trading under high volatility
            signals = hft_simulator.generate_trading_signals(count=750, time_window_minutes=15)
            results = [hft_simulator.simulate_order_execution(signal) for signal in signals[:150]]
            
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            
            # Should maintain reasonable success rate even under high volatility
            assert success_rate >= 0.90, f"High volatility success rate {success_rate:.2%} below 90%"
            
            avg_latency = statistics.mean([r.total_latency_ms for r in successful_results])
            assert avg_latency < 80, f"High volatility latency {avg_latency:.1f}ms exceeds 80ms"
            
            print(f"   ✓ High volatility success rate: {success_rate:.2%}")
            print(f"   ✓ High volatility average latency: {avg_latency:.1f}ms")
    
    def test_extreme_volatility_survival(self, mock_environment_variables, hft_simulator):
        """Test system survival during extreme volatility."""
        with patch.dict('os.environ', mock_environment_variables):
            scenario = MarketVolatilityScenario(
                volatility_level='extreme',
                price_change_rate=20.0,  # 20% per minute
                volume_spike_factor=10.0,
                duration_minutes=5
            )
            
            # This will initially fail - implementation needed
            market_conditions = hft_simulator.simulate_market_volatility(scenario)
            
            # Test trading under extreme volatility
            signals = hft_simulator.generate_trading_signals(count=200, time_window_minutes=5)
            results = [hft_simulator.simulate_order_execution(signal) for signal in signals[:50]]
            
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            
            # Focus on system survival rather than high performance
            assert success_rate >= 0.70, f"Extreme volatility success rate {success_rate:.2%} below 70%"
            
            # No crashes or system failures
            crash_results = [r for r in results if r.error_msg and 'crash' in r.error_msg.lower()]
            assert len(crash_results) == 0, f"System crashed during extreme volatility: {len(crash_results)} crashes"
            
            print(f"   ✓ Extreme volatility survival rate: {success_rate:.2%}")


class TestConcurrentMultiAssetTrading:
    """Test concurrent trading across multiple assets."""
    
    @pytest.fixture
    def concurrent_manager(self):
        """Provide concurrent trading manager."""
        return ConcurrentTradingManager(max_concurrent_assets=50)
    
    def test_10_asset_concurrent_trading(self, mock_environment_variables, concurrent_manager):
        """Test concurrent trading across 10 assets."""
        with patch.dict('os.environ', mock_environment_variables):
            assets = [f"TOKEN{i}/USD" for i in range(10)]
            duration_minutes = 5
            
            # This will initially fail - implementation needed
            results = concurrent_manager.start_concurrent_trading(assets, duration_minutes)
            
            assert len(results) == 10, f"Expected results for 10 assets, got {len(results)}"
            
            # All assets should trade successfully
            successful_assets = [r for r in results if r['success']]
            success_rate = len(successful_assets) / len(results)
            
            assert success_rate >= 0.95, f"Multi-asset success rate {success_rate:.2%} below 95%"
            
            # Latency should remain reasonable with concurrent trading
            avg_latencies = [r['avg_latency_ms'] for r in successful_assets]
            overall_avg_latency = statistics.mean(avg_latencies)
            
            assert overall_avg_latency < 120, f"Concurrent trading latency {overall_avg_latency:.1f}ms exceeds 120ms"
    
    def test_25_asset_concurrent_trading(self, mock_environment_variables, concurrent_manager):
        """Test concurrent trading across 25 assets."""
        with patch.dict('os.environ', mock_environment_variables):
            assets = [f"TOKEN{i}/USD" for i in range(25)]
            duration_minutes = 10
            
            # This will initially fail - implementation needed
            results = concurrent_manager.start_concurrent_trading(assets, duration_minutes)
            
            successful_assets = [r for r in results if r['success']]
            success_rate = len(successful_assets) / len(results)
            
            assert success_rate >= 0.90, f"25-asset success rate {success_rate:.2%} below 90%"
            
            # Check resource usage
            resource_usage = concurrent_manager.monitor_resource_usage()
            
            assert resource_usage['memory_usage_percent'] < 80, f"Memory usage {resource_usage['memory_usage_percent']:.1f}% too high"
            assert resource_usage['cpu_usage_percent'] < 85, f"CPU usage {resource_usage['cpu_usage_percent']:.1f}% too high"
    
    def test_50_asset_concurrent_trading_stress(self, mock_environment_variables, concurrent_manager):
        """Test concurrent trading across 50 assets (stress test)."""
        with patch.dict('os.environ', mock_environment_variables):
            assets = [f"TOKEN{i}/USD" for i in range(50)]
            duration_minutes = 3  # Shorter duration for stress test
            
            # This will initially fail - implementation needed
            results = concurrent_manager.start_concurrent_trading(assets, duration_minutes)
            
            successful_assets = [r for r in results if r['success']]
            success_rate = len(successful_assets) / len(results)
            
            # Lower success rate acceptable for stress test
            assert success_rate >= 0.80, f"50-asset stress success rate {success_rate:.2%} below 80%"
            
            # System should not crash
            system_failures = [r for r in results if 'system_failure' in r.get('error_type', '')]
            assert len(system_failures) == 0, f"System failures during 50-asset stress test: {len(system_failures)}"
            
            print(f"   ✓ 50-asset stress test success rate: {success_rate:.2%}")


class TestEmergencyStopPerformance:
    """Test emergency stop performance under various load conditions."""
    
    @pytest.fixture
    def hft_simulator(self):
        """Provide HFT simulator."""
        return HFTPerformanceSimulator()
    
    def test_emergency_stop_no_load(self, mock_environment_variables, hft_simulator):
        """Test emergency stop performance with no active load."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            stop_latency_ms = hft_simulator.trigger_emergency_stop()
            
            assert stop_latency_ms < 5, f"Emergency stop with no load took {stop_latency_ms:.1f}ms, target <5ms"
            assert hft_simulator.emergency_stop_triggered, "Emergency stop should be triggered"
    
    def test_emergency_stop_moderate_load(self, mock_environment_variables, hft_simulator):
        """Test emergency stop performance under moderate trading load."""
        with patch.dict('os.environ', mock_environment_variables):
            # Generate moderate load
            signals = hft_simulator.generate_trading_signals(count=100, time_window_minutes=2)
            
            # Start background trading
            def background_trading():
                for signal in signals[:50]:
                    hft_simulator.simulate_order_execution(signal)
                    time.sleep(0.01)  # 100ms between trades
            
            trading_thread = threading.Thread(target=background_trading)
            trading_thread.start()
            
            # Trigger emergency stop during trading
            time.sleep(0.5)  # Let some trading happen
            stop_latency_ms = hft_simulator.trigger_emergency_stop()
            
            trading_thread.join(timeout=5)
            
            assert stop_latency_ms < 10, f"Emergency stop under load took {stop_latency_ms:.1f}ms, target <10ms"
    
    def test_emergency_stop_high_load(self, mock_environment_variables, hft_simulator):
        """Test emergency stop performance under high trading load."""
        with patch.dict('os.environ', mock_environment_variables):
            # Generate high load
            signals = hft_simulator.generate_trading_signals(count=500, time_window_minutes=1)
            
            # Start multiple background trading threads
            def high_frequency_trading():
                for signal in signals[:100]:
                    hft_simulator.simulate_order_execution(signal)
                    time.sleep(0.001)  # 1ms between trades
            
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=high_frequency_trading)
                threads.append(thread)
                thread.start()
            
            # Trigger emergency stop during high-frequency trading
            time.sleep(0.2)
            stop_latency_ms = hft_simulator.trigger_emergency_stop()
            
            # Wait for threads to complete
            for thread in threads:
                thread.join(timeout=3)
            
            assert stop_latency_ms < 15, f"Emergency stop under high load took {stop_latency_ms:.1f}ms, target <15ms"


class TestMLRLPipelinePerformance:
    """Test ML/RL pipeline performance in HFT scenarios."""
    
    @pytest.fixture
    def hft_simulator(self):
        """Provide HFT simulator.""" 
        return HFTPerformanceSimulator()
    
    def test_ml_prediction_throughput_hft(self, mock_environment_variables, hft_simulator):
        """Test ML prediction throughput for HFT scenarios."""
        with patch.dict('os.environ', mock_environment_variables):
            # Generate signals for throughput testing
            signals = hft_simulator.generate_trading_signals(count=1000, time_window_minutes=1)
            
            start_time = time.time()
            
            predictions = []
            for signal in signals:
                # This will initially fail - implementation needed
                prediction, latency_ms = hft_simulator.simulate_ml_prediction(signal)
                predictions.append((prediction, latency_ms))
                
                # Individual prediction should be fast
                assert latency_ms < 1, f"ML prediction took {latency_ms:.1f}ms, target <1ms"
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Throughput should exceed 1000 predictions per second
            throughput = len(predictions) / total_time
            assert throughput >= 1000, f"ML prediction throughput {throughput:.1f}/sec below 1000/sec target"
            
            print(f"   ✓ ML prediction throughput: {throughput:.1f} predictions/sec")
    
    def test_rl_decision_throughput_hft(self, mock_environment_variables, hft_simulator):
        """Test RL decision throughput for HFT scenarios."""
        with patch.dict('os.environ', mock_environment_variables):
            signals = hft_simulator.generate_trading_signals(count=1000, time_window_minutes=1)
            
            start_time = time.time()
            
            decisions = []
            for signal in signals:
                # Mock ML prediction for RL input
                ml_prediction = {'confidence': 0.8, 'direction': 'buy', 'strength': 0.7}
                
                # This will initially fail - implementation needed
                action, confidence, latency_ms = hft_simulator.simulate_rl_decision(signal, ml_prediction)
                decisions.append((action, confidence, latency_ms))
                
                # Individual decision should be fast
                assert latency_ms < 1, f"RL decision took {latency_ms:.1f}ms, target <1ms"
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Throughput should exceed 1000 decisions per second
            throughput = len(decisions) / total_time
            assert throughput >= 1000, f"RL decision throughput {throughput:.1f}/sec below 1000/sec target"
            
            print(f"   ✓ RL decision throughput: {throughput:.1f} decisions/sec")
    
    def test_integrated_ml_rl_pipeline_throughput(self, mock_environment_variables, hft_simulator):
        """Test integrated ML+RL pipeline throughput."""
        with patch.dict('os.environ', mock_environment_variables):
            signals = hft_simulator.generate_trading_signals(count=500, time_window_minutes=1)
            
            start_time = time.time()
            
            pipeline_results = []
            for signal in signals:
                # This will initially fail - implementation needed
                result = hft_simulator.simulate_order_execution(signal)
                pipeline_results.append(result)
                
                # Complete pipeline should be under 1 second
                assert result.total_latency_ms < 1000, f"Pipeline latency {result.total_latency_ms:.1f}ms exceeds 1s"
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Pipeline throughput should exceed 500 complete pipelines per second
            throughput = len(pipeline_results) / total_time
            assert throughput >= 500, f"Pipeline throughput {throughput:.1f}/sec below 500/sec target"
            
            # Validate pipeline component performance
            successful_results = [r for r in pipeline_results if r.success]
            avg_ml_latency = statistics.mean([r.ml_prediction_latency_ms for r in successful_results])
            avg_rl_latency = statistics.mean([r.rl_decision_latency_ms for r in successful_results])
            avg_safety_latency = statistics.mean([r.safety_validation_latency_ms for r in successful_results])
            
            assert avg_ml_latency < 0.5, f"Average ML latency {avg_ml_latency:.2f}ms too high"
            assert avg_rl_latency < 0.5, f"Average RL latency {avg_rl_latency:.2f}ms too high"
            assert avg_safety_latency < 50, f"Average safety latency {avg_safety_latency:.1f}ms exceeds 50ms"
            
            print(f"   ✓ Integrated pipeline throughput: {throughput:.1f} pipelines/sec")
            print(f"   ✓ Component latencies - ML: {avg_ml_latency:.2f}ms, RL: {avg_rl_latency:.2f}ms, Safety: {avg_safety_latency:.1f}ms")


class TestComprehensiveHFTValidation:
    """Comprehensive HFT performance validation."""
    
    def test_comprehensive_hft_performance_suite(self, mock_environment_variables, performance_tracker):
        """Run comprehensive HFT performance validation."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== COMPREHENSIVE HFT PERFORMANCE VALIDATION ===")
            
            hft_simulator = HFTPerformanceSimulator()
            concurrent_manager = ConcurrentTradingManager()
            
            performance_tracker.start_timing('comprehensive_hft_validation')
            
            # 1. Sustained Trading Test
            print("\n1. Sustained Trading Performance:")
            signals = hft_simulator.generate_trading_signals(count=1000, time_window_minutes=2)
            sustained_results = [hft_simulator.simulate_order_execution(signal) for signal in signals[:200]]
            
            sustained_success_rate = len([r for r in sustained_results if r.success]) / len(sustained_results)
            sustained_avg_latency = statistics.mean([r.total_latency_ms for r in sustained_results if r.success])
            
            print(f"   ✓ Sustained success rate: {sustained_success_rate:.2%}")
            print(f"   ✓ Sustained average latency: {sustained_avg_latency:.1f}ms")
            
            # 2. Volatility Resilience Test
            print("\n2. Market Volatility Resilience:")
            volatility_scenario = MarketVolatilityScenario('high', 3.0, 2.0, 5)
            volatility_conditions = hft_simulator.simulate_market_volatility(volatility_scenario)
            volatility_results = [hft_simulator.simulate_order_execution(signal) for signal in signals[:100]]
            
            volatility_success_rate = len([r for r in volatility_results if r.success]) / len(volatility_results)
            print(f"   ✓ High volatility success rate: {volatility_success_rate:.2%}")
            
            # 3. Concurrent Asset Trading Test
            print("\n3. Concurrent Multi-Asset Trading:")
            assets = [f"TOKEN{i}/USD" for i in range(20)]
            concurrent_results = concurrent_manager.start_concurrent_trading(assets, 3)
            
            concurrent_success_rate = len([r for r in concurrent_results if r['success']]) / len(concurrent_results)
            print(f"   ✓ Concurrent asset success rate: {concurrent_success_rate:.2%}")
            
            # 4. Emergency Stop Test
            print("\n4. Emergency Stop Performance:")
            emergency_latency = hft_simulator.trigger_emergency_stop()
            print(f"   ✓ Emergency stop latency: {emergency_latency:.1f}ms")
            
            # 5. Pipeline Throughput Test
            print("\n5. ML/RL Pipeline Throughput:")
            pipeline_signals = signals[:100]
            pipeline_start = time.time()
            pipeline_results = [hft_simulator.simulate_order_execution(signal) for signal in pipeline_signals]
            pipeline_time = time.time() - pipeline_start
            
            pipeline_throughput = len(pipeline_results) / pipeline_time
            print(f"   ✓ Pipeline throughput: {pipeline_throughput:.1f} ops/sec")
            
            performance_tracker.end_timing('comprehensive_hft_validation')
            
            # Overall HFT Performance Validation
            print("\n=== HFT PERFORMANCE SUMMARY ===")
            print(f"✓ Sustained trading: {sustained_success_rate:.2%} success, {sustained_avg_latency:.1f}ms avg latency")
            print(f"✓ Volatility resilience: {volatility_success_rate:.2%} success under high volatility")
            print(f"✓ Concurrent assets: {concurrent_success_rate:.2%} success across 20 assets")
            print(f"✓ Emergency stop: {emergency_latency:.1f}ms response time")
            print(f"✓ Pipeline throughput: {pipeline_throughput:.1f} complete pipelines/sec")
            
            # Assert HFT performance targets
            assert sustained_success_rate >= 0.95, f"Sustained trading success rate {sustained_success_rate:.2%} below 95%"
            assert sustained_avg_latency < 100, f"Sustained trading latency {sustained_avg_latency:.1f}ms exceeds 100ms"
            assert volatility_success_rate >= 0.85, f"Volatility resilience {volatility_success_rate:.2%} below 85%"
            assert concurrent_success_rate >= 0.90, f"Concurrent asset success {concurrent_success_rate:.2%} below 90%"
            assert emergency_latency < 15, f"Emergency stop latency {emergency_latency:.1f}ms exceeds 15ms"
            assert pipeline_throughput >= 500, f"Pipeline throughput {pipeline_throughput:.1f} below 500 ops/sec"


# Mark all tests as HFT performance tests
pytestmark = [pytest.mark.performance, pytest.mark.hft]