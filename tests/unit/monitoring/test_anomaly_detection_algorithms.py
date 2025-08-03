"""
Test suite for anomaly detection algorithms for trading patterns and system metrics.

This module provides comprehensive TDD tests for ML-powered anomaly detection
algorithms used in the intelligent alerting system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch

# The classes we'll implement
from src.monitoring.anomaly_detection_algorithms import (
    TradingPatternAnomalyDetector,
    SystemMetricsAnomalyDetector,
    MarketAnomalyDetector,
    EnsembleAnomalyDetector,
    AnomalyResult,
    AnomalyType,
    AnomalyAlgorithm
)


class TestTradingPatternAnomalyDetector:
    """Test trading pattern anomaly detection algorithms."""
    
    @pytest.fixture
    def trading_detector(self):
        """Create trading pattern anomaly detector."""
        config = {
            'volume_threshold_multiplier': 3.0,
            'price_change_threshold': 0.05,
            'liquidity_threshold': 0.1,
            'wash_trading_detection': True,
            'pump_dump_detection': True,
            'flash_crash_detection': True
        }
        return TradingPatternAnomalyDetector(config)
    
    @pytest.fixture
    def normal_trading_data(self):
        """Create normal trading data."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'price': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),
            'volume': np.random.normal(1000, 100, n_samples),
            'bid_ask_spread': np.random.normal(0.01, 0.001, n_samples),
            'trade_count': np.random.poisson(50, n_samples),
            'market_cap': np.full(n_samples, 1000000)
        })
    
    @pytest.fixture
    def anomalous_trading_data(self):
        """Create trading data with anomalies."""
        normal_data = self.normal_trading_data()
        
        # Add volume spike anomaly
        normal_data.loc[50, 'volume'] = 10000  # 10x normal volume
        
        # Add flash crash anomaly
        normal_data.loc[60:65, 'price'] *= 0.8  # 20% price drop
        
        # Add pump and dump pattern
        normal_data.loc[70:75, 'price'] *= 1.3  # 30% price spike
        normal_data.loc[76:80, 'price'] *= 0.9  # followed by dump
        
        return normal_data
    
    @pytest.mark.asyncio
    async def test_volume_spike_detection(self, trading_detector, normal_trading_data):
        """Test detection of volume spikes."""
        # Create data with volume spike
        data = normal_trading_data.copy()
        data.loc[50, 'volume'] = 5000  # 5x normal volume
        
        anomalies = await trading_detector.detect_volume_anomalies(data)
        
        # Should detect volume spike
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_SPIKE]
        assert len(volume_anomalies) > 0
        assert any(a.metadata.get('volume_multiplier', 0) > 3.0 for a in volume_anomalies)
    
    @pytest.mark.asyncio
    async def test_flash_crash_detection(self, trading_detector, normal_trading_data):
        """Test detection of flash crashes."""
        # Create data with flash crash
        data = normal_trading_data.copy()
        base_price = data.loc[50, 'price']
        data.loc[50:55, 'price'] = base_price * 0.7  # 30% crash
        
        anomalies = await trading_detector.detect_price_anomalies(data)
        
        # Should detect flash crash
        crash_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.FLASH_CRASH]
        assert len(crash_anomalies) > 0
        assert any(a.metadata.get('price_drop_percent', 0) > 20 for a in crash_anomalies)
    
    @pytest.mark.asyncio
    async def test_pump_and_dump_detection(self, trading_detector, normal_trading_data):
        """Test detection of pump and dump patterns."""
        # Create pump and dump pattern
        data = normal_trading_data.copy()
        base_price = data.loc[50, 'price']
        
        # Pump phase
        data.loc[50:55, 'price'] = base_price * 1.4  # 40% pump
        data.loc[50:55, 'volume'] *= 3  # High volume during pump
        
        # Dump phase
        data.loc[56:60, 'price'] = base_price * 0.9  # Dump below original
        data.loc[56:60, 'volume'] *= 2  # High volume during dump
        
        anomalies = await trading_detector.detect_pump_dump_patterns(data)
        
        # Should detect pump and dump
        pump_dump_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.PUMP_AND_DUMP]
        assert len(pump_dump_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_wash_trading_detection(self, trading_detector, normal_trading_data):
        """Test detection of wash trading patterns."""
        # Create wash trading pattern (high volume, low price movement)
        data = normal_trading_data.copy()
        
        # High volume with minimal price movement
        data.loc[50:60, 'volume'] *= 5
        data.loc[50:60, 'price'] = data.loc[50, 'price']  # No price movement
        data.loc[50:60, 'trade_count'] *= 3  # Many small trades
        
        anomalies = await trading_detector.detect_wash_trading(data)
        
        # Should detect wash trading
        wash_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.WASH_TRADING]
        assert len(wash_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_liquidity_anomaly_detection(self, trading_detector, normal_trading_data):
        """Test detection of liquidity anomalies."""
        # Create liquidity crisis
        data = normal_trading_data.copy()
        data.loc[50:60, 'bid_ask_spread'] *= 10  # Wide spreads indicate low liquidity
        data.loc[50:60, 'volume'] *= 0.1  # Low volume
        
        anomalies = await trading_detector.detect_liquidity_anomalies(data)
        
        # Should detect liquidity anomaly
        liquidity_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.LIQUIDITY_CRISIS]
        assert len(liquidity_anomalies) > 0
    
    def test_trading_pattern_feature_extraction(self, trading_detector, normal_trading_data):
        """Test extraction of trading pattern features."""
        features = trading_detector._extract_trading_features(normal_trading_data)
        
        # Should extract relevant features
        expected_features = [
            'volume_rolling_mean', 'volume_rolling_std', 'price_volatility',
            'price_momentum', 'trade_size_avg', 'bid_ask_spread_normalized'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_detector_configuration_validation(self):
        """Test validation of detector configuration."""
        # Test valid configuration
        valid_config = {
            'volume_threshold_multiplier': 3.0,
            'price_change_threshold': 0.05
        }
        detector = TradingPatternAnomalyDetector(valid_config)
        assert detector.config['volume_threshold_multiplier'] == 3.0
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            invalid_config = {'volume_threshold_multiplier': -1.0}
            TradingPatternAnomalyDetector(invalid_config)


class TestSystemMetricsAnomalyDetector:
    """Test system metrics anomaly detection algorithms."""
    
    @pytest.fixture
    def system_detector(self):
        """Create system metrics anomaly detector."""
        config = {
            'cpu_threshold': 0.8,
            'memory_threshold': 0.85,
            'disk_threshold': 0.9,
            'response_time_threshold': 1000,  # ms
            'error_rate_threshold': 0.05,
            'detection_window_minutes': 5
        }
        return SystemMetricsAnomalyDetector(config)
    
    @pytest.fixture
    def normal_system_data(self):
        """Create normal system metrics data."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'cpu_usage': np.random.normal(0.4, 0.1, n_samples),
            'memory_usage': np.random.normal(0.5, 0.1, n_samples),
            'disk_usage': np.random.normal(0.3, 0.05, n_samples),
            'response_time_ms': np.random.normal(200, 50, n_samples),
            'error_rate': np.random.normal(0.01, 0.005, n_samples),
            'network_io_mb': np.random.normal(100, 20, n_samples),
            'active_connections': np.random.poisson(50, n_samples)
        })
    
    @pytest.mark.asyncio
    async def test_cpu_spike_detection(self, system_detector, normal_system_data):
        """Test detection of CPU spikes."""
        # Create data with CPU spike
        data = normal_system_data.copy()
        data.loc[50:55, 'cpu_usage'] = 0.95  # High CPU usage
        
        anomalies = await system_detector.detect_cpu_anomalies(data)
        
        # Should detect CPU spike
        cpu_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CPU_SPIKE]
        assert len(cpu_anomalies) > 0
        assert any(a.metadata.get('cpu_usage', 0) > 0.8 for a in cpu_anomalies)
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, system_detector, normal_system_data):
        """Test detection of memory leaks."""
        # Create data with gradual memory increase (memory leak pattern)
        data = normal_system_data.copy()
        data.loc[50:, 'memory_usage'] = np.linspace(0.5, 0.9, len(data) - 50)
        
        anomalies = await system_detector.detect_memory_anomalies(data)
        
        # Should detect memory leak pattern
        memory_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.MEMORY_LEAK]
        assert len(memory_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_response_time_degradation(self, system_detector, normal_system_data):
        """Test detection of response time degradation."""
        # Create data with response time degradation
        data = normal_system_data.copy()
        data.loc[50:60, 'response_time_ms'] = 2000  # High response times
        
        anomalies = await system_detector.detect_performance_anomalies(data)
        
        # Should detect response time anomaly
        perf_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION]
        assert len(perf_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_error_rate_spike_detection(self, system_detector, normal_system_data):
        """Test detection of error rate spikes."""
        # Create data with error rate spike
        data = normal_system_data.copy()
        data.loc[50:55, 'error_rate'] = 0.1  # 10% error rate
        
        anomalies = await system_detector.detect_error_anomalies(data)
        
        # Should detect error rate spike
        error_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.ERROR_RATE_SPIKE]
        assert len(error_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_network_anomaly_detection(self, system_detector, normal_system_data):
        """Test detection of network anomalies."""
        # Create data with network anomaly
        data = normal_system_data.copy()
        data.loc[50:55, 'network_io_mb'] = 1000  # Unusually high network I/O
        
        anomalies = await system_detector.detect_network_anomalies(data)
        
        # Should detect network anomaly
        network_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.NETWORK_ANOMALY]
        assert len(network_anomalies) > 0
    
    def test_system_health_scoring(self, system_detector, normal_system_data):
        """Test system health scoring algorithm."""
        health_score = system_detector.calculate_health_score(normal_system_data.iloc[-10:])
        
        # Normal data should have high health score
        assert 0.8 <= health_score <= 1.0
        
        # Test with degraded system
        degraded_data = normal_system_data.copy()
        degraded_data.iloc[-10:] = degraded_data.iloc[-10:] * 2  # Double all metrics
        
        degraded_score = system_detector.calculate_health_score(degraded_data.iloc[-10:])
        assert degraded_score < health_score
    
    def test_anomaly_severity_calculation(self, system_detector):
        """Test anomaly severity calculation."""
        # Test different severity levels
        low_severity = system_detector._calculate_anomaly_severity(0.81, 0.8)  # Just above threshold
        high_severity = system_detector._calculate_anomaly_severity(0.95, 0.8)  # Well above threshold
        
        assert low_severity < high_severity
        assert 0 <= low_severity <= 1
        assert 0 <= high_severity <= 1


class TestMarketAnomalyDetector:
    """Test market-wide anomaly detection algorithms."""
    
    @pytest.fixture
    def market_detector(self):
        """Create market anomaly detector."""
        config = {
            'market_correlation_threshold': 0.8,
            'volatility_spike_threshold': 2.0,
            'market_crash_threshold': 0.1,  # 10% market drop
            'sector_analysis_enabled': True
        }
        return MarketAnomalyDetector(config)
    
    @pytest.fixture
    def market_data(self):
        """Create market data for multiple assets."""
        np.random.seed(42)
        n_samples = 100
        n_assets = 10
        
        timestamps = pd.date_range('2025-01-01', periods=n_samples, freq='1min')
        
        # Create correlated market data
        market_returns = np.random.normal(0, 0.01, n_samples)
        
        data = {}
        for i in range(n_assets):
            asset_returns = market_returns + np.random.normal(0, 0.005, n_samples)
            data[f'asset_{i}'] = pd.DataFrame({
                'timestamp': timestamps,
                'price': 100 * np.exp(np.cumsum(asset_returns)),
                'volume': np.random.normal(1000, 100, n_samples),
                'market_cap': np.random.normal(1000000, 100000, n_samples)
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_market_crash_detection(self, market_detector, market_data):
        """Test detection of market-wide crashes."""
        # Simulate market crash
        for asset_name, asset_data in market_data.items():
            asset_data.loc[50:55, 'price'] *= 0.85  # 15% drop across all assets
        
        anomalies = await market_detector.detect_market_crashes(market_data)
        
        # Should detect market crash
        crash_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.MARKET_CRASH]
        assert len(crash_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_correlation_breakdown_detection(self, market_detector, market_data):
        """Test detection of correlation breakdowns."""
        # Break correlation for one asset
        asset_data = market_data['asset_0']
        asset_data.loc[50:60, 'price'] *= 1.2  # This asset moves differently
        
        anomalies = await market_detector.detect_correlation_anomalies(market_data)
        
        # Should detect correlation breakdown
        corr_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CORRELATION_BREAKDOWN]
        assert len(corr_anomalies) > 0
    
    @pytest.mark.asyncio
    async def test_volatility_clustering_detection(self, market_detector, market_data):
        """Test detection of volatility clustering."""
        # Create volatility clustering pattern
        for asset_name, asset_data in market_data.items():
            # High volatility period
            asset_data.loc[50:60, 'price'] *= np.random.normal(1, 0.1, 11)
        
        anomalies = await market_detector.detect_volatility_anomalies(market_data)
        
        # Should detect volatility clustering
        vol_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLATILITY_CLUSTERING]
        assert len(vol_anomalies) > 0
    
    def test_market_correlation_calculation(self, market_detector, market_data):
        """Test market correlation calculation."""
        correlations = market_detector._calculate_market_correlations(market_data)
        
        # Should have correlation matrix
        assert correlations.shape[0] == len(market_data)
        assert correlations.shape[1] == len(market_data)
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(correlations), 1.0)
    
    def test_sector_analysis(self, market_detector, market_data):
        """Test sector-based anomaly analysis."""
        # Add sector information
        sectors = {'asset_0': 'tech', 'asset_1': 'tech', 'asset_2': 'finance'}
        
        sector_anomalies = market_detector._analyze_sector_anomalies(market_data, sectors)
        
        # Should analyze sectors separately
        assert isinstance(sector_anomalies, dict)


class TestEnsembleAnomalyDetector:
    """Test ensemble anomaly detection that combines multiple algorithms."""
    
    @pytest.fixture
    def ensemble_detector(self):
        """Create ensemble anomaly detector."""
        config = {
            'voting_strategy': 'weighted',
            'confidence_threshold': 0.7,
            'algorithm_weights': {
                'isolation_forest': 0.3,
                'statistical': 0.3,
                'neural_network': 0.4
            }
        }
        return EnsembleAnomalyDetector(config)
    
    @pytest.fixture
    def mock_detectors(self):
        """Create mock individual detectors."""
        trading_detector = MagicMock(spec=TradingPatternAnomalyDetector)
        system_detector = MagicMock(spec=SystemMetricsAnomalyDetector)
        market_detector = MagicMock(spec=MarketAnomalyDetector)
        
        return {
            'trading': trading_detector,
            'system': system_detector,
            'market': market_detector
        }
    
    @pytest.mark.asyncio
    async def test_ensemble_voting(self, ensemble_detector, mock_detectors):
        """Test ensemble voting mechanism."""
        # Configure mock detectors to return different results
        mock_detectors['trading'].detect_anomalies.return_value = [
            AnomalyResult(confidence=0.8, anomaly_type=AnomalyType.VOLUME_SPIKE)
        ]
        mock_detectors['system'].detect_anomalies.return_value = [
            AnomalyResult(confidence=0.6, anomaly_type=AnomalyType.CPU_SPIKE)
        ]
        mock_detectors['market'].detect_anomalies.return_value = []
        
        # Set detectors
        ensemble_detector.detectors = mock_detectors
        
        # Test ensemble detection
        test_data = {'mock': 'data'}
        results = await ensemble_detector.detect_anomalies(test_data)
        
        # Should combine results based on voting strategy
        assert len(results) >= 1  # High confidence trading anomaly should be included
    
    def test_confidence_aggregation(self, ensemble_detector):
        """Test confidence score aggregation."""
        # Test different aggregation methods
        confidences = [0.8, 0.6, 0.9]
        
        # Weighted average
        weighted_confidence = ensemble_detector._aggregate_confidence(
            confidences, 
            weights=[0.3, 0.3, 0.4],
            method='weighted_average'
        )
        expected = 0.8 * 0.3 + 0.6 * 0.3 + 0.9 * 0.4
        assert abs(weighted_confidence - expected) < 0.01
        
        # Maximum confidence
        max_confidence = ensemble_detector._aggregate_confidence(
            confidences, 
            method='maximum'
        )
        assert max_confidence == 0.9
    
    def test_algorithm_weight_validation(self):
        """Test validation of algorithm weights."""
        # Test valid weights
        valid_config = {
            'algorithm_weights': {
                'isolation_forest': 0.5,
                'statistical': 0.5
            }
        }
        detector = EnsembleAnomalyDetector(valid_config)
        assert sum(detector.config['algorithm_weights'].values()) == 1.0
        
        # Test invalid weights (don't sum to 1)
        with pytest.raises(ValueError):
            invalid_config = {
                'algorithm_weights': {
                    'isolation_forest': 0.3,
                    'statistical': 0.3
                }
            }
            EnsembleAnomalyDetector(invalid_config)
    
    def test_anomaly_type_consensus(self, ensemble_detector):
        """Test anomaly type consensus mechanism."""
        # Multiple detectors agree on anomaly type
        anomaly_results = [
            AnomalyResult(confidence=0.8, anomaly_type=AnomalyType.VOLUME_SPIKE),
            AnomalyResult(confidence=0.7, anomaly_type=AnomalyType.VOLUME_SPIKE),
            AnomalyResult(confidence=0.6, anomaly_type=AnomalyType.CPU_SPIKE)
        ]
        
        consensus_type = ensemble_detector._determine_consensus_type(anomaly_results)
        assert consensus_type == AnomalyType.VOLUME_SPIKE


class TestAnomalyResult:
    """Test anomaly result data structure."""
    
    def test_anomaly_result_creation(self):
        """Test creation of anomaly result."""
        result = AnomalyResult(
            confidence=0.8,
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            timestamp=datetime.now(),
            affected_metrics=['volume', 'price'],
            metadata={'volume_multiplier': 5.0}
        )
        
        assert result.confidence == 0.8
        assert result.anomaly_type == AnomalyType.VOLUME_SPIKE
        assert 'volume' in result.affected_metrics
        assert result.metadata['volume_multiplier'] == 5.0
    
    def test_anomaly_result_serialization(self):
        """Test serialization of anomaly result."""
        result = AnomalyResult(
            confidence=0.8,
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            affected_metrics=['volume']
        )
        
        serialized = result.to_dict()
        
        assert serialized['confidence'] == 0.8
        assert serialized['anomaly_type'] == AnomalyType.VOLUME_SPIKE.value
        assert 'volume' in serialized['affected_metrics']
    
    def test_anomaly_result_comparison(self):
        """Test comparison of anomaly results."""
        result1 = AnomalyResult(confidence=0.8, anomaly_type=AnomalyType.VOLUME_SPIKE)
        result2 = AnomalyResult(confidence=0.6, anomaly_type=AnomalyType.CPU_SPIKE)
        
        # Higher confidence should be "greater"
        assert result1 > result2
        assert result2 < result1


class TestAnomalyAlgorithm:
    """Test base anomaly algorithm class."""
    
    def test_algorithm_interface(self):
        """Test anomaly algorithm interface."""
        # Should be able to create instances of concrete algorithms
        algorithm = AnomalyAlgorithm.create_algorithm('isolation_forest', {})
        assert algorithm is not None
        
        # Should support common interface
        assert hasattr(algorithm, 'fit')
        assert hasattr(algorithm, 'predict')
        assert hasattr(algorithm, 'get_anomaly_score')
    
    def test_algorithm_configuration(self):
        """Test algorithm configuration validation."""
        # Test valid configuration
        config = {'contamination': 0.1, 'random_state': 42}
        algorithm = AnomalyAlgorithm.create_algorithm('isolation_forest', config)
        assert algorithm.config['contamination'] == 0.1
        
        # Test default configuration
        algorithm_default = AnomalyAlgorithm.create_algorithm('isolation_forest', {})
        assert hasattr(algorithm_default, 'config')
    
    def test_algorithm_registry(self):
        """Test algorithm registry for different types."""
        available_algorithms = AnomalyAlgorithm.get_available_algorithms()
        
        expected_algorithms = [
            'isolation_forest', 'one_class_svm', 'local_outlier_factor',
            'statistical_outlier', 'neural_network_autoencoder'
        ]
        
        for algorithm in expected_algorithms:
            assert algorithm in available_algorithms


class TestPerformanceAndScalability:
    """Test performance and scalability of anomaly detection."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'metric1': np.random.normal(0, 1, n_samples),
            'metric2': np.random.normal(0, 1, n_samples),
            'metric3': np.random.normal(0, 1, n_samples)
        })
        
        detector = TradingPatternAnomalyDetector({})
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        anomalies = await detector.detect_anomalies(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with streaming data."""
        detector = SystemMetricsAnomalyDetector({})
        
        # Process data in chunks to test memory efficiency
        chunk_size = 1000
        total_processed = 0
        
        for i in range(5):  # 5 chunks
            chunk_data = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=chunk_size, freq='1min'),
                'cpu_usage': np.random.normal(0.4, 0.1, chunk_size),
                'memory_usage': np.random.normal(0.5, 0.1, chunk_size)
            })
            
            anomalies = await detector.detect_anomalies(chunk_data)
            total_processed += len(chunk_data)
        
        assert total_processed == 5 * chunk_size
    
    def test_parallel_processing(self):
        """Test parallel processing capabilities."""
        detector = EnsembleAnomalyDetector({})
        
        # Should support parallel processing
        assert hasattr(detector, 'enable_parallel_processing')
        
        # Test configuration
        detector.enable_parallel_processing(max_workers=4)
        assert detector.max_workers == 4