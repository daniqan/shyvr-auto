"""
Comprehensive failing tests for Phase 2.4.1 - Transformer Metrics Monitoring

This module contains failing tests that define the expected behavior for transformer-specific 
metrics monitoring components. Following TDD methodology, these tests will fail initially as 
the implementation doesn't exist yet.

Test Coverage Areas:
- Attention entropy monitoring
- Prediction confidence tracking
- Model performance by market regime
- Resource usage monitoring (memory, compute)
- Gradient flow health checks
- Training stability indicators
- Embedding drift monitoring
- Token embedding stability
- Positional encoding drift

Requirements:
- <1 hour detection time for significant drift
- Comprehensive model health monitoring
- Real-time metrics collection and analysis
- Integration with existing monitoring systems
- Support for all transformer models (iTransformer, PatchTST, TimesMixer, TimesFM)
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import MagicMock, patch
import warnings
import time
import psutil
import os

# Import existing monitoring components
from src.monitoring.cloud_monitoring import CloudMonitoringManager
from src.monitoring.operational_analytics import OperationalAnalyticsManager

# These imports will fail initially - this is expected for TDD
try:
    from src.monitoring.transformer_metrics import (
        TransformerMetricsCollector,
        AttentionEntropyMonitor,
        PredictionConfidenceTracker,
        MarketRegimePerformanceMonitor,
        ResourceUsageMonitor,
        GradientFlowHealthChecker,
        TrainingStabilityIndicator,
        EmbeddingDriftMonitor,
        TransformerHealthDashboard,
        TransformerMetricsResult,
        AttentionHealthMetrics,
        PerformanceMetrics,
        ResourceMetrics,
        EmbeddingMetrics,
        ModelHealthStatus
    )
except ImportError:
    # Expected to fail initially - create placeholder classes for testing
    class TransformerMetricsCollector:
        pass
    
    class AttentionEntropyMonitor:
        pass
    
    class PredictionConfidenceTracker:
        pass
    
    class MarketRegimePerformanceMonitor:
        pass
    
    class ResourceUsageMonitor:
        pass
    
    class GradientFlowHealthChecker:
        pass
    
    class TrainingStabilityIndicator:
        pass
    
    class EmbeddingDriftMonitor:
        pass
    
    class TransformerHealthDashboard:
        pass
    
    class TransformerMetricsResult:
        pass
    
    class AttentionHealthMetrics:
        pass
    
    class PerformanceMetrics:
        pass
    
    class ResourceMetrics:
        pass
    
    class EmbeddingMetrics:
        pass
    
    class ModelHealthStatus:
        pass


class TestTransformerMetricsCollector:
    """Test suite for TransformerMetricsCollector main class"""
    
    @pytest.fixture
    def mock_transformer_models(self):
        """Create mock transformer models with realistic interfaces"""
        models = {}
        
        # Mock iTransformer
        itransformer = MagicMock()
        itransformer.model_type = 'iTransformer'
        itransformer.parameters.return_value = [torch.rand(100, 512) for _ in range(20)]
        itransformer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        itransformer.get_embeddings.return_value = torch.rand(100, 512)
        itransformer.get_prediction_confidence.return_value = torch.tensor([0.85, 0.92, 0.78, 0.96])
        models['iTransformer'] = itransformer
        
        # Mock PatchTST
        patchtst = MagicMock()
        patchtst.model_type = 'PatchTST'
        patchtst.parameters.return_value = [torch.rand(50, 512) for _ in range(15)]
        patchtst.get_attention_weights.return_value = torch.rand(4, 8, 20, 20)
        patchtst.get_embeddings.return_value = torch.rand(20, 512)
        patchtst.get_prediction_confidence.return_value = torch.tensor([0.88, 0.79, 0.91, 0.84])
        models['PatchTST'] = patchtst
        
        # Mock TimesMixer
        timesmixer = MagicMock()
        timesmixer.model_type = 'TimesMixer'
        timesmixer.parameters.return_value = [torch.rand(80, 512) for _ in range(18)]
        timesmixer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        timesmixer.get_embeddings.return_value = torch.rand(100, 512)
        timesmixer.get_prediction_confidence.return_value = torch.tensor([0.82, 0.89, 0.76, 0.93])
        models['TimesMixer'] = timesmixer
        
        # Mock TimesFM
        timesfm = MagicMock()
        timesfm.model_type = 'TimesFM'
        timesfm.parameters.return_value = [torch.rand(200, 768) for _ in range(24)]
        timesfm.get_attention_weights.return_value = torch.rand(4, 12, 512, 512)
        timesfm.get_embeddings.return_value = torch.rand(512, 768)
        timesfm.get_prediction_confidence.return_value = torch.tensor([0.91, 0.87, 0.95, 0.83])
        models['TimesFM'] = timesfm
        
        return models
    
    @pytest.fixture
    def market_regime_data(self):
        """Create market regime data for testing"""
        regimes = {
            'bull_market': {
                'timestamps': pd.date_range('2024-01-01', periods=500, freq='1H'),
                'btc_returns': np.random.normal(0.02, 0.05, 500),  # Positive trend
                'volatility': np.random.exponential(0.015, 500),   # Lower volatility
                'regime_label': 'bull'
            },
            'bear_market': {
                'timestamps': pd.date_range('2024-03-01', periods=500, freq='1H'),
                'btc_returns': np.random.normal(-0.015, 0.08, 500),  # Negative trend  
                'volatility': np.random.exponential(0.03, 500),      # Higher volatility
                'regime_label': 'bear'
            },
            'sideways_market': {
                'timestamps': pd.date_range('2024-05-01', periods=500, freq='1H'),
                'btc_returns': np.random.normal(0.001, 0.02, 500),  # Flat trend
                'volatility': np.random.exponential(0.01, 500),     # Low volatility
                'regime_label': 'sideways'
            },
            'high_volatility': {
                'timestamps': pd.date_range('2024-07-01', periods=500, freq='1H'),
                'btc_returns': np.random.normal(0.005, 0.12, 500),  # High variance
                'volatility': np.random.exponential(0.05, 500),     # Very high volatility
                'regime_label': 'volatile'
            }
        }
        return regimes
    
    def test_transformer_metrics_collector_initialization(self, mock_transformer_models):
        """Test TransformerMetricsCollector initialization"""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=60,
            metrics_retention_hours=24,
            health_check_threshold=0.8
        )
        
        # Expected initialization behavior
        assert collector.models == mock_transformer_models
        assert collector.collection_interval_seconds == 60
        assert collector.metrics_retention_hours == 24
        assert collector.health_check_threshold == 0.8
        assert hasattr(collector, 'attention_entropy_monitor')
        assert hasattr(collector, 'confidence_tracker')
        assert hasattr(collector, 'resource_monitor')
        assert hasattr(collector, 'gradient_health_checker')
        assert hasattr(collector, 'embedding_drift_monitor')
        assert hasattr(collector, 'metrics_history')
    
    def test_collect_all_metrics(self, mock_transformer_models, market_regime_data):
        """Test comprehensive metrics collection"""
        collector = TransformerMetricsCollector(models=mock_transformer_models)
        
        # Collect metrics for all models
        metrics_result = collector.collect_all_metrics(
            current_market_regime='bull',
            trading_data=pd.DataFrame({
                'timestamp': market_regime_data['bull_market']['timestamps'][:100],
                'btc_price': np.random.randn(100) * 1000 + 45000,
                'eth_price': np.random.randn(100) * 100 + 3000,
                'volume': np.random.exponential(1000, 100)
            })
        )
        
        # Expected metrics structure
        assert isinstance(metrics_result, TransformerMetricsResult)
        assert metrics_result.timestamp is not None
        assert metrics_result.overall_health_status in ['healthy', 'warning', 'critical']
        assert 'iTransformer' in metrics_result.model_metrics
        assert 'PatchTST' in metrics_result.model_metrics
        assert 'TimesMixer' in metrics_result.model_metrics
        assert 'TimesFM' in metrics_result.model_metrics
        
        # Each model should have comprehensive metrics
        for model_type, model_metrics in metrics_result.model_metrics.items():
            assert hasattr(model_metrics, 'attention_health')
            assert hasattr(model_metrics, 'performance_metrics')
            assert hasattr(model_metrics, 'resource_metrics')
            assert hasattr(model_metrics, 'embedding_metrics')
            assert hasattr(model_metrics, 'health_status')
    
    def test_real_time_metrics_streaming(self, mock_transformer_models):
        """Test real-time metrics streaming capabilities"""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            streaming_enabled=True,
            stream_buffer_size=100
        )
        
        # Start streaming
        collector.start_streaming()
        assert collector.is_streaming is True
        
        # Simulate multiple metrics collections
        streaming_results = []
        for i in range(5):
            result = collector.collect_streaming_metrics()
            streaming_results.append(result)
            time.sleep(0.1)  # Small delay to simulate real-time
        
        # Stop streaming
        collector.stop_streaming()
        assert collector.is_streaming is False
        
        # Validate streaming results
        assert len(streaming_results) == 5
        assert all(isinstance(result, TransformerMetricsResult) for result in streaming_results)
        assert all(result.collection_latency_ms < 100 for result in streaming_results)
    
    def test_metrics_detection_within_one_hour(self, mock_transformer_models):
        """Test that significant drift is detected within 1 hour (success criteria)"""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            detection_window_minutes=60,
            drift_sensitivity=0.1
        )
        
        # Simulate 1 hour of metrics collection (60 minutes, 1 per minute)
        detection_time_minutes = 0
        significant_drift_detected = False
        
        for minute in range(60):
            # Gradually introduce drift patterns
            drift_factor = minute / 60.0
            
            # Modify model behavior to simulate drift
            for model in mock_transformer_models.values():
                # Simulate attention entropy decline
                entropy_drift = torch.rand(4, 8) * (1 - drift_factor * 0.5)
                model.get_attention_entropy = MagicMock(return_value=entropy_drift)
                
                # Simulate confidence decline
                confidence_drift = torch.tensor([0.9, 0.85, 0.8, 0.75]) * (1 - drift_factor * 0.3)
                model.get_prediction_confidence = MagicMock(return_value=confidence_drift)
            
            # Collect metrics
            metrics_result = collector.collect_all_metrics()
            
            # Check for significant drift detection
            if metrics_result.overall_health_status == 'critical' or \
               any(model_metrics.health_status == 'critical' 
                   for model_metrics in metrics_result.model_metrics.values()):
                significant_drift_detected = True
                detection_time_minutes = minute + 1
                break
        
        # Success criteria: detect significant drift within 1 hour
        assert significant_drift_detected, "Significant drift should be detected within 1 hour"
        assert detection_time_minutes <= 60, f"Drift detected after {detection_time_minutes} minutes, should be ≤60"


class TestAttentionEntropyMonitor:
    """Test attention entropy monitoring functionality"""
    
    def test_attention_entropy_monitor_initialization(self):
        """Test AttentionEntropyMonitor initialization"""
        monitor = AttentionEntropyMonitor(
            entropy_baseline_threshold=2.0,
            entropy_warning_threshold=1.5,
            entropy_critical_threshold=1.0,
            head_diversity_threshold=0.8
        )
        
        assert monitor.entropy_baseline_threshold == 2.0
        assert monitor.entropy_warning_threshold == 1.5
        assert monitor.entropy_critical_threshold == 1.0
        assert monitor.head_diversity_threshold == 0.8
        assert hasattr(monitor, 'baseline_entropy_stats')
    
    def test_compute_attention_entropy(self, mock_transformer_models):
        """Test attention entropy computation"""
        monitor = AttentionEntropyMonitor()
        
        # Test entropy computation for different models
        for model_type, model in mock_transformer_models.items():
            attention_weights = model.get_attention_weights()
            entropy_metrics = monitor.compute_attention_entropy(attention_weights)
            
            assert isinstance(entropy_metrics, AttentionHealthMetrics)
            assert hasattr(entropy_metrics, 'mean_entropy')
            assert hasattr(entropy_metrics, 'entropy_std')
            assert hasattr(entropy_metrics, 'head_entropy_distribution')
            assert hasattr(entropy_metrics, 'entropy_health_status')
            
            # Validate entropy properties
            assert entropy_metrics.mean_entropy >= 0.0
            assert entropy_metrics.entropy_std >= 0.0
            assert len(entropy_metrics.head_entropy_distribution) > 0
    
    def test_detect_entropy_degradation(self, mock_transformer_models):
        """Test detection of attention entropy degradation"""
        monitor = AttentionEntropyMonitor()
        
        # Establish baseline
        model = mock_transformer_models['iTransformer']
        baseline_attention = model.get_attention_weights()
        baseline_entropy = monitor.compute_attention_entropy(baseline_attention)
        monitor.set_baseline_entropy(baseline_entropy)
        
        # Create degraded attention (more concentrated, lower entropy)
        degraded_attention = torch.softmax(baseline_attention * 3.0, dim=-1)  # Sharper distribution
        degraded_entropy = monitor.compute_attention_entropy(degraded_attention)
        
        degradation_result = monitor.detect_entropy_degradation(
            current_entropy=degraded_entropy,
            baseline_entropy=baseline_entropy
        )
        
        assert degradation_result.has_degradation is True
        assert degradation_result.degradation_severity in ['low', 'moderate', 'high']
        assert degradation_result.entropy_decline_ratio > 0.0
        assert 'head_specific_degradation' in degradation_result.metadata
    
    def test_monitor_head_diversity(self, mock_transformer_models):
        """Test monitoring of attention head diversity"""
        monitor = AttentionEntropyMonitor()
        
        model = mock_transformer_models['iTransformer']
        attention_weights = model.get_attention_weights()
        
        # Test head diversity analysis
        diversity_metrics = monitor.monitor_head_diversity(attention_weights)
        
        assert hasattr(diversity_metrics, 'head_similarity_matrix')
        assert hasattr(diversity_metrics, 'diversity_index')
        assert hasattr(diversity_metrics, 'specialization_scores')
        assert hasattr(diversity_metrics, 'redundant_heads')
        
        # Validate diversity properties
        assert 0.0 <= diversity_metrics.diversity_index <= 1.0
        assert diversity_metrics.head_similarity_matrix.shape[0] == attention_weights.shape[1]  # n_heads
        assert len(diversity_metrics.specialization_scores) == attention_weights.shape[1]
    
    def test_entropy_anomaly_detection(self, mock_transformer_models):
        """Test anomaly detection in attention entropy patterns"""
        monitor = AttentionEntropyMonitor(anomaly_detection_enabled=True)
        
        # Collect baseline entropy measurements
        model = mock_transformer_models['iTransformer']
        baseline_entropies = []
        
        for _ in range(50):  # 50 baseline measurements
            attention = torch.rand(4, 8, 100, 100)
            attention = torch.softmax(attention, dim=-1)
            entropy_metrics = monitor.compute_attention_entropy(attention)
            baseline_entropies.append(entropy_metrics.mean_entropy)
        
        monitor.fit_anomaly_detector(baseline_entropies)
        
        # Test anomaly detection with outlier
        anomalous_attention = torch.zeros(4, 8, 100, 100)
        anomalous_attention[:, :, 0, 0] = 1.0  # Extremely concentrated attention
        anomalous_entropy = monitor.compute_attention_entropy(anomalous_attention)
        
        anomaly_result = monitor.detect_entropy_anomaly(anomalous_entropy.mean_entropy)
        
        assert anomaly_result.is_anomaly is True
        assert anomaly_result.anomaly_score > 0.5
        assert anomaly_result.confidence > 0.8


class TestPredictionConfidenceTracker:
    """Test prediction confidence tracking functionality"""
    
    def test_confidence_tracker_initialization(self):
        """Test PredictionConfidenceTracker initialization"""
        tracker = PredictionConfidenceTracker(
            confidence_window_size=100,
            low_confidence_threshold=0.7,
            high_confidence_threshold=0.9,
            trend_detection_periods=10
        )
        
        assert tracker.confidence_window_size == 100
        assert tracker.low_confidence_threshold == 0.7
        assert tracker.high_confidence_threshold == 0.9
        assert tracker.trend_detection_periods == 10
        assert hasattr(tracker, 'confidence_history')
    
    def test_track_prediction_confidence(self, mock_transformer_models):
        """Test prediction confidence tracking"""
        tracker = PredictionConfidenceTracker()
        
        # Track confidence for multiple predictions
        confidence_results = []
        
        for model_type, model in mock_transformer_models.items():
            confidence_scores = model.get_prediction_confidence()
            confidence_result = tracker.track_confidence(
                model_type=model_type,
                confidence_scores=confidence_scores,
                prediction_targets=['BTC_1h', 'ETH_1h', 'SOL_1h', 'BTC_4h']
            )
            confidence_results.append(confidence_result)
        
        # Validate tracking results
        assert len(confidence_results) == len(mock_transformer_models)
        
        for result in confidence_results:
            assert hasattr(result, 'mean_confidence')
            assert hasattr(result, 'confidence_std')
            assert hasattr(result, 'low_confidence_count')
            assert hasattr(result, 'confidence_distribution')
            assert hasattr(result, 'confidence_trend')
            
            # Validate confidence properties
            assert 0.0 <= result.mean_confidence <= 1.0
            assert result.confidence_std >= 0.0
            assert result.low_confidence_count >= 0
    
    def test_detect_confidence_degradation(self, mock_transformer_models):
        """Test detection of confidence degradation trends"""
        tracker = PredictionConfidenceTracker(trend_detection_periods=5)
        
        model = mock_transformer_models['iTransformer']
        
        # Simulate declining confidence over time
        declining_confidences = []
        for i in range(20):
            # Gradually decrease confidence
            base_confidence = torch.tensor([0.9, 0.85, 0.8, 0.75])
            decline_factor = i * 0.02  # 2% decline per period
            current_confidence = base_confidence - decline_factor
            declining_confidences.append(current_confidence)
            
            tracker.track_confidence(
                model_type='iTransformer',
                confidence_scores=current_confidence,
                prediction_targets=['BTC_1h', 'ETH_1h', 'SOL_1h', 'BTC_4h']
            )
        
        # Detect degradation trend
        degradation_result = tracker.detect_confidence_degradation('iTransformer')
        
        assert degradation_result.has_degradation is True
        assert degradation_result.degradation_rate < 0  # Negative slope
        assert degradation_result.degradation_significance > 0.8  # Strong trend
        assert 'trend_analysis' in degradation_result.metadata
    
    def test_confidence_by_market_conditions(self, mock_transformer_models, market_regime_data):
        """Test confidence tracking by different market conditions"""
        tracker = PredictionConfidenceTracker(market_condition_aware=True)
        
        model = mock_transformer_models['iTransformer']
        
        # Track confidence across different market regimes
        regime_confidence_results = {}
        
        for regime_name, regime_data in market_regime_data.items():
            # Simulate regime-specific confidence patterns
            if regime_name == 'bull_market':
                confidence_base = torch.tensor([0.92, 0.88, 0.85, 0.90])
            elif regime_name == 'bear_market':
                confidence_base = torch.tensor([0.78, 0.82, 0.75, 0.80])
            elif regime_name == 'sideways_market':
                confidence_base = torch.tensor([0.85, 0.87, 0.83, 0.86])
            else:  # high_volatility
                confidence_base = torch.tensor([0.70, 0.75, 0.68, 0.72])
            
            result = tracker.track_confidence_by_regime(
                model_type='iTransformer',
                confidence_scores=confidence_base,
                market_regime=regime_name,
                volatility_level=regime_data['volatility'][:4].mean()
            )
            
            regime_confidence_results[regime_name] = result
        
        # Validate regime-specific results
        assert len(regime_confidence_results) == 4
        
        # Bull market should have higher confidence than bear market
        assert regime_confidence_results['bull_market'].mean_confidence > \
               regime_confidence_results['bear_market'].mean_confidence
        
        # High volatility should have lower confidence
        assert regime_confidence_results['high_volatility'].mean_confidence < \
               regime_confidence_results['sideways_market'].mean_confidence
    
    def test_confidence_calibration_analysis(self, mock_transformer_models):
        """Test prediction confidence calibration analysis"""
        tracker = PredictionConfidenceTracker(calibration_analysis_enabled=True)
        
        model = mock_transformer_models['iTransformer']
        
        # Simulate predictions with known accuracy
        predictions_data = []
        
        for i in range(100):
            confidence_scores = torch.rand(4)  # Random confidence
            actual_accuracy = torch.rand(4)    # Random accuracy
            
            predictions_data.append({
                'confidence': confidence_scores,
                'accuracy': actual_accuracy,
                'timestamp': datetime.now() - timedelta(hours=i)
            })
        
        # Analyze calibration
        calibration_result = tracker.analyze_confidence_calibration(
            predictions_data=predictions_data,
            model_type='iTransformer'
        )
        
        assert hasattr(calibration_result, 'calibration_error')
        assert hasattr(calibration_result, 'reliability_diagram')
        assert hasattr(calibration_result, 'overconfidence_ratio')
        assert hasattr(calibration_result, 'underconfidence_ratio')
        assert hasattr(calibration_result, 'calibration_slope')
        
        # Validate calibration metrics
        assert 0.0 <= calibration_result.calibration_error <= 1.0
        assert 0.0 <= calibration_result.overconfidence_ratio <= 1.0
        assert 0.0 <= calibration_result.underconfidence_ratio <= 1.0


class TestMarketRegimePerformanceMonitor:
    """Test market regime performance monitoring"""
    
    def test_regime_performance_monitor_initialization(self):
        """Test MarketRegimePerformanceMonitor initialization"""
        monitor = MarketRegimePerformanceMonitor(
            regimes=['bull', 'bear', 'sideways', 'volatile'],
            performance_window_hours=24,
            regime_transition_sensitivity=0.1
        )
        
        assert monitor.regimes == ['bull', 'bear', 'sideways', 'volatile']
        assert monitor.performance_window_hours == 24
        assert monitor.regime_transition_sensitivity == 0.1
        assert hasattr(monitor, 'regime_performance_history')
    
    def test_monitor_performance_by_regime(self, mock_transformer_models, market_regime_data):
        """Test performance monitoring across different market regimes"""
        monitor = MarketRegimePerformanceMonitor()
        
        performance_results = {}
        
        for regime_name, regime_data in market_regime_data.items():
            # Simulate model performance metrics for each regime
            performance_metrics = {
                'accuracy': 0.85 if regime_name == 'bull_market' else 0.75,
                'precision': 0.82 if regime_name == 'bull_market' else 0.70,
                'recall': 0.88 if regime_name == 'bull_market' else 0.72,
                'f1_score': 0.85 if regime_name == 'bull_market' else 0.71,
                'sharpe_ratio': 1.2 if regime_name == 'bull_market' else 0.8,
                'max_drawdown': 0.05 if regime_name == 'bull_market' else 0.12
            }
            
            result = monitor.monitor_regime_performance(
                model_type='iTransformer',
                regime=regime_name,
                performance_metrics=performance_metrics,
                market_data=pd.DataFrame({
                    'timestamp': regime_data['timestamps'][:100],
                    'returns': regime_data['btc_returns'][:100],
                    'volatility': regime_data['volatility'][:100]
                })
            )
            
            performance_results[regime_name] = result
        
        # Validate regime performance results
        assert len(performance_results) == len(market_regime_data)
        
        for regime, result in performance_results.items():
            assert hasattr(result, 'regime_name')
            assert hasattr(result, 'performance_scores')
            assert hasattr(result, 'relative_performance')
            assert hasattr(result, 'regime_stability')
            assert result.regime_name == regime
    
    def test_detect_regime_performance_degradation(self, mock_transformer_models, market_regime_data):
        """Test detection of performance degradation within regimes"""
        monitor = MarketRegimePerformanceMonitor()
        
        # Establish baseline performance for bull market
        baseline_metrics = {
            'accuracy': 0.85,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05
        }
        
        monitor.set_baseline_performance('bull_market', baseline_metrics)
        
        # Simulate degraded performance
        degraded_metrics = {
            'accuracy': 0.70,  # Significant drop
            'sharpe_ratio': 0.8,  # Significant drop
            'max_drawdown': 0.15  # Significant increase (worse)
        }
        
        degradation_result = monitor.detect_performance_degradation(
            regime='bull_market',
            current_metrics=degraded_metrics
        )
        
        assert degradation_result.has_degradation is True
        assert degradation_result.degradation_severity in ['moderate', 'severe']
        assert 'accuracy' in degradation_result.degraded_metrics
        assert 'sharpe_ratio' in degradation_result.degraded_metrics
        assert degradation_result.overall_degradation_score > 0.2
    
    def test_regime_transition_detection(self, market_regime_data):
        """Test detection of market regime transitions"""
        monitor = MarketRegimePerformanceMonitor(regime_transition_sensitivity=0.1)
        
        # Simulate regime transition from bull to bear
        transition_data = pd.concat([
            pd.DataFrame({
                'timestamp': market_regime_data['bull_market']['timestamps'][-50:],
                'returns': market_regime_data['bull_market']['btc_returns'][-50:],
                'volatility': market_regime_data['bull_market']['volatility'][-50:]
            }),
            pd.DataFrame({
                'timestamp': market_regime_data['bear_market']['timestamps'][:50],
                'returns': market_regime_data['bear_market']['btc_returns'][:50],
                'volatility': market_regime_data['bear_market']['volatility'][:50]
            })
        ]).reset_index(drop=True)
        
        transition_result = monitor.detect_regime_transition(
            historical_data=transition_data,
            current_regime='bull_market'
        )
        
        assert hasattr(transition_result, 'transition_detected')
        assert hasattr(transition_result, 'from_regime')
        assert hasattr(transition_result, 'to_regime')
        assert hasattr(transition_result, 'transition_confidence')
        assert hasattr(transition_result, 'transition_timestamp')
        
        if transition_result.transition_detected:
            assert transition_result.from_regime == 'bull_market'
            assert transition_result.to_regime in ['bear_market', 'volatile', 'sideways']
            assert 0.0 <= transition_result.transition_confidence <= 1.0


class TestResourceUsageMonitor:
    """Test resource usage monitoring functionality"""
    
    def test_resource_monitor_initialization(self):
        """Test ResourceUsageMonitor initialization"""
        monitor = ResourceUsageMonitor(
            memory_warning_threshold_mb=1000,
            memory_critical_threshold_mb=2000,
            gpu_utilization_threshold=0.8,
            cpu_utilization_threshold=0.7
        )
        
        assert monitor.memory_warning_threshold_mb == 1000
        assert monitor.memory_critical_threshold_mb == 2000
        assert monitor.gpu_utilization_threshold == 0.8
        assert monitor.cpu_utilization_threshold == 0.7
        assert hasattr(monitor, 'resource_history')
    
    def test_monitor_memory_usage(self, mock_transformer_models):
        """Test memory usage monitoring for transformer models"""
        monitor = ResourceUsageMonitor()
        
        # Monitor memory usage for each model
        memory_results = {}
        
        for model_type, model in mock_transformer_models.items():
            # Mock memory usage based on model size
            if model_type == 'TimesFM':
                mock_memory_mb = 1500  # Larger foundation model
            elif model_type == 'iTransformer':
                mock_memory_mb = 800
            elif model_type == 'TimesMixer':
                mock_memory_mb = 900
            else:  # PatchTST
                mock_memory_mb = 600
            
            memory_result = monitor.monitor_memory_usage(
                model_type=model_type,
                model=model,
                current_memory_mb=mock_memory_mb
            )
            
            memory_results[model_type] = memory_result
        
        # Validate memory monitoring results
        assert len(memory_results) == len(mock_transformer_models)
        
        for model_type, result in memory_results.items():
            assert hasattr(result, 'current_memory_mb')
            assert hasattr(result, 'memory_growth_rate')
            assert hasattr(result, 'memory_status')  # 'normal', 'warning', 'critical'
            assert hasattr(result, 'peak_memory_mb')
            assert hasattr(result, 'memory_efficiency_score')
            
            assert result.current_memory_mb > 0
            assert result.memory_status in ['normal', 'warning', 'critical']
    
    def test_monitor_gpu_utilization(self, mock_transformer_models):
        """Test GPU utilization monitoring"""
        monitor = ResourceUsageMonitor()
        
        # Mock GPU utilization data
        gpu_utilization_data = {
            'gpu_memory_used_mb': 3500,
            'gpu_memory_total_mb': 8000,
            'gpu_utilization_percent': 85.0,
            'gpu_temperature_celsius': 72.0,
            'gpu_power_draw_watts': 180.0
        }
        
        gpu_result = monitor.monitor_gpu_utilization(
            gpu_stats=gpu_utilization_data,
            active_models=list(mock_transformer_models.keys())
        )
        
        assert hasattr(gpu_result, 'gpu_memory_utilization')
        assert hasattr(gpu_result, 'gpu_compute_utilization')
        assert hasattr(gpu_result, 'gpu_temperature')
        assert hasattr(gpu_result, 'gpu_power_efficiency')
        assert hasattr(gpu_result, 'gpu_health_status')
        
        # Validate GPU metrics
        assert 0.0 <= gpu_result.gpu_memory_utilization <= 1.0
        assert 0.0 <= gpu_result.gpu_compute_utilization <= 1.0
        assert gpu_result.gpu_temperature > 0
        assert gpu_result.gpu_health_status in ['normal', 'warning', 'critical']
    
    def test_detect_resource_anomalies(self, mock_transformer_models):
        """Test detection of resource usage anomalies"""
        monitor = ResourceUsageMonitor(anomaly_detection_enabled=True)
        
        # Establish baseline resource usage
        baseline_resources = []
        for i in range(50):
            normal_usage = ResourceMetrics(
                memory_mb=800 + np.random.normal(0, 50),
                cpu_percent=30 + np.random.normal(0, 5),
                gpu_percent=70 + np.random.normal(0, 10),
                inference_time_ms=50 + np.random.normal(0, 5)
            )
            baseline_resources.append(normal_usage)
        
        monitor.fit_resource_anomaly_detector(baseline_resources)
        
        # Test anomaly detection with extreme usage
        anomalous_usage = ResourceMetrics(
            memory_mb=2000,  # 2.5x normal
            cpu_percent=80,   # 2.5x normal
            gpu_percent=95,   # 1.3x normal
            inference_time_ms=200  # 4x normal
        )
        
        anomaly_result = monitor.detect_resource_anomaly(anomalous_usage)
        
        assert anomaly_result.is_anomaly is True
        assert anomaly_result.anomaly_score > 0.7
        assert 'memory' in anomaly_result.anomalous_metrics
        assert 'cpu' in anomaly_result.anomalous_metrics
        assert 'inference_time' in anomaly_result.anomalous_metrics
    
    def test_resource_optimization_recommendations(self, mock_transformer_models):
        """Test resource optimization recommendations"""
        monitor = ResourceUsageMonitor(optimization_recommendations_enabled=True)
        
        # Simulate suboptimal resource usage
        resource_data = {
            'iTransformer': ResourceMetrics(memory_mb=1200, cpu_percent=40, gpu_percent=60, inference_time_ms=80),
            'PatchTST': ResourceMetrics(memory_mb=700, cpu_percent=25, gpu_percent=45, inference_time_ms=45),
            'TimesMixer': ResourceMetrics(memory_mb=950, cpu_percent=35, gpu_percent=55, inference_time_ms=70),
            'TimesFM': ResourceMetrics(memory_mb=1800, cpu_percent=50, gpu_percent=85, inference_time_ms=120)
        }
        
        optimization_result = monitor.generate_optimization_recommendations(resource_data)
        
        assert hasattr(optimization_result, 'recommendations')
        assert hasattr(optimization_result, 'potential_savings')
        assert hasattr(optimization_result, 'priority_actions')
        
        # Should have recommendations for high-usage models
        assert len(optimization_result.recommendations) > 0
        assert any('TimesFM' in rec for rec in optimization_result.recommendations)
        assert optimization_result.potential_savings.memory_mb > 0


class TestGradientFlowHealthChecker:
    """Test gradient flow health checking functionality"""
    
    def test_gradient_health_checker_initialization(self):
        """Test GradientFlowHealthChecker initialization"""
        checker = GradientFlowHealthChecker(
            gradient_norm_threshold=10.0,
            vanishing_gradient_threshold=1e-7,
            exploding_gradient_threshold=100.0,
            layer_wise_analysis_enabled=True
        )
        
        assert checker.gradient_norm_threshold == 10.0
        assert checker.vanishing_gradient_threshold == 1e-7
        assert checker.exploding_gradient_threshold == 100.0
        assert checker.layer_wise_analysis_enabled is True
        assert hasattr(checker, 'gradient_history')
    
    def test_analyze_gradient_flow(self, mock_transformer_models):
        """Test gradient flow analysis"""
        checker = GradientFlowHealthChecker()
        
        # Create mock gradients for different models
        gradient_results = {}
        
        for model_type, model in mock_transformer_models.items():
            # Mock gradients with realistic patterns
            mock_gradients = []
            n_layers = len(list(model.parameters()))
            
            for i, param in enumerate(model.parameters()):
                if i < n_layers // 3:  # Early layers - smaller gradients
                    grad = torch.randn_like(param) * 0.01
                elif i < 2 * n_layers // 3:  # Middle layers - normal gradients
                    grad = torch.randn_like(param) * 0.1
                else:  # Later layers - larger gradients
                    grad = torch.randn_like(param) * 0.5
                
                param.grad = grad
                mock_gradients.append(grad)
            
            gradient_result = checker.analyze_gradient_flow(
                model=model,
                model_type=model_type
            )
            
            gradient_results[model_type] = gradient_result
        
        # Validate gradient analysis results
        assert len(gradient_results) == len(mock_transformer_models)
        
        for model_type, result in gradient_results.items():
            assert hasattr(result, 'overall_gradient_norm')
            assert hasattr(result, 'layer_wise_norms')
            assert hasattr(result, 'vanishing_layers')
            assert hasattr(result, 'exploding_layers')
            assert hasattr(result, 'gradient_flow_health')
            
            assert result.overall_gradient_norm >= 0.0
            assert len(result.layer_wise_norms) > 0
            assert result.gradient_flow_health in ['healthy', 'warning', 'critical']
    
    def test_detect_vanishing_gradients(self, mock_transformer_models):
        """Test detection of vanishing gradients"""
        checker = GradientFlowHealthChecker(vanishing_gradient_threshold=1e-6)
        
        model = mock_transformer_models['iTransformer']
        
        # Create vanishing gradient scenario
        for i, param in enumerate(model.parameters()):
            if i < 10:  # First 10 layers have vanishing gradients
                param.grad = torch.full_like(param, 1e-8)  # Very small gradients
            else:
                param.grad = torch.randn_like(param) * 0.1  # Normal gradients
        
        vanishing_result = checker.detect_vanishing_gradients(model, 'iTransformer')
        
        assert vanishing_result.has_vanishing_gradients is True
        assert len(vanishing_result.vanishing_layers) > 0
        assert vanishing_result.vanishing_severity in ['moderate', 'severe']
        assert 'gradient_statistics' in vanishing_result.metadata
    
    def test_detect_exploding_gradients(self, mock_transformer_models):
        """Test detection of exploding gradients"""
        checker = GradientFlowHealthChecker(exploding_gradient_threshold=50.0)
        
        model = mock_transformer_models['iTransformer']
        
        # Create exploding gradient scenario
        for i, param in enumerate(model.parameters()):
            if i >= 15:  # Last layers have exploding gradients
                param.grad = torch.full_like(param, 100.0)  # Very large gradients
            else:
                param.grad = torch.randn_like(param) * 0.1  # Normal gradients
        
        exploding_result = checker.detect_exploding_gradients(model, 'iTransformer')
        
        assert exploding_result.has_exploding_gradients is True
        assert len(exploding_result.exploding_layers) > 0
        assert exploding_result.exploding_severity in ['moderate', 'severe']
        assert 'gradient_statistics' in exploding_result.metadata
    
    def test_gradient_flow_optimization_suggestions(self, mock_transformer_models):
        """Test gradient flow optimization suggestions"""
        checker = GradientFlowHealthChecker(optimization_suggestions_enabled=True)
        
        model = mock_transformer_models['iTransformer']
        
        # Create mixed gradient health scenario
        for i, param in enumerate(model.parameters()):
            if i < 5:  # Vanishing gradients
                param.grad = torch.full_like(param, 1e-8)
            elif i > 15:  # Exploding gradients
                param.grad = torch.full_like(param, 200.0)
            else:  # Normal gradients
                param.grad = torch.randn_like(param) * 0.1
        
        optimization_result = checker.generate_optimization_suggestions(model, 'iTransformer')
        
        assert hasattr(optimization_result, 'suggestions')
        assert hasattr(optimization_result, 'priority_level')
        assert hasattr(optimization_result, 'expected_improvements')
        
        # Should suggest gradient clipping for exploding gradients
        suggestions_text = ' '.join(optimization_result.suggestions)
        assert 'gradient clipping' in suggestions_text.lower() or 'clip' in suggestions_text.lower()
        
        # Should suggest learning rate adjustment or normalization for vanishing gradients
        assert any(keyword in suggestions_text.lower() 
                  for keyword in ['learning rate', 'normalization', 'initialization'])


class TestTrainingStabilityIndicator:
    """Test training stability indicators"""
    
    def test_stability_indicator_initialization(self):
        """Test TrainingStabilityIndicator initialization"""
        indicator = TrainingStabilityIndicator(
            loss_smoothing_window=20,
            stability_threshold=0.05,
            convergence_patience=10
        )
        
        assert indicator.loss_smoothing_window == 20
        assert indicator.stability_threshold == 0.05
        assert indicator.convergence_patience == 10
        assert hasattr(indicator, 'training_history')
    
    def test_analyze_training_stability(self, mock_transformer_models):
        """Test training stability analysis"""
        indicator = TrainingStabilityIndicator()
        
        # Simulate training history with different stability patterns
        training_scenarios = {
            'stable_training': {
                'losses': [1.0 - 0.05 * i + np.random.normal(0, 0.01) for i in range(50)],
                'expected_stability': 'stable'
            },
            'unstable_training': {
                'losses': [1.0 + 0.1 * np.sin(i) + np.random.normal(0, 0.1) for i in range(50)],
                'expected_stability': 'unstable'
            },
            'divergent_training': {
                'losses': [1.0 + 0.1 * i + np.random.normal(0, 0.05) for i in range(50)],
                'expected_stability': 'divergent'
            }
        }
        
        stability_results = {}
        
        for scenario_name, scenario_data in training_scenarios.items():
            stability_result = indicator.analyze_training_stability(
                loss_history=scenario_data['losses'],
                model_type='iTransformer',
                scenario_name=scenario_name
            )
            
            stability_results[scenario_name] = stability_result
        
        # Validate stability analysis
        for scenario, result in stability_results.items():
            assert hasattr(result, 'stability_score')
            assert hasattr(result, 'stability_status')
            assert hasattr(result, 'loss_trend')
            assert hasattr(result, 'volatility_measure')
            assert hasattr(result, 'convergence_indicator')
            
            assert 0.0 <= result.stability_score <= 1.0
            assert result.stability_status in ['stable', 'unstable', 'divergent', 'converged']
    
    def test_detect_training_divergence(self):
        """Test detection of training divergence"""
        indicator = TrainingStabilityIndicator()
        
        # Create divergent loss pattern
        divergent_losses = [0.5 + 0.1 * i + 0.02 * i**1.5 for i in range(30)]
        
        divergence_result = indicator.detect_training_divergence(
            loss_history=divergent_losses,
            model_type='iTransformer'
        )
        
        assert divergence_result.is_divergent is True
        assert divergence_result.divergence_point > 0
        assert divergence_result.divergence_rate > 0
        assert 'loss_growth_analysis' in divergence_result.metadata
    
    def test_convergence_analysis(self):
        """Test training convergence analysis"""
        indicator = TrainingStabilityIndicator(convergence_patience=5)
        
        # Create converged loss pattern
        converged_losses = []
        for i in range(50):
            if i < 30:
                loss = 1.0 - 0.03 * i + np.random.normal(0, 0.01)
            else:
                loss = 0.1 + np.random.normal(0, 0.005)  # Converged to low value
            converged_losses.append(loss)
        
        convergence_result = indicator.analyze_convergence(
            loss_history=converged_losses,
            model_type='iTransformer'
        )
        
        assert convergence_result.has_converged is True
        assert convergence_result.convergence_point > 0
        assert convergence_result.final_loss < 0.2
        assert convergence_result.convergence_confidence > 0.8


class TestEmbeddingDriftMonitor:
    """Test embedding drift monitoring functionality"""
    
    def test_embedding_drift_monitor_initialization(self):
        """Test EmbeddingDriftMonitor initialization"""
        monitor = EmbeddingDriftMonitor(
            embedding_similarity_threshold=0.9,
            drift_detection_window=100,
            position_encoding_drift_threshold=0.1
        )
        
        assert monitor.embedding_similarity_threshold == 0.9
        assert monitor.drift_detection_window == 100
        assert monitor.position_encoding_drift_threshold == 0.1
        assert hasattr(monitor, 'baseline_embeddings')
    
    def test_monitor_token_embedding_stability(self, mock_transformer_models):
        """Test token embedding stability monitoring"""
        monitor = EmbeddingDriftMonitor()
        
        embedding_results = {}
        
        for model_type, model in mock_transformer_models.items():
            current_embeddings = model.get_embeddings()
            
            # Create baseline embeddings
            baseline_embeddings = current_embeddings + torch.randn_like(current_embeddings) * 0.01
            monitor.set_baseline_embeddings(model_type, baseline_embeddings)
            
            # Monitor stability
            stability_result = monitor.monitor_token_embedding_stability(
                model_type=model_type,
                current_embeddings=current_embeddings
            )
            
            embedding_results[model_type] = stability_result
        
        # Validate embedding stability results
        for model_type, result in embedding_results.items():
            assert hasattr(result, 'embedding_similarity')
            assert hasattr(result, 'drift_magnitude')
            assert hasattr(result, 'drift_detected')
            assert hasattr(result, 'affected_tokens')
            assert hasattr(result, 'stability_score')
            
            assert 0.0 <= result.embedding_similarity <= 1.0
            assert result.drift_magnitude >= 0.0
            assert 0.0 <= result.stability_score <= 1.0
    
    def test_monitor_positional_encoding_drift(self, mock_transformer_models):
        """Test positional encoding drift monitoring"""
        monitor = EmbeddingDriftMonitor()
        
        # Create positional encodings for testing
        seq_lengths = {'iTransformer': 100, 'PatchTST': 20, 'TimesMixer': 100, 'TimesFM': 512}
        
        for model_type, seq_len in seq_lengths.items():
            # Create baseline positional encodings
            d_model = 512 if model_type != 'TimesFM' else 768
            baseline_pos_enc = torch.randn(seq_len, d_model)
            monitor.set_baseline_positional_encoding(model_type, baseline_pos_enc)
            
            # Create slightly drifted positional encodings
            current_pos_enc = baseline_pos_enc + torch.randn_like(baseline_pos_enc) * 0.05
            
            drift_result = monitor.monitor_positional_encoding_drift(
                model_type=model_type,
                current_positional_encoding=current_pos_enc
            )
            
            assert hasattr(drift_result, 'position_drift_magnitude')
            assert hasattr(drift_result, 'drift_pattern')
            assert hasattr(drift_result, 'affected_positions')
            assert hasattr(drift_result, 'encoding_health_status')
            
            assert drift_result.position_drift_magnitude >= 0.0
            assert drift_result.encoding_health_status in ['healthy', 'warning', 'critical']
    
    def test_detect_embedding_distribution_shift(self, mock_transformer_models):
        """Test detection of embedding distribution shifts"""
        monitor = EmbeddingDriftMonitor()
        
        model = mock_transformer_models['iTransformer']
        baseline_embeddings = model.get_embeddings()
        
        # Create distribution-shifted embeddings
        shifted_embeddings = baseline_embeddings.clone()
        shifted_embeddings[:50, :] *= 1.5  # Shift first half of embeddings
        shifted_embeddings[50:, :] *= 0.7  # Shift second half differently
        
        shift_result = monitor.detect_embedding_distribution_shift(
            baseline_embeddings=baseline_embeddings,
            current_embeddings=shifted_embeddings,
            model_type='iTransformer'
        )
        
        assert hasattr(shift_result, 'distribution_shift_detected')
        assert hasattr(shift_result, 'shift_magnitude')
        assert hasattr(shift_result, 'shift_type')  # 'uniform', 'localized', 'bimodal', etc.
        assert hasattr(shift_result, 'statistical_significance')
        
        assert shift_result.distribution_shift_detected is True
        assert shift_result.shift_magnitude > 0.0
        assert 0.0 <= shift_result.statistical_significance <= 1.0
    
    def test_embedding_semantic_consistency_check(self, mock_transformer_models):
        """Test semantic consistency of embeddings"""
        monitor = EmbeddingDriftMonitor(semantic_consistency_enabled=True)
        
        # Create embeddings that should be semantically similar
        related_tokens = ['BTC', 'bitcoin', 'Bitcoin']
        unrelated_tokens = ['price', 'volume', 'timestamp']
        
        # Mock embedding lookup
        related_embeddings = torch.randn(3, 512)
        related_embeddings[1] = related_embeddings[0] + torch.randn(512) * 0.1  # Similar to BTC
        related_embeddings[2] = related_embeddings[0] + torch.randn(512) * 0.15  # Similar to BTC
        
        unrelated_embeddings = torch.randn(3, 512)
        
        consistency_result = monitor.check_semantic_consistency(
            token_groups={'crypto_related': related_tokens, 'market_data': unrelated_tokens},
            embeddings={'crypto_related': related_embeddings, 'market_data': unrelated_embeddings},
            model_type='iTransformer'
        )
        
        assert hasattr(consistency_result, 'intra_group_similarity')
        assert hasattr(consistency_result, 'inter_group_similarity')
        assert hasattr(consistency_result, 'semantic_drift_score')
        assert hasattr(consistency_result, 'consistency_violations')
        
        # Related tokens should have higher intra-group similarity
        assert consistency_result.intra_group_similarity['crypto_related'] > \
               consistency_result.inter_group_similarity


class TestTransformerHealthDashboard:
    """Test comprehensive transformer health dashboard"""
    
    def test_health_dashboard_initialization(self):
        """Test TransformerHealthDashboard initialization"""
        dashboard = TransformerHealthDashboard(
            update_interval_seconds=30,
            alert_thresholds={
                'attention_entropy': 0.8,
                'confidence': 0.7,
                'memory_usage': 0.9,
                'gradient_health': 0.6
            },
            visualization_enabled=True
        )
        
        assert dashboard.update_interval_seconds == 30
        assert dashboard.alert_thresholds['attention_entropy'] == 0.8
        assert dashboard.visualization_enabled is True
        assert hasattr(dashboard, 'health_history')
    
    def test_generate_comprehensive_health_report(self, mock_transformer_models, market_regime_data):
        """Test generation of comprehensive health report"""
        dashboard = TransformerHealthDashboard()
        
        # Mock comprehensive metrics
        mock_metrics = {
            'attention_health': AttentionHealthMetrics(
                mean_entropy=2.1,
                entropy_std=0.3,
                head_entropy_distribution=[2.0, 2.2, 1.9, 2.3, 2.1, 2.0, 2.2, 1.8],
                entropy_health_status='healthy'
            ),
            'performance_metrics': PerformanceMetrics(
                accuracy=0.87,
                precision=0.84,
                recall=0.89,
                f1_score=0.86,
                confidence_mean=0.83
            ),
            'resource_metrics': ResourceMetrics(
                memory_mb=950,
                cpu_percent=45,
                gpu_percent=78,
                inference_time_ms=65
            ),
            'embedding_metrics': EmbeddingMetrics(
                embedding_stability=0.92,
                position_encoding_health=0.95,
                semantic_consistency=0.88
            )
        }
        
        health_report = dashboard.generate_comprehensive_health_report(
            models=mock_transformer_models,
            current_metrics=mock_metrics,
            market_context='bull_market'
        )
        
        assert hasattr(health_report, 'overall_health_score')
        assert hasattr(health_report, 'model_health_breakdown')
        assert hasattr(health_report, 'critical_issues')
        assert hasattr(health_report, 'recommendations')
        assert hasattr(health_report, 'performance_trends')
        
        # Validate health score
        assert 0.0 <= health_report.overall_health_score <= 1.0
        assert len(health_report.model_health_breakdown) == len(mock_transformer_models)
    
    def test_real_time_health_monitoring(self, mock_transformer_models):
        """Test real-time health monitoring capabilities"""
        dashboard = TransformerHealthDashboard(
            real_time_monitoring=True,
            monitoring_frequency_seconds=1
        )
        
        # Start real-time monitoring
        dashboard.start_real_time_monitoring(mock_transformer_models)
        assert dashboard.is_monitoring is True
        
        # Collect health snapshots
        health_snapshots = []
        for _ in range(3):
            snapshot = dashboard.capture_health_snapshot()
            health_snapshots.append(snapshot)
            time.sleep(0.1)
        
        # Stop monitoring
        dashboard.stop_real_time_monitoring()
        assert dashboard.is_monitoring is False
        
        # Validate snapshots
        assert len(health_snapshots) == 3
        for snapshot in health_snapshots:
            assert hasattr(snapshot, 'timestamp')
            assert hasattr(snapshot, 'health_scores')
            assert hasattr(snapshot, 'alert_level')
            assert snapshot.alert_level in ['normal', 'warning', 'critical']
    
    def test_alert_generation_and_notification(self, mock_transformer_models):
        """Test health alert generation and notification"""
        dashboard = TransformerHealthDashboard(
            alert_enabled=True,
            notification_channels=['email', 'slack', 'webhook']
        )
        
        # Create critical health scenario
        critical_metrics = {
            'attention_health': AttentionHealthMetrics(
                mean_entropy=0.5,  # Very low - critical
                entropy_health_status='critical'
            ),
            'resource_metrics': ResourceMetrics(
                memory_mb=3000,  # Very high - critical
                gpu_percent=95   # Very high - critical
            )
        }
        
        alert_result = dashboard.generate_health_alerts(
            current_metrics=critical_metrics,
            model_type='iTransformer'
        )
        
        assert len(alert_result.alerts) > 0
        assert any(alert.severity == 'critical' for alert in alert_result.alerts)
        assert any('attention entropy' in alert.message.lower() for alert in alert_result.alerts)
        assert any('memory' in alert.message.lower() for alert in alert_result.alerts)
        
        # Test notification dispatch
        notification_result = dashboard.dispatch_notifications(alert_result.alerts)
        
        assert hasattr(notification_result, 'notifications_sent')
        assert hasattr(notification_result, 'failed_notifications')
        assert notification_result.notifications_sent > 0


class TestIntegrationWithExistingMonitoring:
    """Test integration with existing monitoring systems"""
    
    def test_cloud_monitoring_integration(self, mock_transformer_models):
        """Test integration with CloudMonitoringManager"""
        # Mock existing cloud monitoring
        cloud_monitor = MagicMock(spec=CloudMonitoringManager)
        
        # Create transformer metrics collector with cloud integration
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            cloud_monitoring=cloud_monitor,
            cloud_integration_enabled=True
        )
        
        # Collect metrics and send to cloud monitoring
        metrics_result = collector.collect_all_metrics()
        integration_result = collector.send_metrics_to_cloud(metrics_result)
        
        assert integration_result.success is True
        assert integration_result.metrics_sent > 0
        assert hasattr(integration_result, 'cloud_metric_ids')
        
        # Verify cloud monitoring calls
        assert cloud_monitor.send_custom_metric.called
        call_args = cloud_monitor.send_custom_metric.call_args_list
        assert len(call_args) > 0
    
    def test_operational_analytics_integration(self, mock_transformer_models):
        """Test integration with OperationalAnalyticsManager"""
        # Mock existing operational analytics
        analytics_manager = MagicMock(spec=OperationalAnalyticsManager)
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            operational_analytics=analytics_manager,
            analytics_integration_enabled=True
        )
        
        # Collect metrics and update analytics
        metrics_result = collector.collect_all_metrics()
        analytics_result = collector.update_operational_analytics(metrics_result)
        
        assert analytics_result.success is True
        assert hasattr(analytics_result, 'analytics_updated')
        assert hasattr(analytics_result, 'dashboard_updated')
        
        # Verify analytics manager calls
        assert analytics_manager.update_transformer_metrics.called
    
    def test_intelligent_alerting_integration(self, mock_transformer_models):
        """Test integration with intelligent alerting system"""
        from src.monitoring.intelligent_alerting import IntelligentAlertingSystem
        
        # Mock intelligent alerting
        alert_system = MagicMock(spec=IntelligentAlertingSystem)
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            intelligent_alerting=alert_system,
            auto_alerting_enabled=True
        )
        
        # Simulate critical metrics that should trigger alerts
        critical_metrics = TransformerMetricsResult(
            timestamp=datetime.now(),
            overall_health_status='critical',
            model_metrics={
                'iTransformer': MagicMock(health_status='critical'),
                'TimesFM': MagicMock(health_status='warning')
            }
        )
        
        alerting_result = collector.trigger_intelligent_alerts(critical_metrics)
        
        assert alerting_result.alerts_generated > 0
        assert hasattr(alerting_result, 'alert_priorities')
        assert hasattr(alerting_result, 'escalation_triggered')
        
        # Verify intelligent alerting calls
        assert alert_system.create_alert.called


class TestPerformanceAndScalability:
    """Test performance and scalability requirements"""
    
    def test_metrics_collection_latency(self, mock_transformer_models):
        """Test that metrics collection meets latency requirements"""
        collector = TransformerMetricsCollector(models=mock_transformer_models)
        
        # Test collection latency
        start_time = time.time()
        metrics_result = collector.collect_all_metrics()
        collection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance requirement: metrics collection should be fast
        assert collection_time < 500, f"Metrics collection took {collection_time:.2f}ms, should be <500ms"
        assert metrics_result is not None
    
    def test_concurrent_metrics_collection(self, mock_transformer_models):
        """Test concurrent metrics collection for multiple models"""
        import threading
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            concurrent_collection_enabled=True
        )
        
        results = {}
        threads = []
        
        def collect_metrics_for_model(model_type):
            try:
                result = collector.collect_model_specific_metrics(model_type)
                results[model_type] = result
            except Exception as e:
                results[model_type] = f"Error: {e}"
        
        # Start concurrent collection
        start_time = time.time()
        
        for model_type in mock_transformer_models.keys():
            thread = threading.Thread(target=collect_metrics_for_model, args=(model_type,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = (time.time() - start_time) * 1000
        
        # Performance requirement: concurrent collection should be efficient
        assert total_time < 1000, f"Concurrent collection took {total_time:.2f}ms, should be <1000ms"
        assert len(results) == len(mock_transformer_models)
        assert all(not isinstance(result, str) or not result.startswith("Error") 
                  for result in results.values())
    
    def test_memory_efficiency_metrics_collection(self, mock_transformer_models):
        """Test memory efficiency of metrics collection"""
        import psutil
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            memory_efficient_mode=True
        )
        
        # Perform multiple metrics collections
        for _ in range(20):
            metrics_result = collector.collect_all_metrics()
            # Force garbage collection to test memory efficiency
            import gc
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency requirement
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB, should be <50MB"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for transformer metrics"""
    
    def test_handle_corrupted_model_state(self, mock_transformer_models):
        """Test handling of corrupted model states"""
        collector = TransformerMetricsCollector(models=mock_transformer_models)
        
        # Corrupt model by removing required methods
        corrupted_model = mock_transformer_models['iTransformer']
        del corrupted_model.get_attention_weights
        
        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = collector.collect_model_specific_metrics('iTransformer')
        
        # Should return partial results or safe defaults
        assert result is not None
        assert hasattr(result, 'collection_errors')
        assert len(result.collection_errors) > 0
    
    def test_handle_missing_model_attributes(self, mock_transformer_models):
        """Test handling of missing model attributes"""
        collector = TransformerMetricsCollector(models=mock_transformer_models)
        
        # Create model with missing attributes
        incomplete_model = MagicMock()
        incomplete_model.model_type = 'IncompleteModel'
        # Missing get_attention_weights, get_embeddings, etc.
        
        mock_transformer_models['IncompleteModel'] = incomplete_model
        
        # Should handle gracefully
        result = collector.collect_model_specific_metrics('IncompleteModel')
        
        assert result is not None
        assert hasattr(result, 'partial_metrics')
        assert result.collection_status == 'partial'
    
    def test_handle_resource_constraints(self, mock_transformer_models):
        """Test handling of resource constraints"""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            resource_constraint_handling=True,
            max_memory_usage_mb=100  # Very low limit
        )
        
        # Should adapt to resource constraints
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = collector.collect_all_metrics()
        
        assert result is not None
        assert hasattr(result, 'resource_limited')
        assert result.resource_limited is True
    
    def test_handle_network_failures_cloud_integration(self, mock_transformer_models):
        """Test handling of network failures in cloud integration"""
        # Mock cloud monitoring with network failure
        cloud_monitor = MagicMock()
        cloud_monitor.send_custom_metric.side_effect = ConnectionError("Network unavailable")
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            cloud_monitoring=cloud_monitor,
            cloud_integration_enabled=True,
            failure_resilient=True
        )
        
        metrics_result = collector.collect_all_metrics()
        
        # Should not fail and should handle network error gracefully
        integration_result = collector.send_metrics_to_cloud(metrics_result)
        
        assert integration_result.success is False
        assert hasattr(integration_result, 'failure_reason')
        assert 'network' in integration_result.failure_reason.lower()
        assert hasattr(integration_result, 'local_backup_saved')
        assert integration_result.local_backup_saved is True