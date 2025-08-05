"""
Comprehensive failing integration tests for Phase 2.4.1 - Transformer Monitoring Integration

This module contains failing integration tests that define the expected behavior for 
transformer monitoring system integration. Following TDD methodology, these tests will 
fail initially as the implementation doesn't exist yet.

Test Coverage Areas:
- Integration with existing DriftDetector
- Automated alerts for drift events
- <1 hour detection time for significant drift
- Comprehensive model health monitoring
- Seamless integration with existing monitoring
- Integration with intelligent alerting system
- Operational analytics updates

Requirements:
- <1 hour detection time for significant drift
- Integration with existing monitoring systems
- Support for all transformer models (iTransformer, PatchTST, TimesMixer, TimesFM)
- Real-time drift detection capabilities
- Comprehensive model health monitoring
- Automated alert generation and escalation
- Performance monitoring with <100ms latency for critical checks
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import MagicMock, patch, AsyncMock
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing monitoring components
from src.monitoring.drift_detection import DriftDetector, DriftAnalysisResult, DriftSeverity, DriftType
from src.monitoring.cloud_monitoring import CloudMonitoringManager
from src.monitoring.intelligent_alerting import IntelligentAlertingSystem
from src.monitoring.operational_analytics import OperationalAnalyticsManager

# These imports will fail initially - this is expected for TDD
try:
    from src.monitoring.transformer_drift_detection import (
        TransformerDriftDetector,
        TransformerDriftResult,
        AttentionPatternDrift,
        TemporalFocusDrift
    )
    from src.monitoring.transformer_metrics import (
        TransformerMetricsCollector,
        TransformerMetricsResult,
        AttentionHealthMetrics,
        ResourceMetrics
    )
    from src.monitoring.transformer_monitoring_integration import (
        TransformerMonitoringIntegrator,
        IntegratedMonitoringSystem,
        MonitoringOrchestrator,
        AlertCoordinator,
        MetricsAggregator,
        MonitoringDashboardIntegrator,
        RealTimeMonitoringPipeline,
        MonitoringHealthChecker
    )
except ImportError:
    # Expected to fail initially - create placeholder classes for testing
    class TransformerDriftDetector:
        pass
    
    class TransformerMetricsCollector:
        pass
    
    class TransformerMonitoringIntegrator:
        pass
    
    class IntegratedMonitoringSystem:
        pass
    
    class MonitoringOrchestrator:
        pass
    
    class AlertCoordinator:
        pass
    
    class MetricsAggregator:
        pass
    
    class MonitoringDashboardIntegrator:
        pass
    
    class RealTimeMonitoringPipeline:
        pass
    
    class MonitoringHealthChecker:
        pass
    
    class TransformerDriftResult:
        pass
    
    class TransformerMetricsResult:
        pass
    
    class AttentionPatternDrift:
        pass
    
    class TemporalFocusDrift:
        pass
    
    class AttentionHealthMetrics:
        pass
    
    class ResourceMetrics:
        pass


class TestTransformerMonitoringIntegration:
    """Integration tests for transformer monitoring system"""
    
    @pytest.fixture
    def existing_monitoring_systems(self):
        """Create mock existing monitoring systems"""
        systems = {}
        
        # Mock DriftDetector
        drift_detector = MagicMock(spec=DriftDetector)
        drift_detector.is_fitted = True
        drift_detector.detect_drift.return_value = MagicMock(
            has_drift=False,
            overall_severity=DriftSeverity.NONE,
            overall_drift_score=0.05
        )
        systems['drift_detector'] = drift_detector
        
        # Mock CloudMonitoringManager
        cloud_monitor = MagicMock(spec=CloudMonitoringManager)
        cloud_monitor.send_custom_metric.return_value = {'success': True}
        cloud_monitor.create_alert_policy.return_value = {'policy_id': 'test_policy_123'}
        systems['cloud_monitoring'] = cloud_monitor
        
        # Mock IntelligentAlertingSystem
        alert_system = MagicMock(spec=IntelligentAlertingSystem)
        alert_system.create_alert.return_value = {'alert_id': 'alert_123', 'status': 'created'}
        alert_system.escalate_alert.return_value = {'escalation_id': 'escalation_456'}
        systems['intelligent_alerting'] = alert_system
        
        # Mock OperationalAnalyticsManager
        analytics_manager = MagicMock(spec=OperationalAnalyticsManager)
        analytics_manager.update_metrics.return_value = {'status': 'updated'}
        analytics_manager.generate_report.return_value = {'report_id': 'report_789'}
        systems['operational_analytics'] = analytics_manager
        
        return systems
    
    @pytest.fixture
    def mock_transformer_models(self):
        """Create mock transformer models for testing"""
        models = {}
        
        # Mock iTransformer
        itransformer = MagicMock()
        itransformer.model_type = 'iTransformer'
        itransformer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        itransformer.get_embeddings.return_value = torch.rand(100, 512)
        itransformer.get_prediction_confidence.return_value = torch.tensor([0.85, 0.92, 0.78, 0.96])
        itransformer.get_memory_usage.return_value = 850  # MB
        models['iTransformer'] = itransformer
        
        # Mock PatchTST
        patchtst = MagicMock()
        patchtst.model_type = 'PatchTST'
        patchtst.get_attention_weights.return_value = torch.rand(4, 8, 20, 20)
        patchtst.get_embeddings.return_value = torch.rand(20, 512)
        patchtst.get_prediction_confidence.return_value = torch.tensor([0.88, 0.79, 0.91, 0.84])
        patchtst.get_memory_usage.return_value = 650  # MB
        models['PatchTST'] = patchtst
        
        # Mock TimesMixer
        timesmixer = MagicMock()
        timesmixer.model_type = 'TimesMixer'
        timesmixer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        timesmixer.get_embeddings.return_value = torch.rand(100, 512)
        timesmixer.get_prediction_confidence.return_value = torch.tensor([0.82, 0.89, 0.76, 0.93])
        timesmixer.get_memory_usage.return_value = 750  # MB
        models['TimesMixer'] = timesmixer
        
        # Mock TimesFM
        timesfm = MagicMock()
        timesfm.model_type = 'TimesFM'
        timesfm.get_attention_weights.return_value = torch.rand(4, 12, 512, 512)
        timesfm.get_embeddings.return_value = torch.rand(512, 768)
        timesfm.get_prediction_confidence.return_value = torch.tensor([0.91, 0.87, 0.95, 0.83])
        timesfm.get_memory_usage.return_value = 1200  # MB
        models['TimesFM'] = timesfm
        
        return models
    
    @pytest.fixture
    def trading_data_stream(self):
        """Create realistic trading data stream for testing"""
        np.random.seed(42)
        
        # Generate 1000 hours of data (simulate real-time stream)
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='1H')
        
        # Create realistic crypto price movements with volatility regimes
        base_price = 45000
        price_changes = []
        volatility_regime = 'normal'
        
        for i in range(1000):
            # Change volatility regime occasionally
            if i % 200 == 0:
                volatility_regime = np.random.choice(['low', 'normal', 'high'], p=[0.3, 0.5, 0.2])
            
            if volatility_regime == 'low':
                change = np.random.normal(0, 0.01)
            elif volatility_regime == 'normal':
                change = np.random.normal(0, 0.02)
            else:  # high volatility
                change = np.random.normal(0, 0.05)
            
            price_changes.append(change)
        
        # Generate correlated prices
        btc_prices = [base_price]
        for change in price_changes:
            btc_prices.append(btc_prices[-1] * (1 + change))
        
        eth_prices = [btc_prices[i] * 0.065 + np.random.normal(0, 20) for i in range(len(btc_prices))]
        sol_prices = [btc_prices[i] * 0.0025 + np.random.normal(0, 2) for i in range(len(btc_prices))]
        
        volumes = np.random.lognormal(10, 1, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': btc_prices[:-1],  # Remove extra element
            'eth_price': eth_prices[:-1],
            'sol_price': sol_prices[:-1],
            'volume': volumes,
            'volatility_regime': [
                'high' if abs(price_changes[i]) > 0.03 else 
                'low' if abs(price_changes[i]) < 0.01 else 'normal'
                for i in range(len(price_changes))
            ]
        })


class TestMonitoringOrchestrator:
    """Test the main monitoring orchestrator"""
    
    def test_monitoring_orchestrator_initialization(self, existing_monitoring_systems, mock_transformer_models):
        """Test MonitoringOrchestrator initialization and system integration"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            integration_mode='comprehensive',
            orchestration_interval_seconds=30
        )
        
        # Expected initialization behavior
        assert orchestrator.transformer_models == mock_transformer_models
        assert orchestrator.existing_systems == existing_monitoring_systems
        assert orchestrator.integration_mode == 'comprehensive'
        assert orchestrator.orchestration_interval_seconds == 30
        
        # Should initialize integrated components
        assert hasattr(orchestrator, 'transformer_drift_detector')
        assert hasattr(orchestrator, 'transformer_metrics_collector')
        assert hasattr(orchestrator, 'alert_coordinator')
        assert hasattr(orchestrator, 'metrics_aggregator')
        assert hasattr(orchestrator, 'dashboard_integrator')
        assert hasattr(orchestrator, 'real_time_pipeline')
        
        # Should validate system compatibility
        assert orchestrator.is_system_compatible() is True
        assert orchestrator.integration_health_score > 0.8
    
    def test_comprehensive_monitoring_orchestration(self, existing_monitoring_systems, mock_transformer_models, trading_data_stream):
        """Test comprehensive monitoring orchestration"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems
        )
        
        # Start comprehensive monitoring
        orchestration_result = orchestrator.orchestrate_comprehensive_monitoring(
            trading_data=trading_data_stream.iloc[-100:],  # Recent data
            monitoring_duration_minutes=60
        )
        
        # Expected orchestration results
        assert orchestration_result.orchestration_status == 'active'
        assert orchestration_result.systems_monitored == len(mock_transformer_models) + len(existing_monitoring_systems)
        assert orchestration_result.alerts_generated >= 0
        assert orchestration_result.metrics_collected > 0
        assert orchestration_result.integration_health_score > 0.7
        
        # Should coordinate all monitoring components
        assert hasattr(orchestration_result, 'drift_detection_results')
        assert hasattr(orchestration_result, 'metrics_collection_results')
        assert hasattr(orchestration_result, 'alert_coordination_results')
        assert hasattr(orchestration_result, 'dashboard_update_results')
    
    def test_monitoring_system_health_check(self, existing_monitoring_systems, mock_transformer_models):
        """Test monitoring system health checking"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            health_check_enabled=True
        )
        
        # Perform comprehensive health check
        health_check_result = orchestrator.perform_system_health_check()
        
        assert health_check_result.overall_health_status in ['healthy', 'degraded', 'critical']
        assert hasattr(health_check_result, 'component_health')
        assert hasattr(health_check_result, 'integration_health')
        assert hasattr(health_check_result, 'performance_metrics')
        assert hasattr(health_check_result, 'recommendations')
        
        # Each monitored system should have health status
        for system_name in existing_monitoring_systems.keys():
            assert system_name in health_check_result.component_health
        
        for model_type in mock_transformer_models.keys():
            assert model_type in health_check_result.component_health


class TestIntegratedMonitoringSystem:
    """Test the integrated monitoring system"""
    
    def test_integrated_system_initialization(self, existing_monitoring_systems, mock_transformer_models):
        """Test IntegratedMonitoringSystem initialization"""
        integrated_system = IntegratedMonitoringSystem(
            transformer_models=mock_transformer_models,
            existing_monitoring=existing_monitoring_systems,
            integration_strategy='seamless',
            failover_enabled=True
        )
        
        assert integrated_system.transformer_models == mock_transformer_models
        assert integrated_system.integration_strategy == 'seamless'
        assert integrated_system.failover_enabled is True
        
        # Should create unified monitoring interface
        assert hasattr(integrated_system, 'unified_drift_detection')
        assert hasattr(integrated_system, 'unified_metrics_collection')
        assert hasattr(integrated_system, 'unified_alerting')
        assert hasattr(integrated_system, 'fallback_systems')
    
    def test_seamless_drift_detection_integration(self, existing_monitoring_systems, mock_transformer_models, trading_data_stream):
        """Test seamless integration with existing DriftDetector"""
        integrated_system = IntegratedMonitoringSystem(
            transformer_models=mock_transformer_models,
            existing_monitoring=existing_monitoring_systems
        )
        
        # Test unified drift detection
        current_data = trading_data_stream.iloc[-50:].copy()
        current_data['volume'] *= 1.5  # Introduce some drift
        
        # Mock attention weights showing drift
        current_attention = {
            'iTransformer': torch.rand(4, 8, 100, 100) * 0.8,  # Reduced attention entropy
            'PatchTST': torch.rand(4, 8, 20, 20),
            'TimesMixer': torch.rand(4, 8, 100, 100),
            'TimesFM': torch.rand(4, 12, 512, 512)
        }
        
        unified_drift_result = integrated_system.detect_unified_drift(
            trading_data=current_data,
            attention_weights=current_attention
        )
        
        # Should combine traditional and transformer-specific drift detection
        assert hasattr(unified_drift_result, 'traditional_drift')
        assert hasattr(unified_drift_result, 'transformer_drift')
        assert hasattr(unified_drift_result, 'combined_drift_score')
        assert hasattr(unified_drift_result, 'drift_consensus')
        
        # Should provide unified drift assessment
        assert 0.0 <= unified_drift_result.combined_drift_score <= 1.0
        assert unified_drift_result.drift_consensus in ['no_drift', 'possible_drift', 'confirmed_drift']
    
    def test_detection_time_within_one_hour(self, existing_monitoring_systems, mock_transformer_models, trading_data_stream):
        """Test that significant drift is detected within 1 hour (success criteria)"""
        integrated_system = IntegratedMonitoringSystem(
            transformer_models=mock_transformer_models,
            existing_monitoring=existing_monitoring_systems,
            detection_sensitivity='high',
            monitoring_frequency_minutes=1
        )
        
        # Simulate 1 hour of monitoring with gradual drift introduction
        detection_results = []
        drift_detected_at_minute = None
        
        for minute in range(60):  # 60 minutes = 1 hour
            # Introduce gradual drift
            drift_factor = minute / 60.0
            
            # Create drifted trading data
            current_data = trading_data_stream.iloc[minute:minute+10].copy()
            if drift_factor > 0.3:  # Start introducing drift after 18 minutes
                current_data['volume'] *= (1 + drift_factor)
                current_data['btc_price'] *= (1 + drift_factor * 0.1)
            
            # Create drifted attention patterns
            current_attention = {}
            for model_type, model in mock_transformer_models.items():
                baseline_attention = model.get_attention_weights()
                if drift_factor > 0.3:
                    # Introduce attention drift (more concentrated attention)
                    drifted_attention = torch.softmax(baseline_attention * (1 + drift_factor * 2), dim=-1)
                else:
                    drifted_attention = baseline_attention
                current_attention[model_type] = drifted_attention
            
            # Detect drift
            drift_result = integrated_system.detect_unified_drift(
                trading_data=current_data,
                attention_weights=current_attention
            )
            
            detection_results.append((minute, drift_result))
            
            # Check if significant drift detected
            if (drift_result.combined_drift_score > 0.7 and 
                drift_result.drift_consensus == 'confirmed_drift' and
                drift_detected_at_minute is None):
                drift_detected_at_minute = minute
                break
        
        # Success criteria: detect significant drift within 1 hour (60 minutes)
        assert drift_detected_at_minute is not None, "Significant drift should be detected within 1 hour"
        assert drift_detected_at_minute < 60, f"Drift detected at minute {drift_detected_at_minute}, should be <60"
        
        # Validate detection quality
        final_result = detection_results[drift_detected_at_minute][1]
        assert final_result.combined_drift_score > 0.7
        assert final_result.drift_consensus == 'confirmed_drift'
    
    def test_system_failover_functionality(self, existing_monitoring_systems, mock_transformer_models):
        """Test system failover when components fail"""
        integrated_system = IntegratedMonitoringSystem(
            transformer_models=mock_transformer_models,
            existing_monitoring=existing_monitoring_systems,
            failover_enabled=True,
            redundancy_level='high'
        )
        
        # Simulate component failure
        existing_monitoring_systems['cloud_monitoring'].send_custom_metric.side_effect = Exception("Service unavailable")
        
        # System should failover gracefully
        failover_result = integrated_system.handle_component_failure(
            failed_component='cloud_monitoring',
            failure_type='service_unavailable'
        )
        
        assert failover_result.failover_successful is True
        assert failover_result.backup_systems_activated > 0
        assert hasattr(failover_result, 'fallback_strategy')
        assert hasattr(failover_result, 'recovery_plan')
        
        # Should continue monitoring with reduced functionality
        monitoring_result = integrated_system.continue_monitoring_with_failover()
        assert monitoring_result.monitoring_active is True
        assert monitoring_result.functionality_level in ['full', 'reduced', 'minimal']


class TestAlertCoordinator:
    """Test alert coordination and escalation"""
    
    def test_alert_coordinator_initialization(self, existing_monitoring_systems):
        """Test AlertCoordinator initialization"""
        coordinator = AlertCoordinator(
            existing_alerting=existing_monitoring_systems['intelligent_alerting'],
            cloud_monitoring=existing_monitoring_systems['cloud_monitoring'],
            escalation_thresholds={
                'attention_drift': 0.8,
                'performance_degradation': 0.7,
                'resource_exhaustion': 0.9
            },
            coordination_strategy='intelligent'
        )
        
        assert coordinator.escalation_thresholds['attention_drift'] == 0.8
        assert coordinator.coordination_strategy == 'intelligent'
        assert hasattr(coordinator, 'alert_aggregator')
        assert hasattr(coordinator, 'escalation_manager')
        assert hasattr(coordinator, 'notification_dispatcher')
    
    def test_coordinated_alert_generation(self, existing_monitoring_systems, mock_transformer_models):
        """Test coordinated alert generation from multiple sources"""
        coordinator = AlertCoordinator(
            existing_alerting=existing_monitoring_systems['intelligent_alerting'],
            cloud_monitoring=existing_monitoring_systems['cloud_monitoring']
        )
        
        # Simulate multiple alert sources
        transformer_alerts = [
            {
                'source': 'iTransformer',
                'type': 'attention_drift',
                'severity': 'high',
                'confidence': 0.85,
                'message': 'Attention entropy decreased by 40%'
            },
            {
                'source': 'TimesFM',
                'type': 'resource_usage',
                'severity': 'medium',
                'confidence': 0.75,
                'message': 'Memory usage increased to 1.5GB'
            }
        ]
        
        traditional_alerts = [
            {
                'source': 'feature_drift',
                'type': 'data_drift',
                'severity': 'medium',
                'confidence': 0.70,
                'message': 'Volume feature showing drift'
            }
        ]
        
        coordination_result = coordinator.coordinate_alerts(
            transformer_alerts=transformer_alerts,
            traditional_alerts=traditional_alerts
        )
        
        # Should intelligently coordinate and prioritize alerts
        assert hasattr(coordination_result, 'coordinated_alerts')
        assert hasattr(coordination_result, 'alert_priorities')
        assert hasattr(coordination_result, 'escalation_decisions')
        assert hasattr(coordination_result, 'notification_plan')
        
        # Should prioritize high-confidence, high-severity alerts
        high_priority_alerts = [alert for alert in coordination_result.coordinated_alerts 
                              if alert.priority == 'high']
        assert len(high_priority_alerts) > 0
    
    def test_intelligent_alert_deduplication(self, existing_monitoring_systems):
        """Test intelligent alert deduplication and aggregation"""
        coordinator = AlertCoordinator(
            existing_alerting=existing_monitoring_systems['intelligent_alerting'],
            deduplication_enabled=True,
            aggregation_window_minutes=5
        )
        
        # Simulate similar alerts from different sources
        similar_alerts = [
            {
                'source': 'iTransformer',
                'type': 'attention_drift',
                'message': 'Attention patterns changed significantly',
                'timestamp': datetime.now()
            },
            {
                'source': 'PatchTST', 
                'type': 'attention_drift',
                'message': 'Attention distribution drift detected',
                'timestamp': datetime.now()
            },
            {
                'source': 'drift_detector',
                'type': 'feature_drift',
                'message': 'Feature importance changed',
                'timestamp': datetime.now()
            }
        ]
        
        deduplication_result = coordinator.deduplicate_and_aggregate_alerts(similar_alerts)
        
        assert hasattr(deduplication_result, 'deduplicated_alerts')
        assert hasattr(deduplication_result, 'aggregated_alerts')
        assert hasattr(deduplication_result, 'duplicate_count')
        
        # Should reduce similar alerts to aggregated ones
        assert len(deduplication_result.deduplicated_alerts) < len(similar_alerts)
        assert deduplication_result.duplicate_count > 0
    
    def test_escalation_management(self, existing_monitoring_systems):
        """Test alert escalation management"""
        coordinator = AlertCoordinator(
            existing_alerting=existing_monitoring_systems['intelligent_alerting'],
            escalation_enabled=True,
            escalation_levels=['team', 'manager', 'executive']
        )
        
        # Simulate critical alert requiring escalation
        critical_alert = {
            'id': 'critical_alert_001',
            'severity': 'critical',
            'confidence': 0.95,
            'type': 'system_failure',
            'message': 'Multiple transformer models showing severe drift',
            'impact': 'trading_performance_degradation',
            'estimated_loss': 50000  # USD
        }
        
        escalation_result = coordinator.manage_alert_escalation(critical_alert)
        
        assert hasattr(escalation_result, 'escalation_path')
        assert hasattr(escalation_result, 'notifications_sent')
        assert hasattr(escalation_result, 'escalation_timeline')
        assert hasattr(escalation_result, 'response_requirements')
        
        # Critical alerts should escalate to highest levels
        assert 'executive' in escalation_result.escalation_path
        assert escalation_result.notifications_sent > 0


class TestMetricsAggregator:
    """Test metrics aggregation across monitoring systems"""
    
    def test_metrics_aggregator_initialization(self, existing_monitoring_systems):
        """Test MetricsAggregator initialization"""
        aggregator = MetricsAggregator(
            existing_analytics=existing_monitoring_systems['operational_analytics'],
            cloud_monitoring=existing_monitoring_systems['cloud_monitoring'],
            aggregation_strategy='comprehensive',
            retention_policy_days=30
        )
        
        assert aggregator.aggregation_strategy == 'comprehensive'
        assert aggregator.retention_policy_days == 30
        assert hasattr(aggregator, 'metric_processors')
        assert hasattr(aggregator, 'data_warehouse')
        assert hasattr(aggregator, 'visualization_engine')
    
    def test_comprehensive_metrics_aggregation(self, existing_monitoring_systems, mock_transformer_models):
        """Test comprehensive metrics aggregation from all sources"""
        aggregator = MetricsAggregator(
            existing_analytics=existing_monitoring_systems['operational_analytics'],
            cloud_monitoring=existing_monitoring_systems['cloud_monitoring']
        )
        
        # Mock metrics from different sources
        transformer_metrics = {
            'iTransformer': {
                'attention_entropy': 2.1,
                'prediction_confidence': 0.85,
                'memory_usage_mb': 850,
                'inference_time_ms': 45
            },
            'TimesFM': {
                'attention_entropy': 1.9,
                'prediction_confidence': 0.88,
                'memory_usage_mb': 1200,
                'inference_time_ms': 78
            }
        }
        
        traditional_metrics = {
            'drift_detection': {
                'feature_drift_score': 0.15,
                'concept_drift_detected': False,
                'detection_latency_ms': 23
            },
            'system_performance': {
                'cpu_usage_percent': 45,
                'memory_usage_percent': 67,
                'gpu_utilization_percent': 82
            }
        }
        
        aggregation_result = aggregator.aggregate_comprehensive_metrics(
            transformer_metrics=transformer_metrics,
            traditional_metrics=traditional_metrics,
            timestamp=datetime.now()
        )
        
        # Should create unified metrics view
        assert hasattr(aggregation_result, 'unified_metrics')
        assert hasattr(aggregation_result, 'metric_correlations')
        assert hasattr(aggregation_result, 'performance_indicators')
        assert hasattr(aggregation_result, 'trend_analysis')
        
        # Should calculate cross-system correlations
        assert hasattr(aggregation_result.metric_correlations, 'attention_performance_correlation')
        assert hasattr(aggregation_result.metric_correlations, 'resource_efficiency_correlation')
    
    def test_real_time_metrics_streaming(self, existing_monitoring_systems, mock_transformer_models):
        """Test real-time metrics streaming and aggregation"""
        aggregator = MetricsAggregator(
            existing_analytics=existing_monitoring_systems['operational_analytics'],
            real_time_streaming=True,
            stream_buffer_size=100
        )
        
        # Start streaming
        streaming_result = aggregator.start_real_time_streaming()
        assert streaming_result.streaming_active is True
        
        # Simulate metrics stream
        metrics_stream = []
        for i in range(10):
            metric_point = {
                'timestamp': datetime.now() - timedelta(seconds=i),
                'source': 'iTransformer',
                'metrics': {
                    'attention_entropy': 2.0 + np.random.normal(0, 0.1),
                    'confidence': 0.85 + np.random.normal(0, 0.05),
                    'memory_mb': 850 + np.random.normal(0, 20)
                }
            }
            metrics_stream.append(metric_point)
        
        # Process streaming metrics
        stream_processing_result = aggregator.process_metrics_stream(metrics_stream)
        
        assert hasattr(stream_processing_result, 'processed_count')
        assert hasattr(stream_processing_result, 'aggregated_metrics')
        assert hasattr(stream_processing_result, 'anomalies_detected')
        assert hasattr(stream_processing_result, 'trend_updates')
        
        assert stream_processing_result.processed_count == len(metrics_stream)
        
        # Stop streaming
        stop_result = aggregator.stop_real_time_streaming()
        assert stop_result.streaming_active is False
    
    def test_cross_system_correlation_analysis(self, existing_monitoring_systems):
        """Test cross-system correlation analysis"""
        aggregator = MetricsAggregator(
            existing_analytics=existing_monitoring_systems['operational_analytics'],
            correlation_analysis_enabled=True
        )
        
        # Historical metrics for correlation analysis
        historical_data = {
            'timestamps': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'attention_entropy': np.random.normal(2.0, 0.2, 100),
            'prediction_accuracy': np.random.normal(0.85, 0.05, 100),
            'system_latency': np.random.normal(50, 10, 100),
            'memory_usage': np.random.normal(800, 100, 100)
        }
        
        # Add some correlations
        for i in range(100):
            # Lower attention entropy correlates with lower accuracy
            if historical_data['attention_entropy'][i] < 1.8:
                historical_data['prediction_accuracy'][i] *= 0.9
            
            # Higher memory usage correlates with higher latency
            if historical_data['memory_usage'][i] > 900:
                historical_data['system_latency'][i] *= 1.2
        
        correlation_result = aggregator.analyze_cross_system_correlations(historical_data)
        
        assert hasattr(correlation_result, 'correlation_matrix')
        assert hasattr(correlation_result, 'significant_correlations')
        assert hasattr(correlation_result, 'causal_relationships')
        assert hasattr(correlation_result, 'predictive_indicators')
        
        # Should detect engineered correlations
        significant_corrs = correlation_result.significant_correlations
        assert any('attention_entropy' in corr and 'prediction_accuracy' in corr 
                  for corr in significant_corrs)
        assert any('memory_usage' in corr and 'system_latency' in corr 
                  for corr in significant_corrs)


class TestRealTimeMonitoringPipeline:
    """Test real-time monitoring pipeline"""
    
    def test_real_time_pipeline_initialization(self, existing_monitoring_systems, mock_transformer_models):
        """Test RealTimeMonitoringPipeline initialization"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            pipeline_latency_target_ms=100,
            parallel_processing=True,
            buffer_size=1000
        )
        
        assert pipeline.pipeline_latency_target_ms == 100
        assert pipeline.parallel_processing is True
        assert pipeline.buffer_size == 1000
        assert hasattr(pipeline, 'data_ingestion_stage')
        assert hasattr(pipeline, 'processing_stage')
        assert hasattr(pipeline, 'analysis_stage')
        assert hasattr(pipeline, 'alert_stage')
        assert hasattr(pipeline, 'output_stage')
    
    def test_real_time_data_ingestion(self, existing_monitoring_systems, mock_transformer_models, trading_data_stream):
        """Test real-time data ingestion and processing"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems
        )
        
        # Start pipeline
        pipeline_start_result = pipeline.start_pipeline()
        assert pipeline_start_result.pipeline_active is True
        
        # Simulate real-time data ingestion
        ingestion_results = []
        
        for i in range(10):  # 10 data points
            # Create real-time data point
            data_point = {
                'timestamp': datetime.now(),
                'trading_data': trading_data_stream.iloc[i:i+1].to_dict('records')[0],
                'attention_weights': {
                    model_type: model.get_attention_weights()
                    for model_type, model in mock_transformer_models.items()
                },
                'system_metrics': {
                    'cpu_percent': 45 + np.random.normal(0, 5),
                    'memory_percent': 65 + np.random.normal(0, 10),
                    'gpu_percent': 80 + np.random.normal(0, 8)
                }
            }
            
            # Ingest data point
            ingestion_result = pipeline.ingest_real_time_data(data_point)
            ingestion_results.append(ingestion_result)
        
        # Validate ingestion results
        assert len(ingestion_results) == 10
        for result in ingestion_results:
            assert hasattr(result, 'ingestion_latency_ms')
            assert hasattr(result, 'processing_queued')
            assert hasattr(result, 'buffer_utilization')
            
            # Performance requirement: <100ms ingestion latency
            assert result.ingestion_latency_ms < 100
        
        # Stop pipeline
        pipeline_stop_result = pipeline.stop_pipeline()
        assert pipeline_stop_result.pipeline_active is False
    
    def test_parallel_processing_performance(self, existing_monitoring_systems, mock_transformer_models):
        """Test parallel processing performance"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            parallel_processing=True,
            worker_threads=4
        )
        
        # Create batch of data for parallel processing
        data_batch = []
        for i in range(20):
            data_point = {
                'id': i,
                'timestamp': datetime.now(),
                'attention_weights': {
                    'iTransformer': torch.rand(1, 8, 100, 100),
                    'TimesFM': torch.rand(1, 12, 512, 512)
                },
                'complexity': 'high' if i % 3 == 0 else 'medium'
            }
            data_batch.append(data_point)
        
        # Process batch in parallel
        start_time = time.time()
        parallel_result = pipeline.process_batch_parallel(data_batch)
        parallel_time = (time.time() - start_time) * 1000  # ms
        
        # Compare with serial processing
        start_time = time.time()
        serial_result = pipeline.process_batch_serial(data_batch)
        serial_time = (time.time() - start_time) * 1000  # ms
        
        # Parallel should be faster for large batches
        assert parallel_time < serial_time * 0.8  # At least 20% improvement
        assert parallel_result.processed_count == len(data_batch)
        assert serial_result.processed_count == len(data_batch)
        
        # Quality should be maintained
        assert parallel_result.processing_quality_score > 0.95
        assert abs(parallel_result.processing_quality_score - serial_result.processing_quality_score) < 0.05
    
    def test_pipeline_bottleneck_detection(self, existing_monitoring_systems, mock_transformer_models):
        """Test pipeline bottleneck detection and optimization"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            bottleneck_detection_enabled=True,
            optimization_enabled=True
        )
        
        # Simulate pipeline with bottleneck
        pipeline_stages = {
            'ingestion': {'target_ms': 10, 'actual_ms': 8},
            'preprocessing': {'target_ms': 20, 'actual_ms': 18},
            'analysis': {'target_ms': 50, 'actual_ms': 120},  # Bottleneck
            'alerting': {'target_ms': 15, 'actual_ms': 12},
            'output': {'target_ms': 5, 'actual_ms': 4}
        }
        
        bottleneck_result = pipeline.detect_and_optimize_bottlenecks(pipeline_stages)
        
        assert hasattr(bottleneck_result, 'bottlenecks_detected')
        assert hasattr(bottleneck_result, 'optimization_applied')
        assert hasattr(bottleneck_result, 'performance_improvement')
        assert hasattr(bottleneck_result, 'recommended_actions')
        
        # Should detect analysis stage as bottleneck
        assert 'analysis' in bottleneck_result.bottlenecks_detected
        assert bottleneck_result.optimization_applied is True
        assert bottleneck_result.performance_improvement > 0


class TestMonitoringDashboardIntegrator:
    """Test monitoring dashboard integration"""
    
    def test_dashboard_integrator_initialization(self, existing_monitoring_systems):
        """Test MonitoringDashboardIntegrator initialization"""
        integrator = MonitoringDashboardIntegrator(
            existing_systems=existing_monitoring_systems,
            dashboard_type='comprehensive',
            update_frequency_seconds=30,
            visualization_enabled=True
        )
        
        assert integrator.dashboard_type == 'comprehensive'
        assert integrator.update_frequency_seconds == 30
        assert integrator.visualization_enabled is True
        assert hasattr(integrator, 'dashboard_components')
        assert hasattr(integrator, 'visualization_engine')
        assert hasattr(integrator, 'real_time_updater')
    
    def test_transformer_monitoring_dashboard_creation(self, existing_monitoring_systems, mock_transformer_models):
        """Test creation of transformer monitoring dashboard"""
        integrator = MonitoringDashboardIntegrator(
            existing_systems=existing_monitoring_systems
        )
        
        # Create comprehensive dashboard
        dashboard_result = integrator.create_transformer_monitoring_dashboard(
            transformer_models=mock_transformer_models,
            dashboard_layout='grid',
            interactive_features=['drill_down', 'filtering', 'alerting']
        )
        
        assert hasattr(dashboard_result, 'dashboard_id')
        assert hasattr(dashboard_result, 'dashboard_components')
        assert hasattr(dashboard_result, 'data_sources')
        assert hasattr(dashboard_result, 'update_endpoints')
        
        # Should include components for each transformer model
        components = dashboard_result.dashboard_components
        assert 'attention_health_panel' in components
        assert 'performance_metrics_panel' in components
        assert 'resource_usage_panel' in components
        assert 'drift_detection_panel' in components
        assert 'alert_management_panel' in components
        
        # Should support all transformer models
        for model_type in mock_transformer_models.keys():
            assert f'{model_type}_specific_panel' in components
    
    def test_real_time_dashboard_updates(self, existing_monitoring_systems, mock_transformer_models):
        """Test real-time dashboard updates"""
        integrator = MonitoringDashboardIntegrator(
            existing_systems=existing_monitoring_systems,
            real_time_updates=True,
            update_frequency_seconds=5
        )
        
        # Create dashboard
        dashboard_result = integrator.create_transformer_monitoring_dashboard(
            transformer_models=mock_transformer_models
        )
        
        # Start real-time updates
        update_start_result = integrator.start_real_time_updates(dashboard_result.dashboard_id)
        assert update_start_result.updates_active is True
        
        # Simulate real-time data changes
        for i in range(5):
            # Mock new metrics
            new_metrics = {
                'timestamp': datetime.now(),
                'iTransformer': {
                    'attention_entropy': 2.0 + i * 0.1,
                    'confidence': 0.85 - i * 0.01,
                    'memory_mb': 850 + i * 10
                },
                'system_health': {
                    'overall_score': 0.9 - i * 0.02,
                    'alerts_active': i
                }
            }
            
            # Update dashboard
            update_result = integrator.update_dashboard_real_time(
                dashboard_id=dashboard_result.dashboard_id,
                new_metrics=new_metrics
            )
            
            assert update_result.update_successful is True
            assert update_result.update_latency_ms < 100  # Should be fast
            
            time.sleep(0.1)  # Small delay between updates
        
        # Stop real-time updates
        update_stop_result = integrator.stop_real_time_updates(dashboard_result.dashboard_id)
        assert update_stop_result.updates_active is False
    
    def test_dashboard_alert_integration(self, existing_monitoring_systems):
        """Test dashboard integration with alerting system"""
        integrator = MonitoringDashboardIntegrator(
            existing_systems=existing_monitoring_systems,
            alert_integration_enabled=True
        )
        
        # Create alert-enabled dashboard
        dashboard_result = integrator.create_transformer_monitoring_dashboard(
            transformer_models={'iTransformer': MagicMock()},
            alert_features=['visual_alerts', 'sound_alerts', 'email_integration']
        )
        
        # Test alert visualization
        test_alert = {
            'id': 'test_alert_001',
            'severity': 'high',
            'type': 'attention_drift',
            'message': 'Significant attention pattern drift detected',
            'source': 'iTransformer',
            'timestamp': datetime.now()
        }
        
        alert_integration_result = integrator.integrate_alert_with_dashboard(
            dashboard_id=dashboard_result.dashboard_id,
            alert=test_alert
        )
        
        assert alert_integration_result.alert_displayed is True
        assert hasattr(alert_integration_result, 'visualization_type')
        assert hasattr(alert_integration_result, 'user_notification_sent')
        assert hasattr(alert_integration_result, 'alert_persistence')
        
        # High severity alerts should be prominently displayed
        assert alert_integration_result.visualization_type in ['popup', 'banner', 'highlight']
        assert alert_integration_result.user_notification_sent is True


class TestPerformanceAndScalability:
    """Test performance and scalability of integrated monitoring"""
    
    def test_monitoring_latency_requirements(self, existing_monitoring_systems, mock_transformer_models):
        """Test that monitoring meets latency requirements"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            performance_mode='optimized'
        )
        
        # Test critical monitoring latency
        start_time = time.time()
        
        # Perform comprehensive monitoring cycle
        monitoring_result = orchestrator.perform_monitoring_cycle()
        
        total_latency = (time.time() - start_time) * 1000  # ms
        
        # Performance requirement: <100ms for critical monitoring checks
        assert total_latency < 100, f"Monitoring cycle took {total_latency:.2f}ms, should be <100ms"
        assert monitoring_result is not None
        assert monitoring_result.cycle_successful is True
    
    def test_concurrent_model_monitoring(self, existing_monitoring_systems, mock_transformer_models):
        """Test concurrent monitoring of multiple transformer models"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            concurrent_monitoring=True,
            max_concurrent_workers=4
        )
        
        # Test concurrent monitoring
        start_time = time.time()
        
        concurrent_result = orchestrator.monitor_all_models_concurrent()
        
        concurrent_time = (time.time() - start_time) * 1000  # ms
        
        # Compare with sequential monitoring
        start_time = time.time()
        sequential_result = orchestrator.monitor_all_models_sequential()
        sequential_time = (time.time() - start_time) * 1000  # ms
        
        # Concurrent should be faster for multiple models
        assert concurrent_time < sequential_time * 0.7  # At least 30% improvement
        assert concurrent_result.models_monitored == len(mock_transformer_models)
        assert sequential_result.models_monitored == len(mock_transformer_models)
        
        # Quality should be maintained
        assert concurrent_result.monitoring_quality_score > 0.95
    
    def test_high_frequency_monitoring_stability(self, existing_monitoring_systems, mock_transformer_models):
        """Test stability under high-frequency monitoring"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            high_frequency_mode=True,
            monitoring_interval_ms=100  # 10 Hz
        )
        
        # Start high-frequency monitoring
        pipeline.start_pipeline()
        
        monitoring_results = []
        start_time = time.time()
        
        # Run for 10 seconds at high frequency
        while (time.time() - start_time) < 10:
            result = pipeline.perform_monitoring_cycle()
            monitoring_results.append(result)
            time.sleep(0.1)  # 100ms intervals
        
        pipeline.stop_pipeline()
        
        # Validate stability
        assert len(monitoring_results) >= 90  # Should complete ~100 cycles
        
        # Check for performance degradation
        early_results = monitoring_results[:10]
        late_results = monitoring_results[-10:]
        
        early_avg_latency = np.mean([r.cycle_latency_ms for r in early_results])
        late_avg_latency = np.mean([r.cycle_latency_ms for r in late_results])
        
        # Latency should remain stable (no significant degradation)
        assert late_avg_latency < early_avg_latency * 1.2  # Max 20% degradation
        
        # Success rate should remain high
        success_rate = sum(1 for r in monitoring_results if r.cycle_successful) / len(monitoring_results)
        assert success_rate > 0.98  # >98% success rate
    
    def test_memory_efficiency_under_load(self, existing_monitoring_systems, mock_transformer_models):
        """Test memory efficiency under monitoring load"""
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            memory_efficient_mode=True
        )
        
        # Perform intensive monitoring
        for i in range(100):
            monitoring_result = orchestrator.perform_comprehensive_monitoring()
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency requirement
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, should be <100MB"


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""
    
    def test_component_failure_recovery(self, existing_monitoring_systems, mock_transformer_models):
        """Test recovery from component failures"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            fault_tolerance_enabled=True,
            auto_recovery_enabled=True
        )
        
        # Simulate component failure
        existing_monitoring_systems['cloud_monitoring'].send_custom_metric.side_effect = Exception("Service failed")
        
        # Orchestrator should handle failure gracefully
        recovery_result = orchestrator.handle_component_failure(
            component='cloud_monitoring',
            failure_exception=Exception("Service failed")
        )
        
        assert recovery_result.recovery_successful is True
        assert hasattr(recovery_result, 'fallback_activated')
        assert hasattr(recovery_result, 'service_level_maintained')
        assert hasattr(recovery_result, 'estimated_recovery_time')
        
        # Should continue monitoring with degraded functionality
        continued_monitoring = orchestrator.continue_monitoring_after_failure()
        assert continued_monitoring.monitoring_active is True
        assert continued_monitoring.functionality_level in ['degraded', 'minimal']
    
    def test_data_corruption_handling(self, existing_monitoring_systems, mock_transformer_models):
        """Test handling of corrupted monitoring data"""
        pipeline = RealTimeMonitoringPipeline(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            data_validation_enabled=True,
            corruption_recovery_enabled=True
        )
        
        # Create corrupted data
        corrupted_data = {
            'timestamp': 'invalid_timestamp',
            'attention_weights': {
                'iTransformer': torch.full((4, 8, 100, 100), float('nan'))  # NaN values
            },
            'system_metrics': {
                'cpu_percent': -50,  # Invalid negative CPU
                'memory_percent': 150  # Invalid >100% memory
            }
        }
        
        # Pipeline should handle corrupted data gracefully
        corruption_handling_result = pipeline.handle_corrupted_data(corrupted_data)
        
        assert corruption_handling_result.corruption_detected is True
        assert hasattr(corruption_handling_result, 'data_cleaned')
        assert hasattr(corruption_handling_result, 'fallback_data_used')
        assert hasattr(corruption_handling_result, 'data_quality_score')
        
        # Should continue with cleaned/fallback data
        assert corruption_handling_result.monitoring_continued is True
        assert corruption_handling_result.data_quality_score > 0.0
    
    def test_network_failure_resilience(self, existing_monitoring_systems, mock_transformer_models):
        """Test resilience to network failures"""
        integrator = MonitoringDashboardIntegrator(
            existing_systems=existing_monitoring_systems,
            network_resilience_enabled=True,
            offline_mode_enabled=True
        )
        
        # Simulate network failure
        for system in existing_monitoring_systems.values():
            for method in dir(system):
                if method.startswith('send') or method.startswith('post'):
                    getattr(system, method).side_effect = ConnectionError("Network unavailable")
        
        # Should switch to offline mode
        offline_result = integrator.switch_to_offline_mode()
        
        assert offline_result.offline_mode_active is True
        assert hasattr(offline_result, 'local_storage_enabled')
        assert hasattr(offline_result, 'data_queuing_enabled')
        assert hasattr(offline_result, 'sync_plan')
        
        # Should continue monitoring locally
        local_monitoring_result = integrator.perform_offline_monitoring(
            transformer_models=mock_transformer_models
        )
        
        assert local_monitoring_result.monitoring_active is True
        assert local_monitoring_result.data_stored_locally is True
        
        # Should queue data for later sync
        assert hasattr(local_monitoring_result, 'queued_data_count')
        assert local_monitoring_result.queued_data_count > 0
    
    def test_cascading_failure_prevention(self, existing_monitoring_systems, mock_transformer_models):
        """Test prevention of cascading failures"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            cascade_prevention_enabled=True,
            circuit_breaker_enabled=True
        )
        
        # Simulate multiple component failures
        failure_cascade = [
            ('cloud_monitoring', ConnectionError("Cloud service down")),
            ('intelligent_alerting', TimeoutError("Alert service timeout")),
            ('operational_analytics', RuntimeError("Analytics service error"))
        ]
        
        cascade_result = orchestrator.handle_potential_cascade(failure_cascade)
        
        assert cascade_result.cascade_prevented is True
        assert hasattr(cascade_result, 'circuit_breakers_activated')
        assert hasattr(cascade_result, 'isolation_boundaries')
        assert hasattr(cascade_result, 'minimal_service_maintained')
        
        # Should maintain core monitoring functionality
        assert cascade_result.minimal_service_maintained is True
        assert len(cascade_result.circuit_breakers_activated) == len(failure_cascade)
        
        # Should isolate failures to prevent spread
        core_monitoring_result = orchestrator.perform_core_monitoring_only()
        assert core_monitoring_result.monitoring_active is True
        assert core_monitoring_result.critical_functions_operational is True


class TestComplianceAndAuditing:
    """Test compliance and auditing features"""
    
    def test_monitoring_audit_trail(self, existing_monitoring_systems, mock_transformer_models):
        """Test comprehensive monitoring audit trail"""
        orchestrator = MonitoringOrchestrator(
            transformer_models=mock_transformer_models,
            existing_systems=existing_monitoring_systems,
            audit_trail_enabled=True,
            compliance_mode='strict'
        )
        
        # Perform monitored activities
        activities = [
            orchestrator.detect_drift(),
            orchestrator.collect_metrics(),
            orchestrator.generate_alerts(),
            orchestrator.update_dashboards()
        ]
        
        # Generate audit trail
        audit_result = orchestrator.generate_audit_trail()
        
        assert hasattr(audit_result, 'audit_entries')
        assert hasattr(audit_result, 'compliance_status')
        assert hasattr(audit_result, 'data_lineage')
        assert hasattr(audit_result, 'access_log')
        
        # Should track all monitoring activities
        assert len(audit_result.audit_entries) >= len(activities)
        
        # Each entry should have required fields
        for entry in audit_result.audit_entries:
            assert hasattr(entry, 'timestamp')
            assert hasattr(entry, 'action')
            assert hasattr(entry, 'user_context')
            assert hasattr(entry, 'data_accessed')
            assert hasattr(entry, 'results')
    
    def test_data_retention_compliance(self, existing_monitoring_systems):
        """Test data retention compliance"""
        aggregator = MetricsAggregator(
            existing_analytics=existing_monitoring_systems['operational_analytics'],
            retention_compliance_enabled=True,
            retention_policies={
                'metrics_data': '90_days',
                'alert_history': '1_year',
                'audit_logs': '7_years',
                'personal_data': '30_days'
            }
        )
        
        # Test retention policy enforcement
        retention_result = aggregator.enforce_retention_policies()
        
        assert hasattr(retention_result, 'policies_enforced')
        assert hasattr(retention_result, 'data_purged')
        assert hasattr(retention_result, 'compliance_violations')
        assert hasattr(retention_result, 'retention_schedule')
        
        # Should comply with all policies
        assert len(retention_result.compliance_violations) == 0
        assert retention_result.policies_enforced > 0
    
    def test_privacy_protection(self, existing_monitoring_systems):
        """Test privacy protection in monitoring data"""
        pipeline = RealTimeMonitoringPipeline(
            existing_systems=existing_monitoring_systems,
            privacy_protection_enabled=True,
            data_anonymization_enabled=True
        )
        
        # Test data with potential PII
        monitoring_data = {
            'user_id': 'user123',
            'ip_address': '192.168.1.100',
            'system_metrics': {
                'hostname': 'trading-server-01',
                'username': 'trader_john'
            },
            'trading_data': {
                'account_balance': 50000,
                'positions': ['BTC-LONG-1000']
            }
        }
        
        privacy_result = pipeline.apply_privacy_protection(monitoring_data)
        
        assert hasattr(privacy_result, 'anonymized_data')
        assert hasattr(privacy_result, 'pii_removed')
        assert hasattr(privacy_result, 'privacy_compliance_score')
        
        # Should remove or anonymize PII
        anonymized = privacy_result.anonymized_data
        assert 'user123' not in str(anonymized)  # User ID should be anonymized
        assert '192.168.1.100' not in str(anonymized)  # IP should be anonymized
        assert privacy_result.privacy_compliance_score > 0.9