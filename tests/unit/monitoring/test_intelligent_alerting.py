"""
Test suite for intelligent alerting system with ML-powered anomaly detection.

This module provides comprehensive TDD tests for Phase 6.1: Enhanced Monitoring & Alerting,
including intelligent alerting, anomaly detection, alert correlation, and escalation policies.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# The classes we'll implement
from src.monitoring.intelligent_alerting import (
    IntelligentAlertingSystem,
    AnomalyDetector,
    AlertCorrelationEngine,
    AlertRoutingManager,
    AlertAnalytics,
    AlertSeverity,
    AlertRule,
    Alert,
    AnomalyType,
    CorrelationRule
)


class TestIntelligentAlertingSystem:
    """Test the main intelligent alerting system."""
    
    @pytest.fixture
    def mock_anomaly_detector(self):
        """Mock anomaly detector."""
        detector = MagicMock(spec=AnomalyDetector)
        detector.detect_anomalies = AsyncMock(return_value=[])
        detector.is_healthy = MagicMock(return_value=True)
        return detector
    
    @pytest.fixture
    def mock_correlation_engine(self):
        """Mock alert correlation engine."""
        engine = MagicMock(spec=AlertCorrelationEngine)
        engine.correlate_alerts = AsyncMock(return_value=[])
        engine.is_healthy = MagicMock(return_value=True)
        return engine
    
    @pytest.fixture
    def mock_routing_manager(self):
        """Mock alert routing manager."""
        manager = MagicMock(spec=AlertRoutingManager)
        manager.route_alert = AsyncMock()
        manager.is_healthy = MagicMock(return_value=True)
        return manager
    
    @pytest.fixture
    def intelligent_alerting_system(self, mock_anomaly_detector, mock_correlation_engine, mock_routing_manager):
        """Create intelligent alerting system with mocked dependencies."""
        system = IntelligentAlertingSystem(
            anomaly_detector=mock_anomaly_detector,
            correlation_engine=mock_correlation_engine,
            routing_manager=mock_routing_manager
        )
        return system
    
    def test_intelligent_alerting_system_initialization(self, intelligent_alerting_system):
        """Test intelligent alerting system initialization."""
        assert intelligent_alerting_system.anomaly_detector is not None
        assert intelligent_alerting_system.correlation_engine is not None
        assert intelligent_alerting_system.routing_manager is not None
        assert intelligent_alerting_system.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_intelligent_alerting_system(self, intelligent_alerting_system):
        """Test starting the intelligent alerting system."""
        await intelligent_alerting_system.start()
        assert intelligent_alerting_system.is_running is True
    
    @pytest.mark.asyncio
    async def test_stop_intelligent_alerting_system(self, intelligent_alerting_system):
        """Test stopping the intelligent alerting system."""
        await intelligent_alerting_system.start()
        await intelligent_alerting_system.stop()
        assert intelligent_alerting_system.is_running is False
    
    @pytest.mark.asyncio
    async def test_process_alert_flow(self, intelligent_alerting_system):
        """Test the complete alert processing flow."""
        # Create test data (more samples to meet minimum requirement)
        test_metrics = {
            'trading_volume': [100, 150, 200, 180, 170, 160, 140, 155, 175, 190, 200, 185],
            'price_change': [0.01, 0.02, 0.05, 0.03, 0.02, 0.015, 0.025, 0.03, 0.02, 0.04, 0.05, 0.03],
            'timestamp': pd.date_range('2025-01-01', periods=12, freq='1h')
        }
        
        # Mock anomaly detection
        expected_anomalies = [
            Alert(
                id="test_anomaly_1",
                severity=AlertSeverity.WARNING,
                message="Trading volume anomaly detected",
                source="AnomalyDetector",
                anomaly_type=AnomalyType.TRADING_PATTERN,
                metadata={'volume': 200}
            )
        ]
        intelligent_alerting_system.anomaly_detector.detect_anomalies.return_value = expected_anomalies
        
        # Mock correlation engine to return the same alerts
        intelligent_alerting_system.correlation_engine.correlate_alerts.return_value = expected_anomalies
        
        # Process metrics
        alerts = await intelligent_alerting_system.process_metrics(test_metrics)
        
        # Verify anomaly detection was called
        intelligent_alerting_system.anomaly_detector.detect_anomalies.assert_called_once_with(test_metrics)
        
        # Verify correlation engine was called
        intelligent_alerting_system.correlation_engine.correlate_alerts.assert_called_once()
        
        # Verify routing manager was called for each alert
        assert intelligent_alerting_system.routing_manager.route_alert.call_count == len(expected_anomalies)
    
    def test_system_health_check(self, intelligent_alerting_system):
        """Test system health checking."""
        health_status = intelligent_alerting_system.get_health_status()
        
        assert 'anomaly_detector' in health_status
        assert 'correlation_engine' in health_status
        assert 'routing_manager' in health_status
        assert 'overall_healthy' in health_status
        assert health_status['overall_healthy'] is True


class TestAnomalyDetector:
    """Test the ML-powered anomaly detection system."""
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create anomaly detector."""
        config = {
            'isolation_forest_contamination': 0.1,
            'statistical_threshold': 3.0,
            'ml_model_threshold': 0.8,
            'min_samples_for_detection': 10
        }
        return AnomalyDetector(config)
    
    @pytest.mark.asyncio
    async def test_trading_pattern_anomaly_detection(self, anomaly_detector):
        """Test detection of trading pattern anomalies."""
        # Create test data with clear anomaly (more samples to meet minimum requirement)
        test_data = {
            'trading_volume': [100, 110, 105, 120, 115, 100, 95, 105, 110, 1000, 125, 130],  # 1000 is anomaly
            'price_change': [0.01, 0.02, 0.015, 0.025, 0.02, 0.01, 0.015, 0.02, 0.025, 0.02, 0.025, 0.03],
            'timestamp': pd.date_range('2025-01-01', periods=12, freq='1h')
        }
        
        anomalies = await anomaly_detector.detect_anomalies(test_data)
        
        # Should detect the volume anomaly
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.TRADING_PATTERN]
        assert len(volume_anomalies) > 0
        assert any('volume' in a.message.lower() for a in volume_anomalies)
    
    @pytest.mark.asyncio
    async def test_system_metrics_anomaly_detection(self, anomaly_detector):
        """Test detection of system metrics anomalies."""
        # Create test data with system metrics anomaly (more samples)
        test_data = {
            'cpu_usage': [0.2, 0.3, 0.25, 0.35, 0.3, 0.25, 0.4, 0.35, 0.3, 0.95, 0.4, 0.35],  # 0.95 is anomaly
            'memory_usage': [0.4, 0.45, 0.42, 0.48, 0.46, 0.47, 0.5, 0.49, 0.42, 0.47, 0.5, 0.49],
            'response_time': [10, 12, 11, 13, 12, 11, 14, 13, 15, 11, 14, 13],
            'timestamp': pd.date_range('2025-01-01', periods=12, freq='1h')
        }
        
        anomalies = await anomaly_detector.detect_anomalies(test_data)
        
        # Should detect the CPU usage anomaly
        system_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.SYSTEM_METRICS]
        assert len(system_anomalies) > 0
        assert any('cpu' in a.message.lower() for a in system_anomalies)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, anomaly_detector):
        """Test handling of insufficient data for anomaly detection."""
        # Create test data with too few samples
        test_data = {
            'trading_volume': [100, 110, 105],  # Only 3 samples
            'timestamp': pd.date_range('2025-01-01', periods=3, freq='1h')
        }
        
        anomalies = await anomaly_detector.detect_anomalies(test_data)
        
        # Should return empty list for insufficient data
        assert len(anomalies) == 0
    
    def test_isolation_forest_detection(self, anomaly_detector):
        """Test isolation forest-based anomaly detection."""
        # Create sample data
        normal_data = np.random.normal(0, 1, (100, 2))
        anomaly_data = np.array([[5, 5], [-5, -5]])  # Clear outliers
        data = np.vstack([normal_data, anomaly_data])
        
        anomaly_scores = anomaly_detector._detect_isolation_forest_anomalies(data)
        
        # Should detect anomalies in the outlier points
        assert len(anomaly_scores) == len(data)
        assert max(anomaly_scores[-2:]) > max(anomaly_scores[:-2])  # Outliers have higher scores
    
    def test_statistical_anomaly_detection(self):
        """Test statistical-based anomaly detection."""
        # Create detector with lower threshold for testing
        config = {
            'statistical_threshold': 2.0,  # Lower threshold for testing
            'min_samples_for_detection': 5
        }
        detector = AnomalyDetector(config)
        
        # Create data with clear statistical anomaly
        data = [10, 12, 11, 13, 12, 10, 14, 1000]  # 1000 is statistical anomaly (z-score > 2)
        
        anomalies = detector._detect_statistical_anomalies(data, 'test_metric')
        
        # Should detect the statistical anomaly
        assert len(anomalies) > 0
        assert any(a.metadata.get('value') == 1000 for a in anomalies)
    
    def test_detector_health_check(self, anomaly_detector):
        """Test anomaly detector health checking."""
        assert anomaly_detector.is_healthy() is True
        
        # Test with error state
        anomaly_detector._error_count = 10
        assert anomaly_detector.is_healthy() is False


class TestAlertCorrelationEngine:
    """Test the alert correlation and deduplication engine."""
    
    @pytest.fixture
    def correlation_engine(self):
        """Create alert correlation engine."""
        config = {
            'correlation_window_minutes': 5,
            'similarity_threshold': 0.8,
            'max_alerts_per_correlation': 10
        }
        return AlertCorrelationEngine(config)
    
    @pytest.fixture
    def sample_alerts(self):
        """Create sample alerts for testing."""
        now = datetime.now()
        return [
            Alert(
                id="alert_1",
                severity=AlertSeverity.WARNING,
                message="High CPU usage detected",
                source="SystemMonitor",
                timestamp=now,
                metadata={'cpu': 0.9}
            ),
            Alert(
                id="alert_2",
                severity=AlertSeverity.WARNING,
                message="High CPU utilization observed",
                source="SystemMonitor",
                timestamp=now + timedelta(minutes=1),
                metadata={'cpu': 0.95}
            ),
            Alert(
                id="alert_3",
                severity=AlertSeverity.CRITICAL,
                message="Memory usage critical",
                source="SystemMonitor",
                timestamp=now + timedelta(minutes=2),
                metadata={'memory': 0.98}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_alert_correlation(self, correlation_engine, sample_alerts):
        """Test basic alert correlation functionality."""
        correlated_alerts = await correlation_engine.correlate_alerts(sample_alerts)
        
        # Should correlate similar CPU alerts
        cpu_alerts = [a for a in correlated_alerts if 'cpu' in a.message.lower()]
        assert len(cpu_alerts) <= 2  # Should deduplicate similar CPU alerts
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, correlation_engine):
        """Test alert deduplication for identical alerts."""
        now = datetime.now()
        duplicate_alerts = [
            Alert(
                id="alert_1",
                severity=AlertSeverity.WARNING,
                message="Identical alert message",
                source="TestSource",
                timestamp=now
            ),
            Alert(
                id="alert_2",
                severity=AlertSeverity.WARNING,
                message="Identical alert message",
                source="TestSource",
                timestamp=now + timedelta(seconds=30)
            )
        ]
        
        correlated_alerts = await correlation_engine.correlate_alerts(duplicate_alerts)
        
        # Should deduplicate to single alert
        assert len(correlated_alerts) == 1
    
    def test_alert_similarity_calculation(self, correlation_engine):
        """Test alert similarity calculation."""
        alert1 = Alert(
            id="alert_1",
            severity=AlertSeverity.WARNING,
            message="High CPU usage detected",
            source="SystemMonitor"
        )
        alert2 = Alert(
            id="alert_2",
            severity=AlertSeverity.WARNING,
            message="High CPU utilization observed",
            source="SystemMonitor"
        )
        
        similarity = correlation_engine._calculate_alert_similarity(alert1, alert2)
        
        # Should have high similarity
        assert similarity > 0.7
    
    def test_correlation_rules(self, correlation_engine):
        """Test custom correlation rules."""
        rule = CorrelationRule(
            name="cpu_memory_correlation",
            conditions=['cpu_usage > 0.8', 'memory_usage > 0.8'],
            action="merge_as_system_overload"
        )
        
        correlation_engine.add_correlation_rule(rule)
        assert len(correlation_engine.correlation_rules) == 1
        assert correlation_engine.correlation_rules[0].name == "cpu_memory_correlation"


class TestAlertRoutingManager:
    """Test the alert routing and escalation system."""
    
    @pytest.fixture
    def routing_manager(self):
        """Create alert routing manager."""
        config = {
            'escalation_thresholds': {
                'warning': timedelta(minutes=10),
                'critical': timedelta(minutes=5),
                'emergency': timedelta(minutes=1)
            },
            'notification_channels': ['email', 'slack', 'pagerduty']
        }
        return AlertRoutingManager(config)
    
    @pytest.fixture
    def test_alert(self):
        """Create test alert."""
        return Alert(
            id="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            source="TestSource",
            metadata={'key': 'value'}
        )
    
    @pytest.mark.asyncio
    async def test_alert_routing(self, routing_manager, test_alert):
        """Test basic alert routing functionality."""
        # Mock notification service
        routing_manager.notification_service = AsyncMock()
        
        await routing_manager.route_alert(test_alert)
        
        # Should route alert to notification service
        routing_manager.notification_service.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_escalation_policies(self, routing_manager):
        """Test alert escalation policies."""
        # Create alert that should escalate
        old_alert = Alert(
            id="old_alert",
            severity=AlertSeverity.WARNING,
            message="Old unresolved alert",
            source="TestSource",
            timestamp=datetime.now() - timedelta(minutes=15)  # Old enough to escalate
        )
        
        # Route the alert first so it gets tracked
        await routing_manager.route_alert(old_alert)
        
        # Mock escalation
        routing_manager._escalate_alert = AsyncMock()
        
        await routing_manager.check_escalations()
        
        # Should escalate the old alert
        routing_manager._escalate_alert.assert_called()
    
    def test_routing_rules(self, routing_manager):
        """Test alert routing rules configuration."""
        # Test adding routing rule
        rule = {
            'condition': 'severity == "CRITICAL"',
            'channels': ['pagerduty', 'slack'],
            'immediate': True
        }
        
        routing_manager.add_routing_rule(rule)
        assert len(routing_manager.routing_rules) == 1
    
    def test_notification_channel_selection(self, routing_manager, test_alert):
        """Test notification channel selection based on alert severity."""
        channels = routing_manager._select_notification_channels(test_alert)
        
        # Should select appropriate channels for warning level
        assert 'email' in channels or 'slack' in channels
        
        # Test for critical alert
        critical_alert = Alert(
            id="critical_alert",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
            source="TestSource"
        )
        
        critical_channels = routing_manager._select_notification_channels(critical_alert)
        assert 'pagerduty' in critical_channels


class TestAlertAnalytics:
    """Test the alert analytics and insights system."""
    
    @pytest.fixture
    def alert_analytics(self):
        """Create alert analytics system."""
        return AlertAnalytics()
    
    @pytest.fixture
    def sample_alert_history(self):
        """Create sample alert history."""
        now = datetime.now()
        return [
            Alert(
                id="alert_1",
                severity=AlertSeverity.WARNING,
                message="CPU usage high",
                source="SystemMonitor",
                timestamp=now - timedelta(hours=1)
            ),
            Alert(
                id="alert_2",
                severity=AlertSeverity.CRITICAL,
                message="Memory critical",
                source="SystemMonitor",
                timestamp=now - timedelta(hours=2)
            ),
            Alert(
                id="alert_3",
                severity=AlertSeverity.WARNING,
                message="Trading volume spike",
                source="TradingMonitor",
                timestamp=now - timedelta(hours=3)
            )
        ]
    
    def test_alert_frequency_analysis(self, alert_analytics, sample_alert_history):
        """Test alert frequency analysis."""
        frequency_stats = alert_analytics.analyze_alert_frequency(sample_alert_history)
        
        assert 'total_alerts' in frequency_stats
        assert 'alerts_per_hour' in frequency_stats
        assert 'alerts_by_severity' in frequency_stats
        assert frequency_stats['total_alerts'] == 3
    
    def test_alert_pattern_detection(self, alert_analytics, sample_alert_history):
        """Test alert pattern detection."""
        patterns = alert_analytics.detect_alert_patterns(sample_alert_history)
        
        assert 'recurring_sources' in patterns
        assert 'time_patterns' in patterns
        assert 'SystemMonitor' in patterns['recurring_sources']
    
    def test_alert_fatigue_metrics(self, alert_analytics):
        """Test alert fatigue metrics calculation."""
        # Create many similar alerts
        now = datetime.now()
        similar_alerts = [
            Alert(
                id=f"alert_{i}",
                severity=AlertSeverity.WARNING,
                message="Similar warning message",
                source="TestSource",
                timestamp=now - timedelta(minutes=i)
            )
            for i in range(20)
        ]
        
        fatigue_metrics = alert_analytics.calculate_alert_fatigue(similar_alerts)
        
        assert 'fatigue_score' in fatigue_metrics
        assert 'duplicate_percentage' in fatigue_metrics
        assert fatigue_metrics['fatigue_score'] > 0
    
    def test_alert_resolution_insights(self, alert_analytics, sample_alert_history):
        """Test alert resolution insights."""
        # Mark some alerts as resolved
        sample_alert_history[0].resolved = True
        sample_alert_history[0].resolution_time = timedelta(minutes=30)
        
        insights = alert_analytics.generate_resolution_insights(sample_alert_history)
        
        assert 'resolution_rate' in insights
        assert 'average_resolution_time' in insights
        assert 'unresolved_count' in insights


class TestAlertSeverity:
    """Test alert severity classification."""
    
    def test_severity_levels(self):
        """Test severity level enumeration."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"
    
    def test_severity_comparison(self):
        """Test severity level comparison."""
        assert AlertSeverity.INFO < AlertSeverity.WARNING
        assert AlertSeverity.WARNING < AlertSeverity.CRITICAL
        assert AlertSeverity.CRITICAL < AlertSeverity.EMERGENCY
    
    def test_severity_escalation(self):
        """Test severity escalation logic."""
        current_severity = AlertSeverity.WARNING
        escalated = current_severity.escalate()
        assert escalated == AlertSeverity.CRITICAL
        
        # Test maximum escalation
        max_severity = AlertSeverity.EMERGENCY
        assert max_severity.escalate() == AlertSeverity.EMERGENCY


class TestAlertRule:
    """Test alert rule configuration and evaluation."""
    
    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            name="high_cpu_rule",
            condition="cpu_usage > 0.8",
            severity=AlertSeverity.WARNING,
            message_template="CPU usage is {cpu_usage}%"
        )
        
        assert rule.name == "high_cpu_rule"
        assert rule.severity == AlertSeverity.WARNING
    
    def test_alert_rule_evaluation(self):
        """Test alert rule condition evaluation."""
        rule = AlertRule(
            name="memory_rule",
            condition="memory_usage > 0.9",
            severity=AlertSeverity.CRITICAL,
            message_template="Memory usage critical: {memory_usage}%"
        )
        
        # Test condition that should trigger
        data = {'memory_usage': 0.95}
        assert rule.evaluate(data) is True
        
        # Test condition that should not trigger
        data = {'memory_usage': 0.8}
        assert rule.evaluate(data) is False
    
    def test_alert_rule_message_formatting(self):
        """Test alert rule message formatting."""
        rule = AlertRule(
            name="trading_rule",
            condition="volume > 1000",
            severity=AlertSeverity.WARNING,
            message_template="High trading volume detected: {volume} units"
        )
        
        data = {'volume': 1500}
        message = rule.format_message(data)
        assert "1500" in message
        assert "High trading volume" in message


class TestIntegrationWithExistingMonitoring:
    """Test integration with existing monitoring infrastructure."""
    
    def test_integration_with_existing_alerting(self):
        """Test integration with existing AlertingService."""
        from src.monitoring.alerting import AlertingService, AlertLevel
        
        # Should be able to create both systems
        existing_alerting = AlertingService()
        intelligent_alerting = IntelligentAlertingSystem()
        
        # Should work together
        assert existing_alerting is not None
        assert intelligent_alerting is not None
    
    def test_integration_with_drift_detection(self):
        """Test integration with existing drift detection."""
        # Create mock drift detector
        class MockDriftDetector:
            def detect_drift(self, data):
                return {"drift_score": 0.5, "drift_detected": True}
        
        # Should integrate with drift detection for anomaly context
        drift_detector = MockDriftDetector()
        intelligent_alerting = IntelligentAlertingSystem()
        
        # Test integration
        result = intelligent_alerting.integrate_drift_detection(drift_detector)
        assert result is True
        assert hasattr(intelligent_alerting, '_drift_detector')
    
    @pytest.mark.asyncio
    async def test_integration_with_safety_systems(self):
        """Test integration with safety systems."""
        # Create mock safety controller
        class MockSafetyController:
            async def emergency_stop(self, reason=None, alert_id=None):
                return {"status": "stopped", "reason": reason, "alert_id": alert_id}
        
        # Should integrate with safety systems for emergency alerts
        emergency_controller = MockSafetyController()
        intelligent_alerting = IntelligentAlertingSystem()
        
        # Test integration
        result = intelligent_alerting.integrate_safety_systems(emergency_controller)
        assert result is True
        assert hasattr(intelligent_alerting, '_safety_controller')
        assert hasattr(intelligent_alerting, '_emergency_alert_handler')