"""
Continuous Performance Monitoring Framework - Phase 7.2
Real-time performance monitoring and alerting for production systems.

Features:
- Real-time performance metric collection
- Automated alerting based on thresholds
- Performance trend tracking
- Dashboard integration
- Historical data analysis
- Predictive performance analytics
- Incident response automation

Following TDD methodology - tests written first, then implementations.
"""
import pytest
import asyncio
import time
import threading
import json
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass, field, asdict
from collections import deque
import statistics

from tests.conftest_integration import (
    mock_environment_variables,
    integration_test_helper,
    performance_tracker
)


@dataclass
class PerformanceMetric:
    """Real-time performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'threshold': self.threshold
        }


@dataclass
class PerformanceAlert:
    """Performance alert generated when thresholds are exceeded."""
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'info', 'warning', 'critical'
    message: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class MonitoringConfiguration:
    """Configuration for performance monitoring."""
    metrics_collection_interval_seconds: float = 5.0
    alert_check_interval_seconds: float = 10.0
    metric_retention_hours: int = 24
    alert_retention_hours: int = 168  # 1 week
    
    # Performance thresholds
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    memory_warning_mb: float = 1000.0
    memory_critical_mb: float = 2000.0
    cpu_warning_percent: float = 80.0
    cpu_critical_percent: float = 95.0
    error_rate_warning_percent: float = 1.0
    error_rate_critical_percent: float = 5.0


class MetricsCollector:
    """Collects performance metrics from various system components."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.metrics_queue = queue.Queue()
        self.registered_collectors: List[Callable[[], List[PerformanceMetric]]] = []
        
    def register_collector(self, collector_func: Callable[[], List[PerformanceMetric]]) -> None:
        """Register a metric collector function."""
        # This will initially fail - implementation needed
        pass
    
    def start_collection(self) -> None:
        """Start continuous metrics collection."""
        # This will initially fail - implementation needed
        pass
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        # This will initially fail - implementation needed
        pass
    
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        # This will initially fail - implementation needed
        pass
    
    def collect_trading_metrics(self) -> List[PerformanceMetric]:
        """Collect trading system performance metrics."""
        # This will initially fail - implementation needed
        pass
    
    def collect_ml_rl_metrics(self) -> List[PerformanceMetric]:
        """Collect ML/RL system performance metrics."""
        # This will initially fail - implementation needed
        pass
    
    def collect_database_metrics(self) -> List[PerformanceMetric]:
        """Collect database performance metrics."""
        # This will initially fail - implementation needed
        pass
    
    def get_latest_metrics(self, count: int = 100) -> List[PerformanceMetric]:
        """Get latest collected metrics."""
        # This will initially fail - implementation needed
        pass
    
    def _collection_loop(self) -> None:
        """Main collection loop running in separate thread."""
        # This will initially fail - implementation needed
        pass


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.notification_handlers: List[Callable[[PerformanceAlert], None]] = []
        
    def add_notification_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add notification handler for alerts."""
        # This will initially fail - implementation needed
        pass
    
    def check_thresholds(self, metrics: List[PerformanceMetric]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        # This will initially fail - implementation needed
        pass
    
    def create_alert(self, metric: PerformanceMetric, threshold: float, severity: str) -> PerformanceAlert:
        """Create new performance alert."""
        # This will initially fail - implementation needed
        pass
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        # This will initially fail - implementation needed
        pass
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        # This will initially fail - implementation needed
        pass
    
    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history for specified time period."""
        # This will initially fail - implementation needed
        pass
    
    def send_notification(self, alert: PerformanceAlert) -> None:
        """Send alert notification to registered handlers."""
        # This will initially fail - implementation needed
        pass


class TrendAnalyzer:
    """Analyzes performance trends for predictive monitoring."""
    
    def __init__(self):
        self.metric_buffers: Dict[str, deque] = {}
        self.trend_window_minutes = 30
        self.prediction_horizon_minutes = 15
        
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add metric to trend analysis."""
        # This will initially fail - implementation needed
        pass
    
    def analyze_trend(self, metric_name: str) -> Dict[str, Any]:
        """Analyze trend for specific metric."""
        # This will initially fail - implementation needed
        pass
    
    def predict_future_values(self, metric_name: str, minutes_ahead: int = 15) -> List[float]:
        """Predict future metric values based on current trend."""
        # This will initially fail - implementation needed
        pass
    
    def detect_anomalies(self, metric_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies in metric patterns."""
        # This will initially fail - implementation needed
        pass
    
    def calculate_trend_slope(self, values: List[float], timestamps: List[datetime]) -> float:
        """Calculate trend slope for time series data."""
        # This will initially fail - implementation needed
        pass
    
    def get_trend_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metric trends."""
        # This will initially fail - implementation needed
        pass


class PerformanceDashboard:
    """Performance dashboard for real-time monitoring visualization."""
    
    def __init__(self):
        self.dashboard_data: Dict[str, Any] = {}
        self.update_interval_seconds = 5.0
        self.active = False
        
    def start_dashboard(self) -> None:
        """Start dashboard data updates."""
        # This will initially fail - implementation needed
        pass
    
    def stop_dashboard(self) -> None:
        """Stop dashboard updates."""
        # This will initially fail - implementation needed
        pass
    
    def update_dashboard_data(self, metrics: List[PerformanceMetric], alerts: List[PerformanceAlert]) -> None:
        """Update dashboard with latest data."""
        # This will initially fail - implementation needed
        pass
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        # This will initially fail - implementation needed
        pass
    
    def create_metric_chart_data(self, metric_name: str, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Create chart data for specific metric."""
        # This will initially fail - implementation needed
        pass
    
    def create_alert_summary(self, alerts: List[PerformanceAlert]) -> Dict[str, Any]:
        """Create alert summary for dashboard."""
        # This will initially fail - implementation needed
        pass


class IncidentResponder:
    """Automated incident response for performance issues."""
    
    def __init__(self):
        self.response_rules: List[Dict[str, Any]] = []
        self.active_incidents: Dict[str, Dict[str, Any]] = {}
        
    def add_response_rule(self, rule: Dict[str, Any]) -> None:
        """Add automated response rule."""
        # This will initially fail - implementation needed
        pass
    
    def handle_alert(self, alert: PerformanceAlert) -> Optional[Dict[str, Any]]:
        """Handle performance alert with automated response."""
        # This will initially fail - implementation needed
        pass
    
    def trigger_emergency_stop(self, reason: str) -> bool:
        """Trigger emergency stop of trading system."""
        # This will initially fail - implementation needed
        pass
    
    def scale_resources(self, resource_type: str, scale_factor: float) -> bool:
        """Scale system resources automatically."""
        # This will initially fail - implementation needed
        pass
    
    def restart_component(self, component_name: str) -> bool:
        """Restart system component."""
        # This will initially fail - implementation needed
        pass
    
    def create_incident(self, alert: PerformanceAlert) -> str:
        """Create incident from alert."""
        # This will initially fail - implementation needed
        pass
    
    def resolve_incident(self, incident_id: str) -> bool:
        """Resolve incident."""
        # This will initially fail - implementation needed
        pass


class ContinuousPerformanceMonitor:
    """Main continuous performance monitoring system."""
    
    def __init__(self, config: Optional[MonitoringConfiguration] = None):
        self.config = config or MonitoringConfiguration()
        self.metrics_collector = MetricsCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.trend_analyzer = TrendAnalyzer()
        self.dashboard = PerformanceDashboard()
        self.incident_responder = IncidentResponder()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        # This will initially fail - implementation needed
        pass
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        # This will initially fail - implementation needed
        pass
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        # This will initially fail - implementation needed
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # This will initially fail - implementation needed
        pass
    
    def configure_alerts(self, alert_config: Dict[str, Any]) -> None:
        """Configure alert thresholds and rules."""
        # This will initially fail - implementation needed
        pass
    
    def add_custom_metric_collector(self, name: str, collector_func: Callable[[], List[PerformanceMetric]]) -> None:
        """Add custom metric collector."""
        # This will initially fail - implementation needed
        pass
    
    def export_metrics(self, format_type: str = 'json', time_range_hours: int = 1) -> str:
        """Export metrics data in specified format."""
        # This will initially fail - implementation needed
        pass
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        # This will initially fail - implementation needed
        pass


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def config(self):
        """Provide monitoring configuration."""
        return MonitoringConfiguration(
            metrics_collection_interval_seconds=1.0,
            alert_check_interval_seconds=2.0,
            metric_retention_hours=1
        )
    
    @pytest.fixture
    def metrics_collector(self, config):
        """Provide metrics collector."""
        return MetricsCollector(config)
    
    def test_metrics_collector_initialization(self, metrics_collector, config):
        """Test metrics collector initialization."""
        assert metrics_collector.config == config, "Should use provided configuration"
        assert not metrics_collector.active, "Should start inactive"
        assert isinstance(metrics_collector.registered_collectors, list), "Should initialize collectors list"
    
    def test_collector_registration(self, metrics_collector):
        """Test registering metric collector functions."""
        def sample_collector():
            return [PerformanceMetric(
                name="test_metric",
                value=42.0,
                unit="units",
                timestamp=datetime.now()
            )]
        
        # This will initially fail - implementation needed
        metrics_collector.register_collector(sample_collector)
        
        assert len(metrics_collector.registered_collectors) == 1, "Should register collector"
        
        # Test collector execution
        metrics = metrics_collector.registered_collectors[0]()
        assert len(metrics) == 1, "Collector should return metrics"
        assert metrics[0].name == "test_metric", "Should return correct metric"
    
    def test_system_metrics_collection(self, metrics_collector):
        """Test system metrics collection."""
        # This will initially fail - implementation needed
        metrics = metrics_collector.collect_system_metrics()
        
        assert len(metrics) > 0, "Should collect system metrics"
        
        metric_names = [m.name for m in metrics]
        expected_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
        
        for expected in expected_metrics:
            assert any(expected in name for name in metric_names), f"Should collect {expected} metric"
    
    def test_trading_metrics_collection(self, mock_environment_variables, metrics_collector):
        """Test trading system metrics collection."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            metrics = metrics_collector.collect_trading_metrics()
            
            assert len(metrics) > 0, "Should collect trading metrics"
            
            metric_names = [m.name for m in metrics]
            expected_metrics = ['order_execution_latency', 'trade_success_rate', 'position_count']
            
            for expected in expected_metrics:
                assert any(expected in name for name in metric_names), f"Should collect {expected} metric"
    
    def test_ml_rl_metrics_collection(self, mock_environment_variables, metrics_collector):
        """Test ML/RL system metrics collection."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            metrics = metrics_collector.collect_ml_rl_metrics()
            
            assert len(metrics) > 0, "Should collect ML/RL metrics"
            
            metric_names = [m.name for m in metrics]
            expected_metrics = ['ml_prediction_latency', 'rl_decision_latency', 'model_accuracy']
            
            for expected in expected_metrics:
                assert any(expected in name for name in metric_names), f"Should collect {expected} metric"
    
    def test_continuous_collection_lifecycle(self, metrics_collector):
        """Test continuous metrics collection lifecycle."""
        assert not metrics_collector.active, "Should start inactive"
        
        # This will initially fail - implementation needed
        metrics_collector.start_collection()
        assert metrics_collector.active, "Should be active after start"
        
        # Let it collect for a short time
        time.sleep(2.5)  # Should collect at least 2 intervals
        
        metrics_collector.stop_collection()
        assert not metrics_collector.active, "Should be inactive after stop"
        
        # Check collected metrics
        latest_metrics = metrics_collector.get_latest_metrics(10)
        assert len(latest_metrics) > 0, "Should have collected metrics"


class TestAlertManager:
    """Test alert management functionality."""
    
    @pytest.fixture
    def config(self):
        """Provide monitoring configuration."""
        return MonitoringConfiguration(
            latency_warning_ms=100.0,
            latency_critical_ms=500.0,
            memory_warning_mb=1000.0,
            memory_critical_mb=2000.0
        )
    
    @pytest.fixture
    def alert_manager(self, config):
        """Provide alert manager."""
        return AlertManager(config)
    
    @pytest.fixture
    def sample_metrics(self):
        """Provide sample metrics for testing."""
        return [
            PerformanceMetric(
                name="order_execution_latency",
                value=150.0,  # Above warning threshold
                unit="ms",
                timestamp=datetime.now(),
                threshold=100.0
            ),
            PerformanceMetric(
                name="memory_usage",
                value=2500.0,  # Above critical threshold
                unit="MB",
                timestamp=datetime.now(),
                threshold=2000.0
            ),
            PerformanceMetric(
                name="cpu_usage",
                value=50.0,  # Normal level
                unit="percent",
                timestamp=datetime.now(),
                threshold=80.0
            )
        ]
    
    def test_alert_manager_initialization(self, alert_manager, config):
        """Test alert manager initialization."""
        assert alert_manager.config == config, "Should use provided configuration"
        assert len(alert_manager.active_alerts) == 0, "Should start with no alerts"
        assert len(alert_manager.alert_history) == 0, "Should start with empty history"
    
    def test_threshold_checking_and_alert_generation(self, alert_manager, sample_metrics):
        """Test threshold checking and alert generation."""
        # This will initially fail - implementation needed
        alerts = alert_manager.check_thresholds(sample_metrics)
        
        assert len(alerts) >= 2, "Should generate alerts for metrics exceeding thresholds"
        
        # Check for latency warning alert
        latency_alerts = [a for a in alerts if 'latency' in a.metric_name]
        assert len(latency_alerts) > 0, "Should generate latency alert"
        assert latency_alerts[0].severity == 'warning', "Should be warning severity"
        
        # Check for memory critical alert
        memory_alerts = [a for a in alerts if 'memory' in a.metric_name]
        assert len(memory_alerts) > 0, "Should generate memory alert"
        assert memory_alerts[0].severity == 'critical', "Should be critical severity"
    
    def test_alert_creation_and_management(self, alert_manager, sample_metrics):
        """Test alert creation and management."""
        metric = sample_metrics[0]  # Latency metric
        
        # This will initially fail - implementation needed
        alert = alert_manager.create_alert(metric, 100.0, 'warning')
        
        assert alert.alert_id is not None, "Alert should have ID"
        assert alert.metric_name == metric.name, "Alert should reference correct metric"
        assert alert.current_value == metric.value, "Alert should record current value"
        assert alert.severity == 'warning', "Alert should have correct severity"
        assert not alert.resolved, "Alert should start unresolved"
        
        # Test alert resolution
        resolved = alert_manager.resolve_alert(alert.alert_id)
        assert resolved, "Should successfully resolve alert"
    
    def test_active_alerts_management(self, alert_manager, sample_metrics):
        """Test active alerts management."""
        # Generate alerts
        alerts = alert_manager.check_thresholds(sample_metrics)
        
        # Check active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) >= 2, "Should have active alerts"
        
        # Filter by severity
        critical_alerts = alert_manager.get_active_alerts(severity_filter='critical')
        warning_alerts = alert_manager.get_active_alerts(severity_filter='warning')
        
        assert len(critical_alerts) >= 1, "Should have critical alerts"
        assert len(warning_alerts) >= 1, "Should have warning alerts"
    
    def test_notification_handling(self, alert_manager, sample_metrics):
        """Test notification handling."""
        notifications_received = []
        
        def test_handler(alert: PerformanceAlert):
            notifications_received.append(alert)
        
        # This will initially fail - implementation needed
        alert_manager.add_notification_handler(test_handler)
        
        # Generate alerts (should trigger notifications)
        alerts = alert_manager.check_thresholds(sample_metrics)
        
        # Check notifications were sent
        assert len(notifications_received) >= 2, "Should send notifications for generated alerts"
        assert any(alert.severity == 'critical' for alert in notifications_received), "Should notify for critical alerts"


class TestTrendAnalyzer:
    """Test trend analysis functionality."""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Provide trend analyzer."""
        return TrendAnalyzer()
    
    @pytest.fixture
    def trending_metrics(self):
        """Provide metrics showing a trend."""
        metrics = []
        base_time = datetime.now() - timedelta(minutes=30)
        
        for i in range(30):
            # Create upward trend: base value + trend component + noise
            value = 50.0 + (i * 2.0) + (i % 3)  # Increasing trend
            metrics.append(PerformanceMetric(
                name="latency_trend_test",
                value=value,
                unit="ms",
                timestamp=base_time + timedelta(minutes=i)
            ))
        
        return metrics
    
    def test_trend_analyzer_initialization(self, trend_analyzer):
        """Test trend analyzer initialization."""
        assert isinstance(trend_analyzer.metric_buffers, dict), "Should initialize metric buffers"
        assert trend_analyzer.trend_window_minutes > 0, "Should have positive trend window"
        assert trend_analyzer.prediction_horizon_minutes > 0, "Should have positive prediction horizon"
    
    def test_metric_addition_and_buffering(self, trend_analyzer, trending_metrics):
        """Test adding metrics to trend analysis."""
        for metric in trending_metrics:
            # This will initially fail - implementation needed
            trend_analyzer.add_metric(metric)
        
        assert "latency_trend_test" in trend_analyzer.metric_buffers, "Should buffer metric"
        assert len(trend_analyzer.metric_buffers["latency_trend_test"]) > 0, "Should store metric values"
    
    def test_trend_analysis(self, trend_analyzer, trending_metrics):
        """Test trend analysis functionality."""
        # Add metrics to analyzer
        for metric in trending_metrics:
            trend_analyzer.add_metric(metric)
        
        # This will initially fail - implementation needed
        trend_result = trend_analyzer.analyze_trend("latency_trend_test")
        
        assert trend_result is not None, "Should return trend analysis"
        assert "direction" in trend_result, "Should identify trend direction"
        assert "slope" in trend_result, "Should calculate slope"
        assert "confidence" in trend_result, "Should provide confidence"
        
        # Should detect upward trend
        assert trend_result["direction"] in ["up", "increasing", "rising"], "Should detect upward trend"
        assert trend_result["slope"] > 0, "Slope should be positive for upward trend"
    
    def test_future_value_prediction(self, trend_analyzer, trending_metrics):
        """Test future value prediction."""
        # Add metrics to analyzer
        for metric in trending_metrics:
            trend_analyzer.add_metric(metric)
        
        # This will initially fail - implementation needed
        predictions = trend_analyzer.predict_future_values("latency_trend_test", minutes_ahead=10)
        
        assert len(predictions) > 0, "Should return predictions"
        assert len(predictions) <= 10, "Should not exceed requested prediction count"
        
        # Predictions should be reasonable (continuing the trend)
        last_actual_value = trending_metrics[-1].value
        first_prediction = predictions[0]
        
        assert first_prediction > last_actual_value, "Prediction should continue upward trend"
    
    def test_anomaly_detection(self, trend_analyzer):
        """Test anomaly detection in metrics."""
        # Create metrics with an anomaly
        base_time = datetime.now() - timedelta(minutes=20)
        anomaly_metrics = []
        
        for i in range(20):
            if i == 10:  # Anomaly at position 10
                value = 200.0  # Spike
            else:
                value = 50.0 + (i % 3)  # Normal pattern
            
            anomaly_metrics.append(PerformanceMetric(
                name="anomaly_test",
                value=value,
                unit="ms",
                timestamp=base_time + timedelta(minutes=i)
            ))
        
        # Add metrics to analyzer
        for metric in anomaly_metrics:
            trend_analyzer.add_metric(metric)
        
        # This will initially fail - implementation needed
        anomalies = trend_analyzer.detect_anomalies("anomaly_test")
        
        assert len(anomalies) > 0, "Should detect anomalies"
        assert any(anomaly["value"] == 200.0 for anomaly in anomalies), "Should detect the spike"


class TestContinuousPerformanceMonitor:
    """Test integrated continuous performance monitoring."""
    
    @pytest.fixture
    def monitor_config(self):
        """Provide monitoring configuration for testing."""
        return MonitoringConfiguration(
            metrics_collection_interval_seconds=1.0,
            alert_check_interval_seconds=2.0,
            metric_retention_hours=1,
            latency_warning_ms=50.0,
            latency_critical_ms=200.0
        )
    
    @pytest.fixture
    def performance_monitor(self, monitor_config):
        """Provide continuous performance monitor."""
        return ContinuousPerformanceMonitor(monitor_config)
    
    def test_monitor_initialization(self, performance_monitor, monitor_config):
        """Test monitor initialization with all components."""
        assert performance_monitor.config == monitor_config, "Should use provided configuration"
        assert performance_monitor.metrics_collector is not None, "Should initialize metrics collector"
        assert performance_monitor.alert_manager is not None, "Should initialize alert manager"
        assert performance_monitor.trend_analyzer is not None, "Should initialize trend analyzer"
        assert performance_monitor.dashboard is not None, "Should initialize dashboard"
        assert performance_monitor.incident_responder is not None, "Should initialize incident responder"
        assert not performance_monitor.monitoring_active, "Should start inactive"
    
    def test_monitoring_lifecycle(self, performance_monitor):
        """Test complete monitoring lifecycle."""
        # This will initially fail - implementation needed
        performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active, "Should be active after start"
        
        # Let monitoring run for a short time
        time.sleep(3.0)
        
        # Check monitoring status
        status = performance_monitor.get_monitoring_status()
        assert status is not None, "Should return monitoring status"
        assert status.get("active", False), "Status should show monitoring as active"
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active, "Should be inactive after stop"
    
    def test_performance_summary_generation(self, mock_environment_variables, performance_monitor):
        """Test performance summary generation."""
        with patch.dict('os.environ', mock_environment_variables):
            performance_monitor.start_monitoring()
            
            # Let monitoring collect some data
            time.sleep(2.5)
            
            # This will initially fail - implementation needed
            summary = performance_monitor.get_performance_summary()
            
            assert summary is not None, "Should return performance summary"
            assert "metrics" in summary, "Summary should include metrics"
            assert "alerts" in summary, "Summary should include alerts"
            assert "trends" in summary, "Summary should include trends"
            
            performance_monitor.stop_monitoring()
    
    def test_custom_metric_collector_integration(self, performance_monitor):
        """Test adding custom metric collectors."""
        custom_metrics_called = False
        
        def custom_collector():
            nonlocal custom_metrics_called
            custom_metrics_called = True
            return [PerformanceMetric(
                name="custom_metric",
                value=123.45,
                unit="custom_units",
                timestamp=datetime.now()
            )]
        
        # This will initially fail - implementation needed
        performance_monitor.add_custom_metric_collector("test_collector", custom_collector)
        
        performance_monitor.start_monitoring()
        time.sleep(2.0)
        performance_monitor.stop_monitoring()
        
        assert custom_metrics_called, "Custom collector should be called during monitoring"
    
    def test_alert_configuration_and_triggering(self, performance_monitor):
        """Test alert configuration and triggering."""
        # Configure custom alert thresholds
        alert_config = {
            "latency_warning_ms": 25.0,
            "latency_critical_ms": 100.0,
            "memory_warning_mb": 500.0
        }
        
        # This will initially fail - implementation needed
        performance_monitor.configure_alerts(alert_config)
        
        # Simulate metric that would trigger alert
        high_latency_metric = PerformanceMetric(
            name="test_latency",
            value=75.0,  # Above warning, below critical
            unit="ms",
            timestamp=datetime.now(),
            threshold=25.0
        )
        
        performance_monitor.start_monitoring()
        
        # Manually add metric to trigger alert
        performance_monitor.metrics_collector.metrics_queue.put(high_latency_metric)
        
        time.sleep(3.0)  # Wait for alert processing
        
        # Check for generated alerts
        active_alerts = performance_monitor.alert_manager.get_active_alerts()
        
        performance_monitor.stop_monitoring()
        
        # Should generate alert for high latency
        assert len(active_alerts) > 0, "Should generate alerts for threshold violations"
    
    def test_metrics_export_functionality(self, performance_monitor):
        """Test metrics export functionality."""
        performance_monitor.start_monitoring()
        time.sleep(2.0)
        
        # This will initially fail - implementation needed
        exported_data = performance_monitor.export_metrics(format_type='json', time_range_hours=1)
        
        performance_monitor.stop_monitoring()
        
        assert exported_data is not None, "Should export metrics data"
        assert len(exported_data) > 0, "Exported data should not be empty"
        
        # Validate JSON format
        if exported_data:
            try:
                json.loads(exported_data)
                json_valid = True
            except json.JSONDecodeError:
                json_valid = False
            
            assert json_valid, "Exported data should be valid JSON"


class TestIntegratedMonitoringScenarios:
    """Test integrated monitoring scenarios."""
    
    def test_end_to_end_performance_monitoring(self, mock_environment_variables, performance_tracker):
        """Test complete end-to-end performance monitoring scenario."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== END-TO-END PERFORMANCE MONITORING TEST ===")
            
            # Initialize monitoring with realistic configuration
            config = MonitoringConfiguration(
                metrics_collection_interval_seconds=0.5,
                alert_check_interval_seconds=1.0,
                latency_warning_ms=50.0,
                latency_critical_ms=200.0,
                memory_warning_mb=500.0,
                memory_critical_mb=1000.0
            )
            
            monitor = ContinuousPerformanceMonitor(config)
            
            # Track notifications
            alerts_received = []
            
            def alert_handler(alert):
                alerts_received.append(alert)
                print(f"   📢 Alert: {alert.metric_name} = {alert.current_value}{alert.threshold and f' (threshold: {alert.threshold_value})' or ''}")
            
            monitor.alert_manager.add_notification_handler(alert_handler)
            
            performance_tracker.start_timing('end_to_end_monitoring')
            
            try:
                # Start monitoring
                print("\n1. Starting Continuous Monitoring:")
                monitor.start_monitoring()
                
                # Let it collect baseline metrics
                time.sleep(2.0)
                print("   ✓ Baseline metrics collected")
                
                # Simulate performance events
                print("\n2. Simulating Performance Events:")
                
                # Add high latency metrics to trigger alerts
                for i in range(3):
                    high_latency_metric = PerformanceMetric(
                        name="simulated_latency",
                        value=150.0 + i * 50,  # Increasing latency
                        unit="ms",
                        timestamp=datetime.now(),
                        threshold=50.0
                    )
                    monitor.trend_analyzer.add_metric(high_latency_metric)
                
                time.sleep(2.0)  # Allow alert processing
                
                # Check monitoring status
                print("\n3. Checking Monitoring Status:")
                status = monitor.get_monitoring_status()
                print(f"   ✓ Monitoring active: {status.get('active', False)}")
                print(f"   ✓ Metrics collected: {status.get('metrics_count', 0)}")
                print(f"   ✓ Active alerts: {status.get('active_alerts_count', 0)}")
                
                # Generate performance summary
                print("\n4. Generating Performance Summary:")
                summary = monitor.get_performance_summary()
                print(f"   ✓ Total metrics: {len(summary.get('metrics', []))}")
                print(f"   ✓ Alert history: {len(summary.get('alerts', []))}")
                print(f"   ✓ Trend analysis: {len(summary.get('trends', {}))}")
                
                # Test metrics export
                print("\n5. Testing Metrics Export:")
                exported_metrics = monitor.export_metrics('json', 1)
                export_size = len(exported_metrics) if exported_metrics else 0
                print(f"   ✓ Exported data size: {export_size} characters")
                
                performance_tracker.end_timing('end_to_end_monitoring')
                
                # Validate end-to-end functionality
                assert status["active"], "Monitoring should be active"
                assert status.get("metrics_count", 0) > 0, "Should have collected metrics"
                assert len(summary.get("metrics", [])) > 0, "Summary should include metrics"
                assert export_size > 0, "Should export metrics data"
                
                print("\n=== END-TO-END MONITORING VALIDATED ✓ ===")
                
            finally:
                # Clean up
                monitor.stop_monitoring()
                print("   ✓ Monitoring stopped")


# Mark all tests as continuous monitoring tests
pytestmark = [pytest.mark.performance, pytest.mark.monitoring]