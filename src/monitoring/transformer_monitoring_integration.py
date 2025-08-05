"""
Transformer Monitoring Integration System

This module coordinates all transformer monitoring components including:
- MonitoringOrchestrator for coordinating all components
- AlertCoordinator for transformer-specific alerts
- MetricsAggregator for cross-system correlation
- RealTimeMonitoringPipeline for streaming
- Dashboard integration
"""

import asyncio
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .transformer_drift_detection import TransformerDriftDetector
from .transformer_metrics import TransformerMetricsCollector, TransformerHealthDashboard
from .drift_detection import EnhancedDriftDetector
from .intelligent_alerting import IntelligentAlertingSystem
from .operational_analytics import OperationalAnalyticsManager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = structlog.get_logger(__name__)


class MonitoringState(Enum):
    """Enumeration for monitoring system states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class MonitoringConfiguration:
    """Configuration for monitoring system."""
    monitoring_interval_seconds: int = 30
    drift_detection_window: int = 100
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'drift_confidence': 0.8,
        'attention_entropy': 1.5,
        'memory_usage_mb': 2000,
        'performance_degradation': 0.15
    })
    real_time_enabled: bool = True
    dashboard_enabled: bool = True
    cloud_integration_enabled: bool = False
    failure_resilient: bool = True


@dataclass
class MonitoringResult:
    """Result from monitoring pipeline."""
    timestamp: datetime
    drift_detected: bool
    metrics_collected: bool
    alerts_generated: int
    health_status: str
    processing_latency_ms: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MonitoringOrchestrator:
    """Orchestrates all transformer monitoring components."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        reference_data: Any,
        feature_columns: List[str],
        config: Optional[MonitoringConfiguration] = None
    ):
        """
        Initialize monitoring orchestrator.
        
        Args:
            models: Dictionary of transformer models
            reference_data: Reference data for drift detection
            feature_columns: Feature columns for analysis
            config: Monitoring configuration
        """
        self.models = models
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.config = config or MonitoringConfiguration()
        
        self.state = MonitoringState.STOPPED
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize components
        self._initialize_components()
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # Enhanced drift detector with transformer support
            self.drift_detector = EnhancedDriftDetector(
                reference_data=self.reference_data,
                feature_columns=self.feature_columns,
                transformer_models=self.models
            )
            self.drift_detector.fit()
            
            # Transformer metrics collector
            self.metrics_collector = TransformerMetricsCollector(
                models=self.models,
                collection_interval_seconds=self.config.monitoring_interval_seconds,
                metrics_retention_hours=self.config.metrics_retention_hours
            )
            
            # Health dashboard
            if self.config.dashboard_enabled:
                self.health_dashboard = TransformerHealthDashboard(
                    update_interval_seconds=self.config.monitoring_interval_seconds,
                    alert_thresholds=self.config.alert_thresholds
                )
            else:
                self.health_dashboard = None
            
            # Alert coordinator
            self.alert_coordinator = AlertCoordinator(
                alert_thresholds=self.config.alert_thresholds
            )
            
            # Metrics aggregator
            self.metrics_aggregator = MetricsAggregator()
            
            # Real-time pipeline
            if self.config.real_time_enabled:
                self.real_time_pipeline = RealTimeMonitoringPipeline(
                    orchestrator=self,
                    monitoring_interval=self.config.monitoring_interval_seconds
                )
            else:
                self.real_time_pipeline = None
            
            self.logger.info("Monitoring components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring components: {e}")
            self.state = MonitoringState.ERROR
            raise
    
    def start_monitoring(self) -> bool:
        """
        Start the monitoring system.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.state != MonitoringState.STOPPED:
            self.logger.warning(f"Cannot start monitoring from state {self.state}")
            return False
        
        try:
            self.state = MonitoringState.STARTING
            self.stop_event.clear()
            
            # Start dashboard monitoring
            if self.health_dashboard:
                self.health_dashboard.start_real_time_monitoring(self.models)
            
            # Start real-time pipeline
            if self.real_time_pipeline:
                self.real_time_pipeline.start()
            
            # Start main monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.state = MonitoringState.RUNNING
            self.logger.info("Monitoring system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.state = MonitoringState.ERROR
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop the monitoring system.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.state != MonitoringState.RUNNING:
            self.logger.warning(f"Cannot stop monitoring from state {self.state}")
            return False
        
        try:
            self.state = MonitoringState.STOPPING
            self.stop_event.set()
            
            # Stop real-time pipeline
            if self.real_time_pipeline:
                self.real_time_pipeline.stop()
            
            # Stop dashboard monitoring
            if self.health_dashboard:
                self.health_dashboard.stop_real_time_monitoring()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            
            self.state = MonitoringState.STOPPED
            self.logger.info("Monitoring system stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            self.state = MonitoringState.ERROR
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Perform monitoring cycle
                result = self.execute_monitoring_cycle()
                
                # Log results
                self.logger.info(
                    "Monitoring cycle completed",
                    drift_detected=result.drift_detected,
                    alerts_generated=result.alerts_generated,
                    health_status=result.health_status,
                    latency_ms=result.processing_latency_ms
                )
                
                # Wait for next cycle
                self.stop_event.wait(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if not self.config.failure_resilient:
                    break
                # Continue with next cycle if failure resilient
                self.stop_event.wait(self.config.monitoring_interval_seconds)
    
    def execute_monitoring_cycle(
        self,
        current_data: Optional[Any] = None,
        current_attention_weights: Optional[Dict[str, Any]] = None
    ) -> MonitoringResult:
        """
        Execute a single monitoring cycle.
        
        Args:
            current_data: Current data for analysis
            current_attention_weights: Current attention weights
            
        Returns:
            MonitoringResult
        """
        start_time = time.time()
        errors = []
        drift_detected = False
        metrics_collected = False
        alerts_generated = 0
        health_status = 'unknown'
        
        try:
            # Collect metrics
            metrics_result = self.metrics_collector.collect_all_metrics()
            metrics_collected = True
            health_status = metrics_result.overall_health_status
            
            # Detect drift if data is available
            if current_data is not None:
                drift_result = self.drift_detector.detect_combined_drift(
                    current_data, current_attention_weights
                )
                drift_detected = drift_result['has_combined_drift']
                
                # Generate alerts
                if drift_detected:
                    alerts = self.alert_coordinator.generate_alerts(
                        drift_result, metrics_result
                    )
                    alerts_generated = len(alerts)
            
            # Aggregate metrics
            aggregated_metrics = self.metrics_aggregator.aggregate_metrics([metrics_result])
            
            # Update dashboard
            if self.health_dashboard:
                self.health_dashboard.generate_comprehensive_health_report(
                    self.models, metrics_result.model_metrics
                )
            
        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"Error in monitoring cycle: {e}")
        
        processing_latency = (time.time() - start_time) * 1000
        
        return MonitoringResult(
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            metrics_collected=metrics_collected,
            alerts_generated=alerts_generated,
            health_status=health_status,
            processing_latency_ms=processing_latency,
            errors=errors
        )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        return {
            'state': self.state.value,
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time()),
            'models_monitored': list(self.models.keys()),
            'components_active': {
                'drift_detector': self.drift_detector is not None,
                'metrics_collector': self.metrics_collector is not None,
                'health_dashboard': self.health_dashboard is not None,
                'alert_coordinator': self.alert_coordinator is not None,
                'real_time_pipeline': self.real_time_pipeline is not None
            },
            'configuration': {
                'monitoring_interval': self.config.monitoring_interval_seconds,
                'real_time_enabled': self.config.real_time_enabled,
                'dashboard_enabled': self.config.dashboard_enabled
            }
        }


class AlertCoordinator:
    """Coordinates alerts from different monitoring components."""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        """
        Initialize alert coordinator.
        
        Args:
            alert_thresholds: Alert thresholds for different metrics
        """
        self.alert_thresholds = alert_thresholds
        self.alert_history = []
        self.alert_suppression = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def generate_alerts(
        self,
        drift_result: Dict[str, Any],
        metrics_result: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate coordinated alerts from drift and metrics results.
        
        Args:
            drift_result: Drift detection result
            metrics_result: Metrics collection result
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        current_time = datetime.now()
        
        # Process drift alerts
        if drift_result.get('has_combined_drift', False):
            alert = {
                'type': 'combined_drift',
                'severity': self._determine_severity(drift_result['overall_drift_score']),
                'message': f"Combined drift detected (score: {drift_result['overall_drift_score']:.3f})",
                'timestamp': current_time,
                'metadata': drift_result
            }
            
            if self._should_send_alert(alert):
                alerts.append(alert)
        
        # Process metrics alerts
        if hasattr(metrics_result, 'overall_health_status'):
            if metrics_result.overall_health_status in ['warning', 'critical']:
                alert = {
                    'type': 'health_degradation',
                    'severity': metrics_result.overall_health_status,
                    'message': f"Model health degradation detected: {metrics_result.overall_health_status}",
                    'timestamp': current_time,
                    'metadata': {'metrics_result': metrics_result}
                }
                
                if self._should_send_alert(alert):
                    alerts.append(alert)
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Maintain alert history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return alerts
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine alert severity based on drift score."""
        if drift_score > 0.3:
            return 'critical'
        elif drift_score > 0.15:
            return 'warning'
        else:
            return 'info'
    
    def _should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Determine if alert should be sent (implements suppression logic).
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if alert should be sent, False otherwise
        """
        alert_key = f"{alert['type']}_{alert['severity']}"
        current_time = alert['timestamp']
        
        # Check if this type of alert was recently sent
        if alert_key in self.alert_suppression:
            last_sent = self.alert_suppression[alert_key]
            suppression_period = timedelta(minutes=15)  # 15-minute suppression
            
            if current_time - last_sent < suppression_period:
                return False
        
        # Update suppression timestamp
        self.alert_suppression[alert_key] = current_time
        return True
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]
        
        # Count alerts by type and severity
        alert_counts = {}
        severity_counts = {'info': 0, 'warning': 0, 'critical': 0}
        
        for alert in recent_alerts:
            alert_type = alert['type']
            severity = alert['severity']
            
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'alert_counts_by_type': alert_counts,
            'alert_counts_by_severity': severity_counts,
            'time_period_hours': hours,
            'most_recent_alert': recent_alerts[-1] if recent_alerts else None
        }


class MetricsAggregator:
    """Aggregates metrics from different monitoring components."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.aggregation_history = []
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def aggregate_metrics(self, metrics_results: List[Any]) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple sources.
        
        Args:
            metrics_results: List of metrics results
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_results:
            return {}
        
        aggregated = {
            'timestamp': datetime.now(),
            'source_count': len(metrics_results),
            'overall_health_score': 0.0,
            'aggregated_metrics': {},
            'correlation_analysis': {}
        }
        
        # Aggregate health scores
        health_scores = []
        for result in metrics_results:
            if hasattr(result, 'model_metrics'):
                for model_type, model_metrics in result.model_metrics.items():
                    if hasattr(model_metrics, 'get') and 'overall_score' in model_metrics:
                        health_scores.append(model_metrics['overall_score'])
        
        if health_scores:
            aggregated['overall_health_score'] = sum(health_scores) / len(health_scores)
        
        # Aggregate key metrics
        attention_entropies = []
        memory_usages = []
        confidence_scores = []
        
        for result in metrics_results:
            if hasattr(result, 'model_metrics'):
                for model_type, model_metrics in result.model_metrics.items():
                    # Extract attention entropy
                    if ('attention_health' in model_metrics and 
                        hasattr(model_metrics['attention_health'], 'mean_entropy')):
                        attention_entropies.append(model_metrics['attention_health'].mean_entropy)
                    
                    # Extract memory usage
                    if ('resource_metrics' in model_metrics and 
                        hasattr(model_metrics['resource_metrics'], 'memory_mb')):
                        memory_usages.append(model_metrics['resource_metrics'].memory_mb)
                    
                    # Extract confidence scores
                    if ('performance_metrics' in model_metrics and 
                        hasattr(model_metrics['performance_metrics'], 'confidence_mean')):
                        confidence_scores.append(model_metrics['performance_metrics'].confidence_mean)
        
        # Calculate aggregated statistics
        if attention_entropies:
            aggregated['aggregated_metrics']['attention_entropy'] = {
                'mean': sum(attention_entropies) / len(attention_entropies),
                'min': min(attention_entropies),
                'max': max(attention_entropies),
                'count': len(attention_entropies)
            }
        
        if memory_usages:
            aggregated['aggregated_metrics']['memory_usage'] = {
                'mean': sum(memory_usages) / len(memory_usages),
                'min': min(memory_usages),
                'max': max(memory_usages),
                'total': sum(memory_usages),
                'count': len(memory_usages)
            }
        
        if confidence_scores:
            aggregated['aggregated_metrics']['confidence'] = {
                'mean': sum(confidence_scores) / len(confidence_scores),
                'min': min(confidence_scores),
                'max': max(confidence_scores),
                'count': len(confidence_scores)
            }
        
        # Perform correlation analysis
        if len(attention_entropies) > 1 and len(confidence_scores) > 1:
            # Simple correlation between attention entropy and confidence
            if len(attention_entropies) == len(confidence_scores):
                import numpy as np
                correlation = np.corrcoef(attention_entropies, confidence_scores)[0, 1]
                aggregated['correlation_analysis']['entropy_confidence_correlation'] = correlation
        
        # Store in history
        self.aggregation_history.append(aggregated)
        
        # Maintain history size
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
        
        return aggregated
    
    def get_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get trends for a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            hours: Number of hours to look back
            
        Returns:
            Trend analysis dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_aggregations = [
            agg for agg in self.aggregation_history
            if agg['timestamp'] > cutoff_time
        ]
        
        if not recent_aggregations:
            return {'trend': 'unknown', 'data_points': 0}
        
        # Extract metric values
        values = []
        timestamps = []
        
        for agg in recent_aggregations:
            if (metric_name in agg.get('aggregated_metrics', {}) and
                'mean' in agg['aggregated_metrics'][metric_name]):
                values.append(agg['aggregated_metrics'][metric_name]['mean'])
                timestamps.append(agg['timestamp'])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(values)}
        
        # Calculate trend
        import numpy as np
        time_points = list(range(len(values)))
        trend_slope = np.polyfit(time_points, values, 1)[0]
        
        if trend_slope > 0.01:
            trend_direction = 'increasing'
        elif trend_slope < -0.01:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        return {
            'trend': trend_direction,
            'slope': trend_slope,
            'data_points': len(values),
            'current_value': values[-1],
            'change_from_start': values[-1] - values[0],
            'timestamps': timestamps
        }


class RealTimeMonitoringPipeline:
    """Real-time monitoring pipeline for streaming data."""
    
    def __init__(
        self,
        orchestrator: MonitoringOrchestrator,
        monitoring_interval: int = 30
    ):
        """
        Initialize real-time monitoring pipeline.
        
        Args:
            orchestrator: MonitoringOrchestrator instance
            monitoring_interval: Monitoring interval in seconds
        """
        self.orchestrator = orchestrator
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        self.data_buffer = []
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def start(self):
        """Start the real-time monitoring pipeline."""
        if self.is_running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self.pipeline_thread.start()
        self.logger.info("Real-time monitoring pipeline started")
    
    def stop(self):
        """Stop the real-time monitoring pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5.0)
        
        self.logger.info("Real-time monitoring pipeline stopped")
    
    def _pipeline_loop(self):
        """Main pipeline processing loop."""
        while not self.stop_event.is_set():
            try:
                # Process buffered data
                if self.data_buffer:
                    self._process_buffer()
                
                # Wait for next processing cycle
                self.stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in pipeline loop: {e}")
                if not self.orchestrator.config.failure_resilient:
                    break
    
    def add_data(self, data: Any, attention_weights: Optional[Dict[str, Any]] = None):
        """
        Add data to the processing buffer.
        
        Args:
            data: Data to process
            attention_weights: Associated attention weights
        """
        entry = {
            'timestamp': datetime.now(),
            'data': data,
            'attention_weights': attention_weights
        }
        
        self.data_buffer.append(entry)
        
        # Maintain buffer size
        if len(self.data_buffer) > 1000:
            self.data_buffer = self.data_buffer[-1000:]
    
    def _process_buffer(self):
        """Process data in the buffer."""
        if not self.data_buffer:
            return
        
        try:
            # Get most recent data
            latest_entry = self.data_buffer[-1]
            
            # Execute monitoring cycle
            result = self.orchestrator.execute_monitoring_cycle(
                current_data=latest_entry['data'],
                current_attention_weights=latest_entry['attention_weights']
            )
            
            self.logger.debug(
                "Real-time processing completed",
                drift_detected=result.drift_detected,
                health_status=result.health_status,
                latency_ms=result.processing_latency_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error processing buffer: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_running': self.is_running,
            'buffer_size': len(self.data_buffer),
            'monitoring_interval': self.monitoring_interval,
            'last_processed': self.data_buffer[-1]['timestamp'] if self.data_buffer else None
        }