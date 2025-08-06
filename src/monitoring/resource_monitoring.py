"""
Resource Monitoring for Dynamic Resource Allocation

Provides monitoring capabilities for resource usage, performance metrics,
and alerting for the dynamic resource allocation system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Optional Google Cloud imports
try:
    from google.cloud import monitoring_v3
except ImportError:
    monitoring_v3 = None

try:
    from google.cloud import logging as cloud_logging
except ImportError:
    cloud_logging = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: float
    memory_usage_gb: float
    memory_limit_gb: float
    cpu_utilization: float
    cpu_limit: float
    request_count: int
    response_time_ms: float
    error_count: int
    model_type: str = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    threshold: float
    duration_seconds: int
    notification_channels: List[str]
    enabled: bool = True


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResourceUsageMonitor:
    """
    Resource Usage Monitor
    
    Monitors resource usage patterns and provides real-time metrics
    for the dynamic resource allocation system.
    """
    
    def __init__(self, project_id: str, region: str):
        """Initialize resource usage monitor"""
        self.project_id = project_id
        self.region = region
        self.monitoring_client = monitoring_v3.MetricServiceClient() if monitoring_v3 else None
        self.metrics_buffer = []
        self.monitoring_active = False
        
        logger.info(f"Initialized ResourceUsageMonitor for {project_id}")

    async def start_monitoring(self, collection_interval_seconds: int = 30):
        """Start continuous resource monitoring"""
        try:
            self.monitoring_active = True
            logger.info(f"Started resource monitoring with {collection_interval_seconds}s interval")
            
            while self.monitoring_active:
                metrics = await self.collect_resource_metrics()
                if metrics:
                    self.metrics_buffer.append(metrics)
                    
                    # Keep buffer size manageable
                    if len(self.metrics_buffer) > 1000:
                        self.metrics_buffer = self.metrics_buffer[-500:]
                
                await asyncio.sleep(collection_interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
            self.monitoring_active = False

    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        logger.info("Stopped resource monitoring")

    async def collect_resource_metrics(self) -> Optional[ResourceMetrics]:
        """Collect current resource metrics"""
        try:
            # Mock implementation - in production this would query GCP monitoring
            current_time = time.time()
            
            # Simulate realistic metrics
            memory_usage = 4.2 + (time.time() % 60) / 60.0 * 2.0  # 4.2-6.2 GB
            cpu_usage = 0.4 + (time.time() % 30) / 30.0 * 0.4  # 40-80% CPU
            
            metrics = ResourceMetrics(
                timestamp=current_time,
                memory_usage_gb=memory_usage,
                memory_limit_gb=8.0,
                cpu_utilization=cpu_usage,
                cpu_limit=6.0,
                request_count=25 + int(time.time() % 10),
                response_time_ms=75.0 + (time.time() % 20),
                error_count=0,
                model_type="itransformer"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return None

    def get_metrics_history(self, duration_minutes: int = 60) -> List[ResourceMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]

    def get_current_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages"""
        if not self.metrics_buffer:
            return {"memory": 0.0, "cpu": 0.0}
        
        latest = self.metrics_buffer[-1]
        return {
            "memory": (latest.memory_usage_gb / latest.memory_limit_gb) * 100,
            "cpu": latest.cpu_utilization * 100
        }

    def analyze_usage_trends(self, window_minutes: int = 30) -> Dict[str, Any]:
        """Analyze resource usage trends"""
        try:
            history = self.get_metrics_history(window_minutes)
            if len(history) < 2:
                return {"trend": "insufficient_data"}
            
            # Calculate trends
            memory_values = [m.memory_usage_gb for m in history]
            cpu_values = [m.cpu_utilization for m in history]
            latency_values = [m.response_time_ms for m in history]
            
            memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
            
            analysis = {
                "memory_trend": memory_trend,
                "cpu_trend": cpu_trend,
                "avg_memory_gb": sum(memory_values) / len(memory_values),
                "avg_cpu_utilization": sum(cpu_values) / len(cpu_values),
                "avg_latency_ms": sum(latency_values) / len(latency_values),
                "data_points": len(history)
            }
            
            logger.info(f"Usage trend analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing usage trends: {e}")
            return {"trend": "error"}


class PerformanceMetricsCollector:
    """
    Performance Metrics Collector
    
    Collects and aggregates performance metrics for resource allocation decisions.
    """
    
    def __init__(self, project_id: str):
        """Initialize performance metrics collector"""
        self.project_id = project_id
        self.monitoring_client = monitoring_v3.MetricServiceClient() if monitoring_v3 else None
        self.performance_history = []
        
        logger.info("Initialized PerformanceMetricsCollector")

    async def collect_performance_metrics(self, model_types: List[str]) -> Dict[str, Any]:
        """Collect performance metrics for specified model types"""
        try:
            metrics = {}
            
            for model_type in model_types:
                model_metrics = await self._collect_model_metrics(model_type)
                metrics[model_type] = model_metrics
            
            # Add timestamp
            metrics["timestamp"] = time.time()
            self.performance_history.append(metrics)
            
            # Limit history size
            if len(self.performance_history) > 2880:  # 24 hours at 30s intervals
                self.performance_history = self.performance_history[-1440:]
            
            logger.info(f"Collected performance metrics for {len(model_types)} models")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {}

    async def _collect_model_metrics(self, model_type: str) -> Dict[str, Any]:
        """Collect metrics for a specific model type"""
        try:
            # Mock implementation - in production would query actual metrics
            base_latency = {"itransformer": 80, "patchtst": 65, "timesmixer": 95, "timesfm": 120}
            base_throughput = {"itransformer": 45, "patchtst": 55, "timesmixer": 38, "timesfm": 32}
            
            # Add some variability
            variance = (time.time() % 10) / 10.0 - 0.5  # -0.5 to 0.5
            
            metrics = {
                "inference_latency_ms": base_latency.get(model_type, 85) + (variance * 20),
                "throughput_rps": base_throughput.get(model_type, 40) + (variance * 10),
                "cache_hit_rate": 0.75 + (variance * 0.2),
                "memory_efficiency": 0.82 + (variance * 0.1),
                "error_rate": max(0, 0.01 + (variance * 0.005))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {model_type}: {e}")
            return {}

    def get_performance_summary(self, model_type: str, duration_minutes: int = 30) -> Dict[str, Any]:
        """Get performance summary for a model type"""
        try:
            cutoff_time = time.time() - (duration_minutes * 60)
            relevant_metrics = [
                m for m in self.performance_history 
                if m.get("timestamp", 0) >= cutoff_time and model_type in m
            ]
            
            if not relevant_metrics:
                return {"status": "no_data"}
            
            # Aggregate metrics
            latencies = [m[model_type]["inference_latency_ms"] for m in relevant_metrics]
            throughputs = [m[model_type]["throughput_rps"] for m in relevant_metrics]
            cache_rates = [m[model_type]["cache_hit_rate"] for m in relevant_metrics]
            
            summary = {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "avg_throughput_rps": sum(throughputs) / len(throughputs),
                "avg_cache_hit_rate": sum(cache_rates) / len(cache_rates),
                "data_points": len(relevant_metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"status": "error"}


class ResourceAlertManager:
    """
    Resource Alert Manager
    
    Manages alerting for resource allocation and performance issues.
    """
    
    def __init__(self, project_id: str, notification_channels: List[str]):
        """Initialize resource alert manager"""
        self.project_id = project_id
        self.notification_channels = notification_channels
        self.monitoring_client = monitoring_v3.MetricServiceClient() if monitoring_v3 else None
        self.alert_rules = []
        self.active_alerts = {}
        
        logger.info(f"Initialized ResourceAlertManager with {len(notification_channels)} channels")

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    async def evaluate_alerts(self, current_metrics: ResourceMetrics):
        """Evaluate all alert rules against current metrics"""
        try:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                should_alert = await self._evaluate_rule(rule, current_metrics)
                alert_key = f"{rule.name}_{current_metrics.model_type}"
                
                if should_alert and alert_key not in self.active_alerts:
                    # New alert
                    await self._trigger_alert(rule, current_metrics)
                    self.active_alerts[alert_key] = {
                        "rule": rule,
                        "triggered_at": time.time(),
                        "metrics": current_metrics
                    }
                elif not should_alert and alert_key in self.active_alerts:
                    # Alert resolved
                    await self._resolve_alert(rule, current_metrics)
                    del self.active_alerts[alert_key]
                    
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")

    async def _evaluate_rule(self, rule: AlertRule, metrics: ResourceMetrics) -> bool:
        """Evaluate a single alert rule"""
        try:
            metric_value = None
            
            if rule.metric == "memory_usage_percent":
                metric_value = (metrics.memory_usage_gb / metrics.memory_limit_gb) * 100
            elif rule.metric == "cpu_utilization_percent":
                metric_value = metrics.cpu_utilization * 100
            elif rule.metric == "response_time_ms":
                metric_value = metrics.response_time_ms
            elif rule.metric == "error_rate_percent":
                metric_value = (metrics.error_count / max(metrics.request_count, 1)) * 100
            
            if metric_value is None:
                return False
            
            return metric_value > rule.threshold
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return False

    async def _trigger_alert(self, rule: AlertRule, metrics: ResourceMetrics):
        """Trigger an alert"""
        try:
            alert_message = (
                f"Alert: {rule.name}\n"
                f"Model: {metrics.model_type}\n"
                f"Metric: {rule.metric} exceeded threshold {rule.threshold}\n"
                f"Timestamp: {datetime.fromtimestamp(metrics.timestamp)}\n"
                f"Memory: {metrics.memory_usage_gb:.1f}GB / {metrics.memory_limit_gb}GB\n"
                f"CPU: {metrics.cpu_utilization:.1%}\n"
                f"Response Time: {metrics.response_time_ms:.1f}ms"
            )
            
            # In production, this would send to actual notification channels
            logger.warning(f"ALERT TRIGGERED: {alert_message}")
            
            # Could integrate with Slack, PagerDuty, etc.
            for channel in rule.notification_channels:
                await self._send_notification(channel, alert_message, AlertSeverity.HIGH)
                
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    async def _resolve_alert(self, rule: AlertRule, metrics: ResourceMetrics):
        """Resolve an alert"""
        try:
            resolve_message = (
                f"RESOLVED: {rule.name}\n"
                f"Model: {metrics.model_type}\n"
                f"Alert condition no longer met\n"
                f"Timestamp: {datetime.fromtimestamp(metrics.timestamp)}"
            )
            
            logger.info(f"ALERT RESOLVED: {resolve_message}")
            
            for channel in rule.notification_channels:
                await self._send_notification(channel, resolve_message, AlertSeverity.LOW)
                
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")

    async def _send_notification(self, channel: str, message: str, severity: AlertSeverity):
        """Send notification to a channel"""
        try:
            # Mock implementation - in production would integrate with actual services
            logger.info(f"Sending {severity.value} notification to {channel}: {message}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def get_active_alerts(self) -> Dict[str, Any]:
        """Get all currently active alerts"""
        return {
            alert_key: {
                "rule_name": alert["rule"].name,
                "triggered_at": alert["triggered_at"],
                "duration_seconds": time.time() - alert["triggered_at"],
                "model_type": alert["metrics"].model_type
            }
            for alert_key, alert in self.active_alerts.items()
        }

    def create_default_alert_rules(self) -> List[AlertRule]:
        """Create default alert rules for transformer resource monitoring"""
        default_rules = [
            AlertRule(
                name="high_memory_usage",
                metric="memory_usage_percent",
                threshold=90.0,
                duration_seconds=300,
                notification_channels=self.notification_channels
            ),
            AlertRule(
                name="high_cpu_utilization", 
                metric="cpu_utilization_percent",
                threshold=85.0,
                duration_seconds=300,
                notification_channels=self.notification_channels
            ),
            AlertRule(
                name="high_response_time",
                metric="response_time_ms",
                threshold=1000.0,
                duration_seconds=180,
                notification_channels=self.notification_channels
            ),
            AlertRule(
                name="high_error_rate",
                metric="error_rate_percent", 
                threshold=5.0,
                duration_seconds=120,
                notification_channels=self.notification_channels
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
        
        logger.info(f"Created {len(default_rules)} default alert rules")
        return default_rules