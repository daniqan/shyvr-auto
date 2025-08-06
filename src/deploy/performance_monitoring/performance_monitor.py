"""
Performance Monitor

Real-time performance monitoring system for production environments.
Provides continuous monitoring, alerting, and performance tracking.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringState(Enum):
    """Performance monitoring states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    monitoring_interval_seconds: int = 10
    metrics_retention_minutes: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    enable_real_time_alerts: bool = True
    enable_trend_analysis: bool = True
    max_concurrent_monitors: int = 5


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    active_connections: int
    queue_length: int
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Performance alert rule"""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[float] = None


class PerformanceMonitor:
    """
    Real-time Performance Monitor
    
    Provides continuous monitoring of system performance metrics
    with configurable alerting and trend analysis.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.state = MonitoringState.STOPPED
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_rules: List[AlertRule] = []
        self.alerts_triggered: List[Dict[str, Any]] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        logger.info("Performance monitor initialized")
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name="response_time_ms",
                threshold=500.0,
                comparison="gt",
                severity="high",
                cooldown_minutes=2
            ),
            AlertRule(
                name="low_throughput",
                metric_name="throughput_rps",
                threshold=10.0,
                comparison="lt",
                severity="medium",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                threshold=0.05,
                comparison="gt",
                severity="critical",
                cooldown_minutes=1
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_utilization",
                threshold=90.0,
                comparison="gt",
                severity="high",
                cooldown_minutes=3
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_utilization",
                threshold=95.0,
                comparison="gt",
                severity="critical",
                cooldown_minutes=2
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    async def start_monitoring(self) -> Dict[str, Any]:
        """Start performance monitoring"""
        if self.state == MonitoringState.RUNNING:
            return {"success": False, "message": "Monitoring already running"}
        
        try:
            self.state = MonitoringState.STARTING
            
            # Start monitoring loop
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.state = MonitoringState.RUNNING
            logger.info("Performance monitoring started")
            
            return {
                "success": True,
                "state": self.state.value,
                "config": {
                    "interval_seconds": self.config.monitoring_interval_seconds,
                    "retention_minutes": self.config.metrics_retention_minutes,
                    "alert_rules_count": len(self.alert_rules)
                }
            }
            
        except Exception as e:
            self.state = MonitoringState.ERROR
            logger.error(f"Failed to start monitoring: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop performance monitoring"""
        if self.state == MonitoringState.STOPPED:
            return {"success": True, "message": "Monitoring already stopped"}
        
        try:
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.state = MonitoringState.STOPPED
            logger.info("Performance monitoring stopped")
            
            return {"success": True, "state": self.state.value}
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return {"success": False, "error": str(e)}
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.state == MonitoringState.RUNNING:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check alert rules
                await self._check_alert_rules(metrics)
                
                # Notify callbacks
                await self._notify_metrics_callbacks(metrics)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # Simulate metric collection (in real implementation, this would
        # collect from actual system monitoring APIs)
        current_time = time.time()
        
        # Simulate realistic metrics with some variability
        base_response_time = 80 + (time.time() % 100)  # 80-180ms
        base_throughput = 150 + (time.time() % 50)     # 150-200 rps
        base_error_rate = 0.01 + (time.time() % 0.02)  # 0.01-0.03
        base_cpu = 60 + (time.time() % 30)             # 60-90%
        base_memory = 50 + (time.time() % 40)          # 50-90%
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            response_time_ms=base_response_time,
            throughput_rps=base_throughput,
            error_rate=base_error_rate,
            cpu_utilization=base_cpu,
            memory_utilization=base_memory,
            active_connections=int(100 + (time.time() % 200)),
            queue_length=int(time.time() % 10),
            custom_metrics={}
        )
        
        return metrics
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history"""
        self.metrics_history.append(metrics)
        
        # Limit history size based on retention policy
        retention_seconds = self.config.metrics_retention_minutes * 60
        cutoff_time = time.time() - retention_seconds
        
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
    
    async def _check_alert_rules(self, metrics: PerformanceMetrics):
        """Check alert rules against current metrics"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.last_triggered and 
                current_time - rule.last_triggered < rule.cooldown_minutes * 60):
                continue
            
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is None:
                continue
            
            # Check threshold
            alert_triggered = False
            if rule.comparison == "gt" and metric_value > rule.threshold:
                alert_triggered = True
            elif rule.comparison == "lt" and metric_value < rule.threshold:
                alert_triggered = True
            elif rule.comparison == "eq" and abs(metric_value - rule.threshold) < 0.001:
                alert_triggered = True
            
            if alert_triggered:
                await self._trigger_alert(rule, metrics, metric_value)
                rule.last_triggered = current_time
    
    async def _trigger_alert(self, rule: AlertRule, metrics: PerformanceMetrics, metric_value: float):
        """Trigger a performance alert"""
        alert = {
            "alert_id": f"{rule.name}_{int(time.time())}",
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "metric_value": metric_value,
            "threshold": rule.threshold,
            "severity": rule.severity,
            "timestamp": metrics.timestamp,
            "message": f"{rule.name}: {rule.metric_name} ({metric_value}) exceeds threshold ({rule.threshold})"
        }
        
        self.alerts_triggered.append(alert)
        
        # Notify alert callbacks
        await self._notify_alert_callbacks(alert)
        
        logger.warning(f"Performance alert triggered: {alert['message']}")
    
    async def _notify_metrics_callbacks(self, metrics: PerformanceMetrics):
        """Notify registered metrics callbacks"""
        for callback in self.metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    async def _notify_alert_callbacks(self, alert: Dict[str, Any]):
        """Notify registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        retention_seconds = self.config.metrics_retention_minutes * 60
        cutoff_time = time.time() - retention_seconds
        
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # Also clean up old alerts (keep last 100)
        if len(self.alerts_triggered) > 100:
            self.alerts_triggered = self.alerts_triggered[-100:]
    
    def register_metrics_callback(self, callback: Callable):
        """Register callback for metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_history(self, time_range_minutes: int = 30) -> List[PerformanceMetrics]:
        """Get metrics history for specified time range"""
        cutoff_time = time.time() - (time_range_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_recent_alerts(self, time_range_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts within time range"""
        cutoff_time = time.time() - (time_range_minutes * 60)
        return [
            alert for alert in self.alerts_triggered
            if alert["timestamp"] >= cutoff_time
        ]
    
    def add_alert_rule(self, rule: AlertRule) -> Dict[str, Any]:
        """Add custom alert rule"""
        try:
            self.alert_rules.append(rule)
            logger.info(f"Added alert rule: {rule.name}")
            return {"success": True, "rule_added": rule.name}
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return {"success": False, "error": str(e)}
    
    def remove_alert_rule(self, rule_name: str) -> Dict[str, Any]:
        """Remove alert rule by name"""
        original_count = len(self.alert_rules)
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        
        if len(self.alert_rules) < original_count:
            logger.info(f"Removed alert rule: {rule_name}")
            return {"success": True, "rule_removed": rule_name}
        else:
            return {"success": False, "error": "Rule not found"}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "state": self.state.value,
            "metrics_count": len(self.metrics_history),
            "alert_rules_count": len(self.alert_rules),
            "recent_alerts_count": len(self.get_recent_alerts(60)),
            "uptime_seconds": time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time()),
            "last_collection": self.metrics_history[-1].timestamp if self.metrics_history else None
        }
    
    async def get_performance_summary(self, time_range_minutes: int = 30) -> Dict[str, Any]:
        """Get performance summary for time range"""
        metrics_data = self.get_metrics_history(time_range_minutes)
        
        if not metrics_data:
            return {"error": "No metrics data available"}
        
        # Calculate summary statistics
        response_times = [m.response_time_ms for m in metrics_data]
        throughputs = [m.throughput_rps for m in metrics_data]
        error_rates = [m.error_rate for m in metrics_data]
        cpu_utils = [m.cpu_utilization for m in metrics_data]
        memory_utils = [m.memory_utilization for m in metrics_data]
        
        summary = {
            "time_range_minutes": time_range_minutes,
            "data_points": len(metrics_data),
            "response_time": {
                "avg": statistics.mean(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p95": sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 0 else 0
            },
            "throughput": {
                "avg": statistics.mean(throughputs),
                "min": min(throughputs),
                "max": max(throughputs)
            },
            "error_rate": {
                "avg": statistics.mean(error_rates),
                "max": max(error_rates)
            },
            "resource_utilization": {
                "cpu_avg": statistics.mean(cpu_utils),
                "cpu_max": max(cpu_utils),
                "memory_avg": statistics.mean(memory_utils),
                "memory_max": max(memory_utils)
            },
            "alerts_in_period": len([
                a for a in self.alerts_triggered
                if a["timestamp"] >= time.time() - (time_range_minutes * 60)
            ])
        }
        
        return summary