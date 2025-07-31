"""Model preservation monitoring and observability system.

This module provides comprehensive monitoring capabilities for model preservation operations,
including Prometheus metrics collection, Grafana dashboard configuration, Cloud Monitoring
alerts, and structured logging.
"""
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

import structlog
from prometheus_client import (
    CollectorRegistry, 
    Counter, 
    Histogram, 
    Gauge, 
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import AlertPolicy, NotificationChannel


class MetricNames:
    """Constants for Prometheus metric names."""
    
    MODEL_SAVE_DURATION = "model_save_duration_seconds"
    MODEL_LOAD_DURATION = "model_load_duration_seconds"
    MODEL_CACHE_HIT_RATE = "model_cache_hit_rate"
    PRESERVATION_ERRORS_TOTAL = "preservation_errors_total"
    MODEL_SIZE_BYTES = "model_size_bytes"
    MODELS_PRESERVED_TOTAL = "models_preserved_total"
    CACHE_EVICTIONS_TOTAL = "cache_evictions_total"
    STORAGE_OPERATIONS_TOTAL = "storage_operations_total"


class AlertPolicies:
    """Constants for Cloud Monitoring alert policy configurations."""
    
    SAVE_FAILURE_ALERT = {
        "display_name": "Model Save Failure Rate Alert",
        "conditions": [{
            "display_name": "High save failure rate",
            "condition_threshold": {
                "filter": 'metric.type="custom.googleapis.com/model_preservation/errors"',
                "comparison": "COMPARISON_GREATER_THAN",
                "threshold_value": {"double_value": 5},
                "duration": {"seconds": 300}
            }
        }],
        "alert_strategy": {
            "auto_close": {"seconds": 86400}
        },
        "combiner": "OR"
    }
    
    STORAGE_QUOTA_ALERT = {
        "display_name": "Storage Quota Warning Alert",
        "conditions": [{
            "display_name": "Storage usage approaching limit",
            "condition_threshold": {
                "filter": 'metric.type="gcs.googleapis.com/storage/total_bytes"',
                "comparison": "COMPARISON_GREATER_THAN",
                "threshold_value": {"double_value": 900 * 1024 * 1024 * 1024},  # 900GB
                "duration": {"seconds": 600}
            }
        }],
        "alert_strategy": {
            "auto_close": {"seconds": 86400}
        },
        "combiner": "OR"
    }
    
    PERFORMANCE_DEGRADATION_ALERT = {
        "display_name": "Model Performance Degradation Alert",
        "conditions": [
            {
                "display_name": "High save duration",
                "condition_threshold": {
                    "filter": 'metric.type="custom.googleapis.com/model_preservation/save_duration"',
                    "comparison": "COMPARISON_GREATER_THAN",
                    "threshold_value": {"double_value": 10.0},
                    "duration": {"seconds": 300}
                }
            },
            {
                "display_name": "High load duration",
                "condition_threshold": {
                    "filter": 'metric.type="custom.googleapis.com/model_preservation/load_duration"',
                    "comparison": "COMPARISON_GREATER_THAN",
                    "threshold_value": {"double_value": 5.0},
                    "duration": {"seconds": 300}
                }
            }
        ],
        "alert_strategy": {
            "auto_close": {"seconds": 86400}
        },
        "combiner": "OR"
    }
    
    CACHE_MISS_RATE_ALERT = {
        "display_name": "High Cache Miss Rate Alert",
        "conditions": [{
            "display_name": "Cache miss rate too high",
            "condition_threshold": {
                "filter": 'metric.type="custom.googleapis.com/model_preservation/cache_hit_rate"',
                "comparison": "COMPARISON_LESS_THAN",
                "threshold_value": {"double_value": 0.5},  # 50% hit rate
                "duration": {"seconds": 600}
            }
        }],
        "alert_strategy": {
            "auto_close": {"seconds": 86400}
        },
        "combiner": "OR"
    }


class PreservationMetricsCollector:
    """Prometheus metrics collector for model preservation operations."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector with Prometheus metrics.
        
        Args:
            registry: Prometheus registry to use. If None, creates a new registry.
        """
        # Use a new registry by default to avoid conflicts during testing
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()
        
        # Initialize all metrics
        try:
            self._init_metrics()
        except ValueError as e:
            # Handle metric registration conflicts gracefully
            if "Duplicated timeseries" in str(e) or "already exists" in str(e):
                # Use a fresh registry to avoid conflicts
                self.registry = CollectorRegistry()
                self._init_metrics()
            else:
                raise
        
        # Track cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # Model save duration histogram
        self.model_save_duration = Histogram(
            MetricNames.MODEL_SAVE_DURATION,
            "Time taken to save models to storage",
            ["model_type", "priority", "mode", "success"],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Model load duration histogram
        self.model_load_duration = Histogram(
            MetricNames.MODEL_LOAD_DURATION,
            "Time taken to load models from storage",
            ["model_type", "cache_hit", "success"],
            registry=self.registry,
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Cache hit rate gauge
        self.model_cache_hit_rate = Gauge(
            MetricNames.MODEL_CACHE_HIT_RATE,
            "Model cache hit rate",
            ["model_type"],
            registry=self.registry
        )
        
        # Preservation errors counter
        self.preservation_errors_total = Counter(
            MetricNames.PRESERVATION_ERRORS_TOTAL,
            "Total preservation operation errors",
            ["operation", "model_type", "error_type"],
            registry=self.registry
        )
        
        # Model size gauge
        self.model_size_bytes = Gauge(
            MetricNames.MODEL_SIZE_BYTES,
            "Size of preserved models in bytes",
            ["model_type", "version"],
            registry=self.registry
        )
        
        # Models preserved counter
        self.models_preserved_total = Counter(
            MetricNames.MODELS_PRESERVED_TOTAL,
            "Total number of models preserved",
            ["model_type", "priority"],
            registry=self.registry
        )
        
        # Cache evictions counter
        self.cache_evictions_total = Counter(
            MetricNames.CACHE_EVICTIONS_TOTAL,
            "Total cache evictions",
            ["cache_level", "model_type"],
            registry=self.registry
        )
        
        # Storage operations counter
        self.storage_operations_total = Counter(
            MetricNames.STORAGE_OPERATIONS_TOTAL,
            "Total storage operations",
            ["operation", "success"],
            registry=self.registry
        )
    
    def record_save_duration(
        self, 
        model_type: str, 
        duration: float, 
        model_size: int,
        success: bool,
        priority: str = "medium",
        mode: str = "unknown"
    ):
        """Record model save operation metrics.
        
        Args:
            model_type: Type of model being saved
            duration: Time taken in seconds
            model_size: Size of model in bytes
            success: Whether operation succeeded
            priority: Priority level of the operation
            mode: Operating mode when save occurred
        """
        with self._lock:
            # Record save duration
            self.model_save_duration.labels(
                model_type=model_type,
                priority=priority,
                mode=mode,
                success=str(success)
            ).observe(duration)
            
            # Record model size
            self.model_size_bytes.labels(
                model_type=model_type,
                version="latest"
            ).set(model_size)
            
            # Increment preserved models counter if successful
            if success:
                self.models_preserved_total.labels(
                    model_type=model_type,
                    priority=priority
                ).inc()
    
    def record_load_duration(
        self,
        model_type: str,
        duration: float,
        cache_hit: bool,
        success: bool
    ):
        """Record model load operation metrics.
        
        Args:
            model_type: Type of model being loaded
            duration: Time taken in seconds
            cache_hit: Whether this was a cache hit
            success: Whether operation succeeded
        """
        with self._lock:
            # Record load duration
            self.model_load_duration.labels(
                model_type=model_type,
                cache_hit=str(cache_hit),
                success=str(success)
            ).observe(duration)
            
            # Update cache statistics
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            
            # Update cache hit rate
            total_requests = self._cache_hits + self._cache_misses
            if total_requests > 0:
                hit_rate = self._cache_hits / total_requests
                self.model_cache_hit_rate.labels(
                    model_type=model_type
                ).set(hit_rate)
    
    def record_cache_hit(self, model_type: str):
        """Record a cache hit event.
        
        Args:
            model_type: Type of model that was cached
        """
        with self._lock:
            self._cache_hits += 1
            total_requests = self._cache_hits + self._cache_misses
            if total_requests > 0:
                hit_rate = self._cache_hits / total_requests
                self.model_cache_hit_rate.labels(
                    model_type=model_type
                ).set(hit_rate)
    
    def record_cache_miss(self, model_type: str):
        """Record a cache miss event.
        
        Args:
            model_type: Type of model that missed the cache
        """
        with self._lock:
            self._cache_misses += 1
            total_requests = self._cache_hits + self._cache_misses
            if total_requests > 0:
                hit_rate = self._cache_hits / total_requests
                self.model_cache_hit_rate.labels(
                    model_type=model_type
                ).set(hit_rate)
    
    def record_cache_eviction(self, model_type: str, size_bytes: int, cache_level: str = "memory"):
        """Record a cache eviction event.
        
        Args:
            model_type: Type of model that was evicted
            size_bytes: Size of evicted model
            cache_level: Level of cache (memory, disk, etc.)
        """
        self.cache_evictions_total.labels(
            cache_level=cache_level,
            model_type=model_type
        ).inc()
    
    def record_error(
        self,
        operation: str,
        model_type: str,
        error_type: str
    ):
        """Record a preservation error.
        
        Args:
            operation: Type of operation that failed
            model_type: Type of model involved
            error_type: Category of error
        """
        self.preservation_errors_total.labels(
            operation=operation,
            model_type=model_type,
            error_type=error_type
        ).inc()
    
    def record_storage_operation(
        self,
        operation: str,
        size_bytes: int,
        duration: float,
        success: bool
    ):
        """Record a storage operation.
        
        Args:
            operation: Type of storage operation
            size_bytes: Size of data operated on
            duration: Time taken for operation
            success: Whether operation succeeded
        """
        self.storage_operations_total.labels(
            operation=operation,
            success=str(success)
        ).inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing current metrics summary
        """
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
            
            # Get current metric values
            total_saves = sum(
                sample.value for sample in self.models_preserved_total.collect()[0].samples
            )
            
            total_errors = sum(
                sample.value for sample in self.preservation_errors_total.collect()[0].samples
            )
            
            error_rate = total_errors / (total_saves + total_errors) if (total_saves + total_errors) > 0 else 0.0
            
            return {
                "total_saves": int(total_saves),
                "total_loads": total_requests,
                "cache_hit_rate": cache_hit_rate,
                "error_rate": error_rate,
                "total_errors": int(total_errors),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses
            }


class GrafanaDashboardConfig:
    """Grafana dashboard configuration generator for model preservation metrics."""
    
    def __init__(self):
        """Initialize dashboard configuration."""
        self.dashboard_json = {
            "dashboard": {
                "title": "Model Preservation Metrics",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        self._panel_id = 1
    
    def add_metrics_panel(
        self,
        title: str,
        metric_query: str,
        panel_type: str = "graph",
        unit: str = "short",
        width: int = 12,
        height: int = 8
    ):
        """Add a metrics panel to the dashboard.
        
        Args:
            title: Panel title
            metric_query: Prometheus query for the metric
            panel_type: Type of panel (graph, stat, etc.)
            unit: Unit for the metric
            width: Panel width
            height: Panel height
        """
        panel = {
            "id": self._panel_id,
            "title": title,
            "type": panel_type,
            "gridPos": {
                "h": height,
                "w": width,
                "x": 0,
                "y": 0
            },
            "targets": [{
                "expr": metric_query,
                "refId": "A"
            }],
            "fieldConfig": {
                "defaults": {
                    "unit": unit
                }
            }
        }
        
        if panel_type == "graph":
            panel["xAxis"] = {"show": True}
            panel["yAxes"] = [{
                "label": unit,
                "show": True
            }]
        
        self.dashboard_json["dashboard"]["panels"].append(panel)
        self._panel_id += 1
    
    def create_preservation_dashboard(self) -> Dict[str, Any]:
        """Create complete preservation monitoring dashboard.
        
        Returns:
            Complete Grafana dashboard configuration
        """
        # Reset panels
        self.dashboard_json["dashboard"]["panels"] = []
        self._panel_id = 1
        
        # Model Save Duration panel
        self.add_metrics_panel(
            title="Model Save Duration",
            metric_query=f"rate({MetricNames.MODEL_SAVE_DURATION}[5m])",
            panel_type="graph",
            unit="seconds"
        )
        
        # Model Load Duration panel
        self.add_metrics_panel(
            title="Model Load Duration",
            metric_query=f"rate({MetricNames.MODEL_LOAD_DURATION}[5m])",
            panel_type="graph",
            unit="seconds"
        )
        
        # Cache Hit Rate panel
        self.add_metrics_panel(
            title="Cache Hit Rate",
            metric_query=f"{MetricNames.MODEL_CACHE_HIT_RATE}",
            panel_type="stat",
            unit="percentunit"
        )
        
        # Preservation Errors panel
        self.add_metrics_panel(
            title="Preservation Errors",
            metric_query=f"rate({MetricNames.PRESERVATION_ERRORS_TOTAL}[5m])",
            panel_type="graph",
            unit="ops"
        )
        
        # Model Sizes panel
        self.add_metrics_panel(
            title="Model Sizes",
            metric_query=f"{MetricNames.MODEL_SIZE_BYTES}",
            panel_type="graph",
            unit="bytes"
        )
        
        # Storage Operations panel
        self.add_metrics_panel(
            title="Storage Operations",
            metric_query=f"rate({MetricNames.STORAGE_OPERATIONS_TOTAL}[5m])",
            panel_type="graph",
            unit="ops"
        )
        
        # Add alert configurations to panels
        for panel in self.dashboard_json["dashboard"]["panels"]:
            if "Errors" in panel["title"] or "Duration" in panel["title"]:
                panel["alert"] = {
                    "enabled": True,
                    "conditions": [
                        {
                            "query": {"refId": "A"},
                            "reducer": {"type": "avg"},
                            "evaluator": {
                                "params": [5.0] if "Duration" in panel["title"] else [1.0],
                                "type": "gt"
                            }
                        }
                    ]
                }
        
        return self.dashboard_json["dashboard"]
    
    def export_dashboard_json(self, dashboard: Dict[str, Any]) -> str:
        """Export dashboard configuration as JSON string.
        
        Args:
            dashboard: Dashboard configuration dictionary
            
        Returns:
            JSON string representation of dashboard
        """
        return json.dumps(dashboard, indent=2)
    
    def validate_dashboard_config(self, dashboard: Dict[str, Any]) -> bool:
        """Validate dashboard configuration structure.
        
        Args:
            dashboard: Dashboard configuration to validate
            
        Returns:
            True if configuration is valid
        """
        required_fields = ["title", "panels"]
        
        for field in required_fields:
            if field not in dashboard:
                return False
        
        # Validate panels structure
        for panel in dashboard["panels"]:
            panel_required = ["id", "title", "type", "targets"]
            for field in panel_required:
                if field not in panel:
                    return False
        
        return True


class CloudMonitoringAlerter:
    """Cloud Monitoring alert policy management."""
    
    def __init__(self, project_id: str):
        """Initialize Cloud Monitoring alerter.
        
        Args:
            project_id: Google Cloud project ID
        """
        self.project_id = project_id
        self.client = monitoring_v3.AlertPolicyServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def create_save_failure_alert(
        self,
        threshold: int = 5,
        duration_minutes: int = 5,
        notification_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create alert policy for model save failures.
        
        Args:
            threshold: Number of failures to trigger alert
            duration_minutes: Duration to evaluate threshold over
            notification_channels: List of notification channel names
            
        Returns:
            Created alert policy configuration
        """
        policy_config = AlertPolicies.SAVE_FAILURE_ALERT.copy()
        
        # Update threshold and duration
        condition = policy_config["conditions"][0]
        condition["condition_threshold"]["threshold_value"]["double_value"] = threshold
        condition["condition_threshold"]["duration"]["seconds"] = duration_minutes * 60
        
        # Add notification channels if provided
        if notification_channels:
            policy_config["notification_channels"] = notification_channels
        
        return policy_config
    
    def create_storage_quota_alert(
        self,
        threshold_gb: int = 900,
        notification_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create alert policy for storage quota warnings.
        
        Args:
            threshold_gb: Storage threshold in GB to trigger alert
            notification_channels: List of notification channel names
            
        Returns:
            Created alert policy configuration
        """
        policy_config = AlertPolicies.STORAGE_QUOTA_ALERT.copy()
        
        # Convert GB to bytes
        threshold_bytes = threshold_gb * 1024 * 1024 * 1024
        
        # Update threshold
        condition = policy_config["conditions"][0]
        condition["condition_threshold"]["threshold_value"]["double_value"] = threshold_bytes
        
        # Add notification channels if provided
        if notification_channels:
            policy_config["notification_channels"] = notification_channels
        
        return policy_config
    
    def create_performance_degradation_alert(
        self,
        save_duration_threshold: float = 10.0,
        load_duration_threshold: float = 5.0,
        notification_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create alert policy for performance degradation.
        
        Args:
            save_duration_threshold: Save duration threshold in seconds
            load_duration_threshold: Load duration threshold in seconds
            notification_channels: List of notification channel names
            
        Returns:
            Created alert policy configuration
        """
        policy_config = AlertPolicies.PERFORMANCE_DEGRADATION_ALERT.copy()
        
        # Update thresholds
        policy_config["conditions"][0]["condition_threshold"]["threshold_value"]["double_value"] = save_duration_threshold
        policy_config["conditions"][1]["condition_threshold"]["threshold_value"]["double_value"] = load_duration_threshold
        
        # Add notification channels if provided
        if notification_channels:
            policy_config["notification_channels"] = notification_channels
        
        return policy_config
    
    def deploy_all_alert_policies(self) -> List[Dict[str, Any]]:
        """Deploy all standard alert policies.
        
        Returns:
            List of deployed alert policy configurations
        """
        deployed_policies = []
        
        # Deploy save failure alert
        save_alert = self.create_save_failure_alert()
        try:
            created_save = self.client.create_alert_policy(
                parent=self.project_name,
                alert_policy=save_alert
            )
            deployed_policies.append({"name": created_save.name, "type": "save_failure"})
        except Exception as e:
            # In real implementation, would log the error
            deployed_policies.append({"error": str(e), "type": "save_failure"})
        
        # Deploy storage quota alert
        quota_alert = self.create_storage_quota_alert()
        try:
            created_quota = self.client.create_alert_policy(
                parent=self.project_name,
                alert_policy=quota_alert
            )
            deployed_policies.append({"name": created_quota.name, "type": "storage_quota"})
        except Exception as e:
            deployed_policies.append({"error": str(e), "type": "storage_quota"})
        
        # Deploy performance degradation alert
        perf_alert = self.create_performance_degradation_alert()
        try:
            created_perf = self.client.create_alert_policy(
                parent=self.project_name,
                alert_policy=perf_alert
            )
            deployed_policies.append({"name": created_perf.name, "type": "performance_degradation"})
        except Exception as e:
            deployed_policies.append({"error": str(e), "type": "performance_degradation"})
        
        return deployed_policies
    
    def list_existing_policies(self) -> List[Dict[str, Any]]:
        """List existing alert policies.
        
        Returns:
            List of existing alert policies
        """
        try:
            policies = list(self.client.list_alert_policies(parent=self.project_name))
            result = []
            for policy in policies:
                # Handle both dict and object-based policies (for testing vs real API)
                if isinstance(policy, dict):
                    result.append({
                        "name": policy.get("name", ""),
                        "display_name": policy.get("display_name", ""),
                        "enabled": policy.get("enabled", True)
                    })
                else:
                    result.append({
                        "name": getattr(policy, 'name', ''),
                        "display_name": getattr(policy, 'display_name', ''),
                        "enabled": getattr(policy, 'enabled', True)
                    })
            return result
        except Exception as e:
            return [{"error": str(e)}]
    
    def update_alert_policy(
        self,
        policy_name: str,
        new_threshold: Union[int, float]
    ) -> Dict[str, Any]:
        """Update existing alert policy threshold.
        
        Args:
            policy_name: Name of policy to update
            new_threshold: New threshold value
            
        Returns:
            Updated alert policy configuration
        """
        try:
            # Get existing policy
            policy = self.client.get_alert_policy(name=policy_name)
            
            # Update threshold in first condition
            if policy.conditions:
                policy.conditions[0].condition_threshold.threshold_value.double_value = new_threshold
            
            # Update the policy
            updated_policy = self.client.update_alert_policy(alert_policy=policy)
            
            return {
                "name": updated_policy.name,
                "display_name": updated_policy.display_name,
                "updated_threshold": new_threshold
            }
        except Exception as e:
            return {"error": str(e)}


class StructuredLogger:
    """Structured logging for model preservation operations."""
    
    def __init__(
        self,
        component: str,
        log_level: str = "INFO",
        format_json: bool = True
    ):
        """Initialize structured logger.
        
        Args:
            component: Component name for logging context
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            format_json: Whether to format logs as JSON
        """
        self.component = component
        self.logger = structlog.get_logger(component)
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if format_json else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        if format_json:
            self._json_formatter = True
        
        # Set log level
        import logging
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        self.logger.setLevel(level_map.get(log_level.upper(), logging.INFO))
    
    def log_save_operation(
        self,
        model_type: str,
        version: str,
        duration: float,
        size_bytes: int,
        success: bool,
        storage_path: str,
        **context
    ):
        """Log model save operation.
        
        Args:
            model_type: Type of model saved
            version: Model version
            duration: Operation duration in seconds
            size_bytes: Model size in bytes
            success: Whether operation succeeded
            storage_path: Storage location path
            **context: Additional context information
        """
        log_data = {
            "operation": "save",
            "model_type": model_type,
            "version": version,
            "duration_seconds": duration,
            "size_bytes": size_bytes,
            "success": success,
            "storage_path": storage_path,
            **context
        }
        
        if success:
            self.logger.info("Model save completed", **log_data)
        else:
            self.logger.error("Model save failed", **log_data)
    
    def log_load_operation(
        self,
        model_type: str,
        version: str,
        duration: float,
        cache_hit: bool,
        success: bool,
        **context
    ):
        """Log model load operation.
        
        Args:
            model_type: Type of model loaded
            version: Model version
            duration: Operation duration in seconds
            cache_hit: Whether this was a cache hit
            success: Whether operation succeeded
            **context: Additional context information
        """
        log_data = {
            "operation": "load",
            "model_type": model_type,
            "version": version,
            "duration_seconds": duration,
            "cache_hit": cache_hit,
            "success": success,
            **context
        }
        
        if success:
            self.logger.info("Model load completed", **log_data)
        else:
            self.logger.error("Model load failed", **log_data)
    
    def log_error(
        self,
        operation: str,
        model_type: str,
        version: str,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        **context
    ):
        """Log preservation error.
        
        Args:
            operation: Type of operation that failed
            model_type: Type of model involved
            version: Model version
            error_type: Category of error
            error_message: Error message
            traceback: Error traceback if available
            **context: Additional context information
        """
        log_data = {
            "operation": operation,
            "model_type": model_type,
            "version": version,
            "error_type": error_type,
            "error_message": error_message,
            **context
        }
        
        if traceback:
            log_data["traceback"] = traceback
        
        self.logger.error("Preservation operation failed", **log_data)
    
    def log_cache_event(
        self,
        event_type: str,
        model_type: str,
        cache_level: str,
        size_bytes: Optional[int] = None,
        **context
    ):
        """Log cache event.
        
        Args:
            event_type: Type of cache event (hit, miss, eviction)
            model_type: Type of model involved
            cache_level: Level of cache (memory, disk)
            size_bytes: Size of cached item if applicable
            **context: Additional context information
        """
        log_data = {
            "event_type": event_type,
            "model_type": model_type,
            "cache_level": cache_level,
            **context
        }
        
        if size_bytes is not None:
            log_data["size_bytes"] = size_bytes
        
        self.logger.debug("Cache event", **log_data)
    
    def log_metrics_export(
        self,
        metrics_summary: Dict[str, Any],
        export_timestamp: datetime,
        export_destination: str,
        **context
    ):
        """Log metrics export event.
        
        Args:
            metrics_summary: Summary of exported metrics
            export_timestamp: When export occurred
            export_destination: Where metrics were exported
            **context: Additional context information
        """
        log_data = {
            "metrics_summary": metrics_summary,
            "export_timestamp": export_timestamp.isoformat(),
            "export_destination": export_destination,
            **context
        }
        
        self.logger.info("Metrics exported", **log_data)