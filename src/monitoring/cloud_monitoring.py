"""
Google Cloud Monitoring integration for Shyvr RLTE trading system.

This module provides comprehensive integration with Google Cloud Monitoring,
including custom metrics collection, alerting policies, and notification channels.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from contextlib import asynccontextmanager

from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboard import v1 as dashboard_v1
from google.api_core import exceptions as gcp_exceptions

from .base import MetricsCollector, MetricsRegistry


logger = logging.getLogger(__name__)


@dataclass
class CloudMetricDescriptor:
    """Descriptor for a custom Cloud Monitoring metric."""
    
    name: str
    display_name: str
    description: str
    metric_kind: str  # GAUGE, CUMULATIVE, DELTA
    value_type: str   # DOUBLE, INT64, BOOL, STRING
    unit: str = ""
    labels: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []


@dataclass
class AlertPolicyCondition:
    """Configuration for an alert policy condition."""
    
    filter: str
    comparison: str  # COMPARISON_GT, COMPARISON_LT, etc.
    threshold_value: float
    duration: str = "300s"  # Duration the condition must be met
    aggregation_alignment_period: str = "60s"
    aggregation_per_series_aligner: str = "ALIGN_RATE"
    aggregation_cross_series_reducer: str = "REDUCE_MEAN"


@dataclass
class NotificationChannel:
    """Configuration for notification channels."""
    
    type: str  # email, slack, webhook, sms
    display_name: str
    labels: Dict[str, str]
    enabled: bool = True


class CloudMonitoringClient:
    """
    Google Cloud Monitoring client for managing custom metrics and alerts.
    
    This class provides a high-level interface for:
    - Creating custom metric descriptors
    - Writing time series data
    - Managing alert policies
    - Setting up notification channels
    """
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize Cloud Monitoring client.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials (optional)
        """
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        
        try:
            self.client = monitoring_v3.MetricServiceClient()
            self.alert_client = monitoring_v3.AlertPolicyServiceClient()
            self.notification_client = monitoring_v3.NotificationChannelServiceClient()
            self.dashboard_client = dashboard_v1.DashboardsServiceClient()
            
            logger.info(f"Initialized Cloud Monitoring client for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Monitoring client: {e}")
            raise
    
    async def create_metric_descriptor(self, descriptor: CloudMetricDescriptor) -> bool:
        """
        Create a custom metric descriptor in Cloud Monitoring.
        
        Args:
            descriptor: Metric descriptor configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metric_descriptor = monitoring_v3.MetricDescriptor(
                type=f"custom.googleapis.com/shyvr_rlte/{descriptor.name}",
                metric_kind=getattr(monitoring_v3.MetricDescriptor.MetricKind, descriptor.metric_kind),
                value_type=getattr(monitoring_v3.MetricDescriptor.ValueType, descriptor.value_type),
                display_name=descriptor.display_name,
                description=descriptor.description,
                unit=descriptor.unit,
                labels=[
                    monitoring_v3.LabelDescriptor(
                        key=label["key"],
                        value_type=getattr(
                            monitoring_v3.LabelDescriptor.ValueType, 
                            label.get("value_type", "STRING")
                        ),
                        description=label.get("description", "")
                    )
                    for label in descriptor.labels
                ]
            )
            
            created_descriptor = self.client.create_metric_descriptor(
                name=self.project_name,
                metric_descriptor=metric_descriptor
            )
            
            logger.info(f"Created metric descriptor: {created_descriptor.type}")
            return True
            
        except gcp_exceptions.AlreadyExists:
            logger.info(f"Metric descriptor already exists: {descriptor.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create metric descriptor {descriptor.name}: {e}")
            return False
    
    async def write_time_series(
        self, 
        metric_type: str, 
        value: Union[float, int], 
        labels: Dict[str, str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Write a time series data point to Cloud Monitoring.
        
        Args:
            metric_type: Custom metric type name
            value: Metric value
            labels: Metric labels
            timestamp: Data point timestamp (defaults to now)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            if labels is None:
                labels = {}
            
            # Convert timestamp to protobuf Timestamp
            timestamp_pb = monitoring_v3.TimeInterval()
            timestamp_pb.end_time.seconds = int(timestamp.timestamp())
            timestamp_pb.end_time.nanos = int((timestamp.timestamp() % 1) * 1e9)
            
            # Create time series point
            point = monitoring_v3.Point(
                interval=timestamp_pb,
                value=monitoring_v3.TypedValue(double_value=float(value))
            )
            
            # Create resource and metric labels
            resource = monitoring_v3.MonitoredResource(
                type="gce_instance",  # or "global" for global resources
                labels={"instance_id": "shyvr-rlte", "zone": "us-central1-a"}
            )
            
            metric = monitoring_v3.Metric(
                type=f"custom.googleapis.com/shyvr_rlte/{metric_type}",
                labels=labels
            )
            
            series = monitoring_v3.TimeSeries(
                metric=metric,
                resource=resource,
                points=[point]
            )
            
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
            
            logger.debug(f"Wrote time series data: {metric_type} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write time series data for {metric_type}: {e}")
            return False
    
    async def create_alert_policy(
        self,
        display_name: str,
        conditions: List[AlertPolicyCondition],
        notification_channels: List[str],
        documentation: str = "",
        enabled: bool = True
    ) -> Optional[str]:
        """
        Create an alert policy in Cloud Monitoring.
        
        Args:
            display_name: Alert policy display name
            conditions: List of alert conditions
            notification_channels: List of notification channel IDs
            documentation: Documentation text for the alert
            enabled: Whether the alert policy is enabled
            
        Returns:
            Alert policy name if successful, None otherwise
        """
        try:
            # Convert conditions to protobuf format
            alert_conditions = []
            for i, condition in enumerate(conditions):
                alert_condition = monitoring_v3.AlertPolicy.Condition(
                    display_name=f"{display_name} Condition {i+1}",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=condition.filter,
                        comparison=getattr(
                            monitoring_v3.ComparisonType, 
                            condition.comparison
                        ),
                        threshold_value=condition.threshold_value,
                        duration={"seconds": int(condition.duration.rstrip('s'))},
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": int(condition.aggregation_alignment_period.rstrip('s'))},
                                per_series_aligner=getattr(
                                    monitoring_v3.Aggregation.Aligner,
                                    condition.aggregation_per_series_aligner
                                ),
                                cross_series_reducer=getattr(
                                    monitoring_v3.Aggregation.Reducer,
                                    condition.aggregation_cross_series_reducer
                                )
                            )
                        ]
                    )
                )
                alert_conditions.append(alert_condition)
            
            # Create alert policy
            policy = monitoring_v3.AlertPolicy(
                display_name=display_name,
                conditions=alert_conditions,
                notification_channels=notification_channels,
                documentation=monitoring_v3.AlertPolicy.Documentation(
                    content=documentation,
                    mime_type="text/markdown"
                ),
                enabled=enabled,
                combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND
            )
            
            created_policy = self.alert_client.create_alert_policy(
                name=self.project_name,
                alert_policy=policy
            )
            
            logger.info(f"Created alert policy: {created_policy.name}")
            return created_policy.name
            
        except Exception as e:
            logger.error(f"Failed to create alert policy {display_name}: {e}")
            return None
    
    async def create_notification_channel(self, channel: NotificationChannel) -> Optional[str]:
        """
        Create a notification channel.
        
        Args:
            channel: Notification channel configuration
            
        Returns:
            Channel name if successful, None otherwise
        """
        try:
            notification_channel = monitoring_v3.NotificationChannel(
                type=f"email" if channel.type == "email" else channel.type,
                display_name=channel.display_name,
                labels=channel.labels,
                enabled=channel.enabled
            )
            
            created_channel = self.notification_client.create_notification_channel(
                name=self.project_name,
                notification_channel=notification_channel
            )
            
            logger.info(f"Created notification channel: {created_channel.name}")
            return created_channel.name
            
        except Exception as e:
            logger.error(f"Failed to create notification channel {channel.display_name}: {e}")
            return None
    
    async def list_alert_policies(self) -> List[Dict[str, Any]]:
        """List all alert policies for the project."""
        try:
            policies = []
            for policy in self.alert_client.list_alert_policies(name=self.project_name):
                policies.append({
                    "name": policy.name,
                    "display_name": policy.display_name,
                    "enabled": policy.enabled,
                    "conditions_count": len(policy.conditions),
                    "notification_channels": list(policy.notification_channels)
                })
            return policies
            
        except Exception as e:
            logger.error(f"Failed to list alert policies: {e}")
            return []
    
    async def delete_alert_policy(self, policy_name: str) -> bool:
        """Delete an alert policy."""
        try:
            self.alert_client.delete_alert_policy(name=policy_name)
            logger.info(f"Deleted alert policy: {policy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete alert policy {policy_name}: {e}")
            return False


class CloudMonitoringCollector(MetricsCollector):
    """
    Metrics collector that integrates with Google Cloud Monitoring.
    
    This collector automatically sends metrics to both Prometheus (for Grafana)
    and Cloud Monitoring (for GCP-native monitoring and alerting).
    """
    
    def __init__(self, registry: MetricsRegistry, project_id: str):
        """
        Initialize the Cloud Monitoring collector.
        
        Args:
            registry: Prometheus metrics registry
            project_id: GCP project ID
        """
        super().__init__(registry)
        self.project_id = project_id
        self.cloud_client = CloudMonitoringClient(project_id)
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._last_flush_time = time.time()
        self._flush_interval = 60  # Flush every 60 seconds
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info("Initialized Cloud Monitoring collector")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring the monitoring system."""
        self.cloud_metrics_sent = self.registry.get_counter(
            "cloud_monitoring_metrics_sent_total",
            "Number of metrics sent to Cloud Monitoring",
            ["metric_type", "status"]
        )
        
        self.cloud_alerts_created = self.registry.get_counter(
            "cloud_monitoring_alerts_created_total",
            "Number of alert policies created",
            ["status"]
        )
        
        self.cloud_monitoring_errors = self.registry.get_counter(
            "cloud_monitoring_errors_total",
            "Number of Cloud Monitoring errors",
            ["operation", "error_type"]
        )
    
    async def record_trading_metric(
        self, 
        metric_name: str, 
        value: Union[float, int, Decimal], 
        labels: Dict[str, str] = None
    ):
        """
        Record a trading-related metric to both Prometheus and Cloud Monitoring.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        if labels is None:
            labels = {}
        
        # Convert Decimal to float for Cloud Monitoring
        if isinstance(value, Decimal):
            value = float(value)
        
        # Buffer metric for batch sending
        self._metrics_buffer.append({
            "metric_type": f"trading/{metric_name}",
            "value": value,
            "labels": labels,
            "timestamp": datetime.utcnow()
        })
        
        # Flush if buffer is full or time interval exceeded
        if (len(self._metrics_buffer) >= 100 or 
            time.time() - self._last_flush_time >= self._flush_interval):
            await self._flush_metrics()
    
    async def record_safety_metric(
        self, 
        metric_name: str, 
        value: Union[float, int], 
        labels: Dict[str, str] = None
    ):
        """Record a safety/risk-related metric."""
        if labels is None:
            labels = {}
        
        self._metrics_buffer.append({
            "metric_type": f"safety/{metric_name}",
            "value": value,
            "labels": labels,
            "timestamp": datetime.utcnow()
        })
        
        if (len(self._metrics_buffer) >= 100 or 
            time.time() - self._last_flush_time >= self._flush_interval):
            await self._flush_metrics()
    
    async def record_system_metric(
        self, 
        metric_name: str, 
        value: Union[float, int], 
        labels: Dict[str, str] = None
    ):
        """Record a system health metric."""
        if labels is None:
            labels = {}
        
        self._metrics_buffer.append({
            "metric_type": f"system/{metric_name}",
            "value": value,
            "labels": labels,
            "timestamp": datetime.utcnow()
        })
        
        if (len(self._metrics_buffer) >= 100 or 
            time.time() - self._last_flush_time >= self._flush_interval):
            await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush buffered metrics to Cloud Monitoring."""
        if not self._metrics_buffer:
            return
        
        metrics_to_send = self._metrics_buffer.copy()
        self._metrics_buffer.clear()
        self._last_flush_time = time.time()
        
        success_count = 0
        error_count = 0
        
        for metric_data in metrics_to_send:
            try:
                success = await self.cloud_client.write_time_series(
                    metric_type=metric_data["metric_type"],
                    value=metric_data["value"],
                    labels=metric_data["labels"],
                    timestamp=metric_data["timestamp"]
                )
                
                if success:
                    success_count += 1
                    self.cloud_metrics_sent.labels(
                        metric_type=metric_data["metric_type"],
                        status="success"
                    ).inc()
                else:
                    error_count += 1
                    self.cloud_metrics_sent.labels(
                        metric_type=metric_data["metric_type"],
                        status="error"
                    ).inc()
                    
            except Exception as e:
                error_count += 1
                self.cloud_monitoring_errors.labels(
                    operation="write_time_series",
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Failed to send metric {metric_data['metric_type']}: {e}")
        
        logger.info(f"Flushed {success_count} metrics to Cloud Monitoring ({error_count} errors)")
    
    def collect_metrics(self):
        """Collect metrics from the monitoring system itself."""
        # This is called by the base class for Prometheus metrics
        pass
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        return {
            "cloud_monitoring_metrics_sent_total": "Number of metrics sent to Cloud Monitoring",
            "cloud_monitoring_alerts_created_total": "Number of alert policies created",
            "cloud_monitoring_errors_total": "Number of Cloud Monitoring errors"
        }
    
    async def setup_standard_alert_policies(self) -> Dict[str, str]:
        """
        Set up standard alert policies for the trading system.
        
        Returns:
            Dictionary mapping alert names to policy IDs
        """
        policies = {}
        
        # Create notification channels first (example email)
        email_channel = NotificationChannel(
            type="email",
            display_name="Shyvr RLTE Alerts",
            labels={"email_address": "alerts@yourcompany.com"}
        )
        
        email_channel_id = await self.cloud_client.create_notification_channel(email_channel)
        if not email_channel_id:
            logger.error("Failed to create email notification channel")
            return policies
        
        # Trading System Down Alert
        trading_down_condition = AlertPolicyCondition(
            filter='resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/system/uptime"',
            comparison="COMPARISON_LT",
            threshold_value=0.5,
            duration="60s"
        )
        
        trading_down_policy = await self.cloud_client.create_alert_policy(
            display_name="Shyvr RLTE - Trading System Down",
            conditions=[trading_down_condition],
            notification_channels=[email_channel_id],
            documentation="Alert when the trading system goes down or becomes unresponsive."
        )
        
        if trading_down_policy:
            policies["system_down"] = trading_down_policy
        
        # High Risk Level Alert
        high_risk_condition = AlertPolicyCondition(
            filter='resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/safety/risk_level"',
            comparison="COMPARISON_GT",
            threshold_value=0.8,
            duration="120s"
        )
        
        high_risk_policy = await self.cloud_client.create_alert_policy(
            display_name="Shyvr RLTE - High Risk Level",
            conditions=[high_risk_condition],
            notification_channels=[email_channel_id],
            documentation="Alert when overall risk level exceeds 80% for more than 2 minutes."
        )
        
        if high_risk_policy:
            policies["high_risk"] = high_risk_policy
        
        # Large Loss Alert
        large_loss_condition = AlertPolicyCondition(
            filter='resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/trading/daily_pnl"',
            comparison="COMPARISON_LT",
            threshold_value=-1000.0,
            duration="0s"
        )
        
        large_loss_policy = await self.cloud_client.create_alert_policy(
            display_name="Shyvr RLTE - Large Daily Loss",
            conditions=[large_loss_condition],
            notification_channels=[email_channel_id],
            documentation="Alert immediately when daily loss exceeds $1000."
        )
        
        if large_loss_policy:
            policies["large_loss"] = large_loss_policy
        
        logger.info(f"Created {len(policies)} standard alert policies")
        return policies
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Cloud Monitoring integration.
        
        Returns:
            Health check results
        """
        health_status = {
            "status": "healthy",
            "cloud_monitoring_connected": False,
            "metrics_buffer_size": len(self._metrics_buffer),
            "last_flush_time": self._last_flush_time,
            "project_id": self.project_id,
            "errors": []
        }
        
        try:
            # Test connection by listing existing alert policies
            policies = await self.cloud_client.list_alert_policies()
            health_status["cloud_monitoring_connected"] = True
            health_status["alert_policies_count"] = len(policies)
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(f"Cloud Monitoring connection failed: {e}")
        
        # Check buffer size
        if len(self._metrics_buffer) > 1000:
            health_status["status"] = "degraded"
            health_status["errors"].append("Large metrics buffer indicates sending issues")
        
        return health_status


@asynccontextmanager
async def cloud_monitoring_context(project_id: str, registry: MetricsRegistry):
    """
    Async context manager for Cloud Monitoring integration.
    
    This ensures proper cleanup of resources and final metric flushing.
    """
    collector = CloudMonitoringCollector(registry, project_id)
    
    try:
        yield collector
    finally:
        # Flush any remaining metrics
        await collector._flush_metrics()
        logger.info("Cloud Monitoring context cleaned up")