"""
Resource Monitoring Integration for Dynamic Resource Allocation

Provides integration between resource monitoring systems and dynamic
resource allocation with GCP Cloud Monitoring and custom metrics.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CustomMetric:
    """Custom monitoring metric definition"""
    name: str
    type: str
    description: str
    unit: str
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


@dataclass
class MetricDataPoint:
    """Metric data point"""
    metric_name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.timestamp is None:
            self.timestamp = time.time()


class ResourceMonitoringIntegrator:
    """
    Resource Monitoring Integrator
    
    Integrates resource monitoring with GCP Cloud Monitoring for
    dynamic resource allocation metrics and alerting.
    """
    
    def __init__(self, project_id: str, region: str, service_name: str,
                 monitoring_enabled: bool = True, alert_channels: List[str] = None):
        """Initialize resource monitoring integrator"""
        self.project_id = project_id
        self.region = region
        self.service_name = service_name
        self.monitoring_enabled = monitoring_enabled
        self.alert_channels = alert_channels or []
        
        # Initialize monitoring client
        try:
            self.monitoring_client = monitoring_v3.MetricServiceClient() if monitoring_v3 else None
            self.project_path = f"projects/{project_id}" if monitoring_v3 else None
        except Exception as e:
            logger.warning(f"Failed to initialize monitoring client: {e}")
            self.monitoring_client = None
            self.project_path = None
        
        # Metric definitions and data
        self.custom_metrics = {}
        self.metric_buffer = []
        self.alert_policies = {}
        
        logger.info(f"Initialized ResourceMonitoringIntegrator for {service_name}")

    async def create_custom_metrics(self, metrics_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create custom metrics for resource allocation monitoring"""
        try:
            logger.info(f"Creating {len(metrics_config)} custom metrics")
            created_metrics = []
            
            for metric_name, config in metrics_config.items():
                custom_metric = CustomMetric(
                    name=metric_name,
                    type=config.get("type", "gauge"),
                    description=config.get("description", ""),
                    unit=config.get("unit", "1"),
                    labels=config.get("labels", [])
                )
                
                # Simulate metric creation (in production, would use GCP API)
                await asyncio.sleep(0.1)
                
                self.custom_metrics[metric_name] = custom_metric
                created_metrics.append(metric_name)
                
                logger.info(f"Created custom metric: {metric_name}")
            
            result = {
                "success": True,
                "created_metrics": created_metrics,
                "total_created": len(created_metrics)
            }
            
            logger.info(f"Successfully created {len(created_metrics)} custom metrics")
            return result
            
        except Exception as e:
            logger.error(f"Error creating custom metrics: {e}")
            return {
                "success": False,
                "error": str(e),
                "created_metrics": []
            }

    async def send_metric_data(self, metric_name: str, value: float, 
                             labels: Dict[str, str] = None) -> bool:
        """Send metric data point"""
        try:
            if metric_name not in self.custom_metrics:
                logger.warning(f"Metric {metric_name} not found in custom metrics")
                return False
            
            data_point = MetricDataPoint(
                metric_name=metric_name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            self.metric_buffer.append(data_point)
            
            # In production, would send to GCP monitoring
            logger.debug(f"Queued metric data: {metric_name} = {value}")
            
            # Flush buffer periodically
            if len(self.metric_buffer) > 100:
                await self._flush_metric_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending metric data: {e}")
            return False

    async def _flush_metric_buffer(self):
        """Flush metric buffer to monitoring service"""
        try:
            if not self.metric_buffer:
                return
            
            logger.info(f"Flushing {len(self.metric_buffer)} metric data points")
            
            # Mock flushing - in production would batch send to GCP
            await asyncio.sleep(0.1)
            
            flushed_count = len(self.metric_buffer)
            self.metric_buffer.clear()
            
            logger.info(f"Flushed {flushed_count} metric data points")
            
        except Exception as e:
            logger.error(f"Error flushing metric buffer: {e}")

    async def create_alert_policies(self, alert_policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create alert policies for resource monitoring"""
        try:
            logger.info(f"Creating {len(alert_policies)} alert policies")
            created_policies = []
            
            for policy_config in alert_policies:
                policy_name = policy_config.get("name")
                condition = policy_config.get("condition")
                notification_channels = policy_config.get("notification_channels", [])
                
                # Mock policy creation
                await asyncio.sleep(0.1)
                
                policy_id = f"policy-{hash(policy_name) % 10000}"
                
                alert_policy = {
                    "name": policy_name,
                    "id": policy_id,
                    "condition": condition,
                    "notification_channels": notification_channels,
                    "created_at": time.time(),
                    "enabled": True
                }
                
                self.alert_policies[policy_id] = alert_policy
                created_policies.append(alert_policy)
                
                logger.info(f"Created alert policy: {policy_name}")
            
            result = {
                "success": True,
                "created_policies": created_policies,
                "total_created": len(created_policies)
            }
            
            logger.info(f"Successfully created {len(created_policies)} alert policies")
            return result
            
        except Exception as e:
            logger.error(f"Error creating alert policies: {e}")
            return {
                "success": False,
                "error": str(e),
                "created_policies": []
            }

    async def monitor_resource_allocation_performance(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Monitor resource allocation performance metrics"""
        try:
            logger.info(f"Starting resource allocation performance monitoring for {duration_seconds}s")
            start_time = time.time()
            
            performance_metrics = {
                "allocation_latency_samples": [],
                "resource_utilization_samples": [],
                "cost_efficiency_samples": [],
                "scaling_events": [],
                "alert_events": []
            }
            
            # Monitor for specified duration (cap for testing)
            monitor_duration = min(duration_seconds, 10.0)
            sample_interval = 1.0
            
            while (time.time() - start_time) < monitor_duration:
                # Collect performance sample
                sample = await self._collect_performance_sample()
                
                performance_metrics["allocation_latency_samples"].append(sample["allocation_latency"])
                performance_metrics["resource_utilization_samples"].append(sample["resource_utilization"])
                performance_metrics["cost_efficiency_samples"].append(sample["cost_efficiency"])
                
                # Check for scaling events
                if sample.get("scaling_event"):
                    performance_metrics["scaling_events"].append({
                        "timestamp": time.time(),
                        "type": sample["scaling_event"]["type"],
                        "reason": sample["scaling_event"]["reason"]
                    })
                
                await asyncio.sleep(sample_interval)
            
            # Calculate summary statistics
            summary = self._calculate_monitoring_summary(performance_metrics)
            
            monitoring_result = {
                "monitoring_duration_seconds": time.time() - start_time,
                "samples_collected": len(performance_metrics["allocation_latency_samples"]),
                "performance_metrics": performance_metrics,
                "summary": summary,
                "status": "completed"
            }
            
            logger.info(f"Resource allocation performance monitoring completed")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Error monitoring resource allocation performance: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _collect_performance_sample(self) -> Dict[str, Any]:
        """Collect a single performance sample"""
        try:
            import random
            
            # Mock performance sample
            sample = {
                "allocation_latency": random.uniform(10, 50),  # ms
                "resource_utilization": {
                    "cpu": random.uniform(30, 80),
                    "memory": random.uniform(40, 85),
                    "network": random.uniform(20, 60)
                },
                "cost_efficiency": random.uniform(0.7, 0.95),
                "timestamp": time.time()
            }
            
            # Occasionally add scaling events
            if random.random() < 0.1:  # 10% chance
                sample["scaling_event"] = {
                    "type": random.choice(["scale_up", "scale_down"]),
                    "reason": "resource_threshold_triggered"
                }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error collecting performance sample: {e}")
            return {}

    def _calculate_monitoring_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for monitoring data"""
        try:
            latency_samples = metrics.get("allocation_latency_samples", [])
            utilization_samples = metrics.get("resource_utilization_samples", [])
            cost_samples = metrics.get("cost_efficiency_samples", [])
            
            summary = {}
            
            if latency_samples:
                summary["allocation_latency"] = {
                    "avg_ms": sum(latency_samples) / len(latency_samples),
                    "max_ms": max(latency_samples),
                    "min_ms": min(latency_samples)
                }
            
            if utilization_samples:
                cpu_values = [s["cpu"] for s in utilization_samples]
                memory_values = [s["memory"] for s in utilization_samples]
                
                summary["resource_utilization"] = {
                    "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                    "avg_memory_percent": sum(memory_values) / len(memory_values),
                    "max_cpu_percent": max(cpu_values),
                    "max_memory_percent": max(memory_values)
                }
            
            if cost_samples:
                summary["cost_efficiency"] = {
                    "avg_score": sum(cost_samples) / len(cost_samples),
                    "min_score": min(cost_samples),
                    "max_score": max(cost_samples)
                }
            
            summary["scaling_events_count"] = len(metrics.get("scaling_events", []))
            summary["alert_events_count"] = len(metrics.get("alert_events", []))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating monitoring summary: {e}")
            return {}

    async def get_resource_allocation_metrics(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get resource allocation metrics for specified time range"""
        try:
            end_time = time.time()
            start_time = end_time - (time_range_minutes * 60)
            
            # Mock metrics retrieval
            import random
            
            metrics = {
                "time_range": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": time_range_minutes
                },
                "allocation_performance": {
                    "total_allocations": random.randint(50, 200),
                    "successful_allocations": random.randint(45, 195),
                    "failed_allocations": random.randint(0, 10),
                    "avg_allocation_time_ms": random.uniform(15, 45)
                },
                "resource_efficiency": {
                    "avg_cpu_utilization": random.uniform(45, 75),
                    "avg_memory_utilization": random.uniform(50, 80),
                    "resource_waste_percent": random.uniform(5, 20)
                },
                "cost_metrics": {
                    "estimated_hourly_cost": random.uniform(8, 25),
                    "cost_per_allocation": random.uniform(0.01, 0.05),
                    "cost_efficiency_score": random.uniform(0.75, 0.95)
                },
                "scaling_activity": {
                    "scale_up_events": random.randint(0, 5),
                    "scale_down_events": random.randint(0, 3),
                    "auto_scaling_efficiency": random.uniform(0.8, 0.95)
                }
            }
            
            logger.info(f"Retrieved resource allocation metrics for {time_range_minutes} minutes")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting resource allocation metrics: {e}")
            return {}

    async def create_monitoring_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring dashboard for resource allocation"""
        try:
            dashboard_name = dashboard_config.get("name", "Resource Allocation Dashboard")
            panels = dashboard_config.get("panels", [])
            
            logger.info(f"Creating monitoring dashboard: {dashboard_name}")
            
            # Mock dashboard creation
            await asyncio.sleep(0.5)
            
            dashboard_id = f"dashboard-{hash(dashboard_name) % 10000}"
            
            dashboard = {
                "id": dashboard_id,
                "name": dashboard_name,
                "panels": len(panels),
                "created_at": time.time(),
                "url": f"https://console.cloud.google.com/monitoring/dashboards/custom/{dashboard_id}",
                "status": "active"
            }
            
            result = {
                "success": True,
                "dashboard": dashboard,
                "dashboard_url": dashboard["url"]
            }
            
            logger.info(f"Created monitoring dashboard: {dashboard_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating monitoring dashboard: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        try:
            status = {
                "monitoring_enabled": self.monitoring_enabled,
                "custom_metrics_count": len(self.custom_metrics),
                "alert_policies_count": len(self.alert_policies),
                "metric_buffer_size": len(self.metric_buffer),
                "last_flush_time": time.time() - 300,  # Mock last flush
                "client_connected": self.monitoring_client is not None,
                "project_id": self.project_id,
                "service_name": self.service_name
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup_monitoring_resources(self) -> Dict[str, Any]:
        """Cleanup monitoring resources"""
        try:
            logger.info("Starting monitoring resources cleanup")
            
            # Flush remaining metrics
            await self._flush_metric_buffer()
            
            # Mock cleanup operations
            cleanup_results = {
                "metrics_flushed": len(self.metric_buffer),
                "custom_metrics_cleaned": len(self.custom_metrics),
                "alert_policies_cleaned": len(self.alert_policies)
            }
            
            # Clear local state
            self.custom_metrics.clear()
            self.alert_policies.clear()
            self.metric_buffer.clear()
            
            logger.info(f"Monitoring resources cleanup completed: {cleanup_results}")
            return {
                "success": True,
                "cleanup_results": cleanup_results
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up monitoring resources: {e}")
            return {
                "success": False,
                "error": str(e)
            }