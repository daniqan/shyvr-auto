#!/usr/bin/env python3

"""
Phase 8.2: Production Monitoring Dashboard
Comprehensive monitoring system for Shyvr RLTE production deployment
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests
from google.cloud import monitoring_v3
from google.cloud import logging as cloud_logging
from google.cloud import run_v2
import asyncio
import aiohttp


@dataclass
class MonitoringMetric:
    """Production monitoring metric definition"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str]
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class SLATarget:
    """Service Level Agreement target definition"""
    metric_name: str
    target_value: float
    measurement_window: str  # e.g., '5m', '1h', '24h'
    operator: str  # 'lt', 'gt', 'eq'
    description: str


@dataclass
class AlertCondition:
    """Alert condition definition"""
    name: str
    condition: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    notification_channels: List[str]
    description: str


class ProductionMonitoringDashboard:
    """
    Comprehensive production monitoring dashboard for Shyvr RLTE
    Provides real-time monitoring, SLA tracking, and alerting
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.service_name = "shyvr-rlte"
        
        # Initialize Google Cloud clients
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.logging_client = cloud_logging.Client()
        self.run_client = run_v2.ServicesClient()
        
        # Configuration
        self.setup_monitoring_config()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring_config(self):
        """Setup monitoring configuration and SLA targets"""
        
        # SLA Targets
        self.sla_targets = [
            SLATarget(
                metric_name="request_latency_p99",
                target_value=2000.0,  # 2 seconds
                measurement_window="5m",
                operator="lt",
                description="99th percentile request latency < 2 seconds"
            ),
            SLATarget(
                metric_name="error_rate",
                target_value=5.0,  # 5%
                measurement_window="5m", 
                operator="lt",
                description="Error rate < 5%"
            ),
            SLATarget(
                metric_name="availability",
                target_value=99.9,  # 99.9%
                measurement_window="24h",
                operator="gt",
                description="Service availability > 99.9%"
            ),
            SLATarget(
                metric_name="trading_execution_latency",
                target_value=500.0,  # 500ms
                measurement_window="1m",
                operator="lt",
                description="Trading execution latency < 500ms"
            ),
            SLATarget(
                metric_name="ml_prediction_latency",
                target_value=100.0,  # 100ms
                measurement_window="1m",
                operator="lt",
                description="ML prediction latency < 100ms"
            ),
            SLATarget(
                metric_name="safety_validation_latency",
                target_value=50.0,  # 50ms
                measurement_window="1m",
                operator="lt",
                description="Safety validation latency < 50ms"
            )
        ]
        
        # Alert Conditions
        self.alert_conditions = [
            AlertCondition(
                name="high_error_rate",
                condition="error_rate > 10",
                severity="CRITICAL",
                notification_channels=["email", "slack"],
                description="Error rate exceeds 10%"
            ),
            AlertCondition(
                name="high_latency",
                condition="request_latency_p99 > 5000",
                severity="WARNING",
                notification_channels=["slack"],
                description="Request latency exceeds 5 seconds"
            ),
            AlertCondition(
                name="trading_system_down",
                condition="trading_requests_count == 0 for 5m",
                severity="CRITICAL",
                notification_channels=["email", "slack", "pager"],
                description="No trading requests received for 5 minutes"
            ),
            AlertCondition(
                name="memory_high",
                condition="memory_utilization > 85",
                severity="WARNING",
                notification_channels=["slack"],
                description="Memory utilization exceeds 85%"
            ),
            AlertCondition(
                name="cpu_high",
                condition="cpu_utilization > 80",
                severity="WARNING",
                notification_channels=["slack"],
                description="CPU utilization exceeds 80%"
            )
        ]
    
    async def collect_system_metrics(self) -> List[MonitoringMetric]:
        """Collect comprehensive system metrics"""
        metrics = []
        
        try:
            # Cloud Run metrics
            run_metrics = await self._collect_cloud_run_metrics()
            metrics.extend(run_metrics)
            
            # Application metrics
            app_metrics = await self._collect_application_metrics()
            metrics.extend(app_metrics)
            
            # Trading metrics
            trading_metrics = await self._collect_trading_metrics()
            metrics.extend(trading_metrics)
            
            # ML/RL metrics
            ml_metrics = await self._collect_ml_rl_metrics()
            metrics.extend(ml_metrics)
            
            # Safety metrics
            safety_metrics = await self._collect_safety_metrics()
            metrics.extend(safety_metrics)
            
            # Transformer metrics
            transformer_metrics = await self._collect_transformer_metrics()
            metrics.extend(transformer_metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _collect_cloud_run_metrics(self) -> List[MonitoringMetric]:
        """Collect Cloud Run service metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Request count
            request_count = await self._query_monitoring_metric(
                "run.googleapis.com/request_count"
            )
            metrics.append(MonitoringMetric(
                name="request_count",
                value=request_count,
                unit="count",
                timestamp=now,
                labels={"service": self.service_name},
                threshold_warning=1000.0,
                threshold_critical=2000.0
            ))
            
            # Request latency
            request_latency = await self._query_monitoring_metric(
                "run.googleapis.com/request_latencies"
            )
            metrics.append(MonitoringMetric(
                name="request_latency_p99",
                value=request_latency,
                unit="ms",
                timestamp=now,
                labels={"service": self.service_name, "percentile": "99"},
                threshold_warning=2000.0,
                threshold_critical=5000.0
            ))
            
            # Container CPU utilization
            cpu_utilization = await self._query_monitoring_metric(
                "run.googleapis.com/container/cpu/utilizations"
            )
            metrics.append(MonitoringMetric(
                name="cpu_utilization",
                value=cpu_utilization * 100,
                unit="percent",
                timestamp=now,
                labels={"service": self.service_name},
                threshold_warning=80.0,
                threshold_critical=90.0
            ))
            
            # Container memory utilization
            memory_utilization = await self._query_monitoring_metric(
                "run.googleapis.com/container/memory/utilizations"
            )
            metrics.append(MonitoringMetric(
                name="memory_utilization",
                value=memory_utilization * 100,
                unit="percent",
                timestamp=now,
                labels={"service": self.service_name},
                threshold_warning=85.0,
                threshold_critical=95.0
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting Cloud Run metrics: {e}")
        
        return metrics
    
    async def _collect_application_metrics(self) -> List[MonitoringMetric]:
        """Collect application-specific metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Health check metrics
            health_status = await self._check_application_health()
            metrics.append(MonitoringMetric(
                name="health_status",
                value=1.0 if health_status else 0.0,
                unit="bool",
                timestamp=now,
                labels={"service": self.service_name},
                threshold_critical=1.0
            ))
            
            # Error rate from logs
            error_rate = await self._calculate_error_rate()
            metrics.append(MonitoringMetric(
                name="error_rate",
                value=error_rate,
                unit="percent",
                timestamp=now,
                labels={"service": self.service_name},
                threshold_warning=5.0,
                threshold_critical=10.0
            ))
            
            # Active connections
            active_connections = await self._get_active_connections()
            metrics.append(MonitoringMetric(
                name="active_connections",
                value=active_connections,
                unit="count",
                timestamp=now,
                labels={"service": self.service_name}
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
        
        return metrics
    
    async def _collect_trading_metrics(self) -> List[MonitoringMetric]:
        """Collect trading system metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Trading execution latency
            trading_latency = await self._get_trading_latency()
            metrics.append(MonitoringMetric(
                name="trading_execution_latency",
                value=trading_latency,
                unit="ms",
                timestamp=now,
                labels={"component": "trading"},
                threshold_warning=500.0,
                threshold_critical=1000.0
            ))
            
            # Active positions
            active_positions = await self._get_active_positions_count()
            metrics.append(MonitoringMetric(
                name="active_positions",
                value=active_positions,
                unit="count",
                timestamp=now,
                labels={"component": "portfolio"}
            ))
            
            # Risk score
            risk_score = await self._get_current_risk_score()
            metrics.append(MonitoringMetric(
                name="risk_score",
                value=risk_score,
                unit="score",
                timestamp=now,
                labels={"component": "risk"},
                threshold_warning=80.0,
                threshold_critical=90.0
            ))
            
            # Trading volume (24h)
            trading_volume = await self._get_trading_volume_24h()
            metrics.append(MonitoringMetric(
                name="trading_volume_24h",
                value=trading_volume,
                unit="usd",
                timestamp=now,
                labels={"component": "trading"}
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
        
        return metrics
    
    async def _collect_ml_rl_metrics(self) -> List[MonitoringMetric]:
        """Collect ML/RL system metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # ML prediction latency
            ml_latency = await self._get_ml_prediction_latency()
            metrics.append(MonitoringMetric(
                name="ml_prediction_latency",
                value=ml_latency,
                unit="ms",
                timestamp=now,
                labels={"component": "ml"},
                threshold_warning=100.0,
                threshold_critical=200.0
            ))
            
            # Model accuracy
            model_accuracy = await self._get_model_accuracy()
            metrics.append(MonitoringMetric(
                name="model_accuracy",
                value=model_accuracy,
                unit="percent",
                timestamp=now,
                labels={"component": "ml"},
                threshold_warning=70.0,
                threshold_critical=60.0
            ))
            
            # RL episode rewards
            rl_rewards = await self._get_rl_episode_rewards()
            metrics.append(MonitoringMetric(
                name="rl_episode_rewards",
                value=rl_rewards,
                unit="score",
                timestamp=now,
                labels={"component": "rl"}
            ))
            
            # Experience replay buffer size
            buffer_size = await self._get_experience_buffer_size()
            metrics.append(MonitoringMetric(
                name="experience_buffer_size",
                value=buffer_size,
                unit="count",
                timestamp=now,
                labels={"component": "rl"}
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting ML/RL metrics: {e}")
        
        return metrics
    
    async def _collect_safety_metrics(self) -> List[MonitoringMetric]:
        """Collect safety system metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Safety validation latency
            safety_latency = await self._get_safety_validation_latency()
            metrics.append(MonitoringMetric(
                name="safety_validation_latency",
                value=safety_latency,
                unit="ms",
                timestamp=now,
                labels={"component": "safety"},
                threshold_warning=50.0,
                threshold_critical=100.0
            ))
            
            # Emergency stop status
            emergency_status = await self._get_emergency_stop_status()
            metrics.append(MonitoringMetric(
                name="emergency_stop_active",
                value=1.0 if emergency_status else 0.0,
                unit="bool",
                timestamp=now,
                labels={"component": "safety"},
                threshold_critical=0.0
            ))
            
            # Circuit breaker status
            circuit_breaker_status = await self._get_circuit_breaker_status()
            metrics.append(MonitoringMetric(
                name="circuit_breaker_triggered",
                value=circuit_breaker_status,
                unit="count",
                timestamp=now,
                labels={"component": "safety"}
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting safety metrics: {e}")
        
        return metrics
    
    async def _collect_transformer_metrics(self) -> List[MonitoringMetric]:
        """Collect transformer model metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Transformer models to monitor
            transformer_models = ['itransformer', 'patchtst', 'timesmixer', 'timesfm']
            
            for model_name in transformer_models:
                # Memory usage
                memory_usage = await self._get_transformer_memory_usage(model_name)
                metrics.append(MonitoringMetric(
                    name="transformer_memory_usage",
                    value=memory_usage,
                    unit="mb",
                    timestamp=now,
                    labels={"model_name": model_name, "component": "transformer"},
                    threshold_warning=6553.6,  # 80% of 8Gi
                    threshold_critical=7372.8  # 90% of 8Gi
                ))
                
                # Inference latency
                inference_latency = await self._get_transformer_inference_latency(model_name)
                metrics.append(MonitoringMetric(
                    name="transformer_inference_latency",
                    value=inference_latency,
                    unit="ms",
                    timestamp=now,
                    labels={"model_name": model_name, "component": "transformer"},
                    threshold_warning=500.0,
                    threshold_critical=1000.0
                ))
                
                # Cache hit rate
                cache_hit_rate = await self._get_transformer_cache_hit_rate(model_name)
                metrics.append(MonitoringMetric(
                    name="transformer_cache_hit_rate",
                    value=cache_hit_rate,
                    unit="percent",
                    timestamp=now,
                    labels={"model_name": model_name, "component": "transformer"},
                    threshold_warning=60.0,
                    threshold_critical=50.0
                ))
                
                # Model health
                health_status = await self._get_transformer_health_status(model_name)
                metrics.append(MonitoringMetric(
                    name="transformer_health_status",
                    value=1.0 if health_status else 0.0,
                    unit="bool",
                    timestamp=now,
                    labels={"model_name": model_name, "component": "transformer"},
                    threshold_critical=1.0
                ))
                
                # Loading time
                loading_time = await self._get_transformer_loading_time(model_name)
                metrics.append(MonitoringMetric(
                    name="transformer_loading_time",
                    value=loading_time,
                    unit="ms",
                    timestamp=now,
                    labels={"model_name": model_name, "component": "transformer"},
                    threshold_warning=3000.0,
                    threshold_critical=5000.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting transformer metrics: {e}")
        
        return metrics
    
    async def evaluate_sla_compliance(self, metrics: List[MonitoringMetric]) -> Dict[str, Any]:
        """Evaluate SLA compliance based on collected metrics"""
        sla_results = {}
        
        for sla in self.sla_targets:
            # Find matching metric
            matching_metrics = [m for m in metrics if m.name == sla.metric_name]
            
            if not matching_metrics:
                sla_results[sla.metric_name] = {
                    "status": "NO_DATA",
                    "current_value": None,
                    "target_value": sla.target_value,
                    "compliance": False
                }
                continue
            
            current_value = matching_metrics[0].value
            
            # Evaluate compliance
            if sla.operator == "lt":
                compliance = current_value < sla.target_value
            elif sla.operator == "gt":
                compliance = current_value > sla.target_value
            elif sla.operator == "eq":
                compliance = abs(current_value - sla.target_value) < 0.01
            else:
                compliance = False
            
            sla_results[sla.metric_name] = {
                "status": "COMPLIANT" if compliance else "VIOLATION",
                "current_value": current_value,
                "target_value": sla.target_value,
                "compliance": compliance,
                "description": sla.description
            }
        
        return sla_results
    
    async def check_alert_conditions(self, metrics: List[MonitoringMetric]) -> List[Dict[str, Any]]:
        """Check alert conditions and generate alerts"""
        alerts = []
        
        for condition in self.alert_conditions:
            # Simple condition evaluation (in production, use more sophisticated parsing)
            alert_triggered = await self._evaluate_alert_condition(condition, metrics)
            
            if alert_triggered:
                alerts.append({
                    "name": condition.name,
                    "severity": condition.severity,
                    "description": condition.description,
                    "timestamp": datetime.utcnow().isoformat(),
                    "condition": condition.condition,
                    "notification_channels": condition.notification_channels
                })
        
        return alerts
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # Collect all metrics
        metrics = await self.collect_system_metrics()
        
        # Evaluate SLA compliance
        sla_compliance = await self.evaluate_sla_compliance(metrics)
        
        # Check alerts
        alerts = await self.check_alert_conditions(metrics)
        
        # Calculate overall health score
        health_score = await self._calculate_health_score(metrics, sla_compliance)
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": health_score,
            "metrics": [asdict(m) for m in metrics],
            "sla_compliance": sla_compliance,
            "active_alerts": alerts,
            "system_status": "HEALTHY" if health_score > 80 else "DEGRADED" if health_score > 60 else "CRITICAL",
            "uptime": await self._calculate_uptime(),
            "deployment_info": await self._get_deployment_info()
        }
        
        return dashboard_data
    
    async def _query_monitoring_metric(self, metric_type: str) -> float:
        """Query Google Cloud Monitoring for specific metric"""
        try:
            # This is a simplified implementation
            # In production, use proper time series queries
            project_name = f"projects/{self.project_id}"
            
            # For now, return mock data
            return 50.0  # Mock value
            
        except Exception as e:
            self.logger.error(f"Error querying metric {metric_type}: {e}")
            return 0.0
    
    async def _check_application_health(self) -> bool:
        """Check application health endpoint"""
        try:
            # In production, query the actual health endpoint
            return True  # Mock implementation
        except Exception:
            return False
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate from logs"""
        try:
            # Query logs for error rate calculation
            return 2.5  # Mock error rate
        except Exception:
            return 0.0
    
    async def _get_active_connections(self) -> float:
        """Get number of active connections"""
        return 25.0  # Mock value
    
    async def _get_trading_latency(self) -> float:
        """Get trading execution latency"""
        return 150.0  # Mock value
    
    async def _get_active_positions_count(self) -> float:
        """Get number of active trading positions"""
        return 5.0  # Mock value
    
    async def _get_current_risk_score(self) -> float:
        """Get current portfolio risk score"""
        return 45.0  # Mock value
    
    async def _get_trading_volume_24h(self) -> float:
        """Get 24-hour trading volume"""
        return 50000.0  # Mock value
    
    async def _get_ml_prediction_latency(self) -> float:
        """Get ML prediction latency"""
        return 75.0  # Mock value
    
    async def _get_model_accuracy(self) -> float:
        """Get current model accuracy"""
        return 85.5  # Mock value
    
    async def _get_rl_episode_rewards(self) -> float:
        """Get recent RL episode rewards"""
        return 12.5  # Mock value
    
    async def _get_experience_buffer_size(self) -> float:
        """Get experience replay buffer size"""
        return 10000.0  # Mock value
    
    async def _get_safety_validation_latency(self) -> float:
        """Get safety validation latency"""
        return 25.0  # Mock value
    
    async def _get_emergency_stop_status(self) -> bool:
        """Get emergency stop status"""
        return False  # Mock value
    
    async def _get_circuit_breaker_status(self) -> float:
        """Get circuit breaker trigger count"""
        return 0.0  # Mock value
    
    async def _get_transformer_memory_usage(self, model_name: str) -> float:
        """Get memory usage for specific transformer model"""
        # Mock values based on model type
        memory_usage_map = {
            'itransformer': 1200.0,
            'patchtst': 950.0,
            'timesmixer': 1600.0,
            'timesfm': 7200.0
        }
        return memory_usage_map.get(model_name, 1000.0)
    
    async def _get_transformer_inference_latency(self, model_name: str) -> float:
        """Get inference latency for specific transformer model"""
        # Mock values based on model type
        latency_map = {
            'itransformer': 85.0,
            'patchtst': 120.0,
            'timesmixer': 95.0,
            'timesfm': 1100.0
        }
        return latency_map.get(model_name, 100.0)
    
    async def _get_transformer_cache_hit_rate(self, model_name: str) -> float:
        """Get cache hit rate for specific transformer model"""
        # Mock values based on model type
        cache_rate_map = {
            'itransformer': 75.0,
            'patchtst': 82.0,
            'timesmixer': 45.0,
            'timesfm': 68.0
        }
        return cache_rate_map.get(model_name, 70.0)
    
    async def _get_transformer_health_status(self, model_name: str) -> bool:
        """Get health status for specific transformer model"""
        # Mock healthy status (in production, would check actual model health)
        health_map = {
            'itransformer': True,
            'patchtst': True,
            'timesmixer': False,  # Simulating unhealthy state due to low cache rate
            'timesfm': False      # Simulating unhealthy state due to high memory/latency
        }
        return health_map.get(model_name, True)
    
    async def _get_transformer_loading_time(self, model_name: str) -> float:
        """Get loading time for specific transformer model"""
        # Mock values based on model complexity
        loading_time_map = {
            'itransformer': 2500.0,
            'patchtst': 1800.0,
            'timesmixer': 3200.0,
            'timesfm': 4500.0
        }
        return loading_time_map.get(model_name, 2000.0)
    
    async def _evaluate_alert_condition(self, condition: AlertCondition, metrics: List[MonitoringMetric]) -> bool:
        """Evaluate a specific alert condition"""
        # Simplified condition evaluation
        # In production, implement proper condition parser
        return False  # Mock evaluation
    
    async def _calculate_health_score(self, metrics: List[MonitoringMetric], sla_compliance: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        compliant_slas = sum(1 for sla in sla_compliance.values() if sla.get("compliance", False))
        total_slas = len(sla_compliance)
        
        if total_slas == 0:
            return 50.0
        
        health_score = (compliant_slas / total_slas) * 100
        return health_score
    
    async def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        # Mock uptime calculation
        return "99.95%"
    
    async def _get_deployment_info(self) -> Dict[str, Any]:
        """Get current deployment information"""
        return {
            "version": "1.0.0",
            "deployed_at": "2025-08-03T12:00:00Z",
            "environment": "production",
            "region": self.region
        }


class MonitoringDashboardServer:
    """HTTP server for monitoring dashboard"""
    
    def __init__(self, dashboard: ProductionMonitoringDashboard, port: int = 8080):
        self.dashboard = dashboard
        self.port = port
        self.logger = logging.getLogger(__name__)
    
    async def start_server(self):
        """Start the monitoring dashboard server"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/dashboard', self.get_dashboard)
        app.router.add_get('/metrics', self.get_metrics)
        app.router.add_get('/health', self.get_health)
        app.router.add_get('/sla', self.get_sla_status)
        app.router.add_get('/alerts', self.get_alerts)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        self.logger.info(f"Monitoring dashboard started on port {self.port}")
    
    async def get_dashboard(self, request):
        """Get complete dashboard data"""
        from aiohttp import web
        
        try:
            dashboard_data = await self.dashboard.generate_dashboard_data()
            return web.json_response(dashboard_data)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to generate dashboard data: {e}"}, 
                status=500
            )
    
    async def get_metrics(self, request):
        """Get current metrics"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_system_metrics()
            return web.json_response([asdict(m) for m in metrics])
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect metrics: {e}"}, 
                status=500
            )
    
    async def get_health(self, request):
        """Get system health status"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_system_metrics()
            sla_compliance = await self.dashboard.evaluate_sla_compliance(metrics)
            health_score = await self.dashboard._calculate_health_score(metrics, sla_compliance)
            
            return web.json_response({
                "status": "healthy" if health_score > 80 else "degraded" if health_score > 60 else "critical",
                "health_score": health_score,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get health status: {e}"}, 
                status=500
            )
    
    async def get_sla_status(self, request):
        """Get SLA compliance status"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_system_metrics()
            sla_compliance = await self.dashboard.evaluate_sla_compliance(metrics)
            return web.json_response(sla_compliance)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get SLA status: {e}"}, 
                status=500
            )
    
    async def get_alerts(self, request):
        """Get active alerts"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_system_metrics()
            alerts = await self.dashboard.check_alert_conditions(metrics)
            return web.json_response(alerts)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get alerts: {e}"}, 
                status=500
            )


async def main():
    """Main function to run the monitoring dashboard"""
    import os
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "shyvr-rlte")
    region = os.getenv("GCP_REGION", "us-central1")
    
    # Create dashboard
    dashboard = ProductionMonitoringDashboard(project_id, region)
    
    # Create server
    server = MonitoringDashboardServer(dashboard)
    
    # Start server
    await server.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down monitoring dashboard")


if __name__ == "__main__":
    asyncio.run(main())