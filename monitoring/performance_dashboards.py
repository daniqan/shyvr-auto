#!/usr/bin/env python3

"""
Performance Tracking Dashboards for Production Monitoring

Comprehensive performance tracking dashboards that integrate with existing monitoring infrastructure
to provide real-time visibility into system performance, resource utilization, and cost efficiency.

Key Features:
1. Real-time transformer model performance metrics
2. Inference latency tracking for all models
3. Memory usage and resource allocation monitoring
4. Throughput and request rate tracking
5. Ensemble performance metrics
6. Cost efficiency metrics
7. Historical performance trends
8. SLA compliance monitoring
9. Performance degradation alerts
10. Model comparison views
11. Cloud Run scaling metrics
12. Production health status
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import structlog
from google.cloud import monitoring_v3
from google.cloud import logging as cloud_logging
from google.cloud import run_v2
import aiohttp

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric definition"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str]
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    trend_direction: Optional[str] = None  # 'up', 'down', 'stable'
    percentile: Optional[int] = None


@dataclass
class PerformanceTarget:
    """Performance target definition for SLA compliance"""
    metric_name: str
    target_value: float
    operator: str  # 'lt', 'gt', 'eq'
    measurement_window: str
    description: str
    priority: str = 'medium'  # 'high', 'medium', 'low'


@dataclass
class CostMetric:
    """Cost efficiency metric"""
    component: str
    cost_usd: float
    usage_units: float
    cost_per_unit: float
    timestamp: datetime
    budget_percent: Optional[float] = None


@dataclass
class ThroughputMetric:
    """Throughput and request rate metric"""
    endpoint: str
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    timestamp: datetime
    response_time_p50: Optional[float] = None
    response_time_p95: Optional[float] = None
    response_time_p99: Optional[float] = None


class PerformanceTrackingDashboard:
    """
    Comprehensive performance tracking dashboard that integrates with existing monitoring infrastructure
    to provide real-time performance visibility and historical analysis.
    """

    def __init__(self, project_id: str, region: str = "us-central1", service_name: str = "shyvr-rlte"):
        self.project_id = project_id
        self.region = region
        self.service_name = service_name
        
        # Initialize Google Cloud clients
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.logging_client = cloud_logging.Client()
        self.run_client = run_v2.ServicesClient()
        
        # Performance targets
        self.performance_targets = self._setup_performance_targets()
        
        # Metrics history storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.cost_history = deque(maxlen=1000)
        self.throughput_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Model configurations
        self.transformer_models = ['itransformer', 'patchtst', 'timesmixer', 'timesfm']
        self.ensemble_models = ['voting_ensemble', 'weighted_ensemble', 'stacking_ensemble']
        
        self.logger = structlog.get_logger(self.__class__.__name__)

    def _setup_performance_targets(self) -> List[PerformanceTarget]:
        """Setup comprehensive performance targets for monitoring"""
        return [
            # Latency targets
            PerformanceTarget(
                metric_name="transformer_inference_latency_p99",
                target_value=500.0,  # 500ms
                operator="lt",
                measurement_window="5m",
                description="99th percentile transformer inference latency < 500ms",
                priority="high"
            ),
            PerformanceTarget(
                metric_name="ensemble_prediction_latency_p95",
                target_value=200.0,  # 200ms
                operator="lt",
                measurement_window="5m",
                description="95th percentile ensemble prediction latency < 200ms",
                priority="high"
            ),
            PerformanceTarget(
                metric_name="api_response_time_p95",
                target_value=1000.0,  # 1 second
                operator="lt",
                measurement_window="5m",
                description="95th percentile API response time < 1 second",
                priority="high"
            ),
            # Throughput targets
            PerformanceTarget(
                metric_name="requests_per_second",
                target_value=100.0,  # 100 RPS
                operator="gt",
                measurement_window="1m",
                description="System should handle > 100 requests per second",
                priority="medium"
            ),
            PerformanceTarget(
                metric_name="transformer_predictions_per_minute",
                target_value=1000.0,  # 1000 predictions per minute
                operator="gt",
                measurement_window="1m",
                description="Transformer models should generate > 1000 predictions per minute",
                priority="medium"
            ),
            # Resource targets
            PerformanceTarget(
                metric_name="memory_utilization",
                target_value=85.0,  # 85%
                operator="lt",
                measurement_window="5m",
                description="Memory utilization should be < 85%",
                priority="high"
            ),
            PerformanceTarget(
                metric_name="cpu_utilization",
                target_value=80.0,  # 80%
                operator="lt",
                measurement_window="5m",
                description="CPU utilization should be < 80%",
                priority="high"
            ),
            # Accuracy targets
            PerformanceTarget(
                metric_name="model_accuracy",
                target_value=85.0,  # 85%
                operator="gt",
                measurement_window="1h",
                description="Model accuracy should be > 85%",
                priority="high"
            ),
            # Cost efficiency targets
            PerformanceTarget(
                metric_name="cost_per_prediction",
                target_value=0.001,  # $0.001 per prediction
                operator="lt",
                measurement_window="1h",
                description="Cost per prediction should be < $0.001",
                priority="medium"
            )
        ]

    async def collect_transformer_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive transformer model performance metrics"""
        metrics = []
        now = datetime.utcnow()
        
        for model_name in self.transformer_models:
            try:
                # Inference latency percentiles
                for percentile in [50, 95, 99]:
                    latency = await self._get_transformer_latency_percentile(model_name, percentile)
                    metrics.append(PerformanceMetric(
                        name=f"transformer_inference_latency_p{percentile}",
                        value=latency,
                        unit="ms",
                        timestamp=now,
                        labels={"model": model_name, "component": "transformer"},
                        target_value=500.0 if percentile == 99 else 300.0,
                        threshold_warning=400.0 if percentile == 99 else 200.0,
                        threshold_critical=800.0 if percentile == 99 else 500.0,
                        percentile=percentile
                    ))
                
                # Memory usage
                memory_usage = await self._get_transformer_memory_usage(model_name)
                memory_percent = (memory_usage / 8192.0) * 100  # 8Gi = 8192MB
                metrics.append(PerformanceMetric(
                    name="transformer_memory_usage",
                    value=memory_usage,
                    unit="mb",
                    timestamp=now,
                    labels={"model": model_name, "component": "transformer"},
                    target_value=6553.6,  # 80% of 8Gi
                    threshold_warning=6553.6,
                    threshold_critical=7372.8  # 90% of 8Gi
                ))
                
                # Throughput (predictions per second)
                throughput = await self._get_transformer_throughput(model_name)
                metrics.append(PerformanceMetric(
                    name="transformer_throughput",
                    value=throughput,
                    unit="predictions/sec",
                    timestamp=now,
                    labels={"model": model_name, "component": "transformer"},
                    target_value=10.0,  # 10 predictions per second
                    threshold_warning=5.0,
                    threshold_critical=2.0
                ))
                
                # Cache hit rate
                cache_hit_rate = await self._get_transformer_cache_hit_rate(model_name)
                metrics.append(PerformanceMetric(
                    name="transformer_cache_hit_rate",
                    value=cache_hit_rate * 100,  # Convert to percentage
                    unit="percent",
                    timestamp=now,
                    labels={"model": model_name, "component": "transformer"},
                    target_value=80.0,
                    threshold_warning=60.0,
                    threshold_critical=40.0
                ))
                
                # Model accuracy (recent performance)
                accuracy = await self._get_transformer_accuracy(model_name)
                metrics.append(PerformanceMetric(
                    name="transformer_accuracy",
                    value=accuracy,
                    unit="percent",
                    timestamp=now,
                    labels={"model": model_name, "component": "transformer"},
                    target_value=85.0,
                    threshold_warning=80.0,
                    threshold_critical=75.0
                ))
                
                # Queue depth (pending requests)
                queue_depth = await self._get_transformer_queue_depth(model_name)
                metrics.append(PerformanceMetric(
                    name="transformer_queue_depth",
                    value=queue_depth,
                    unit="count",
                    timestamp=now,
                    labels={"model": model_name, "component": "transformer"},
                    target_value=10.0,
                    threshold_warning=50.0,
                    threshold_critical=100.0
                ))
                
            except Exception as e:
                self.logger.error(f"Error collecting transformer metrics for {model_name}: {e}")
        
        return metrics

    async def collect_ensemble_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect ensemble model performance metrics"""
        metrics = []
        now = datetime.utcnow()
        
        for ensemble_name in self.ensemble_models:
            try:
                # Prediction latency
                latency = await self._get_ensemble_latency(ensemble_name)
                metrics.append(PerformanceMetric(
                    name="ensemble_prediction_latency",
                    value=latency,
                    unit="ms",
                    timestamp=now,
                    labels={"ensemble": ensemble_name, "component": "ensemble"},
                    target_value=200.0,
                    threshold_warning=300.0,
                    threshold_critical=500.0
                ))
                
                # Consensus accuracy
                consensus_accuracy = await self._get_ensemble_consensus_accuracy(ensemble_name)
                metrics.append(PerformanceMetric(
                    name="ensemble_consensus_accuracy",
                    value=consensus_accuracy,
                    unit="percent",
                    timestamp=now,
                    labels={"ensemble": ensemble_name, "component": "ensemble"},
                    target_value=90.0,
                    threshold_warning=85.0,
                    threshold_critical=80.0
                ))
                
                # Model agreement rate
                agreement_rate = await self._get_ensemble_agreement_rate(ensemble_name)
                metrics.append(PerformanceMetric(
                    name="ensemble_agreement_rate",
                    value=agreement_rate,
                    unit="percent",
                    timestamp=now,
                    labels={"ensemble": ensemble_name, "component": "ensemble"},
                    target_value=70.0,
                    threshold_warning=60.0,
                    threshold_critical=50.0
                ))
                
            except Exception as e:
                self.logger.error(f"Error collecting ensemble metrics for {ensemble_name}: {e}")
        
        return metrics

    async def collect_throughput_metrics(self) -> List[ThroughputMetric]:
        """Collect comprehensive throughput metrics"""
        throughput_metrics = []
        now = datetime.utcnow()
        
        endpoints = [
            '/api/v1/predict',
            '/api/v1/ensemble/predict',
            '/api/v1/health',
            '/api/v1/metrics',
            '/dashboard',
            '/api/v1/trading/signal'
        ]
        
        for endpoint in endpoints:
            try:
                # Request rate
                rps = await self._get_endpoint_request_rate(endpoint)
                success_count = await self._get_endpoint_success_count(endpoint)
                error_count = await self._get_endpoint_error_count(endpoint)
                
                # Response time percentiles
                p50_latency = await self._get_endpoint_latency_percentile(endpoint, 50)
                p95_latency = await self._get_endpoint_latency_percentile(endpoint, 95)
                p99_latency = await self._get_endpoint_latency_percentile(endpoint, 99)
                
                throughput_metrics.append(ThroughputMetric(
                    endpoint=endpoint,
                    requests_per_second=rps,
                    successful_requests=success_count,
                    failed_requests=error_count,
                    timestamp=now,
                    response_time_p50=p50_latency,
                    response_time_p95=p95_latency,
                    response_time_p99=p99_latency
                ))
                
            except Exception as e:
                self.logger.error(f"Error collecting throughput metrics for {endpoint}: {e}")
        
        return throughput_metrics

    async def collect_cost_efficiency_metrics(self) -> List[CostMetric]:
        """Collect cost efficiency metrics"""
        cost_metrics = []
        now = datetime.utcnow()
        
        components = {
            'cloud_run': {'service_name': self.service_name},
            'cloud_sql': {'instance_name': 'shyvr-rlte-db'},
            'cloud_storage': {'bucket_name': 'shyvr-rlte-models'},
            'cloud_logging': {'project_id': self.project_id},
            'cloud_monitoring': {'project_id': self.project_id}
        }
        
        for component, config in components.items():
            try:
                # Get cost and usage data
                cost_usd = await self._get_component_cost(component, config)
                usage_units = await self._get_component_usage(component, config)
                cost_per_unit = cost_usd / usage_units if usage_units > 0 else 0.0
                budget_percent = await self._get_budget_utilization(component)
                
                cost_metrics.append(CostMetric(
                    component=component,
                    cost_usd=cost_usd,
                    usage_units=usage_units,
                    cost_per_unit=cost_per_unit,
                    timestamp=now,
                    budget_percent=budget_percent
                ))
                
            except Exception as e:
                self.logger.error(f"Error collecting cost metrics for {component}: {e}")
        
        return cost_metrics

    async def collect_cloud_run_scaling_metrics(self) -> List[PerformanceMetric]:
        """Collect Cloud Run scaling and resource metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Active instances
            active_instances = await self._get_cloud_run_active_instances()
            metrics.append(PerformanceMetric(
                name="cloud_run_active_instances",
                value=active_instances,
                unit="count",
                timestamp=now,
                labels={"service": self.service_name, "component": "cloud_run"},
                target_value=5.0,
                threshold_warning=10.0,
                threshold_critical=20.0
            ))
            
            # Request concurrency
            concurrency = await self._get_cloud_run_concurrency()
            metrics.append(PerformanceMetric(
                name="cloud_run_concurrency",
                value=concurrency,
                unit="requests",
                timestamp=now,
                labels={"service": self.service_name, "component": "cloud_run"},
                target_value=80.0,  # 80% of max concurrency
                threshold_warning=90.0,
                threshold_critical=95.0
            ))
            
            # Cold start rate
            cold_start_rate = await self._get_cloud_run_cold_start_rate()
            metrics.append(PerformanceMetric(
                name="cloud_run_cold_start_rate",
                value=cold_start_rate,
                unit="percent",
                timestamp=now,
                labels={"service": self.service_name, "component": "cloud_run"},
                target_value=5.0,  # < 5% cold starts
                threshold_warning=10.0,
                threshold_critical=20.0
            ))
            
            # Startup latency
            startup_latency = await self._get_cloud_run_startup_latency()
            metrics.append(PerformanceMetric(
                name="cloud_run_startup_latency",
                value=startup_latency,
                unit="ms",
                timestamp=now,
                labels={"service": self.service_name, "component": "cloud_run"},
                target_value=5000.0,  # 5 seconds
                threshold_warning=10000.0,
                threshold_critical=15000.0
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting Cloud Run scaling metrics: {e}")
        
        return metrics

    async def generate_performance_comparison_view(self) -> Dict[str, Any]:
        """Generate model performance comparison view"""
        comparison_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_comparisons": {},
            "ensemble_comparisons": {},
            "performance_rankings": {}
        }
        
        try:
            # Transformer model comparisons
            for model_name in self.transformer_models:
                latency = await self._get_transformer_latency_percentile(model_name, 95)
                memory = await self._get_transformer_memory_usage(model_name)
                accuracy = await self._get_transformer_accuracy(model_name)
                throughput = await self._get_transformer_throughput(model_name)
                cost_per_prediction = await self._get_model_cost_per_prediction(model_name)
                
                comparison_data["model_comparisons"][model_name] = {
                    "latency_p95_ms": latency,
                    "memory_usage_mb": memory,
                    "accuracy_percent": accuracy,
                    "throughput_rps": throughput,
                    "cost_per_prediction_usd": cost_per_prediction,
                    "efficiency_score": self._calculate_efficiency_score(
                        latency, memory, accuracy, throughput, cost_per_prediction
                    )
                }
            
            # Ensemble comparisons
            for ensemble_name in self.ensemble_models:
                latency = await self._get_ensemble_latency(ensemble_name)
                accuracy = await self._get_ensemble_consensus_accuracy(ensemble_name)
                agreement = await self._get_ensemble_agreement_rate(ensemble_name)
                
                comparison_data["ensemble_comparisons"][ensemble_name] = {
                    "prediction_latency_ms": latency,
                    "consensus_accuracy_percent": accuracy,
                    "agreement_rate_percent": agreement,
                    "ensemble_score": self._calculate_ensemble_score(latency, accuracy, agreement)
                }
            
            # Performance rankings
            comparison_data["performance_rankings"] = await self._generate_performance_rankings()
            
        except Exception as e:
            self.logger.error(f"Error generating performance comparison view: {e}")
        
        return comparison_data

    async def generate_historical_trends_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate historical performance trends analysis"""
        trends_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_period_hours": hours_back,
            "latency_trends": {},
            "throughput_trends": {},
            "accuracy_trends": {},
            "cost_trends": {},
            "anomalies_detected": []
        }
        
        try:
            # Analyze latency trends
            for model_name in self.transformer_models:
                history = self.metrics_history[f"transformer_inference_latency_p95_{model_name}"]
                if history:
                    trend = self._analyze_metric_trend(list(history))
                    trends_data["latency_trends"][model_name] = {
                        "current_value": history[-1] if history else 0.0,
                        "trend_direction": trend["direction"],
                        "trend_magnitude": trend["magnitude"],
                        "volatility": trend["volatility"],
                        "anomaly_count": trend["anomaly_count"]
                    }
            
            # Analyze throughput trends
            for endpoint, history in self.throughput_history.items():
                if history:
                    trend = self._analyze_throughput_trend(list(history))
                    trends_data["throughput_trends"][endpoint] = {
                        "current_rps": history[-1].requests_per_second if history else 0.0,
                        "trend_direction": trend["direction"],
                        "peak_rps": trend["peak_rps"],
                        "avg_response_time": trend["avg_response_time"]
                    }
            
            # Detect performance anomalies
            anomalies = await self._detect_performance_anomalies(hours_back)
            trends_data["anomalies_detected"] = anomalies
            
        except Exception as e:
            self.logger.error(f"Error generating historical trends analysis: {e}")
        
        return trends_data

    async def generate_sla_compliance_report(self) -> Dict[str, Any]:
        """Generate SLA compliance report"""
        compliance_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_compliance_score": 0.0,
            "target_compliance": {},
            "violations": [],
            "recommendations": []
        }
        
        try:
            compliant_targets = 0
            total_targets = len(self.performance_targets)
            
            for target in self.performance_targets:
                current_value = await self._get_current_metric_value(target.metric_name)
                compliance = self._evaluate_target_compliance(target, current_value)
                
                compliance_report["target_compliance"][target.metric_name] = {
                    "description": target.description,
                    "target_value": target.target_value,
                    "current_value": current_value,
                    "compliant": compliance["compliant"],
                    "compliance_percentage": compliance["percentage"],
                    "priority": target.priority
                }
                
                if compliance["compliant"]:
                    compliant_targets += 1
                else:
                    compliance_report["violations"].append({
                        "metric": target.metric_name,
                        "description": target.description,
                        "target": target.target_value,
                        "current": current_value,
                        "severity": self._calculate_violation_severity(target, current_value),
                        "priority": target.priority
                    })
            
            # Calculate overall compliance score
            compliance_report["overall_compliance_score"] = (compliant_targets / total_targets) * 100
            
            # Generate recommendations
            compliance_report["recommendations"] = await self._generate_performance_recommendations(
                compliance_report["violations"]
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SLA compliance report: {e}")
        
        return compliance_report

    async def generate_comprehensive_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard data"""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "healthy",
            "performance_score": 0.0,
            "transformer_metrics": [],
            "ensemble_metrics": [],
            "throughput_metrics": [],
            "cost_metrics": [],
            "cloud_run_metrics": [],
            "comparison_view": {},
            "historical_trends": {},
            "sla_compliance": {},
            "alerts": []
        }
        
        try:
            # Collect all metrics
            transformer_metrics = await self.collect_transformer_performance_metrics()
            ensemble_metrics = await self.collect_ensemble_performance_metrics()
            throughput_metrics = await self.collect_throughput_metrics()
            cost_metrics = await self.collect_cost_efficiency_metrics()
            cloud_run_metrics = await self.collect_cloud_run_scaling_metrics()
            
            # Convert to dictionary format
            dashboard_data["transformer_metrics"] = [asdict(m) for m in transformer_metrics]
            dashboard_data["ensemble_metrics"] = [asdict(m) for m in ensemble_metrics]
            dashboard_data["throughput_metrics"] = [asdict(m) for m in throughput_metrics]
            dashboard_data["cost_metrics"] = [asdict(m) for m in cost_metrics]
            dashboard_data["cloud_run_metrics"] = [asdict(m) for m in cloud_run_metrics]
            
            # Generate advanced views
            dashboard_data["comparison_view"] = await self.generate_performance_comparison_view()
            dashboard_data["historical_trends"] = await self.generate_historical_trends_analysis()
            dashboard_data["sla_compliance"] = await self.generate_sla_compliance_report()
            
            # Calculate overall performance score
            dashboard_data["performance_score"] = await self._calculate_overall_performance_score(
                transformer_metrics, ensemble_metrics, throughput_metrics
            )
            
            # Determine system status
            dashboard_data["system_status"] = self._determine_system_status(dashboard_data["performance_score"])
            
            # Generate performance alerts
            dashboard_data["alerts"] = await self._generate_performance_alerts(
                transformer_metrics + ensemble_metrics + cloud_run_metrics
            )
            
            # Store metrics history
            await self._store_metrics_history(transformer_metrics, throughput_metrics)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive dashboard data: {e}")
        
        return dashboard_data

    # Mock implementation methods (replace with actual metric collection)
    async def _get_transformer_latency_percentile(self, model_name: str, percentile: int) -> float:
        """Get transformer model latency percentile"""
        latency_map = {
            'itransformer': {50: 85.0, 95: 150.0, 99: 220.0},
            'patchtst': {50: 120.0, 95: 180.0, 99: 280.0},
            'timesmixer': {50: 95.0, 95: 160.0, 99: 240.0},
            'timesfm': {50: 800.0, 95: 1100.0, 99: 1400.0}
        }
        return latency_map.get(model_name, {}).get(percentile, 100.0)
    
    async def _get_transformer_memory_usage(self, model_name: str) -> float:
        """Get transformer model memory usage in MB"""
        memory_map = {
            'itransformer': 1200.0,
            'patchtst': 950.0,
            'timesmixer': 1600.0,
            'timesfm': 7200.0
        }
        return memory_map.get(model_name, 1000.0)
    
    async def _get_transformer_throughput(self, model_name: str) -> float:
        """Get transformer model throughput (predictions per second)"""
        throughput_map = {
            'itransformer': 15.0,
            'patchtst': 12.0,
            'timesmixer': 14.0,
            'timesfm': 3.0
        }
        return throughput_map.get(model_name, 10.0)
    
    async def _get_transformer_cache_hit_rate(self, model_name: str) -> float:
        """Get transformer model cache hit rate (0-1)"""
        cache_map = {
            'itransformer': 0.75,
            'patchtst': 0.82,
            'timesmixer': 0.45,
            'timesfm': 0.68
        }
        return cache_map.get(model_name, 0.70)
    
    async def _get_transformer_accuracy(self, model_name: str) -> float:
        """Get transformer model accuracy percentage"""
        accuracy_map = {
            'itransformer': 87.5,
            'patchtst': 84.2,
            'timesmixer': 86.8,
            'timesfm': 89.1
        }
        return accuracy_map.get(model_name, 85.0)
    
    async def _get_transformer_queue_depth(self, model_name: str) -> float:
        """Get transformer model queue depth"""
        return 5.0  # Mock value
    
    async def _get_ensemble_latency(self, ensemble_name: str) -> float:
        """Get ensemble prediction latency"""
        latency_map = {
            'voting_ensemble': 180.0,
            'weighted_ensemble': 165.0,
            'stacking_ensemble': 220.0
        }
        return latency_map.get(ensemble_name, 180.0)
    
    async def _get_ensemble_consensus_accuracy(self, ensemble_name: str) -> float:
        """Get ensemble consensus accuracy"""
        accuracy_map = {
            'voting_ensemble': 89.5,
            'weighted_ensemble': 91.2,
            'stacking_ensemble': 92.8
        }
        return accuracy_map.get(ensemble_name, 90.0)
    
    async def _get_ensemble_agreement_rate(self, ensemble_name: str) -> float:
        """Get ensemble model agreement rate"""
        agreement_map = {
            'voting_ensemble': 72.0,
            'weighted_ensemble': 78.5,
            'stacking_ensemble': 68.0
        }
        return agreement_map.get(ensemble_name, 70.0)
    
    async def _get_endpoint_request_rate(self, endpoint: str) -> float:
        """Get endpoint request rate (RPS)"""
        return 25.0  # Mock value
    
    async def _get_endpoint_success_count(self, endpoint: str) -> int:
        """Get endpoint successful request count"""
        return 1000  # Mock value
    
    async def _get_endpoint_error_count(self, endpoint: str) -> int:
        """Get endpoint error count"""
        return 10  # Mock value
    
    async def _get_endpoint_latency_percentile(self, endpoint: str, percentile: int) -> float:
        """Get endpoint latency percentile"""
        return 150.0  # Mock value
    
    async def _get_component_cost(self, component: str, config: Dict[str, str]) -> float:
        """Get component cost in USD"""
        cost_map = {
            'cloud_run': 45.0,
            'cloud_sql': 120.0,
            'cloud_storage': 15.0,
            'cloud_logging': 8.0,
            'cloud_monitoring': 12.0
        }
        return cost_map.get(component, 10.0)
    
    async def _get_component_usage(self, component: str, config: Dict[str, str]) -> float:
        """Get component usage units"""
        usage_map = {
            'cloud_run': 1000000.0,  # requests
            'cloud_sql': 168.0,      # hours
            'cloud_storage': 100.0,  # GB
            'cloud_logging': 1000.0, # entries
            'cloud_monitoring': 500.0 # metrics
        }
        return usage_map.get(component, 100.0)
    
    async def _get_budget_utilization(self, component: str) -> float:
        """Get budget utilization percentage"""
        return 65.0  # Mock value
    
    async def _get_cloud_run_active_instances(self) -> float:
        """Get Cloud Run active instances count"""
        return 3.0  # Mock value
    
    async def _get_cloud_run_concurrency(self) -> float:
        """Get Cloud Run request concurrency"""
        return 75.0  # Mock value
    
    async def _get_cloud_run_cold_start_rate(self) -> float:
        """Get Cloud Run cold start rate percentage"""
        return 8.0  # Mock value
    
    async def _get_cloud_run_startup_latency(self) -> float:
        """Get Cloud Run startup latency in ms"""
        return 4500.0  # Mock value
    
    async def _get_model_cost_per_prediction(self, model_name: str) -> float:
        """Get cost per prediction for model"""
        cost_map = {
            'itransformer': 0.0008,
            'patchtst': 0.0006,
            'timesmixer': 0.0009,
            'timesfm': 0.0025
        }
        return cost_map.get(model_name, 0.001)
    
    def _calculate_efficiency_score(self, latency: float, memory: float, accuracy: float, 
                                   throughput: float, cost: float) -> float:
        """Calculate model efficiency score"""
        # Normalize metrics and calculate composite score
        normalized_latency = max(0, 100 - (latency / 10))  # Lower is better
        normalized_memory = max(0, 100 - (memory / 100))   # Lower is better
        normalized_accuracy = accuracy                      # Higher is better
        normalized_throughput = min(100, throughput * 5)   # Higher is better
        normalized_cost = max(0, 100 - (cost * 1000))     # Lower is better
        
        return (normalized_latency + normalized_memory + normalized_accuracy + 
                normalized_throughput + normalized_cost) / 5
    
    def _calculate_ensemble_score(self, latency: float, accuracy: float, agreement: float) -> float:
        """Calculate ensemble performance score"""
        normalized_latency = max(0, 100 - (latency / 5))
        normalized_accuracy = accuracy
        normalized_agreement = agreement
        
        return (normalized_latency + normalized_accuracy + normalized_agreement) / 3
    
    async def _generate_performance_rankings(self) -> Dict[str, Any]:
        """Generate performance rankings"""
        return {
            "best_latency": "itransformer",
            "best_accuracy": "timesfm",
            "best_efficiency": "patchtst",
            "best_throughput": "itransformer",
            "best_ensemble": "stacking_ensemble"
        }
    
    def _analyze_metric_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze metric trend from historical values"""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0, "volatility": 0.0, "anomaly_count": 0}
        
        # Simple trend analysis
        recent = values[-5:] if len(values) >= 5 else values
        trend = (recent[-1] - recent[0]) / len(recent)
        
        return {
            "direction": "up" if trend > 0.1 else "down" if trend < -0.1 else "stable",
            "magnitude": abs(trend),
            "volatility": sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / len(recent),
            "anomaly_count": 0  # Mock value
        }
    
    def _analyze_throughput_trend(self, metrics: List[ThroughputMetric]) -> Dict[str, Any]:
        """Analyze throughput trend"""
        if not metrics:
            return {"direction": "stable", "peak_rps": 0.0, "avg_response_time": 0.0}
        
        rps_values = [m.requests_per_second for m in metrics]
        response_times = [m.response_time_p95 for m in metrics if m.response_time_p95]
        
        return {
            "direction": "up" if rps_values[-1] > rps_values[0] else "down",
            "peak_rps": max(rps_values),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0
        }
    
    async def _detect_performance_anomalies(self, hours_back: int) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        return []  # Mock implementation
    
    async def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value for a metric"""
        return 85.0  # Mock value
    
    def _evaluate_target_compliance(self, target: PerformanceTarget, current_value: float) -> Dict[str, Any]:
        """Evaluate compliance for a performance target"""
        if target.operator == "lt":
            compliant = current_value < target.target_value
            percentage = min(100, (target.target_value - current_value) / target.target_value * 100)
        elif target.operator == "gt":
            compliant = current_value > target.target_value
            percentage = min(100, (current_value - target.target_value) / target.target_value * 100)
        else:  # eq
            compliant = abs(current_value - target.target_value) < 0.01
            percentage = 100.0 if compliant else 0.0
        
        return {"compliant": compliant, "percentage": max(0, percentage)}
    
    def _calculate_violation_severity(self, target: PerformanceTarget, current_value: float) -> str:
        """Calculate violation severity"""
        deviation = abs(current_value - target.target_value) / target.target_value
        if deviation > 0.5:
            return "critical"
        elif deviation > 0.2:
            return "high"
        else:
            return "medium"
    
    async def _generate_performance_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        for violation in violations:
            if "latency" in violation["metric"]:
                recommendations.append(f"Consider optimizing {violation['metric']} - current value {violation['current']} exceeds target {violation['target']}")
            elif "memory" in violation["metric"]:
                recommendations.append(f"Memory optimization needed for {violation['metric']}")
        return recommendations
    
    async def _calculate_overall_performance_score(self, transformer_metrics: List[PerformanceMetric],
                                                   ensemble_metrics: List[PerformanceMetric],
                                                   throughput_metrics: List[ThroughputMetric]) -> float:
        """Calculate overall performance score"""
        return 85.0  # Mock implementation
    
    def _determine_system_status(self, performance_score: float) -> str:
        """Determine system status based on performance score"""
        if performance_score >= 90:
            return "excellent"
        elif performance_score >= 80:
            return "healthy"
        elif performance_score >= 70:
            return "degraded"
        else:
            return "critical"
    
    async def _generate_performance_alerts(self, all_metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate performance alerts"""
        alerts = []
        for metric in all_metrics:
            if metric.threshold_critical and metric.value > metric.threshold_critical:
                alerts.append({
                    "severity": "critical",
                    "metric": metric.name,
                    "current_value": metric.value,
                    "threshold": metric.threshold_critical,
                    "message": f"{metric.name} critical threshold exceeded",
                    "timestamp": metric.timestamp.isoformat()
                })
            elif metric.threshold_warning and metric.value > metric.threshold_warning:
                alerts.append({
                    "severity": "warning",
                    "metric": metric.name,
                    "current_value": metric.value,
                    "threshold": metric.threshold_warning,
                    "message": f"{metric.name} warning threshold exceeded",
                    "timestamp": metric.timestamp.isoformat()
                })
        return alerts
    
    async def _store_metrics_history(self, transformer_metrics: List[PerformanceMetric],
                                   throughput_metrics: List[ThroughputMetric]):
        """Store metrics in history for trend analysis"""
        for metric in transformer_metrics:
            key = f"{metric.name}_{metric.labels.get('model', '')}"
            self.metrics_history[key].append(metric.value)
        
        for metric in throughput_metrics:
            self.throughput_history[metric.endpoint].append(metric)


class PerformanceDashboardServer:
    """HTTP server for performance tracking dashboard"""
    
    def __init__(self, dashboard: PerformanceTrackingDashboard, port: int = 8081):
        self.dashboard = dashboard
        self.port = port
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def start_server(self):
        """Start the performance dashboard server"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/performance/dashboard', self.get_dashboard)
        app.router.add_get('/performance/metrics', self.get_metrics)
        app.router.add_get('/performance/transformers', self.get_transformer_metrics)
        app.router.add_get('/performance/ensembles', self.get_ensemble_metrics)
        app.router.add_get('/performance/throughput', self.get_throughput_metrics)
        app.router.add_get('/performance/costs', self.get_cost_metrics)
        app.router.add_get('/performance/comparison', self.get_comparison_view)
        app.router.add_get('/performance/trends', self.get_trends_analysis)
        app.router.add_get('/performance/sla', self.get_sla_compliance)
        app.router.add_get('/performance/health', self.get_health_status)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        self.logger.info(f"Performance tracking dashboard started on port {self.port}")
    
    async def get_dashboard(self, request):
        """Get comprehensive dashboard data"""
        from aiohttp import web
        
        try:
            dashboard_data = await self.dashboard.generate_comprehensive_dashboard_data()
            return web.json_response(dashboard_data)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to generate dashboard data: {e}"}, 
                status=500
            )
    
    async def get_metrics(self, request):
        """Get all performance metrics"""
        from aiohttp import web
        
        try:
            transformer_metrics = await self.dashboard.collect_transformer_performance_metrics()
            ensemble_metrics = await self.dashboard.collect_ensemble_performance_metrics()
            cloud_run_metrics = await self.dashboard.collect_cloud_run_scaling_metrics()
            
            all_metrics = {
                "transformer_metrics": [asdict(m) for m in transformer_metrics],
                "ensemble_metrics": [asdict(m) for m in ensemble_metrics],
                "cloud_run_metrics": [asdict(m) for m in cloud_run_metrics]
            }
            return web.json_response(all_metrics)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect metrics: {e}"}, 
                status=500
            )
    
    async def get_transformer_metrics(self, request):
        """Get transformer performance metrics"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_transformer_performance_metrics()
            return web.json_response([asdict(m) for m in metrics])
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect transformer metrics: {e}"}, 
                status=500
            )
    
    async def get_ensemble_metrics(self, request):
        """Get ensemble performance metrics"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_ensemble_performance_metrics()
            return web.json_response([asdict(m) for m in metrics])
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect ensemble metrics: {e}"}, 
                status=500
            )
    
    async def get_throughput_metrics(self, request):
        """Get throughput metrics"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_throughput_metrics()
            return web.json_response([asdict(m) for m in metrics])
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect throughput metrics: {e}"}, 
                status=500
            )
    
    async def get_cost_metrics(self, request):
        """Get cost efficiency metrics"""
        from aiohttp import web
        
        try:
            metrics = await self.dashboard.collect_cost_efficiency_metrics()
            return web.json_response([asdict(m) for m in metrics])
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to collect cost metrics: {e}"}, 
                status=500
            )
    
    async def get_comparison_view(self, request):
        """Get performance comparison view"""
        from aiohttp import web
        
        try:
            comparison = await self.dashboard.generate_performance_comparison_view()
            return web.json_response(comparison)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to generate comparison view: {e}"}, 
                status=500
            )
    
    async def get_trends_analysis(self, request):
        """Get historical trends analysis"""
        from aiohttp import web
        
        try:
            hours_back = int(request.query.get('hours', 24))
            trends = await self.dashboard.generate_historical_trends_analysis(hours_back)
            return web.json_response(trends)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to generate trends analysis: {e}"}, 
                status=500
            )
    
    async def get_sla_compliance(self, request):
        """Get SLA compliance report"""
        from aiohttp import web
        
        try:
            compliance = await self.dashboard.generate_sla_compliance_report()
            return web.json_response(compliance)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to generate SLA compliance report: {e}"}, 
                status=500
            )
    
    async def get_health_status(self, request):
        """Get system health status"""
        from aiohttp import web
        
        try:
            dashboard_data = await self.dashboard.generate_comprehensive_dashboard_data()
            health_status = {
                "status": dashboard_data["system_status"],
                "performance_score": dashboard_data["performance_score"],
                "timestamp": dashboard_data["timestamp"],
                "active_alerts": len(dashboard_data["alerts"]),
                "sla_compliance_score": dashboard_data["sla_compliance"]["overall_compliance_score"]
            }
            return web.json_response(health_status)
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get health status: {e}"}, 
                status=500
            )


async def main():
    """Main function to run the performance tracking dashboard"""
    import os
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "shyvr-rlte")
    region = os.getenv("GCP_REGION", "us-central1")
    
    # Create performance tracking dashboard
    dashboard = PerformanceTrackingDashboard(project_id, region)
    
    # Create server
    server = PerformanceDashboardServer(dashboard)
    
    # Start server
    await server.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down performance tracking dashboard")


if __name__ == "__main__":
    asyncio.run(main())