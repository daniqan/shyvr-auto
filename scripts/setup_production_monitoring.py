#!/usr/bin/env python3
"""
Production Database Monitoring and Alerting Setup
Creates comprehensive monitoring for the RL experience database
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from google.cloud import monitoring_v3
from google.cloud import logging as cloud_logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


class ProductionMonitoringSetup:
    """Production monitoring and alerting setup"""
    
    def __init__(self):
        self.project_id = "shvyr-ai-bots"
        self.instance_name = "shyvr-rlte-db-prod"
        self.database_name = "shyvr_rlte_prod"
        
        # Initialize clients
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.alerting_client = monitoring_v3.AlertPolicyServiceClient()
        self.logging_client = cloud_logging.Client()
        
        self.project_name = f"projects/{self.project_id}"
    
    def create_custom_metrics(self) -> List[Dict]:
        """Create custom metrics for RL experience storage"""
        metrics = []
        
        # RL Experience Storage Metrics
        rl_metrics = [
            {
                "type": "custom.googleapis.com/rl/experience_storage/total_experiences",
                "labels": [
                    {"key": "session_id", "value_type": "STRING"},
                    {"key": "trading_mode", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "INT64",
                "display_name": "Total RL Experiences",
                "description": "Total number of RL experiences stored in database"
            },
            {
                "type": "custom.googleapis.com/rl/experience_storage/experiences_per_minute",
                "labels": [
                    {"key": "trading_mode", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "RL Experiences Per Minute",
                "description": "Rate of RL experience storage per minute"
            },
            {
                "type": "custom.googleapis.com/rl/experience_storage/average_reward",
                "labels": [
                    {"key": "session_id", "value_type": "STRING"},
                    {"key": "trading_mode", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Average RL Reward",
                "description": "Average reward from RL experiences"
            },
            {
                "type": "custom.googleapis.com/rl/experience_storage/query_latency",
                "labels": [
                    {"key": "query_type", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "RL Database Query Latency",
                "description": "Latency of RL experience database queries"
            },
            {
                "type": "custom.googleapis.com/rl/experience_storage/active_sessions",
                "labels": [],
                "metric_kind": "GAUGE",
                "value_type": "INT64",
                "display_name": "Active RL Training Sessions",
                "description": "Number of active RL training sessions"
            }
        ]
        
        # Transformer Model Metrics
        transformer_metrics = [
            {
                "type": "custom.googleapis.com/transformer/memory_usage_mb",
                "labels": [
                    {"key": "model_type", "value_type": "STRING"},
                    {"key": "model_name", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Transformer Memory Usage (MB)",
                "description": "Memory usage of transformer models in MB"
            },
            {
                "type": "custom.googleapis.com/transformer/inference_latency_ms",
                "labels": [
                    {"key": "model_type", "value_type": "STRING"},
                    {"key": "model_name", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Transformer Inference Latency (ms)",
                "description": "Inference latency of transformer models in milliseconds"
            },
            {
                "type": "custom.googleapis.com/transformer/cache_hit_rate",
                "labels": [
                    {"key": "model_type", "value_type": "STRING"},
                    {"key": "model_name", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Transformer Cache Hit Rate",
                "description": "Cache hit rate for transformer model predictions"
            },
            {
                "type": "custom.googleapis.com/transformer/loading_time_ms",
                "labels": [
                    {"key": "model_type", "value_type": "STRING"},
                    {"key": "model_name", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Transformer Loading Time (ms)",
                "description": "Time taken to load transformer models in milliseconds"
            },
            {
                "type": "custom.googleapis.com/transformer/health_status",
                "labels": [
                    {"key": "model_type", "value_type": "STRING"},
                    {"key": "model_name", "value_type": "STRING"}
                ],
                "metric_kind": "GAUGE",
                "value_type": "DOUBLE",
                "display_name": "Transformer Health Status",
                "description": "Health status of transformer models (1.0=healthy, 0.0=unhealthy)"
            }
        ]
        
        # Combine all metrics
        all_metrics = rl_metrics + transformer_metrics
        
        for metric_config in all_metrics:
            try:
                descriptor = monitoring_v3.MetricDescriptor(
                    type=metric_config["type"],
                    metric_kind=getattr(monitoring_v3.MetricDescriptor.MetricKind, metric_config["metric_kind"]),
                    value_type=getattr(monitoring_v3.MetricDescriptor.ValueType, metric_config["value_type"]),
                    display_name=metric_config["display_name"],
                    description=metric_config["description"],
                    labels=[
                        monitoring_v3.LabelDescriptor(
                            key=label["key"],
                            value_type=getattr(monitoring_v3.LabelDescriptor.ValueType, label["value_type"])
                        ) for label in metric_config["labels"]
                    ]
                )
                
                # Create the metric descriptor
                created_descriptor = self.monitoring_client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=descriptor
                )
                
                logger.info(f"Created custom metric: {metric_config['type']}")
                metrics.append({
                    "type": metric_config["type"],
                    "descriptor": created_descriptor,
                    "status": "created"
                })
                
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Metric already exists: {metric_config['type']}")
                    metrics.append({
                        "type": metric_config["type"],
                        "status": "exists"
                    })
                else:
                    logger.error(f"Failed to create metric {metric_config['type']}: {e}")
                    metrics.append({
                        "type": metric_config["type"],
                        "status": "failed",
                        "error": str(e)
                    })
        
        return metrics
    
    def create_alert_policies(self) -> List[Dict]:
        """Create alert policies for RL experience monitoring"""
        policies = []
        
        # Alert Policy Configurations
        alert_configs = [
            {
                "display_name": "RL Database High Connection Count",
                "documentation": "Alert when Cloud SQL instance has too many connections",
                "conditions": [{
                    "display_name": "High connection count",
                    "condition_threshold": {
                        "filter": f'resource.type="cloudsql_database" AND resource.labels.database_id="{self.project_id}:{self.instance_name}" AND metric.type="cloudsql.googleapis.com/database/network/connections"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 80,
                        "duration": "300s",  # 5 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "RL Experience Storage Low Performance",
                "documentation": "Alert when RL experience queries are slow",
                "conditions": [{
                    "display_name": "High query latency",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/rl/experience_storage/query_latency"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 2.0,  # 2 seconds
                        "duration": "180s",  # 3 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "RL Experience Storage Rate Anomaly",
                "documentation": "Alert when experience storage rate is abnormally low",
                "conditions": [{
                    "display_name": "Low experience storage rate",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/rl/experience_storage/experiences_per_minute"',
                        "comparison": "COMPARISON_LESS_THAN",
                        "threshold_value": 1.0,  # Less than 1 experience per minute
                        "duration": "600s",  # 10 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_RATE"
                        }]
                    }
                }]
            },
            {
                "display_name": "RL Database High CPU Usage",
                "documentation": "Alert when Cloud SQL CPU usage is high",
                "conditions": [{
                    "display_name": "High CPU utilization",
                    "condition_threshold": {
                        "filter": f'resource.type="cloudsql_database" AND resource.labels.database_id="{self.project_id}:{self.instance_name}" AND metric.type="cloudsql.googleapis.com/database/cpu/utilization"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 0.8,  # 80% CPU
                        "duration": "300s",  # 5 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "RL Database Memory Usage High",
                "documentation": "Alert when Cloud SQL memory usage is high", 
                "conditions": [{
                    "display_name": "High memory utilization",
                    "condition_threshold": {
                        "filter": f'resource.type="cloudsql_database" AND resource.labels.database_id="{self.project_id}:{self.instance_name}" AND metric.type="cloudsql.googleapis.com/database/memory/utilization"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 0.85,  # 85% memory
                        "duration": "300s",  # 5 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "Transformer High Memory Usage",
                "documentation": "Alert when transformer model memory usage exceeds 90% of 8Gi",
                "conditions": [{
                    "display_name": "Transformer memory usage > 90% of 8Gi",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/transformer/memory_usage_mb"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 7372.8,  # 90% of 8192MB
                        "duration": "300s",  # 5 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "Transformer Slow Inference Latency",
                "documentation": "Alert when transformer inference latency exceeds 1000ms",
                "conditions": [{
                    "display_name": "Transformer inference latency > 1000ms",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/transformer/inference_latency_ms"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "threshold_value": 1000.0,
                        "duration": "180s",  # 3 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_PERCENTILE_99"
                        }]
                    }
                }]
            },
            {
                "display_name": "Transformer Low Cache Hit Rate",
                "documentation": "Alert when transformer cache hit rate is below 50%",
                "conditions": [{
                    "display_name": "Transformer cache hit rate < 50%",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/transformer/cache_hit_rate"',
                        "comparison": "COMPARISON_LESS_THAN",
                        "threshold_value": 0.5,
                        "duration": "600s",  # 10 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            },
            {
                "display_name": "Transformer Model Unhealthy",
                "documentation": "Alert when transformer model health status indicates unhealthy state",
                "conditions": [{
                    "display_name": "Transformer health status unhealthy",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/transformer/health_status"',
                        "comparison": "COMPARISON_LESS_THAN",
                        "threshold_value": 1.0,
                        "duration": "120s",  # 2 minutes
                        "aggregations": [{
                            "alignment_period": "60s",
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }]
            }
        ]
        
        for config in alert_configs:
            try:
                # Build conditions
                conditions = []
                for cond_config in config["conditions"]:
                    condition = monitoring_v3.AlertPolicy.Condition(
                        display_name=cond_config["display_name"],
                        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                            filter=cond_config["condition_threshold"]["filter"],
                            comparison=getattr(
                                monitoring_v3.ComparisonType, 
                                cond_config["condition_threshold"]["comparison"]
                            ),
                            threshold_value=cond_config["condition_threshold"]["threshold_value"],
                            duration=monitoring_v3.Duration(
                                seconds=int(cond_config["condition_threshold"]["duration"][:-1])
                            ),
                            aggregations=[
                                monitoring_v3.Aggregation(
                                    alignment_period=monitoring_v3.Duration(
                                        seconds=int(agg["alignment_period"][:-1])
                                    ),
                                    per_series_aligner=getattr(
                                        monitoring_v3.Aggregation.Aligner,
                                        agg["per_series_aligner"]
                                    )
                                ) for agg in cond_config["condition_threshold"]["aggregations"]
                            ]
                        )
                    )
                    conditions.append(condition)
                
                # Create alert policy
                policy = monitoring_v3.AlertPolicy(
                    display_name=config["display_name"],
                    documentation=monitoring_v3.AlertPolicy.Documentation(
                        content=config["documentation"]
                    ),
                    conditions=conditions,
                    combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
                    enabled=True
                )
                
                created_policy = self.alerting_client.create_alert_policy(
                    name=self.project_name,
                    alert_policy=policy
                )
                
                logger.info(f"Created alert policy: {config['display_name']}")
                policies.append({
                    "name": config["display_name"],
                    "policy": created_policy,
                    "status": "created"
                })
                
            except Exception as e:
                if "already exists" in str(e) or "duplicate" in str(e).lower():
                    logger.info(f"Alert policy already exists: {config['display_name']}")
                    policies.append({
                        "name": config["display_name"],
                        "status": "exists"
                    })
                else:
                    logger.error(f"Failed to create alert policy {config['display_name']}: {e}")
                    policies.append({
                        "name": config["display_name"],
                        "status": "failed",
                        "error": str(e)
                    })
        
        return policies
    
    def setup_log_based_alerts(self) -> List[Dict]:
        """Set up log-based alerts for database errors"""
        log_alerts = []
        
        # Log-based alert configurations
        log_alert_configs = [
            {
                "display_name": "RL Database Connection Errors",
                "documentation": "Alert on database connection failures",
                "log_filter": f'''
                resource.type="cloudsql_database"
                resource.labels.database_id="{self.project_id}:{self.instance_name}"
                (jsonPayload.message=~".*connection.*failed.*" OR
                 jsonPayload.message=~".*timeout.*" OR
                 jsonPayload.message=~".*refused.*")
                severity>=ERROR
                ''',
                "threshold": 5,  # 5 errors in time window
                "duration": "300s"  # 5 minutes
            },
            {
                "display_name": "RL Experience Storage Errors",
                "documentation": "Alert on RL experience storage errors",
                "log_filter": '''
                resource.type="gce_instance" OR resource.type="cloud_function"
                (jsonPayload.message=~".*rl.*experience.*error.*" OR
                 jsonPayload.message=~".*experience.*storage.*failed.*" OR
                 textPayload=~".*rl.*experience.*error.*")
                severity>=ERROR
                ''',
                "threshold": 3,  # 3 errors in time window
                "duration": "180s"  # 3 minutes
            }
        ]
        
        for config in log_alert_configs:
            try:
                # Create log-based condition
                condition = monitoring_v3.AlertPolicy.Condition(
                    display_name=f"Log match: {config['display_name']}",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=f'resource.type="logging" AND metric.type="logging.googleapis.com/user/rl_database_errors"',
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                        threshold_value=config["threshold"],
                        duration=monitoring_v3.Duration(
                            seconds=int(config["duration"][:-1])
                        )
                    )
                )
                
                # Create the alert policy
                policy = monitoring_v3.AlertPolicy(
                    display_name=config["display_name"],
                    documentation=monitoring_v3.AlertPolicy.Documentation(
                        content=config["documentation"]
                    ),
                    conditions=[condition],
                    combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
                    enabled=True
                )
                
                created_policy = self.alerting_client.create_alert_policy(
                    name=self.project_name,
                    alert_policy=policy
                )
                
                logger.info(f"Created log-based alert: {config['display_name']}")
                log_alerts.append({
                    "name": config["display_name"],
                    "policy": created_policy,
                    "status": "created"
                })
                
            except Exception as e:
                logger.warning(f"Log-based alert creation failed (expected): {config['display_name']}: {e}")
                log_alerts.append({
                    "name": config["display_name"],
                    "status": "partial",
                    "note": "Log-based alerts require manual setup in Cloud Logging"
                })
        
        return log_alerts
    
    def create_dashboard_config(self) -> Dict:
        """Create Grafana dashboard configuration for RL monitoring"""
        dashboard_config = {
            "dashboard": {
                "title": "RL Experience Storage Production Monitoring",
                "tags": ["rl", "database", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Total RL Experiences",
                        "type": "stat",
                        "targets": [{
                            "expr": 'custom_googleapis_com_rl_experience_storage_total_experiences',
                            "legendFormat": "{{trading_mode}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "short"
                            }
                        }
                    },
                    {
                        "title": "Experience Storage Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": 'rate(custom_googleapis_com_rl_experience_storage_experiences_per_minute[5m])',
                            "legendFormat": "{{trading_mode}}"
                        }],
                        "yAxes": [{
                            "label": "Experiences/min",
                            "min": 0
                        }]
                    },
                    {
                        "title": "Average RL Reward",
                        "type": "graph",
                        "targets": [{
                            "expr": 'custom_googleapis_com_rl_experience_storage_average_reward',
                            "legendFormat": "{{session_id}}"
                        }],
                        "yAxes": [{
                            "label": "Reward"
                        }]
                    },
                    {
                        "title": "Database Query Latency",
                        "type": "graph",
                        "targets": [{
                            "expr": 'custom_googleapis_com_rl_experience_storage_query_latency',
                            "legendFormat": "{{query_type}}"
                        }],
                        "yAxes": [{
                            "label": "Seconds",
                            "min": 0
                        }]
                    },
                    {
                        "title": "Cloud SQL CPU Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": f'cloudsql_googleapis_com_database_cpu_utilization{{database_id="{self.project_id}:{self.instance_name}"}}',
                            "legendFormat": "CPU %"
                        }],
                        "yAxes": [{
                            "label": "Percentage",
                            "min": 0,
                            "max": 100
                        }]
                    },
                    {
                        "title": "Cloud SQL Memory Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": f'cloudsql_googleapis_com_database_memory_utilization{{database_id="{self.project_id}:{self.instance_name}"}}',
                            "legendFormat": "Memory %"
                        }],
                        "yAxes": [{
                            "label": "Percentage",
                            "min": 0,
                            "max": 100
                        }]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        return dashboard_config
    
    async def setup_monitoring(self) -> Dict[str, Any]:
        """Set up complete production monitoring"""
        results = {
            "custom_metrics": [],
            "alert_policies": [],
            "log_alerts": [],
            "dashboard_config": {},
            "status": "success"
        }
        
        try:
            logger.info("Setting up production monitoring for RL experience storage...")
            
            # Create custom metrics
            logger.info("Creating custom metrics...")
            results["custom_metrics"] = self.create_custom_metrics()
            
            # Create alert policies  
            logger.info("Creating alert policies...")
            results["alert_policies"] = self.create_alert_policies()
            
            # Set up log-based alerts
            logger.info("Setting up log-based alerts...")
            results["log_alerts"] = self.setup_log_based_alerts()
            
            # Create dashboard configuration
            logger.info("Creating dashboard configuration...")
            results["dashboard_config"] = self.create_dashboard_config()
            
            # Save dashboard config to file
            dashboard_file = Path(__file__).parent.parent / "monitoring" / "grafana" / "rl_production_dashboard.json"
            dashboard_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dashboard_file, 'w') as f:
                json.dump(results["dashboard_config"], f, indent=2)
            
            logger.info(f"Dashboard configuration saved to: {dashboard_file}")
            
            # Summary
            metrics_created = sum(1 for m in results["custom_metrics"] if m["status"] == "created")
            alerts_created = sum(1 for a in results["alert_policies"] if a["status"] == "created")
            
            logger.info(f"Monitoring setup completed:")
            logger.info(f"  - Custom metrics: {metrics_created} created")
            logger.info(f"  - Alert policies: {alerts_created} created")
            logger.info(f"  - Dashboard config: saved")
            
            return results
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results


async def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    setup = ProductionMonitoringSetup()
    results = await setup.setup_monitoring()
    
    if results["status"] == "success":
        print("\n✅ Production monitoring setup completed successfully!")
        print(f"📊 Created {len(results['custom_metrics'])} custom metrics")
        print(f"🚨 Created {len(results['alert_policies'])} alert policies")
        print(f"📝 Dashboard configuration saved")
    else:
        print(f"\n❌ Monitoring setup failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())