# Monitoring and Observability Setup Guide

## Table of Contents
1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Transformer-Specific Monitoring](#transformer-specific-monitoring)
4. [GCP Cloud Monitoring Integration](#gcp-cloud-monitoring-integration)
5. [Custom Metrics and Dashboards](#custom-metrics-and-dashboards)
6. [Alert Policies and Notification](#alert-policies-and-notification)
7. [Health Check Endpoints](#health-check-endpoints)
8. [Performance Monitoring](#performance-monitoring)
9. [Log Aggregation and Analysis](#log-aggregation-and-analysis)
10. [Distributed Tracing](#distributed-tracing)
11. [Business Metrics Monitoring](#business-metrics-monitoring)
12. [Incident Response Integration](#incident-response-integration)

## Overview

This guide provides comprehensive instructions for setting up monitoring and observability for the transformer-enabled RLTE system. The monitoring stack includes transformer-specific metrics, traditional application metrics, business intelligence, and proactive alerting.

### Key Monitoring Components
- **GCP Cloud Monitoring**: Primary observability platform
- **Custom Transformer Metrics**: Model-specific performance tracking
- **Health Check Endpoints**: Service availability monitoring
- **Alert Policies**: Proactive incident detection
- **Performance Analytics**: Resource utilization tracking
- **Business Metrics**: Trading performance monitoring

### Monitoring Objectives
1. **System Reliability**: 99.9% uptime monitoring
2. **Performance Optimization**: Sub-100ms inference latency
3. **Resource Efficiency**: Cost-optimized resource utilization
4. **Model Performance**: Transformer accuracy and drift detection
5. **Business Impact**: Trading performance correlation

## Monitoring Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    RLTE Application                         │
│  ┌─────────────────────────────────────────────────────────│
│  │  Transformer Models | ML Pipeline | RL Agent            │
│  │  ┌─────────────────────────────────────────────────────┐│
│  │  │          Metrics Collection Layer                   ││
│  │  │  • Performance Metrics  • Health Checks            ││
│  │  │  • Resource Usage      • Business Metrics          ││
│  │  └─────────────────────────────────────────────────────┘│
│  └─────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────┘
              │                    │                    │
    ┌─────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │ Cloud Monitoring │  │  Cloud Logging  │  │  Cloud Trace    │
    │   • Metrics      │  │   • App Logs    │  │  • Distributed  │
    │   • Dashboards   │  │   • Audit Logs  │  │    Tracing      │
    │   • Alerts       │  │   • Error Logs  │  │   • Latency     │
    └─────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                    │                    │
    ┌─────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │   Alertmanager   │  │   Log Analysis  │  │  APM Dashboard  │
    │   • Notifications│  │   • Error       │  │  • Performance  │
    │   • Escalation   │  │     Detection   │  │    Analytics    │
    │   • Suppression  │  │   • Trends      │  │  • Bottlenecks  │
    └──────────────────┘  └─────────────────┘  └─────────────────┘
```

### Monitoring Stack Components
```yaml
monitoring_stack:
  primary_platform: "GCP Cloud Monitoring"
  log_aggregation: "GCP Cloud Logging"
  tracing: "GCP Cloud Trace"
  metrics_collection: "Prometheus-compatible"
  visualization: "Cloud Monitoring Dashboards"
  alerting: "Cloud Monitoring Alert Policies"
  notification_channels:
    - email
    - slack
    - pagerduty
```

## Transformer-Specific Monitoring

### Model Performance Metrics

#### 1. Core Transformer Metrics
```python
# Transformer-specific metrics collected by the system
TRANSFORMER_METRICS = {
    # Memory utilization
    'transformer_memory_usage_bytes': {
        'description': 'Memory consumption by transformer model',
        'labels': ['model_type', 'model_version', 'instance_id'],
        'unit': 'bytes',
        'threshold_critical': 7_372_800_000,  # 7GB (90% of 8GB)
        'threshold_warning': 6_442_450_944,   # 6GB (75% of 8GB)
    },
    
    # Inference performance
    'transformer_inference_latency_ms': {
        'description': 'Time taken for model inference',
        'labels': ['model_type', 'batch_size', 'sequence_length'],
        'unit': 'milliseconds',
        'threshold_critical': 1000,  # 1 second
        'threshold_warning': 500,    # 500ms
    },
    
    # Cache performance
    'transformer_cache_hit_rate': {
        'description': 'Model cache hit rate percentage',
        'labels': ['model_type', 'cache_type'],
        'unit': 'percentage',
        'threshold_critical': 30,    # Below 30% hit rate
        'threshold_warning': 50,     # Below 50% hit rate
    },
    
    # Model health
    'transformer_model_health': {
        'description': 'Overall model health status',
        'labels': ['model_type', 'health_check_type'],
        'unit': 'boolean',
        'values': {'unhealthy': 0, 'healthy': 1},
    },
    
    # Attention pattern metrics
    'transformer_attention_entropy': {
        'description': 'Attention pattern entropy for drift detection',
        'labels': ['model_type', 'layer', 'head'],
        'unit': 'entropy',
        'threshold_warning': 'dynamic',  # Based on baseline
    }
}
```

#### 2. Model-Specific Resource Profiles
```yaml
# Resource monitoring per transformer model
resource_profiles:
  iTransformer:
    memory_baseline: 4_000_000_000    # 4GB baseline
    cpu_baseline: 2.0                 # 2 CPU cores
    inference_latency_p95: 200        # 200ms P95 latency
    cache_hit_target: 70              # 70% cache hit rate
    
  PatchTST:
    memory_baseline: 3_000_000_000    # 3GB baseline
    cpu_baseline: 2.0                 # 2 CPU cores
    inference_latency_p95: 150        # 150ms P95 latency
    cache_hit_target: 75              # 75% cache hit rate
    
  TimesMixer:
    memory_baseline: 5_000_000_000    # 5GB baseline
    cpu_baseline: 3.0                 # 3 CPU cores
    inference_latency_p95: 300        # 300ms P95 latency
    cache_hit_target: 65              # 65% cache hit rate
    
  TimesFM:
    memory_baseline: 6_000_000_000    # 6GB baseline
    cpu_baseline: 4.0                 # 4 CPU cores
    inference_latency_p95: 400        # 400ms P95 latency
    cache_hit_target: 60              # 60% cache hit rate
```

### Dynamic Resource Monitoring

#### 1. Resource Allocation Metrics
```python
RESOURCE_ALLOCATION_METRICS = {
    # Dynamic allocation decisions
    'resource_allocation_events': {
        'description': 'Resource allocation decision events',
        'labels': ['allocation_type', 'model_type', 'decision_reason'],
        'unit': 'count',
    },
    
    # Resource utilization efficiency
    'resource_utilization_efficiency': {
        'description': 'Resource utilization efficiency score',
        'labels': ['resource_type', 'model_type'],
        'unit': 'percentage',
        'threshold_warning': 50,  # Below 50% efficiency
    },
    
    # Auto-scaling events
    'autoscaling_events': {
        'description': 'Auto-scaling events and decisions',
        'labels': ['scaling_direction', 'trigger_reason', 'model_type'],
        'unit': 'count',
    },
    
    # Cost optimization metrics
    'resource_cost_per_prediction': {
        'description': 'Cost per prediction by model type',
        'labels': ['model_type', 'resource_allocation'],
        'unit': 'currency',
    }
}
```

#### 2. Model Lifecycle Monitoring
```python
LIFECYCLE_METRICS = {
    # Model loading performance
    'model_loading_duration_seconds': {
        'description': 'Time taken to load model',
        'labels': ['model_type', 'model_version', 'loading_stage'],
        'unit': 'seconds',
        'threshold_warning': 300,  # 5 minutes
    },
    
    # Hot-swap operations
    'model_hotswap_success_rate': {
        'description': 'Success rate of model hot-swap operations',
        'labels': ['model_type', 'swap_strategy'],
        'unit': 'percentage',
        'threshold_critical': 90,  # Below 90% success
    },
    
    # Version rollback events
    'model_rollback_events': {
        'description': 'Model version rollback events',
        'labels': ['model_type', 'rollback_reason', 'from_version', 'to_version'],
        'unit': 'count',
    }
}
```

## GCP Cloud Monitoring Integration

### Setup and Configuration

#### 1. Enable Required APIs
```bash
# Enable necessary GCP APIs
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable cloudtrace.googleapis.com
gcloud services enable clouderrorreporting.googleapis.com
```

#### 2. Service Account Setup
```bash
# Create monitoring service account
gcloud iam service-accounts create rlte-monitoring \
  --description="Service account for RLTE monitoring" \
  --display-name="RLTE Monitoring"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rlte-monitoring@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rlte-monitoring@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rlte-monitoring@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudtrace.agent"
```

#### 3. Deploy Monitoring Infrastructure
```bash
# Deploy transformer monitoring setup
./deploy/monitoring/setup_transformer_monitoring.py --project-id=$PROJECT_ID

# Create custom metrics
python -c "
from src.monitoring.gcp_transformer_monitoring import GCPTransformerMonitoring
monitor = GCPTransformerMonitoring('$PROJECT_ID')
monitor.setup_custom_metrics()
print('Custom metrics created successfully')
"

# Deploy dashboards
gcloud monitoring dashboards create \
  --config-from-file=monitoring/transformer-dashboard.json

# Configure alert policies
python deploy/monitoring/setup_alert_policies.py --project-id=$PROJECT_ID
```

### Custom Metrics Creation

#### 1. Transformer Metrics Setup
```python
# Custom metrics for transformer monitoring
from google.cloud import monitoring_v3

def create_transformer_metrics(project_id: str):
    """Create custom metrics for transformer monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # Memory usage metric
    memory_descriptor = monitoring_v3.MetricDescriptor(
        type="custom.googleapis.com/transformer/memory_usage",
        metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
        value_type=monitoring_v3.MetricDescriptor.ValueType.INT64,
        description="Memory usage by transformer models",
        display_name="Transformer Memory Usage",
        labels=[
            monitoring_v3.LabelDescriptor(
                key="model_type",
                value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                description="Type of transformer model"
            ),
            monitoring_v3.LabelDescriptor(
                key="instance_id",
                value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                description="Cloud Run instance ID"
            )
        ]
    )
    
    # Inference latency metric
    latency_descriptor = monitoring_v3.MetricDescriptor(
        type="custom.googleapis.com/transformer/inference_latency",
        metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
        value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
        description="Inference latency for transformer models",
        display_name="Transformer Inference Latency",
        unit="ms",
        labels=[
            monitoring_v3.LabelDescriptor(
                key="model_type",
                value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                description="Type of transformer model"
            ),
            monitoring_v3.LabelDescriptor(
                key="batch_size",
                value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                description="Batch size for inference"
            )
        ]
    )
    
    # Create metrics
    try:
        client.create_metric_descriptor(
            name=project_name, 
            metric_descriptor=memory_descriptor
        )
        client.create_metric_descriptor(
            name=project_name, 
            metric_descriptor=latency_descriptor
        )
        print("Custom transformer metrics created successfully")
    except Exception as e:
        if "already exists" in str(e):
            print("Metrics already exist, skipping creation")
        else:
            raise e
```

## Custom Metrics and Dashboards

### Transformer Performance Dashboard

#### 1. Dashboard Configuration
```json
{
  "displayName": "RLTE Transformer Performance",
  "description": "Comprehensive monitoring for transformer models",
  "dashboardFilters": [
    {
      "filterType": "RESOURCE_LABEL",
      "labelKey": "service_name",
      "stringValue": "shyvr-rlte"
    }
  ],
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Transformer Memory Usage",
          "scorecard": {
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"custom.googleapis.com/transformer/memory_usage\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_MEAN",
                  "crossSeriesReducer": "REDUCE_MAX",
                  "groupByFields": ["metric.label.model_type"]
                }
              }
            },
            "sparkChartView": {
              "sparkChartType": "SPARK_LINE"
            },
            "gaugeView": {
              "lowerBound": 0,
              "upperBound": 8589934592
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Inference Latency by Model",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/transformer/inference_latency\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN",
                      "groupByFields": ["metric.label.model_type"]
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      }
    ]
  }
}
```

#### 2. Resource Utilization Dashboard
```json
{
  "displayName": "RLTE Resource Utilization",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 12,
        "height": 4,
        "widget": {
          "title": "CPU Utilization by Model Type",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "STACKED_AREA"
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "Memory Utilization",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ]
          }
        }
      }
    ]
  }
}
```

### Business Intelligence Dashboard

#### 1. Trading Performance Metrics
```json
{
  "displayName": "RLTE Trading Performance",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Prediction Accuracy by Model",
          "scorecard": {
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"custom.googleapis.com/trading/prediction_accuracy\"",
                "aggregation": {
                  "alignmentPeriod": "3600s",
                  "perSeriesAligner": "ALIGN_MEAN",
                  "groupByFields": ["metric.label.model_type"]
                }
              }
            },
            "sparkChartView": {
              "sparkChartType": "SPARK_LINE"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Trading Performance Correlation",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/trading/sharpe_ratio\"",
                    "aggregation": {
                      "alignmentPeriod": "3600s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ]
          }
        }
      }
    ]
  }
}
```

## Alert Policies and Notification

### Critical Alert Policies

#### 1. High Memory Usage Alert
```yaml
# Alert for high transformer memory usage
display_name: "RLTE - High Transformer Memory Usage"
documentation:
  content: "Transformer model memory usage exceeds 90% of allocated 8GB"
  mime_type: "text/markdown"

conditions:
  - display_name: "Memory usage > 7GB"
    condition_threshold:
      filter: 'metric.type="custom.googleapis.com/transformer/memory_usage"'
      comparison: COMPARISON_GREATER_THAN
      threshold_value: 7516192768  # 7GB in bytes
      duration: 300s  # 5 minutes
      aggregations:
        - alignment_period: 60s
          per_series_aligner: ALIGN_MEAN
          cross_series_reducer: REDUCE_MAX

notification_channels:
  - "projects/PROJECT_ID/notificationChannels/EMAIL_CHANNEL_ID"
  - "projects/PROJECT_ID/notificationChannels/SLACK_CHANNEL_ID"

alert_strategy:
  auto_close: 86400s  # 24 hours
```

#### 2. Slow Inference Latency Alert
```yaml
display_name: "RLTE - Slow Transformer Inference"
documentation:
  content: "Transformer inference latency exceeds 1000ms threshold"

conditions:
  - display_name: "Inference latency > 1000ms"
    condition_threshold:
      filter: 'metric.type="custom.googleapis.com/transformer/inference_latency"'
      comparison: COMPARISON_GREATER_THAN
      threshold_value: 1000.0  # 1000ms
      duration: 180s  # 3 minutes
      aggregations:
        - alignment_period: 60s
          per_series_aligner: ALIGN_PERCENTILE_95

severity: CRITICAL
notification_channels:
  - "projects/PROJECT_ID/notificationChannels/PAGERDUTY_CHANNEL_ID"
  - "projects/PROJECT_ID/notificationChannels/SLACK_CHANNEL_ID"
```

#### 3. Model Health Alert
```yaml
display_name: "RLTE - Transformer Model Unhealthy"
documentation:
  content: "Transformer model health check failing"

conditions:
  - display_name: "Model health status unhealthy"
    condition_threshold:
      filter: 'metric.type="custom.googleapis.com/transformer/model_health"'
      comparison: COMPARISON_EQUAL
      threshold_value: 0  # 0 = unhealthy
      duration: 60s  # 1 minute

severity: CRITICAL
notification_channels:
  - "projects/PROJECT_ID/notificationChannels/PAGERDUTY_CHANNEL_ID"
  - "projects/PROJECT_ID/notificationChannels/EMAIL_CHANNEL_ID"
  - "projects/PROJECT_ID/notificationChannels/SLACK_CHANNEL_ID"
```

### Notification Channel Setup

#### 1. Email Notifications
```bash
# Create email notification channel
gcloud alpha monitoring channels create \
  --display-name="RLTE Operations Team" \
  --type=email \
  --channel-labels=email_address=ops@shyvr.ai
```

#### 2. Slack Integration
```bash
# Create Slack notification channel
gcloud alpha monitoring channels create \
  --display-name="RLTE Slack Alerts" \
  --type=slack \
  --channel-labels=channel_name=#rlte-alerts,url=SLACK_WEBHOOK_URL
```

#### 3. PagerDuty Integration
```bash
# Create PagerDuty notification channel
gcloud alpha monitoring channels create \
  --display-name="RLTE PagerDuty" \
  --type=pagerduty \
  --channel-labels=service_key=PAGERDUTY_SERVICE_KEY
```

## Health Check Endpoints

### Comprehensive Health Monitoring

#### 1. Main Health Endpoint
```python
# /health endpoint implementation
@app.get("/health")
async def health_check():
    """Comprehensive system health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "environment": config.environment,
        "checks": {}
    }
    
    # Database connectivity
    try:
        await database_health_check()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Transformer models health
    try:
        model_health = await transformer_health_check()
        health_status["checks"]["transformers"] = model_health
        if any(status != "healthy" for status in model_health.values()):
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["transformers"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # External APIs
    api_health = await api_health_check()
    health_status["checks"]["apis"] = api_health
    
    # Memory usage
    memory_usage = get_memory_usage()
    health_status["checks"]["memory_usage_mb"] = memory_usage
    if memory_usage > 7000:  # 7GB threshold
        health_status["status"] = "degraded"
    
    return health_status
```

#### 2. Transformer-Specific Health Checks
```python
# /health/transformers endpoint
@app.get("/health/transformers")
async def transformer_health_check():
    """Detailed transformer model health check."""
    from src.monitoring.transformer_health_endpoints import TransformerHealthEndpoints
    
    health_checker = TransformerHealthEndpoints()
    return await health_checker.get_comprehensive_health()

# Expected response
{
    "overall_status": "healthy",
    "models": {
        "itransformer": {
            "status": "healthy",
            "memory_usage_mb": 3840,
            "cache_hit_rate": 0.75,
            "avg_inference_latency_ms": 180,
            "last_prediction_time": "2025-08-06T10:00:00Z"
        },
        "patchtst": {
            "status": "healthy", 
            "memory_usage_mb": 2560,
            "cache_hit_rate": 0.80,
            "avg_inference_latency_ms": 120
        },
        "timesmixer": {
            "status": "degraded",
            "memory_usage_mb": 4800,
            "cache_hit_rate": 0.45,
            "avg_inference_latency_ms": 350,
            "warnings": ["cache hit rate below threshold"]
        },
        "timesfm": {
            "status": "healthy",
            "memory_usage_mb": 5120,
            "cache_hit_rate": 0.65,
            "avg_inference_latency_ms": 280
        }
    },
    "resource_summary": {
        "total_memory_usage_mb": 16320,
        "memory_limit_mb": 8192,
        "memory_utilization_percent": 81.6
    }
}
```

#### 3. Individual Model Health Endpoints
```python
# Individual model health checks
@app.get("/health/models/{model_type}")
async def individual_model_health(model_type: str):
    """Health check for individual transformer model."""
    
    model_health_map = {
        "itransformer": check_itransformer_health,
        "patchtst": check_patchtst_health,
        "timesmixer": check_timesmixer_health,
        "timesfm": check_timesfm_health
    }
    
    if model_type not in model_health_map:
        raise HTTPException(status_code=404, detail="Model not found")
    
    health_checker = model_health_map[model_type]
    return await health_checker()
```

### Health Check Automation

#### 1. Kubernetes Probes Configuration
```yaml
# Health check configuration for Cloud Run
apiVersion: serving.knative.dev/v1
kind: Service
spec:
  template:
    spec:
      containers:
      - image: us-central1-docker.pkg.dev/PROJECT_ID/shyvr-ai-prod/shyvr-rlte
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
```

#### 2. External Health Monitoring
```bash
# External monitoring with curl
#!/bin/bash
# scripts/health_monitor.sh

HEALTH_ENDPOINT="https://your-service-url/health"
EXPECTED_STATUS="healthy"
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

check_health() {
    local response
    response=$(curl -s -f "$HEALTH_ENDPOINT" || echo "ERROR")
    
    if [[ "$response" == "ERROR" ]]; then
        send_alert "Health endpoint unreachable"
        return 1
    fi
    
    local status
    status=$(echo "$response" | jq -r '.status')
    
    if [[ "$status" != "$EXPECTED_STATUS" ]]; then
        send_alert "System status: $status"
        return 1
    fi
    
    echo "✅ Health check passed: $status"
    return 0
}

send_alert() {
    local message=$1
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 RLTE Health Alert: $message\"}" \
        "$ALERT_WEBHOOK"
}

# Run health check
check_health
```

## Performance Monitoring

### System Performance Metrics

#### 1. Resource Utilization Tracking
```python
# Performance metrics collection
PERFORMANCE_METRICS = {
    # CPU utilization
    'system_cpu_utilization': {
        'description': 'Overall system CPU utilization',
        'threshold_warning': 80,    # 80% CPU usage
        'threshold_critical': 95,   # 95% CPU usage
    },
    
    # Memory utilization
    'system_memory_utilization': {
        'description': 'System memory utilization',
        'threshold_warning': 85,    # 85% memory usage
        'threshold_critical': 95,   # 95% memory usage
    },
    
    # Request throughput
    'request_throughput': {
        'description': 'Requests per second',
        'labels': ['endpoint', 'method', 'status_code'],
        'threshold_warning': 1000,  # High request volume
    },
    
    # Response time percentiles
    'response_time_percentiles': {
        'description': 'Response time distribution',
        'percentiles': [50, 75, 90, 95, 99],
        'threshold_warning': 500,   # P95 > 500ms
        'threshold_critical': 1000, # P95 > 1000ms
    }
}
```

#### 2. Performance Profiling Integration
```python
# Continuous profiling setup
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def performance_profiler(profile_name: str):
    """Context manager for performance profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        # Save profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Send to monitoring
        profile_data = {
            'profile_name': profile_name,
            'timestamp': datetime.utcnow().isoformat(),
            'top_functions': stats.get_stats_profile().func_profiles
        }
        
        # Send to Cloud Monitoring
        send_profile_metrics(profile_data)

# Usage in transformer inference
async def predict_with_profiling(model_type: str, data: np.ndarray):
    with performance_profiler(f"transformer_inference_{model_type}"):
        return await model.predict(data)
```

### Application Performance Monitoring (APM)

#### 1. Distributed Tracing Setup
```python
# Cloud Trace integration
from google.cloud import trace_v1
import opentelemetry
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Setup Cloud Trace exporter
cloud_trace_exporter = CloudTraceSpanExporter()
span_processor = BatchSpanProcessor(cloud_trace_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace transformer inference
async def traced_transformer_inference(model_type: str, input_data):
    with tracer.start_as_current_span(f"transformer_inference_{model_type}") as span:
        span.set_attribute("model_type", model_type)
        span.set_attribute("input_size", len(input_data))
        
        start_time = time.time()
        result = await transformer_predict(model_type, input_data)
        
        span.set_attribute("inference_duration_ms", 
                          (time.time() - start_time) * 1000)
        span.set_attribute("prediction_confidence", result.confidence)
        
        return result
```

#### 2. Custom Span Attributes
```python
# Detailed tracing for transformer operations
TRACE_ATTRIBUTES = {
    'transformer_inference': [
        'model_type',
        'model_version', 
        'batch_size',
        'sequence_length',
        'attention_heads',
        'inference_duration_ms',
        'memory_usage_bytes',
        'cache_hit_rate',
        'prediction_confidence'
    ],
    
    'model_lifecycle': [
        'operation_type',  # load, warm, activate, deactivate
        'model_type',
        'model_version',
        'operation_duration_ms',
        'success',
        'error_message'
    ],
    
    'resource_allocation': [
        'allocation_type',
        'current_memory_gb',
        'target_memory_gb',
        'current_cpu',
        'target_cpu',
        'allocation_reason',
        'allocation_duration_ms'
    ]
}
```

## Log Aggregation and Analysis

### Structured Logging Setup

#### 1. Application Logging Configuration
```python
# Structured logging with Cloud Logging
import structlog
from google.cloud import logging as cloud_logging

# Initialize Cloud Logging
logging_client = cloud_logging.Client()
logging_client.setup_logging()

# Configure structlog
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
```

#### 2. Transformer-Specific Logging
```python
# Specialized logging for transformer operations
class TransformerLogger:
    def __init__(self):
        self.logger = structlog.get_logger("transformer")
    
    def log_inference(self, model_type: str, metrics: dict):
        """Log transformer inference with metrics."""
        self.logger.info(
            "transformer_inference_completed",
            model_type=model_type,
            inference_latency_ms=metrics['latency'],
            memory_usage_bytes=metrics['memory'],
            cache_hit_rate=metrics['cache_hit_rate'],
            batch_size=metrics['batch_size'],
            prediction_confidence=metrics.get('confidence', 0.0)
        )
    
    def log_model_lifecycle(self, operation: str, model_type: str, 
                          success: bool, duration_ms: float):
        """Log model lifecycle operations."""
        self.logger.info(
            "model_lifecycle_operation",
            operation=operation,
            model_type=model_type,
            success=success,
            duration_ms=duration_ms,
            severity="INFO" if success else "WARNING"
        )
    
    def log_resource_allocation(self, allocation_event: dict):
        """Log resource allocation decisions."""
        self.logger.info(
            "resource_allocation_event",
            **allocation_event,
            component="dynamic_resource_allocator"
        )

# Usage
transformer_logger = TransformerLogger()
transformer_logger.log_inference("iTransformer", {
    "latency": 180.5,
    "memory": 3840000000,
    "cache_hit_rate": 0.75,
    "batch_size": 1
})
```

### Log-Based Metrics

#### 1. Error Rate Monitoring
```yaml
# Log-based metric for error tracking
metric_name: "error_rate_by_component"
filter: |
  severity >= ERROR AND 
  jsonPayload.component IS NOT NULL
label_extractors:
  component: "EXTRACT(jsonPayload.component)"
  error_type: "EXTRACT(jsonPayload.error_type)"
metric_descriptor:
  metric_kind: DELTA
  value_type: INT64
```

#### 2. Performance Insights from Logs
```yaml
# Latency distribution from logs
metric_name: "inference_latency_distribution"
filter: |
  jsonPayload.event_type = "transformer_inference_completed" AND
  jsonPayload.inference_latency_ms IS NOT NULL
label_extractors:
  model_type: "EXTRACT(jsonPayload.model_type)"
value_extractor: "EXTRACT(jsonPayload.inference_latency_ms)"
metric_descriptor:
  metric_kind: GAUGE
  value_type: DOUBLE
```

## Business Metrics Monitoring

### Trading Performance Metrics

#### 1. Model Accuracy Tracking
```python
# Business intelligence metrics
BUSINESS_METRICS = {
    # Prediction accuracy by model
    'prediction_accuracy_by_model': {
        'description': 'Prediction accuracy percentage by transformer model',
        'labels': ['model_type', 'time_horizon', 'market_condition'],
        'calculation': 'correct_predictions / total_predictions * 100',
        'target_threshold': 65,  # 65% accuracy target
    },
    
    # Trading performance correlation
    'trading_sharpe_ratio': {
        'description': 'Risk-adjusted returns (Sharpe ratio)',
        'labels': ['model_type', 'trading_strategy'],
        'calculation': '(portfolio_return - risk_free_rate) / portfolio_volatility',
        'target_threshold': 1.5,  # 1.5 Sharpe ratio target
    },
    
    # Model contribution to returns
    'model_contribution_to_returns': {
        'description': 'Individual model contribution to portfolio returns',
        'labels': ['model_type', 'time_period'],
        'unit': 'percentage',
    },
    
    # Risk metrics
    'portfolio_drawdown': {
        'description': 'Maximum drawdown from peak',
        'labels': ['strategy', 'time_period'],
        'threshold_warning': 10,  # 10% drawdown warning
        'threshold_critical': 20, # 20% drawdown critical
    }
}
```

#### 2. Model Performance Correlation
```python
# Correlation analysis between model performance and business outcomes
class BusinessMetricsCollector:
    def __init__(self):
        self.logger = structlog.get_logger("business_metrics")
        self.metrics_client = monitoring_v3.MetricServiceClient()
    
    async def track_prediction_accuracy(self, model_type: str, 
                                      predictions: List[dict], 
                                      actuals: List[dict]):
        """Track model prediction accuracy."""
        correct_predictions = sum(
            1 for pred, actual in zip(predictions, actuals)
            if abs(pred['value'] - actual['value']) < pred.get('tolerance', 0.05)
        )
        
        accuracy = correct_predictions / len(predictions) * 100
        
        # Send to monitoring
        series = monitoring_v3.TimeSeries(
            metric=monitoring_v3.Metric(
                type="custom.googleapis.com/trading/prediction_accuracy",
                labels={
                    "model_type": model_type,
                    "time_horizon": "1h",  # Configurable
                }
            ),
            resource=monitoring_v3.MonitoredResource(
                type="cloud_run_revision",
                labels={"service_name": "shyvr-rlte"}
            ),
            points=[monitoring_v3.Point(
                interval=monitoring_v3.TimeInterval(
                    end_time={"seconds": int(time.time())}
                ),
                value=monitoring_v3.TypedValue(double_value=accuracy)
            )]
        )
        
        self.metrics_client.create_time_series(
            name=f"projects/{PROJECT_ID}",
            time_series=[series]
        )
        
        self.logger.info(
            "prediction_accuracy_tracked",
            model_type=model_type,
            accuracy=accuracy,
            total_predictions=len(predictions)
        )
    
    async def track_trading_performance(self, model_type: str, 
                                     returns: List[float], 
                                     benchmark_returns: List[float]):
        """Track trading performance metrics."""
        import numpy as np
        
        # Calculate Sharpe ratio
        excess_returns = np.array(returns) - np.array(benchmark_returns)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Send metrics to monitoring
        await self.send_business_metric("trading/sharpe_ratio", sharpe_ratio, 
                                      {"model_type": model_type})
        await self.send_business_metric("trading/max_drawdown", abs(max_drawdown), 
                                      {"model_type": model_type})
        
        self.logger.info(
            "trading_performance_tracked",
            model_type=model_type,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_returns=sum(returns)
        )
```

## Incident Response Integration

### Automated Incident Response

#### 1. Alert Escalation Matrix
```yaml
escalation_matrix:
  level_1_alerts:
    - high_memory_usage
    - slow_inference_latency
    - low_cache_performance
    escalation_delay: 300  # 5 minutes
    notification_channels: ["email", "slack"]
  
  level_2_alerts:
    - model_health_failure
    - system_unavailable
    - critical_error_rate
    escalation_delay: 180  # 3 minutes
    notification_channels: ["email", "slack", "pagerduty"]
  
  level_3_alerts:
    - trading_system_failure
    - data_corruption
    - security_breach
    escalation_delay: 60   # 1 minute
    notification_channels: ["pagerduty", "phone", "executive_escalation"]
```

#### 2. Automated Recovery Procedures
```python
# Automated incident response
class IncidentResponseManager:
    def __init__(self):
        self.logger = structlog.get_logger("incident_response")
        self.recovery_procedures = {
            'high_memory_usage': self.handle_high_memory,
            'model_health_failure': self.handle_model_failure,
            'slow_inference': self.handle_slow_inference,
        }
    
    async def handle_high_memory(self, alert_data: dict):
        """Handle high memory usage incidents."""
        model_type = alert_data.get('model_type')
        
        self.logger.warning(
            "high_memory_incident_detected",
            model_type=model_type,
            memory_usage_mb=alert_data.get('memory_usage_mb')
        )
        
        # Step 1: Try garbage collection
        await self.trigger_garbage_collection()
        
        # Step 2: Clear model caches
        await self.clear_model_cache(model_type)
        
        # Step 3: If still high, restart the problematic model
        if await self.check_memory_usage() > 7000:
            await self.restart_model(model_type)
        
        # Step 4: If still high, scale up resources
        if await self.check_memory_usage() > 7000:
            await self.scale_up_resources()
        
        self.logger.info("high_memory_incident_handled")
    
    async def handle_model_failure(self, alert_data: dict):
        """Handle model health failure incidents."""
        model_type = alert_data.get('model_type')
        
        # Step 1: Try model restart
        restart_success = await self.restart_model(model_type)
        
        if not restart_success:
            # Step 2: Rollback to previous version
            await self.rollback_model_version(model_type)
        
        # Step 3: If still failing, disable model and use fallback
        if not await self.check_model_health(model_type):
            await self.enable_fallback_model(model_type)
        
        self.logger.info(
            "model_failure_incident_handled",
            model_type=model_type,
            fallback_enabled=True
        )
```

### Integration with External Tools

#### 1. PagerDuty Integration
```python
# PagerDuty incident creation
import httpx

class PagerDutyIntegration:
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.base_url = "https://events.pagerduty.com"
    
    async def create_incident(self, alert_data: dict):
        """Create PagerDuty incident."""
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"RLTE Alert: {alert_data['alert_name']}",
                "severity": alert_data.get('severity', 'warning'),
                "source": "shyvr-rlte",
                "component": alert_data.get('component', 'unknown'),
                "custom_details": alert_data
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v2/enqueue",
                json=payload
            )
            
        return response.status_code == 202
```

#### 2. Slack Integration
```python
# Slack notification system
class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_alert(self, alert_data: dict):
        """Send alert to Slack channel."""
        severity_colors = {
            'info': '#36a64f',      # Green
            'warning': '#ff9900',   # Orange
            'critical': '#ff0000',  # Red
        }
        
        color = severity_colors.get(alert_data.get('severity', 'info'))
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 RLTE Alert: {alert_data['alert_name']}",
                    "text": alert_data.get('description', ''),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert_data.get('severity', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Component", 
                            "value": alert_data.get('component', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_data.get('timestamp', ''),
                            "short": True
                        }
                    ],
                    "footer": "RLTE Monitoring System",
                    "ts": int(time.time())
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.webhook_url, json=payload)
        
        return response.status_code == 200
```

## Conclusion

This comprehensive monitoring and observability guide provides all necessary components for effectively monitoring the transformer-enabled RLTE system. The monitoring stack includes:

- **Comprehensive Metrics**: System, application, and business metrics
- **Advanced Alerting**: Multi-level alert policies with automated escalation
- **Health Monitoring**: Detailed health checks for all system components
- **Performance Tracking**: Resource utilization and optimization insights
- **Business Intelligence**: Trading performance correlation and analysis
- **Incident Response**: Automated recovery procedures and external integrations

### Key Benefits
1. **Proactive Monitoring**: Early detection of issues before they impact users
2. **Performance Optimization**: Data-driven insights for resource optimization
3. **Business Correlation**: Direct correlation between system performance and business outcomes
4. **Automated Response**: Reduced MTTR through automated incident response
5. **Comprehensive Observability**: Full visibility into system behavior

### Next Steps
1. Review the [Security and Access Control Guide](SECURITY_ACCESS_CONTROL_GUIDE.md)
2. Familiarize yourself with the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
3. Set up monitoring dashboards and alert policies
4. Test incident response procedures
5. Establish monitoring runbooks and documentation

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-06  
**Maintained By**: ML Development Team