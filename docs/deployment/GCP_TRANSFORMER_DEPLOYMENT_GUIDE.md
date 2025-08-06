# GCP Transformer-Enabled RLTE System Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [System Architecture](#system-architecture)
4. [Deployment Pipeline](#deployment-pipeline)
5. [Resource Configuration](#resource-configuration)
6. [Dynamic Resource Allocation](#dynamic-resource-allocation)
7. [Model Lifecycle Management](#model-lifecycle-management)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Blue-Green Deployment Strategy](#blue-green-deployment-strategy)
10. [Security Configuration](#security-configuration)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)

## Overview

This guide provides comprehensive instructions for deploying the transformer-enabled Shyvr RLTE (Reinforcement Learning Trading Engine) system on Google Cloud Platform (GCP) using Cloud Run. The system integrates state-of-the-art transformer models (iTransformer, PatchTST, TimesMixer, TimesFM) with ML and RL components for advanced cryptocurrency trading.

### Key Features
- **Transformer Model Support**: 4 advanced transformer architectures for time-series prediction
- **Dynamic Resource Allocation**: Automatic scaling based on model complexity and load
- **Model Lifecycle Management**: Hot-swapping, versioning, and rollback capabilities
- **Advanced Monitoring**: Comprehensive observability for transformer-specific metrics
- **Zero-Downtime Deployments**: Blue-green deployment strategy with health checks

## Prerequisites

### GCP Setup Requirements
```bash
# Required GCP Services
- Cloud Run (for container deployment)
- Cloud Build (for CI/CD pipeline)
- Container Registry/Artifact Registry (for Docker images)
- Cloud SQL (PostgreSQL for persistence)
- Cloud Monitoring (for observability)
- Cloud Logging (for log aggregation)
- Secret Manager (for secure configuration)
- Cloud Storage (for model persistence)

# Required GCP APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable storage-component.googleapis.com
```

### Local Development Tools
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install Docker
# Follow platform-specific Docker installation instructions

# Install UV for Python dependency management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up authentication
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Project Configuration
```bash
# Set project variables
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPOSITORY="shyvr-ai-prod"

# Configure gcloud
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
```

## System Architecture

### Transformer-Enabled Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    GCP Cloud Run                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  iTransformer   │  │    PatchTST     │  │ TimesMixer  │  │
│  │   Model         │  │     Model       │  │   Model     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │    TimesFM      │  │   LSTM Models   │  │ RL Agents   │  │
│  │    Model        │  │                 │  │   (DQN)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│           Dynamic Resource Allocator                       │
│      Model Lifecycle Manager | Monitoring                  │
└─────────────────────────────────────────────────────────────┘
             │                      │                      │
    ┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
    │  Cloud SQL      │    │ Cloud Storage   │    │ Cloud Monitoring│
    │ (PostgreSQL)    │    │ (Model Store)   │    │   & Logging     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components
1. **Transformer Models**: 4 specialized time-series prediction models
2. **Dynamic Resource Allocator**: Manages compute resources based on model complexity
3. **Model Lifecycle Manager**: Handles model loading, hot-swapping, and versioning
4. **ML-RL Bridge**: Integrates transformer predictions with reinforcement learning
5. **Monitoring System**: Comprehensive observability and alerting

## Deployment Pipeline

### Automated CI/CD Pipeline

The system uses Google Cloud Build for automated deployments with the following stages:

#### 1. Build Configuration (`config/cloudbuild.yaml`)
```yaml
# Key configuration highlights
options:
  machineType: 'E2_HIGHCPU_16'    # High-CPU for transformer builds
  diskSizeGb: 150                  # Large disk for model caching
timeout: '3600s'                   # 1-hour timeout for transformer builds
```

#### 2. Deployment Stages
```bash
# Stage 1: Code Quality & Security
- Black formatting checks
- Ruff linting
- MyPy type checking
- Trivy security scanning

# Stage 2: Testing
- Unit tests with coverage
- Integration tests
- Performance benchmarks

# Stage 3: Docker Build
- Multi-stage transformer-optimized build
- Model pre-caching
- Security scanning

# Stage 4: Deployment
- Staging deployment (develop branch)
- Production deployment (main branch)
- Blue-green deployment strategy
```

#### 3. Resource-Optimized Configuration

**Staging Environment**:
```bash
Memory: 6Gi
CPU: 4 cores
Concurrency: 20
Timeout: 2700s (45 minutes)
Max Instances: 5
```

**Production Environment**:
```bash
Memory: 8Gi
CPU: 6 cores
Concurrency: 15
Timeout: 4200s (70 minutes)
Max Instances: 10
Min Instances: 1
```

## Resource Configuration

### Transformer-Specific Settings

#### Environment Variables for Optimization
```bash
# Core transformer settings
TRANSFORMER_OPTIMIZED=true
TRANSFORMER_BATCH_SIZE=1
TRANSFORMER_MAX_LENGTH=512
TORCH_COMPILE_MODE=reduce-overhead
TRANSFORMERS_CACHE=/app/models/cache
TOKENIZERS_PARALLELISM=false

# CPU optimization
OMP_NUM_THREADS=6
MKL_NUM_THREADS=6
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
TORCH_INFERENCE_MODE=1
```

#### Resource Profiles by Model Type
```python
# Model-specific resource requirements
TRANSFORMER_RESOURCE_PROFILES = {
    'iTransformer': {
        'base_memory_gb': 4,
        'base_cpu': 2,
        'scaling_factor': 1.2,
        'complexity_factor': 1.0
    },
    'PatchTST': {
        'base_memory_gb': 3,
        'base_cpu': 2, 
        'scaling_factor': 1.0,
        'complexity_factor': 0.8
    },
    'TimesMixer': {
        'base_memory_gb': 5,
        'base_cpu': 3,
        'scaling_factor': 1.5,
        'complexity_factor': 1.5
    },
    'TimesFM': {
        'base_memory_gb': 6,
        'base_cpu': 4,
        'scaling_factor': 2.0,
        'complexity_factor': 2.0
    }
}
```

## Dynamic Resource Allocation

### Overview
The Dynamic Resource Allocator automatically adjusts compute resources based on:
- Model complexity and type
- Current system load
- Memory pressure
- Cost optimization preferences

### Key Components

#### 1. DynamicResourceAllocator (`src/deploy/dynamic_resource_allocator.py`)
```python
# Core functionality
- Real-time resource monitoring
- Load-based scaling decisions
- Cost-aware resource allocation  
- Multi-model resource balancing
- Memory pressure detection
```

#### 2. TransformerResourceManager
```python
# Model-specific optimizations
- Resource profile management
- Performance monitoring
- Bottleneck detection
- Usage analytics
```

#### 3. Auto-Scaling Policies
```python
SCALING_POLICIES = {
    'cpu_threshold_scale_up': 80,      # Scale up at 80% CPU
    'cpu_threshold_scale_down': 30,    # Scale down at 30% CPU
    'memory_threshold_scale_up': 85,   # Scale up at 85% memory
    'memory_threshold_scale_down': 40, # Scale down at 40% memory
    'latency_threshold_ms': 100,       # Scale up if latency > 100ms
    'cooldown_scale_up_seconds': 180,  # 3-minute cooldown
    'cooldown_scale_down_seconds': 300 # 5-minute cooldown
}
```

### Deployment Commands

#### Manual Resource Allocation
```bash
# Update Cloud Run service with specific resources
gcloud run deploy shyvr-rlte \
  --memory=8Gi \
  --cpu=6 \
  --concurrency=15 \
  --timeout=4200 \
  --max-instances=10 \
  --min-instances=1 \
  --region=us-central1
```

#### Dynamic Scaling Configuration
```bash
# Enable auto-scaling
gcloud run services update shyvr-rlte \
  --min-instances=1 \
  --max-instances=10 \
  --cpu-boost \
  --region=us-central1
```

## Model Lifecycle Management

### Overview
The Model Lifecycle Manager provides hot-swapping capabilities for zero-downtime model updates.

### Key Features
1. **Hot-Swapping**: Replace models without service interruption
2. **Version Control**: Semantic versioning with rollback capabilities
3. **Health Monitoring**: Continuous model performance tracking
4. **Traffic Migration**: Gradual traffic shifting for safe deployments

### Model Lifecycle States
```python
class ModelLifecycleState:
    LOADING = "loading"      # Model being loaded
    WARMING = "warming"      # Model warming up
    ACTIVE = "active"        # Model serving traffic
    DRAINING = "draining"    # Model being phased out
    INACTIVE = "inactive"    # Model not serving traffic
    FAILED = "failed"        # Model in error state
```

### Hot-Swap Process
```bash
# 1. Load new model version
POST /api/models/{model_id}/load
{
    "version": "2.1.0",
    "strategy": "blue_green"
}

# 2. Warm up new model
POST /api/models/{model_id}/warmup
{
    "samples": 100,
    "timeout_seconds": 300
}

# 3. Activate new model
POST /api/models/{model_id}/activate
{
    "traffic_percentage": 10  # Start with 10% traffic
}

# 4. Gradual traffic migration
POST /api/models/{model_id}/migrate-traffic
{
    "target_percentage": 100,
    "migration_steps": [10, 25, 50, 75, 100],
    "step_duration_seconds": 300
}
```

### Model Versioning
```python
# Semantic versioning format: MAJOR.MINOR.PATCH
# Example: 2.1.3
# - MAJOR: Breaking changes to model architecture
# - MINOR: New features or significant improvements  
# - PATCH: Bug fixes and minor improvements

# Compatibility matrix
COMPATIBILITY_RULES = {
    "major_version_compatibility": False,  # Major versions incompatible
    "minor_version_compatibility": True,   # Minor versions compatible
    "patch_version_compatibility": True,   # Patch versions compatible
    "rollback_generations": 3              # Keep 3 versions for rollback
}
```

## Monitoring and Observability

### Transformer-Specific Metrics

#### 1. Model Performance Metrics
```yaml
# Memory usage monitoring
transformer_memory_usage_bytes:
  description: "Memory consumption by transformer model"
  labels: [model_type, model_version, instance_id]
  threshold: 7372800000  # 7GB (90% of 8GB)

# Inference latency tracking  
transformer_inference_latency_ms:
  description: "Time taken for model inference"
  labels: [model_type, batch_size]
  threshold: 1000  # 1 second

# Cache performance
transformer_cache_hit_rate:
  description: "Model cache hit rate percentage"
  labels: [model_type, cache_type]
  threshold: 50  # Minimum 50% hit rate

# Model health status
transformer_model_health:
  description: "Overall model health status"
  labels: [model_type, health_check_type]
  values: [0=unhealthy, 1=healthy]
```

#### 2. Alert Policies
```yaml
# High memory usage alert
high_memory_usage:
  condition: transformer_memory_usage_bytes > 7372800000
  duration: 300s  # 5 minutes
  severity: WARNING
  notification_channels: ["email", "slack"]

# Slow inference alert  
slow_inference:
  condition: transformer_inference_latency_ms > 1000
  duration: 180s  # 3 minutes
  severity: CRITICAL
  notification_channels: ["email", "slack", "pagerduty"]

# Low cache hit rate
low_cache_performance:
  condition: transformer_cache_hit_rate < 50
  duration: 600s  # 10 minutes
  severity: WARNING
  notification_channels: ["email"]

# Model health failure
model_unhealthy:
  condition: transformer_model_health == 0
  duration: 60s   # 1 minute
  severity: CRITICAL
  notification_channels: ["email", "slack", "pagerduty"]
```

#### 3. Custom Dashboards

The system includes comprehensive monitoring dashboards:

**Model Health Overview**:
- Model status by type
- Memory usage trends
- Inference latency percentiles
- Cache performance metrics

**Resource Utilization**:
- CPU usage by model
- Memory pressure indicators
- Auto-scaling events
- Cost analysis

**Performance Analytics**:
- Prediction accuracy trends
- Model comparison metrics
- Feature importance analysis
- Trading performance correlation

### Monitoring Setup Commands

#### 1. Deploy Monitoring Infrastructure
```bash
# Create custom metrics
./deploy/monitoring/setup_transformer_monitoring.py

# Deploy dashboards
gcloud monitoring dashboards create --config-from-file=monitoring/transformer-dashboard.json

# Configure alert policies
gcloud alpha monitoring policies create --policy-from-file=monitoring/alert-policies.yaml
```

#### 2. Health Check Endpoints
```bash
# Overall system health
curl https://your-service-url/health

# Transformer-specific health
curl https://your-service-url/health/transformers

# Individual model health
curl https://your-service-url/health/models/itransformer
curl https://your-service-url/health/models/patchtst
curl https://your-service-url/health/models/timesmixer
curl https://your-service-url/health/models/timesfm
```

## Blue-Green Deployment Strategy

### Overview
The system implements a sophisticated blue-green deployment strategy with gradual traffic migration and automatic rollback capabilities.

### Deployment Flow

#### 1. Staging Deployment (Develop Branch)
```bash
# Automatic deployment on develop branch push
# Resource allocation: 6Gi RAM, 4 CPU, 20 concurrency

# Traffic migration strategy
1. Deploy new version with 0% traffic
2. Run health checks for 60 seconds
3. Migrate 50% traffic to new version
4. Monitor for 60 seconds
5. If healthy: migrate 100% traffic
6. If unhealthy: rollback to previous version
```

#### 2. Production Deployment (Main Branch)
```bash
# Conservative production deployment
# Resource allocation: 8Gi RAM, 6 CPU, 15 concurrency

# Traffic migration strategy
1. Deploy new version with 0% traffic
2. Run comprehensive health checks (2 minutes)
3. Migrate 10% traffic to new version (conservative start)
4. Monitor for 2 minutes with extended validation
5. If healthy: migrate to 50% traffic
6. Monitor for 2 minutes
7. If healthy: migrate to 100% traffic
8. If unhealthy at any stage: immediate rollback
```

### Manual Deployment Commands

#### 1. Run Blue-Green Deployment
```bash
# Deploy to staging
./deploy/blue_green_deployment.sh staging latest

# Deploy to production
./deploy/blue_green_deployment.sh production v2.1.0
```

#### 2. Manual Traffic Control
```bash
# Gradual traffic migration
gcloud run services update-traffic shyvr-rlte \
  --to-revisions=new-revision=25,old-revision=75 \
  --region=us-central1

# Complete migration
gcloud run services update-traffic shyvr-rlte \
  --to-revisions=new-revision=100 \
  --region=us-central1

# Emergency rollback
gcloud run services update-traffic shyvr-rlte \
  --to-revisions=old-revision=100 \
  --region=us-central1
```

### Health Check Validation
```bash
# Comprehensive health check script
HEALTH_CHECKS=(
  "/health"                    # Overall system health
  "/health/transformers"       # Transformer models health  
  "/health/database"          # Database connectivity
  "/health/models/ready"      # Model readiness check
  "/metrics"                  # Prometheus metrics endpoint
)

# Extended validation for production
VALIDATION_ITERATIONS=10
VALIDATION_INTERVAL=15  # seconds
ERROR_THRESHOLD=2       # Allow 2 failures out of 10 checks
```

## Security Configuration

### Access Control

#### 1. Service Account Setup
```bash
# Create service account for Cloud Run
gcloud iam service-accounts create shyvr-rlte-service \
  --description="Service account for RLTE system" \
  --display-name="RLTE Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:shyvr-rlte-service@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:shyvr-rlte-service@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:shyvr-rlte-service@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### 2. Secret Management
```bash
# Store secrets in Secret Manager
gcloud secrets create TELEGRAM_TOKEN --data-file=telegram-token.txt
gcloud secrets create DB_PASSWORD --data-file=db-password.txt
gcloud secrets create API_KEYS --data-file=api-keys.json

# Grant access to service account
gcloud secrets add-iam-policy-binding TELEGRAM_TOKEN \
  --member="serviceAccount:shyvr-rlte-service@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### 3. Network Security
```bash
# Configure VPC connector for secure database access
gcloud compute networks create rlte-vpc --subnet-mode=regional

gcloud compute networks subnets create rlte-subnet \
  --network=rlte-vpc \
  --range=10.0.0.0/24 \
  --region=us-central1

# Create VPC connector
gcloud compute networks vpc-access connectors create rlte-connector \
  --region=us-central1 \
  --subnet=rlte-subnet \
  --subnet-project=$PROJECT_ID \
  --min-instances=2 \
  --max-instances=10
```

### Environment-Specific Security

#### Production Security Measures
```yaml
security_config:
  # Authentication
  jwt_expiry_hours: 12        # Shorter token lifetime
  api_key_rotation: weekly    # Regular key rotation
  
  # Rate limiting  
  rate_limit_per_minute: 60   # Conservative rate limits
  burst_capacity: 10          # Limited burst capacity
  
  # Logging
  audit_logging: enabled      # Full audit trail
  sensitive_data_masking: enabled
  log_retention_days: 90      # Extended retention
  
  # Network
  allowed_origins: ["https://app.shyvr.ai"]
  cors_enabled: false         # Disable in production
  
  # Container security
  run_as_non_root: true      # Non-root execution
  read_only_root_fs: true    # Read-only filesystem
  drop_capabilities: ["ALL"] # Drop all capabilities
```

#### Secret Rotation Schedule
```bash
# Automated secret rotation (monthly)
# Implemented via Cloud Scheduler + Cloud Functions

# API Keys: Monthly rotation
# Database passwords: Quarterly rotation  
# JWT secrets: Weekly rotation
# TLS certificates: Annual rotation
```

## Performance Optimization

### Transformer Model Optimizations

#### 1. Docker Build Optimizations
```dockerfile
# Multi-stage build with model pre-caching
FROM python:3.12-slim as transformer-builder

# CPU-optimized PyTorch installation
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch>=2.7.1+cpu

# Flash Attention with fallback
RUN (pip install flash-attn --no-build-isolation || \
     echo "Flash Attention fallback: using xformers") && \
    pip install xformers

# Model quantization support
RUN pip install bitsandbytes optimum onnx onnxruntime

# Pre-cache transformer models
RUN python -c "
from transformers import AutoTokenizer, AutoModel;
tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1');
model = AutoModel.from_pretrained('distilbert-base-uncased');
print('Models pre-cached successfully')
"
```

#### 2. Runtime Optimizations
```python
# Transformer inference optimizations
OPTIMIZATION_CONFIG = {
    # Memory management
    'low_cpu_mem_usage': True,
    'torch_dtype': 'float16',           # Half precision
    'device_map': 'cpu',                # CPU-only deployment
    
    # Attention optimizations  
    'use_flash_attention': True,        # If available
    'attention_dropout': 0.0,           # Disable during inference
    
    # Compilation
    'torch_compile': True,              # Enable torch.compile
    'compile_mode': 'reduce-overhead',  # Optimization mode
    
    # Batching
    'batch_size': 1,                    # Single requests
    'max_length': 512,                  # Reasonable sequence length
    
    # Caching
    'use_cache': True,                  # Enable KV cache
    'cache_implementation': 'static',    # Static cache
}
```

#### 3. Load Balancing Strategy
```yaml
# Cloud Run load balancing configuration
load_balancing:
  # CPU allocation strategy
  cpu_allocation: "1000m per instance"  # 1 CPU per instance
  memory_allocation: "2Gi per instance" # 2GB per instance
  
  # Concurrency settings
  max_concurrent_requests: 15           # Conservative concurrency
  target_concurrent_requests: 10        # Target utilization
  
  # Auto-scaling
  min_instances: 1                      # Always warm
  max_instances: 10                     # Scale limit
  scale_up_cooldown: "3m"              # Scaling cooldown
  scale_down_cooldown: "5m"            # Conservative scale-down
```

### Cost Optimization Strategies

#### 1. Resource Right-Sizing
```python
# Cost-aware resource allocation
COST_OPTIMIZATION = {
    # CPU preferences
    'prefer_cpu_over_memory': True,     # CPU is more cost-effective
    'cpu_utilization_target': 75,       # Target 75% CPU utilization
    
    # Memory management
    'memory_utilization_target': 85,    # Target 85% memory utilization
    'enable_swap': False,               # Disable swap for predictability
    
    # Instance management
    'preemptible_instances': False,     # Use regular instances for stability
    'committed_use_discounts': True,    # Consider CUD for predictable workloads
    
    # Traffic optimization
    'connection_pooling': True,         # Reduce connection overhead
    'request_batching': True,           # Batch requests when possible
}
```

#### 2. Storage Cost Optimization
```bash
# Model storage optimization
gsutil lifecycle set lifecycle-config.json gs://your-model-bucket

# lifecycle-config.json
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 30}
    },
    {
      "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
      "condition": {"age": 90}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 365}
    }
  ]
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Symptoms: OOM kills, high memory usage
# Solutions:
1. Check memory usage: kubectl top pods
2. Analyze memory leaks: Enable memory profiling
3. Optimize model loading: Use model sharding
4. Increase memory allocation: Update Cloud Run config

# Emergency mitigation
gcloud run services update shyvr-rlte \
  --memory=12Gi \
  --region=us-central1
```

#### 2. Model Loading Failures
```bash
# Symptoms: Model fails to load, timeout errors
# Solutions:
1. Check model cache: Verify /app/models/cache
2. Increase timeout: Update Cloud Run timeout
3. Verify model integrity: Check model checksums
4. Review model size: Consider model quantization

# Debug model loading
curl https://your-service-url/debug/models/status
```

#### 3. Performance Degradation
```bash
# Symptoms: High latency, low throughput
# Solutions:
1. Monitor metrics: Check Cloud Monitoring dashboards
2. Analyze bottlenecks: Use Cloud Profiler
3. Optimize attention: Enable Flash Attention
4. Scale resources: Increase CPU/memory

# Performance analysis
curl https://your-service-url/debug/performance/profile
```

#### 4. Deployment Failures
```bash
# Symptoms: Build failures, deployment errors
# Solutions:
1. Check build logs: gcloud builds log <BUILD_ID>
2. Verify Docker build: Test locally
3. Check resource limits: Review Cloud Run quotas
4. Validate secrets: Ensure all secrets exist

# Manual deployment debug
gcloud run deploy shyvr-rlte-debug \
  --image=us-central1-docker.pkg.dev/$PROJECT_ID/shyvr-ai-prod/shyvr-rlte:debug \
  --region=us-central1 \
  --memory=8Gi \
  --cpu=6
```

### Monitoring Commands for Troubleshooting

#### 1. System Health Checks
```bash
# Overall system status
curl https://your-service-url/health | jq .

# Model-specific health
curl https://your-service-url/health/transformers | jq .

# Resource utilization
curl https://your-service-url/metrics | grep -E "(memory|cpu|latency)"
```

#### 2. Log Analysis
```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=shyvr-rlte" \
  --location=us-central1

# Error logs
gcloud logging read "resource.type=cloud_run_revision AND severity=ERROR" \
  --limit=50 \
  --format=json
```

#### 3. Performance Metrics
```bash
# Query monitoring metrics
gcloud monitoring metrics list --filter="transformer"

# Get specific metric data
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/transformer/memory_usage"' \
  --interval-end-time=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --interval-start-time=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)
```

### Emergency Procedures

#### 1. Emergency Rollback
```bash
# Immediate rollback to previous version
./deploy/emergency_rollback.sh production

# Manual rollback
PREVIOUS_REVISION=$(gcloud run revisions list \
  --service=shyvr-rlte \
  --region=us-central1 \
  --format="value(metadata.name)" \
  --limit=2 | tail -1)

gcloud run services update-traffic shyvr-rlte \
  --to-revisions=$PREVIOUS_REVISION=100 \
  --region=us-central1
```

#### 2. Emergency Scaling
```bash
# Scale up immediately
gcloud run services update shyvr-rlte \
  --max-instances=20 \
  --region=us-central1

# Scale down to save costs
gcloud run services update shyvr-rlte \
  --max-instances=5 \
  --min-instances=0 \
  --region=us-central1
```

#### 3. Circuit Breaker Activation
```bash
# Activate emergency circuit breaker
curl -X POST https://your-service-url/admin/circuit-breaker/activate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "high_error_rate", "duration": 300}'

# Deactivate circuit breaker
curl -X POST https://your-service-url/admin/circuit-breaker/deactivate \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Conclusion

This comprehensive deployment guide provides all necessary information for deploying and managing the transformer-enabled RLTE system on GCP. The system is designed for production-scale cryptocurrency trading with advanced ML/RL capabilities, comprehensive monitoring, and robust operational procedures.

For additional support and advanced configurations, refer to the specific component documentation in the `/docs/deployment/` directory.

### Next Steps
1. Review the [Configuration Management Guide](CONFIG_MANAGEMENT_GUIDE.md)
2. Set up [Monitoring and Observability](MONITORING_OBSERVABILITY_GUIDE.md)
3. Configure [Security and Access Control](SECURITY_ACCESS_CONTROL_GUIDE.md)
4. Familiarize yourself with the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

### Support Resources
- **Documentation**: `/docs/deployment/`
- **Monitoring Dashboards**: Cloud Monitoring Console
- **Logs**: Cloud Logging Console  
- **Metrics**: Custom transformer metrics in Cloud Monitoring
- **Alerts**: Configured alert policies for proactive monitoring

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-06  
**Maintained By**: ML Development Team