# Production Configuration Templates
# Transformer-Enabled RLTE Trading System

This directory contains comprehensive production configuration templates for deploying the transformer-enabled RLTE trading system on GCP Cloud Run.

## Overview

These templates are based on actual implementations and tested configurations from the RLTE system codebase. They provide production-ready configurations with transformer-specific optimizations, auto-scaling, monitoring, and security features.

## Template Structure

```
config/templates/
├── production/
│   ├── cloud_run_service.yaml          # Production Cloud Run service configuration
│   ├── env.production.yaml             # Production environment variables
│   ├── secret_management.yaml          # GCP Secret Manager configuration
│   ├── autoscaling_resource_allocation.yaml  # Auto-scaling and resource policies
│   └── monitoring_alerting.yaml        # Monitoring and alerting configuration
├── staging/
│   ├── cloud_run_service.yaml          # Staging Cloud Run service configuration
│   └── env.staging.yaml                # Staging environment variables
└── README.md                           # This file
```

## Key Features

### 🚀 **Production-Ready Configurations**
- **Transformer-Optimized Resources**: Based on actual resource profiles from `DynamicResourceAllocator`
- **Production Security**: GCP Secret Manager integration with automated rotation
- **Auto-Scaling Policies**: Model-aware scaling based on actual performance characteristics
- **Comprehensive Monitoring**: Transformer-specific metrics and alerting rules

### 🔧 **Actual Implementation Based**
All configurations are derived from actual implementations in the codebase:
- Resource allocations from `src/deploy/dynamic_resource_allocator.py`
- Monitoring metrics from `src/monitoring/transformer_*`
- Model configurations from `src/ml_analysis/config.py`
- Alert thresholds from tested performance requirements

### 🏗️ **Multi-Environment Support**
- **Production**: 8Gi RAM, 6 CPU, optimized for high availability and performance
- **Staging**: 6Gi RAM, 4 CPU, optimized for testing and cost efficiency

## Template Details

### 1. Cloud Run Service Configuration

#### Production (`production/cloud_run_service.yaml`)
```yaml
# Resource allocation
memory: "8Gi"
cpu: "6"
concurrency: 15  # Optimized for transformer workloads
timeout: 4200s   # 70 minutes for transformer operations

# Auto-scaling
min_instances: 2
max_instances: 100
target_cpu_utilization: 70%
target_memory_utilization: 80%
```

#### Key Features:
- **Transformer-Optimized Resources**: Based on actual model profiles
- **Extended Timeouts**: Account for transformer model loading (120s startup)
- **Blue-Green Deployment**: Zero-downtime deployment support
- **Health Checks**: Comprehensive startup, liveness, and readiness probes

### 2. Environment Variables

#### Production (`production/env.production.yaml`)
Comprehensive environment configuration with 100+ variables covering:

- **Application Settings**: Environment, logging, security
- **Transformer Models**: All 4 model types (iTransformer, PatchTST, TimesMixer, TimesFM)
- **PyTorch Optimization**: CPU-optimized settings for Cloud Run
- **Resource Allocation**: Dynamic scaling and model-specific profiles
- **API Integration**: External API keys and rate limits
- **Monitoring**: Metrics, alerts, and health check configurations

#### Key Sections:
```yaml
# Transformer model architecture settings
TRANSFORMER_D_MODEL=128
TRANSFORMER_NHEAD=8
ITRANSFORMER_INVERT_TIME_ATTENTION=true
PATCHTST_PATCH_LEN=16
TIMESMIXER_DOWN_SAMPLING_LAYERS=3

# PyTorch optimization
TORCH_COMPILE_MODE=reduce-overhead
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
OMP_NUM_THREADS=6

# Dynamic resource allocation
DYNAMIC_RESOURCE_ALLOCATION_ENABLED=true
AUTO_SCALING_ENABLED=true
```

### 3. Secret Management

#### Production (`production/secret_management.yaml`)
Complete GCP Secret Manager configuration including:

- **Database Credentials**: Production PostgreSQL with automated rotation
- **API Keys**: External service keys with security policies
- **Trading Secrets**: High-security trading system authentication
- **Access Control**: RBAC policies and audit logging
- **Compliance**: DLP inspection and compliance labeling

#### Key Features:
```yaml
# Automated rotation
rotation_period: "2592000s"  # 30 days for DB credentials
rotation_period: "1296000s"  # 15 days for trading secrets

# Customer-managed encryption
customer_managed_encryption:
  kms_key_name: "projects/${GCP_PROJECT_ID}/locations/global/keyRings/shyvr-rlte-production/cryptoKeys/secrets-key"

# Access control
- role: "roles/secretmanager.secretAccessor"
  members:
  - "serviceAccount:shyvr-rlte-production@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
```

### 4. Auto-Scaling and Resource Allocation

#### Production (`production/autoscaling_resource_allocation.yaml`)
Comprehensive scaling configuration based on actual model characteristics:

#### Model Resource Profiles:
```yaml
# iTransformer Profile
itransformer:
  memory_requirements:
    base: "4Gi"
    max: "12Gi"
  cpu_requirements:
    base: 2
    max: 8
  attention_complexity: 1.2
  scaling_policies:
    scale_up_threshold: 0.8
    cooldown_period_up: "180s"

# TimesFM Profile (largest model)
timesfm:
  memory_requirements:
    base: "6Gi"
    max: "20Gi"
  cpu_requirements:
    base: 4
    max: 12
  attention_complexity: 2.0
```

#### Auto-Scaling Policies:
- **CPU Scaling**: 80% scale up, 30% scale down
- **Memory Scaling**: 85% scale up, 40% scale down  
- **Latency-Based**: Scale up at 100ms P95 latency
- **Cost Optimization**: Peak hours scaling with budget constraints

### 5. Monitoring and Alerting

#### Production (`production/monitoring_alerting.yaml`)
Comprehensive monitoring based on actual implementation:

#### Custom Metrics (13 total):
- **Model Health**: Binary health status per model type
- **Memory Usage**: Current usage with 8Gi limit tracking
- **Inference Latency**: P95/P99 percentiles with thresholds
- **Cache Performance**: Hit rates with 50% minimum threshold
- **Attention Patterns**: Entropy for drift detection
- **Drift Detection**: Model drift scores with ML alerting

#### Alert Policies (6 critical alerts):
```yaml
# Critical memory usage (>90% of 8Gi)
threshold_value: 7372.8  # MB
duration: "300s"

# Slow inference (>1000ms P99)
threshold_value: 1000    # milliseconds
duration: "180s"

# Model health failure
threshold_value: 0       # Binary health check
duration: "60s"
```

#### Dashboards:
- **Transformer Overview**: Main production monitoring dashboard
- **Model Performance**: Detailed performance analysis and diagnostics
- **Resource Utilization**: Model-specific resource tracking

## Usage Instructions

### 1. Prerequisites

- GCP Project with Cloud Run, Secret Manager, and Cloud Monitoring APIs enabled
- Service account with appropriate permissions
- Domain and SSL certificates for production deployment

### 2. Deployment Steps

#### Step 1: Environment Setup
```bash
# Copy and customize environment variables
cp config/templates/production/env.production.yaml .env.production

# Replace placeholder values with actual secrets
# NEVER commit .env.production to version control
```

#### Step 2: Secret Management Setup
```bash
# Apply secret management configuration
gcloud deployment-manager deployments create shyvr-secrets \
  --config config/templates/production/secret_management.yaml

# Upload actual secret values to Secret Manager
gcloud secrets versions add database-credentials --data-file=db-credentials.json
gcloud secrets versions add api-keys --data-file=api-keys.json
```

#### Step 3: Deploy Cloud Run Service
```bash
# Build and deploy using the service configuration
gcloud run services replace config/templates/production/cloud_run_service.yaml \
  --region=us-central1

# Configure traffic routing
gcloud run services update-traffic shyvr-rlte-production \
  --to-latest --region=us-central1
```

#### Step 4: Setup Monitoring
```bash
# Deploy monitoring configuration
python deploy/monitoring/setup_transformer_monitoring.py \
  --config config/templates/production/monitoring_alerting.yaml
```

### 3. Configuration Customization

#### Environment-Specific Values
Replace these placeholders in your actual deployment:

```yaml
# Project and Infrastructure
GCP_PROJECT_ID: "your-gcp-project-id"
GCS_BUCKET: "your-model-storage-bucket"  
CLOUD_SQL_CONNECTION_NAME: "your-project:region:instance"

# Security (use strong, unique values)
SECRET_KEY: "generate-strong-32-char-secret"
DB_PASSWORD: "generate-strong-database-password"
JWT_SECRET: "generate-strong-jwt-secret"

# API Keys (obtain from respective services)
OPENAI_API_KEY: "your-actual-openai-key"
HELIUS_API_KEY: "your-actual-helius-key"
ETHERSCAN_API_KEY: "your-actual-etherscan-key"

# Monitoring and Alerts
ALERT_EMAIL_CRITICAL: "alerts@your-domain.com"
SLACK_WEBHOOK_URL: "your-slack-webhook-url"
```

#### Resource Tuning
Adjust resource allocations based on your specific requirements:

```yaml
# For higher throughput
memory: "12Gi"
cpu: "8"
max_instances: 150

# For cost optimization
memory: "6Gi" 
cpu: "4"
max_instances: 50
```

## Monitoring and Operations

### Health Checks
The system includes comprehensive health checks:

- **Startup Probe**: `/health/startup` - Model loading validation
- **Liveness Probe**: `/health/live` - Container health
- **Readiness Probe**: `/health/ready` - Traffic routing readiness
- **Transformer Health**: `/health/transformer` - Model-specific health

### Key Metrics to Monitor

#### Performance Metrics:
- **Inference Latency**: Target <100ms P95, Alert >1000ms P99
- **Memory Usage**: Target <80%, Alert >90% of allocated memory
- **Cache Hit Rate**: Target >70%, Alert <50%
- **Model Health**: Target 100%, Alert on any unhealthy models

#### Business Metrics:
- **Prediction Accuracy**: Monitor model drift and performance degradation
- **Trading Performance**: Correlation with model predictions
- **System Availability**: Target 99.9% uptime

### Troubleshooting

#### Common Issues:

1. **High Memory Usage**
   - Check model loading patterns
   - Verify attention mechanism efficiency
   - Review cache performance
   - Consider memory optimization settings

2. **Slow Inference**
   - Analyze sequence length distribution
   - Check model complexity factors
   - Review PyTorch compilation settings
   - Monitor resource utilization

3. **Model Health Failures**
   - Check model loading logs
   - Verify service health endpoints
   - Review deployment status
   - Consider model reload or rollback

## Security Considerations

### Production Security Features:
- **Secret Management**: All secrets stored in GCP Secret Manager
- **Automated Rotation**: 15-30 day rotation schedules for critical secrets
- **Access Control**: RBAC with time-based and conditional access
- **Encryption**: Customer-managed encryption keys for sensitive data
- **Audit Logging**: Comprehensive access and change logging
- **Network Security**: VPC isolation and firewall rules
- **Container Security**: Non-root execution and read-only filesystems

### Compliance:
- **SOC 2 Type II**: Audit logging and access controls
- **GDPR**: Data governance and retention policies
- **PCI DSS**: Secure handling of trading-related data
- **DLP**: Data loss prevention for sensitive content

## Cost Optimization

### Cost Management Features:
- **Peak Hours Scaling**: Reduced resources during off-peak hours
- **Budget Constraints**: Daily and monthly budget limits with alerts
- **Resource Optimization**: CPU/memory ratio optimization
- **Cost Allocation**: Detailed labeling for cost tracking
- **Right-Sizing**: Model-specific resource profiles to avoid over-provisioning

### Estimated Costs (Production):
- **Base Infrastructure**: ~$200-400/month
- **Peak Usage**: ~$500-1000/month during high trading activity
- **Monitoring**: ~$50-100/month for custom metrics and dashboards

## Support and Maintenance

### Regular Maintenance Tasks:
1. **Weekly**: Review performance metrics and cost reports
2. **Monthly**: Update API keys and review security policies
3. **Quarterly**: Evaluate model performance and resource utilization
4. **Semi-Annually**: Security audit and compliance review

### Escalation Procedures:
- **Performance Issues**: Performance team via Slack alerts
- **Security Incidents**: On-call engineer via PagerDuty
- **Model Issues**: ML team via email and Slack
- **Critical System**: Immediate escalation to engineering leadership

## Version History

- **v1.0.0** (2025-08-06): Initial production configuration templates
  - Complete transformer model support
  - GCP Cloud Run optimized deployment
  - Comprehensive monitoring and alerting
  - Production-ready security and compliance features

## Contributing

When updating these templates:
1. Base changes on actual implementation updates
2. Test in staging environment before production
3. Update documentation with new features or changes
4. Validate security and compliance requirements
5. Update version history and notify operations team

## Related Documentation

- [GCP Transformer Deployment Guide](../../docs/deployment/GCP_TRANSFORMER_DEPLOYMENT_GUIDE.md)
- [Configuration Management Guide](../../docs/deployment/CONFIG_MANAGEMENT_GUIDE.md)  
- [Monitoring and Observability Guide](../../docs/deployment/MONITORING_OBSERVABILITY_GUIDE.md)
- [Security and Access Control Guide](../../docs/deployment/SECURITY_ACCESS_CONTROL_GUIDE.md)
- [Troubleshooting Guide](../../docs/deployment/TROUBLESHOOTING_GUIDE.md)