# Production Rollout Documentation - Transformer Deployment

## Rollout Overview

This document provides comprehensive operational procedures for the production deployment of Transformer models within the existing Shyvr RLTE infrastructure. The rollout leverages and enhances the current deployment pipeline without replacing core components.

### Integration with Existing Deployment Pipeline

The transformer deployment **integrates with** the existing infrastructure:
- **Primary Orchestrator**: `./deploy/deploy.sh` remains the main deployment command
- **Pipeline Engine**: `automated_deployment_pipeline.sh` handles comprehensive validation
- **Traffic Management**: `blue_green_deployment.sh` manages zero-downtime deployments
- **Rollback System**: `automated_rollback.sh` provides emergency recovery

### How Transformers Enhance Current System

Transformers **enhance** rather than replace the existing system:
- **Ensemble Approach**: Transformers work alongside existing LSTM models
- **Gradual Rollout**: Progressive traffic migration ensures system stability
- **Fallback Capability**: Automatic fallback to LSTM models if transformer issues occur
- **Resource Optimization**: Phase 3.2.6 optimizations ensure efficient resource usage
- **Monitoring Integration**: Existing monitoring systems extended with transformer-specific metrics

### Key Deployment Commands Remain Unchanged

Core deployment commands maintain consistency:
```bash
# Standard production deployment
./deploy/deploy.sh production standard

# Blue-green deployment
./deploy/deploy.sh production blue-green

# Validation only
./deploy/deploy.sh production validation-only

# Rollback to previous version
./deploy/deploy.sh production rollback
```

## Pre-Deployment Checklist

### Verify Phase 3.2.6 Optimizations in Place

Before deploying transformer models, ensure all Phase 3.2.6 optimizations are active:

**✅ Resource Configuration Verification:**
```bash
# Verify optimized resource settings are loaded
grep -r "8Gi\|6.*CPU\|4200" deploy/configs/
```

**✅ Performance Optimizations Active:**
- Memory optimization: <2GB usage vs 8GB allocation
- Inference latency: <43ms average response time
- Throughput capability: 960-4,952 RPS validated
- Cost efficiency: 60% reduction in operational costs

**✅ Model Cache Configuration:**
```bash
# Verify model cache directories exist
ls -la /app/models/cache/
# Should contain: iTransformer/, PatchTST/, TimesMixer/, TimesFM/
```

### Resource Requirements Production

**Production Environment Specifications:**
- **Memory**: 8Gi RAM allocated (actual usage <2GB)
- **CPU**: 6 vCPU cores with CPU boost enabled
- **Timeout**: 4200 seconds for model initialization
- **Concurrency**: 15 concurrent requests maximum
- **Auto-scaling**: 1-10 instances based on load
- **Network**: Cloud SQL proxy connection enabled

**Staging Environment Specifications:**
- **Memory**: 4Gi RAM (reduced for staging)
- **CPU**: 4 vCPU cores
- **Timeout**: 3600 seconds
- **Concurrency**: 10 concurrent requests maximum
- **Auto-scaling**: 1-5 instances

### Configuration Files Ready

**Required Configuration Files:**
- `deploy/configs/transformer_rollout.yaml` - Progressive rollout settings ✅ READY
- `deploy/configs/transformer_models.json` - Model-specific configurations ✅ READY
- `deploy/modules/transformer_config_loader.sh` - Configuration injection module
- `deploy/modules/transformer_validation.sh` - Validation enhancement module

**Verify Configuration Integrity:**
```bash
# Validate YAML configuration syntax
python -c "import yaml; yaml.safe_load(open('deploy/configs/transformer_rollout.yaml'))"

# Validate JSON model configuration
python -c "import json; json.load(open('deploy/configs/transformer_models.json'))"
```

### Health Monitoring Endpoints Verified

**Core Health Endpoints:**
- `/health` - General system health (existing)
- `/health/transformers` - Transformer-specific health status
- `/health/iTransformer` - iTransformer model health
- `/health/PatchTST` - PatchTST model health  
- `/health/TimesMixer` - TimesMixer model health
- `/health/TimesFM` - TimesFM model health

**Health Check Validation:**
```bash
# Test health endpoints before deployment
SERVICE_URL=$(gcloud run services describe shyvr-rlte-staging --region=us-central1 --format="value(status.url)")
curl -f "$SERVICE_URL/health"
```

## Deployment Procedures

### Staging Deployment First

**Always deploy to staging environment first:**

```bash
# Step 1: Deploy to staging with standard pipeline
./deploy/deploy.sh staging standard latest

# Step 2: Validate staging deployment
./deploy/deploy.sh staging validation-only

# Step 3: Run integration tests on staging
python -m pytest tests/integration/ --staging
```

**Staging Validation Checklist:**
- [ ] Health endpoints responding within 30 seconds
- [ ] All transformer models loading successfully
- [ ] Memory usage under 2GB per instance
- [ ] Response time under 100ms for predictions
- [ ] Integration tests passing at 100%

### Canary Deployment (10% Traffic)

**Production canary deployment with 10% traffic:**

```bash
# Enable canary mode for transformer deployment
CANARY_MODE=true TRANSFORMER_MODEL_TYPE=iTransformer ./deploy/deploy.sh production blue-green
```

**Canary Configuration (from transformer_rollout.yaml):**
- **Traffic**: 10% to new revision, 90% to stable revision
- **Duration**: 5 minutes monitoring period
- **Validation**: Health checks every 20 seconds
- **Rollback Trigger**: >5% error rate or >2000ms response time

**Canary Monitoring Commands:**
```bash
# Monitor canary deployment logs
gcloud run logs tail shyvr-rlte --region us-central1

# Check traffic distribution
gcloud run services describe shyvr-rlte --region us-central1 --format="table(spec.traffic[].revisionName,spec.traffic[].percent)"

# Monitor performance metrics
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health | jq .
```

### Progressive Rollout Stages

**Stage 1: Limited Rollout (25% Traffic)**

```bash
# After successful canary, increase to 25% traffic
# This happens automatically via transformer_rollout.yaml configuration
# Monitor for 10 minutes with enhanced validation
```

**Stage 2: Expanded Rollout (50% Traffic)**

```bash
# Increase to 50% traffic after Stage 1 validation
# Monitor for 15 minutes with full validation suite
# Include transformer model accuracy checks
```

**Stage 3: Full Rollout (100% Traffic)**

```bash
# Complete rollout to 100% traffic
# Monitor for 30 minutes with end-to-end validation
# All validation checks including prediction quality assessment
```

**Progressive Rollout Monitoring:**
- **Error Rate**: Must remain <5% throughout all stages
- **Response Time**: Must stay <2000ms (target <100ms)
- **Resource Usage**: Memory <8GB, CPU <80%
- **Model Accuracy**: No degradation >15% from baseline
- **Prediction Quality**: Quality score >0.7

### Monitoring at Each Stage

**Comprehensive Monitoring Strategy:**

**Real-time Metrics Monitoring:**
```bash
# Monitor system metrics during rollout
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="shyvr-rlte"' --limit=50 --format json

# Monitor transformer-specific metrics
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/metrics/transformers
```

**Performance Validation at Each Stage:**
- **Stage Validation**: Automated checks every 20 seconds
- **Health Thresholds**: 95% success rate required
- **Resource Monitoring**: Real-time memory and CPU tracking
- **Model Performance**: Prediction accuracy validation
- **End-to-end Testing**: Complete workflow validation

### Rollback Procedures Using Existing Scripts

**Automatic Rollback Triggers:**
- Error rate >10%
- Response time >3000ms
- 5 consecutive health check failures
- Resource utilization >90%
- Model accuracy drop >15%

**Emergency Rollback Command:**
```bash
# Immediate rollback to previous stable version
./deploy/deploy.sh production rollback

# Emergency rollback with automated recovery
./deploy/automated_rollback.sh production emergency
```

**Rollback Validation:**
```bash
# Verify rollback completed successfully
./deploy/validate_deployment.sh

# Confirm service stability after rollback
gcloud run services describe shyvr-rlte --region us-central1 --format="table(status.traffic[].revisionName,status.traffic[].percent)"
```

## Integration Points

### How deploy.sh Loads Transformer Configs

The main deployment orchestrator automatically detects transformer deployments:

```bash
# In deploy.sh, transformer configuration loading:
load_transformer_config() {
    if [[ -n "${TRANSFORMER_MODEL_TYPE:-}" ]]; then
        local config_loader_module="$SCRIPT_DIR/modules/transformer_config_loader.sh"
        if [[ -f "$config_loader_module" ]]; then
            source "$config_loader_module"
            if load_transformer_configurations "$TRANSFORMER_MODEL_TYPE" "$ENVIRONMENT"; then
                export_all_config
            fi
        fi
    fi
}
```

**Configuration Loading Process:**
1. **Detection**: Environment variable `TRANSFORMER_MODEL_TYPE` triggers config loading
2. **Loading**: `transformer_config_loader.sh` module injects model-specific settings
3. **Validation**: Configuration integrity checked before deployment proceeds
4. **Export**: All transformer settings exported to deployment environment

### How blue_green_deployment.sh Handles Canary Mode

Enhanced blue-green deployment supports canary mode:

```bash
# Canary mode detection in blue_green_deployment.sh:
if [[ "${CANARY_MODE:-false}" == "true" ]]; then
    if execute_canary_deployment && execute_blue_green_migration; then
        deployment_success=true
    fi
fi
```

**Canary Integration Features:**
- **Traffic Control**: Precise traffic percentage control (10%, 25%, 50%, 100%)
- **Monitoring Integration**: Enhanced health checks during canary phase
- **Automatic Progression**: Configuration-driven rollout stages
- **Rollback Integration**: Seamless rollback if canary validation fails

### How automated_deployment_pipeline.sh Validates Transformers

The comprehensive pipeline includes transformer-specific validation:

**Enhanced Pipeline Stages:**
1. **Infrastructure Setup** - Verify transformer-optimized resources
2. **Pre-deployment Validation** - Standard validation plus transformer checks
3. **Security Scan** - Include transformer model security validation
4. **Transformer Validation** - New stage for model-specific validation
5. **Build and Test** - Enhanced testing with transformer integration
6. **Blue-green Deployment** - Enhanced with canary mode support

**Transformer Validation Stage:**
```bash
# Stage 4 enhancement in automated_deployment_pipeline.sh
run_transformer_validation() {
    if [[ -n "${TRANSFORMER_MODEL_TYPE:-}" ]]; then
        log_info "Running transformer-specific validation..."
        source "$SCRIPT_DIR/modules/transformer_validation.sh"
        validate_transformer_deployment || return 1
    fi
}
```

### Health Endpoint Monitoring

**Enhanced Health Endpoint Structure:**
- **Main Health**: `/health` - Overall system status
- **Model Health**: `/health/transformers` - All transformer models status
- **Specific Model Health**: `/health/{MODEL_TYPE}` - Individual model status

**Health Response Format:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-06T12:00:00Z",
  "models": {
    "iTransformer": {"status": "healthy", "memory": "1.2GB", "latency": "43ms"},
    "PatchTST": {"status": "healthy", "memory": "0.8GB", "latency": "31ms"},
    "TimesMixer": {"status": "healthy", "memory": "1.5GB", "latency": "52ms"},
    "TimesFM": {"status": "healthy", "memory": "2.1GB", "latency": "67ms"}
  },
  "system": {
    "cpu_usage": "45%",
    "memory_usage": "5.6GB/8GB",
    "uptime": "72h 15m"
  }
}
```

## Operational Commands

### Standard Deployment

**Production Standard Deployment:**
```bash
# Deploy latest version to production with full pipeline
./deploy/deploy.sh production standard

# Deploy specific version
IMAGE_TAG=v1.2.3 ./deploy/deploy.sh production standard
```

### Canary Deployment

**Production Canary with Transformer Support:**
```bash
# Enable canary mode with transformer model
CANARY_MODE=true ./deploy/deploy.sh production blue-green

# Canary with specific transformer model
CANARY_MODE=true TRANSFORMER_MODEL_TYPE=iTransformer ./deploy/deploy.sh production blue-green
```

### Transformer-Specific Deployment

**Deploy Specific Transformer Model:**
```bash
# Deploy iTransformer model
TRANSFORMER_MODEL_TYPE=iTransformer ./deploy/deploy.sh production standard

# Deploy PatchTST model  
TRANSFORMER_MODEL_TYPE=PatchTST ./deploy/deploy.sh production standard

# Deploy TimesMixer model
TRANSFORMER_MODEL_TYPE=TimesMixer ./deploy/deploy.sh production standard

# Deploy TimesFM foundation model
TRANSFORMER_MODEL_TYPE=TimesFM ./deploy/deploy.sh production standard
```

### Rollback Operations

**Standard Rollback:**
```bash
# Rollback to previous stable version
./deploy/deploy.sh production rollback

# Emergency rollback with minimal validation
FORCE_DEPLOY=true ./deploy/deploy.sh production rollback
```

**Advanced Rollback Operations:**
```bash
# Rollback to specific revision
./deploy/automated_rollback.sh production specific <revision-name>

# Emergency rollback with full logging
./deploy/automated_rollback.sh production emergency
```

### Validation Commands

**Deployment Validation:**
```bash
# Validate current deployment
./deploy/deploy.sh production validation-only

# Comprehensive validation with transformer checks
./deploy/validate_deployment.sh

# Production readiness checklist
./deploy/production_readiness_checklist.sh
```

## Monitoring and Validation

### Performance Metrics to Track

**System Performance Metrics:**
- **Response Time**: Target <100ms, Alerting threshold >2000ms
- **Throughput**: Target >500 RPS, Current capability 960-4,952 RPS
- **Memory Usage**: Target <2GB, Allocation limit 8GB
- **CPU Utilization**: Target <80%, Monitor for spikes >90%
- **Error Rate**: Target <1%, Alert threshold >5%

**Transformer-Specific Metrics:**
- **Model Loading Time**: Target <60s, Monitor >120s
- **Inference Latency**: Target <50ms per model, Alert >100ms
- **Memory Per Model**: Track individual model memory consumption
- **Model Accuracy**: Monitor for >15% degradation from baseline
- **Attention Pattern Quality**: Monitor attention coherence scores

**Business Metrics:**
- **Prediction Quality**: Target quality score >0.7
- **Trading Performance**: Monitor Sharpe ratio improvements
- **False Signal Rate**: Target <25% of baseline LSTM rate
- **Model Utilization**: Track usage across different transformer models

### SLA Compliance Checks

**Uptime SLA: 99.9%**
- **Monitoring**: Continuous availability monitoring
- **Measurement**: Monthly uptime calculation
- **Alerting**: Immediate alerts for service unavailability

**Performance SLA: <100ms Response Time**
- **Monitoring**: Real-time latency tracking
- **Measurement**: 95th percentile response time
- **Alerting**: Alerts when >2000ms sustained for >5 minutes

**Capacity SLA: >500 RPS**
- **Monitoring**: Throughput measurement during peak hours
- **Measurement**: Peak capacity during market hours
- **Alerting**: Degradation below 80% of capacity

### Resource Utilization Monitoring

**Memory Monitoring:**
```bash
# Monitor memory usage in real-time
gcloud run services describe shyvr-rlte --region us-central1 --format="table(spec.template.spec.template.spec.containers[0].resources.limits.memory)"

# Track memory consumption trends
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/metrics/memory
```

**CPU Monitoring:**
```bash
# Monitor CPU utilization
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.message~"CPU"' --limit=20

# Track CPU trends during peak trading hours
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/metrics/cpu
```

**Auto-scaling Monitoring:**
- **Instance Count**: Monitor scaling events during peak hours
- **Scaling Triggers**: Track memory and CPU thresholds
- **Cold Start Impact**: Monitor first request latency for new instances

### Model-Specific Health Checks

**Individual Model Health Validation:**
```bash
# Check iTransformer health
curl -f $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health/iTransformer

# Check all transformer models
for model in iTransformer PatchTST TimesMixer TimesFM; do
    echo "Checking $model..."
    curl -f "$(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health/$model"
done
```

**Model Performance Validation:**
```bash
# Validate prediction quality
curl -X POST $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/validate/prediction \
     -H "Content-Type: application/json" \
     -d '{"test_data": "sample_market_data"}'

# Test ensemble prediction
curl -X POST $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/predict/ensemble \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTC", "timeframe": "1h"}'
```

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: Transformer Model Loading Timeout**
```
Symptoms: 504 Gateway Timeout during deployment
Root Cause: Model initialization exceeding 4200s timeout
```

**Solution:**
```bash
# Check model cache status
ls -la /app/models/cache/
# If cache is empty, models download on first request

# Increase timeout temporarily for first deployment
gcloud run services update shyvr-rlte --timeout=6000 --region us-central1

# Pre-warm models in staging
TRANSFORMER_MODEL_TYPE=iTransformer ./deploy/deploy.sh staging validation-only
```

**Issue 2: High Memory Usage During Model Loading**
```
Symptoms: Out of memory errors, service restarts
Root Cause: Multiple models loading simultaneously
```

**Solution:**
```bash
# Enable sequential model loading
export MODEL_LOADING_MODE=sequential
TRANSFORMER_MODEL_TYPE=iTransformer ./deploy/deploy.sh production blue-green

# Monitor memory during deployment
watch "gcloud run logs tail shyvr-rlte --region us-central1 | grep -i memory"
```

**Issue 3: Canary Deployment Stuck at 10% Traffic**
```
Symptoms: Traffic not progressing beyond canary stage
Root Cause: Health checks failing or validation thresholds not met
```

**Solution:**
```bash
# Check canary validation status
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health/canary

# Force progression if health checks pass manually
gcloud run services update-traffic shyvr-rlte --to-revisions=LATEST=25 --region us-central1

# Review canary configuration
cat deploy/configs/transformer_rollout.yaml | grep -A 10 canary
```

**Issue 4: Model Prediction Accuracy Degradation**
```
Symptoms: Prediction quality scores dropping below 0.7
Root Cause: Model drift or data distribution changes
```

**Solution:**
```bash
# Trigger model drift analysis
curl -X POST $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/analyze/drift

# Fallback to LSTM models temporarily
FALLBACK_MODE=true ./deploy/deploy.sh production rollback

# Retrain transformer with recent data
python scripts/retrain_transformers.py --recent-data-only
```

### Log Locations

**Primary Log Sources:**
```bash
# Cloud Run service logs
gcloud run logs tail shyvr-rlte --region us-central1

# Deployment pipeline logs
ls -la deployment_pipeline_*.log

# Transformer-specific logs
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.component="transformer"'

# Blue-green deployment logs
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.deployment_type="blue_green"'
```

**Log Analysis Commands:**
```bash
# Search for errors during deployment
gcloud logging read 'resource.type="cloud_run_revision" AND severity>=ERROR' --limit=50

# Monitor health check failures
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.message~"health.*fail"' --limit=20

# Track memory usage patterns
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.message~"memory"' --limit=30
```

### Health Check Failures

**Common Health Check Failure Scenarios:**

**Scenario 1: Model Initialization Failure**
```bash
# Check model loading status
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health/models | jq '.models'

# Verify model files exist
gcloud run execute shyvr-rlte --region us-central1 --command="ls -la /app/models/cache/"

# Check model configuration
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/debug/config
```

**Scenario 2: Database Connection Issues**
```bash
# Test database connectivity
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/health/database

# Check Cloud SQL instance status
gcloud sql instances describe shyvr-rlte-db --project=shvyr-ai-bots

# Verify connection configuration
gcloud secrets versions access latest --secret=DATABASE_URL --project=shvyr-ai-bots
```

**Scenario 3: Resource Exhaustion**
```bash
# Check resource limits
gcloud run services describe shyvr-rlte --region us-central1 --format="table(spec.template.spec.template.spec.containers[0].resources)"

# Monitor current resource usage
curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/metrics/resources

# Scale up resources if needed
gcloud run services update shyvr-rlte --memory=8Gi --cpu=6 --region us-central1
```

### Resource Exhaustion Handling

**Memory Exhaustion Recovery:**
```bash
# Immediate response: Scale up memory
gcloud run services update shyvr-rlte --memory=10Gi --region us-central1

# Long-term: Optimize model loading
export MODEL_LOADING_STRATEGY=lazy
TRANSFORMER_MODEL_TYPE=PatchTST ./deploy/deploy.sh production standard

# Monitor memory patterns
watch "curl -s $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)")/metrics/memory"
```

**CPU Exhaustion Recovery:**
```bash
# Scale CPU resources
gcloud run services update shyvr-rlte --cpu=8 --region us-central1

# Enable CPU boost for better performance
gcloud run services update shyvr-rlte --cpu-boost --region us-central1

# Optimize transformer inference
export TORCH_COMPILE_MODE=max-autotune
./deploy/deploy.sh production standard
```

**Network Timeout Recovery:**
```bash
# Increase request timeout
gcloud run services update shyvr-rlte --timeout=4200 --region us-central1

# Check concurrent request limits
gcloud run services update shyvr-rlte --concurrency=15 --region us-central1

# Monitor network latency
ping -c 5 $(gcloud run services describe shyvr-rlte --region us-central1 --format="value(status.url)" | sed 's|https://||')
```

---

## Deployment Validation Checklist

### Pre-Deployment Validation
- [ ] Phase 3.2.6 optimizations verified and active
- [ ] Resource requirements validated (8Gi RAM, 6 CPU)
- [ ] Configuration files present and validated
- [ ] Staging environment deployment successful
- [ ] Health endpoints responding correctly
- [ ] Integration tests passing at 100%

### During Deployment Validation
- [ ] Canary deployment (10% traffic) health checks passing
- [ ] Progressive rollout stages completing within SLA
- [ ] Error rate remaining below 5% threshold
- [ ] Response time staying under 2000ms
- [ ] Memory usage under 8GB allocation
- [ ] Model-specific health endpoints responding

### Post-Deployment Validation
- [ ] Full traffic (100%) deployment successful
- [ ] All transformer models loaded and operational
- [ ] End-to-end prediction workflow functional
- [ ] Monitoring systems capturing transformer metrics
- [ ] Rollback procedures tested and verified
- [ ] Operations team notification completed

### Production Sign-off Requirements
- [ ] SLA compliance validated (99.9% uptime, <100ms latency)
- [ ] Performance metrics within acceptable ranges
- [ ] Security validation completed
- [ ] Monitoring and alerting operational
- [ ] Incident response procedures updated
- [ ] Operations team training completed

---

*This document serves as the definitive guide for transformer model production deployment. All procedures integrate with existing infrastructure while providing transformer-specific enhancements for optimal performance and reliability.*

**Document Version**: 1.0  
**Last Updated**: August 6, 2025  
**Phase**: 3.3.3 Implementation  
**Status**: Production Ready