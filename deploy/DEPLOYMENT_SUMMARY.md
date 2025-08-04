# Shyvr RLTE Deployment Infrastructure Summary

## Overview

This directory contains a clean, maintainable deployment infrastructure for the Shyvr RLTE AI-augmented cryptocurrency trading system. The refactored deployment system provides a unified interface for all deployment operations while maintaining zero-downtime blue-green deployments, automated rollbacks, comprehensive monitoring, and disaster recovery.

## Quick Start

### Unified Deployment Runner (Recommended)
```bash
# Deploy to staging
./deploy/deploy.sh staging

# Deploy to production
./deploy/deploy.sh production

# Emergency deployment
./deploy/deploy.sh production emergency

# Validation only
./deploy/deploy.sh staging validation-only

# Rollback
./deploy/deploy.sh production rollback
```

### Direct Script Usage (Advanced)
```bash
# 1. Validate readiness
./deploy/production_readiness_checklist.sh

# 2. Deploy with comprehensive pipeline
./deploy/automated_deployment_pipeline.sh production

# 3. Validate deployment
./deploy/validate_deployment.sh
```

## Deployment Scripts

### Main Deployment Interface
- **`deploy.sh`** - **NEW**: Unified deployment runner that orchestrates all deployment modes
- **`deploy-utils.sh`** - **NEW**: Shared utilities and functions for consistent deployment operations

### Core Deployment Pipeline
- **`automated_deployment_pipeline.sh`** - Complete 8-stage automated deployment pipeline with validation
- **`blue_green_deployment.sh`** - Zero-downtime blue-green deployment with gradual traffic migration
- **`validate_deployment.sh`** - Comprehensive post-deployment health and functionality validation
- **`automated_rollback.sh`** - Multi-strategy automated rollback with emergency capabilities

### Pre-Deployment Validation
- **`production_readiness_checklist.sh`** - Comprehensive 10-section pre-deployment validation

### Infrastructure Setup
- **`setup_secrets.sh`** - Secret management and configuration
- **`setup_cloud_sql.sh`** - Database infrastructure setup
- **`setup_monitoring.sh`** - Basic monitoring setup
- **`setup_production_monitoring.sh`** - Production monitoring and alerting setup
- **`setup_gcs_infrastructure.sh`** - Google Cloud Storage setup for model preservation
- **`setup_experience_database.sh`** - RL experience database setup
- **`setup_build_triggers.sh`** - CI/CD build trigger configuration
- **`sla_monitoring_setup.sh`** - SLA monitoring configuration

### Validation & Testing Utilities
- **`validate_secrets.py`** - Secret validation utility
- **`validate_gcs_setup.py`** - GCS configuration validation
- **`test_cloud_sql_connectivity.py`** - Database connectivity testing
- **`final_security_audit.py`** - Security audit automation

### Maintenance & Operations
- **`rotate_secrets.py`** - Secret rotation automation

### Archive (Legacy Scripts)
- **`archive/deploy_latest.sh`** - Legacy deployment script (superseded by automated_deployment_pipeline.sh)
- **`archive/deploy_and_configure.sh`** - Simple orchestrator (functionality moved to deploy.sh)
- **`archive/final_deployment_package.sh`** - Complex Phase 8.2 script (functionality integrated into main pipeline)

## Key Features

### 🚀 Zero-Downtime Deployments
- Blue-green deployment strategy
- Gradual traffic migration (10% → 50% → 100%)
- Automatic rollback on health check failures
- Session affinity preservation

### 🔒 Production Safety
- Comprehensive pre-deployment validation
- Multi-stage health checks
- Emergency rollback capabilities
- Trading safety system integration

### 📊 Monitoring & Observability
- Custom metrics for trading performance
- Automated alerting policies
- Production dashboards
- Log-based metrics and analysis

### 🔄 Automation
- End-to-end deployment pipeline
- Automated infrastructure setup
- Secret management automation
- Health monitoring with auto-recovery

## Deployment Modes

### Standard Deployment (Default)
- **Command**: `./deploy/deploy.sh <environment>`
- **Process**: Full 8-stage automated pipeline with comprehensive validation
- **Use Case**: Regular deployments with full safety checks

### Emergency Deployment
- **Command**: `./deploy/deploy.sh <environment> emergency`
- **Process**: Minimal validation, fastest deployment possible
- **Use Case**: Critical hotfixes and emergency deployments

### Blue-Green Only
- **Command**: `./deploy/deploy.sh <environment> blue-green`
- **Process**: Blue-green deployment with post-validation
- **Use Case**: When pre-built image already exists

### Validation Only
- **Command**: `./deploy/deploy.sh <environment> validation-only`
- **Process**: Run all validation checks without deployment
- **Use Case**: Pre-deployment verification and health checks

### Rollback
- **Command**: `./deploy/deploy.sh <environment> rollback`
- **Process**: Automated rollback to previous version
- **Use Case**: Emergency rollback scenarios

## Deployment Environments

### Staging Environment
- **Service**: `shyvr-rlte-staging`
- **Purpose**: Pre-production testing and validation
- **Traffic Strategy**: 50% → 100% migration
- **Resource Allocation**: 4Gi memory, 2 CPU

### Production Environment
- **Service**: `shyvr-rlte`
- **Purpose**: Live trading system
- **Traffic Strategy**: 10% → 50% → 100% migration
- **Resource Allocation**: 6Gi memory, 4 CPU
- **Enhanced Safety**: Extended health checks, emergency controls

## Configuration Files

### Cloud Build
- **`../config/cloudbuild.yaml`** - CI/CD pipeline with blue-green deployment

### Docker
- **`../docker/docker-compose.production.yml`** - Production container orchestration
- **`../Dockerfile`** - Optimized multi-stage build for ML/RL workloads

### Monitoring
- **`../config/monitoring_config.json`** - Monitoring configuration
- **`../monitoring/grafana/`** - Grafana dashboards and alerts

## Emergency Procedures

### Immediate Rollback
```bash
# Using unified runner (recommended)
./deploy/deploy.sh production rollback

# Direct rollback script
./deploy/automated_rollback.sh production emergency
```

### Health Check
```bash
curl -s https://your-service-url/health | jq .
```

### Service Logs
```bash
gcloud run logs tail shyvr-rlte --region us-central1 --follow
```

### Emergency Stop Trading
```bash
curl -X POST https://your-service-url/api/emergency-stop
```

### Emergency Deployment
```bash
# Skip all validation for fastest deployment
./deploy/deploy.sh production emergency
```

## Documentation

- **[Production Deployment Runbook](../docs/deployment/PRODUCTION_DEPLOYMENT_RUNBOOK.md)** - Complete operational procedures
- **[Comprehensive Deployment Tutorial](../docs/deployment/COMPREHENSIVE_DEPLOYMENT_TUTORIAL.md)** - Step-by-step deployment guide
- **[Deployment Readiness Checklist](../docs/deployment/DEPLOYMENT_READINESS_CHECKLIST.md)** - Pre-deployment validation

## Security Considerations

### Secret Management
- All secrets stored in Google Secret Manager
- Automatic secret validation before deployment
- Support for secret rotation without downtime

### Access Controls
- Service account with minimal required permissions
- Network isolation for database access
- Container security with non-root user

### Data Protection
- Encrypted data at rest and in transit
- Database backup automation
- Model preservation with versioning

## Performance Optimizations

### Container Optimization
- Multi-stage Docker builds
- CPU-only PyTorch for container efficiency
- Optimized Python dependencies

### Resource Management
- Memory allocation for ML model caching
- CPU boost for faster cold starts
- Intelligent concurrency settings

### Caching Strategy
- Model preservation caching
- Database connection pooling
- Application-level caching

## Monitoring Metrics

### System Metrics
- CPU utilization (target: <80%)
- Memory usage (target: <85%)
- Request latency (target: <2000ms)
- Error rate (target: <5%)

### Trading Metrics
- Active position count
- P&L tracking
- Risk score monitoring
- Emergency stop triggers

### ML/RL Metrics
- Model accuracy trends
- RL episode rewards
- Experience storage capacity
- Model loading performance

## Support and Escalation

### Deployment Issues
1. Check deployment logs
2. Review health check failures
3. Initiate rollback if needed
4. Escalate to on-call engineer

### Production Issues
1. Monitor alerts and dashboards
2. Use emergency procedures if critical
3. Follow incident response playbook
4. Document resolution for post-mortem

---

## Deployment Workflow

### Typical Deployment Process

1. **Validation Phase**
   ```bash
   # Validate environment and configuration
   ./deploy/deploy.sh staging validation-only
   ```

2. **Staging Deployment**
   ```bash
   # Deploy to staging for testing
   ./deploy/deploy.sh staging
   ```

3. **Production Deployment**
   ```bash
   # Deploy to production with full pipeline
   ./deploy/deploy.sh production
   ```

4. **Emergency Scenarios**
   ```bash
   # Emergency deployment (skip validation)
   ./deploy/deploy.sh production emergency
   
   # Emergency rollback
   ./deploy/deploy.sh production rollback
   ```

### Environment Variables for Customization

- **`DRY_RUN=true`** - Show what would be deployed without executing
- **`SKIP_TESTS=true`** - Skip testing phases in pipeline
- **`FORCE_DEPLOY=true`** - Skip validation failures
- **`LOG_DIR=/path/to/logs`** - Custom log directory

### Script Dependencies

The unified deployment system automatically handles dependencies:
- `deploy.sh` → orchestrates all deployment modes
- `deploy-utils.sh` → provides shared functions
- `automated_deployment_pipeline.sh` → comprehensive deployment
- `blue_green_deployment.sh` → zero-downtime deployment
- `validate_deployment.sh` → post-deployment validation
- `automated_rollback.sh` → rollback operations

---

**Last Updated**: $(date)  
**Version**: 2.0 (Refactored)  
**Maintained By**: DevOps Team