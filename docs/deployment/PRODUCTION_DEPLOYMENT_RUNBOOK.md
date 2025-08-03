# Production Deployment Runbook - Shyvr RLTE

This runbook provides comprehensive procedures for deploying, operating, and maintaining the Shyvr RLTE AI-augmented cryptocurrency trading system in production.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Procedures](#deployment-procedures)
3. [Health Monitoring](#health-monitoring)
4. [Troubleshooting](#troubleshooting)
5. [Rollback Procedures](#rollback-procedures)
6. [Disaster Recovery](#disaster-recovery)
7. [Operational Procedures](#operational-procedures)
8. [Emergency Contacts](#emergency-contacts)

## Pre-Deployment Checklist

### Prerequisites Validation

Before any production deployment, ensure the following:

#### 1. Development Environment
- [ ] GCP CLI authenticated and configured
- [ ] Docker daemon running
- [ ] Python 3.12+ environment available
- [ ] Access to project `shvyr-ai-bots`

#### 2. Code Quality
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Security scan completed
- [ ] No hardcoded secrets in code
- [ ] Code review completed

#### 3. Infrastructure Readiness
- [ ] Cloud SQL instance healthy
- [ ] Artifact Registry repository accessible
- [ ] Required secrets configured in Secret Manager
- [ ] Service account permissions validated
- [ ] GCS bucket for model preservation available

#### 4. Run Production Readiness Check
```bash
./deploy/production_readiness_checklist.sh
```
This must pass before proceeding with deployment.

## Deployment Procedures

### Automated Deployment (Recommended)

The automated deployment pipeline handles the complete deployment process:

```bash
# Deploy to staging first
./deploy/automated_deployment_pipeline.sh staging

# After staging validation, deploy to production
./deploy/automated_deployment_pipeline.sh production
```

### Manual Deployment Steps

If manual deployment is required:

#### 1. Build and Push Image
```bash
# Build Docker image
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/shvyr-ai-bots/shyvr-ai-prod/shyvr-rlte:$(date +%Y%m%d-%H%M%S) .

# Push to registry
docker push us-central1-docker.pkg.dev/shvyr-ai-bots/shyvr-ai-prod/shyvr-rlte:$(date +%Y%m%d-%H%M%S)
```

#### 2. Blue-Green Deployment
```bash
./deploy/blue_green_deployment.sh production <image_tag>
```

#### 3. Validate Deployment
```bash
./deploy/validate_deployment.sh
```

### Post-Deployment Verification

After deployment completion:

1. **Health Check Validation**
   ```bash
   curl -s https://your-service-url/health | jq .
   ```

2. **Critical Endpoint Testing**
   ```bash
   # Test configuration endpoint
   curl -s https://your-service-url/config | jq .

   # Test metrics endpoint
   curl -s https://your-service-url/metrics
   ```

3. **Trading System Verification**
   - [ ] ML models loading successfully
   - [ ] RL agent initialized
   - [ ] Database connectivity confirmed
   - [ ] Safety systems active
   - [ ] Model preservation functional

## Health Monitoring

### Real-Time Monitoring

#### 1. Service Health
Monitor the main health endpoint:
```bash
watch -n 30 'curl -s https://your-service-url/health | jq .'
```

#### 2. Key Metrics to Monitor
- CPU utilization (< 80%)
- Memory usage (< 85%)
- Request latency (< 2000ms)
- Error rate (< 5%)
- Database connection health
- ML model performance
- RL agent episode rewards

#### 3. GCP Console Monitoring
- **Cloud Run Metrics**: https://console.cloud.google.com/run/detail/us-central1/shyvr-rlte/metrics
- **Logs Explorer**: https://console.cloud.google.com/logs/query
- **Error Reporting**: https://console.cloud.google.com/errors

### Automated Monitoring

The production monitoring setup includes:
- Custom metrics for trading performance
- Alerting policies for critical conditions
- Uptime checks for service availability
- Log-based metrics for application events

#### Critical Alerts

Monitor for these critical conditions:
1. **Emergency Stop Triggered** - Immediate response required
2. **High Error Rate** - Investigate within 5 minutes
3. **Service Down** - Immediate response required
4. **Database Connectivity Lost** - Immediate response required
5. **ML Model Failures** - Investigate within 15 minutes

## Troubleshooting

### Common Issues and Solutions

#### 1. Deployment Failures

**Symptom**: Blue-green deployment fails during health checks
```
Health check $i failed
❌ Blue-green deployment failed
```

**Solution**:
1. Check logs for specific error:
   ```bash
   gcloud run logs tail shyvr-rlte --region us-central1 --limit 100
   ```
2. Common causes:
   - Database connection issues
   - Missing or invalid secrets
   - ML model loading failures
   - Resource constraints

**Recovery**:
```bash
# Automatic rollback should trigger, but if not:
./deploy/automated_rollback.sh production emergency
```

#### 2. High Memory Usage

**Symptom**: Memory usage > 85% for extended periods

**Investigation**:
```bash
# Check current resource allocation
gcloud run services describe shyvr-rlte --region us-central1 --format="value(spec.template.spec.containers[0].resources.limits.memory)"

# Review memory-intensive operations in logs
gcloud run logs tail shyvr-rlte --region us-central1 | grep -i "memory\|oom"
```

**Solution**:
1. Increase memory allocation:
   ```bash
   gcloud run services update shyvr-rlte --region us-central1 --memory 8Gi
   ```
2. Optimize model caching strategies
3. Review ML model memory usage

#### 3. ML Model Loading Failures

**Symptom**: ML models fail to initialize

**Investigation**:
```bash
# Check model-specific logs
gcloud run logs tail shyvr-rlte --region us-central1 | grep -i "model\|ml\|lstm"

# Verify GCS bucket access
gsutil ls gs://shyvr-models-prod/
```

**Solution**:
1. Verify GCS permissions
2. Check model file integrity
3. Review model preservation configuration

#### 4. Database Connection Issues

**Symptom**: Database health checks failing

**Investigation**:
```bash
# Check Cloud SQL instance status
gcloud sql instances describe shyvr-rlte-db-prod

# Test connectivity
./deploy/test_cloud_sql_connectivity.py
```

**Solution**:
1. Verify Cloud SQL instance is running
2. Check connection pool configuration
3. Validate database credentials in Secret Manager

### Emergency Procedures

#### 1. Emergency Rollback

For critical production issues:
```bash
# Immediate rollback to previous working version
./deploy/automated_rollback.sh production emergency
```

#### 2. Emergency Stop Trading

If trading needs to be stopped immediately:
```bash
# Access the emergency stop endpoint (requires authentication)
curl -X POST https://your-service-url/api/emergency-stop \
  -H "Authorization: Bearer <emergency-token>"
```

#### 3. Scale Down Service

To reduce load during issues:
```bash
gcloud run services update shyvr-rlte --region us-central1 --min-instances 0 --max-instances 1
```

## Rollback Procedures

### Automated Rollback

The system supports several rollback strategies:

#### 1. Previous Revision Rollback
```bash
./deploy/automated_rollback.sh production previous
```

#### 2. Specific Revision Rollback
```bash
# List available revisions
gcloud run revisions list --service shyvr-rlte --region us-central1

# Rollback to specific revision
./deploy/automated_rollback.sh production <revision-name>
```

#### 3. Emergency Rollback
```bash
# Fastest rollback with minimal validation
./deploy/automated_rollback.sh production emergency
```

### Manual Rollback

If automated rollback fails:

```bash
# Get previous revision
PREVIOUS_REVISION=$(gcloud run revisions list --service shyvr-rlte --region us-central1 --format="value(metadata.name)" --sort-by="~metadata.creationTimestamp" --limit 2 | tail -1)

# Switch traffic
gcloud run services update-traffic shyvr-rlte --region us-central1 --to-revisions="$PREVIOUS_REVISION=100"
```

### Post-Rollback Validation

After any rollback:
1. Verify service health
2. Test critical endpoints
3. Monitor for 30 minutes
4. Document rollback reason and resolution

## Disaster Recovery

### Backup and Recovery

#### 1. Database Backups
- Automated daily backups configured in Cloud SQL
- Manual backup before major deployments:
  ```bash
  gcloud sql backups create --instance shyvr-rlte-db-prod
  ```

#### 2. Model Preservation Backups
- ML/RL models automatically backed up to GCS
- Manual model backup:
  ```bash
  gsutil -m cp -r gs://shyvr-models-prod/ gs://shyvr-models-backup/$(date +%Y%m%d)/
  ```

#### 3. Configuration Backups
- Secrets stored in Secret Manager (versioned)
- Configuration files in version control

### Recovery Procedures

#### 1. Complete Service Recovery

If the entire service needs to be rebuilt:

```bash
# Deploy from last known good configuration
git checkout <last-known-good-commit>
./deploy/automated_deployment_pipeline.sh production
```

#### 2. Database Recovery

If database corruption occurs:

```bash
# List available backups
gcloud sql backups list --instance shyvr-rlte-db-prod

# Restore from backup
gcloud sql backups restore <backup-id> --restore-instance shyvr-rlte-db-prod
```

#### 3. Model Recovery

If ML/RL models are corrupted:

```bash
# Restore models from backup
gsutil -m cp -r gs://shyvr-models-backup/<backup-date>/* gs://shyvr-models-prod/
```

## Operational Procedures

### Daily Operations

#### 1. Morning Health Check (9:00 AM EST)
```bash
# Run comprehensive health check
./deploy/validate_deployment.sh

# Check overnight trading activity
gcloud run logs tail shyvr-rlte --region us-central1 --since "24h" | grep -i "trade\|position"

# Verify ML model performance
curl -s https://your-service-url/health | jq '.components.ml_models'
```

#### 2. Resource Monitoring
- Review CPU and memory trends
- Check error rates and latency
- Monitor trading performance metrics

#### 3. Log Analysis
```bash
# Check for errors in last 24 hours
gcloud run logs tail shyvr-rlte --region us-central1 --since "24h" | grep -i "error\|exception\|failure"

# Monitor trading safety triggers
gcloud run logs tail shyvr-rlte --region us-central1 --since "24h" | grep -i "emergency\|circuit.*breaker\|risk.*limit"
```

### Weekly Operations

#### 1. Performance Review
- Analyze weekly trading performance
- Review ML model accuracy trends
- Check RL agent learning progress

#### 2. Security Audit
```bash
# Run security validation
./deploy/production_readiness_checklist.sh

# Check secret rotation needs
gcloud secrets list --format="table(name,createTime)" | grep -E "(TOKEN|KEY|PASSWORD)"
```

#### 3. Capacity Planning
- Review resource utilization trends
- Plan for scaling if needed
- Evaluate model storage requirements

### Monthly Operations

#### 1. Model Maintenance
- Review and update ML models
- Analyze RL agent performance
- Clean up old model versions

#### 2. Infrastructure Review
- Review GCP resource usage and costs
- Update dependencies and base images
- Perform security updates

#### 3. Disaster Recovery Testing
- Test backup and restore procedures
- Validate rollback mechanisms
- Update emergency procedures

### Deployment Schedule

- **Staging Deployments**: Any time during business hours
- **Production Deployments**: Tuesday/Thursday 10:00-12:00 EST (avoid Monday/Friday)
- **Emergency Deployments**: Any time with appropriate approvals
- **Maintenance Windows**: First Sunday of each month 2:00-4:00 AM EST

## Emergency Contacts

### Primary Team
- **DevOps Lead**: [Contact Information]
- **ML/AI Engineer**: [Contact Information]
- **Trading System Lead**: [Contact Information]

### Escalation
- **Technical Manager**: [Contact Information]
- **On-Call Engineer**: [On-call rotation system]

### External Dependencies
- **GCP Support**: [Support case system]
- **Third-party API Support**: [Vendor-specific contacts]

### Emergency Communication
- **Slack Channel**: #rlte-alerts
- **Email List**: rlte-ops@company.com
- **Phone Tree**: [Emergency phone tree]

---

## Quick Reference Commands

### Deployment
```bash
# Full automated deployment
./deploy/automated_deployment_pipeline.sh production

# Blue-green deployment only
./deploy/blue_green_deployment.sh production <tag>

# Rollback
./deploy/automated_rollback.sh production previous
```

### Monitoring
```bash
# Health check
curl -s https://your-service-url/health | jq .

# View logs
gcloud run logs tail shyvr-rlte --region us-central1 --follow

# Check metrics
curl -s https://your-service-url/metrics
```

### Troubleshooting
```bash
# Service status
gcloud run services describe shyvr-rlte --region us-central1

# Recent deployments
gcloud run revisions list --service shyvr-rlte --region us-central1

# Database connectivity
./deploy/test_cloud_sql_connectivity.py
```

---

**Document Version**: 1.0  
**Last Updated**: $(date)  
**Review Cycle**: Monthly  
**Next Review**: [Date + 1 month]