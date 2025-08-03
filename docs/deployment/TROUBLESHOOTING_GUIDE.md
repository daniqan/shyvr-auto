# Production Troubleshooting Guide
## Shyvr RLTE AI Trading System

### Overview

This comprehensive troubleshooting guide provides step-by-step procedures for diagnosing and resolving common issues in the Shyvr RLTE production environment. The guide is organized by system component and includes both automated diagnostic tools and manual investigation procedures.

---

## Quick Diagnostic Tools

### System Health Check
```bash
# Run comprehensive health check
./scripts/production_health_check.py

# Quick service status
curl -s https://shyvr-rlte.run.app/health | jq

# Service logs (last 100 lines)
gcloud run logs tail shyvr-rlte --region=us-central1 --limit=100
```

### Performance Quick Check
```bash
# CPU and memory usage
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"

# Request latency
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="latency" --limit=20
```

---

## 1. Service Availability Issues

### Symptoms
- Service returns 5xx errors
- Health check failures
- Connection timeouts
- Service unreachable

### Diagnostic Steps

#### Step 1: Check Service Status
```bash
# Check Cloud Run service status
gcloud run services describe shyvr-rlte --region=us-central1

# Check service endpoints
curl -I https://shyvr-rlte.run.app/health
curl -I https://shyvr-rlte.run.app/api/status
```

#### Step 2: Review Recent Deployments
```bash
# Check recent revisions
gcloud run revisions list --service=shyvr-rlte --region=us-central1 --limit=5

# Check revision status
gcloud run revisions describe REVISION_NAME --region=us-central1
```

#### Step 3: Analyze Error Logs
```bash
# Get error logs from last hour
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="severity>=ERROR" --since="1h"

# Check startup errors
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="startup" --limit=50
```

### Common Resolutions

#### **Container Startup Failures**
```bash
# Check container build
docker build -t shyvr-rlte:debug .
docker run --rm shyvr-rlte:debug python -c "import src; print('Import successful')"

# Verify environment variables
gcloud run services describe shyvr-rlte --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].env)"
```

#### **Resource Constraints**
```bash
# Increase memory and CPU
gcloud run services update shyvr-rlte \
  --memory=8Gi \
  --cpu=4 \
  --max-instances=10 \
  --region=us-central1
```

#### **Configuration Issues**
```bash
# Verify secrets are accessible
./deploy/validate_secrets.py

# Check database connectivity
python ./scripts/test_cloud_sql_connectivity.py
```

---

## 2. Performance Degradation

### Symptoms
- High response times
- Request timeouts
- Memory or CPU alerts
- Slow trading execution

### Diagnostic Steps

#### Step 1: Performance Metrics Analysis
```bash
# Check current resource usage
gcloud monitoring timeseries list \
  --filter='metric.type="run.googleapis.com/container/cpu/utilizations"' \
  --interval.end-time=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --interval.start-time=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)

# Request latency analysis
gcloud monitoring timeseries list \
  --filter='metric.type="run.googleapis.com/request_latencies"' \
  --interval.end-time=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --interval.start-time=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)
```

#### Step 2: Application Performance Profiling
```bash
# Enable profiling endpoint (if available)
curl -s https://shyvr-rlte.run.app/debug/profiling/cpu

# Check slow requests
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="duration>2000" --limit=20
```

#### Step 3: Database Performance Check
```bash
# Check Cloud SQL metrics
gcloud sql operations list --instance=shyvr-rlte-db --limit=10

# Database connection analysis
python ./scripts/analyze_database_performance.py
```

### Common Resolutions

#### **Memory Issues**
```bash
# Check memory usage patterns
gcloud monitoring timeseries list \
  --filter='metric.type="run.googleapis.com/container/memory/utilizations"'

# Increase memory allocation
gcloud run services update shyvr-rlte --memory=16Gi --region=us-central1

# Check for memory leaks
curl -s https://shyvr-rlte.run.app/debug/memory-stats
```

#### **CPU Bottlenecks**
```bash
# Increase CPU allocation
gcloud run services update shyvr-rlte --cpu=8 --region=us-central1

# Check CPU-intensive operations
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="cpu_time" --limit=20
```

#### **Database Performance**
```bash
# Optimize database connections
python ./scripts/optimize_database_performance.py

# Check slow queries
gcloud sql operations list --instance=shyvr-rlte-db --filter="operationType=IMPORT"
```

---

## 3. Trading System Issues

### Symptoms
- Trading orders not executing
- Incorrect position calculations
- Risk management alerts
- Exchange connectivity issues

### Diagnostic Steps

#### Step 1: Trading System Health
```bash
# Check trading service status
curl -s https://shyvr-rlte.run.app/api/trading/status | jq

# Verify exchange connections
curl -s https://shyvr-rlte.run.app/api/exchanges/health | jq

# Check active positions
curl -s https://shyvr-rlte.run.app/api/positions/summary | jq
```

#### Step 2: Exchange API Connectivity
```bash
# Test Hyperliquid connection
python -c "
from src.dex.hyperliquid_client import HyperliquidClient
client = HyperliquidClient()
print(client.test_connection())
"

# Test Jupiter connection
python -c "
from src.dex.jupiter_client import JupiterClient
client = JupiterClient()
print(client.test_connection())
"
```

#### Step 3: Trading Logs Analysis
```bash
# Check trading execution logs
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="trading_execution" --limit=50

# Risk management logs
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="risk_management" --limit=30
```

### Common Resolutions

#### **Exchange API Issues**
```bash
# Verify API keys
./deploy/validate_secrets.py --filter="API_KEY"

# Reset API connections
curl -X POST https://shyvr-rlte.run.app/api/exchanges/reconnect

# Check rate limiting
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="rate_limit" --limit=10
```

#### **Position Synchronization**
```bash
# Force position sync
curl -X POST https://shyvr-rlte.run.app/api/positions/sync

# Verify position accuracy
python ./scripts/verify_position_accuracy.py
```

#### **Risk Management Issues**
```bash
# Check risk thresholds
curl -s https://shyvr-rlte.run.app/api/risk/thresholds | jq

# Emergency risk override (use carefully)
curl -X POST https://shyvr-rlte.run.app/api/risk/emergency-override \
  -H "Content-Type: application/json" \
  -d '{"reason": "production issue", "duration": 300}'
```

---

## 4. ML/RL System Issues

### Symptoms
- Model prediction failures
- Low prediction accuracy
- RL training issues
- Model loading errors

### Diagnostic Steps

#### Step 1: Model System Health
```bash
# Check ML model status
curl -s https://shyvr-rlte.run.app/api/ml/models/status | jq

# RL agent status
curl -s https://shyvr-rlte.run.app/api/rl/agent/status | jq

# Model loading performance
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="model_loading" --limit=20
```

#### Step 2: Model Performance Analysis
```bash
# Check prediction accuracy
curl -s https://shyvr-rlte.run.app/api/ml/performance/accuracy | jq

# RL episode rewards
curl -s https://shyvr-rlte.run.app/api/rl/performance/rewards | jq

# Model memory usage
curl -s https://shyvr-rlte.run.app/debug/ml-memory-usage | jq
```

#### Step 3: Model Storage Verification
```bash
# Check GCS model storage
gsutil ls gs://shyvr-rlte-models-$PROJECT_ID/

# Verify model files
python ./scripts/verify_model_integrity.py

# Check model registry
curl -s https://shyvr-rlte.run.app/api/models/registry | jq
```

### Common Resolutions

#### **Model Loading Issues**
```bash
# Clear model cache
curl -X POST https://shyvr-rlte.run.app/api/ml/cache/clear

# Reload models
curl -X POST https://shyvr-rlte.run.app/api/ml/models/reload

# Check model file integrity
python ./scripts/validate_model_files.py
```

#### **Prediction Performance**
```bash
# Restart ML service components
curl -X POST https://shyvr-rlte.run.app/api/ml/restart

# Update model weights
curl -X POST https://shyvr-rlte.run.app/api/ml/models/update

# Performance optimization
curl -X POST https://shyvr-rlte.run.app/api/ml/optimize
```

#### **RL Training Issues**
```bash
# Check experience replay buffer
curl -s https://shyvr-rlte.run.app/api/rl/experience/status | jq

# Restart RL training
curl -X POST https://shyvr-rlte.run.app/api/rl/training/restart

# Clear corrupted experiences
python ./scripts/clean_experience_buffer.py
```

---

## 5. Database Issues

### Symptoms
- Database connection errors
- Slow query performance
- Transaction timeouts
- Data inconsistencies

### Diagnostic Steps

#### Step 1: Database Connectivity
```bash
# Test database connection
python ./scripts/test_cloud_sql_connectivity.py

# Check connection pool status
curl -s https://shyvr-rlte.run.app/debug/db-pool-status | jq

# Database instance status
gcloud sql instances describe shyvr-rlte-db --format="value(state)"
```

#### Step 2: Performance Analysis
```bash
# Check slow queries
gcloud sql operations list --instance=shyvr-rlte-db --filter="operationType=QUERY" --limit=10

# Database metrics
gcloud monitoring timeseries list \
  --filter='metric.type="cloudsql.googleapis.com/database/cpu/utilization"'

# Connection count
gcloud monitoring timeseries list \
  --filter='metric.type="cloudsql.googleapis.com/database/network/connections"'
```

#### Step 3: Data Integrity Check
```bash
# Run database validation
python ./scripts/validate_database_schema.py

# Check for data corruption
python ./scripts/check_data_integrity.py

# Verify recent migrations
python ./scripts/verify_migration_status.py
```

### Common Resolutions

#### **Connection Issues**
```bash
# Restart database connections
curl -X POST https://shyvr-rlte.run.app/debug/db-restart-connections

# Update connection configuration
gcloud sql instances patch shyvr-rlte-db \
  --database-flags=max_connections=200

# Check firewall rules
gcloud sql instances describe shyvr-rlte-db --format="value(settings.ipConfiguration)"
```

#### **Performance Optimization**
```bash
# Optimize database performance
python ./scripts/optimize_database_performance.py

# Update statistics
gcloud sql operations create \
  --instance=shyvr-rlte-db \
  --type=ANALYZE_TABLE
```

#### **Data Recovery**
```bash
# Create backup
gcloud sql backups create \
  --instance=shyvr-rlte-db \
  --description="Emergency backup $(date)"

# Restore from backup (if needed)
gcloud sql backups restore BACKUP_ID \
  --restore-instance=shyvr-rlte-db
```

---

## 6. Security Issues

### Symptoms
- Authentication failures
- Unauthorized access attempts
- API key issues
- Certificate problems

### Diagnostic Steps

#### Step 1: Security Status Check
```bash
# Check authentication system
curl -s https://shyvr-rlte.run.app/api/auth/status | jq

# Verify SSL certificate
curl -I https://shyvr-rlte.run.app/

# Check secret manager access
gcloud secrets list --filter="name:shyvr-rlte"
```

#### Step 2: Access Log Analysis
```bash
# Check unauthorized access attempts
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="401 OR 403" --limit=50

# API key usage analysis
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="api_key" --limit=30

# Security alert logs
gcloud run logs tail shyvr-rlte --region=us-central1 --filter="security_alert" --limit=20
```

#### Step 3: Compliance Verification
```bash
# Run security audit
python ./scripts/security_audit.py

# Check compliance status
curl -s https://shyvr-rlte.run.app/api/compliance/status | jq

# Verify data encryption
python ./scripts/verify_encryption.py
```

### Common Resolutions

#### **Authentication Issues**
```bash
# Regenerate API keys
./deploy/rotate_secrets.py --keys="API_KEYS"

# Reset authentication system
curl -X POST https://shyvr-rlte.run.app/api/auth/reset

# Update secret manager
gcloud secrets versions add SECRET_NAME --data-file=new_secret.txt
```

#### **Certificate Problems**
```bash
# Check certificate expiry
curl -s https://shyvr-rlte.run.app/ | openssl x509 -noout -dates

# Update SSL configuration
gcloud run services update shyvr-rlte \
  --region=us-central1 \
  --update-env-vars SSL_CERT_PATH=/etc/ssl/certs
```

---

## 7. Monitoring & Alerting Issues

### Symptoms
- Missing alerts
- False positive alerts
- Dashboard loading issues
- Metric collection problems

### Diagnostic Steps

#### Step 1: Monitoring System Check
```bash
# Check monitoring service
gcloud services list --enabled --filter="name:monitoring.googleapis.com"

# Verify alert policies
gcloud alpha monitoring policies list --filter="enabled=true"

# Check notification channels
gcloud alpha monitoring channels list
```

#### Step 2: Metric Collection Analysis
```bash
# Check custom metrics
gcloud monitoring metrics list --filter="metric.type:custom.googleapis.com"

# Verify metric ingestion
gcloud monitoring timeseries list \
  --filter='metric.type="custom.googleapis.com/trading/execution_latency"' \
  --interval.end-time=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --interval.start-time=$(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%SZ)
```

#### Step 3: Dashboard Verification
```bash
# Check dashboard configuration
gcloud monitoring dashboards list

# Verify dashboard access
curl -s "https://monitoring.googleapis.com/v1/projects/$PROJECT_ID/dashboards"
```

### Common Resolutions

#### **Alert Configuration**
```bash
# Recreate alert policies
./deploy/sla_monitoring_setup.sh

# Test notification channels
gcloud alpha monitoring channels verify CHANNEL_ID

# Update alert thresholds
python ./scripts/update_alert_thresholds.py
```

#### **Metric Collection**
```bash
# Restart metric collection
curl -X POST https://shyvr-rlte.run.app/debug/restart-metrics

# Verify metric format
python ./scripts/validate_custom_metrics.py

# Clear metric cache
curl -X POST https://shyvr-rlte.run.app/debug/clear-metric-cache
```

---

## 8. Emergency Procedures

### Complete System Recovery

#### **Emergency Stop Sequence**
```bash
# 1. Stop all trading
curl -X POST https://shyvr-rlte.run.app/api/emergency-stop

# 2. Verify all positions are safe
./scripts/verify_position_safety.sh

# 3. Scale service to minimum
gcloud run services update shyvr-rlte --max-instances=1 --region=us-central1

# 4. Enable maintenance mode
curl -X POST https://shyvr-rlte.run.app/api/maintenance-mode/enable
```

#### **Service Restart Sequence**
```bash
# 1. Deploy latest known good version
./deploy/deploy_latest.sh --version=KNOWN_GOOD_VERSION

# 2. Validate deployment
./deploy/validate_deployment.sh

# 3. Gradually restore traffic
./deploy/gradual_traffic_restoration.sh

# 4. Resume trading (manual approval required)
curl -X POST https://shyvr-rlte.run.app/api/resume-trading
```

### Disaster Recovery

#### **Complete Infrastructure Recovery**
```bash
# 1. Deploy to backup region
./deploy/disaster_recovery_deployment.sh --region=us-east1

# 2. Restore database from backup
./scripts/restore_database_backup.sh --backup-id=LATEST

# 3. Sync model storage
./scripts/sync_model_storage.sh --source=primary --target=backup

# 4. Update DNS routing
./scripts/update_dns_routing.sh --region=us-east1
```

---

## 9. Preventive Maintenance

### Daily Checks
```bash
# Run daily health check
./scripts/daily_health_check.sh

# Check system performance
./scripts/performance_check.sh --comprehensive

# Verify backup integrity
./scripts/verify_backup_integrity.sh
```

### Weekly Maintenance
```bash
# Database maintenance
./scripts/weekly_database_maintenance.sh

# Model performance review
./scripts/weekly_model_review.sh

# Security audit
./scripts/weekly_security_audit.sh
```

### Monthly Tasks
```bash
# Comprehensive system audit
./scripts/monthly_system_audit.sh

# Performance optimization
./scripts/monthly_optimization.sh

# Documentation updates
./scripts/update_documentation.sh
```

---

## 10. Escalation Procedures

### Internal Escalation
1. **Level 1**: On-call Engineer (0-15 minutes)
2. **Level 2**: Technical Lead (15-30 minutes)
3. **Level 3**: Operations Manager (30-60 minutes)
4. **Level 4**: CTO/Executive (1+ hours)

### External Escalation
1. **Cloud Provider Support**: For infrastructure issues
2. **Exchange Support**: For trading connectivity issues
3. **Security Vendor**: For security incidents
4. **Regulatory Bodies**: For compliance issues

### Emergency Contacts
- **On-call Engineer**: +1-xxx-xxx-xxxx
- **Technical Lead**: +1-xxx-xxx-xxxx
- **Operations Manager**: +1-xxx-xxx-xxxx
- **Security Team**: security@shyvr.ai
- **Legal/Compliance**: compliance@shyvr.ai

---

**Last Updated**: August 3, 2025  
**Document Version**: 1.0  
**Next Review**: November 3, 2025