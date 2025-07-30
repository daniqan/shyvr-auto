# Shyvr RLTE - Complete Deployment Guide

This directory contains comprehensive deployment automation for the Shyvr AI-augmented cryptocurrency trading bot with RL capabilities. This guide provides everything needed to deploy and maintain the fully operational application in production, staging, and development environments.

## 🎯 Quick Deployment

### Complete Application Deployment
```bash
# Deploy entire application with database and monitoring
./deploy/deploy_and_configure.sh

# OR step-by-step deployment
./deploy/deploy_latest.sh           # Deploy application to Cloud Run
./deploy/validate_deployment.sh     # Validate deployment health
```

### Database and Infrastructure Setup
```bash
# Deploy RL experience database
./deploy/setup_experience_database.sh --environment production --dry-run
./deploy/setup_experience_database.sh --environment production

# Setup monitoring and alerting
./deploy/setup_monitoring.sh --project-id YOUR_PROJECT_ID --alert-email alerts@company.com
```

---

## 📋 Deployment Scripts Overview

### Core Deployment Scripts

#### `deploy_and_configure.sh`
**Purpose**: Complete deployment pipeline - deploys application and configures Telegram webhook
- ✅ Automated Cloud Run deployment
- ✅ Telegram webhook configuration
- ✅ Service URL retrieval and configuration
- ✅ End-to-end deployment validation

#### `deploy_latest.sh`
**Purpose**: Production-ready application deployment with advanced error handling
- ✅ Multi-stage Docker builds and optimization
- ✅ Rollback capabilities with traffic allocation
- ✅ Comprehensive health checks and validation
- ✅ Deployment logging and tracking
- ✅ Container registry management
- ✅ Cloud Run service configuration

#### `setup_experience_database.sh`
**Purpose**: RL experience database infrastructure deployment
- ✅ Multi-environment Cloud SQL deployment (dev/staging/prod)
- ✅ Automated schema migration and validation
- ✅ Secret management integration
- ✅ Database monitoring setup
- ✅ Performance optimization and scaling

#### `setup_monitoring.sh`
**Purpose**: Comprehensive monitoring and alerting infrastructure
- ✅ GCP monitoring and logging setup
- ✅ Grafana dashboard deployment
- ✅ Custom metrics and alerts configuration
- ✅ Performance monitoring for trading operations
- ✅ Database and application health monitoring

#### `validate_deployment.sh`
**Purpose**: Post-deployment validation and health checks
- ✅ Service availability and response validation
- ✅ Database connectivity testing
- ✅ RL experience storage validation
- ✅ API endpoint health checks
- ✅ Telegram webhook validation

### Utility Scripts

#### `setup_secrets.sh`
- Secret management and rotation
- Environment-specific secret configuration
- Service account key management

#### `setup_build_triggers.sh`
- Cloud Build CI/CD pipeline setup
- Automated deployment triggers
- Build notification configuration

#### `rotate_secrets.py`
- Automated secret rotation
- Security compliance maintenance
- Database credential updates

---

## 🚀 Manual Deployment Workflow

### Pre-Deployment Checklist

1. **Environment Setup**
   ```bash
   # Authenticate with Google Cloud
   gcloud auth login
   gcloud config set project shvyr-ai-bots
   
   # Set environment variables
   export PROJECT_ID="shvyr-ai-bots"
   export ENVIRONMENT="production"  # or staging, development
   export ALERT_EMAIL="alerts@company.com"
   ```

2. **Dependencies Verification**
   ```bash
   # Verify Docker and gcloud are installed
   docker --version
   gcloud --version
   
   # Check project permissions
   gcloud projects get-iam-policy $PROJECT_ID
   ```

### Deployment Execution Order

#### Step 1: Database Infrastructure (if not exists)
```bash
# Deploy database with dry-run first
./deploy/setup_experience_database.sh --environment $ENVIRONMENT --dry-run

# Execute database deployment
./deploy/setup_experience_database.sh --environment $ENVIRONMENT --verbose
```

#### Step 2: Secrets and Configuration
```bash
# Setup application secrets
./deploy/setup_secrets.sh --environment $ENVIRONMENT

# Validate secret configuration
./deploy/validate_secrets.py --environment $ENVIRONMENT
```

#### Step 3: Monitoring Infrastructure
```bash
# Deploy monitoring stack
./deploy/setup_monitoring.sh \
  --project-id $PROJECT_ID \
  --environment $ENVIRONMENT \
  --alert-email $ALERT_EMAIL
```

#### Step 4: Application Deployment
```bash
# Deploy application to Cloud Run
./deploy/deploy_latest.sh

# Or use complete pipeline
./deploy/deploy_and_configure.sh
```

#### Step 5: Post-Deployment Validation
```bash
# Comprehensive deployment validation
./deploy/validate_deployment.sh

# Test database connectivity
./deploy/test_cloud_sql_connectivity.py --environment $ENVIRONMENT
```

### Environment-Specific Configurations

| Environment | Cloud Run CPU | Memory | Database Tier | Storage | Monitoring |
|-------------|---------------|--------|---------------|---------|------------|
| Development | 1 CPU        | 2Gi    | db-f1-micro   | 10GB    | Basic      |
| Staging     | 2 CPU        | 4Gi    | db-g1-small   | 20GB    | Standard   |
| Production  | 4 CPU        | 8Gi    | db-n1-standard-1 | 100GB | Advanced   |

---

## 🔧 Pre and Post Deployment Procedures

### Pre-Deployment Requirements

#### System Requirements
- Google Cloud CLI (gcloud) installed and authenticated
- Docker installed and running
- Python 3.8+ with uv package manager
- Network access to Google Cloud APIs

#### Required APIs
```bash
# APIs enabled automatically by deployment scripts
- Cloud Run API
- Cloud SQL Admin API
- Secret Manager API
- Cloud Build API
- Cloud Monitoring API
- Cloud Logging API
- Container Registry API
```

#### Required Permissions
- Cloud Run Admin
- Cloud SQL Admin
- Secret Manager Admin
- Service Account Admin
- Monitoring Admin
- Project Editor (recommended) or granular IAM roles

#### Configuration Files
- `config/config.yaml` - Application configuration
- `database/migrations/` - Database schema files
- `monitoring/grafana/dashboard.json` - Grafana dashboards

### Post-Deployment Validation

#### Automated Health Checks
```bash
# Run comprehensive validation suite
./deploy/validate_deployment.sh

# Check specific components
curl -f https://YOUR_SERVICE_URL/health
curl -f https://YOUR_SERVICE_URL/dashboard
```

#### Manual Verification Steps
1. **Service Accessibility**
   - Verify Cloud Run service is receiving traffic
   - Test Telegram bot responsiveness
   - Validate dashboard accessibility

2. **Database Connectivity**
   - Confirm database connections are established
   - Verify RL experience storage is functional
   - Test query performance

3. **Monitoring Setup**
   - Check Grafana dashboards are populated
   - Verify alerting rules are active
   - Test notification channels

#### Configuration Updates
```bash
# Update application environment variables if needed
gcloud run services update shyvr-rlte \
  --region us-central1 \
  --set-env-vars="LOG_LEVEL=INFO"

# Update secrets
gcloud secrets versions add TELEGRAM_TOKEN --data-file=token.txt
```

---

## 📊 Dashboard Access Instructions

### Grafana Dashboard
```bash
# Dashboard is deployed as part of monitoring setup
# Access URL: https://YOUR_DOMAIN/grafana
# Default credentials managed via Secret Manager

# To get dashboard URL:
gcloud run services describe grafana-service \
  --region us-central1 \
  --format="value(status.url)"
```

### Application Dashboard
```bash
# Built-in web dashboard at /dashboard endpoint
# URL: https://YOUR_SERVICE_URL/dashboard

# Features:
- Real-time trading metrics
- RL agent performance
- Portfolio status
- System health monitoring
- Database metrics
```

### Cloud Console Monitoring
- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud SQL**: https://console.cloud.google.com/sql
- **Monitoring**: https://console.cloud.google.com/monitoring
- **Logs**: https://console.cloud.google.com/logs

---

## 🏭 Production Operation Guidelines

### Service Management
```bash
# View service status
gcloud run services describe shyvr-rlte --region us-central1

# Scale service
gcloud run services update shyvr-rlte \
  --region us-central1 \
  --min-instances 1 \
  --max-instances 10

# Update service
./deploy/deploy_latest.sh  # Automated rollback on failure
```

### Database Operations
```bash
# Monitor database performance
gcloud sql operations list --instance shyvr-rlte-db-prod

# Database maintenance
uv run python scripts/manage_experience_lifecycle.py maintenance

# Backup database
gcloud sql backups create --instance shyvr-rlte-db-prod
```

### Secret Management
```bash
# List secrets
gcloud secrets list

# Update secret
./deploy/rotate_secrets.py --environment production --secret TELEGRAM_TOKEN

# Validate secrets
./deploy/validate_secrets.py --environment production
```

### Log Management
```bash
# View application logs
gcloud logs read "resource.type=cloud_run_revision" --limit 100

# View database logs
gcloud logs read "resource.type=cloud_sql_database" --limit 100

# Export logs for analysis
gcloud logs read --format json > application_logs.json
```

---

## 📈 Monitoring and Maintenance Procedures

### Daily Monitoring Tasks
```bash
# Check system health
./deploy/validate_deployment.sh

# Monitor RL experience storage
uv run python scripts/manage_experience_lifecycle.py monitor

# Review critical alerts
gcloud alpha monitoring policies list
```

### Weekly Maintenance
```bash
# Database maintenance and optimization
uv run python scripts/manage_experience_lifecycle.py maintenance

# Clean up old RL experiences
uv run python scripts/manage_experience_lifecycle.py cleanup

# Review and archive logs
gcloud logging sinks list
```

### Monthly Tasks
```bash
# Rotate secrets
./deploy/rotate_secrets.py --environment production

# Update dependencies and redeploy
./deploy/deploy_latest.sh

# Database performance review
gcloud sql operations list --instance shyvr-rlte-db-prod --limit 100
```

### Performance Monitoring

#### Key Metrics to Monitor
- **Response Time**: <2s for API endpoints
- **Database Connections**: <80% of max connections
- **Memory Usage**: <85% of allocated memory
- **RL Training Performance**: Model convergence metrics
- **Trading Accuracy**: Win rate and profit metrics

#### Alert Thresholds
```yaml
# Critical Alerts
- Service downtime > 1 minute
- Database connections > 90%
- Memory usage > 95%
- Error rate > 5%

# Warning Alerts
- Response time > 3s
- Database connections > 80%
- Memory usage > 85%
- Disk usage > 80%
```

### Troubleshooting Common Issues

#### Service Not Responding
```bash
# Check service logs
gcloud logs read "resource.type=cloud_run_revision" --limit 50

# Restart service (force new revision)
gcloud run services update shyvr-rlte --region us-central1 --tag=restart

# Rollback if needed
gcloud run services update-traffic shyvr-rlte --to-revisions=REVISION-NAME=100
```

#### Database Connectivity Issues
```bash
# Test database connection
./deploy/test_cloud_sql_connectivity.py --environment production

# Check Cloud SQL status
gcloud sql instances describe shyvr-rlte-db-prod

# Restart database if needed
gcloud sql instances restart shyvr-rlte-db-prod
```

#### High Memory Usage
```bash
# Scale up instance
gcloud run services update shyvr-rlte \
  --region us-central1 \
  --memory 16Gi

# Monitor memory patterns
gcloud monitoring time-series list --filter='resource.type="cloud_run_revision"'
```

---

## 🔐 Security and Compliance

### Security Best Practices
- All secrets stored in Google Secret Manager
- Database connections use Cloud SQL Auth Proxy
- Container images scanned for vulnerabilities
- HTTPS-only communication enforced
- Service accounts follow principle of least privilege

### Backup and Recovery
```bash
# Automated database backups (configured in setup)
gcloud sql backups list --instance shyvr-rlte-db-prod

# Manual backup
gcloud sql backups create --instance shyvr-rlte-db-prod

# Restore from backup
gcloud sql backups restore BACKUP_ID --restore-instance shyvr-rlte-db-prod
```

### Disaster Recovery
- Multi-region deployment capabilities
- Automated failover procedures
- Data replication strategies
- Recovery time objectives (RTO): <30 minutes
- Recovery point objectives (RPO): <1 hour

---

## 🛠️ Development and Staging Workflows

### Development Environment
```bash
# Deploy to development
export ENVIRONMENT="development"
./deploy/setup_experience_database.sh --environment development
./deploy/deploy_latest.sh  # Uses development configuration
```

### Staging Deployment
```bash
# Deploy to staging for testing
export ENVIRONMENT="staging"
./deploy/deploy_and_configure.sh

# Run integration tests
./deploy/validate_deployment.sh
```

### Production Promotion
```bash
# Promote staging to production
export ENVIRONMENT="production"
./deploy/deploy_latest.sh --dry-run  # Always dry-run first
./deploy/deploy_latest.sh            # Deploy to production
```

---

## 📞 Support and Troubleshooting

### Getting Help
```bash
# Script help and documentation
./deploy/deploy_latest.sh --help
./deploy/setup_experience_database.sh --help
./deploy/setup_monitoring.sh --help
```

### Log Locations
- **Application Logs**: Cloud Logging (cloud_run_revision)
- **Database Logs**: Cloud Logging (cloud_sql_database)
- **Deployment Logs**: Local `deployment_*.log` files
- **Build Logs**: Cloud Build history

### Emergency Procedures
```bash
# Emergency rollback
gcloud run services update-traffic shyvr-rlte --to-revisions=PREVIOUS-REVISION=100

# Emergency shutdown
gcloud run services update shyvr-rlte --region us-central1 --min-instances 0

# Emergency database read-only mode
gcloud sql instances patch shyvr-rlte-db-prod --database-flags=default_transaction_read_only=on
```

### Contact Information
- **System Alerts**: Configured via `--alert-email` parameter
- **Documentation**: `docs/` directory in project root
- **Issue Tracking**: GitHub Issues (if applicable)

This comprehensive deployment guide ensures reliable, scalable, and maintainable operation of the Shyvr RLTE trading system across all environments.