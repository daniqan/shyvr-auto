# Shyvr RLTE Deployment

This directory contains deployment scripts for the Shyvr AI Reinforcement Learning Trading Engine (RLTE).

## 📚 Complete Documentation

All comprehensive deployment documentation has been moved to:
**[`docs/deployment/`](../docs/deployment/)**

### Quick Start

For complete deployment instructions, see:
- **[Comprehensive Deployment Tutorial](../docs/deployment/COMPREHENSIVE_DEPLOYMENT_TUTORIAL.md)** - Step-by-step deployment from scratch
- **[Deployment README](../docs/deployment/README.md)** - Complete deployment workflow and script documentation

### Key Documentation

- **[Production Usage Guide](../docs/deployment/PRODUCTION_USAGE_GUIDE.md)** - Complete system operation manual
- **[Dashboard Access Guide](../docs/deployment/DASHBOARD_ACCESS_OPERATION_GUIDE.md)** - Dashboard usage and monitoring
- **[Deployment Readiness Checklist](../docs/deployment/DEPLOYMENT_READINESS_CHECKLIST.md)** - Pre-deployment validation
- **[Monitoring System](../docs/deployment/MONITORING_SYSTEM.md)** - Monitoring and alerting setup

## 🚀 Quick Deployment

```bash
# 1. Pre-deployment validation
./scripts/run_all_validations.sh

# 2. Complete deployment
./deploy_and_configure.sh

# 3. Post-deployment verification  
./validate_deployment.sh
```

## 📁 Deployment Scripts

This directory contains the following deployment scripts:

- `deploy_and_configure.sh` - Main deployment orchestrator
- `deploy_latest.sh` - Core production deployment script  
- `setup_cloud_sql.sh` - Database infrastructure setup
- `setup_secrets.sh` - API keys and secrets management
- `setup_monitoring.sh` - Monitoring and alerting setup
- `setup_build_triggers.sh` - CI/CD pipeline configuration
- `validate_deployment.sh` - Post-deployment health checks
- `validate_secrets.py` - Secret validation utility

For detailed documentation of each script, see [docs/deployment/README.md](../docs/deployment/README.md).