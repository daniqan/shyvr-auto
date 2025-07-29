# RL Experience Database Deployment Automation

This directory contains deployment automation scripts for the RL experience storage system, implementing Phase 6.2 of the RL experience storage infrastructure.

## Quick Start

### Deploy to Development Environment
```bash
./deploy/setup_experience_database.sh --environment development --dry-run
./deploy/setup_experience_database.sh --environment development
```

### Deploy to Staging Environment
```bash
./deploy/setup_experience_database.sh --environment staging --dry-run
./deploy/setup_experience_database.sh --environment staging
```

### Deploy to Production Environment
```bash
./deploy/setup_experience_database.sh --environment production --dry-run
./deploy/setup_experience_database.sh --environment production
```

### Validate Existing Deployment
```bash
./deploy/setup_experience_database.sh --environment production --validate-only
```

## Scripts Overview

### `setup_experience_database.sh`
**Purpose**: Comprehensive deployment automation for RL experience database infrastructure

**Features**:
- ✅ Multi-environment support (development, staging, production)
- ✅ Automated Cloud SQL instance creation and configuration
- ✅ Environment-specific database scaling and settings
- ✅ RL experience schema migration and validation
- ✅ Database health checks and monitoring integration
- ✅ Secret management with Google Secret Manager
- ✅ Dry-run capability for safe testing
- ✅ Comprehensive error handling and status reporting

**Environment Configurations**:

| Environment | Instance Tier | Storage Size | Database Name | Instance Name |
|-------------|---------------|--------------|---------------|---------------|
| Development | db-f1-micro   | 10GB         | shyvr_rlte_dev | shyvr-rlte-db-dev |
| Staging     | db-g1-small   | 20GB         | shyvr_rlte_staging | shyvr-rlte-db-staging |
| Production  | db-n1-standard-1 | 100GB     | shyvr_rlte_prod | shyvr-rlte-db-prod |

**Options**:
```
--environment ENV        Environment: development, staging, production
--project-id PROJECT     Google Cloud Project ID 
--region REGION          Cloud SQL region
--instance-name NAME     Custom Cloud SQL instance name
--database-name NAME     Custom database name
--database-user USER     Custom database user
--tier TIER              Custom instance tier
--storage-size SIZE      Custom storage size
--dry-run                Show what would be done without executing
--skip-migration         Skip database schema migration
--skip-monitoring        Skip monitoring setup
--force-recreate         Force recreation of existing resources
--validate-only          Only validate existing setup
--verbose, -v            Enable verbose logging
--help, -h               Show help message
```

## Prerequisites

1. **Google Cloud CLI**: Installed and authenticated
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Required APIs**: The script will enable these automatically
   - Cloud SQL Admin API
   - Cloud Resource Manager API
   - Secret Manager API
   - Cloud Monitoring API
   - Cloud Logging API

3. **Python Environment**: Python 3.x with database utilities
   - Script supports both `python3` and `uv run python` execution

4. **Permissions**: The authenticated user needs:
   - Cloud SQL Admin
   - Secret Manager Admin
   - Monitoring Admin
   - Project Editor (or specific IAM roles)

## Integration with Existing Infrastructure

### Database Schema
The deployment script integrates with the existing database infrastructure:
- Uses `database/migrations/004_create_rl_experience_schema.sql`
- Runs `database/init_database.py` for schema initialization
- Maintains compatibility with existing user and activity logging systems

### Monitoring Integration
- Integrates with `deploy/setup_monitoring.sh` for comprehensive monitoring
- Creates database-specific alerts and metrics
- Supports existing Grafana and Prometheus infrastructure

### Secret Management
The script creates and manages these secrets per environment:
- `postgres-password-{environment}`: Root user password
- `db-password-{environment}`: Application user password
- `database-url-{environment}`: Complete connection string

## Usage Examples

### Basic Deployment
```bash
# Development environment with defaults
./deploy/setup_experience_database.sh --environment development

# Production with custom configuration
./deploy/setup_experience_database.sh \
  --environment production \
  --tier db-n1-standard-2 \
  --storage-size 200GB
```

### Testing and Validation
```bash
# Dry run to see what would be done
./deploy/setup_experience_database.sh --environment staging --dry-run

# Skip migration for existing database
./deploy/setup_experience_database.sh \
  --environment production \
  --skip-migration

# Validate existing deployment
./deploy/setup_experience_database.sh \
  --environment production \
  --validate-only
```

### Troubleshooting
```bash
# Verbose logging for debugging
./deploy/setup_experience_database.sh \
  --environment development \
  --verbose

# Force recreation of resources
./deploy/setup_experience_database.sh \
  --environment development \
  --force-recreate \
  --dry-run
```

## Environment Variables

The script respects these environment variables:

```bash
# Core configuration
export PROJECT_ID="your-gcp-project"
export ENVIRONMENT="production"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Database overrides
export RL_EXPERIENCE_DB_HOST="custom-host"
export RL_EXPERIENCE_DB_PORT="5432"
export RL_EXPERIENCE_DB_NAME="custom_database"
export RL_EXPERIENCE_DB_USER="custom_user"

# Monitoring configuration
export ALERT_EMAIL_CRITICAL="alerts@company.com"
export ALERT_EMAIL_WARNING="warnings@company.com"
```

## Post-Deployment Steps

1. **Update Application Configuration**
   ```yaml
   # config/config.yaml
   rl:
     experience_storage:
       database:
         host: "/cloudsql/CONNECTION_NAME"
         database: "DATABASE_NAME"
         username: "DATABASE_USER"
         # Password loaded from Secret Manager
   ```

2. **Test RL Experience Collection**
   ```bash
   # Run experience lifecycle management
   uv run python scripts/manage_experience_lifecycle.py monitor
   ```

3. **Monitor Database Performance**
   - Check Cloud SQL metrics in GCP Console
   - Monitor Grafana dashboards
   - Review application logs for database connectivity

4. **Configure Automated Lifecycle Management**
   ```bash
   # Test cleanup and archival
   uv run python scripts/manage_experience_lifecycle.py cleanup --dry-run
   ```

## Security Considerations

- All database passwords are auto-generated and stored in Secret Manager
- Cloud SQL instances use deletion protection by default
- Application users have limited database permissions
- Connections use Cloud SQL Auth Proxy for security
- Secrets are environment-specific to prevent cross-environment access

## Monitoring and Alerting

The deployment includes:

### Database Metrics
- Connection count and usage
- Query performance and slow queries
- Storage utilization
- Memory and CPU usage

### RL Experience Metrics
- Experience insertion rate
- Experience retrieval performance
- Training session activity
- Storage growth trends

### Alerts
- High database connections
- Slow query performance
- Storage space warnings
- Failed database operations

## Maintenance

### Regular Tasks
```bash
# Monitor storage usage
uv run python scripts/manage_experience_lifecycle.py monitor

# Run database maintenance
uv run python scripts/manage_experience_lifecycle.py maintenance

# Archive old experiences
uv run python scripts/manage_experience_lifecycle.py archive
```

### Scaling
```bash
# Upgrade instance tier (requires manual confirmation)
gcloud sql instances patch INSTANCE_NAME --tier=db-n1-standard-2

# Increase storage (automatic scaling is enabled by default)
gcloud sql instances patch INSTANCE_NAME --storage-size=200GB
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Permission Denied**
   - Ensure service account has required IAM roles
   - Check project permissions

3. **Instance Already Exists**
   ```bash
   # Use --force-recreate or different instance name
   ./deploy/setup_experience_database.sh --force-recreate
   ```

4. **Migration Failures**
   ```bash
   # Skip migration and run manually
   ./deploy/setup_experience_database.sh --skip-migration
   uv run python database/init_database.py --verbose
   ```

### Getting Help
```bash
# Show detailed help
./deploy/setup_experience_database.sh --help

# Enable verbose logging
./deploy/setup_experience_database.sh --verbose
```

## Integration with CI/CD

The deployment script is designed for integration with automated pipelines:

```yaml
# Example GitHub Actions workflow
- name: Deploy RL Experience Database
  run: |
    ./deploy/setup_experience_database.sh \
      --environment staging \
      --project-id ${{ secrets.GCP_PROJECT_ID }}
  env:
    GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCP_SA_KEY }}
```

For production deployments, always use `--dry-run` first and require manual approval.