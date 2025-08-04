# Infrastructure Setup Orchestrator

The `setup-infrastructure.sh` script orchestrates all infrastructure setup for Shyvr RLTE, ensuring proper dependency order and robust pre-deployment checks.

## Overview

This script coordinates the following infrastructure components in order:

1. **Enable APIs** - Required Google Cloud APIs
2. **Service Account** - Creates and configures service accounts with proper IAM roles
3. **Secret Manager** - Sets up all required secrets and API keys
4. **Cloud SQL** - Creates PostgreSQL databases with proper configuration
5. **GCS Infrastructure** - Storage buckets for model preservation
6. **Monitoring** - Cloud monitoring and alerting setup

## Usage

### Basic Usage

```bash
# Validate existing infrastructure
./deploy/setup-infrastructure.sh --mode validate --environment production

# Setup missing infrastructure
./deploy/setup-infrastructure.sh --mode setup-if-needed --environment staging

# Full setup with dry run first
./deploy/setup-infrastructure.sh --mode setup --environment production --dry-run
./deploy/setup-infrastructure.sh --mode setup --environment production
```

### Integration with Deployment Pipeline

The deployment pipeline automatically calls this script as Stage 0:

- **Staging Environment**: Uses `setup-if-needed` mode (creates missing components)
- **Production Environment**: Uses `validate` mode (fails if components missing)

```bash
# Skip infrastructure setup in deployment pipeline
SKIP_INFRASTRUCTURE=true ./deploy/automated_deployment_pipeline.sh staging

# Normal deployment (infrastructure setup included)
./deploy/automated_deployment_pipeline.sh staging
```

## Modes

### validate
- Only validates existing infrastructure
- Fails if any required components are missing or misconfigured
- Recommended for production deployments
- No changes made to infrastructure

### setup
- Creates all infrastructure components
- Updates existing components with new configuration
- Use when you want to ensure all components are properly configured

### setup-if-needed
- Creates only missing infrastructure components
- Safest mode for existing environments
- Recommended for staging deployments
- Minimizes changes to existing infrastructure

### force-recreate
- **DANGEROUS**: Deletes and recreates all infrastructure
- Requires explicit confirmation for production
- Use only when you need to completely rebuild infrastructure

## Environment-Specific Behavior

### Staging
- Reduced resource allocations (smaller database tiers, fewer replicas)
- More permissive settings for development
- Automatic infrastructure creation when missing

### Production
- Full resource allocations with high availability
- Strict validation requirements
- Deletion protection on critical resources
- Infrastructure validation only (no automatic creation)

### Development
- Minimal resource allocations
- Basic feature set
- Suitable for local development and testing

## Dry Run Mode

Always test with `--dry-run` first to see what changes would be made:

```bash
./deploy/setup-infrastructure.sh --mode setup --environment production --dry-run
```

## Error Handling

The script uses robust error handling:

- **Critical Failures**: APIs, Service Account, Secrets (for validate mode)
- **Non-Critical Failures**: Monitoring setup
- **Automatic Rollback**: Not implemented (too dangerous for infrastructure)
- **Detailed Logging**: Complete audit trail of all operations

## Dependencies

Each stage depends on previous stages:

1. APIs must be enabled first
2. Service Account needs APIs
3. Secrets need Service Account permissions
4. Cloud SQL needs Secrets for passwords
5. GCS needs Service Account permissions
6. Monitoring needs all previous components

## Output

The script provides:

- **Progress Tracking**: Real-time status of each stage
- **Detailed Summary**: Complete report of what was created/validated
- **Integration Commands**: Next steps for deployment pipeline
- **Troubleshooting**: Clear error messages and recovery steps

## Security

- Service accounts follow principle of least privilege
- Secrets are stored in Google Secret Manager
- No hardcoded credentials in scripts
- Workload Identity preferred over service account keys
- Production safety checks for destructive operations

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have proper GCP IAM roles
2. **API Not Enabled**: Script will enable required APIs automatically
3. **Secrets Missing**: Use `setup-if-needed` mode to create missing secrets
4. **Resource Conflicts**: Use validate mode to check existing resources

### Recovery

If infrastructure setup fails:

1. Check the detailed error messages
2. Fix the underlying issue
3. Re-run the script (it's idempotent)
4. Use validate mode to verify fixes

### Manual Cleanup

If you need to clean up infrastructure manually:

```bash
# Delete secrets (be careful!)
gcloud secrets delete SECRET_NAME --project=PROJECT_ID

# Delete service accounts
gcloud iam service-accounts delete EMAIL --project=PROJECT_ID

# Note: Cloud SQL deletion requires manual confirmation for safety
```

## Integration

This orchestrator integrates with:

- **Deployment Pipeline**: Automatic infrastructure validation/setup
- **Individual Setup Scripts**: Calls each component's setup script
- **Monitoring**: Integrated with deployment logging and metrics
- **CI/CD**: Can be called from build systems and automation

The infrastructure setup is designed to be robust, safe, and easy to use across different environments and deployment scenarios.