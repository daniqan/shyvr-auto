# GCS Infrastructure Setup Guide

This guide provides step-by-step instructions for setting up Google Cloud Storage infrastructure for the Shyvr RLTE model preservation system.

## Overview

The GCS infrastructure consists of:
- **Production bucket** (`shyvr-models-prod`): For production model storage
- **Staging bucket** (`shyvr-models-staging`): For testing and staging
- **Lifecycle policies**: For cost optimization and data management
- **IAM permissions**: For secure access control
- **Audit logging**: For compliance and monitoring
- **VPC Service Controls**: For additional security (optional)

## Prerequisites

1. **Google Cloud CLI**: Install and authenticate with `gcloud`
2. **Project access**: Must have Owner or Editor role on the GCP project
3. **APIs enabled**: The setup script will enable required APIs automatically
4. **Service account**: Cloud Run service account must exist

## Quick Start

### 1. Set Environment Variables

```bash
export GCP_PROJECT_ID="your-project-id"
export CLOUD_RUN_SA="shyvr-rlte@your-project-id.iam.gserviceaccount.com"
export GCP_REGION="us-central1"  # Optional, defaults to us-central1
export VPC_NAME="shyvr-vpc"      # Optional, for VPC Service Controls
```

### 2. Run Setup Script

```bash
cd deploy
./setup_gcs_infrastructure.sh
```

### 3. Validate Setup

```bash
./validate_gcs_setup.py
```

## Detailed Setup Steps

### Step 1: Prepare Environment

1. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Verify permissions**:
   ```bash
   gcloud projects get-iam-policy YOUR_PROJECT_ID --flatten="bindings[].members" --filter="bindings.members:YOUR_EMAIL"
   ```

### Step 2: Run Infrastructure Setup

The setup script performs the following operations:

1. **Enable required APIs**:
   - Cloud Storage API
   - Cloud Logging API
   - Cloud Monitoring API
   - Cloud Asset API
   - Access Context Manager API

2. **Create GCS buckets**:
   - Production bucket with STANDARD storage class
   - Staging bucket with NEARLINE storage class
   - Both in specified region (default: us-central1)

3. **Configure bucket settings**:
   - Enable versioning on both buckets
   - Apply lifecycle policies for cost optimization
   - Enable uniform bucket-level access
   - Enforce public access prevention

4. **Set up IAM permissions**:
   - Grant Storage Object Admin role to Cloud Run service account
   - Scope permissions to specific buckets only
   - Grant Legacy Bucket Reader for listing operations

5. **Configure security**:
   - Set up VPC Service Controls (if organization available)
   - Enable audit logging for all storage operations
   - Configure access policies

### Step 3: Verify Configuration

The validation script checks:

- ✅ Bucket existence and accessibility
- ✅ Versioning enabled
- ✅ Lifecycle policies applied
- ✅ Storage classes configured
- ✅ Security settings (uniform access, public access prevention)
- ✅ IAM permissions working
- ✅ Read/write/delete operations
- ✅ Encryption configuration

## Configuration Details

### Lifecycle Policy

The lifecycle policy (`lifecycle.json`) includes:

1. **Transition to NEARLINE**: After 30 days
2. **Transition to COLDLINE**: After 90 days
3. **Transition to ARCHIVE**: After 365 days
4. **Delete archived objects**: After 7 years (2555 days)
5. **Version management**: Keep only 10 newest versions
6. **Cleanup non-current versions**: Delete after 7 days
7. **Abort incomplete uploads**: After 1 day

### IAM Permissions

The Cloud Run service account receives:

- `roles/storage.objectAdmin` with bucket-specific conditions
- `roles/storage.legacyBucketReader` for bucket listing

Conditions limit access to only the model storage buckets.

### Security Features

- **Uniform bucket-level access**: Simplifies permission management
- **Public access prevention**: Prevents accidental public exposure
- **Audit logging**: Tracks all access and modifications
- **VPC Service Controls**: Additional network-level security (optional)

## Usage in Application

### Environment Variables

Set these in your Cloud Run service:

```bash
GCS_MODEL_BUCKET=shyvr-models-prod
GCS_STAGING_BUCKET=shyvr-models-staging
MODEL_PRESERVATION_ENABLED=true
MODEL_CACHE_DIR=/app/models/cache
```

### Python Code Example

```python
from src.model_preservation.manager import PreservationManager

# Initialize with GCS bucket
manager = PreservationManager(
    gcs_bucket_name="shyvr-models-prod",
    enable_db_handler=True
)

# Save model
await manager.save_model(
    model=my_model,
    model_type="dqn_agent",
    version="1.0.0",
    metadata={"accuracy": 0.95}
)

# Load model
model = await manager.load_model(
    model_type="dqn_agent",
    version="latest"
)
```

## Monitoring and Alerting

### Metrics to Monitor

- Storage usage and costs
- Upload/download latencies
- Error rates and failures
- Permission violations
- Lifecycle policy effectiveness

### Recommended Alerts

1. **Storage costs**: Alert when monthly costs exceed threshold
2. **Access errors**: Alert on repeated permission failures
3. **Lifecycle failures**: Alert when policies fail to execute
4. **Upload failures**: Alert on model save failures

## Troubleshooting

### Common Issues

#### 1. Permission Denied Errors

**Symptoms**: `403 Forbidden` errors when accessing buckets

**Solutions**:
- Verify service account has correct roles
- Check IAM policy bindings
- Ensure service account is being used by Cloud Run
- Validate bucket-specific conditions in IAM

#### 2. Bucket Not Found

**Symptoms**: `404 Not Found` errors

**Solutions**:
- Verify bucket names in environment variables
- Check if buckets exist in correct project
- Ensure project ID is set correctly

#### 3. Lifecycle Policy Issues

**Symptoms**: Objects not transitioning storage classes

**Solutions**:
- Verify lifecycle policy syntax
- Check policy application with `gsutil lifecycle get`
- Wait for policy evaluation (can take 24 hours)

#### 4. Version Limit Not Working

**Symptoms**: More than 10 versions of objects

**Solutions**:
- Check lifecycle rule for `numNewerVersions`
- Verify versioning is enabled on bucket
- Wait for lifecycle evaluation

### Debugging Commands

```bash
# Check bucket configuration
gsutil ls -L -b gs://shyvr-models-prod

# Verify lifecycle policy
gsutil lifecycle get gs://shyvr-models-prod

# Test permissions
gsutil -i SERVICE_ACCOUNT cp test.txt gs://shyvr-models-prod/

# Check IAM policies
gcloud projects get-iam-policy PROJECT_ID

# View audit logs
gcloud logging read "resource.type=gcs_bucket" --limit=10
```

## Cost Optimization

### Storage Class Strategy

- **STANDARD**: Active models (first 30 days)
- **NEARLINE**: Recent models (30-90 days)
- **COLDLINE**: Older models (90-365 days)
- **ARCHIVE**: Long-term retention (1-7 years)

### Cost Monitoring

1. Set up billing alerts
2. Monitor storage class distribution
3. Review lifecycle policy effectiveness
4. Clean up unnecessary versions

### Best Practices

- Use compression for model storage
- Implement proper version management
- Regular cleanup of test/temporary objects
- Monitor access patterns for optimization

## Security Best Practices

1. **Principle of least privilege**: Grant minimal required access
2. **Service account keys**: Avoid downloading and storing keys
3. **Audit regularly**: Review access logs and permissions
4. **Network security**: Use VPC Service Controls for sensitive data
5. **Encryption**: Enable customer-managed encryption keys if required

## Disaster Recovery

### Backup Strategy

- Cross-region replication for critical models
- Regular testing of restore procedures
- Documentation of recovery steps
- Automated backup verification

### Recovery Procedures

1. **Bucket deletion**: Restore from cross-region backup
2. **Data corruption**: Use object versioning to restore
3. **Permission issues**: Use Terraform/scripts to recreate IAM
4. **Regional outage**: Failover to backup region

## Next Steps

After successful setup:

1. **Deploy application**: Update Cloud Run with GCS configuration
2. **Test model preservation**: Run end-to-end tests
3. **Set up monitoring**: Configure alerts and dashboards
4. **Documentation**: Update team documentation
5. **Training**: Train team on new procedures

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review validation script output
3. Check Google Cloud console for detailed errors
4. Contact DevOps team for infrastructure issues

## File Reference

- `setup_gcs_infrastructure.sh`: Main setup script
- `lifecycle.json`: Bucket lifecycle policy configuration
- `validate_gcs_setup.py`: Validation and testing script
- `README_GCS_SETUP.md`: This documentation file

---

**Last Updated**: 2025-07-31  
**Version**: 1.0  
**Phase**: 1.1 - GCS Infrastructure Setup