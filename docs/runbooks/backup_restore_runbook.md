# Backup and Restore Operations Runbook

## Overview

This runbook covers daily backup and restore operations for the ShyvRAI-RLTE production environment. It includes procedures for scheduled backups, manual backups, cross-region replication, and various restoration scenarios.

## Backup Architecture

### Components

- **Database**: PostgreSQL on Cloud SQL
- **ML Models**: Stored in Cloud Storage
- **Application State**: Configuration and metadata
- **Training Data**: Historical RL experience data

### Storage Locations

- **Primary**: `gs://shyvr-rlte-backups` (us-central1)
- **Secondary**: `gs://shyvr-rlte-backups-east` (us-east1)  
- **Archive**: `gs://shyvr-rlte-backups-archive` (europe-west1)

## Backup Types

### 1. Full Backup

Complete snapshot of all system components:

```python
from src.deploy.backup_restore import BackupManager

backup_manager = BackupManager(
    project_id="shvyr-ai-bots",
    primary_storage="gs://shyvr-rlte-backups",
    backup_regions=["us-east1", "europe-west1"],
    config={
        "compression_enabled": True,
        "encryption_enabled": True,
        "verification_enabled": True
    }
)

# Execute full backup
backup_result = await backup_manager.execute_full_backup({
    "components": [
        "database",
        "ml_models", 
        "training_checkpoints",
        "configurations",
        "application_state"
    ],
    "description": "Scheduled daily full backup",
    "retention_days": 30
})
```

### 2. Incremental Backup

Changes since last backup:

```python
from src.deploy.backup_restore import IncrementalBackupManager

incremental_manager = IncrementalBackupManager(
    project_id="shvyr-ai-bots",
    config={"base_backup_required": True}
)

# Create incremental backup
await incremental_manager.create_incremental_backup(
    base_backup_id="full_backup_20241201_040000",
    component_data={
        "database": {"tables": ["experience_replay", "model_metrics"]},
        "models": {"changed_models": ["dqn_agent_v2"]}
    }
)
```

### 3. Cross-Region Backup

Replicated across multiple regions:

```python
from src.deploy.backup_restore import CrossRegionBackupReplicator

replicator = CrossRegionBackupReplicator(
    project_id="shvyr-ai-bots",
    config={
        "replication_strategy": "async",
        "verification_enabled": True
    }
)

# Replicate to backup regions
await replicator.replicate_backup(
    backup_id="backup_20241201_120000",
    target_regions=["us-east1", "europe-west1"]
)
```

## Scheduled Backup Operations

### Daily Full Backup

```python
from src.deploy.backup_restore import BackupScheduler
from src.deploy.backup_restore import BackupJob

scheduler = BackupScheduler(
    project_id="shvyr-ai-bots",
    config={
        "notification_enabled": True,
        "failure_retry_count": 3
    }
)

# Schedule daily backup at 4:00 AM UTC
daily_job = BackupJob(
    name="daily_full_backup",
    schedule="0 4 * * *",
    backup_type="full",
    components=["database", "ml_models", "configurations"],
    retention_days=30,
    cross_region_replication=True
)

await scheduler.schedule_backup_job(daily_job)
```

### Hourly Incremental Backup

```python
# Schedule hourly incremental backups
hourly_job = BackupJob(
    name="hourly_incremental_backup", 
    schedule="0 * * * *",
    backup_type="incremental",
    components=["database", "training_checkpoints"],
    retention_days=7,
    base_backup_pattern="daily_full_backup"
)

await scheduler.schedule_backup_job(hourly_job)
```

### Model Checkpoint Backup

```python
# Backup ML model checkpoints after training
model_job = BackupJob(
    name="model_checkpoint_backup",
    trigger="training_completion",
    backup_type="selective",
    components=["ml_models", "training_metrics"],
    retention_days=90
)

await scheduler.schedule_backup_job(model_job)
```

## Manual Backup Procedures

### Pre-Deployment Backup

Before major deployments:

```bash
# Using the production backup script
cd /Users/kendo/daniqan/shyvrai-rlte
python scripts/manage_production_backups.py --manual-backup

# Or programmatically
uv run python -c "
import asyncio
from src.deploy.backup_restore import BackupManager

async def pre_deployment_backup():
    manager = BackupManager(
        project_id='shvyr-ai-bots',
        primary_storage='gs://shyvr-rlte-backups',
        backup_regions=['us-east1']
    )
    
    result = await manager.execute_full_backup({
        'description': 'Pre-deployment backup',
        'tags': ['deployment', 'manual'],
        'priority': 'high'
    })
    
    print(f'Backup created: {result[\"backup_id\"]}')

asyncio.run(pre_deployment_backup())
"
```

### Critical Data Backup

For urgent data preservation:

```python
from src.deploy.backup_restore import CriticalDataBackup

critical_backup = CriticalDataBackup(
    project_id="shvyr-ai-bots",
    config={"expedited_processing": True}
)

# Backup critical components immediately
await critical_backup.execute_critical_backup({
    "components": ["user_data", "active_models", "current_training_state"],
    "priority": "emergency",
    "description": "Critical data backup before emergency maintenance"
})
```

## Restoration Procedures

### 1. Point-in-Time Restore

Restore to specific timestamp:

```python
from src.deploy.backup_restore import RestoreManager

restore_manager = RestoreManager(
    project_id="shvyr-ai-bots",
    config={
        "verification_enabled": True,
        "rollback_enabled": True,
        "notification_enabled": True
    }
)

# Restore to 2 hours ago
target_timestamp = time.time() - 7200  # 2 hours
restore_result = await restore_manager.execute_point_in_time_restore(
    target_timestamp=target_timestamp,
    components=["database", "ml_models"]
)
```

### 2. Selective Component Restore

Restore specific components:

```python
# Restore only ML models from specific backup
await restore_manager.execute_selective_restore(
    backup_id="backup_20241201_040000",
    components=[
        "ml_models/dqn_agent",
        "ml_models/policy_network",
        "training_checkpoints/latest"
    ]
)
```

### 3. Cross-Region Restore

Restore from backup region:

```python
from src.deploy.backup_restore import CrossRegionRestoreManager

cross_region_restore = CrossRegionRestoreManager(
    project_id="shvyr-ai-bots",
    config={"source_region_priority": ["us-east1", "europe-west1"]}
)

# Restore from backup region
await cross_region_restore.execute_cross_region_restore(
    backup_id="backup_20241201_040000",
    source_region="us-east1",
    target_region="us-central1"
)
```

### 4. Database-Only Restore

For database corruption issues:

```bash
# Stop application services first
gcloud run services update shyvr-rlte --region=us-central1 --min-instances=0

# Execute database restore
uv run python -c "
import asyncio
from src.deploy.backup_restore import DatabaseRestoreManager

async def restore_database():
    db_restore = DatabaseRestoreManager(
        project_id='shvyr-ai-bots',
        instance_name='shyvr-rlte-db-prod'
    )
    
    await db_restore.execute_database_restore(
        backup_id='db_backup_20241201_040000',
        target_database='shyvr_rlte_prod'
    )

asyncio.run(restore_database())
"

# Restart services after restore
gcloud run services update shyvr-rlte --region=us-central1 --min-instances=1
```

## Backup Verification

### Automated Verification

```python
from src.deploy.backup_restore import BackupVerifier

verifier = BackupVerifier(
    project_id="shvyr-ai-bots",
    config={
        "integrity_check_enabled": True,
        "sample_restore_test": True,
        "checksum_verification": True
    }
)

# Verify backup integrity
verification_result = await verifier.verify_backup_integrity(
    backup_id="backup_20241201_040000",
    verification_level="comprehensive"
)

if verification_result["status"] == "valid":
    print("✅ Backup verification successful")
else:
    print(f"❌ Backup verification failed: {verification_result['errors']}")
```

### Manual Verification

```bash
# Verify recent backups
python scripts/manage_production_backups.py --list-backups --limit=5

# Test restore to temporary instance
python scripts/manage_production_backups.py --test-restore backup_20241201_040000
```

## Monitoring and Alerting

### Backup Success Monitoring

```python
from src.deploy.backup_restore import BackupMonitor

monitor = BackupMonitor(
    project_id="shvyr-ai-bots",
    config={
        "alert_on_failure": True,
        "success_rate_threshold": 99.0,
        "notification_channels": ["slack", "email"]
    }
)

# Check backup health
health_status = await monitor.check_backup_health()
```

### Storage Usage Monitoring

```bash
# Check backup storage usage
gsutil du -sh gs://shyvr-rlte-backups/
gsutil du -sh gs://shyvr-rlte-backups-east/
gsutil du -sh gs://shyvr-rlte-backups-archive/

# Monitor costs
gcloud billing budgets list --billing-account=YOUR_BILLING_ACCOUNT
```

## Backup Retention Management

### Lifecycle Policies

```python
from src.deploy.backup_restore import RetentionPolicyManager

retention_manager = RetentionPolicyManager(
    project_id="shvyr-ai-bots",
    config={"dry_run": False}
)

# Apply retention policies
await retention_manager.apply_retention_policy({
    "daily_backups": {"retention_days": 30},
    "weekly_backups": {"retention_days": 90}, 
    "monthly_backups": {"retention_days": 365},
    "archive_backups": {"retention_days": 2555}  # 7 years
})
```

### Manual Cleanup

```bash
# List old backups
gsutil ls -l gs://shyvr-rlte-backups/ | grep "2023-"

# Remove backups older than retention policy
uv run python -c "
import asyncio
from src.deploy.backup_restore import BackupCleanupManager

async def cleanup_old_backups():
    cleanup = BackupCleanupManager('shvyr-ai-bots')
    await cleanup.cleanup_expired_backups(dry_run=False)

asyncio.run(cleanup_old_backups())
"
```

## Emergency Procedures

### Rapid Backup Creation

For emergency situations requiring immediate backup:

```python
from src.deploy.backup_restore import EmergencyBackupManager

emergency_backup = EmergencyBackupManager(
    project_id="shvyr-ai-bots",
    config={"priority": "emergency", "expedited": True}
)

# Create emergency backup
await emergency_backup.create_emergency_backup({
    "components": ["critical_data", "active_models"],
    "description": "Emergency backup before system maintenance",
    "replicate_immediately": True
})
```

### Rapid Restore

For critical system restoration:

```python
from src.deploy.backup_restore import RapidRestoreManager

rapid_restore = RapidRestoreManager(
    project_id="shvyr-ai-bots", 
    config={"parallel_processing": True}
)

# Execute rapid restore
await rapid_restore.execute_rapid_restore(
    backup_id="latest_valid",
    priority_components=["database", "critical_models"]
)
```

## Testing Procedures

### Monthly Backup Testing

```bash
# Run comprehensive backup/restore tests
uv run pytest tests/unit/deploy/test_backup_restore.py -v

# Test specific backup scenarios  
uv run pytest tests/unit/deploy/test_backup_restore.py::TestBackupManager::test_cross_region_replication -v
```

### Disaster Recovery Drill

```python
# Simulate disaster scenario
from src.deploy.backup_restore import DisasterSimulator

simulator = DisasterSimulator(
    project_id="shvyr-ai-bots",
    config={"test_environment": True}
)

# Run DR drill
drill_results = await simulator.execute_disaster_drill({
    "scenario": "primary_region_failure",
    "components": ["database", "ml_models"],
    "verify_recovery": True
})
```

## Troubleshooting

### Common Issues

#### Backup Fails with Storage Error

```bash
# Check storage bucket permissions
gsutil iam get gs://shyvr-rlte-backups

# Verify service account permissions
gcloud projects get-iam-policy shvyr-ai-bots --flatten="bindings[].members" --filter="bindings.members:serviceAccount"
```

#### Restore Timeout

```python
# Increase timeout for large restores
config = {
    "restore_timeout_minutes": 120,  # 2 hours
    "chunk_size": "100MB",
    "parallel_threads": 4
}
```

#### Cross-Region Replication Failure

```bash
# Check network connectivity between regions
gcloud compute networks connectivity-tests create test-cross-region \
  --source-project=shvyr-ai-bots \
  --destination-project=shvyr-ai-bots

# Verify regional quotas
gcloud compute regions describe us-east1 --project=shvyr-ai-bots
```

### Debug Commands

```bash
# Enable debug logging
export BACKUP_DEBUG=true

# Check backup job status
gcloud scheduler jobs describe daily_full_backup --location=us-central1

# View backup operation logs
gcloud logging read "resource.type=cloud_function AND textPayload:backup" --limit=50
```

## Reference Information

### Backup IDs Format

```
Format: {type}_{component}_{timestamp}
Examples:
- full_system_20241201_040000
- incr_database_20241201_050000  
- manual_models_20241201_120000
```

### Storage Paths

```
gs://shyvr-rlte-backups/
├── daily/          # Daily full backups
├── hourly/         # Incremental backups  
├── manual/         # Manual backups
├── emergency/      # Emergency backups
└── archive/        # Long-term archive
```

### Configuration Files

- Backup config: `/src/deploy/config/backup_config.yaml`
- Retention policies: `/src/deploy/config/retention_policies.yaml`
- Monitoring config: `/src/deploy/config/monitoring_config.yaml`

### Useful Commands

```bash
# Check backup status
uv run python scripts/manage_production_backups.py --report

# List recent backups
uv run python scripts/manage_production_backups.py --list-backups

# Create manual backup
uv run python scripts/manage_production_backups.py --manual-backup

# Test backup integrity
uv run python scripts/manage_production_backups.py --test-restore BACKUP_ID
```