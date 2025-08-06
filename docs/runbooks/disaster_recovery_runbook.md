# Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for disaster recovery operations in the ShyvRAI-RLTE production environment. The system is deployed on GCP Cloud Run with automated disaster recovery capabilities.

## Key Components

- **DisasterRecoveryManager**: Manages overall disaster detection and recovery orchestration
- **BackupManager**: Handles cross-region backup replication and restoration
- **FailoverController**: Manages automated regional failover procedures
- **EmergencyModelPreservation**: Preserves ML models during disaster events

## RTO/RPO Targets

- **Recovery Time Objective (RTO)**: 15 minutes for critical services
- **Recovery Point Objective (RPO)**: 5 minutes for data loss tolerance
- **Service Availability**: 99.9% uptime target

## Disaster Scenarios

### 1. Regional Outage

**Symptoms:**
- Primary region (us-central1) becomes unavailable
- Cloud Run services unable to start
- Database connectivity lost
- Monitoring alerts triggered

**Recovery Steps:**

1. **Assess Situation**
   ```bash
   # Check regional status
   gcloud status
   gcloud compute regions describe us-central1
   ```

2. **Activate Disaster Recovery**
   ```python
   from src.deploy.disaster_recovery import DisasterRecoveryManager
   
   dr_manager = DisasterRecoveryManager(
       project_id="shvyr-ai-bots",
       primary_region="us-central1",
       backup_regions=["us-east1", "europe-west1"],
       config={
           "rto_minutes": 15,
           "rpo_minutes": 5,
           "monitoring_enabled": True
       }
   )
   
   # Execute regional failover
   disaster_event = {
       "type": "regional_outage",
       "affected_region": "us-central1",
       "severity": "critical",
       "timestamp": time.time()
   }
   
   await dr_manager.execute_regional_failover(
       disaster_event,
       target_region="us-east1",
       system_state={}
   )
   ```

3. **Verify Failover**
   - Check service endpoints in backup region
   - Validate database connectivity
   - Test critical application workflows
   - Monitor performance metrics

### 2. Database Corruption

**Symptoms:**
- Database queries returning errors
- Data integrity issues detected
- Backup verification failures

**Recovery Steps:**

1. **Stop Applications**
   ```bash
   # Scale down Cloud Run services
   gcloud run services update shyvr-rlte --region=us-central1 --min-instances=0
   ```

2. **Execute Database Recovery**
   ```python
   # Point-in-time recovery
   await dr_manager.execute_database_recovery(
       disaster_event={
           "type": "database_corruption",
           "severity": "high"
       },
       recovery_strategy="point_in_time",
       target_timestamp=time.time() - 3600  # 1 hour ago
   )
   ```

3. **Validate Recovery**
   - Test database connectivity
   - Verify data integrity
   - Run application smoke tests

### 3. Model Corruption/Loss

**Symptoms:**
- ML model serving errors
- Model inference failures
- Training checkpoints corrupted

**Recovery Steps:**

1. **Activate Emergency Preservation**
   ```python
   from src.model_preservation.emergency_preservation import EmergencyModelPreservation
   
   emergency_preservation = EmergencyModelPreservation(
       project_id="shvyr-ai-bots",
       storage_bucket="shyvr-rlte-models-backup"
   )
   
   # Trigger emergency backup of critical models
   await emergency_preservation.trigger_emergency_preservation(
       disaster_event={
           "type": "model_corruption",
           "severity": "critical"
       },
       priority_models=["dqn_agent", "policy_network"]
   )
   ```

2. **Execute Model Recovery**
   ```python
   from src.deploy.backup_restore import RestoreManager
   
   restore_manager = RestoreManager(
       project_id="shvyr-ai-bots",
       config={"verification_enabled": True}
   )
   
   # Restore from latest backup
   await restore_manager.execute_selective_restore(
       backup_id="model_backup_20241201_120000",
       components=["ml_models", "training_checkpoints"]
   )
   ```

## Backup Procedures

### Automated Backups

Automated backups run daily at 04:00 UTC:

```python
from src.deploy.backup_restore import BackupScheduler

scheduler = BackupScheduler(
    project_id="shvyr-ai-bots",
    config={"retention_days": 30}
)

# Schedule daily backup
await scheduler.schedule_backup_job(BackupJob(
    name="daily_full_backup",
    schedule="0 4 * * *",  # 4 AM daily
    backup_type="full",
    components=["database", "models", "configurations"]
))
```

### Manual Backups

For critical changes or before major deployments:

```python
from src.deploy.backup_restore import BackupManager

backup_manager = BackupManager(
    project_id="shvyr-ai-bots",
    primary_storage="gs://shyvr-rlte-backups",
    backup_regions=["us-east1", "europe-west1"]
)

# Create manual backup with cross-region replication
await backup_manager.execute_backup_with_replication(
    backup_data={
        "components": ["database", "models"],
        "description": "Pre-deployment backup"
    },
    replication_regions=["us-east1", "europe-west1"]
)
```

### Backup Verification

Regular backup integrity checks:

```python
from src.deploy.backup_restore import BackupVerifier

verifier = BackupVerifier(
    project_id="shvyr-ai-bots",
    config={"deep_verification": True}
)

# Verify latest backup
await verifier.verify_backup_integrity(
    backup_id="latest",
    verification_level="comprehensive"
)
```

## Failover Procedures

### Automated Failover

The system monitors service health and triggers automatic failover:

```python
from src.deploy.disaster_recovery import FailoverController

failover_controller = FailoverController(
    service_name="shyvr-rlte",
    config={
        "health_check_interval": 30,
        "failure_threshold": 3,
        "auto_failover_enabled": True
    }
)

# Setup monitoring
await failover_controller.setup_failover_monitoring({
    "primary_endpoints": ["https://shyvr-rlte-us-central1.run.app"],
    "backup_endpoints": ["https://shyvr-rlte-us-east1.run.app"],
    "health_check_path": "/health"
})
```

### Manual Failover

For planned maintenance or emergency situations:

```python
# Execute traffic migration
await failover_controller.execute_traffic_migration({
    "source_region": "us-central1",
    "target_region": "us-east1",
    "migration_type": "immediate",
    "rollback_enabled": True
})
```

## Testing Procedures

### Disaster Recovery Drills

Monthly DR drills should be conducted:

```bash
# Run disaster recovery tests
uv run pytest tests/unit/deploy/test_disaster_recovery.py -v

# Test specific scenarios
uv run pytest tests/unit/deploy/test_disaster_recovery.py::TestDisasterRecoveryManager::test_regional_failover -v
```

### Backup Testing

Regular backup restore tests:

```bash
# Test backup creation and restoration
uv run python -c "
from src.deploy.backup_restore import BackupManager, RestoreManager
import asyncio

async def test_backup_restore():
    # Create test backup
    backup_manager = BackupManager('shvyr-ai-bots', 'gs://test-backups', ['us-east1'])
    backup_id = await backup_manager.execute_full_backup({'test': True})
    
    # Test restoration
    restore_manager = RestoreManager('shvyr-ai-bots', {'test_mode': True})
    await restore_manager.execute_point_in_time_restore(time.time())
    
    print('Backup/restore test completed successfully')

asyncio.run(test_backup_restore())
"
```

## Monitoring and Alerting

### Key Metrics

- Service availability: `>99.9%`
- Response time: `<500ms P95`
- Database connection pool: `<80% utilization`
- Backup success rate: `100%`
- Recovery time: `<15 minutes`

### Alert Conditions

```yaml
# Example monitoring alerts
alerts:
  - name: "Service Unavailable"
    condition: "availability < 99.5%"
    duration: "5 minutes"
    severity: "critical"
    
  - name: "High Response Time"
    condition: "response_time_p95 > 1000ms"
    duration: "10 minutes"
    severity: "warning"
    
  - name: "Backup Failure"
    condition: "backup_success_rate < 100%"
    duration: "1 minute"
    severity: "high"
```

## Communication Plan

### Incident Response Team

- **Incident Commander**: DevOps Lead
- **Technical Lead**: Senior Developer
- **Communications**: Product Manager
- **Database Expert**: Data Engineer

### Communication Channels

- **Primary**: Slack #incident-response
- **Secondary**: Email distribution list
- **External**: Status page updates
- **Escalation**: Phone tree for critical issues

### Status Updates

- **Initial**: Within 15 minutes of detection
- **Progress**: Every 30 minutes during active incident
- **Resolution**: Within 1 hour of resolution

## Recovery Validation

### Service Health Checks

```bash
# Validate service endpoints
curl -f https://shyvr-rlte-us-east1.run.app/health
curl -f https://shyvr-rlte-us-east1.run.app/api/v1/status

# Database connectivity
gcloud sql instances describe shyvr-rlte-db-prod --project=shvyr-ai-bots
```

### Data Integrity Verification

```python
# Run data integrity checks
from src.deploy.backup_restore import DataIntegrityChecker

integrity_checker = DataIntegrityChecker()
await integrity_checker.verify_data_consistency()
```

### Performance Validation

```bash
# Load testing after recovery
uv run python scripts/load_test.py --target https://shyvr-rlte-us-east1.run.app
```

## Post-Incident Procedures

### 1. Documentation

- Update incident log with timeline
- Document lessons learned
- Review and update procedures

### 2. Root Cause Analysis

- Identify failure points
- Implement preventive measures
- Update monitoring and alerting

### 3. Process Improvement

- Review RTO/RPO targets
- Update disaster recovery procedures
- Conduct team training

## Contact Information

- **On-Call Engineer**: [Pagerduty rotation]
- **Database Administrator**: [Contact details]
- **Cloud Infrastructure**: [GCP Support]
- **Executive Escalation**: [Leadership contacts]

## Reference Links

- [GCP Status Page](https://status.cloud.google.com/)
- [Production Monitoring Dashboard](https://console.cloud.google.com/monitoring)
- [Backup Management Script](../scripts/manage_production_backups.py)
- [Test Suite](../tests/unit/deploy/test_disaster_recovery.py)