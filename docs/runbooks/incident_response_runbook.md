# Incident Response Runbook

## Overview

This runbook defines the incident response procedures for the ShyvRAI-RLTE production environment. It covers incident classification, escalation procedures, communication protocols, and coordination with disaster recovery operations.

## Incident Classification

### Severity Levels

#### P0 - Critical (RTO: 15 minutes)
- Complete service outage
- Data corruption or loss
- Security breach
- Regional infrastructure failure

**Examples:**
- All users unable to access the service
- Database corruption preventing operations
- ML models completely unavailable

#### P1 - High (RTO: 1 hour) 
- Significant service degradation
- Partial functionality loss
- Performance issues affecting >50% users
- Backup system failures

**Examples:**
- API response times >5 seconds
- ML inference failures for some models
- Cross-region replication issues

#### P2 - Medium (RTO: 4 hours)
- Minor service degradation
- Non-critical feature unavailable
- Performance issues affecting <25% users

**Examples:**
- Non-essential API endpoints failing
- Monitoring dashboard issues
- Training pipeline delays

#### P3 - Low (RTO: 24 hours)
- Cosmetic issues
- Documentation problems
- Non-urgent technical debt

## Incident Response Team

### Roles and Responsibilities

#### Incident Commander
- Overall incident coordination
- Decision making authority
- External communication
- Resource allocation

#### Technical Lead
- Technical investigation and resolution
- Coordinate with engineering teams
- Implement fixes and workarounds
- Post-incident technical analysis

#### Communications Lead
- Internal status updates
- External customer communication
- Documentation updates
- Stakeholder notifications

#### Database Expert
- Database-related incidents
- Backup and restore operations
- Data integrity verification
- Performance optimization

#### ML/AI Specialist
- Model-related incidents
- Training pipeline issues
- Model deployment problems
- Performance analysis

### Contact Information

```yaml
incident_contacts:
  incident_commander:
    primary: "John Doe <john.doe@company.com>"
    backup: "Jane Smith <jane.smith@company.com>"
    phone: "+1-555-0123"
    
  technical_lead:
    primary: "Alice Engineer <alice@company.com>"
    backup: "Bob Developer <bob@company.com>"
    phone: "+1-555-0124"
    
  communications:
    primary: "Carol PM <carol@company.com>"
    backup: "Dave Marketing <dave@company.com>"
    phone: "+1-555-0125"
```

## Incident Detection

### Automated Monitoring

```python
# Example monitoring setup
from src.deploy.disaster_recovery import DisasterRecoveryManager

dr_manager = DisasterRecoveryManager(
    project_id="shvyr-ai-bots",
    primary_region="us-central1",
    backup_regions=["us-east1", "europe-west1"],
    config={
        "monitoring_enabled": True,
        "alert_thresholds": {
            "response_time_p95": 1000,  # ms
            "error_rate": 0.05,          # 5%
            "availability": 0.999        # 99.9%
        }
    }
)

# Setup automated detection
await dr_manager.setup_disaster_detection({
    "health_checks": ["/health", "/api/v1/status"],
    "check_interval": 30,  # seconds
    "failure_threshold": 3,
    "alert_channels": ["slack", "pagerduty", "email"]
})
```

### Alert Sources

- **GCP Monitoring**: Service availability, error rates
- **Application Logs**: Error patterns, performance issues  
- **Database Monitoring**: Connection issues, query failures
- **ML Pipeline**: Training failures, model serving errors
- **User Reports**: Direct feedback via support channels

### Alert Routing

```yaml
alert_routing:
  P0_critical:
    - pagerduty: "immediate"
    - slack: "#incident-response"
    - email: "oncall@company.com"
    
  P1_high:
    - slack: "#alerts-high"
    - email: "engineering@company.com"
    
  P2_medium:
    - slack: "#alerts-medium"
    
  P3_low:
    - email: "engineering-daily@company.com"
```

## Incident Response Workflow

### 1. Detection and Alerting (0-5 minutes)

```bash
# Initial detection
ALERT_TIME=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"

echo "🚨 INCIDENT DETECTED: ${INCIDENT_ID}"
echo "Time: ${ALERT_TIME}"
echo "Source: [Monitoring/User Report/Other]"
```

### 2. Initial Assessment (5-10 minutes)

```python
# Quick health check
import asyncio
from src.deploy.disaster_recovery import HealthCheckManager

async def initial_assessment():
    health_checker = HealthCheckManager(
        project_id="shvyr-ai-bots"
    )
    
    # Check critical services
    results = await health_checker.check_critical_services([
        "shyvr-rlte-api",
        "shyvr-rlte-ml-service", 
        "shyvr-rlte-db",
        "shyvr-rlte-training-pipeline"
    ])
    
    # Determine severity
    if results["overall_health"] < 0.5:
        return "P0"
    elif results["overall_health"] < 0.8:
        return "P1"
    else:
        return "P2"

severity = asyncio.run(initial_assessment())
print(f"Initial severity assessment: {severity}")
```

### 3. Team Assembly (10-15 minutes)

```bash
# Activate incident response team
SEVERITY="P0"  # From assessment

if [ "$SEVERITY" = "P0" ]; then
    # Page all team members
    echo "Paging full incident response team for P0"
    # Send PagerDuty alerts
    # Post to #incident-response Slack channel
elif [ "$SEVERITY" = "P1" ]; then
    # Page technical team
    echo "Alerting technical team for P1"
fi
```

### 4. Situation Assessment (15-30 minutes)

```python
# Comprehensive situation analysis
from src.deploy.disaster_recovery import SituationAnalyzer

analyzer = SituationAnalyzer(
    project_id="shvyr-ai-bots"
)

situation = await analyzer.analyze_current_situation({
    "incident_id": "INC-20241201-120000",
    "initial_symptoms": ["api_errors", "high_latency"],
    "affected_regions": ["us-central1"],
    "user_impact": "high"
})

print(f"""
Situation Analysis:
- Root cause hypothesis: {situation['likely_causes']}
- Affected systems: {situation['affected_systems']}
- User impact: {situation['user_impact_assessment']}
- Recommended actions: {situation['recommended_actions']}
""")
```

### 5. Disaster Recovery Decision (30-45 minutes)

```python
# Determine if DR activation is needed
from src.deploy.disaster_recovery import DisasterRecoveryDecisionEngine

decision_engine = DisasterRecoveryDecisionEngine()

dr_decision = await decision_engine.evaluate_dr_need({
    "incident_severity": "P0",
    "affected_services": ["api", "database"],
    "estimated_recovery_time": 60,  # minutes
    "rto_target": 15,  # minutes
    "user_impact_percentage": 100
})

if dr_decision["activate_dr"]:
    print("🚨 ACTIVATING DISASTER RECOVERY")
    
    # Execute disaster recovery
    from src.deploy.disaster_recovery import DisasterRecoveryManager
    
    dr_manager = DisasterRecoveryManager(
        project_id="shvyr-ai-bots",
        primary_region="us-central1", 
        backup_regions=["us-east1", "europe-west1"]
    )
    
    disaster_event = {
        "type": "service_outage",
        "severity": "critical",
        "incident_id": "INC-20241201-120000",
        "timestamp": time.time()
    }
    
    await dr_manager.execute_regional_failover(
        disaster_event,
        target_region="us-east1",
        system_state={}
    )
```

## Communication Protocols

### Internal Communication

#### Slack Channels

```yaml
slack_channels:
  primary: "#incident-response"
  engineering: "#engineering-alerts"
  leadership: "#leadership-updates"
  customer_success: "#customer-impact"
```

#### Status Updates

```bash
# Initial notification (within 15 minutes)
curl -X POST -H 'Content-type: application/json' \
  --data '{
    "text": "🚨 P0 INCIDENT: Service Outage - INC-20241201-120000",
    "channel": "#incident-response",
    "username": "Incident Bot",
    "attachments": [{
      "color": "danger",
      "fields": [
        {"title": "Incident ID", "value": "INC-20241201-120000", "short": true},
        {"title": "Severity", "value": "P0 - Critical", "short": true},
        {"title": "Status", "value": "Investigating", "short": true},
        {"title": "ETA", "value": "TBD", "short": true}
      ]
    }]
  }' \
  $SLACK_WEBHOOK_URL

# Progress updates (every 30 minutes)
curl -X POST -H 'Content-type: application/json' \
  --data '{
    "text": "📊 UPDATE: INC-20241201-120000",
    "channel": "#incident-response",
    "attachments": [{
      "color": "warning", 
      "fields": [
        {"title": "Progress", "value": "Disaster recovery activated, failing over to us-east1", "short": false},
        {"title": "ETA", "value": "10 minutes", "short": true},
        {"title": "User Impact", "value": "Service partially restored", "short": true}
      ]
    }]
  }' \
  $SLACK_WEBHOOK_URL
```

### External Communication

#### Status Page Updates

```python
from src.deploy.incident_response import StatusPageManager

status_manager = StatusPageManager(
    service_url="https://status.shyvrai.com",
    api_key="your_status_api_key"
)

# Initial incident notification
await status_manager.create_incident({
    "name": "Service Unavailable",
    "status": "investigating", 
    "impact": "major",
    "components": ["API", "ML Service", "Web Application"],
    "message": "We are investigating reports of service unavailability. Updates will be provided every 30 minutes."
})

# Progress updates
await status_manager.update_incident({
    "incident_id": "incident_123",
    "status": "identified",
    "message": "Issue identified as regional infrastructure failure. Activating disaster recovery procedures."
})

# Resolution
await status_manager.resolve_incident({
    "incident_id": "incident_123",
    "status": "resolved",
    "message": "Service fully restored. All systems operating normally."
})
```

#### Customer Notifications

```python
# Email notification to customers
from src.deploy.incident_response import CustomerNotificationManager

notification_manager = CustomerNotificationManager()

# For P0 incidents affecting all users
await notification_manager.send_incident_notification({
    "incident_id": "INC-20241201-120000",
    "severity": "P0",
    "template": "service_outage",
    "recipients": "all_customers",
    "message": {
        "subject": "Service Disruption - Investigation Underway",
        "body": "We are currently experiencing a service disruption and are working to resolve it as quickly as possible. We will provide updates every 30 minutes."
    }
})
```

## Incident Resolution Procedures

### Database Recovery

```python
# For database-related incidents
from src.deploy.backup_restore import RestoreManager

async def database_recovery_procedure():
    restore_manager = RestoreManager(
        project_id="shvyr-ai-bots",
        config={"verification_enabled": True}
    )
    
    # 1. Stop application services
    print("Stopping application services...")
    # Scale down Cloud Run services
    
    # 2. Restore database from backup
    print("Restoring database from backup...")
    restore_result = await restore_manager.execute_point_in_time_restore(
        target_timestamp=time.time() - 3600,  # 1 hour ago
        components=["database"]
    )
    
    # 3. Verify data integrity
    print("Verifying data integrity...")
    # Run integrity checks
    
    # 4. Restart services
    print("Restarting application services...")
    # Scale up Cloud Run services
    
    return restore_result

# Execute if database recovery needed
if incident_type == "database_corruption":
    recovery_result = await database_recovery_procedure()
```

### Service Recovery

```python
# For service-related incidents
from src.deploy.disaster_recovery import FailoverController

async def service_recovery_procedure():
    failover_controller = FailoverController(
        service_name="shyvr-rlte",
        config={"health_check_enabled": True}
    )
    
    # 1. Execute traffic migration to healthy region
    migration_result = await failover_controller.execute_traffic_migration({
        "source_region": "us-central1",
        "target_region": "us-east1", 
        "migration_type": "immediate"
    })
    
    # 2. Verify service health in new region
    health_check = await failover_controller.verify_service_health("us-east1")
    
    # 3. Update DNS and load balancer configs
    # Update configurations
    
    return {"migration": migration_result, "health": health_check}
```

### Model Recovery

```python
# For ML model-related incidents
from src.model_preservation.emergency_preservation import EmergencyModelPreservation

async def model_recovery_procedure():
    emergency_preservation = EmergencyModelPreservation(
        project_id="shvyr-ai-bots",
        storage_bucket="shyvr-rlte-models-backup"
    )
    
    # 1. Backup current state if possible
    try:
        await emergency_preservation.trigger_emergency_preservation({
            "type": "model_corruption",
            "severity": "high"
        })
    except Exception as e:
        print(f"Unable to backup current state: {e}")
    
    # 2. Restore from last known good backup
    from src.deploy.backup_restore import RestoreManager
    restore_manager = RestoreManager(project_id="shvyr-ai-bots")
    
    restore_result = await restore_manager.execute_selective_restore(
        backup_id="latest_valid_model_backup",
        components=["ml_models", "training_checkpoints"]
    )
    
    # 3. Validate model functionality
    # Run model validation tests
    
    return restore_result
```

## Post-Incident Procedures

### Immediate Actions (0-24 hours)

```python
# Document incident timeline
from src.deploy.incident_response import IncidentDocumentationManager

doc_manager = IncidentDocumentationManager()

incident_summary = {
    "incident_id": "INC-20241201-120000",
    "severity": "P0",
    "duration_minutes": 45,
    "root_cause": "Regional infrastructure failure",
    "resolution": "Disaster recovery failover to us-east1",
    "user_impact": "Complete service outage for 45 minutes",
    "timeline": [
        {"time": "12:00", "event": "Initial alerts received"},
        {"time": "12:05", "event": "Incident team assembled"},
        {"time": "12:15", "event": "Disaster recovery activated"},
        {"time": "12:30", "event": "Service restored in backup region"},
        {"time": "12:45", "event": "Full functionality confirmed"}
    ]
}

await doc_manager.create_incident_report(incident_summary)
```

### Root Cause Analysis (1-7 days)

```python
# Conduct thorough root cause analysis
from src.deploy.incident_response import RootCauseAnalyzer

rca_analyzer = RootCauseAnalyzer()

rca_report = await rca_analyzer.generate_rca_report({
    "incident_id": "INC-20241201-120000",
    "data_sources": [
        "application_logs",
        "infrastructure_metrics", 
        "monitoring_alerts",
        "team_interviews"
    ],
    "analysis_methods": ["5_whys", "timeline_analysis", "fault_tree_analysis"]
})

print(f"""
Root Cause Analysis:
- Primary cause: {rca_report['primary_cause']}
- Contributing factors: {rca_report['contributing_factors']}
- Preventive measures: {rca_report['preventive_measures']}
- Process improvements: {rca_report['process_improvements']}
""")
```

### Action Items and Follow-up

```yaml
# Post-incident action items
action_items:
  - title: "Improve monitoring sensitivity"
    owner: "Engineering Team"
    due_date: "2024-12-15"
    priority: "high"
    
  - title: "Update disaster recovery documentation"
    owner: "DevOps Team" 
    due_date: "2024-12-10"
    priority: "medium"
    
  - title: "Conduct DR drill in backup region"
    owner: "SRE Team"
    due_date: "2024-12-20"
    priority: "high"
    
  - title: "Review and update RTO/RPO targets"
    owner: "Architecture Team"
    due_date: "2024-12-31"
    priority: "low"
```

## Testing and Drills

### Monthly Incident Response Drills

```python
# Simulate incident scenarios
from src.deploy.incident_response import IncidentSimulator

async def monthly_drill():
    simulator = IncidentSimulator(
        project_id="shvyr-ai-bots",
        config={"simulation_mode": True}
    )
    
    # Simulate P1 incident
    drill_result = await simulator.simulate_incident({
        "type": "database_performance_degradation",
        "severity": "P1",
        "duration_minutes": 30,
        "affected_percentage": 75
    })
    
    # Evaluate team response
    evaluation = await simulator.evaluate_response_performance({
        "response_time": drill_result["team_response_time"],
        "communication_effectiveness": drill_result["communication_score"], 
        "resolution_time": drill_result["resolution_time"]
    })
    
    return evaluation

# Run monthly drill
drill_results = await monthly_drill()
print(f"Drill evaluation: {drill_results}")
```

### Quarterly Disaster Recovery Tests

```bash
# Full disaster recovery test
uv run python -c "
import asyncio
from src.deploy.disaster_recovery import DisasterRecoveryTester

async def quarterly_dr_test():
    dr_tester = DisasterRecoveryTester(
        project_id='shvyr-ai-bots',
        config={'full_test_mode': True}
    )
    
    test_results = await dr_tester.execute_full_dr_test({
        'scenario': 'complete_regional_failure',
        'duration_hours': 2,
        'verify_all_components': True
    })
    
    print(f'DR Test Results: {test_results}')

asyncio.run(quarterly_dr_test())
"
```

## Metrics and KPIs

### Incident Response Metrics

```python
from src.deploy.incident_response import IncidentMetricsCalculator

metrics_calculator = IncidentMetricsCalculator()

monthly_metrics = await metrics_calculator.calculate_monthly_metrics({
    "month": "2024-12",
    "metrics": [
        "mean_time_to_detection",
        "mean_time_to_acknowledgment", 
        "mean_time_to_resolution",
        "incident_frequency",
        "severity_distribution"
    ]
})

print(f"""
December 2024 Incident Metrics:
- MTTD: {monthly_metrics['mttd']} minutes
- MTTA: {monthly_metrics['mtta']} minutes  
- MTTR: {monthly_metrics['mttr']} minutes
- Total incidents: {monthly_metrics['total_incidents']}
- P0 incidents: {monthly_metrics['p0_count']}
""")
```

### SLA Compliance

```yaml
sla_targets:
  availability: 99.9%  # 8.77 hours downtime/year
  p0_resolution: 15 minutes
  p1_resolution: 60 minutes
  customer_notification: 15 minutes
  
current_performance:
  availability: 99.95%
  p0_avg_resolution: 12 minutes
  p1_avg_resolution: 45 minutes
  avg_notification_time: 8 minutes
```

## Tools and Resources

### Incident Management Tools

- **PagerDuty**: Alert routing and escalation
- **Slack**: Team coordination and updates
- **Status Page**: Customer communication
- **Google Cloud Console**: Infrastructure monitoring
- **Datadog**: Application monitoring and logs

### Emergency Contacts

```yaml
emergency_contacts:
  google_cloud_support: "+1-877-352-7323"
  pagerduty_support: "+1-844-PAGERDUTY"  
  legal_counsel: "legal@company.com"
  executive_escalation: "ceo@company.com"
```

### Reference Documentation

- [Disaster Recovery Runbook](disaster_recovery_runbook.md)
- [Backup and Restore Runbook](backup_restore_runbook.md)
- [Production Monitoring Dashboard](https://console.cloud.google.com/monitoring)
- [Test Suite Documentation](../../tests/unit/deploy/)

### Quick Reference Commands

```bash
# Check service health
curl -f https://shyvr-rlte-us-central1.run.app/health

# View recent logs
gcloud logging read "resource.type=cloud_run_revision" --limit=100

# Check backup status
python scripts/manage_production_backups.py --report

# Run disaster recovery tests
uv run pytest tests/unit/deploy/test_disaster_recovery.py -v
```