#!/bin/bash

# Basic Production Monitoring Setup for RL Experience Database
# Creates basic monitoring using gcloud CLI

set -e

PROJECT_ID="shvyr-ai-bots"
INSTANCE_NAME="shyvr-rlte-db-prod"
DATABASE_NAME="shyvr_rlte_prod"

echo "🔧 Setting up basic production monitoring..."

# Enable required APIs
echo "📡 Enabling monitoring APIs..."
gcloud services enable monitoring.googleapis.com --project=$PROJECT_ID
gcloud services enable logging.googleapis.com --project=$PROJECT_ID

# Create notification channels (email-based)
echo "📧 Creating notification channels..."

# Create email notification channel (requires manual email verification)
EMAIL_CHANNEL=$(gcloud alpha monitoring channels create \
    --display-name="RL Database Alerts" \
    --type="email" \
    --channel-labels="email_address=douglas.danielj@gmail.com" \
    --project=$PROJECT_ID \
    --format="value(name)" || echo "existing")

echo "📧 Email notification channel: $EMAIL_CHANNEL"

# Create basic alert policies using YAML configurations
echo "🚨 Creating alert policies..."

# CPU usage alert
cat > /tmp/cpu_alert.yaml << EOF
displayName: "RL Database High CPU Usage"
documentation:
  content: "Alert when Cloud SQL CPU usage exceeds 80% for 5 minutes"
conditions:
- displayName: "High CPU utilization"
  conditionThreshold:
    filter: 'resource.type="cloudsql_database" AND resource.labels.database_id="$PROJECT_ID:$INSTANCE_NAME" AND metric.type="cloudsql.googleapis.com/database/cpu/utilization"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0.8
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
combiner: AND
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/cpu_alert.yaml --project=$PROJECT_ID || echo "CPU alert already exists"

# Memory usage alert  
cat > /tmp/memory_alert.yaml << EOF
displayName: "RL Database High Memory Usage"
documentation:
  content: "Alert when Cloud SQL memory usage exceeds 85% for 5 minutes"
conditions:
- displayName: "High memory utilization"
  conditionThreshold:
    filter: 'resource.type="cloudsql_database" AND resource.labels.database_id="$PROJECT_ID:$INSTANCE_NAME" AND metric.type="cloudsql.googleapis.com/database/memory/utilization"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0.85
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN  
combiner: AND
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/memory_alert.yaml --project=$PROJECT_ID || echo "Memory alert already exists"

# Connection count alert
cat > /tmp/connections_alert.yaml << EOF
displayName: "RL Database High Connection Count"
documentation:
  content: "Alert when Cloud SQL connections exceed 80 for 5 minutes"
conditions:
- displayName: "High connection count"
  conditionThreshold:
    filter: 'resource.type="cloudsql_database" AND resource.labels.database_id="$PROJECT_ID:$INSTANCE_NAME" AND metric.type="cloudsql.googleapis.com/database/network/connections"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 80
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
combiner: AND
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/connections_alert.yaml --project=$PROJECT_ID || echo "Connections alert already exists"

# Cleanup temp files
rm -f /tmp/cpu_alert.yaml /tmp/memory_alert.yaml /tmp/connections_alert.yaml

# Create log-based metrics for RL experience errors
echo "📊 Creating log-based metrics..."

gcloud logging metrics create rl_database_errors \
    --description="Count of RL database errors" \
    --log-filter='
    (resource.type="cloudsql_database" AND resource.labels.database_id="'$PROJECT_ID':'$INSTANCE_NAME'") OR
    (jsonPayload.message=~".*rl.*experience.*error.*" OR jsonPayload.message=~".*experience.*storage.*failed.*")
    severity>=ERROR' \
    --project=$PROJECT_ID || echo "Log metric already exists"

gcloud logging metrics create rl_experience_storage_rate \
    --description="Rate of RL experience storage operations" \
    --log-filter='jsonPayload.message=~".*rl.*experience.*stored.*" OR textPayload=~".*experience.*inserted.*"' \
    --project=$PROJECT_ID || echo "Storage rate metric already exists"

# Create dashboard configuration
echo "📊 Creating dashboard configuration..."
mkdir -p monitoring/grafana

cat > monitoring/grafana/rl_production_dashboard.json << EOF
{
  "dashboard": {
    "title": "RL Experience Storage Production Monitoring",
    "tags": ["rl", "database", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Cloud SQL CPU Usage",
        "type": "graph",
        "targets": [{
          "expr": "cloudsql_googleapis_com_database_cpu_utilization{database_id=\"$PROJECT_ID:$INSTANCE_NAME\"}",
          "legendFormat": "CPU %"
        }],
        "yAxes": [{
          "label": "Percentage",
          "min": 0,
          "max": 100
        }]
      },
      {
        "title": "Cloud SQL Memory Usage", 
        "type": "graph",
        "targets": [{
          "expr": "cloudsql_googleapis_com_database_memory_utilization{database_id=\"$PROJECT_ID:$INSTANCE_NAME\"}",
          "legendFormat": "Memory %"
        }],
        "yAxes": [{
          "label": "Percentage",
          "min": 0,
          "max": 100
        }]
      },
      {
        "title": "Database Connections",
        "type": "graph", 
        "targets": [{
          "expr": "cloudsql_googleapis_com_database_network_connections{database_id=\"$PROJECT_ID:$INSTANCE_NAME\"}",
          "legendFormat": "Connections"
        }],
        "yAxes": [{
          "label": "Count",
          "min": 0
        }]
      },
      {
        "title": "RL Database Errors",
        "type": "graph",
        "targets": [{
          "expr": "logging_googleapis_com_user_rl_database_errors",
          "legendFormat": "Errors/min"
        }],
        "yAxes": [{
          "label": "Errors per minute",
          "min": 0
        }]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

# Set up automated backup configuration
echo "💾 Configuring automated backups..."

gcloud sql instances patch $INSTANCE_NAME \
    --backup-start-time=04:00 \
    --backup-location=us \
    --retained-backups-count=7 \
    --retained-transaction-log-days=7 \
    --project=$PROJECT_ID || echo "Backup configuration already set"

# Create health check script
echo "🏥 Creating health check script..."
cat > scripts/production_health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Production Database Health Check
Runs periodic health checks on the RL experience database
"""

import asyncio
import logging
import sys
from pathlib import Path
import asyncpg
from google.cloud import secretmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

async def check_database_health():
    """Perform comprehensive database health check"""
    try:
        # Get credentials
        client = secretmanager.SecretManagerServiceClient()
        password = client.access_secret_version(
            request={"name": "projects/shvyr-ai-bots/secrets/db-password-production/versions/latest"}
        ).payload.data.decode("UTF-8")
        
        # Connect to database
        conn = await asyncpg.connect(
            host="localhost", 
            port=5433,
            database="shyvr_rlte_prod",
            user="rlte_prod_user", 
            password=password
        )
        
        try:
            # Basic connectivity
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            
            # Check table existence
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
            """)
            assert len(tables) == 3
            
            # Check data integrity
            experience_count = await conn.fetchval("SELECT COUNT(*) FROM rl_experiences")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM rl_training_sessions")
            
            # Performance test
            import time
            start = time.time()
            await conn.fetchval("SELECT COUNT(*) FROM rl_experiences LIMIT 1")
            query_time = time.time() - start
            
            health_status = {
                "status": "healthy",
                "experience_count": experience_count,
                "session_count": session_count,
                "query_time_ms": query_time * 1000,
                "tables_exist": len(tables) == 3
            }
            
            logger.info(f"Database health check passed: {health_status}")
            return health_status
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(check_database_health())
    sys.exit(0 if result["status"] == "healthy" else 1)
EOF

chmod +x scripts/production_health_check.py

echo "✅ Basic production monitoring setup completed!"
echo ""
echo "📋 Summary:"
echo "  ✓ Monitoring APIs enabled"
echo "  ✓ Basic alert policies created"
echo "  ✓ Log-based metrics configured" 
echo "  ✓ Dashboard configuration saved"
echo "  ✓ Automated backups configured"
echo "  ✓ Health check script created"
echo ""
echo "🔗 Next steps:"
echo "  1. Configure Grafana with the dashboard JSON"
echo "  2. Set up email notifications (verify email address)"
echo "  3. Run health checks: uv run python scripts/production_health_check.py"
echo "  4. Monitor alerts in Cloud Console"