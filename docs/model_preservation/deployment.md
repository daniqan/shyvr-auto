# Model Preservation System - Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [GCS Storage Setup](#gcs-storage-setup)
- [Cache Configuration](#cache-configuration)
- [Production Deployment](#production-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Security Configuration](#security-configuration)
- [Deployment Checklist](#deployment-checklist)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 200GB+ SSD
- Network: 1 Gbps
- Load balancer support

### Software Dependencies

```bash
# Required software versions
Python >= 3.11
PostgreSQL >= 14
Redis >= 6.2
Docker >= 20.10 (for containerized deployment)
Kubernetes >= 1.24 (for k8s deployment)

# Google Cloud SDK (if using GCS)
gcloud >= 400.0.0
```

### Service Accounts and Permissions

**Google Cloud IAM Roles:**
```bash
# Service account for model preservation
gcloud iam service-accounts create model-preservation-sa \
    --display-name="Model Preservation Service Account"

# Required roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:model-preservation-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:model-preservation-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client"
```

## Environment Setup

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/shyvrai-rlte.git
cd shyvrai-rlte

# Install dependencies with uv
uv sync

# Create environment file
cp env.template .env

# Edit environment variables
vim .env
```

### Environment Variables

Create `.env` file with the following variables:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=shyvr_rlte
POSTGRES_USER=shyvr_user
POSTGRES_PASSWORD=secure_password
DATABASE_URL=postgresql://shyvr_user:secure_password@localhost:5432/shyvr_rlte

# Google Cloud Storage
GOOGLE_CLOUD_PROJECT=your-project-id
MODEL_PRESERVATION_BUCKET=your-model-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
REDIS_DB=0

# Model Preservation Settings
MODEL_PRESERVATION_ENABLED=true
MODEL_CACHE_SIZE_MB=500
MODEL_COMPRESSION_ENABLED=true
MODEL_VERSIONING_STRATEGY=semantic
MODEL_DEFAULT_BRANCH=main

# Security Settings
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60

# Monitoring
ENABLE_MONITORING=true
METRICS_PORT=8080
LOG_LEVEL=INFO

# Performance Settings
MAX_CONCURRENT_UPLOADS=5
UPLOAD_TIMEOUT_SECONDS=300
CACHE_TTL_SECONDS=3600
CONNECTION_POOL_SIZE=20
```

## Configuration

### Model Preservation Configuration

Create `config/model_preservation.yaml`:

```yaml
# Model Preservation Configuration
preservation:
  # Storage configuration
  storage:
    provider: gcs
    bucket: ${MODEL_PRESERVATION_BUCKET}
    prefix: models/
    compression:
      enabled: true
      algorithm: gzip
      level: 6
    
    # Retention policies
    retention:
      default_days: 90
      production_models_days: 365
      cleanup_enabled: true
      cleanup_schedule: "0 2 * * *"  # 2 AM daily
  
  # Database configuration
  database:
    track_metadata: true
    enable_performance_tracking: true
    batch_size: 100
    connection_timeout: 30
    
    # Indexes for performance
    create_indexes: true
    maintenance_schedule: "0 3 * * 0"  # 3 AM weekly
  
  # Caching configuration
  caching:
    enabled: true
    
    # Memory cache
    memory_cache:
      size_mb: 500
      ttl_seconds: 300
      max_entries: 1000
    
    # Redis cache
    redis_cache:
      size_mb: 2000
      ttl_seconds: 3600
      cluster_mode: false
    
    # Disk cache
    disk_cache:
      enabled: true
      size_gb: 10
      path: /var/cache/model-preservation
      cleanup_threshold: 0.8
  
  # Versioning configuration
  versioning:
    strategy: semantic  # semantic, sequential, timestamp
    auto_increment: true
    branch_support: true
    max_versions_per_type: 100
    
    # Branch policies
    branches:
      main:
        protected: true
        auto_merge: false
      development:
        protected: false
        auto_merge: true
        merge_strategy: fast-forward
  
  # Monitoring configuration
  monitoring:
    enabled: true
    track_usage: true
    performance_metrics: true
    alert_on_failures: true
    
    # Thresholds
    thresholds:
      upload_time_seconds: 60
      download_time_seconds: 30
      cache_hit_rate: 0.8
      storage_usage_gb: 100
      error_rate: 0.05
  
  # Security configuration
  security:
    encryption_at_rest: true
    encryption_in_transit: true
    checksum_validation: true
    audit_logging: true
    
    # Access control
    require_authentication: true
    role_based_access: true
    api_rate_limiting: true
```

## Database Setup

### PostgreSQL Installation and Configuration

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE DATABASE shyvr_rlte;
CREATE USER shyvr_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE shyvr_rlte TO shyvr_user;
ALTER USER shyvr_user CREATEDB;
\q
EOF
```

### Database Migration

```bash
# Navigate to database directory
cd database

# Run migrations
python migrate.py --up

# Verify migration status
python migrate.py --status

# Expected output:
# Migration 001_create_initial_schema.sql: Applied
# Migration 002_create_indexes_and_functions.sql: Applied
# Migration 003_create_views_and_permissions.sql: Applied
# Migration 004_create_rl_experience_schema.sql: Applied
# Migration 005_model_preservation_schema.sql: Applied
# Migration 006_model_tags_schema.sql: Applied
# Migration 007_model_branches_schema.sql: Applied
```

### Database Performance Tuning

Add to PostgreSQL configuration (`postgresql.conf`):

```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 256MB

# Connection settings
max_connections = 100
connection_timeout = 30000

# Performance settings
random_page_cost = 1.1
effective_io_concurrency = 200
max_worker_processes = 8
max_parallel_workers_per_gather = 4

# Logging
log_statement = 'mod'
log_duration = on
log_min_duration_statement = 1000

# Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

## GCS Storage Setup

### Create GCS Bucket

```bash
# Set project ID
export PROJECT_ID=your-project-id

# Create bucket for model storage
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://your-model-bucket

# Configure lifecycle policy
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://your-model-bucket
```

### Configure CORS (if needed for web uploads)

```bash
cat > cors.json <<EOF
[
  {
    "origin": ["https://your-domain.com"],
    "method": ["GET", "PUT", "POST", "DELETE"],
    "responseHeader": ["Content-Type", "x-goog-resumable"],
    "maxAgeSeconds": 3600
  }
]
EOF

gsutil cors set cors.json gs://your-model-bucket
```

### Set up IAM permissions

```bash
# Grant service account access to bucket
gsutil iam ch serviceAccount:model-preservation-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://your-model-bucket

# Create and download service account key
gcloud iam service-accounts keys create model-preservation-key.json \
    --iam-account=model-preservation-sa@$PROJECT_ID.iam.gserviceaccount.com
```

## Cache Configuration

### Redis Installation

```bash
# Install Redis (Ubuntu/Debian)
sudo apt install redis-server

# Configure Redis
sudo vim /etc/redis/redis.conf

# Key settings:
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping
# Expected output: PONG
```

### Redis Cluster Setup (Production)

```bash
# Create Redis cluster configuration
mkdir -p /opt/redis-cluster/{7000,7001,7002,7003,7004,7005}

# Create configuration files for each node
for port in {7000..7005}; do
cat > /opt/redis-cluster/$port/redis.conf <<EOF
port $port
cluster-enabled yes
cluster-config-file nodes-$port.conf
cluster-node-timeout 5000
appendonly yes
bind 0.0.0.0
dir /opt/redis-cluster/$port/
EOF
done

# Start Redis cluster nodes
for port in {7000..7005}; do
    redis-server /opt/redis-cluster/$port/redis.conf &
done

# Create cluster
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 --cluster-replicas 1
```

## Production Deployment

### Docker Deployment

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  model-preservation:
    image: shyvr-rlte:latest
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    env_file:
      - .env.production
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - model-cache:/var/cache/model-preservation
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:14
    restart: unless-stopped
    environment:
      POSTGRES_DB: shyvr_rlte
      POSTGRES_USER: shyvr_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/migrations:/docker-entrypoint-initdb.d:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U shyvr_user -d shyvr_rlte"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:6.2-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - model-preservation

volumes:
  postgres_data:
  redis_data:
  model-cache:

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

### Kubernetes Deployment

Create `k8s/model-preservation-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-preservation
  labels:
    app: model-preservation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-preservation
  template:
    metadata:
      labels:
        app: model-preservation
    spec:
      serviceAccountName: model-preservation-sa
      containers:
      - name: model-preservation
        image: gcr.io/PROJECT_ID/model-preservation:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        envFrom:
        - secretRef:
            name: model-preservation-secrets
        - configMapRef:
            name: model-preservation-config
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: cache-volume
          mountPath: /var/cache/model-preservation
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: model-preservation-config
      - name: cache-volume
        persistentVolumeClaim:
          claimName: model-preservation-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: model-preservation-service
spec:
  selector:
    app: model-preservation
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deployment Scripts

Create `deploy.sh`:

```bash
#!/bin/bash
set -e

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-your-project-id}
REGION=${3:-us-central1}

echo "Deploying Model Preservation System to $ENVIRONMENT"

# Build and push Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/model-preservation:latest .
docker push gcr.io/$PROJECT_ID/model-preservation:latest

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/model-preservation-deployment.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/model-preservation

# Run database migrations
echo "Running database migrations..."
kubectl exec -it deployment/model-preservation -- python database/migrate.py --up

# Verify deployment
echo "Verifying deployment..."
kubectl get pods -l app=model-preservation
kubectl logs -l app=model-preservation --tail=50

echo "Deployment completed successfully!"
```

## Monitoring Setup

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "model_preservation_rules.yml"

scrape_configs:
  - job_name: 'model-preservation'
    static_configs:
      - targets: ['model-preservation:8080']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

Create `monitoring/grafana-dashboard.json` (key metrics):

```json
{
  "dashboard": {
    "title": "Model Preservation System",
    "panels": [
      {
        "title": "Model Operations Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_preservation_operations_total[5m])",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "model_preservation_cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ]
      },
      {
        "title": "Storage Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "model_preservation_storage_usage_bytes",
            "legendFormat": "Storage Used"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_preservation_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

Create `monitoring/model_preservation_rules.yml`:

```yaml
groups:
  - name: model_preservation
    rules:
      - alert: ModelPreservationDown
        expr: up{job="model-preservation"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model Preservation service is down"
          description: "Model Preservation service has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(model_preservation_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Model Preservation"
          description: "Error rate is {{ $value }} errors per second"

      - alert: LowCacheHitRate
        expr: model_preservation_cache_hit_rate < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}, below 80%"

      - alert: HighStorageUsage
        expr: model_preservation_storage_usage_gb > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High storage usage"
          description: "Storage usage is {{ $value }}GB, approaching limits"
```

## Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificates (using Let's Encrypt)
sudo apt install certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d your-domain.com -d api.your-domain.com

# Configure auto-renewal
sudo crontab -e
# Add line: 0 12 * * * /usr/bin/certbot renew --quiet
```

### API Security Configuration

```yaml
# In config/security.yaml
security:
  api:
    rate_limiting:
      enabled: true
      requests_per_minute: 100
      burst_size: 20
    
    authentication:
      required: true
      methods: ["jwt", "api_key"]
      jwt:
        secret_key: ${JWT_SECRET_KEY}
        algorithm: "HS256"
        expiration_minutes: 60
    
    authorization:
      role_based: true
      roles:
        admin:
          permissions: ["read", "write", "delete", "admin"]
        user:
          permissions: ["read", "write"]
        readonly:
          permissions: ["read"]
    
    cors:
      enabled: true
      allow_origins: ["https://your-domain.com"]
      allow_methods: ["GET", "POST", "PUT", "DELETE"]
      allow_headers: ["Authorization", "Content-Type"]
```

## Deployment Checklist

### Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] Database migrations tested
- [ ] GCS bucket created and configured
- [ ] Service account keys generated
- [ ] Redis/cache configured
- [ ] SSL certificates obtained
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Security hardening completed
- [ ] Load testing performed

### Deployment Verification

```bash
#!/bin/bash

echo "=== Model Preservation Deployment Verification ==="

# 1. Check service health
echo "Checking service health..."
curl -f http://localhost:8000/health || echo "❌ Health check failed"

# 2. Check database connectivity
echo "Checking database connectivity..."
python -c "
import asyncio
from src.utils.database import DatabaseManager
async def test_db():
    db = DatabaseManager()
    return await db.check_connection()
result = asyncio.run(test_db())
print('✅ Database connected' if result else '❌ Database connection failed')
"

# 3. Check GCS connectivity
echo "Checking GCS connectivity..."
gsutil ls gs://$MODEL_PRESERVATION_BUCKET > /dev/null && echo "✅ GCS accessible" || echo "❌ GCS access failed"

# 4. Check Redis connectivity
echo "Checking Redis connectivity..."
redis-cli ping | grep -q PONG && echo "✅ Redis connected" || echo "❌ Redis connection failed"

# 5. Test API endpoints
echo "Testing API endpoints..."
curl -H "Authorization: Bearer $API_TOKEN" \
     -f http://localhost:8000/api/preservation/models > /dev/null && \
echo "✅ API accessible" || echo "❌ API access failed"

# 6. Check monitoring
echo "Checking monitoring..."
curl -f http://localhost:8080/metrics > /dev/null && echo "✅ Metrics endpoint accessible" || echo "❌ Metrics failed"

echo "=== Verification Complete ==="
```

### Post-Deployment Tasks

1. **Monitor initial performance**:
   ```bash
   # Monitor logs for errors
   kubectl logs -f deployment/model-preservation
   
   # Check resource usage
   kubectl top pods -l app=model-preservation
   ```

2. **Test core functionality**:
   ```bash
   # Test model save/load cycle
   python scripts/test_model_operations.py
   
   # Test rollback functionality
   python scripts/test_rollback.py
   ```

3. **Configure monitoring alerts**:
   ```bash
   # Import Grafana dashboards
   curl -X POST http://grafana:3000/api/dashboards/db \
        -H "Content-Type: application/json" \
        -d @monitoring/grafana-dashboard.json
   ```

4. **Set up automated backups**:
   ```bash
   # Configure backup cron job
   echo "0 2 * * * /opt/scripts/backup-model-preservation.sh" | crontab -
   ```

5. **Document deployment**:
   - Record deployment version
   - Update runbooks
   - Share access credentials securely
   - Update monitoring dashboards

### Rollback Procedures

In case of deployment issues:

```bash
#!/bin/bash
# rollback.sh

echo "Rolling back Model Preservation deployment..."

# 1. Rollback Kubernetes deployment
kubectl rollout undo deployment/model-preservation

# 2. Wait for rollback to complete
kubectl rollout status deployment/model-preservation

# 3. Verify rollback
kubectl get pods -l app=model-preservation

# 4. Check service health
curl -f http://localhost:8000/health

echo "Rollback completed"
```

This deployment guide provides comprehensive instructions for setting up the Model Preservation System in various environments, from local development to production Kubernetes clusters. Follow the checklist carefully to ensure a successful deployment.