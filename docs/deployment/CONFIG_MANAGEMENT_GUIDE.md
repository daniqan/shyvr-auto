# Configuration Management and Environment Setup Guide

## Table of Contents
1. [Overview](#overview)
2. [Configuration Architecture](#configuration-architecture)
3. [Environment-Specific Configurations](#environment-specific-configurations)
4. [Secret Management](#secret-management)
5. [Transformer Model Configuration](#transformer-model-configuration)
6. [Database Configuration](#database-configuration)
7. [API Configuration](#api-configuration)
8. [Monitoring Configuration](#monitoring-configuration)
9. [Security Configuration](#security-configuration)
10. [Environment Variable Management](#environment-variable-management)
11. [Configuration Validation](#configuration-validation)
12. [Best Practices](#best-practices)

## Overview

This guide covers comprehensive configuration management for the transformer-enabled RLTE system, including environment-specific settings, secret management, and configuration validation procedures.

### Configuration Hierarchy
```
Production Configuration Priority:
1. Environment Variables (highest priority)
2. Secret Manager values
3. Production YAML configuration
4. Default fallback values (lowest priority)
```

### Supported Environments
- **Development**: Local development with mock services
- **Test**: Automated testing environment
- **Staging**: Pre-production testing on GCP
- **Production**: Live trading system on GCP

## Configuration Architecture

### Configuration Structure
```
config/
├── config.yaml                    # Base configuration
├── config.test.yaml              # Test environment overrides
├── config.production.yaml        # Production configuration
├── env.production.template       # Environment variable template
├── cloudbuild.yaml              # CI/CD pipeline configuration
└── monitoring_config.json       # Monitoring configuration
```

### Configuration Loading Pattern
```python
# Configuration precedence (highest to lowest)
1. Environment Variables
2. GCP Secret Manager
3. Environment-specific YAML files
4. Base configuration YAML
5. Application defaults
```

## Environment-Specific Configurations

### Development Environment

#### File: `config/config.yaml`
```yaml
# Development configuration with sensible defaults
app:
  name: "shyvr-rlte"
  version: "0.1.0"
  environment: "development"
  debug: true
  host: "127.0.0.1"
  port: 8080

# Local database with Docker
database:
  host: "localhost"
  port: 5432
  database: "rlte_dev"
  username: "rlte_user"
  password: "dev_password"
  pool_size: 5
  max_overflow: 10
  echo: true

# Development API keys (can be test keys)
apis:
  helius:
    api_key: "dev_helius_key"
    base_url: "https://api.helius.xyz"
    rate_limit: 50
  
  birdeye:
    api_key: "dev_birdeye_key"
    base_url: "https://public-api.birdeye.so"
    rate_limit: 30
```

### Test Environment

#### File: `config/config.test.yaml`
```yaml
app:
  environment: "test"
  debug: false
  log_level: "DEBUG"

# In-memory SQLite for tests
database:
  url: "sqlite:///:memory:"
  echo: false

# Mock API configurations
apis:
  helius:
    api_key: "test_key"
    base_url: "http://localhost:8888"  # Mock server
  
  birdeye:
    api_key: "test_key"
    base_url: "http://localhost:8889"  # Mock server

# Transformer test configuration
ml:
  batch_size: 2  # Small batches for testing
  epochs: 1      # Quick training
```

### Production Environment

#### File: `config/config.production.yaml`
```yaml
app:
  name: "shyvr-rlte"
  version: "0.1.0"
  environment: "production"
  log_level: "${LOG_LEVEL:INFO}"
  debug: false
  host: "${HOST:0.0.0.0}"
  port: ${PORT:8080}
  
# Strict production database settings
database:
  host: "${DB_HOST}"  # Required - no default
  port: ${DB_PORT:5432}
  database: "${DB_NAME:shyvr_rlte}"
  username: "${DB_USER:rlte_user}"
  password: "${DB_PASSWORD}"  # Required from Secret Manager
  pool_size: ${DB_POOL_SIZE:20}  # Higher for production
  max_overflow: ${DB_MAX_OVERFLOW:30}
  echo: false  # Never log SQL in production

# Production security settings  
security:
  secret_key: "${SECRET_KEY}"  # Required from Secret Manager
  jwt_algorithm: "${JWT_ALGORITHM:HS256}"
  jwt_expiry_hours: ${JWT_EXPIRY_HOURS:12}  # Shorter for production
  api_key_length: ${API_KEY_LENGTH:32}
  rate_limit_storage: "${RATE_LIMIT_STORAGE:redis}"

# Production API configurations
apis:
  helius:
    api_key: "${HELIUS_API_KEY}"  # Required from Secret Manager
    base_url: "https://api.helius.xyz"
    rate_limit: ${HELIUS_RATE_LIMIT:100}
    timeout: ${HELIUS_TIMEOUT:30}
    
  birdeye:
    api_key: "${BIRDEYE_API_KEY}"  # Required from Secret Manager
    base_url: "https://public-api.birdeye.so"
    rate_limit: ${BIRDEYE_RATE_LIMIT:60}
    timeout: ${BIRDEYE_TIMEOUT:30}

# Conservative production trading settings
trading:
  modes:
    analysis: true
    simulation: true
    live: ${LIVE_TRADING_ENABLED:false}  # Explicitly controlled
  risk_management:
    max_position_size_pct: ${MAX_POSITION_SIZE:2.0}
    max_daily_loss_pct: ${MAX_DAILY_LOSS:3.0}
    max_drawdown_pct: ${MAX_DRAWDOWN:10.0}
    stop_loss_pct: ${STOP_LOSS:5.0}
    take_profit_pct: ${TAKE_PROFIT:15.0}
    max_open_positions: ${MAX_POSITIONS:3}
```

## Secret Management

### GCP Secret Manager Integration

#### 1. Secret Creation
```bash
# Create secrets in GCP Secret Manager
echo "your-telegram-token" | gcloud secrets create TELEGRAM_TOKEN --data-file=-
echo "your-webhook-secret" | gcloud secrets create WEBHOOK_SECRET --data-file=-
echo "your-db-password" | gcloud secrets create DB_PASSWORD --data-file=-
echo "your-helius-key" | gcloud secrets create HELIUS_API_KEY --data-file=-
echo "your-birdeye-key" | gcloud secrets create BIRDEYE_API_KEY --data-file=-
echo "your-openai-key" | gcloud secrets create OPENAI_API_KEY --data-file=-

# Create complex configuration secrets
cat << EOF | gcloud secrets create DATABASE_URL --data-file=-
postgresql://rlte_user:${DB_PASSWORD}@${DB_HOST}:5432/shyvr_rlte
EOF
```

#### 2. Secret Access Configuration
```bash
# Grant service account access to secrets
SERVICE_ACCOUNT="shyvr-rlte-service@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding TELEGRAM_TOKEN \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding DB_PASSWORD \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding HELIUS_API_KEY \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

#### 3. Cloud Run Secret Integration
```yaml
# Cloud Run service configuration
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: shyvr-rlte
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 15
      serviceAccountName: shyvr-rlte-service
      containers:
      - image: us-central1-docker.pkg.dev/PROJECT_ID/shyvr-ai-prod/shyvr-rlte:latest
        resources:
          limits:
            memory: 8Gi
            cpu: 6
        env:
        - name: TELEGRAM_TOKEN
          valueFrom:
            secretKeyRef:
              key: latest
              name: TELEGRAM_TOKEN
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              key: latest
              name: DB_PASSWORD
        - name: HELIUS_API_KEY
          valueFrom:
            secretKeyRef:
              key: latest
              name: HELIUS_API_KEY
```

### Secret Rotation Strategy

#### Automated Rotation Schedule
```bash
# Weekly rotation for high-sensitivity secrets
API_KEYS=("HELIUS_API_KEY" "BIRDEYE_API_KEY" "OPENAI_API_KEY")

# Monthly rotation for moderate-sensitivity secrets  
DB_SECRETS=("DB_PASSWORD" "DATABASE_URL")

# Quarterly rotation for stable secrets
APP_SECRETS=("SECRET_KEY" "JWT_SECRET")
```

#### Rotation Process
```bash
#!/bin/bash
# scripts/rotate_secrets.sh

rotate_secret() {
    local SECRET_NAME=$1
    local NEW_VALUE=$2
    
    # Create new version
    echo "$NEW_VALUE" | gcloud secrets versions add "$SECRET_NAME" --data-file=-
    
    # Wait for propagation
    sleep 30
    
    # Verify new version works
    gcloud run services replace service.yaml --region=us-central1
    
    # Check health after rotation
    curl -f https://your-service-url/health || {
        echo "Health check failed after rotation"
        # Rollback logic here
        exit 1
    }
    
    echo "Secret $SECRET_NAME rotated successfully"
}
```

## Transformer Model Configuration

### Model-Specific Settings

#### 1. iTransformer Configuration
```yaml
ml:
  transformers:
    itransformer:
      # Architecture parameters
      d_model: 256
      n_heads: 8
      n_layers: 6
      d_ff: 1024
      dropout: 0.1
      
      # Training parameters
      learning_rate: 0.0001
      batch_size: 32
      max_epochs: 100
      patience: 15
      
      # Inference parameters
      max_sequence_length: 512
      use_flash_attention: true
      torch_compile: true
      
      # Resource allocation
      memory_gb: 4
      cpu_cores: 2
      scaling_factor: 1.2
```

#### 2. PatchTST Configuration
```yaml
ml:
  transformers:
    patchtst:
      # Patch parameters
      patch_len: 16
      stride: 8
      
      # Architecture parameters
      d_model: 128
      n_heads: 8
      n_layers: 3
      d_ff: 256
      
      # Training parameters
      learning_rate: 0.0005
      batch_size: 64
      max_epochs: 150
      
      # Resource allocation
      memory_gb: 3
      cpu_cores: 2
      scaling_factor: 1.0
```

#### 3. TimesMixer Configuration
```yaml
ml:
  transformers:
    timesmixer:
      # Mixing parameters
      down_sampling_layers: 3
      down_sampling_window: 2
      
      # Architecture parameters
      d_model: 512
      n_heads: 16
      n_layers: 8
      d_ff: 2048
      
      # Training parameters
      learning_rate: 0.00005
      batch_size: 16
      max_epochs: 200
      
      # Resource allocation
      memory_gb: 5
      cpu_cores: 3
      scaling_factor: 1.5
```

#### 4. TimesFM Configuration
```yaml
ml:
  transformers:
    timesfm:
      # Foundation model parameters
      model_size: "200m"  # Options: 200m, 1.3b
      context_len: 512
      horizon_len: 96
      
      # Fine-tuning parameters
      learning_rate: 0.00001
      batch_size: 8
      max_epochs: 50
      freeze_backbone: true
      
      # Resource allocation
      memory_gb: 6
      cpu_cores: 4
      scaling_factor: 2.0
```

### Environment-Specific Model Configuration

#### Production Model Settings
```yaml
# Production transformer configuration
ml:
  # Conservative settings for stability
  batch_size: ${ML_BATCH_SIZE:32}  # Smaller batches for stability
  learning_rate: ${ML_LEARNING_RATE:0.0001}  # Conservative learning rate
  epochs: ${ML_EPOCHS:200}  # More training for production
  validation_split: ${ML_VALIDATION_SPLIT:0.2}
  early_stopping_patience: ${ML_EARLY_STOPPING:15}
  
  # Memory optimization
  low_cpu_mem_usage: true
  torch_dtype: "float16"
  device_map: "cpu"
  
  # Compilation settings
  torch_compile_mode: "${TORCH_COMPILE_MODE:reduce-overhead}"
  torch_inference_mode: true
  
  # Cache configuration
  transformers_cache: "${TRANSFORMERS_CACHE:/app/models/cache}"
  model_cache_size: ${MODEL_CACHE_SIZE:1000}  # MB
  
  # Performance settings
  tokenizers_parallelism: false
  omp_num_threads: ${OMP_NUM_THREADS:6}
  mkl_num_threads: ${MKL_NUM_THREADS:6}
```

## Database Configuration

### Production Database Settings

#### 1. Connection Configuration
```yaml
database:
  # Connection settings
  host: "${DB_HOST}"
  port: ${DB_PORT:5432}
  database: "${DB_NAME:shyvr_rlte}"
  username: "${DB_USER:rlte_user}"
  password: "${DB_PASSWORD}"
  
  # Connection pooling (production-optimized)
  pool_size: ${DB_POOL_SIZE:20}
  max_overflow: ${DB_MAX_OVERFLOW:30}
  pool_pre_ping: true
  pool_recycle: 3600  # 1 hour
  
  # SSL configuration
  sslmode: "require"
  sslcert: "${DB_SSL_CERT_PATH}"
  sslkey: "${DB_SSL_KEY_PATH}"
  sslrootcert: "${DB_SSL_CA_PATH}"
  
  # Query optimization
  echo: false  # Never log SQL in production
  echo_pool: false
  query_timeout: ${DB_QUERY_TIMEOUT:30}
  
  # Performance settings
  statement_timeout: "30s"
  lock_timeout: "10s"
  idle_in_transaction_session_timeout: "30s"
```

#### 2. Migration Configuration
```yaml
database:
  migrations:
    # Migration settings
    auto_upgrade: ${DB_AUTO_UPGRADE:false}
    backup_before_migration: true
    migration_timeout: 600  # 10 minutes
    
    # Rollback configuration
    enable_rollback: true
    rollback_versions: 3
    
    # Migration validation
    validate_before_apply: true
    dry_run_first: ${DB_DRY_RUN:true}
```

#### 3. Experience Database Configuration
```yaml
rl:
  experience_storage:
    enabled: true
    storage_backend: "database"
    max_experiences: ${RL_MAX_EXPERIENCES:100000}
    batch_size: ${RL_EXP_BATCH_SIZE:128}
    prioritized_replay: true
    
    performance:
      cache_size: ${RL_CACHE_SIZE:2000}
      async_operations: true
      compression: ${RL_COMPRESSION:true}
      query_timeout_seconds: ${RL_QUERY_TIMEOUT:45}
      batch_commit_size: ${RL_BATCH_COMMIT:200}
      
    database:
      pool_size: ${RL_DB_POOL_SIZE:10}
      max_overflow: ${RL_DB_OVERFLOW:15}
      timeout_seconds: ${RL_DB_TIMEOUT:45}
      enable_query_logging: false
      connection_retry_attempts: ${RL_DB_RETRIES:5}
      
    lifecycle:
      cleanup_enabled: true
      max_age_days: ${RL_MAX_AGE_DAYS:60}
      cleanup_interval_hours: ${RL_CLEANUP_INTERVAL:12}
      archive_old_experiences: ${RL_ARCHIVE:true}
      min_experiences_to_keep: ${RL_MIN_EXPERIENCES:5000}
```

## API Configuration

### External API Settings

#### 1. Helius API Configuration
```yaml
apis:
  helius:
    api_key: "${HELIUS_API_KEY}"
    base_url: "https://api.helius.xyz"
    rate_limit: ${HELIUS_RATE_LIMIT:100}
    timeout: ${HELIUS_TIMEOUT:30}
    retry_attempts: ${HELIUS_RETRIES:3}
    retry_backoff: ${HELIUS_BACKOFF:1.0}
    
    # Connection pooling
    max_connections: ${HELIUS_MAX_CONN:10}
    max_keepalive_connections: ${HELIUS_KEEPALIVE:5}
    keepalive_expiry: ${HELIUS_KEEPALIVE_EXPIRY:30}
    
    # Health check
    health_check_interval: ${HELIUS_HEALTH_INTERVAL:300}
    health_check_timeout: ${HELIUS_HEALTH_TIMEOUT:10}
```

#### 2. Birdeye API Configuration
```yaml
apis:
  birdeye:
    api_key: "${BIRDEYE_API_KEY}"
    base_url: "https://public-api.birdeye.so"
    rate_limit: ${BIRDEYE_RATE_LIMIT:60}
    timeout: ${BIRDEYE_TIMEOUT:30}
    
    # Request configuration
    max_retries: ${BIRDEYE_MAX_RETRIES:3}
    backoff_factor: ${BIRDEYE_BACKOFF:2.0}
    
    # Data freshness
    cache_ttl: ${BIRDEYE_CACHE_TTL:60}  # seconds
    max_age_tolerance: ${BIRDEYE_MAX_AGE:300}  # 5 minutes
```

#### 3. OpenAI API Configuration
```yaml
agent:
  model_type: "${AGENT_MODEL_TYPE:openai}"
  model_name: "${AGENT_MODEL_NAME:gpt-4o-mini}"
  api_key: "${OPENAI_API_KEY}"
  max_tokens: ${AGENT_MAX_TOKENS:1000}
  temperature: ${AGENT_TEMPERATURE:0.1}
  
  # Request settings
  timeout: ${OPENAI_TIMEOUT:60}
  max_retries: ${OPENAI_MAX_RETRIES:3}
  
  # Rate limiting
  requests_per_minute: ${OPENAI_RPM:100}
  tokens_per_minute: ${OPENAI_TPM:10000}
```

## Monitoring Configuration

### Production Monitoring Settings

#### 1. Application Monitoring
```yaml
monitoring:
  enabled: true
  metrics_port: ${METRICS_PORT:9090}
  health_check_port: ${HEALTH_CHECK_PORT:8081}
  log_level: "${MONITORING_LOG_LEVEL:INFO}"
  
  # Metrics collection
  collection_interval: ${METRICS_INTERVAL:30}  # seconds
  retention_period: ${METRICS_RETENTION:7}     # days
  
  # Health checks
  health_check_enabled: true
  health_check_interval: ${HEALTH_CHECK_INTERVAL:10}
  health_check_timeout: ${HEALTH_CHECK_TIMEOUT:5}
  
  # Performance monitoring
  enable_profiling: ${ENABLE_PROFILING:false}
  profile_sample_rate: ${PROFILE_SAMPLE_RATE:0.01}
```

#### 2. Alert Configuration
```yaml
monitoring:
  alerts:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL}"
    critical_threshold: ${CRITICAL_THRESHOLD:0.95}
    warning_threshold: ${WARNING_THRESHOLD:0.8}
    
    # Alert channels
    email_notifications: ${ALERT_EMAIL_ENABLED:true}
    slack_notifications: ${ALERT_SLACK_ENABLED:true}
    pagerduty_notifications: ${ALERT_PAGERDUTY_ENABLED:false}
    
    # Alert rules
    memory_threshold: ${ALERT_MEMORY_THRESHOLD:90}      # percent
    cpu_threshold: ${ALERT_CPU_THRESHOLD:85}            # percent
    error_rate_threshold: ${ALERT_ERROR_RATE:5}         # percent
    latency_threshold: ${ALERT_LATENCY_MS:1000}         # milliseconds
```

#### 3. Transformer-Specific Monitoring
```yaml
monitoring:
  transformers:
    # Model performance metrics
    track_inference_latency: true
    track_memory_usage: true
    track_cache_performance: true
    track_attention_patterns: ${TRACK_ATTENTION:false}
    
    # Alert thresholds
    memory_usage_threshold: ${TRANSFORMER_MEMORY_THRESHOLD:7000}  # MB
    inference_latency_threshold: ${TRANSFORMER_LATENCY_THRESHOLD:1000}  # ms
    cache_hit_rate_threshold: ${TRANSFORMER_CACHE_THRESHOLD:50}  # percent
    
    # Collection intervals
    metrics_collection_interval: 30  # seconds
    health_check_interval: 60       # seconds
```

## Security Configuration

### Production Security Settings

#### 1. Authentication Configuration
```yaml
security:
  # JWT configuration
  secret_key: "${SECRET_KEY}"
  jwt_algorithm: "${JWT_ALGORITHM:HS256}"
  jwt_expiry_hours: ${JWT_EXPIRY_HOURS:12}
  jwt_refresh_hours: ${JWT_REFRESH_HOURS:24}
  
  # API key configuration
  api_key_length: ${API_KEY_LENGTH:32}
  api_key_expiry_days: ${API_KEY_EXPIRY:90}
  
  # Rate limiting
  rate_limit_storage: "${RATE_LIMIT_STORAGE:redis}"
  rate_limit_per_minute: ${RATE_LIMIT_RPM:60}
  burst_capacity: ${BURST_CAPACITY:10}
  
  # Password policy
  min_password_length: ${MIN_PASSWORD_LENGTH:12}
  require_special_chars: ${REQUIRE_SPECIAL_CHARS:true}
  password_expiry_days: ${PASSWORD_EXPIRY:90}
```

#### 2. Network Security Configuration
```yaml
security:
  # CORS configuration
  cors_enabled: ${CORS_ENABLED:false}  # Disabled in production
  allowed_origins: ["https://app.shyvr.ai"]
  allowed_methods: ["GET", "POST"]
  allowed_headers: ["Authorization", "Content-Type"]
  
  # TLS configuration
  tls_enabled: true
  tls_version: "1.2"
  cipher_suites: ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]
  
  # Request validation
  max_request_size: ${MAX_REQUEST_SIZE:10485760}  # 10MB
  max_header_size: ${MAX_HEADER_SIZE:32768}       # 32KB
  request_timeout: ${REQUEST_TIMEOUT:30}          # seconds
```

#### 3. Data Protection Configuration
```yaml
security:
  # Encryption configuration
  encryption_key: "${ENCRYPTION_KEY}"
  encryption_algorithm: "AES-256-GCM"
  
  # Data masking
  mask_sensitive_data: ${MASK_SENSITIVE_DATA:true}
  mask_api_keys: ${MASK_API_KEYS:true}
  mask_personal_data: ${MASK_PERSONAL_DATA:true}
  
  # Audit logging
  audit_logging_enabled: ${AUDIT_LOGGING:true}
  audit_log_retention_days: ${AUDIT_RETENTION:90}
  audit_log_level: "${AUDIT_LOG_LEVEL:INFO}"
```

## Environment Variable Management

### Environment Variable Template

#### File: `config/env.production.template`
```bash
# Shyvr RLTE Production Environment Variables Template
# Copy this file to .env and fill in the values

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
HOST=0.0.0.0
PORT=8080

# Database Configuration (Cloud SQL)
DB_HOST=10.x.x.x  # Private IP of Cloud SQL instance
DB_PORT=5432
DB_NAME=shyvr_rlte
DB_USER=rlte_user
DB_PASSWORD=  # Set via Secret Manager
DATABASE_URL=  # Set via Secret Manager

# Database Pool Settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_QUERY_TIMEOUT=30

# Security Settings
SECRET_KEY=  # Set via Secret Manager - 64 character random string
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=12
API_KEY_LENGTH=32

# Telegram Configuration
TELEGRAM_TOKEN=  # Set via Secret Manager
WEBHOOK_SECRET=  # Set via Secret Manager

# External API Keys
HELIUS_API_KEY=  # Set via Secret Manager
BIRDEYE_API_KEY=  # Set via Secret Manager
ETHERSCAN_API_KEY=  # Set via Secret Manager
OPENAI_API_KEY=  # Set via Secret Manager
XAI_API_KEY=  # Set via Secret Manager

# Trading Configuration
LIVE_TRADING_ENABLED=false  # Explicitly set to false initially
PAPER_TRADING_ENABLED=true
MAX_POSITION_SIZE=2.0
MAX_DAILY_LOSS=3.0
MAX_DRAWDOWN=10.0

# ML Configuration
ML_BATCH_SIZE=32
ML_LEARNING_RATE=0.0001
ML_EPOCHS=200
ML_EARLY_STOPPING=15

# RL Configuration
RL_TRAINING_EPISODES=20000
RL_EPSILON_START=0.5
RL_EPSILON_END=0.01
RL_LEARNING_RATE=0.00005
RL_BATCH_SIZE=128
RL_MEMORY_SIZE=100000

# Transformer Configuration
TRANSFORMER_OPTIMIZED=true
TRANSFORMER_BATCH_SIZE=1
TRANSFORMER_MAX_LENGTH=512
TORCH_COMPILE_MODE=reduce-overhead
TRANSFORMERS_CACHE=/app/models/cache
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=6
MKL_NUM_THREADS=6
TORCH_INFERENCE_MODE=1

# Model Preservation
MODEL_PRESERVATION_ENABLED=false  # Enable when GCS is configured
GCS_BUCKET=  # Your GCS bucket name
BACKUP_INTERVAL=4
MAX_MODEL_VERSIONS=20

# Monitoring Configuration
METRICS_PORT=9090
HEALTH_CHECK_PORT=8081
MONITORING_LOG_LEVEL=INFO
ALERT_WEBHOOK_URL=  # Set via Secret Manager
GRAFANA_ADMIN_PASSWORD=  # Set via Secret Manager

# Feature Flags
ML_PREDICTIONS_ENABLED=true
RL_DECISIONS_ENABLED=true
RISK_MANAGEMENT_ENABLED=true
AUTOMATED_STOPS_ENABLED=true
EMERGENCY_SHUTDOWN_ENABLED=true

# Performance Settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
```

### Environment Variable Validation

#### Validation Script
```python
# scripts/validate_environment.py
import os
import sys
from typing import Dict, List, Optional

REQUIRED_VARS = {
    # Critical variables that must be present
    'production': [
        'ENVIRONMENT',
        'DB_HOST',
        'DB_PASSWORD',
        'SECRET_KEY',
        'TELEGRAM_TOKEN',
        'HELIUS_API_KEY',
        'BIRDEYE_API_KEY',
    ],
    'staging': [
        'ENVIRONMENT',
        'DB_HOST',
        'DB_PASSWORD',
        'SECRET_KEY',
        'TELEGRAM_TOKEN',
    ],
    'test': [
        'ENVIRONMENT',
    ]
}

VALIDATED_FORMATS = {
    # Variables with specific format requirements
    'ENVIRONMENT': ['development', 'test', 'staging', 'production'],
    'LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    'LIVE_TRADING_ENABLED': ['true', 'false'],
    'DEBUG': ['true', 'false'],
}

def validate_environment(environment: str = 'production') -> bool:
    """Validate environment variables for the specified environment."""
    errors = []
    warnings = []
    
    # Check required variables
    for var in REQUIRED_VARS.get(environment, []):
        if not os.getenv(var):
            errors.append(f"Required variable {var} is not set")
    
    # Check format validation
    for var, valid_values in VALIDATED_FORMATS.items():
        value = os.getenv(var)
        if value and value not in valid_values:
            errors.append(f"Variable {var} has invalid value: {value}")
    
    # Check secret lengths
    secret_vars = ['SECRET_KEY', 'TELEGRAM_TOKEN', 'DB_PASSWORD']
    for var in secret_vars:
        value = os.getenv(var)
        if value and len(value) < 16:
            warnings.append(f"Variable {var} appears to be too short (< 16 chars)")
    
    # Report results
    if errors:
        print("❌ Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️  Environment validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Environment validation passed")
    return True

if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 1 else 'production'
    success = validate_environment(env)
    sys.exit(0 if success else 1)
```

## Configuration Validation

### Validation Procedures

#### 1. Pre-Deployment Validation
```bash
# Validate configuration before deployment
./scripts/validate_environment.py production
./scripts/validate_config.py config/config.production.yaml
./scripts/validate_secrets.py
```

#### 2. Runtime Validation
```python
# Built-in configuration validation
from src.utils.config import ConfigValidator

validator = ConfigValidator()

# Validate all configurations at startup
if not validator.validate_all():
    logger.error("Configuration validation failed")
    sys.exit(1)
    
# Validate specific sections
validator.validate_database_config()
validator.validate_api_config()
validator.validate_transformer_config()
validator.validate_security_config()
```

#### 3. Health Check Validation
```bash
# Configuration health checks
curl https://your-service-url/health/config

# Expected response
{
  "status": "healthy",
  "checks": {
    "database_config": "ok",
    "api_config": "ok",
    "transformer_config": "ok",
    "security_config": "ok",
    "secret_access": "ok"
  },
  "timestamp": "2025-08-06T10:00:00Z"
}
```

## Best Practices

### Configuration Management Best Practices

#### 1. Security Best Practices
```yaml
# ✅ Good practices
- Use Secret Manager for sensitive data
- Rotate secrets regularly
- Use environment-specific configurations
- Validate configurations at startup
- Never commit secrets to version control
- Use least-privilege access policies

# ❌ Avoid these practices
- Hardcoding secrets in configuration files
- Using the same secrets across environments
- Storing secrets in environment variables for production
- Overly permissive access policies
```

#### 2. Configuration Organization
```yaml
# ✅ Recommended structure
config/
├── base/
│   ├── app.yaml           # Application settings
│   ├── database.yaml      # Database configuration
│   └── monitoring.yaml    # Monitoring settings
├── environments/
│   ├── development.yaml   # Dev overrides
│   ├── staging.yaml       # Staging overrides
│   └── production.yaml    # Production overrides
└── templates/
    └── env.template       # Environment variable template
```

#### 3. Validation Strategy
```python
# Multi-level validation approach
1. Schema validation (YAML structure)
2. Type validation (data types)
3. Value validation (ranges, enums)
4. Dependency validation (required relationships)
5. Runtime validation (connectivity tests)
6. Security validation (secret access, permissions)
```

#### 4. Change Management
```yaml
configuration_changes:
  # Change approval process
  approval_required: true
  reviewers: ["tech-lead", "devops-lead"]
  
  # Change testing
  test_in_staging: true
  rollback_plan_required: true
  
  # Change deployment
  gradual_rollout: true
  monitoring_period: "24h"
  
  # Change documentation
  document_changes: true
  update_runbooks: true
```

### Environment-Specific Guidelines

#### Development Environment
- Use local services when possible
- Enable debug logging
- Use test API keys
- Disable security features that impede development

#### Test Environment
- Use in-memory databases for unit tests
- Mock external services
- Enable comprehensive logging
- Use deterministic configurations

#### Staging Environment
- Mirror production as closely as possible
- Use production-like secrets (but separate)
- Enable all monitoring
- Test with realistic data volumes

#### Production Environment
- Minimize configuration changes
- Use Secret Manager for all sensitive data
- Enable all security features
- Comprehensive monitoring and alerting

## Conclusion

This configuration management guide provides comprehensive instructions for managing the transformer-enabled RLTE system across all environments. Following these practices ensures secure, maintainable, and reliable configuration management.

### Key Takeaways
1. Use environment-specific configuration files
2. Leverage GCP Secret Manager for sensitive data
3. Implement comprehensive validation
4. Follow security best practices
5. Maintain clear documentation
6. Establish change management procedures

### Next Steps
1. Review the [Monitoring and Observability Guide](MONITORING_OBSERVABILITY_GUIDE.md)
2. Set up [Security and Access Control](SECURITY_ACCESS_CONTROL_GUIDE.md)
3. Familiarize yourself with the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-06  
**Maintained By**: ML Development Team