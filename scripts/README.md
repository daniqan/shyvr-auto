# Shyvr RLTE Scripts Directory

## Overview

The scripts directory contains operational scripts for the Shyvr RLTE (Reinforcement Learning Trading Engine) system. These scripts manage the complete lifecycle from data collection through model training to production deployment and monitoring.

## Directory Structure

```
scripts/
├── data_collection/        # Data collection and corpus management
├── training/               # Model training pipelines
├── *.py                    # Core operational scripts
├── *.sh                    # Shell scripts for automation
├── *.sql                   # Database schemas and migrations
└── README.md              # This file
```

## Script Categories

### 🗄️ Database Management

| Script | Purpose | Usage |
|--------|---------|-------|
| `init_db.sql` | Database schema initialization | `psql -f init_db.sql` |
| `upgrade_activity_logging.sql` | Activity logging schema upgrade | `psql -f upgrade_activity_logging.sql` |
| `run_migration.py` | Database migration runner | `python run_migration.py` |
| `run_production_migrations.py` | Production migration runner | `python run_production_migrations.py` |
| `optimize_database_performance.py` | Database performance optimization | `python optimize_database_performance.py` |

### 📊 Data Collection & Management

| Script | Purpose | Usage |
|--------|---------|-------|
| `collect_multi_granularity_corpus.py` | Multi-timeframe corpus collection | `python collect_multi_granularity_corpus.py --config config.yaml` |
| `collect_initial_corpus.py` | Single-timeframe corpus collection | `python collect_initial_corpus.py --production` |
| `collect_pepe_only.py` | PEPE token specific collection | `python collect_pepe_only.py` |
| `direct_corpus_collection.py` | Direct collection without pipeline | `python direct_corpus_collection.py` |
| `store_remaining_corpus.py` | Store remaining corpus data | `python store_remaining_corpus.py` |
| `investigate_data_gaps.py` | Identify and analyze data gaps | `python investigate_data_gaps.py` |
| `analyze_experience_growth.py` | RL experience data analysis | `python analyze_experience_growth.py` |

### 🧠 Model Training

| Script | Purpose | Usage |
|--------|---------|-------|
| `training/train_all_models.py` | Unified model training pipeline | `python training/train_all_models.py` |
| `training/train_transformers.py` | Transformer-specific training | `python training/train_transformers.py` |

### 🧪 Testing & Validation

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_all_validations.py` | Orchestrates all validation scripts | `python run_all_validations.py` |
| `run_all_validations.sh` | Shell wrapper for validations | `./run_all_validations.sh` |
| `validate_api_endpoints.py` | API endpoint health validation | `python validate_api_endpoints.py --base-url http://localhost:8080` |
| `validate_database_schema.py` | Database schema validation | `python validate_database_schema.py --project-id shvyr-ai-bots` |
| `validate_ml_rl_models.py` | ML/RL model validation | `python validate_ml_rl_models.py` |
| `validate_monitoring.py` | Monitoring system validation | `python validate_monitoring.py` |
| `validate_dashboard_setup.py` | Dashboard configuration validation | `python validate_dashboard_setup.py` |
| `validate_secrets_comprehensive.py` | Secrets and security validation | `python validate_secrets_comprehensive.py` |

### 📈 Monitoring & Health Checks

| Script | Purpose | Usage |
|--------|---------|-------|
| `production_health_check.py` | Comprehensive production health check | `python production_health_check.py` |
| `benchmark_production_system.py` | Production system benchmarking | `python benchmark_production_system.py --scenario all` |
| `setup_production_monitoring.py` | Production monitoring setup | `python setup_production_monitoring.py` |
| `setup_gcp_monitoring.py` | Google Cloud monitoring setup | `python setup_gcp_monitoring.py` |
| `setup_grafana_gcp.py` | Grafana dashboard setup | `python setup_grafana_gcp.py` |
| `setup_experience_monitoring.py` | RL experience monitoring setup | `python setup_experience_monitoring.py` |
| `manage_experience_lifecycle.py` | RL experience data lifecycle | `python manage_experience_lifecycle.py` |

### 🚀 Deployment & Infrastructure

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy_to_staging.py` | Staging environment deployment | `python deploy_to_staging.py` |
| `build-container.sh` | Docker container build automation | `./build-container.sh build --tag v1.0.0` |
| `pre_deployment_validation.py` | Pre-deployment system validation | `python pre_deployment_validation.py` |
| `post_deployment_smoke_tests.py` | Post-deployment verification | `python post_deployment_smoke_tests.py` |
| `set_webhook.sh` | Webhook configuration | `./set_webhook.sh` |
| `manage_production_backups.py` | Production backup management | `python manage_production_backups.py` |

### 🧪 Testing Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_collection_setup.py` | Data collection setup testing | `python test_collection_setup.py` |
| `test_multi_granularity.py` | Multi-granularity collection testing | `python test_multi_granularity.py` |
| `test_pagination.py` | Pagination system testing | `python test_pagination.py` |
| `test_production_performance.py` | Production performance testing | `python test_production_performance.py` |
| `test_sol_collection.py` | Solana collection testing | `python test_sol_collection.py` |
| `run_tests.py` | General test runner | `python run_tests.py` |
| `run_integration_tests.py` | Integration test suite | `python run_integration_tests.py` |
| `run_transformer_integration_tests.py` | Transformer integration tests | `python run_transformer_integration_tests.py` |
| `run_performance_tests.py` | Performance test suite | `python run_performance_tests.py` |

### 🔧 Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_direct_collection.sh` | Direct collection shell wrapper | `./run_direct_collection.sh` |
| `setup_basic_monitoring.sh` | Basic monitoring setup | `./setup_basic_monitoring.sh` |
| `run_phase_9_validation.py` | Phase 9 validation runner | `python run_phase_9_validation.py` |

## 📁 Subdirectories

### data_collection/
Contains specialized data collection scripts and utilities:
- `collect_corpus.sh` - Unified corpus collection runner
- `check_corpus_features.py` - Feature validation
- `clean_corpus_data.py` - Data cleanup utilities
- `export_corpus_to_gcs.py` - GCS export functionality
- `verify_corpus_data.py` - Data verification tools

### training/
Contains model training scripts and configurations:
- `train_all_models.py` - Unified training pipeline for all models
- `train_transformers.py` - Transformer-specific training
- `config.yaml` - Training configuration

## 🚀 Quick Start Workflows

### 1. Initial System Setup
```bash
# Initialize database
psql -h localhost -U postgres -d shyvr_rlte -f scripts/init_db.sql

# Set up environment
export ENVIRONMENT=development
export PROJECT_ID=shvyr-ai-bots
```

### 2. Data Collection
```bash
# Test collection (2 tokens, 7 days)
./scripts/data_collection/collect_corpus.sh test

# Multi-granularity collection
python scripts/collect_multi_granularity_corpus.py --config config/corpus_collection.yaml

# Production collection with full pagination
./scripts/data_collection/collect_corpus.sh production-multi
```

### 3. Model Training
```bash
# Train all models using collected corpus
python scripts/training/train_all_models.py

# Train specific transformer models
python scripts/training/train_transformers.py --models itransformer,patchtst
```

### 4. Validation & Testing
```bash
# Run complete validation suite
python scripts/run_all_validations.py

# Validate API endpoints
python scripts/validate_api_endpoints.py --base-url http://localhost:8080

# Check production health
python scripts/production_health_check.py
```

### 5. Deployment
```bash
# Deploy to staging
python scripts/deploy_to_staging.py

# Build production container
./scripts/build-container.sh build --tag v1.2.0 --test --push

# Run pre-deployment validation
python scripts/pre_deployment_validation.py
```

## 🔧 Configuration Requirements

### Environment Variables
```bash
# Core configuration
ENVIRONMENT=production|staging|development
PROJECT_ID=shvyr-ai-bots
DATABASE_URL=postgresql://user:pass@host/db

# GCP Configuration  
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCS_BUCKET=shyvr-models-prod

# API Keys (stored in Secret Manager)
HELIUS_API_KEY=<from-secret-manager>
COINGECKO_API_KEY=<from-secret-manager>
```

### Database Connection
Most scripts expect Cloud SQL Proxy running on port 5433:
```bash
~/cloud-sql-proxy --port=5433 shvyr-ai-bots:us-central1:shyvr-rlte-db-prod &
```

## 📊 Key Features

### Multi-Granularity Data Collection
- Supports multiple timeframes: daily, 4-hour, hourly, 15-minute
- Dynamic pagination for comprehensive historical data
- Checkpoint/resume capability for long-running collections
- Automatic gap detection and filling

### Comprehensive Validation
- Database schema and connectivity validation
- ML/RL model loading and functionality checks
- API endpoint health verification
- Security and secrets validation
- Performance benchmarking

### Production Monitoring
- Real-time health checks with transformer-specific monitoring
- Performance benchmarking with configurable scenarios
- GCP monitoring integration
- Grafana dashboard automation

### Unified Training Pipeline
- Supports LSTM and multiple Transformer variants
- Time-series aware train/validation/test splitting
- GCS integration for model storage
- Comprehensive training reports

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check Cloud SQL proxy
   ps aux | grep cloud_sql_proxy
   
   # Restart if needed  
   ~/cloud-sql-proxy --port=5433 shvyr-ai-bots:us-central1:shyvr-rlte-db-prod &
   ```

2. **API Rate Limits**
   ```bash
   # Use conservative rate limiting
   export API_RATE_LIMIT=2.1  # seconds between requests
   ```

3. **Memory Issues During Training**
   ```bash
   # Reduce batch size in config
   python train_all_models.py --batch-size 16
   ```

4. **GCS Authentication**
   ```bash
   # Set credentials
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   
   # Verify access
   gsutil ls gs://shyvr-models-prod/
   ```

## 📈 Performance Guidelines

| Operation | Expected Time | Resource Usage |
|-----------|---------------|----------------|
| Multi-granularity collection (365 days) | 3-4 hours | 8GB RAM, 50GB disk |
| Model training (all models) | 2-3 hours | GPU recommended, 16GB RAM |
| Complete validation suite | 10-15 minutes | 2GB RAM |
| Production health check | < 30 seconds | < 500MB RAM |
| Database migration | 1-5 minutes | Minimal |

## 🔒 Security Considerations

- All secrets managed via Google Cloud Secret Manager
- Database credentials never stored in code
- API keys rotated regularly via automation
- All operations logged for audit purposes
- Non-root container execution enforced

## 📚 Related Documentation

- [Data Collection Guide](data_collection/README.md)
- [Training Pipeline](training/README.md)
- [Production Deployment Guide](../docs/deployment.md)
- [API Documentation](../docs/api.md)

## 🤝 Contributing

1. Create feature branch from `main`
2. Run validation scripts before committing
3. Follow micro-commit structure
4. Update documentation for new scripts
5. Ensure all tests pass before PR submission

---

**Note**: This documentation reflects the actual scripts present in the directory as of the last update. Always verify script existence and parameters before use.