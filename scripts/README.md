# RLTE Scripts Directory

## Overview

The scripts directory contains all operational scripts for the RLTE (Reinforcement Learning Trading Engine) system. These scripts manage the complete lifecycle from data collection through model training to production deployment.

## Directory Structure

```
scripts/
├── data_collection/      # Corpus data collection scripts
├── training/             # Model training pipelines
├── deployment/          # Production deployment automation
├── validation/          # System validation and testing
├── monitoring/          # Performance and health monitoring
└── database/            # Database migrations and management
```

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Initialize database
psql -h localhost -U postgres -d shyvr_rlte -f scripts/init_db.sql

# Set up environment
export ENVIRONMENT=development
export SECRET_KEY="your-secret-key-here"
source .env
```

### 2. Collect Training Data
```bash
# Collect initial corpus (365 days of historical data)
./scripts/data_collection/collect_initial_corpus.sh production-multi

# Collect multi-granularity data
python scripts/collect_multi_granularity_corpus.py --mode production-multi
```

### 3. Train Models
```bash
# Train all models using corpus data
python scripts/training/train_all_models.py

# Or train specific models
python scripts/training/train_all_models.py --models lstm,transformer
```

### 4. Deploy to Production
```bash
# Deploy to staging first
python scripts/deploy_to_staging.py

# Run validation
./scripts/run_all_validations.sh

# Deploy to production
./scripts/build-container.sh
gcloud run deploy --image gcr.io/shvyr-ai-bots/shyvr-rlte:latest
```

## 📊 End-to-End Workflows

### Development Workflow

```mermaid
graph LR
    A[Local Data Collection] --> B[Generate Corpus]
    B --> C[Train Models]
    C --> D[Evaluate Performance]
    D --> E[Local Testing]
    E --> F[Commit Changes]
```

1. **Data Collection Phase**
   ```bash
   # Collect test data (30 days)
   ./scripts/data_collection/collect_initial_corpus.sh test --days 30
   
   # Process into corpus
   python scripts/data_collection/process_corpus.py
   ```

2. **Model Development**
   ```bash
   # Train with small dataset
   python scripts/training/train_all_models.py --mode development
   
   # View training reports
   open /tmp/training_reports/latest_report.pdf
   ```

3. **Testing**
   ```bash
   # Run unit tests
   python scripts/run_tests.py --unit
   
   # Run integration tests
   python scripts/run_integration_tests.py
   ```

### Production Workflow

```mermaid
graph TD
    A[Scheduled Collection] --> B[Multi-Granularity Corpus]
    B --> C[GCS Upload]
    C --> D[Unified Training Pipeline]
    D --> E[Model Validation]
    E --> F[A/B Testing]
    F --> G[Production Deployment]
    G --> H[Performance Monitoring]
```

1. **Automated Data Pipeline**
   ```bash
   # Run daily at 00:00 UTC via cron
   0 0 * * * /path/to/scripts/data_collection/collect_corpus.sh production-daily
   ```

2. **Model Training & Evaluation**
   ```bash
   # Weekly model retraining
   python scripts/training/train_all_models.py \
     --corpus-version latest \
     --upload-to-gcs \
     --generate-report
   ```

3. **Deployment Pipeline**
   ```bash
   # Pre-deployment validation
   python scripts/pre_deployment_validation.py
   
   # Deploy with blue-green strategy
   ./scripts/deployment/blue_green_deploy.sh
   
   # Post-deployment smoke tests
   python scripts/post_deployment_smoke_tests.py
   ```

## 📁 Script Categories

### Data Collection (`data_collection/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `collect_initial_corpus.sh` | Historical data collection | `./collect_initial_corpus.sh production-multi` |
| `collect_corpus.sh` | Incremental collection | `./collect_corpus.sh daily` |
| `process_corpus.py` | Feature engineering | `python process_corpus.py` |
| `export_to_gcs.py` | Upload to cloud storage | `python export_to_gcs.py` |

### Training (`training/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_all_models.py` | Unified training pipeline | `python train_all_models.py` |
| `evaluate_models.py` | Model evaluation | `python evaluate_models.py` |
| `generate_reports.py` | Training reports | `python generate_reports.py` |

### Validation Scripts

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `run_all_validations.sh` | Complete system check | Before deployment |
| `validate_api_endpoints.py` | API health check | After deployment |
| `validate_database_schema.py` | Schema verification | After migrations |
| `validate_ml_rl_models.py` | Model performance | After training |

### Monitoring Scripts

| Script | Purpose | Frequency |
|--------|---------|-----------|
| `production_health_check.py` | System health | Every 5 minutes |
| `benchmark_production_system.py` | Performance metrics | Daily |
| `setup_gcp_monitoring.py` | Cloud monitoring | Once (setup) |
| `setup_grafana_gcp.py` | Dashboard setup | Once (setup) |

### Database Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `init_db.sql` | Initial schema | `psql -f init_db.sql` |
| `run_migration.py` | Schema updates | `python run_migration.py` |
| `optimize_database_performance.py` | Performance tuning | `python optimize_database_performance.py` |

## 🔧 Configuration

### Environment Variables
```bash
# Required
ENVIRONMENT=production|staging|development
SECRET_KEY=<32+ character secret>
DATABASE_URL=postgresql://user:pass@host/db

# GCP
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCS_BUCKET=shyvr-models-prod

# API Keys
HELIUS_API_KEY=<your-key>
LUNARCRUSH_API_KEY=<your-key>
COINGECKO_API_KEY=<your-key>
```

### Configuration Files
- `config/config.yaml` - Main configuration
- `config/training_config.yaml` - Training parameters
- `config/deployment.yaml` - Deployment settings

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check Cloud SQL proxy
   ps aux | grep cloud_sql_proxy
   
   # Restart if needed
   ~/cloud-sql-proxy --port=5433 shvyr-ai-bots:us-central1:shyvr-rlte-db-prod &
   ```

2. **GCS Authentication Error**
   ```bash
   # Set credentials
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   
   # Verify access
   gsutil ls gs://shyvr-models-prod/
   ```

3. **Memory Issues During Training**
   ```bash
   # Reduce batch size
   python train_all_models.py --batch-size 16
   
   # Use subset of data
   python train_all_models.py --sample-size 10000
   ```

4. **API Rate Limits**
   ```bash
   # Use cached data
   export USE_CACHED_DATA=true
   
   # Reduce request rate
   export API_RATE_LIMIT=10  # requests per minute
   ```

## 📈 Performance Benchmarks

| Operation | Expected Time | Resource Usage |
|-----------|--------------|----------------|
| Initial corpus (365 days) | 2-3 hours | 8GB RAM, 50GB disk |
| Multi-granularity processing | 30-45 mins | 16GB RAM |
| Full model training | 4-6 hours | GPU recommended, 32GB RAM |
| Production deployment | 10-15 mins | Minimal |
| Health check | < 5 seconds | < 100MB RAM |

## 🔒 Security Considerations

- Never commit `.env` files or credentials
- Use Secret Manager for production secrets
- Rotate API keys regularly
- Enable audit logging for all operations
- Use service accounts with minimal permissions

## 📚 Related Documentation

- [Data Collection Guide](data_collection/README.md)
- [Training Pipeline](training/README.md)
- [Deployment Procedures](deployment/README.md)
- [API Documentation](../docs/api/README.md)
- [Model Architecture](../src/ml_analysis/README.md)

## 🚦 CI/CD Integration

Scripts are integrated with GitHub Actions:

```yaml
# .github/workflows/train.yml
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: python scripts/training/train_all_models.py
```

## 📊 Monitoring Dashboard

Access production metrics at:
- Grafana: https://monitoring.shyvr.ai
- GCP Console: https://console.cloud.google.com/monitoring
- Custom Dashboard: https://dashboard.shyvr.ai

## 🤝 Contributing

1. Create feature branch
2. Run validation scripts
3. Ensure tests pass
4. Update documentation
5. Submit PR with micro-commits

## 📝 License

Proprietary - Shyvr AI 2024