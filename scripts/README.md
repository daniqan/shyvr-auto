# Shyvr RLTE Validation Scripts

This directory contains comprehensive system validation scripts for ensuring deployment readiness of the Shyvr RLTE system.

## Overview

The validation system consists of multiple specialized scripts that verify different aspects of the system:

1. **Pre-deployment validation** - Overall system readiness
2. **Database validation** - Database connectivity and schema
3. **ML/RL model validation** - Machine learning and reinforcement learning components
4. **API endpoint validation** - API health and functionality
5. **Secret validation** - Secret availability and security
6. **Post-deployment smoke tests** - Deployment verification
7. **Orchestration** - Automated execution of all validations

## Quick Start

### Run All Validations (Recommended)

```bash
# Run all validations with default settings
./scripts/run_all_validations.sh

# Run with custom project ID and base URL
./scripts/run_all_validations.sh "your-project-id" "https://your-service-url"

# Skip API validation if service is not running
./scripts/run_all_validations.sh "your-project-id" "http://localhost:8080" true
```

### Individual Validation Scripts

```bash
# Pre-deployment comprehensive validation
python -m scripts.pre_deployment_validation

# Database schema and connectivity
python -m scripts.validate_database_schema --project-id shvyr-ai-bots

# ML/RL model verification
python -m scripts.validate_ml_rl_models

# Secret availability and security
python -m scripts.validate_secrets_comprehensive --project-id shvyr-ai-bots

# API endpoint health checks
python -m scripts.validate_api_endpoints --base-url http://localhost:8080

# Post-deployment smoke tests
python -m scripts.post_deployment_smoke_tests --base-url https://your-production-url
```

## Validation Scripts Detail

### 1. Pre-deployment Validation (`pre_deployment_validation.py`)

Comprehensive system-wide validation covering:
- Environment and configuration validation
- Python dependencies verification
- Database readiness check
- Secret availability verification
- ML/RL component initialization
- API structure validation
- Monitoring system setup

**Usage:**
```bash
python -m scripts.pre_deployment_validation [--output-file report.json]
```

**Exit Codes:**
- `0`: All validations passed, system ready
- `1`: Critical validations failed, deployment blocked

### 2. Database Schema Validation (`validate_database_schema.py`)

Validates database connectivity, schema structure, and performance:
- Connection establishment and credentials
- Table existence and column structure
- Index and constraint validation
- Data integrity checks
- Performance benchmarking
- RL-specific database features

**Usage:**
```bash
python -m scripts.validate_database_schema \
    --project-id shvyr-ai-bots \
    [--output-file report.json]
```

### 3. ML/RL Model Validation (`validate_ml_rl_models.py`)

Verifies machine learning and reinforcement learning components:
- ML/RL dependency verification
- Model manager functionality
- Feature engineering pipeline
- RL agent initialization
- Experience replay system
- Trading environment validation
- Model persistence testing

**Usage:**
```bash
python -m scripts.validate_ml_rl_models [--output-file report.json]
```

### 4. API Endpoint Validation (`validate_api_endpoints.py`)

Tests API endpoints, authentication, and performance:
- Basic endpoint connectivity
- Health endpoint detailed validation
- Dashboard API functionality
- Static asset serving
- Authentication system testing
- WebSocket endpoint detection
- Performance characteristics
- Error handling validation
- Security header checks

**Usage:**
```bash
python -m scripts.validate_api_endpoints \
    --base-url http://localhost:8080 \
    [--timeout 30] \
    [--output-file report.json]
```

### 5. Secret Validation (`validate_secrets_comprehensive.py`)

Comprehensive secret availability and security validation:
- Secret Manager client initialization
- Critical secret availability (database, Telegram, etc.)
- API key validation
- Trading key security assessment
- Secret format and strength validation
- Performance and versioning checks
- Security report generation

**Usage:**
```bash
python -m scripts.validate_secrets_comprehensive \
    --project-id shvyr-ai-bots \
    [--output-file report.json]
```

### 6. Post-deployment Smoke Tests (`post_deployment_smoke_tests.py`)

Verifies deployed system functionality:
- Basic connectivity testing
- Health endpoint validation
- API functionality verification
- Static asset availability
- Authentication system testing
- Database connectivity through API
- ML/RL system availability
- Performance characteristics
- Error handling validation
- Security measures verification

**Usage:**
```bash
python -m scripts.post_deployment_smoke_tests \
    --base-url https://your-production-url \
    [--timeout 30] \
    [--output-file report.json]
```

### 7. Validation Orchestration (`run_all_validations.py`)

Automated orchestration of all validation scripts:
- Sequential execution of validation scripts
- Timeout management and error handling
- Result aggregation and analysis
- Deployment readiness determination
- Comprehensive reporting
- Recommendation generation

**Usage:**
```bash
python -m scripts.run_all_validations \
    [--project-id shvyr-ai-bots] \
    [--base-url http://localhost:8080] \
    [--skip-api-validation] \
    [--output-file report.json]
```

## Output and Reports

### Report Files

All validation scripts generate detailed JSON reports in the `reports/` directory:

```
reports/
├── pre_deployment_validation_YYYYMMDD_HHMMSS.json
├── database_validation_YYYYMMDD_HHMMSS.json
├── ml_rl_model_validation_YYYYMMDD_HHMMSS.json
├── api_validation_YYYYMMDD_HHMMSS.json
├── secret_validation_YYYYMMDD_HHMMSS.json
├── smoke_test_YYYYMMDD_HHMMSS.json
└── validation_orchestration_YYYYMMDD_HHMMSS.json
```

### Report Structure

Each report contains:
- **Summary**: Overall results and statistics
- **Results by Category**: Detailed test results organized by component
- **Critical Failures**: High-priority issues blocking deployment
- **Performance Metrics**: Execution times and response times
- **Recommendations**: Action items and next steps

### Example Report Analysis

```json
{
  "validation_complete": true,
  "deployment_ready": true,
  "summary": {
    "total_tests": 45,
    "passed_tests": 43,
    "failed_tests": 2,
    "success_rate_percent": 95.6
  },
  "critical_failures": [],
  "recommendations": [
    "🎉 All validation steps passed! System is ready for deployment."
  ]
}
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Pre-deployment Validation
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run validations
        run: ./scripts/run_all_validations.sh
        env:
          GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports
          path: reports/
```

### Cloud Build Integration

```yaml
steps:
  - name: 'python:3.9'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        ./scripts/run_all_validations.sh ${PROJECT_ID}
    env:
      - 'PROJECT_ID=${PROJECT_ID}'
```

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   - Verify Cloud SQL proxy is running
   - Check database credentials in Secret Manager
   - Ensure network connectivity

2. **Secret Access Denied**
   - Verify service account has Secret Manager access
   - Check IAM permissions
   - Confirm project ID is correct

3. **ML/RL Model Loading Issues**
   - Verify PyTorch and dependencies are installed
   - Check available memory for model loading
   - Ensure model files are accessible

4. **API Validation Failures**
   - Confirm service is running and accessible
   - Check firewall and network configuration
   - Verify authentication requirements

### Debug Mode

Enable verbose logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
python -m scripts.pre_deployment_validation
```

### Manual Testing

For debugging specific components:
```bash
# Test database connection manually
python3 -c "
import asyncio
from scripts.validate_database_schema import DatabaseValidator
validator = DatabaseValidator()
result = asyncio.run(validator.get_database_connection())
print(result)
"

# Test secret access manually
python3 -c "
import asyncio
from scripts.validate_secrets_comprehensive import ComprehensiveSecretValidator
validator = ComprehensiveSecretValidator()
result = asyncio.run(validator.get_secret_value('DB_PASSWORD'))
print(result[0])  # Success/failure
"
```

## Performance Considerations

### Execution Times

Typical execution times for validation scripts:
- **Pre-deployment**: 2-5 minutes
- **Database**: 1-3 minutes
- **ML/RL Models**: 2-4 minutes
- **API Endpoints**: 1-3 minutes
- **Secrets**: 1-2 minutes
- **Smoke Tests**: 2-5 minutes

### Optimization Tips

1. **Parallel Execution**: Run non-dependent validations in parallel
2. **Caching**: Cache ML model loading for repeated runs
3. **Selective Validation**: Skip API validation if service not running
4. **Timeout Tuning**: Adjust timeouts based on environment performance

## Security Considerations

### Secret Handling

- Scripts never log secret values
- Reports contain only metadata about secrets
- Access logs are generated for audit purposes
- Minimum required permissions principle

### Network Security

- API validation uses read-only operations
- No destructive operations performed
- Respects rate limits and authentication
- HTTPS validation for production URLs

## Maintenance

### Regular Updates

- Review validation criteria quarterly
- Update dependency checks as new packages added
- Adjust performance thresholds based on infrastructure changes
- Update secret validation as new secrets added

### Version Compatibility

Scripts are compatible with:
- Python 3.9+
- FastAPI 0.68+
- PostgreSQL 12+
- Google Cloud SDK latest

## Support

For issues with validation scripts:
1. Check logs in `reports/` directory
2. Review troubleshooting section above
3. Consult deployment checklist in `docs/DEPLOYMENT_READINESS_CHECKLIST.md`
4. Contact development team with specific error messages and report files

---

**Last Updated**: 2025-01-30
**Script Version**: 1.0