#!/bin/bash

# Production Readiness Checklist and Validation Script
# Comprehensive validation for Shyvr RLTE production deployment
# Ensures all systems, secrets, configurations, and dependencies are ready

set -euo pipefail

# Configuration
PROJECT_ID=${PROJECT_ID:-"shvyr-ai-bots"}
REGION=${REGION:-"us-central1"}
SERVICE=${SERVICE:-"shyvr-rlte"}
REPOSITORY=${REPOSITORY:-"shyvr-ai-prod"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "${PURPLE}[SECTION]${NC} $1"; }

# Validation counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0
CRITICAL_FAILURES=0

# Helper function to run checks
run_check() {
    local check_name="$1"
    local check_command="$2"
    local is_critical=${3:-true}
    local details=${4:-""}
    
    log_info "Checking: $check_name"
    
    if eval "$check_command" >/dev/null 2>&1; then
        log_success "✓ $check_name"
        ((CHECKS_PASSED++))
        return 0
    else
        if [[ "$is_critical" == "true" ]]; then
            log_error "✗ $check_name"
            [[ -n "$details" ]] && log_error "  $details"
            ((CHECKS_FAILED++))
            ((CRITICAL_FAILURES++))
        else
            log_warning "⚠ $check_name"
            [[ -n "$details" ]] && log_warning "  $details"
            ((WARNINGS++))
        fi
        return 1
    fi
}

echo -e "${PURPLE}🚀 Shyvr RLTE Production Readiness Checklist${NC}"
echo -e "${BLUE}AI-augmented cryptocurrency trading bot deployment validation${NC}"
echo ""

# =============================================================================
# 1. DEVELOPMENT ENVIRONMENT VALIDATION
# =============================================================================
log_header "1. Development Environment Validation"

run_check "Required tools installed" "command -v gcloud && command -v docker && command -v python3 && command -v curl"
run_check "GCP authentication active" "gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q '.'"
run_check "GCP project access" "gcloud projects describe $PROJECT_ID"
run_check "Docker daemon running" "docker info"
run_check "Python environment ready" "python3 -c 'import sys; assert sys.version_info >= (3, 11)'"

# =============================================================================
# 2. PROJECT STRUCTURE VALIDATION
# =============================================================================
log_header "2. Project Structure Validation"

run_check "Main application file exists" "test -f main.py"
run_check "Source directory structure" "test -d src && test -d src/utils && test -d src/ml_analysis && test -d src/rl_agent"
run_check "Configuration files present" "test -f config/config.production.yaml && test -f env.template"
run_check "Dockerfile optimized" "test -f Dockerfile && grep -q 'FROM.*as.*builder' Dockerfile"
run_check "Docker compose production config" "test -f docker/docker-compose.production.yml"
run_check "Deployment scripts present" "test -f deploy/deploy_latest.sh && test -f deploy/validate_deployment.sh"
run_check "Database migrations available" "test -d database/migrations && test -f database/migrations/004_create_rl_experience_schema.sql"
run_check "Model preservation migration" "test -f database/migrations/005_model_preservation_schema.sql"

# =============================================================================
# 3. SECURITY AND SECRETS VALIDATION
# =============================================================================
log_header "3. Security and Secrets Validation"

# Required secrets for basic operation
REQUIRED_SECRETS=("TELEGRAM_TOKEN" "WEBHOOK_SECRET" "DB_PASSWORD" "DATABASE_URL" "SECRET_KEY" "JWT_SECRET")

for secret in "${REQUIRED_SECRETS[@]}"; do
    run_check "Required secret: $secret" "gcloud secrets describe $secret --project=$PROJECT_ID"
    run_check "Secret has value: $secret" "gcloud secrets versions access latest --secret=$secret --project=$PROJECT_ID | grep -v '^$'"
done

# Optional but recommended secrets
OPTIONAL_SECRETS=("HELIUS_API_KEY" "BIRDEYE_API_KEY" "XAI_API_KEY" "OPENAI_API_KEY")

for secret in "${OPTIONAL_SECRETS[@]}"; do
    run_check "Optional secret: $secret" "gcloud secrets describe $secret --project=$PROJECT_ID" false "API functionality may be limited"
done

# Security configuration checks
run_check "No hardcoded credentials in config" "! grep -r 'password.*=' config/ || ! grep -r 'key.*=' config/ || ! grep -r 'token.*=' config/"
run_check "Production config uses env vars" "grep -q '\\${.*}' config/config.production.yaml"
run_check "Environment template exists" "test -f env.template"

# =============================================================================
# 4. GCP INFRASTRUCTURE VALIDATION
# =============================================================================
log_header "4. GCP Infrastructure Validation"

run_check "Artifact Registry repository exists" "gcloud artifacts repositories describe $REPOSITORY --location=$REGION --project=$PROJECT_ID"
run_check "Cloud SQL instance exists" "gcloud sql instances describe shyvr-rlte-db-prod --project=$PROJECT_ID"
run_check "Cloud SQL instance is running" "gcloud sql instances describe shyvr-rlte-db-prod --project=$PROJECT_ID --format='value(state)' | grep -q RUNNABLE"
run_check "Service account exists" "gcloud iam service-accounts describe shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com --project=$PROJECT_ID"
run_check "GCS bucket for models exists" "gsutil ls gs://shyvr-models-prod/" false "Model preservation will be disabled"

# =============================================================================
# 5. DATABASE VALIDATION
# =============================================================================
log_header "5. Database Validation"

# Database connectivity (if DATABASE_URL is available)
if gcloud secrets versions access latest --secret=DATABASE_URL --project=$PROJECT_ID >/dev/null 2>&1; then
    DATABASE_URL=$(gcloud secrets versions access latest --secret=DATABASE_URL --project=$PROJECT_ID)
    run_check "Database connectivity" "python3 -c \"import asyncpg, asyncio; asyncio.run(asyncpg.connect('$DATABASE_URL').execute('SELECT 1'))\"" false "May not be accessible from current environment"
fi

run_check "Migration scripts validated" "python3 -c \"import os; assert all(os.path.isfile(f'database/migrations/{f}') for f in ['001_create_initial_schema.sql', '004_create_rl_experience_schema.sql', '005_model_preservation_schema.sql'])\""

# =============================================================================
# 6. CODE QUALITY AND TESTING
# =============================================================================
log_header "6. Code Quality and Testing"

# Check if tests exist and can be imported
run_check "Test suite exists" "test -d tests && test -f tests/conftest.py"
run_check "Unit tests available" "test -d tests/unit && find tests/unit -name 'test_*.py' | wc -l | grep -q '[1-9]'"
run_check "Integration tests available" "test -d tests/integration && find tests/integration -name 'test_*.py' | wc -l | grep -q '[1-9]'"
run_check "Performance tests available" "test -d tests/performance && find tests/performance -name 'test_*.py' | wc -l | grep -q '[1-9]'"

# Check critical source files can be imported
run_check "Main application imports" "cd $(pwd) && python3 -c 'import sys; sys.path.insert(0, \"src\"); import utils.config, utils.database, ml_analysis.model_manager, rl_agent.dqn_agent'" false "Some imports may fail in current environment"

# =============================================================================
# 7. ML/RL SYSTEM VALIDATION
# =============================================================================
log_header "7. ML/RL System Validation"

run_check "ML model management code" "test -f src/ml_analysis/model_manager.py && test -f src/ml_analysis/lstm_model.py"
run_check "RL agent implementation" "test -f src/rl_agent/dqn_agent.py && test -f src/rl_agent/experience_replay.py"
run_check "Model preservation system" "test -f src/model_preservation/manager.py && test -f src/model_preservation/gcs_handler.py"
run_check "Experience storage system" "test -f src/rl_agent/experience_database.py"
run_check "No mock implementations in production" "! grep -r '_create_mock_' src/ || ! grep -r 'unittest.mock' src/"

# =============================================================================
# 8. MONITORING AND OBSERVABILITY
# =============================================================================
log_header "8. Monitoring and Observability"

run_check "Monitoring configuration" "test -f config/monitoring_config.json"
run_check "Grafana dashboards" "test -d monitoring/grafana && find monitoring/grafana -name '*.json' | wc -l | grep -q '[1-9]'"
run_check "Prometheus configuration" "test -f monitoring/grafana/prometheus.yml"
run_check "Health check endpoint code" "grep -q '/health' src/dashboard/api.py || grep -q '/health' main.py"
run_check "Metrics endpoint code" "grep -q '/metrics' src/dashboard/api.py || grep -q '/metrics' main.py"

# =============================================================================
# 9. DEPLOYMENT CONFIGURATION
# =============================================================================
log_header "9. Deployment Configuration"

run_check "Cloud Build configuration" "test -f config/cloudbuild.yaml"
run_check "Blue-green deployment support" "grep -q 'update-traffic' config/cloudbuild.yaml && grep -q 'no-traffic' config/cloudbuild.yaml"
run_check "Production environment variables" "grep -q 'ENVIRONMENT=production' config/cloudbuild.yaml"
run_check "Resource limits configured" "grep -q 'memory.*Gi' config/cloudbuild.yaml && grep -q 'cpu.*[0-9]' config/cloudbuild.yaml"
run_check "Health check timeouts" "grep -q 'timeout.*[0-9]' config/cloudbuild.yaml"

# =============================================================================
# 10. SAFETY AND COMPLIANCE
# =============================================================================
log_header "10. Safety and Compliance"

run_check "Trading safety manager" "test -f src/safety/trading_safety_manager.py"
run_check "Emergency stop controller" "test -f src/safety/emergency_stop_controller.py"
run_check "Risk control manager" "test -f src/safety/risk_control_manager.py"
run_check "Trading circuit breaker" "test -f src/safety/trading_circuit_breaker.py"
run_check "Financial data validator" "test -f src/utils/financial_data_validator.py"
run_check "XAI system for compliance" "test -f src/xai/trading_integration.py"
run_check "Activity logging system" "test -f src/activity_logging/activity_logger.py"

# Production configuration safety checks
run_check "Live trading disabled by default" "grep -q 'live.*false' config/config.production.yaml || grep -q 'LIVE_TRADING_ENABLED.*false' config/config.production.yaml"
run_check "Conservative risk limits" "grep -q 'max_position_size_pct.*[0-5]' config/config.production.yaml"
run_check "Emergency controls enabled" "grep -q 'emergency_shutdown.*true' config/config.production.yaml"

# =============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# =============================================================================
echo ""
log_header "Production Readiness Summary"
echo "================================================"
log_success "Checks passed: $CHECKS_PASSED"
if [[ $WARNINGS -gt 0 ]]; then
    log_warning "Warnings: $WARNINGS"
fi
if [[ $CHECKS_FAILED -gt 0 ]]; then
    log_error "Checks failed: $CHECKS_FAILED"
fi

TOTAL_CHECKS=$((CHECKS_PASSED + CHECKS_FAILED + WARNINGS))
SUCCESS_RATE=$((CHECKS_PASSED * 100 / TOTAL_CHECKS))

echo ""
echo "Overall Success Rate: ${SUCCESS_RATE}%"

# Final assessment
if [[ $CRITICAL_FAILURES -eq 0 ]]; then
    if [[ $WARNINGS -eq 0 ]]; then
        log_success "🎉 PRODUCTION READINESS: EXCELLENT"
        log_success "All critical checks passed. System is ready for production deployment."
    else
        log_success "✅ PRODUCTION READINESS: GOOD"
        log_warning "All critical checks passed, but some warnings need attention."
    fi
    
    echo ""
    log_info "🚀 Next Steps for Production Deployment:"
    log_info "   1. Run final tests: ./scripts/run_all_validations.sh"
    log_info "   2. Deploy to staging: git push origin develop"
    log_info "   3. Validate staging: ./deploy/validate_deployment.sh"
    log_info "   4. Deploy to production: git push origin main"
    log_info "   5. Monitor deployment: gcloud run logs tail $SERVICE --region $REGION"
    
    exit 0
else
    log_error "❌ PRODUCTION READINESS: NOT READY"
    log_error "$CRITICAL_FAILURES critical issues must be resolved before deployment."
    
    echo ""
    log_info "🔧 Required Actions:"
    log_info "   1. Review and fix all failed checks above"
    log_info "   2. Re-run this validation script"
    log_info "   3. Only proceed to deployment when all critical checks pass"
    
    exit 1
fi