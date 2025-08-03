#!/bin/bash

# Phase 8.2: Final Production Validation
# Comprehensive pre-production validation for Shyvr RLTE
# This script validates ALL Phase 1-8 implementations for production readiness

set -euo pipefail

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-shyvr-rlte}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-shyvr-rlte}"
VALIDATION_LOG="/tmp/final_production_validation_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$VALIDATION_LOG"
}

error() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

info() {
    log "${BLUE}INFO: $1${NC}"
}

# Validation results tracking
VALIDATION_RESULTS=()
CRITICAL_FAILURES=0
WARNING_COUNT=0

record_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    VALIDATION_RESULTS+=("$test_name|$status|$details")
    
    if [[ "$status" == "FAIL" ]]; then
        ((CRITICAL_FAILURES++))
        error "$test_name: $details"
    elif [[ "$status" == "WARN" ]]; then
        ((WARNING_COUNT++))
        warning "$test_name: $details"
    else
        success "$test_name: $details"
    fi
}

# Phase 1: Configuration & Infrastructure Validation
validate_phase_1() {
    info "=== Phase 1: Configuration & Infrastructure Validation ==="
    
    # Check GCP project setup
    if gcloud config get-value project &>/dev/null; then
        current_project=$(gcloud config get-value project)
        if [[ "$current_project" == "$PROJECT_ID" ]]; then
            record_result "GCP_PROJECT_CONFIG" "PASS" "Project correctly set to $PROJECT_ID"
        else
            record_result "GCP_PROJECT_CONFIG" "FAIL" "Project mismatch: expected $PROJECT_ID, got $current_project"
        fi
    else
        record_result "GCP_PROJECT_CONFIG" "FAIL" "GCP project not configured"
    fi
    
    # Check required GCP APIs
    local required_apis=(
        "run.googleapis.com"
        "sql-component.googleapis.com"
        "secretmanager.googleapis.com"
        "cloudbuild.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "storage-component.googleapis.com"
    )
    
    for api in "${required_apis[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            record_result "GCP_API_$api" "PASS" "API $api is enabled"
        else
            record_result "GCP_API_$api" "FAIL" "API $api is not enabled"
        fi
    done
    
    # Check service account
    if gcloud iam service-accounts describe "shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com" &>/dev/null; then
        record_result "SERVICE_ACCOUNT" "PASS" "Service account exists"
    else
        record_result "SERVICE_ACCOUNT" "FAIL" "Service account not found"
    fi
    
    # Check configuration files
    local config_files=(
        "config/config.production.yaml"
        "config/cloudbuild.yaml"
        "config/monitoring_config.json"
        "Dockerfile"
        "pyproject.toml"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            record_result "CONFIG_FILE_$file" "PASS" "Configuration file exists"
        else
            record_result "CONFIG_FILE_$file" "FAIL" "Configuration file missing: $file"
        fi
    done
}

# Phase 2: Core Trading System Validation
validate_phase_2() {
    info "=== Phase 2: Core Trading System Validation ==="
    
    # Test core modules import
    local core_modules=(
        "src.wallet.ethereum_wallet"
        "src.wallet.solana_wallet"
        "src.dex.hyperliquid_client"
        "src.dex.jupiter_client"
        "src.dex.uniswap_v3_client"
        "src.portfolio.portfolio_manager"
        "src.portfolio.risk_manager"
        "src.portfolio.pnl_calculator"
    )
    
    for module in "${core_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            record_result "CORE_MODULE_$module" "PASS" "Module imports successfully"
        else
            record_result "CORE_MODULE_$module" "FAIL" "Module import failed: $module"
        fi
    done
    
    # Run core trading tests
    if uv run python -m pytest tests/unit/wallet/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "WALLET_TESTS" "PASS" "Wallet tests passed"
    else
        record_result "WALLET_TESTS" "FAIL" "Wallet tests failed"
    fi
    
    if uv run python -m pytest tests/unit/dex/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "DEX_TESTS" "PASS" "DEX integration tests passed"
    else
        record_result "DEX_TESTS" "FAIL" "DEX integration tests failed"
    fi
    
    if uv run python -m pytest tests/unit/portfolio/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "PORTFOLIO_TESTS" "PASS" "Portfolio management tests passed"
    else
        record_result "PORTFOLIO_TESTS" "FAIL" "Portfolio management tests failed"
    fi
}

# Phase 3: ML Analysis System Validation
validate_phase_3() {
    info "=== Phase 3: ML Analysis System Validation ==="
    
    # Test ML modules
    local ml_modules=(
        "src.ml_analysis.lstm_model"
        "src.ml_analysis.feature_engineer"
        "src.ml_analysis.market_data_aggregator"
        "src.ml_analysis.model_manager"
        "src.ml_analysis.inference_optimizer"
        "src.ml_analysis.batch_prediction"
    )
    
    for module in "${ml_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            record_result "ML_MODULE_$module" "PASS" "ML module imports successfully"
        else
            record_result "ML_MODULE_$module" "FAIL" "ML module import failed: $module"
        fi
    done
    
    # Run ML tests
    if uv run python -m pytest tests/unit/ml_analysis/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "ML_TESTS" "PASS" "ML analysis tests passed"
    else
        record_result "ML_TESTS" "FAIL" "ML analysis tests failed"
    fi
    
    # Test performance optimizations
    if uv run python -m pytest tests/unit/performance/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "ML_PERFORMANCE_TESTS" "PASS" "ML performance tests passed"
    else
        record_result "ML_PERFORMANCE_TESTS" "FAIL" "ML performance tests failed"
    fi
}

# Phase 4: RL Agent System Validation
validate_phase_4() {
    info "=== Phase 4: RL Agent System Validation ==="
    
    # Test RL modules
    local rl_modules=(
        "src.rl_agent.dqn_agent"
        "src.rl_agent.experience_replay"
        "src.rl_agent.experience_database"
        "src.rl_agent.trading_environment"
        "src.rl_agent.reward_engineering"
        "src.rl_agent.training_pipeline"
    )
    
    for module in "${rl_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            record_result "RL_MODULE_$module" "PASS" "RL module imports successfully"
        else
            record_result "RL_MODULE_$module" "FAIL" "RL module import failed: $module"
        fi
    done
    
    # Run RL tests
    if uv run python -m pytest tests/unit/rl_agent/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "RL_TESTS" "PASS" "RL agent tests passed"
    else
        record_result "RL_TESTS" "FAIL" "RL agent tests failed"
    fi
    
    # Test ML-RL integration
    if uv run python -m pytest tests/unit/integration/test_ml_rl_integration.py -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "ML_RL_INTEGRATION" "PASS" "ML-RL integration tests passed"
    else
        record_result "ML_RL_INTEGRATION" "FAIL" "ML-RL integration tests failed"
    fi
}

# Phase 5: Safety & Risk Management Validation
validate_phase_5() {
    info "=== Phase 5: Safety & Risk Management Validation ==="
    
    # Test safety modules
    local safety_modules=(
        "src.safety.emergency_stop_controller"
        "src.safety.risk_control_manager"
        "src.safety.trading_circuit_breaker"
        "src.safety.trading_safety_manager"
        "src.modes.production_safety_checks"
        "src.modes.cross_mode_safety"
    )
    
    for module in "${safety_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            record_result "SAFETY_MODULE_$module" "PASS" "Safety module imports successfully"
        else
            record_result "SAFETY_MODULE_$module" "FAIL" "Safety module import failed: $module"
        fi
    done
    
    # Run safety tests
    if uv run python -m pytest tests/unit/safety/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "SAFETY_TESTS" "PASS" "Safety system tests passed"
    else
        record_result "SAFETY_TESTS" "FAIL" "Safety system tests failed"
    fi
    
    # Test emergency procedures
    if uv run python -m pytest tests/unit/modes/ -k "emergency" -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "EMERGENCY_TESTS" "PASS" "Emergency procedure tests passed"
    else
        record_result "EMERGENCY_TESTS" "FAIL" "Emergency procedure tests failed"
    fi
}

# Phase 6: Model Preservation Validation
validate_phase_6() {
    info "=== Phase 6: Model Preservation Validation ==="
    
    # Test model preservation modules
    local preservation_modules=(
        "src.model_preservation.manager"
        "src.model_preservation.db_handler"
        "src.model_preservation.gcs_handler"
        "src.model_preservation.versioning"
        "src.model_preservation.monitoring"
        "src.model_preservation.ab_testing"
    )
    
    for module in "${preservation_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            record_result "PRESERVATION_MODULE_$module" "PASS" "Preservation module imports successfully"
        else
            record_result "PRESERVATION_MODULE_$module" "FAIL" "Preservation module import failed: $module"
        fi
    done
    
    # Run preservation tests
    if uv run python -m pytest tests/unit/model_preservation/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "PRESERVATION_TESTS" "PASS" "Model preservation tests passed"
    else
        record_result "PRESERVATION_TESTS" "FAIL" "Model preservation tests failed"
    fi
    
    # Check GCS bucket access
    if gsutil ls gs://shyvr-rlte-models-$PROJECT_ID &>/dev/null; then
        record_result "GCS_MODEL_BUCKET" "PASS" "GCS model bucket accessible"
    else
        record_result "GCS_MODEL_BUCKET" "FAIL" "GCS model bucket not accessible"
    fi
}

# Phase 7: Testing & Validation Check
validate_phase_7() {
    info "=== Phase 7: Testing & Validation Check ==="
    
    # Run integration tests
    if uv run python -m pytest tests/integration/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "INTEGRATION_TESTS" "PASS" "Integration tests passed"
    else
        record_result "INTEGRATION_TESTS" "WARN" "Some integration tests failed - check logs"
    fi
    
    # Run performance tests
    if uv run python -m pytest tests/performance/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "PERFORMANCE_TESTS" "PASS" "Performance tests passed"
    else
        record_result "PERFORMANCE_TESTS" "WARN" "Some performance tests failed - check logs"
    fi
    
    # Run security tests
    if uv run python -m pytest tests/security/ -v --tb=short --disable-warnings 2>/dev/null; then
        record_result "SECURITY_TESTS" "PASS" "Security tests passed"
    else
        record_result "SECURITY_TESTS" "FAIL" "Security tests failed"
    fi
}

# Infrastructure Validation
validate_infrastructure() {
    info "=== Infrastructure Validation ==="
    
    # Check Cloud SQL instance
    if gcloud sql instances describe shyvr-rlte-db --region="$REGION" &>/dev/null; then
        record_result "CLOUD_SQL_INSTANCE" "PASS" "Cloud SQL instance exists"
    else
        record_result "CLOUD_SQL_INSTANCE" "FAIL" "Cloud SQL instance not found"
    fi
    
    # Check Cloud Run service
    if gcloud run services describe "$SERVICE_NAME" --region="$REGION" &>/dev/null; then
        record_result "CLOUD_RUN_SERVICE" "PASS" "Cloud Run service exists"
    else
        record_result "CLOUD_RUN_SERVICE" "WARN" "Cloud Run service not found (expected for first deployment)"
    fi
    
    # Check secrets
    local required_secrets=(
        "HYPERLIQUID_API_KEY"
        "ETHEREUM_PRIVATE_KEY"
        "SOLANA_PRIVATE_KEY"
        "DATABASE_URL"
    )
    
    for secret in "${required_secrets[@]}"; do
        if gcloud secrets describe "$secret" &>/dev/null; then
            record_result "SECRET_$secret" "PASS" "Secret $secret exists"
        else
            record_result "SECRET_$secret" "FAIL" "Secret $secret not found"
        fi
    done
}

# Database Validation
validate_database() {
    info "=== Database Validation ==="
    
    # Check database connectivity
    if python scripts/test_cloud_sql_connectivity.py 2>/dev/null; then
        record_result "DATABASE_CONNECTIVITY" "PASS" "Database connection successful"
    else
        record_result "DATABASE_CONNECTIVITY" "FAIL" "Database connection failed"
    fi
    
    # Check database schema
    if uv run python scripts/validate_database_schema.py 2>/dev/null; then
        record_result "DATABASE_SCHEMA" "PASS" "Database schema validation passed"
    else
        record_result "DATABASE_SCHEMA" "FAIL" "Database schema validation failed"
    fi
    
    # Check migrations
    local migration_files=(
        "database/migrations/001_create_initial_schema.sql"
        "database/migrations/002_create_indexes_and_functions.sql"
        "database/migrations/003_create_views_and_permissions.sql"
        "database/migrations/004_create_rl_experience_schema.sql"
        "database/migrations/005_model_preservation_schema.sql"
    )
    
    for migration in "${migration_files[@]}"; do
        if [[ -f "$migration" ]]; then
            record_result "MIGRATION_$migration" "PASS" "Migration file exists"
        else
            record_result "MIGRATION_$migration" "FAIL" "Migration file missing: $migration"
        fi
    done
}

# Container Validation
validate_container() {
    info "=== Container Validation ==="
    
    # Build container image
    if docker build -t shyvr-rlte:validation . &>/dev/null; then
        record_result "CONTAINER_BUILD" "PASS" "Container builds successfully"
    else
        record_result "CONTAINER_BUILD" "FAIL" "Container build failed"
    fi
    
    # Test container startup
    container_id=$(docker run -d --rm shyvr-rlte:validation sleep 30 2>/dev/null)
    if [[ -n "$container_id" ]]; then
        record_result "CONTAINER_STARTUP" "PASS" "Container starts successfully"
        docker stop "$container_id" &>/dev/null || true
    else
        record_result "CONTAINER_STARTUP" "FAIL" "Container startup failed"
    fi
    
    # Check container security
    if docker run --rm shyvr-rlte:validation whoami | grep -q "appuser"; then
        record_result "CONTAINER_SECURITY" "PASS" "Container runs as non-root user"
    else
        record_result "CONTAINER_SECURITY" "FAIL" "Container security issue - running as root"
    fi
}

# Generate validation report
generate_report() {
    info "=== Final Validation Report ==="
    
    local report_file="/tmp/production_validation_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "validation_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_id": "$PROJECT_ID",
  "region": "$REGION",
  "service_name": "$SERVICE_NAME",
  "critical_failures": $CRITICAL_FAILURES,
  "warnings": $WARNING_COUNT,
  "total_tests": ${#VALIDATION_RESULTS[@]},
  "results": [
EOF

    local first=true
    for result in "${VALIDATION_RESULTS[@]}"; do
        IFS='|' read -r test_name status details <<< "$result"
        [[ "$first" == "true" ]] && first=false || echo "    ," >> "$report_file"
        cat >> "$report_file" << EOF
    {
      "test": "$test_name",
      "status": "$status",
      "details": "$details"
    }
EOF
    done
    
    cat >> "$report_file" << EOF
  ],
  "production_ready": $([ $CRITICAL_FAILURES -eq 0 ] && echo "true" || echo "false"),
  "recommendation": "$([ $CRITICAL_FAILURES -eq 0 ] && echo "APPROVED FOR PRODUCTION DEPLOYMENT" || echo "CRITICAL ISSUES MUST BE RESOLVED BEFORE DEPLOYMENT")"
}
EOF

    info "Validation report saved to: $report_file"
    
    # Display summary
    echo
    echo "=================================="
    echo "    FINAL VALIDATION SUMMARY"
    echo "=================================="
    echo "Total Tests: ${#VALIDATION_RESULTS[@]}"
    echo "Critical Failures: $CRITICAL_FAILURES"
    echo "Warnings: $WARNING_COUNT"
    echo "Success Rate: $(( (${#VALIDATION_RESULTS[@]} - CRITICAL_FAILURES - WARNING_COUNT) * 100 / ${#VALIDATION_RESULTS[@]} ))%"
    echo
    
    if [[ $CRITICAL_FAILURES -eq 0 ]]; then
        echo -e "${GREEN}✅ PRODUCTION DEPLOYMENT APPROVED${NC}"
        echo "System is ready for production deployment"
    else
        echo -e "${RED}❌ PRODUCTION DEPLOYMENT BLOCKED${NC}"
        echo "Critical issues must be resolved before deployment"
        exit 1
    fi
    
    if [[ $WARNING_COUNT -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  $WARNING_COUNT warnings detected - review recommended${NC}"
    fi
    
    echo
    echo "Detailed logs: $VALIDATION_LOG"
    echo "Report file: $report_file"
}

# Main execution
main() {
    info "Starting Final Production Validation for Shyvr RLTE"
    info "Validation log: $VALIDATION_LOG"
    
    validate_phase_1
    validate_phase_2
    validate_phase_3
    validate_phase_4
    validate_phase_5
    validate_phase_6
    validate_phase_7
    validate_infrastructure
    validate_database
    validate_container
    
    generate_report
}

# Execute main function
main "$@"