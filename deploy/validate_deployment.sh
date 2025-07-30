#!/bin/bash

# Shyvr RLTE Deployment Validation Script
# Validates deployment health, configuration, and RL experience storage
# Can be run independently or as part of CI/CD pipeline

set -euo pipefail

# Configuration
SERVICE_URL=${1:-""}
PROJECT_ID=${2:-"shvyr-ai-bots"}
SERVICE=${3:-"shyvr-rlte"}
REGION=${4:-"us-central1"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get service URL if not provided
if [[ -z "$SERVICE_URL" ]]; then
    log_info "Getting service URL from Cloud Run..."
    SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
    
    if [[ -z "$SERVICE_URL" ]]; then
        log_error "Could not determine service URL"
        exit 1
    fi
fi

log_info "🔍 Validating Shyvr RLTE deployment: $SERVICE_URL"

# Validation results
TESTS_PASSED=0
TESTS_FAILED=0
WARNINGS=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_critical=${3:-true}
    
    log_info "Testing: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        log_success "✓ $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        if [[ "$is_critical" == "true" ]]; then
            log_error "✗ $test_name"
            ((TESTS_FAILED++))
        else
            log_warning "⚠ $test_name (optional)"
            ((WARNINGS++))
        fi
        return 1
    fi
}

# Basic connectivity tests
log_info "📡 Basic Connectivity Tests"
run_test "Service responds to HTTP requests" "curl -f -s -m 10 '$SERVICE_URL' >/dev/null"
run_test "Health endpoint accessible" "curl -f -s -m 10 '$SERVICE_URL/health' >/dev/null"
run_test "API endpoint accessible" "curl -f -s -m 10 '$SERVICE_URL/api' >/dev/null"

# Health status validation
log_info "🏥 Health Status Validation"

# Get health response
HEALTH_RESPONSE=$(curl -s -m 30 "$SERVICE_URL/health" 2>/dev/null || echo "{}")

# Parse health status
if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    log_success "✓ Overall health status: HEALTHY"
    ((TESTS_PASSED++))
elif echo "$HEALTH_RESPONSE" | grep -q '"status":"degraded"'; then
    log_warning "⚠ Overall health status: DEGRADED"
    ((WARNINGS++))
else
    log_error "✗ Overall health status: UNKNOWN/UNHEALTHY"
    ((TESTS_FAILED++))
fi

# Component health validation
log_info "🧩 Component Health Validation"

# Database health
if echo "$HEALTH_RESPONSE" | grep -q '"database".*"status":"healthy"'; then
    log_success "✓ Database connectivity: HEALTHY"
    ((TESTS_PASSED++))
else
    log_error "✗ Database connectivity: UNHEALTHY"
    ((TESTS_FAILED++))
fi

# ML Models health
if echo "$HEALTH_RESPONSE" | grep -q '"ml_models".*"status":"healthy"'; then
    log_success "✓ ML models: HEALTHY"
    ((TESTS_PASSED++))
elif echo "$HEALTH_RESPONSE" | grep -q '"ml_models"'; then
    log_warning "⚠ ML models: DETECTED but not fully healthy"
    ((WARNINGS++))
else
    log_warning "⚠ ML models: STATUS UNCLEAR"
    ((WARNINGS++))
fi

# RL Agent health
if echo "$HEALTH_RESPONSE" | grep -q '"rl_agent".*"status":"healthy"'; then
    log_success "✓ RL agent: HEALTHY"
    ((TESTS_PASSED++))
elif echo "$HEALTH_RESPONSE" | grep -q '"rl_agent"'; then
    log_warning "⚠ RL agent: DETECTED but not fully healthy"
    ((WARNINGS++))
else
    log_warning "⚠ RL agent: STATUS UNCLEAR"
    ((WARNINGS++))
fi

# Configuration tests
log_info "⚙️ Configuration Validation"
run_test "Configuration endpoint responds" "curl -f -s -m 10 '$SERVICE_URL/config' >/dev/null" false

# Get configuration
CONFIG_RESPONSE=$(curl -s -m 10 "$SERVICE_URL/config" 2>/dev/null || echo "{}")

# Validate trading modes
if echo "$CONFIG_RESPONSE" | grep -q '"modes"'; then
    log_success "✓ Trading modes configured"
    ((TESTS_PASSED++))
    
    # Check specific modes
    if echo "$CONFIG_RESPONSE" | grep -q '"analysis":true'; then
        log_success "  ✓ Analysis mode: ENABLED"
    fi
    
    if echo "$CONFIG_RESPONSE" | grep -q '"simulation":true'; then
        log_success "  ✓ Simulation mode: ENABLED" 
    fi
    
    if echo "$CONFIG_RESPONSE" | grep -q '"live":true'; then
        log_warning "  ⚠ Live trading mode: ENABLED (ensure this is intentional)"
    else
        log_success "  ✓ Live trading mode: DISABLED (safe)"
    fi
else
    log_warning "⚠ Trading modes configuration unclear"
    ((WARNINGS++))
fi

# Metrics endpoint test
log_info "📊 Metrics Validation"
run_test "Metrics endpoint responds" "curl -f -s -m 10 '$SERVICE_URL/metrics' >/dev/null" false

# RL Experience Storage validation
log_info "🧠 RL Experience Storage Validation"

# Check if RL tables are mentioned in health response
if echo "$HEALTH_RESPONSE" | grep -q -i "rl.*experience\|experience.*rl"; then
    log_success "✓ RL experience storage detected in health response"
    ((TESTS_PASSED++))
else
    log_warning "⚠ RL experience storage not clearly indicated in health"
    ((WARNINGS++))
fi

# Database-specific RL validation
if echo "$HEALTH_RESPONSE" | grep -q -i "table.*exist.*3\|tables.*exist.*true"; then
    log_success "✓ Database tables exist (including RL tables)"
    ((TESTS_PASSED++))
else
    log_warning "⚠ Database table status unclear"
    ((WARNINGS++))
fi

# Security validation
log_info "🔒 Security Validation"

# Check for HTTPS
if [[ "$SERVICE_URL" == https://* ]]; then
    log_success "✓ Service uses HTTPS"
    ((TESTS_PASSED++))
else
    log_warning "⚠ Service not using HTTPS"
    ((WARNINGS++))
fi

# Check response headers for security
SECURITY_HEADERS=$(curl -I -s -m 10 "$SERVICE_URL" 2>/dev/null || echo "")

if echo "$SECURITY_HEADERS" | grep -q -i "x-frame-options\|content-security-policy"; then
    log_success "✓ Security headers present"
    ((TESTS_PASSED++))
else
    log_warning "⚠ Security headers not detected"
    ((WARNINGS++))
fi

# Performance validation
log_info "⚡ Performance Validation"

# Response time test
START_TIME=$(date +%s%N)
curl -f -s -m 10 "$SERVICE_URL/health" >/dev/null 2>&1
END_TIME=$(date +%s%N)
RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 )) # Convert to milliseconds

if [[ $RESPONSE_TIME -lt 5000 ]]; then
    log_success "✓ Health endpoint response time: ${RESPONSE_TIME}ms (good)"
    ((TESTS_PASSED++))
elif [[ $RESPONSE_TIME -lt 10000 ]]; then
    log_warning "⚠ Health endpoint response time: ${RESPONSE_TIME}ms (acceptable)"
    ((WARNINGS++))
else
    log_error "✗ Health endpoint response time: ${RESPONSE_TIME}ms (slow)"
    ((TESTS_FAILED++))
fi

# Cloud Run service validation (if gcloud available)
if command -v gcloud >/dev/null 2>&1; then
    log_info "☁️ Cloud Run Service Validation"
    
    # Check service status
    if gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.conditions[0].status)" 2>/dev/null | grep -q "True"; then
        log_success "✓ Cloud Run service status: READY"
        ((TESTS_PASSED++))
    else
        log_error "✗ Cloud Run service status: NOT READY"
        ((TESTS_FAILED++))
    fi
    
    # Check resource allocation
    MEMORY=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(spec.template.spec.containers[0].resources.limits.memory)" 2>/dev/null || echo "unknown")
    CPU=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(spec.template.spec.containers[0].resources.limits.cpu)" 2>/dev/null || echo "unknown")
    
    log_info "  Memory limit: $MEMORY"
    log_info "  CPU limit: $CPU"
    
    # Validate adequate resources for ML/RL workloads
    if [[ "$MEMORY" == *"Gi" ]] && [[ "${MEMORY%Gi}" -ge 4 ]]; then
        log_success "✓ Memory allocation adequate for ML/RL workloads (${MEMORY})"
        ((TESTS_PASSED++))
    else
        log_warning "⚠ Memory allocation may be insufficient for ML/RL workloads (${MEMORY})"
        ((WARNINGS++))
    fi
fi

# Final summary
echo ""
log_info "📋 VALIDATION SUMMARY"
echo "=================="
log_success "Tests passed: $TESTS_PASSED"
if [[ $WARNINGS -gt 0 ]]; then
    log_warning "Warnings: $WARNINGS"
fi
if [[ $TESTS_FAILED -gt 0 ]]; then
    log_error "Tests failed: $TESTS_FAILED"
fi

# Overall result
TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo ""
if [[ $TESTS_FAILED -eq 0 ]]; then
    log_success "🎉 DEPLOYMENT VALIDATION PASSED"
    log_success "Success rate: ${SUCCESS_RATE}% ($TESTS_PASSED/$TOTAL_TESTS tests passed)"
    
    if [[ $WARNINGS -gt 0 ]]; then
        log_warning "Note: $WARNINGS warnings detected - review recommended"
    fi
    
    echo ""
    log_info "🚀 Deployment appears healthy and ready for production use"
    exit 0
else
    log_error "💥 DEPLOYMENT VALIDATION FAILED"
    log_error "Success rate: ${SUCCESS_RATE}% ($TESTS_PASSED/$TOTAL_TESTS tests passed, $TESTS_FAILED failed)"
    
    echo ""
    log_error "🚨 Critical issues detected - deployment needs attention"
    exit 1
fi