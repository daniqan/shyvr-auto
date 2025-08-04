#!/bin/bash

# Automated Deployment Pipeline for Shyvr RLTE
# Comprehensive automation for production-ready deployments
# Includes pre-deployment validation, blue-green deployment, and post-deployment verification

set -euo pipefail

# Configuration
PROJECT_ID=${PROJECT_ID:-"shvyr-ai-bots"}
REGION=${REGION:-"us-central1"}
REPOSITORY=${REPOSITORY:-"shyvr-ai-prod"}
ENVIRONMENT=${1:-"staging"}
SKIP_TESTS=${SKIP_TESTS:-false}
FORCE_DEPLOY=${FORCE_DEPLOY:-false}
SKIP_INFRASTRUCTURE=${SKIP_INFRASTRUCTURE:-false}

# Deployment configuration
DEPLOYMENT_ID=$(date +"%Y%m%d-%H%M%S")
IMAGE_TAG=${IMAGE_TAG:-$DEPLOYMENT_ID}
DEPLOYMENT_LOG="deployment_pipeline_${DEPLOYMENT_ID}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$DEPLOYMENT_LOG"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOYMENT_LOG"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOYMENT_LOG"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"; }
log_header() { echo -e "${PURPLE}[PIPELINE]${NC} $1" | tee -a "$DEPLOYMENT_LOG"; }

# Pipeline stages status
PIPELINE_STAGES=(
    "infrastructure_setup"
    "pre_deployment_validation"
    "security_scan"
    "build_and_test"
    "image_build"
    "blue_green_deployment"
    "health_validation"
    "monitoring_setup"
    "post_deployment_tests"
)

STAGE_STATUS=()
for stage in "${PIPELINE_STAGES[@]}"; do
    STAGE_STATUS["$stage"]="pending"
done

# Error handling and cleanup
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment pipeline failed with exit code $exit_code"
        log_error "Check the deployment log: $DEPLOYMENT_LOG"
        
        # Generate failure report
        generate_failure_report
        
        # Trigger rollback if deployment was in progress
        if [[ "${STAGE_STATUS[blue_green_deployment]}" == "in_progress" ]]; then
            log_warning "Deployment was in progress - triggering rollback"
            ./deploy/automated_rollback.sh "$ENVIRONMENT" emergency
        fi
    fi
}

trap cleanup EXIT

# Progress tracking
update_stage_status() {
    local stage="$1"
    local status="$2"
    STAGE_STATUS["$stage"]="$status"
    log_info "Stage $stage: $status"
}

show_pipeline_progress() {
    log_header "📊 Pipeline Progress"
    for stage in "${PIPELINE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        case "$status" in
            "completed") echo -e "  ✅ $stage" ;;
            "in_progress") echo -e "  🔄 $stage" ;;
            "failed") echo -e "  ❌ $stage" ;;
            *) echo -e "  ⏳ $stage" ;;
        esac
    done
    echo ""
}

# Stage 0: Infrastructure setup
stage_infrastructure_setup() {
    if [[ "$SKIP_INFRASTRUCTURE" == "true" ]]; then
        log_warning "Skipping infrastructure setup (SKIP_INFRASTRUCTURE=true)"
        update_stage_status "infrastructure_setup" "completed"
        return 0
    fi
    
    update_stage_status "infrastructure_setup" "in_progress"
    log_header "🏗️ Stage 0: Infrastructure Setup"
    
    # Determine infrastructure setup mode based on environment
    local infra_mode
    case "$ENVIRONMENT" in
        "production")
            # In production, only validate - don't create missing infrastructure
            infra_mode="validate"
            log_info "Production environment: validating existing infrastructure"
            ;;
        "staging")
            # In staging, create missing infrastructure
            infra_mode="setup-if-needed"
            log_info "Staging environment: creating missing infrastructure"
            ;;
        *)
            # For other environments, create missing infrastructure
            infra_mode="setup-if-needed"
            log_info "Development environment: creating missing infrastructure"
            ;;
    esac
    
    # Run infrastructure setup
    local script_dir="$(dirname "$0")"
    if [[ ! -f "$script_dir/setup-infrastructure.sh" ]]; then
        log_error "Infrastructure setup script not found: $script_dir/setup-infrastructure.sh"
        update_stage_status "infrastructure_setup" "failed"
        return 1
    fi
    
    log_info "Running infrastructure setup in $infra_mode mode"
    
    if "$script_dir/setup-infrastructure.sh" \
        --project-id "$PROJECT_ID" \
        --region "$REGION" \
        --environment "$ENVIRONMENT" \
        --mode "$infra_mode"; then
        
        log_success "Infrastructure setup completed successfully"
        update_stage_status "infrastructure_setup" "completed"
        return 0
    else
        if [[ "$infra_mode" == "validate" ]]; then
            log_error "Infrastructure validation failed - missing or misconfigured components"
            log_error "Please fix infrastructure issues before deploying to production"
        else
            log_error "Infrastructure setup failed - cannot proceed with deployment"
        fi
        update_stage_status "infrastructure_setup" "failed"
        return 1
    fi
}

# Stage 1: Pre-deployment validation
stage_pre_deployment_validation() {
    update_stage_status "pre_deployment_validation" "in_progress"
    log_header "🔍 Stage 1: Pre-deployment Validation"
    
    # Run production readiness checklist
    if [[ "$FORCE_DEPLOY" != "true" ]]; then
        if ! ./deploy/production_readiness_checklist.sh; then
            log_error "Production readiness validation failed"
            update_stage_status "pre_deployment_validation" "failed"
            return 1
        fi
    else
        log_warning "Skipping production readiness validation (FORCE_DEPLOY=true)"
    fi
    
    # Validate environment variables
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        update_stage_status "pre_deployment_validation" "failed"
        return 1
    fi
    
    # Check Docker and gcloud
    for tool in docker gcloud; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool not found: $tool"
            update_stage_status "pre_deployment_validation" "failed"
            return 1
        fi
    done
    
    # Validate GCP authentication and project access
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access GCP project: $PROJECT_ID"
        update_stage_status "pre_deployment_validation" "failed"
        return 1
    fi
    
    update_stage_status "pre_deployment_validation" "completed"
    log_success "Pre-deployment validation completed"
}

# Stage 2: Security scan
stage_security_scan() {
    update_stage_status "security_scan" "in_progress"
    log_header "🔒 Stage 2: Security Scan"
    
    # Check for hardcoded secrets
    log_info "Scanning for hardcoded secrets..."
    if grep -r "password\s*=" src/ config/ || grep -r "key\s*=" src/ config/ || grep -r "token\s*=" src/ config/; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            log_error "Hardcoded secrets detected in source code"
            update_stage_status "security_scan" "failed"
            return 1
        else
            log_warning "Hardcoded secrets detected but continuing (FORCE_DEPLOY=true)"
        fi
    fi
    
    # Validate secret availability
    local required_secrets=("TELEGRAM_TOKEN" "DB_PASSWORD" "SECRET_KEY")
    for secret in "${required_secrets[@]}"; do
        if ! gcloud secrets versions access latest --secret="$secret" --project="$PROJECT_ID" >/dev/null 2>&1; then
            log_error "Required secret not accessible: $secret"
            update_stage_status "security_scan" "failed"
            return 1
        fi
    done
    
    update_stage_status "security_scan" "completed"
    log_success "Security scan completed"
}

# Stage 3: Build and test
stage_build_and_test() {
    update_stage_status "build_and_test" "in_progress"
    log_header "🧪 Stage 3: Build and Test"
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (SKIP_TESTS=true)"
        update_stage_status "build_and_test" "completed"
        return 0
    fi
    
    # Install dependencies
    log_info "Installing dependencies..."
    if ! uv sync --frozen >/dev/null 2>&1; then
        log_error "Failed to install dependencies"
        update_stage_status "build_and_test" "failed"
        return 1
    fi
    
    # Run critical unit tests
    log_info "Running critical unit tests..."
    if ! uv run pytest tests/unit/safety/ tests/unit/model_preservation/ tests/unit/rl_agent/ -v --tb=short >/dev/null 2>&1; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            log_error "Critical unit tests failed"
            update_stage_status "build_and_test" "failed"
            return 1
        else
            log_warning "Unit tests failed but continuing (FORCE_DEPLOY=true)"
        fi
    fi
    
    # Validate Python imports
    log_info "Validating Python imports..."
    if ! python3 -c "import sys; sys.path.insert(0, 'src'); import utils.config, ml_analysis.model_manager, rl_agent.dqn_agent" 2>/dev/null; then
        log_warning "Some Python imports failed (may be environment-specific)"
    fi
    
    update_stage_status "build_and_test" "completed"
    log_success "Build and test completed"
}

# Stage 4: Image build
stage_image_build() {
    update_stage_status "image_build" "in_progress"
    log_header "🐳 Stage 4: Docker Image Build"
    
    local image_name="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/shyvr-rlte"
    
    # Configure Docker authentication
    if ! gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet; then
        log_error "Failed to configure Docker authentication"
        update_stage_status "image_build" "failed"
        return 1
    fi
    
    # Build Docker image
    log_info "Building Docker image..."
    if ! docker build --platform linux/amd64 \
        --tag "$image_name:$IMAGE_TAG" \
        --tag "$image_name:latest" \
        --label "deployment.id=$DEPLOYMENT_ID" \
        --label "environment=$ENVIRONMENT" \
        . >/dev/null 2>&1; then
        log_error "Docker build failed"
        update_stage_status "image_build" "failed"
        return 1
    fi
    
    # Push images
    log_info "Pushing Docker images..."
    for tag in "$IMAGE_TAG" "latest"; do
        if ! docker push "$image_name:$tag" >/dev/null 2>&1; then
            log_error "Failed to push image: $image_name:$tag"
            update_stage_status "image_build" "failed"
            return 1
        fi
    done
    
    update_stage_status "image_build" "completed"
    log_success "Docker image build and push completed"
}

# Stage 5: Blue-green deployment
stage_blue_green_deployment() {
    update_stage_status "blue_green_deployment" "in_progress"
    log_header "🚦 Stage 5: Blue-Green Deployment"
    
    # Execute blue-green deployment
    if ! ./deploy/blue_green_deployment.sh "$ENVIRONMENT" "$IMAGE_TAG"; then
        log_error "Blue-green deployment failed"
        update_stage_status "blue_green_deployment" "failed"
        return 1
    fi
    
    update_stage_status "blue_green_deployment" "completed"
    log_success "Blue-green deployment completed"
}

# Stage 6: Health validation
stage_health_validation() {
    update_stage_status "health_validation" "in_progress"
    log_header "🏥 Stage 6: Health Validation"
    
    # Wait for service to stabilize
    log_info "Waiting for service to stabilize..."
    sleep 60
    
    # Run deployment validation
    if ! ./deploy/validate_deployment.sh; then
        log_error "Health validation failed"
        update_stage_status "health_validation" "failed"
        return 1
    fi
    
    update_stage_status "health_validation" "completed"
    log_success "Health validation completed"
}

# Stage 7: Monitoring setup
stage_monitoring_setup() {
    update_stage_status "monitoring_setup" "in_progress"
    log_header "📊 Stage 7: Monitoring Setup"
    
    # Setup monitoring for production only
    if [[ "$ENVIRONMENT" == "production" ]]; then
        if ! ./deploy/setup_production_monitoring.sh "$ENVIRONMENT"; then
            log_warning "Monitoring setup failed (non-critical)"
        else
            log_success "Production monitoring configured"
        fi
    else
        log_info "Skipping monitoring setup for staging environment"
    fi
    
    update_stage_status "monitoring_setup" "completed"
    log_success "Monitoring setup completed"
}

# Stage 8: Post-deployment tests
stage_post_deployment_tests() {
    update_stage_status "post_deployment_tests" "in_progress"
    log_header "🧪 Stage 8: Post-Deployment Tests"
    
    # Get service URL
    local service_name
    if [[ "$ENVIRONMENT" == "production" ]]; then
        service_name="shyvr-rlte"
    else
        service_name="shyvr-rlte-staging"
    fi
    
    local service_url
    service_url=$(gcloud run services describe "$service_name" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
    
    if [[ -z "$service_url" ]]; then
        log_error "Could not get service URL for post-deployment tests"
        update_stage_status "post_deployment_tests" "failed"
        return 1
    fi
    
    # Test critical endpoints
    local endpoints=("/health" "/config")
    for endpoint in "${endpoints[@]}"; do
        log_info "Testing endpoint: $endpoint"
        if ! curl -f -s -m 30 "$service_url$endpoint" >/dev/null; then
            if [[ "$endpoint" == "/health" ]]; then
                log_error "Critical endpoint failed: $endpoint"
                update_stage_status "post_deployment_tests" "failed"
                return 1
            else
                log_warning "Endpoint failed (non-critical): $endpoint"
            fi
        fi
    done
    
    # Test API responsiveness
    log_info "Testing API responsiveness..."
    local response_time
    response_time=$(curl -o /dev/null -s -w "%{time_total}" "$service_url/health")
    if [[ $(echo "$response_time > 5.0" | bc -l) -eq 1 ]]; then
        log_warning "API response time is slow: ${response_time}s"
    else
        log_success "API response time: ${response_time}s"
    fi
    
    update_stage_status "post_deployment_tests" "completed"
    log_success "Post-deployment tests completed"
}

# Generate deployment report
generate_deployment_report() {
    log_header "📋 Deployment Report"
    
    local service_name
    if [[ "$ENVIRONMENT" == "production" ]]; then
        service_name="shyvr-rlte"
    else
        service_name="shyvr-rlte-staging"
    fi
    
    local service_url
    service_url=$(gcloud run services describe "$service_name" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "N/A")
    
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
    echo "Shyvr RLTE Deployment Report" | tee -a "$DEPLOYMENT_LOG"
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
    echo "Deployment ID: $DEPLOYMENT_ID" | tee -a "$DEPLOYMENT_LOG"
    echo "Environment: $ENVIRONMENT" | tee -a "$DEPLOYMENT_LOG"
    echo "Image Tag: $IMAGE_TAG" | tee -a "$DEPLOYMENT_LOG"
    echo "Service URL: $service_url" | tee -a "$DEPLOYMENT_LOG"
    echo "Deployment Time: $(date)" | tee -a "$DEPLOYMENT_LOG"
    echo "Deployed by: $(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"
    
    echo "Pipeline Stage Results:" | tee -a "$DEPLOYMENT_LOG"
    for stage in "${PIPELINE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        echo "  $stage: $status" | tee -a "$DEPLOYMENT_LOG"
    done
    echo "" | tee -a "$DEPLOYMENT_LOG"
    
    echo "Post-Deployment Actions:" | tee -a "$DEPLOYMENT_LOG"
    echo "  Monitor logs: gcloud run logs tail $service_name --region $REGION" | tee -a "$DEPLOYMENT_LOG"
    echo "  Check health: curl $service_url/health" | tee -a "$DEPLOYMENT_LOG"
    echo "  View metrics: https://console.cloud.google.com/run/detail/$REGION/$service_name/metrics" | tee -a "$DEPLOYMENT_LOG"
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
}

# Generate failure report
generate_failure_report() {
    log_header "💥 Deployment Failure Report"
    
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
    echo "Deployment Failed: $DEPLOYMENT_ID" | tee -a "$DEPLOYMENT_LOG"
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
    echo "Environment: $ENVIRONMENT" | tee -a "$DEPLOYMENT_LOG"
    echo "Failure Time: $(date)" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"
    
    echo "Failed Stages:" | tee -a "$DEPLOYMENT_LOG"
    for stage in "${PIPELINE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        if [[ "$status" == "failed" || "$status" == "in_progress" ]]; then
            echo "  ❌ $stage: $status" | tee -a "$DEPLOYMENT_LOG"
        fi
    done
    echo "" | tee -a "$DEPLOYMENT_LOG"
    
    echo "Recovery Actions:" | tee -a "$DEPLOYMENT_LOG"
    echo "  1. Review deployment log: $DEPLOYMENT_LOG" | tee -a "$DEPLOYMENT_LOG"
    echo "  2. Fix identified issues" | tee -a "$DEPLOYMENT_LOG"
    echo "  3. Re-run deployment pipeline" | tee -a "$DEPLOYMENT_LOG"
    echo "  4. Consider rollback if production is affected" | tee -a "$DEPLOYMENT_LOG"
    echo "================================================" | tee -a "$DEPLOYMENT_LOG"
}

# Main pipeline execution
main() {
    log_header "🚀 Automated Deployment Pipeline for Shyvr RLTE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Log File: $DEPLOYMENT_LOG"
    echo ""
    
    # Initialize all stages as pending
    for stage in "${PIPELINE_STAGES[@]}"; do
        STAGE_STATUS[$stage]="pending"
    done
    
    # Execute pipeline stages
    stage_infrastructure_setup || exit 1
    show_pipeline_progress
    
    stage_pre_deployment_validation || exit 1
    show_pipeline_progress
    
    stage_security_scan || exit 1
    show_pipeline_progress
    
    stage_build_and_test || exit 1
    show_pipeline_progress
    
    stage_image_build || exit 1
    show_pipeline_progress
    
    stage_blue_green_deployment || exit 1
    show_pipeline_progress
    
    stage_health_validation || exit 1
    show_pipeline_progress
    
    stage_monitoring_setup || exit 1
    show_pipeline_progress
    
    stage_post_deployment_tests || exit 1
    show_pipeline_progress
    
    # Generate final report
    generate_deployment_report
    
    log_success "🎉 Deployment pipeline completed successfully!"
    log_info "Full deployment log: $DEPLOYMENT_LOG"
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <environment> [options]"
    echo ""
    echo "Environments:"
    echo "  staging     - Deploy to staging environment"
    echo "  production  - Deploy to production environment"
    echo ""
    echo "Environment Variables:"
    echo "  SKIP_TESTS=true         - Skip testing stages"
    echo "  SKIP_INFRASTRUCTURE=true - Skip infrastructure setup stage"
    echo "  FORCE_DEPLOY=true       - Skip validation failures"
    echo "  IMAGE_TAG=<tag>         - Use specific image tag"
    echo ""
    echo "Examples:"
    echo "  $0 staging"
    echo "  SKIP_TESTS=true $0 production"
    echo "  IMAGE_TAG=v1.2.3 $0 production"
    exit 1
fi

# Execute main function
main