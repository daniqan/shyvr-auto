#!/bin/bash

# Shyvr RLTE Unified Deployment Runner
# Clean, maintainable deployment orchestrator for all environments
# Supports staging, production, and emergency deployment scenarios

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT=${1:-""}
DEPLOYMENT_MODE=${2:-"standard"}  # standard, emergency, validation-only
IMAGE_TAG=${3:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
log_success() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[DEPLOY]${NC} $1"; }
log_error() { echo -e "${RED}[DEPLOY]${NC} $1"; }
log_header() { echo -e "${PURPLE}[DEPLOY]${NC} $1"; }

# Usage information
show_usage() {
    echo "Shyvr RLTE Unified Deployment Runner"
    echo ""
    echo "Usage: $0 <environment> [mode] [image_tag]"
    echo ""
    echo "Environments:"
    echo "  staging     - Deploy to staging environment"
    echo "  production  - Deploy to production environment"
    echo ""
    echo "Modes:"
    echo "  standard         - Full deployment pipeline (default)"
    echo "  emergency        - Emergency deployment with minimal validation"
    echo "  validation-only  - Run validation only, no deployment"
    echo "  blue-green       - Blue-green deployment only"
    echo "  rollback         - Rollback to previous version"
    echo ""
    echo "Examples:"
    echo "  $0 staging"
    echo "  $0 production standard v1.2.3"
    echo "  $0 production emergency"
    echo "  $0 staging validation-only"
    echo "  $0 production rollback"
    echo ""
    echo "Environment Variables:"
    echo "  SKIP_TESTS=true     - Skip testing phases"
    echo "  FORCE_DEPLOY=true   - Skip validation failures"
    echo "  DRY_RUN=true        - Show what would be deployed"
}

# Validate arguments
validate_arguments() {
    if [[ -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required"
        show_usage
        exit 1
    fi
    
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        show_usage
        exit 1
    fi
    
    local valid_modes=("standard" "emergency" "validation-only" "blue-green" "rollback")
    local mode_valid=false
    for mode in "${valid_modes[@]}"; do
        if [[ "$DEPLOYMENT_MODE" == "$mode" ]]; then
            mode_valid=true
            break
        fi
    done
    
    if [[ "$mode_valid" == "false" ]]; then
        log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
        show_usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/main.py" ]]; then
        log_error "Not in project root directory (missing main.py)"
        exit 1
    fi
    
    # Check required deployment scripts exist
    local required_scripts=(
        "automated_deployment_pipeline.sh"
        "blue_green_deployment.sh"
        "validate_deployment.sh"
        "automated_rollback.sh"
        "production_readiness_checklist.sh"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            log_error "Required deployment script missing: $script"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Validation-only mode
run_validation_only() {
    log_header "🔍 Running validation-only mode"
    
    log_info "Running production readiness checklist..."
    if "$SCRIPT_DIR/production_readiness_checklist.sh"; then
        log_success "Production readiness validation passed"
    else
        log_error "Production readiness validation failed"
        exit 1
    fi
    
    # If we have a deployed service, validate it
    local service_name
    if [[ "$ENVIRONMENT" == "production" ]]; then
        service_name="shyvr-rlte"
    else
        service_name="shyvr-rlte-staging"
    fi
    
    if gcloud run services describe "$service_name" --region="us-central1" >/dev/null 2>&1; then
        log_info "Validating existing deployment..."
        if "$SCRIPT_DIR/validate_deployment.sh"; then
            log_success "Existing deployment validation passed"
        else
            log_warning "Existing deployment validation failed"
        fi
    else
        log_info "No existing deployment found to validate"
    fi
    
    log_success "✅ Validation-only mode completed"
}

# Emergency deployment mode
run_emergency_deployment() {
    log_header "🚨 Running emergency deployment mode"
    log_warning "Emergency mode: Minimal validation, maximum speed"
    
    # Set emergency environment variables
    export SKIP_TESTS=true
    export FORCE_DEPLOY=true
    
    log_info "Executing emergency blue-green deployment..."
    if "$SCRIPT_DIR/blue_green_deployment.sh" "$ENVIRONMENT" "$IMAGE_TAG"; then
        log_success "Emergency deployment completed"
        
        # Quick validation
        log_info "Running quick post-deployment validation..."
        if "$SCRIPT_DIR/validate_deployment.sh"; then
            log_success "Emergency deployment validated successfully"
        else
            log_warning "Emergency deployment validation failed - manual review required"
        fi
    else
        log_error "Emergency deployment failed"
        exit 1
    fi
}

# Blue-green deployment mode
run_blue_green_only() {
    log_header "🚦 Running blue-green deployment mode"
    
    log_info "Executing blue-green deployment..."
    if "$SCRIPT_DIR/blue_green_deployment.sh" "$ENVIRONMENT" "$IMAGE_TAG"; then
        log_success "Blue-green deployment completed"
        
        log_info "Running post-deployment validation..."
        if "$SCRIPT_DIR/validate_deployment.sh"; then
            log_success "Blue-green deployment validated successfully"
        else
            log_error "Blue-green deployment validation failed"
            exit 1
        fi
    else
        log_error "Blue-green deployment failed"
        exit 1
    fi
}

# Rollback mode
run_rollback() {
    log_header "🔄 Running rollback mode"
    
    log_warning "This will rollback $ENVIRONMENT to the previous version"
    
    if [[ "${FORCE_DEPLOY:-false}" != "true" ]]; then
        read -p "Are you sure you want to rollback? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    log_info "Executing rollback..."
    if "$SCRIPT_DIR/automated_rollback.sh" "$ENVIRONMENT" "previous"; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# Standard deployment mode
run_standard_deployment() {
    log_header "🚀 Running standard deployment mode"
    
    log_info "Executing comprehensive deployment pipeline..."
    if "$SCRIPT_DIR/automated_deployment_pipeline.sh" "$ENVIRONMENT"; then
        log_success "Standard deployment completed successfully"
    else
        log_error "Standard deployment failed"
        exit 1
    fi
}

# Dry run mode
show_dry_run() {
    log_header "🔍 DRY RUN MODE - No actual deployment will occur"
    log_info "Environment: $ENVIRONMENT"
    log_info "Mode: $DEPLOYMENT_MODE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info ""
    
    case "$DEPLOYMENT_MODE" in
        "validation-only")
            log_info "Would run: production_readiness_checklist.sh"
            log_info "Would run: validate_deployment.sh (if service exists)"
            ;;
        "emergency")
            log_info "Would set: SKIP_TESTS=true FORCE_DEPLOY=true"
            log_info "Would run: blue_green_deployment.sh $ENVIRONMENT $IMAGE_TAG"
            log_info "Would run: validate_deployment.sh"
            ;;
        "blue-green")
            log_info "Would run: blue_green_deployment.sh $ENVIRONMENT $IMAGE_TAG"
            log_info "Would run: validate_deployment.sh"
            ;;
        "rollback")
            log_info "Would run: automated_rollback.sh $ENVIRONMENT previous"
            ;;
        "standard")
            log_info "Would run: automated_deployment_pipeline.sh $ENVIRONMENT"
            ;;
    esac
    
    log_info ""
    log_info "To execute for real, run without DRY_RUN=true"
}

# Main execution
main() {
    log_header "🤖 Shyvr RLTE Unified Deployment Runner"
    log_info "Environment: $ENVIRONMENT"
    log_info "Mode: $DEPLOYMENT_MODE"
    log_info "Image Tag: $IMAGE_TAG"
    echo ""
    
    # Validate arguments
    validate_arguments
    
    # Check prerequisites
    check_prerequisites
    
    # Handle dry run
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        show_dry_run
        exit 0
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Execute based on mode
    case "$DEPLOYMENT_MODE" in
        "validation-only")
            run_validation_only
            ;;
        "emergency")
            run_emergency_deployment
            ;;
        "blue-green")
            run_blue_green_only
            ;;
        "rollback")
            run_rollback
            ;;
        "standard")
            run_standard_deployment
            ;;
        *)
            log_error "Unknown deployment mode: $DEPLOYMENT_MODE"
            exit 1
            ;;
    esac
    
    log_success "🎉 Deployment operation completed successfully!"
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

# Execute main function
main "$@"