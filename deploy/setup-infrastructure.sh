#!/bin/bash

# Infrastructure Setup Orchestrator for Shyvr RLTE
# Coordinates all infrastructure setup scripts in proper dependency order
# Enhanced with comprehensive validation, dry-run support, and integration with deployment pipeline

set -euo pipefail

# Source deployment utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy-utils.sh"

# Configuration with defaults from deploy-utils.sh
PROJECT_ID="${DEFAULT_PROJECT_ID}"
REGION="${DEFAULT_REGION}"
ENVIRONMENT="staging"
MODE="setup"
DRY_RUN=false
SKIP_EXISTING=true
FORCE_RECREATE=false

# Infrastructure setup stages and their order
INFRASTRUCTURE_STAGES=(
    "enable_apis"
    "create_service_account"
    "setup_secrets"
    "setup_cloud_sql"
    "setup_gcs_infrastructure"
    "setup_monitoring"
)

# Track stage status
declare -A STAGE_STATUS
declare -A STAGE_RESULTS

# Initialize all stages as pending
for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
    STAGE_STATUS["$stage"]="pending"
    STAGE_RESULTS["$stage"]=""
done

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Infrastructure Setup Orchestrator for Shyvr RLTE production deployment.
Coordinates all infrastructure components in proper dependency order.

OPTIONS:
    --project-id PROJECT    GCP project ID (default: $PROJECT_ID)
    --region REGION         GCP region (default: $REGION)
    --environment ENV       Environment (staging/production/development, default: $ENVIRONMENT)
    --mode MODE             Setup mode (default: $MODE)
                           - validate: Only validate existing infrastructure
                           - setup: Create missing infrastructure
                           - setup-if-needed: Create only if missing (recommended)
                           - force-recreate: Delete and recreate all infrastructure
    --skip-existing        Skip components that already exist (default: true)
    --force-recreate       Force recreation of all components
    --dry-run              Show what would be done without executing
    --help, -h             Show this help message

MODES:
    validate        - Validate all infrastructure components exist and are configured correctly
    setup           - Create all infrastructure components (will update existing)
    setup-if-needed - Create only missing infrastructure components (safest for production)
    force-recreate  - Delete and recreate all infrastructure (dangerous - use with caution)

ENVIRONMENTS:
    staging         - Create staging infrastructure with reduced resources
    production      - Create production infrastructure with full resources and protection
    development     - Create minimal development infrastructure

EXAMPLES:
    # Validate existing infrastructure
    $0 --mode validate --environment production
    
    # Setup missing staging infrastructure
    $0 --mode setup-if-needed --environment staging
    
    # Full production setup (dry run first)
    $0 --mode setup --environment production --dry-run
    $0 --mode setup --environment production
    
    # Force recreate development environment
    $0 --mode force-recreate --environment development

INTEGRATION WITH DEPLOYMENT PIPELINE:
    The deployment pipeline can call this script in different modes:
    - Staging: setup-if-needed mode (create missing components)
    - Production: validate mode (ensure all components exist)

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                shift 2
                ;;
            --skip-existing)
                SKIP_EXISTING=true
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE=true
                MODE="force-recreate"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                util_log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Validate mode and environment
validate_configuration() {
    # Validate mode
    case "$MODE" in
        validate|setup|setup-if-needed|force-recreate)
            ;;
        *)
            util_log_error "Invalid mode: $MODE"
            util_log_error "Valid modes: validate, setup, setup-if-needed, force-recreate"
            exit 1
            ;;
    esac
    
    # Validate environment
    if ! validate_environment "$ENVIRONMENT"; then
        exit 1
    fi
    
    # Safety check for force-recreate in production
    if [[ "$MODE" == "force-recreate" && "$ENVIRONMENT" == "production" ]]; then
        util_log_error "DANGER: force-recreate mode in production environment!"
        util_log_error "This will delete and recreate ALL infrastructure components."
        
        if [[ "$DRY_RUN" != "true" ]]; then
            echo ""
            read -p "Are you absolutely sure? Type 'DELETE PRODUCTION INFRASTRUCTURE' to confirm: " -r confirm
            if [[ "$confirm" != "DELETE PRODUCTION INFRASTRUCTURE" ]]; then
                util_log_error "Operation cancelled for safety"
                exit 1
            fi
            
            util_log_warning "Proceeding with production infrastructure recreation..."
            sleep 5
        else
            util_log_info "[DRY RUN] Would require confirmation for production recreation"
        fi
    fi
}

# Initialize infrastructure setup
initialize_infrastructure_setup() {
    util_log_header "🏗️ Infrastructure Setup Orchestrator for Shyvr RLTE"
    util_log_info "Environment: $ENVIRONMENT"
    util_log_info "Mode: $MODE"
    util_log_info "Project: $PROJECT_ID"
    util_log_info "Region: $REGION"
    util_log_info "Dry Run: $([ "$DRY_RUN" = "true" ] && echo "YES" || echo "NO")"
    echo ""
    
    # Validate prerequisites
    if ! check_required_tools; then
        util_log_error "Prerequisites check failed"
        exit 1
    fi
    
    if ! check_gcp_auth "$PROJECT_ID"; then
        util_log_error "GCP authentication check failed"
        exit 1
    fi
    
    # Set project context
    gcloud config set project "$PROJECT_ID" --quiet
    util_log_success "Using project: $PROJECT_ID"
    
    # Validate configuration
    validate_configuration
    
    util_log_success "Infrastructure setup initialized"
}

# Update stage status
update_stage_status() {
    local stage="$1"
    local status="$2"
    local result="${3:-}"
    
    STAGE_STATUS["$stage"]="$status"
    STAGE_RESULTS["$stage"]="$result"
    
    case "$status" in
        "in_progress") util_log_info "📋 Starting: $stage" ;;
        "completed") util_log_success "✅ Completed: $stage" ;;
        "failed") util_log_error "❌ Failed: $stage" ;;
        "skipped") util_log_warning "⏭️ Skipped: $stage" ;;
        *) util_log_info "📋 $stage: $status" ;;
    esac
}

# Show infrastructure progress
show_infrastructure_progress() {
    util_log_header "📊 Infrastructure Setup Progress"
    for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        local result="${STAGE_RESULTS[$stage]}"
        
        case "$status" in
            "completed") echo -e "  ✅ $stage $([ -n "$result" ] && echo "($result)")" ;;
            "in_progress") echo -e "  🔄 $stage" ;;
            "failed") echo -e "  ❌ $stage $([ -n "$result" ] && echo "($result)")" ;;
            "skipped") echo -e "  ⏭️ $stage $([ -n "$result" ] && echo "($result)")" ;;
            *) echo -e "  ⏳ $stage" ;;
        esac
    done
    echo ""
}

# Stage 1: Enable required APIs
stage_enable_apis() {
    update_stage_status "enable_apis" "in_progress"
    
    util_log_info "Enabling required Google Cloud APIs"
    
    local apis=(
        "iam.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "run.googleapis.com"
        "secretmanager.googleapis.com"
        "storage-api.googleapis.com"
        "sqladmin.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudbuild.googleapis.com"
        "artifactregistry.googleapis.com"
    )
    
    local enabled_count=0
    local failed_count=0
    
    for api in "${apis[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would enable $api"
            ((enabled_count++))
        else
            util_log_info "Enabling $api"
            if gcloud services enable "$api" --project="$PROJECT_ID" --quiet; then
                ((enabled_count++))
            else
                util_log_warning "Failed to enable $api (may already be enabled)"
                ((failed_count++))
            fi
        fi
    done
    
    if [[ $failed_count -eq 0 ]]; then
        update_stage_status "enable_apis" "completed" "${enabled_count} APIs enabled"
        return 0
    else
        update_stage_status "enable_apis" "failed" "${failed_count} APIs failed"
        return 1
    fi
}

# Stage 2: Create service account
stage_create_service_account() {
    update_stage_status "create_service_account" "in_progress"
    
    local script_args=(
        "--project-id" "$PROJECT_ID"
        "--region" "$REGION"
    )
    
    case "$MODE" in
        validate)
            script_args+=("--validate-only")
            ;;
        setup|setup-if-needed)
            # Default behavior - create if needed
            ;;
        force-recreate)
            # Would need to add delete functionality to service account script
            util_log_warning "force-recreate mode: service account recreation not implemented"
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        script_args+=("--dry-run")
    fi
    
    util_log_info "Setting up service account"
    if "$SCRIPT_DIR/create_service_account.sh" "${script_args[@]}"; then
        update_stage_status "create_service_account" "completed" "Service account configured"
        return 0
    else
        update_stage_status "create_service_account" "failed" "Service account setup failed"
        return 1
    fi
}

# Stage 3: Setup secrets
stage_setup_secrets() {
    update_stage_status "setup_secrets" "in_progress"
    
    local script_args=(
        "--project-id" "$PROJECT_ID"
        "--region" "$REGION"
    )
    
    case "$MODE" in
        validate)
            script_args+=("--validate-only")
            ;;
        setup|setup-if-needed)
            # Check for environment file
            local env_file=""
            if [[ -f ".env.${ENVIRONMENT}" ]]; then
                env_file=".env.${ENVIRONMENT}"
            elif [[ -f "config/env.${ENVIRONMENT}.template" ]]; then
                env_file="config/env.${ENVIRONMENT}.template"
            elif [[ -f "env.template" ]]; then
                env_file="env.template"
            fi
            
            if [[ -n "$env_file" ]]; then
                script_args+=("--env-file" "$env_file")
                util_log_info "Using environment file: $env_file"
            fi
            ;;
        force-recreate)
            util_log_warning "force-recreate mode: secrets recreation requires manual intervention"
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        script_args+=("--dry-run")
    fi
    
    util_log_info "Setting up Secret Manager"
    if "$SCRIPT_DIR/setup_secrets.sh" "${script_args[@]}"; then
        update_stage_status "setup_secrets" "completed" "Secrets configured"
        return 0
    else
        if [[ "$MODE" == "validate" ]]; then
            update_stage_status "setup_secrets" "failed" "Some secrets missing or inaccessible"
            return 1
        else
            update_stage_status "setup_secrets" "failed" "Secret setup failed"
            return 1
        fi
    fi
}

# Stage 4: Setup Cloud SQL
stage_setup_cloud_sql() {
    update_stage_status "setup_cloud_sql" "in_progress"
    
    # Skip Cloud SQL setup in validate mode if secrets aren't available
    if [[ "$MODE" == "validate" ]]; then
        if ! validate_secret "DB_PASSWORD" "$PROJECT_ID"; then
            update_stage_status "setup_cloud_sql" "skipped" "DB_PASSWORD secret not available"
            return 0
        fi
    fi
    
    local script_args=(
        "--project-id" "$PROJECT_ID"
        "--region" "$REGION"
        "--environment" "$ENVIRONMENT"
    )
    
    case "$MODE" in
        validate)
            script_args+=("--validate-only")
            ;;
        setup|setup-if-needed)
            # Default behavior
            ;;
        force-recreate)
            util_log_warning "force-recreate mode: Cloud SQL recreation not implemented (too dangerous)"
            update_stage_status "setup_cloud_sql" "skipped" "force-recreate not supported for Cloud SQL"
            return 0
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        script_args+=("--dry-run")
    fi
    
    util_log_info "Setting up Cloud SQL"
    if "$SCRIPT_DIR/setup_cloud_sql.sh" "${script_args[@]}"; then
        update_stage_status "setup_cloud_sql" "completed" "Cloud SQL configured"
        return 0
    else
        update_stage_status "setup_cloud_sql" "failed" "Cloud SQL setup failed"
        return 1
    fi
}

# Stage 5: Setup GCS infrastructure
stage_setup_gcs_infrastructure() {
    update_stage_status "setup_gcs_infrastructure" "in_progress"
    
    local script_args=(
        "--project-id" "$PROJECT_ID"
        "--region" "$REGION"
        "--environment" "$ENVIRONMENT"
    )
    
    case "$MODE" in
        validate)
            script_args+=("--validate-only")
            ;;
        setup|setup-if-needed)
            # Default behavior
            ;;
        force-recreate)
            script_args+=("--force-recreate")
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        script_args+=("--dry-run")
    fi
    
    util_log_info "Setting up GCS infrastructure"
    if "$SCRIPT_DIR/setup_gcs_infrastructure.sh" "${script_args[@]}"; then
        update_stage_status "setup_gcs_infrastructure" "completed" "GCS buckets configured"
        return 0
    else
        update_stage_status "setup_gcs_infrastructure" "failed" "GCS setup failed"
        return 1
    fi
}

# Stage 6: Setup monitoring
stage_setup_monitoring() {
    update_stage_status "setup_monitoring" "in_progress"
    
    local script_args=(
        "--project-id" "$PROJECT_ID"
        "--region" "$REGION"
        "--environment" "$ENVIRONMENT"
    )
    
    case "$MODE" in
        validate)
            script_args+=("--validate-only")
            ;;
        setup|setup-if-needed)
            # Default behavior
            ;;
        force-recreate)
            script_args+=("--force-recreate")
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        script_args+=("--dry-run")
    fi
    
    # Use appropriate monitoring setup script based on environment
    local monitoring_script="setup_monitoring.sh"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        monitoring_script="setup_production_monitoring.sh"
    fi
    
    util_log_info "Setting up monitoring infrastructure"
    if "$SCRIPT_DIR/$monitoring_script" "${script_args[@]}"; then
        update_stage_status "setup_monitoring" "completed" "Monitoring configured"
        return 0
    else
        # Monitoring setup failure is not critical for basic infrastructure
        update_stage_status "setup_monitoring" "failed" "Monitoring setup failed (non-critical)"
        util_log_warning "Monitoring setup failed but continuing (non-critical for basic infrastructure)"
        return 0
    fi
}

# Execute all infrastructure stages
execute_infrastructure_stages() {
    util_log_info "Executing infrastructure setup stages in dependency order"
    
    local failed_stages=()
    local critical_failure=false
    
    # Execute stages in dependency order
    for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
        local stage_function="stage_${stage}"
        
        util_log_info "Starting stage: $stage"
        
        if "$stage_function"; then
            util_log_success "Stage completed: $stage"
        else
            util_log_error "Stage failed: $stage"
            failed_stages+=("$stage")
            
            # Determine if this is a critical failure
            case "$stage" in
                enable_apis|create_service_account|setup_secrets)
                    critical_failure=true
                    ;;
                setup_cloud_sql)
                    if [[ "$MODE" == "validate" ]]; then
                        critical_failure=true
                    fi
                    ;;
            esac
            
            # Stop on critical failures
            if [[ "$critical_failure" == "true" ]]; then
                util_log_error "Critical infrastructure stage failed: $stage"
                util_log_error "Cannot continue with infrastructure setup"
                break
            fi
        fi
        
        show_infrastructure_progress
    done
    
    # Return status based on failures
    if [[ "$critical_failure" == "true" ]]; then
        return 1
    elif [[ ${#failed_stages[@]} -gt 0 ]]; then
        util_log_warning "Some non-critical stages failed: ${failed_stages[*]}"
        return 0
    else
        return 0
    fi
}

# Generate infrastructure setup summary
generate_infrastructure_summary() {
    util_log_header "📋 Infrastructure Setup Summary"
    
    local total_stages=${#INFRASTRUCTURE_STAGES[@]}
    local completed_stages=0
    local failed_stages=0
    local skipped_stages=0
    
    for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        case "$status" in
            completed) ((completed_stages++)) ;;
            failed) ((failed_stages++)) ;;
            skipped) ((skipped_stages++)) ;;
        esac
    done
    
    echo "================================================"
    echo "Infrastructure Setup Summary"
    echo "================================================"
    echo "Environment: $ENVIRONMENT"
    echo "Mode: $MODE"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Dry Run: $([ "$DRY_RUN" = "true" ] && echo "YES" || echo "NO")"
    echo "Setup Time: $(date)"
    echo ""
    echo "Results:"
    echo "  ✅ Completed: $completed_stages/$total_stages stages"
    echo "  ❌ Failed: $failed_stages/$total_stages stages"
    echo "  ⏭️ Skipped: $skipped_stages/$total_stages stages"
    echo ""
    
    echo "Stage Details:"
    for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
        local status="${STAGE_STATUS[$stage]}"
        local result="${STAGE_RESULTS[$stage]}"
        
        case "$status" in
            completed) echo "  ✅ $stage: $result" ;;
            failed) echo "  ❌ $stage: $result" ;;
            skipped) echo "  ⏭️ $stage: $result" ;;
            *) echo "  ⏳ $stage: $status" ;;
        esac
    done
    echo ""
    
    if [[ "$MODE" == "validate" ]]; then
        if [[ $failed_stages -eq 0 ]]; then
            echo "✅ Infrastructure validation PASSED"
            echo "All required infrastructure components are properly configured."
        else
            echo "❌ Infrastructure validation FAILED"
            echo "Some infrastructure components are missing or misconfigured."
        fi
    else
        if [[ $failed_stages -eq 0 ]]; then
            echo "✅ Infrastructure setup COMPLETED successfully"
            echo "All infrastructure components are ready for deployment."
        else
            echo "⚠️ Infrastructure setup completed with WARNINGS"
            echo "Some non-critical components failed but deployment can proceed."
        fi
    fi
    
    echo ""
    echo "Next Steps:"
    case "$MODE" in
        validate)
            if [[ $failed_stages -eq 0 ]]; then
                echo "1. Infrastructure is ready for deployment"
                echo "2. Proceed with application deployment"
            else
                echo "1. Fix failed infrastructure components"
                echo "2. Re-run validation before deployment"
            fi
            ;;
        setup|setup-if-needed)
            echo "1. Verify infrastructure components are working correctly"
            echo "2. Run validation: $0 --mode validate --environment $ENVIRONMENT"
            echo "3. Proceed with application deployment"
            ;;
        force-recreate)
            echo "1. Verify all components were recreated successfully"
            echo "2. Update any external references to recreated resources"
            echo "3. Test the infrastructure before deploying applications"
            ;;
    esac
    
    echo ""
    echo "Integration with Deployment Pipeline:"
    echo "  # For staging - setup missing components"
    echo "  $0 --mode setup-if-needed --environment staging"
    echo ""
    echo "  # For production - validate all components exist"
    echo "  $0 --mode validate --environment production"
    echo ""
    echo "================================================"
}

# Cleanup function
cleanup_infrastructure_setup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        util_log_error "Infrastructure setup failed with exit code $exit_code"
    else
        util_log_success "Infrastructure setup completed"
    fi
    
    return $exit_code
}

# Main execution function
main() {
    parse_arguments "$@"
    initialize_infrastructure_setup
    
    # Setup cleanup handler
    trap cleanup_infrastructure_setup EXIT
    
    # Execute infrastructure stages
    if execute_infrastructure_stages; then
        generate_infrastructure_summary
        
        if [[ "$MODE" == "validate" ]]; then
            # Check if validation passed
            local validation_failed=false
            for stage in "${INFRASTRUCTURE_STAGES[@]}"; do
                if [[ "${STAGE_STATUS[$stage]}" == "failed" ]]; then
                    validation_failed=true
                    break
                fi
            done
            
            if [[ "$validation_failed" == "true" ]]; then
                util_log_error "Infrastructure validation failed"
                exit 1
            else
                util_log_success "Infrastructure validation passed"
                exit 0
            fi
        else
            util_log_success "🎉 Infrastructure setup completed successfully!"
            exit 0
        fi
    else
        util_log_error "Infrastructure setup failed"
        generate_infrastructure_summary
        exit 1
    fi
}

# Execute main function if script is called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi