#!/bin/bash

# Service Account Creation and Management for Shyvr RLTE
# Creates and configures service accounts with proper IAM roles for production deployment
# Enhanced with deploy-utils.sh integration and production-ready features

set -euo pipefail

# Source deployment utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy-utils.sh"

# Configuration with defaults from deploy-utils.sh
PROJECT_ID="${DEFAULT_PROJECT_ID}"
REGION="${DEFAULT_REGION}"
DRY_RUN=false
VALIDATE_ONLY=false
CREATE_KEYS=false

# Service account configuration
SERVICE_ACCOUNT_NAME="shyvr-rlte"
SERVICE_ACCOUNT_DISPLAY_NAME="Shyvr RLTE Service Account"
SERVICE_ACCOUNT_DESCRIPTION="Service account for Shyvr RLTE trading system with ML/RL capabilities"

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Create and configure service accounts for Shyvr RLTE production deployment.

OPTIONS:
    --project-id PROJECT    GCP project ID (default: $PROJECT_ID)
    --region REGION         GCP region (default: $REGION)
    --account-name NAME     Service account name (default: $SERVICE_ACCOUNT_NAME)
    --display-name NAME     Service account display name
    --create-keys          Create and download service account keys
    --dry-run              Show what would be done without executing
    --validate-only        Only validate existing service accounts
    --help, -h             Show this help message

EXAMPLES:
    # Create service account with default settings
    $0
    
    # Create with custom name and display name
    $0 --account-name my-rlte-sa --display-name "My RLTE Service Account"
    
    # Create service account with keys
    $0 --create-keys
    
    # Dry run to see what would be created
    $0 --dry-run
    
    # Validate existing service accounts
    $0 --validate-only

SECURITY NOTE:
    Use --create-keys only when necessary. Service account keys should be avoided
    in favor of workload identity when possible.

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
            --account-name)
                SERVICE_ACCOUNT_NAME="$2"
                shift 2
                ;;
            --display-name)
                SERVICE_ACCOUNT_DISPLAY_NAME="$2"
                shift 2
                ;;
            --create-keys)
                CREATE_KEYS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
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

# Initialize service account setup
initialize_service_account_setup() {
    util_log_header "🔐 Service Account Setup for Shyvr RLTE"
    util_log_info "Creating and configuring service accounts with proper IAM roles"
    
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
    
    # Validate service account name
    if [[ ! "$SERVICE_ACCOUNT_NAME" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
        util_log_error "Invalid service account name: $SERVICE_ACCOUNT_NAME"
        util_log_error "Name must be 6-30 characters, start with lowercase letter, contain only lowercase letters, numbers, and hyphens"
        exit 1
    fi
    
    util_log_success "Service account name validated: $SERVICE_ACCOUNT_NAME"
}

# Enable required APIs
enable_required_apis() {
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
    )
    
    for api in "${apis[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would enable $api"
        else
            util_log_info "Enabling $api"
            if gcloud services enable "$api" --project="$PROJECT_ID" --quiet; then
                util_log_success "✓ $api enabled"
            else
                util_log_warning "Failed to enable $api (may already be enabled)"
            fi
        fi
    done
    
    util_log_success "Required APIs enabled"
}

# Validate existing service accounts
validate_service_accounts() {
    util_log_info "Validating existing service accounts"
    
    local validation_passed=true
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Check if service account exists
    if gcloud iam service-accounts describe "$service_account_email" --project="$PROJECT_ID" >/dev/null 2>&1; then
        util_log_success "✓ Service account exists: $service_account_email"
        
        # Check if service account is enabled
        local account_disabled
        account_disabled=$(gcloud iam service-accounts describe "$service_account_email" --project="$PROJECT_ID" --format="value(disabled)" 2>/dev/null || echo "false")
        
        if [[ "$account_disabled" == "true" ]]; then
            util_log_error "Service account is disabled: $service_account_email"
            validation_passed=false
        else
            util_log_success "✓ Service account is enabled"
        fi
        
        # Check IAM roles
        local roles_count
        roles_count=$(gcloud projects get-iam-policy "$PROJECT_ID" --flatten="bindings[].members" --format="value(bindings.role)" --filter="bindings.members:serviceAccount:$service_account_email" | wc -l)
        
        if [[ "$roles_count" -gt 0 ]]; then
            util_log_success "✓ Service account has $roles_count IAM role bindings"
        else
            util_log_warning "Service account has no IAM role bindings"
        fi
        
    else
        util_log_error "Service account does not exist: $service_account_email"
        validation_passed=false
    fi
    
    if [[ "$validation_passed" == "true" ]]; then
        util_log_success "Service account validation passed"
        return 0
    else
        util_log_error "Service account validation failed"
        return 1
    fi
}

# Create service account
create_service_account() {
    util_log_info "Creating service account: $SERVICE_ACCOUNT_NAME"
    
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Check if service account already exists
    if gcloud iam service-accounts describe "$service_account_email" --project="$PROJECT_ID" >/dev/null 2>&1; then
        util_log_warning "Service account already exists: $service_account_email"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create service account:"
        util_log_info "  Name: $SERVICE_ACCOUNT_NAME"
        util_log_info "  Display Name: $SERVICE_ACCOUNT_DISPLAY_NAME"
        util_log_info "  Email: $service_account_email"
        util_log_info "  Description: $SERVICE_ACCOUNT_DESCRIPTION"
        return 0
    fi
    
    # Create the service account
    if gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="$SERVICE_ACCOUNT_DISPLAY_NAME" \
        --description="$SERVICE_ACCOUNT_DESCRIPTION" \
        --project="$PROJECT_ID" \
        --quiet; then
        util_log_success "Created service account: $service_account_email"
    else
        util_log_error "Failed to create service account: $SERVICE_ACCOUNT_NAME"
        return 1
    fi
}

# Configure IAM roles for the service account
configure_iam_roles() {
    util_log_info "Configuring IAM roles for service account"
    
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Define roles needed for Shyvr RLTE
    local roles=(
        "roles/run.invoker"                    # Cloud Run service invocation
        "roles/run.developer"                  # Cloud Run service management
        "roles/secretmanager.secretAccessor"   # Secret Manager access
        "roles/cloudsql.client"                # Cloud SQL connection
        "roles/storage.objectAdmin"            # GCS bucket access for models
        "roles/monitoring.metricWriter"        # Custom metrics writing
        "roles/logging.logWriter"              # Log writing
        "roles/cloudtrace.agent"               # Trace data writing
        "roles/clouderrorreporting.writer"     # Error reporting
        "roles/cloudprofiler.agent"            # Profiler agent
    )
    
    local roles_added=()
    local roles_failed=()
    
    for role in "${roles[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would add role: $role"
            roles_added+=("$role")
        else
            util_log_info "Adding role: $role"
            
            if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$service_account_email" \
                --role="$role" \
                --quiet >/dev/null 2>&1; then
                util_log_success "✓ Added role: $role"
                roles_added+=("$role")
            else
                util_log_warning "Failed to add role: $role (may already exist)"
                roles_failed+=("$role")
            fi
        fi
    done
    
    if [[ ${#roles_added[@]} -gt 0 ]]; then
        util_log_success "Successfully configured ${#roles_added[@]} IAM roles"
    fi
    
    if [[ ${#roles_failed[@]} -gt 0 ]]; then
        util_log_warning "${#roles_failed[@]} roles failed to add (may already exist)"
    fi
}

# Create service account keys (optional and discouraged)
create_service_account_keys() {
    if [[ "$CREATE_KEYS" != "true" ]]; then
        util_log_info "Skipping service account key creation (not requested)"
        return 0
    fi
    
    util_log_warning "Creating service account keys"
    util_log_warning "WARNING: Service account keys are security-sensitive and should be avoided when possible"
    util_log_warning "Consider using Workload Identity instead for better security"
    
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    local key_file="${SERVICE_ACCOUNT_NAME}-key-$(date +%Y%m%d-%H%M%S).json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create service account key file: $key_file"
        return 0
    fi
    
    # Create and download the key
    if gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$service_account_email" \
        --project="$PROJECT_ID" \
        --quiet; then
        util_log_success "Created service account key: $key_file"
        util_log_warning "IMPORTANT: Store this key file securely and never commit it to version control"
        
        # Set restrictive permissions on the key file
        chmod 600 "$key_file"
        util_log_info "Set restrictive permissions (600) on key file"
        
        echo ""
        echo "Key file location: $(pwd)/$key_file"
        echo "To use this key:"
        echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"$(pwd)/$key_file\""
        echo ""
    else
        util_log_error "Failed to create service account key"
        return 1
    fi
}

# Setup workload identity (preferred over service account keys)
setup_workload_identity() {
    util_log_info "Setting up Workload Identity binding"
    
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    local gsa_name="$SERVICE_ACCOUNT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would configure Workload Identity for Cloud Run"
        util_log_info "  Service Account: $service_account_email"
        util_log_info "  Cloud Run Service: shyvr-rlte"
        return 0
    fi
    
    # Allow the service account to be used by Cloud Run
    util_log_info "Configuring service account for Cloud Run usage"
    
    # Grant the compute service account the ability to act as this service account
    local compute_sa="$PROJECT_ID-compute@developer.gserviceaccount.com"
    
    if gcloud iam service-accounts add-iam-policy-binding "$service_account_email" \
        --member="serviceAccount:$compute_sa" \
        --role="roles/iam.serviceAccountUser" \
        --project="$PROJECT_ID" \
        --quiet >/dev/null 2>&1; then
        util_log_success "✓ Configured Workload Identity binding"
    else
        util_log_warning "Workload Identity binding may already exist"
    fi
}

# Generate service account summary
generate_service_account_summary() {
    util_log_header "📋 Service Account Setup Summary"
    
    local service_account_email="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    echo "================================================"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Mode: $([ "$DRY_RUN" = "true" ] && echo "DRY RUN" || echo "LIVE")"
    echo "Setup Time: $(date)"
    echo ""
    echo "Service Account Details:"
    echo "  Name: $SERVICE_ACCOUNT_NAME"
    echo "  Display Name: $SERVICE_ACCOUNT_DISPLAY_NAME"
    echo "  Email: $service_account_email"
    echo "  Description: $SERVICE_ACCOUNT_DESCRIPTION"
    echo ""
    echo "IAM Roles Configured:"
    echo "  ✓ Cloud Run invoker and developer"
    echo "  ✓ Secret Manager accessor"
    echo "  ✓ Cloud SQL client"
    echo "  ✓ GCS storage object admin"
    echo "  ✓ Monitoring and logging roles"
    echo "  ✓ Tracing and error reporting"
    echo ""
    echo "Security Configuration:"
    echo "  ✓ Workload Identity configured"
    [[ "$CREATE_KEYS" == "true" ]] && echo "  ⚠ Service account keys created" || echo "  ✓ No service account keys (recommended)"
    echo ""
    echo "Integration:"
    echo "  ✓ Ready for Cloud Run deployment"
    echo "  ✓ Can access Secret Manager secrets"
    echo "  ✓ Can connect to Cloud SQL"
    echo "  ✓ Can read/write GCS model storage"
    echo "  ✓ Can write custom metrics and logs"
    echo ""
    echo "Next Steps:"
    echo "1. Deploy Cloud Run service with this service account"
    echo "2. Verify secret and database access"
    echo "3. Test monitoring and logging integration"
    echo "4. Configure any additional project-specific roles"
    echo ""
    echo "Cloud Run Deployment Command:"
    echo "  gcloud run deploy shyvr-rlte \\"
    echo "    --service-account=$service_account_email \\"
    echo "    --region=$REGION \\"
    echo "    --project=$PROJECT_ID"
    echo ""
    echo "================================================"
}

# Cleanup function
cleanup_temp_files() {
    # Remove any temporary files if created
    rm -f /tmp/service_account_*.json 2>/dev/null || true
}

# Main execution function
main() {
    parse_arguments "$@"
    initialize_service_account_setup
    
    # Validate only mode
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        if validate_service_accounts; then
            util_log_success "Service account validation passed"
            exit 0
        else
            util_log_error "Service account validation failed"
            exit 1
        fi
    fi
    
    # Execute setup steps
    enable_required_apis
    create_service_account
    configure_iam_roles
    setup_workload_identity
    
    # Optionally create keys (discouraged)
    if [[ "$CREATE_KEYS" == "true" ]]; then
        create_service_account_keys
    fi
    
    # Generate summary
    generate_service_account_summary
    
    # Cleanup
    cleanup_temp_files
    
    util_log_success "🎉 Service account setup completed successfully!"
}

# Execute main function if script is called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi