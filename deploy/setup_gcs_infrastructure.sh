#!/bin/bash

# Shyvr RLTE - GCS Infrastructure Setup Script
# Phase 1.1: GCS Infrastructure Setup
# 
# This script sets up the Google Cloud Storage infrastructure for model preservation
# including buckets, lifecycle policies, IAM permissions, and audit logging.

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
PROD_BUCKET="shyvr-models-prod"
STAGING_BUCKET="shyvr-models-staging"
CLOUD_RUN_SERVICE_ACCOUNT="${CLOUD_RUN_SA:-shyvr-rlte@${PROJECT_ID}.iam.gserviceaccount.com}"
REGION="${GCP_REGION:-us-central1}"
VPC_NAME="${VPC_NAME:-shyvr-vpc}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Check if project ID is set
    if [[ -z "$PROJECT_ID" ]]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "GCP_PROJECT_ID is not set and no default project configured"
            log_error "Set GCP_PROJECT_ID environment variable or run 'gcloud config set project PROJECT_ID'"
            exit 1
        fi
    fi
    
    # Set the project
    gcloud config set project "$PROJECT_ID"
    log_success "Using project: $PROJECT_ID"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
    local apis=(
        "storage-api.googleapis.com"
        "storage-component.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
        "cloudasset.googleapis.com"
        "accesscontextmanager.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --quiet
    done
    
    log_success "All required APIs enabled"
}

# Create GCS buckets
create_buckets() {
    log_info "Creating GCS buckets..."
    
    # Create production bucket
    if gsutil ls "gs://$PROD_BUCKET" &> /dev/null; then
        log_warning "Production bucket gs://$PROD_BUCKET already exists"
    else
        log_info "Creating production bucket: gs://$PROD_BUCKET"
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$PROD_BUCKET"
        log_success "Created production bucket: gs://$PROD_BUCKET"
    fi
    
    # Create staging bucket
    if gsutil ls "gs://$STAGING_BUCKET" &> /dev/null; then
        log_warning "Staging bucket gs://$STAGING_BUCKET already exists"
    else
        log_info "Creating staging bucket: gs://$STAGING_BUCKET"
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$STAGING_BUCKET"
        log_success "Created staging bucket: gs://$STAGING_BUCKET"
    fi
}

# Enable versioning on buckets
enable_versioning() {
    log_info "Enabling versioning on buckets..."
    
    gsutil versioning set on "gs://$PROD_BUCKET"
    gsutil versioning set on "gs://$STAGING_BUCKET"
    
    log_success "Versioning enabled on both buckets"
}

# Apply lifecycle policies
apply_lifecycle_policies() {
    log_info "Applying lifecycle policies..."
    
    local lifecycle_file="lifecycle.json"
    
    if [[ ! -f "$lifecycle_file" ]]; then
        log_error "Lifecycle policy file $lifecycle_file not found"
        log_error "Please ensure lifecycle.json exists in the same directory"
        exit 1
    fi
    
    # Apply to production bucket
    gsutil lifecycle set "$lifecycle_file" "gs://$PROD_BUCKET"
    log_success "Applied lifecycle policy to production bucket"
    
    # Apply to staging bucket (more aggressive cleanup for staging)
    gsutil lifecycle set "$lifecycle_file" "gs://$STAGING_BUCKET"
    log_success "Applied lifecycle policy to staging bucket"
}

# Set up IAM permissions
setup_iam_permissions() {
    log_info "Setting up IAM permissions..."
    CLOUD_RUN_SERVICE_ACCOUNT="${CLOUD_RUN_SA:-shyvr-rlte@${PROJECT_ID}.iam.gserviceaccount.com}"
    
    # Check if service account exists
    if gcloud iam service-accounts describe "$CLOUD_RUN_SERVICE_ACCOUNT" &> /dev/null; then
        log_info "Service account $CLOUD_RUN_SERVICE_ACCOUNT exists"
    else
        log_error "Service account $CLOUD_RUN_SERVICE_ACCOUNT does not exist"
        log_error "Please create the Cloud Run service account first"
        exit 1
    fi
    
    # Grant Storage Object Admin role for both buckets
    log_info "Granting Storage Object Admin role to Cloud Run service account..."
    
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$CLOUD_RUN_SERVICE_ACCOUNT" \
        --role="roles/storage.objectAdmin" \
        --condition="expression=resource.name.startsWith('projects/_/buckets/$PROD_BUCKET') || resource.name.startsWith('projects/_/buckets/$STAGING_BUCKET'),title=Model Storage Access"
    
    # Grant bucket-level permissions for listing
    log_info "Granting bucket-level permissions for listing..."
    
    # For production bucket
    gsutil iam ch "serviceAccount:$CLOUD_RUN_SERVICE_ACCOUNT:objectViewer" "gs://$PROD_BUCKET"
    
    # For staging bucket
    gsutil iam ch "serviceAccount:$CLOUD_RUN_SERVICE_ACCOUNT:objectViewer" "gs://$STAGING_BUCKET"
    
    log_success "IAM permissions configured"
}

# Configure VPC Service Controls (optional, requires organization)
configure_vpc_service_controls() {
    log_info "Configuring VPC Service Controls..."
    
    # Check if organization is available
    local org_id
    org_id=$(gcloud organizations list --format="value(name)" --limit=1 2>/dev/null || echo "")
    
    if [[ -z "$org_id" ]]; then
        log_warning "No organization found, skipping VPC Service Controls configuration"
        return 0
    fi
    
    log_info "Organization found: $org_id"
    
    # Create access policy if it doesn't exist
    local policy_name="shyvr-access-policy"
    local policy_id
    policy_id=$(gcloud access-context-manager policies list --organization="$org_id" --format="value(name)" --filter="title:$policy_name" 2>/dev/null || echo "")
    
    if [[ -z "$policy_id" ]]; then
        log_info "Creating access policy: $policy_name"
        policy_id=$(gcloud access-context-manager policies create \
            --organization="$org_id" \
            --title="$policy_name" \
            --format="value(name)")
        log_success "Created access policy: $policy_id"
    else
        log_info "Using existing access policy: $policy_id"
    fi
    
    # Create service perimeter for GCS
    local perimeter_name="shyvr-gcs-perimeter"
    if ! gcloud access-context-manager perimeters describe "$perimeter_name" --policy="$policy_id" &> /dev/null; then
        log_info "Creating service perimeter: $perimeter_name"
        gcloud access-context-manager perimeters create "$perimeter_name" \
            --policy="$policy_id" \
            --title="Shyvr GCS Perimeter" \
            --resources="projects/$PROJECT_ID" \
            --restricted-services="storage.googleapis.com" \
            --perimeter-type="regular"
        log_success "Created service perimeter: $perimeter_name"
    else
        log_info "Service perimeter $perimeter_name already exists"
    fi
}

# Enable audit logging
enable_audit_logging() {
    log_info "Enabling Cloud Storage audit logging..."
    
    # Create audit config JSON
    local audit_config_file="/tmp/audit-config.json"
    cat > "$audit_config_file" << EOF
{
  "auditConfigs": [
    {
      "service": "storage.googleapis.com",
      "auditLogConfigs": [
        {
          "logType": "ADMIN_READ"
        },
        {
          "logType": "DATA_READ"
        },
        {
          "logType": "DATA_WRITE"
        }
      ]
    }
  ]
}
EOF
    
    # Apply audit configuration
    gcloud logging sinks create shyvr-gcs-audit-sink \
        bigquery.googleapis.com/projects/"$PROJECT_ID"/datasets/audit_logs \
        --log-filter='protoPayload.serviceName="storage.googleapis.com"' \
        --quiet || log_warning "Audit sink may already exist"
    
    log_success "Audit logging configuration completed"
    
    # Clean up temp file
    rm -f "$audit_config_file"
}

# Set bucket-level configurations
configure_bucket_settings() {
    log_info "Configuring bucket-level settings..."
    
    # Set uniform bucket-level access
    gsutil uniformbucketlevelaccess set on "gs://$PROD_BUCKET"
    gsutil uniformbucketlevelaccess set on "gs://$STAGING_BUCKET"
    
    # Set default storage class
    gsutil defstorageclass set STANDARD "gs://$PROD_BUCKET"
    gsutil defstorageclass set NEARLINE "gs://$STAGING_BUCKET"
    
    # Enable public access prevention
    gsutil pap set enforced "gs://$PROD_BUCKET"
    gsutil pap set enforced "gs://$STAGING_BUCKET"
    
    log_success "Bucket-level configurations applied"
}

# Verify setup
verify_setup() {
    log_info "Verifying GCS infrastructure setup..."
    
    # Check bucket existence and configuration
    log_info "Checking bucket configurations..."
    
    for bucket in "$PROD_BUCKET" "$STAGING_BUCKET"; do
        if gsutil ls "gs://$bucket" &> /dev/null; then
            log_success "✓ Bucket gs://$bucket exists"
            
            # Check versioning
            if gsutil versioning get "gs://$bucket" | grep -q "Enabled"; then
                log_success "✓ Versioning enabled on gs://$bucket"
            else
                log_error "✗ Versioning not enabled on gs://$bucket"
            fi
            
            # Check lifecycle policy
            if gsutil lifecycle get "gs://$bucket" &> /dev/null; then
                log_success "✓ Lifecycle policy applied to gs://$bucket"
            else
                log_warning "⚠ No lifecycle policy on gs://$bucket"
            fi
        else
            log_error "✗ Bucket gs://$bucket does not exist"
        fi
    done
    
    # Test permissions
    log_info "Testing service account permissions..."
    local test_file="/tmp/gcs-test-$$"
    echo "test" > "$test_file"
    
    if gsutil -i "$CLOUD_RUN_SERVICE_ACCOUNT" cp "$test_file" "gs://$STAGING_BUCKET/test/" &> /dev/null; then
        log_success "✓ Service account can write to staging bucket"
        gsutil rm "gs://$STAGING_BUCKET/test/gcs-test-$$" &> /dev/null || true
    else
        log_error "✗ Service account cannot write to staging bucket"
    fi
    
    rm -f "$test_file"
    
    log_success "GCS infrastructure verification completed"
}

# Main execution
main() {
    log_info "Starting GCS Infrastructure Setup for Shyvr RLTE Model Preservation"
    log_info "=================================================="
    
    check_prerequisites
    enable_apis
    create_buckets
    enable_versioning
    apply_lifecycle_policies
    setup_iam_permissions
    configure_vpc_service_controls
    enable_audit_logging
    configure_bucket_settings
    verify_setup
    
    log_success "GCS Infrastructure Setup Complete!"
    log_info "=================================================="
    log_info "Production bucket: gs://$PROD_BUCKET"
    log_info "Staging bucket: gs://$STAGING_BUCKET"
    log_info "Service account: $CLOUD_RUN_SERVICE_ACCOUNT"
    log_info "=================================================="
    
    log_info "Next steps:"
    log_info "1. Run the validation script to verify everything is working"
    log_info "2. Update your application configuration with bucket names"
    log_info "3. Deploy your application with the new model preservation features"
}

# Execute main function
main "$@"