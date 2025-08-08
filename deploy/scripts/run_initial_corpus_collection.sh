#!/bin/bash

# Initial Corpus Collection Deployment Script
# Production-ready deployment script for executing initial corpus collection
# on Google Cloud Run with proper integration to existing infrastructure

set -euo pipefail

# Source deployment utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"

# Check if deploy-utils.sh exists and source it
if [[ -f "$DEPLOY_DIR/deploy-utils.sh" ]]; then
    source "$DEPLOY_DIR/deploy-utils.sh"
else
    # Fallback logging functions if deploy-utils not available
    util_log_info() { echo "[INFO] $1"; }
    util_log_success() { echo "[SUCCESS] $1"; }
    util_log_warning() { echo "[WARNING] $1"; }
    util_log_error() { echo "[ERROR] $1" >&2; }
    util_log_header() { echo "[HEADER] $1"; }
fi

# Configuration with defaults
PROJECT_ID="${DEFAULT_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${DEFAULT_REGION:-us-central1}"
SERVICE_NAME="initial-corpus-collector"
JOB_NAME="initial-corpus-collection-job"
DRY_RUN=false
VALIDATE_ONLY=false
ENVIRONMENT="production"

# Collection parameters
COLLECTION_TOKENS=""
COLLECTION_DAYS=""
CORPUS_VERSION=""
EXPORT_GCS=true
RESUME_CHECKPOINT=""

# Cloud Run configuration
MEMORY="4Gi"
CPU="2"
TIMEOUT="3600s"  # 1 hour timeout
CONCURRENCY="1"
MAX_RETRIES="1"

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy and execute initial corpus collection on Google Cloud Run.

OPTIONS:
    --environment ENV       Target environment (development/staging/production)
    --project-id PROJECT    GCP project ID (default: current project)
    --region REGION         GCP region (default: us-central1)
    
    Collection Options:
    --tokens TOKENS         Comma-separated token list (default: BTC,ETH,BNB,SOL,ADA,MATIC,AVAX,DOT,LINK,UNI)
    --days DAYS            Number of days to collect (default: environment specific)
    --corpus-version VER    Custom corpus version name
    --no-export-gcs        Skip GCS export after collection
    --resume-checkpoint     Resume from checkpoint file
    
    Execution Options:
    --dry-run              Show what would be deployed without executing
    --validate-only        Only validate prerequisites
    --check-secrets        Check Secret Manager access
    --setup-monitoring     Configure monitoring and alerting
    
    Cloud Run Options:
    --memory MEMORY        Memory allocation (default: 4Gi)
    --cpu CPU              CPU allocation (default: 2)
    --timeout TIMEOUT      Job timeout (default: 3600s)
    
    --help, -h             Show this help message

EXAMPLES:
    # Validate prerequisites for production
    $0 --environment production --validate-only
    
    # Deploy and run development collection (short period)
    $0 --environment development
    
    # Production deployment with full collection
    $0 --environment production --tokens BTC,ETH,SOL --days 180
    
    # Dry run to see what would be deployed
    $0 --environment production --dry-run
    
    # Resume from previous checkpoint
    $0 --environment production --resume-checkpoint gs://shyvr-models-prod/checkpoints/corpus_20240108.json

PREREQUISITES:
    - Google Cloud SDK configured
    - Access to shvyr-ai-bots project
    - Secret Manager API enabled
    - Cloud Run API enabled
    - Required secrets configured in Secret Manager

MONITORING:
    The deployment automatically sets up monitoring and alerting through
    existing infrastructure integration.

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --tokens)
                COLLECTION_TOKENS="$2"
                shift 2
                ;;
            --days)
                COLLECTION_DAYS="$2"
                shift 2
                ;;
            --corpus-version)
                CORPUS_VERSION="$2"
                shift 2
                ;;
            --no-export-gcs)
                EXPORT_GCS=false
                shift
                ;;
            --resume-checkpoint)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --check-secrets)
                CHECK_SECRETS=true
                shift
                ;;
            --setup-monitoring)
                SETUP_MONITORING=true
                shift
                ;;
            --memory)
                MEMORY="$2"
                shift 2
                ;;
            --cpu)
                CPU="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
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

# Initialize deployment
initialize_deployment() {
    util_log_header "🤖 Initial Corpus Collection Deployment"
    util_log_info "Environment: $ENVIRONMENT"
    util_log_info "Project: $PROJECT_ID"
    util_log_info "Region: $REGION"
    util_log_info "Mode: $([ "$DRY_RUN" = "true" ] && echo "DRY RUN" || echo "LIVE")"
    echo ""
    
    # Validate environment
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        util_log_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi
    
    # Set project context
    if ! gcloud config set project "$PROJECT_ID" --quiet; then
        util_log_error "Failed to set project context to $PROJECT_ID"
        exit 1
    fi
    
    util_log_success "Deployment initialized"
}

# Check prerequisites
check_prerequisites() {
    util_log_info "Checking deployment prerequisites..."
    
    local prerequisites_passed=true
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        util_log_error "Google Cloud SDK not found"
        prerequisites_passed=false
    else
        util_log_success "✓ Google Cloud SDK available"
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        util_log_error "Not authenticated with Google Cloud"
        prerequisites_passed=false
    else
        util_log_success "✓ Google Cloud authentication active"
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        util_log_error "Cannot access project: $PROJECT_ID"
        prerequisites_passed=false
    else
        util_log_success "✓ Project access verified"
    fi
    
    # Check required APIs
    local required_apis=(
        "run.googleapis.com"
        "secretmanager.googleapis.com"
        "sqladmin.googleapis.com"
        "storage-api.googleapis.com"
    )
    
    for api in "${required_apis[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            util_log_success "✓ API enabled: $api"
        else
            util_log_warning "⚠ API not enabled: $api"
            if [[ "$DRY_RUN" != "true" ]]; then
                util_log_info "Enabling API: $api"
                gcloud services enable "$api" --quiet || {
                    util_log_error "Failed to enable API: $api"
                    prerequisites_passed=false
                }
            fi
        fi
    done
    
    # Check corpus collection script exists
    local script_path="$PROJECT_ROOT/scripts/collect_initial_corpus.py"
    if [[ ! -f "$script_path" ]]; then
        util_log_error "Corpus collection script not found: $script_path"
        prerequisites_passed=false
    else
        util_log_success "✓ Corpus collection script found"
    fi
    
    # Check Docker context (for Cloud Run)
    if [[ ! -f "$PROJECT_ROOT/Dockerfile" ]]; then
        util_log_warning "⚠ Dockerfile not found in project root"
    else
        util_log_success "✓ Dockerfile found"
    fi
    
    if [[ "$prerequisites_passed" != "true" ]]; then
        util_log_error "Prerequisites check failed"
        exit 1
    fi
    
    util_log_success "Prerequisites check passed"
}

# Check Secret Manager access and required secrets
check_secret_manager() {
    util_log_info "Checking Secret Manager access..."
    
    local required_secrets=(
        "DATABASE_URL"
        "COINGECKO_API_KEY"
    )
    
    local optional_secrets=(
        "LUNARCRUSH_API_KEY"
        "HELIUS_API_KEY"
    )
    
    local secrets_available=0
    local total_secrets=0
    
    # Check required secrets
    for secret in "${required_secrets[@]}"; do
        ((total_secrets++))
        if gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
            util_log_success "✓ Required secret available: $secret"
            ((secrets_available++))
        else
            util_log_error "✗ Required secret missing: $secret"
        fi
    done
    
    # Check optional secrets
    for secret in "${optional_secrets[@]}"; do
        ((total_secrets++))
        if gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
            util_log_success "✓ Optional secret available: $secret"
            ((secrets_available++))
        else
            util_log_info "ℹ Optional secret not configured: $secret"
        fi
    done
    
    util_log_info "Secret Manager check: $secrets_available/$total_secrets secrets available"
    
    # Must have at least required secrets
    if [[ $secrets_available -lt ${#required_secrets[@]} ]]; then
        util_log_error "Required secrets are missing - run setup_secrets.sh first"
        exit 1
    fi
    
    util_log_success "Secret Manager access verified"
}

# Check GCS bucket access
check_gcs_access() {
    util_log_info "Checking GCS bucket access..."
    
    local bucket="shyvr-models-prod"
    
    # Check bucket exists and is accessible
    if gsutil ls "gs://$bucket/" &> /dev/null; then
        util_log_success "✓ GCS bucket accessible: gs://$bucket/"
    else
        util_log_error "✗ Cannot access GCS bucket: gs://$bucket/"
        util_log_info "Ensure bucket exists and you have Storage Admin permissions"
        exit 1
    fi
    
    # Check training-data directory structure
    local training_data_path="gs://$bucket/training-data/"
    if gsutil ls "$training_data_path" &> /dev/null; then
        util_log_success "✓ Training data directory exists"
    else
        util_log_info "Creating training data directory structure..."
        if [[ "$DRY_RUN" != "true" ]]; then
            # Create directory structure by uploading a placeholder
            echo "Initial corpus training data" | gsutil cp - "$training_data_path.placeholder"
            gsutil rm "$training_data_path.placeholder" &> /dev/null || true
        fi
        util_log_success "✓ Training data directory structure ready"
    fi
}

# Build and deploy Cloud Run job
deploy_cloud_run_job() {
    util_log_info "Deploying Cloud Run job for corpus collection..."
    
    local image_name="gcr.io/$PROJECT_ID/initial-corpus-collector:latest"
    local service_account="shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would deploy Cloud Run job with:"
        util_log_info "  Image: $image_name"
        util_log_info "  Memory: $MEMORY"
        util_log_info "  CPU: $CPU"  
        util_log_info "  Timeout: $TIMEOUT"
        util_log_info "  Service Account: $service_account"
        return 0
    fi
    
    # Build container image
    util_log_info "Building container image..."
    if ! gcloud builds submit \
        --tag "$image_name" \
        --project="$PROJECT_ID" \
        "$PROJECT_ROOT"; then
        util_log_error "Failed to build container image"
        exit 1
    fi
    
    util_log_success "Container image built: $image_name"
    
    # Prepare environment variables
    local env_vars=(
        "ENVIRONMENT=$ENVIRONMENT"
        "PROJECT_ID=$PROJECT_ID"
        "GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
    )
    
    if [[ -n "$COLLECTION_TOKENS" ]]; then
        env_vars+=("COLLECTION_TOKENS=$COLLECTION_TOKENS")
    fi
    
    if [[ -n "$COLLECTION_DAYS" ]]; then
        env_vars+=("COLLECTION_DAYS=$COLLECTION_DAYS")
    fi
    
    if [[ -n "$CORPUS_VERSION" ]]; then
        env_vars+=("CORPUS_VERSION=$CORPUS_VERSION")
    fi
    
    if [[ "$EXPORT_GCS" == "false" ]]; then
        env_vars+=("NO_EXPORT_GCS=true")
    fi
    
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        env_vars+=("RESUME_CHECKPOINT=$RESUME_CHECKPOINT")
    fi
    
    # Deploy Cloud Run job
    util_log_info "Deploying Cloud Run job..."
    
    local deploy_cmd=(
        gcloud run jobs replace
        --image="$image_name"
        --region="$REGION"
        --project="$PROJECT_ID"
        --memory="$MEMORY"
        --cpu="$CPU"
        --task-timeout="$TIMEOUT"
        --parallelism=1
        --concurrency="$CONCURRENCY"
        --max-retries="$MAX_RETRIES"
        --service-account="$service_account"
    )
    
    # Add environment variables
    for env_var in "${env_vars[@]}"; do
        deploy_cmd+=(--set-env-vars="$env_var")
    done
    
    # Create job configuration
    cat > "/tmp/corpus-collection-job.yaml" << EOF
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: $JOB_NAME
  labels:
    environment: $ENVIRONMENT
    component: corpus-collection
spec:
  spec:
    template:
      spec:
        template:
          spec:
            containers:
            - image: $image_name
              command: ["python", "/app/scripts/collect_initial_corpus.py"]
              args:
                - "--environment"
                - "$ENVIRONMENT" 
                - "--project-id"
                - "$PROJECT_ID"
                - "--output-json"
$(if [[ -n "$COLLECTION_TOKENS" ]]; then echo "                - \"--tokens\""; echo "                - \"$COLLECTION_TOKENS\""; fi)
$(if [[ -n "$COLLECTION_DAYS" ]]; then echo "                - \"--days\""; echo "                - \"$COLLECTION_DAYS\""; fi)
$(if [[ "$EXPORT_GCS" == "true" ]]; then echo "                - \"--export-gcs\""; fi)
$(if [[ -n "$RESUME_CHECKPOINT" ]]; then echo "                - \"--resume-from-checkpoint\""; echo "                - \"$RESUME_CHECKPOINT\""; fi)
              resources:
                limits:
                  memory: $MEMORY
                  cpu: $CPU
              env:
$(for env_var in "${env_vars[@]}"; do
    IFS='=' read -r key value <<< "$env_var"
    echo "              - name: $key"
    echo "                value: \"$value\""
done)
              # Secret Manager integration
              - name: DATABASE_URL
                valueFrom:
                  secretKeyRef:
                    name: DATABASE_URL
                    key: latest
              - name: COINGECKO_API_KEY  
                valueFrom:
                  secretKeyRef:
                    name: COINGECKO_API_KEY
                    key: latest
            serviceAccountName: $service_account
            restartPolicy: Never
          timeoutSeconds: $(echo "$TIMEOUT" | sed 's/s$//')
EOF
    
    # Deploy the job
    if gcloud run jobs replace "/tmp/corpus-collection-job.yaml" \
        --region="$REGION" \
        --project="$PROJECT_ID"; then
        util_log_success "Cloud Run job deployed: $JOB_NAME"
    else
        util_log_error "Failed to deploy Cloud Run job"
        exit 1
    fi
    
    # Cleanup
    rm -f "/tmp/corpus-collection-job.yaml"
}

# Execute the corpus collection job
execute_collection_job() {
    util_log_info "Executing corpus collection job..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would execute job: $JOB_NAME"
        util_log_info "[DRY RUN] Collection parameters:"
        util_log_info "  Environment: $ENVIRONMENT"
        util_log_info "  Tokens: ${COLLECTION_TOKENS:-default}"
        util_log_info "  Days: ${COLLECTION_DAYS:-environment default}"
        util_log_info "  Export GCS: $EXPORT_GCS"
        return 0
    fi
    
    # Execute the job
    util_log_info "Starting job execution..."
    
    local execution_name
    execution_name=$(gcloud run jobs execute "$JOB_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(metadata.name)" \
        --wait)
    
    if [[ $? -eq 0 ]]; then
        util_log_success "Job execution completed: $execution_name"
        
        # Get job logs
        util_log_info "Fetching job logs..."
        gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB_NAME\"" \
            --limit=100 \
            --format="value(textPayload)" \
            --project="$PROJECT_ID" || true
            
    else
        util_log_error "Job execution failed"
        
        # Get error logs
        util_log_info "Fetching error logs..."
        gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB_NAME\" AND severity>=ERROR" \
            --limit=50 \
            --format="value(textPayload)" \
            --project="$PROJECT_ID" || true
            
        exit 1
    fi
}

# Setup monitoring and alerting
setup_monitoring() {
    util_log_info "Setting up monitoring and alerting..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would setup monitoring for corpus collection job"
        return 0
    fi
    
    # Create log-based alerting for job failures
    local alert_policy_name="initial-corpus-collection-failures"
    
    # Check if alert policy already exists
    if gcloud alpha monitoring policies list \
        --filter="displayName:'$alert_policy_name'" \
        --format="value(name)" \
        --project="$PROJECT_ID" | grep -q .; then
        util_log_info "Monitoring alert policy already exists"
    else
        util_log_info "Creating monitoring alert policy..."
        
        # Create alert policy configuration
        cat > "/tmp/alert-policy.json" << EOF
{
  "displayName": "$alert_policy_name",
  "documentation": {
    "content": "Alert when initial corpus collection job fails"
  },
  "conditions": [
    {
      "displayName": "Cloud Run Job Failure",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB_NAME\" AND severity>=ERROR",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": "0",
        "duration": "60s",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_COUNT"
          }
        ]
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "86400s"
  },
  "enabled": true
}
EOF
        
        gcloud alpha monitoring policies create --policy-from-file="/tmp/alert-policy.json" \
            --project="$PROJECT_ID" || util_log_warning "Failed to create alert policy"
        
        rm -f "/tmp/alert-policy.json"
        util_log_success "Monitoring alert policy created"
    fi
}

# Main execution function
main() {
    parse_arguments "$@"
    initialize_deployment
    
    # Handle validation-only mode
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        check_prerequisites
        check_secret_manager
        check_gcs_access
        util_log_success "✅ All validations passed - ready for corpus collection deployment"
        exit 0
    fi
    
    # Handle secrets check mode
    if [[ "${CHECK_SECRETS:-false}" == "true" ]]; then
        check_secret_manager
        exit 0
    fi
    
    # Handle monitoring setup mode
    if [[ "${SETUP_MONITORING:-false}" == "true" ]]; then
        setup_monitoring
        exit 0
    fi
    
    # Full deployment workflow
    check_prerequisites
    check_secret_manager  
    check_gcs_access
    deploy_cloud_run_job
    
    if [[ "$DRY_RUN" != "true" ]]; then
        execute_collection_job
        setup_monitoring
    fi
    
    # Summary
    util_log_header "🎉 Deployment Summary"
    util_log_success "Environment: $ENVIRONMENT"
    util_log_success "Cloud Run job: $JOB_NAME"
    util_log_success "Region: $REGION"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "DRY RUN completed - no actual deployment performed"
        util_log_info "To execute for real, run without --dry-run flag"
    else
        util_log_success "Corpus collection deployment completed successfully!"
        util_log_info "Monitor job execution in Cloud Console or with:"
        util_log_info "  gcloud run jobs executions list --job=$JOB_NAME --region=$REGION"
    fi
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

# Execute main function
main "$@"