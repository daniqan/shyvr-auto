#!/bin/bash

# Shyvr RLTE Deployment Utilities
# Shared functions and utilities for deployment scripts
# Source this file in other deployment scripts: source "$(dirname "$0")/deploy-utils.sh"

# Configuration defaults
export DEFAULT_PROJECT_ID="shvyr-ai-bots"
export DEFAULT_REGION="us-central1"
export DEFAULT_REPOSITORY="shyvr-ai-prod"

# Colors for consistent output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export PURPLE='\033[0;35m'
export NC='\033[0m'

# Logging functions with consistent formatting
util_log_info() { echo -e "${BLUE}[UTIL]${NC} $1"; }
util_log_success() { echo -e "${GREEN}[UTIL]${NC} $1"; }
util_log_warning() { echo -e "${YELLOW}[UTIL]${NC} $1"; }
util_log_error() { echo -e "${RED}[UTIL]${NC} $1"; }
util_log_header() { echo -e "${PURPLE}[UTIL]${NC} $1"; }

# Environment validation
validate_environment() {
    local env="$1"
    case "$env" in
        staging|production) return 0 ;;
        *) util_log_error "Invalid environment: $env (must be 'staging' or 'production')"; return 1 ;;
    esac
}

# Get service name based on environment
get_service_name() {
    local env="$1"
    case "$env" in
        staging) echo "shyvr-rlte-staging" ;;
        production) echo "shyvr-rlte" ;;
        *) util_log_error "Invalid environment: $env"; return 1 ;;
    esac
}

# Get Cloud SQL instance name based on environment
get_cloud_sql_instance() {
    local env="$1"
    local project_id="${2:-$DEFAULT_PROJECT_ID}"
    case "$env" in
        staging) echo "$project_id:us-central1:shyvr-rlte-db-staging" ;;
        production) echo "$project_id:us-central1:shyvr-rlte-db-prod" ;;
        *) util_log_error "Invalid environment: $env"; return 1 ;;
    esac
}

# Get GCS bucket name for model storage
get_model_bucket() {
    local env="$1"
    case "$env" in
        staging) echo "shyvr-models-staging" ;;
        production) echo "shyvr-models-prod" ;;
        *) util_log_error "Invalid environment: $env"; return 1 ;;
    esac
}

# Check if required tools are installed
check_required_tools() {
    local tools=("gcloud" "docker" "curl" "python3")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        util_log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    return 0
}

# Check GCP authentication
check_gcp_auth() {
    local project_id="${1:-$DEFAULT_PROJECT_ID}"
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
        util_log_error "No active GCP authentication. Run: gcloud auth login"
        return 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$project_id" >/dev/null 2>&1; then
        util_log_error "Cannot access GCP project: $project_id"
        return 1
    fi
    
    return 0
}

# Check if Docker daemon is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        util_log_error "Docker daemon not running"
        return 1
    fi
    return 0
}

# Get current revision of a service
get_current_revision() {
    local service="$1"
    local region="${2:-$DEFAULT_REGION}"
    
    gcloud run services describe "$service" \
        --region="$region" \
        --format="value(status.latestReadyRevisionName)" 2>/dev/null || echo ""
}

# Get service URL
get_service_url() {
    local service="$1"
    local region="${2:-$DEFAULT_REGION}"
    
    gcloud run services describe "$service" \
        --region="$region" \
        --format="value(status.url)" 2>/dev/null || echo ""
}

# Health check with retry logic
health_check_with_retry() {
    local url="$1"
    local max_attempts="${2:-5}"
    local interval="${3:-10}"
    local timeout="${4:-30}"
    
    util_log_info "Performing health check: $url"
    
    for attempt in $(seq 1 "$max_attempts"); do
        if curl -f -s -m "$timeout" "$url" >/dev/null 2>&1; then
            util_log_success "Health check passed (attempt $attempt)"
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            util_log_warning "Health check failed, retrying in ${interval}s (attempt $attempt/$max_attempts)"
            sleep "$interval"
        fi
    done
    
    util_log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Wait for service to be ready
wait_for_service_ready() {
    local service="$1"
    local region="${2:-$DEFAULT_REGION}"
    local max_wait="${3:-300}"
    local check_interval="${4:-10}"
    
    util_log_info "Waiting for service to be ready: $service"
    
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        local status
        status=$(gcloud run services describe "$service" --region="$region" --format="value(status.conditions[0].status)" 2>/dev/null || echo "Unknown")
        
        if [[ "$status" == "True" ]]; then
            util_log_success "Service is ready after ${waited}s"
            return 0
        fi
        
        util_log_info "Service status: $status, waiting... (${waited}/${max_wait}s)"
        sleep "$check_interval"
        waited=$((waited + check_interval))
    done
    
    util_log_error "Service not ready after ${max_wait}s"
    return 1
}

# Check if image exists in Artifact Registry
check_image_exists() {
    local image_url="$1"
    local project_id="${2:-$DEFAULT_PROJECT_ID}"
    
    if gcloud artifacts docker images describe "$image_url" --project="$project_id" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Generate deployment ID
generate_deployment_id() {
    date +"%Y%m%d-%H%M%S"
}

# Get git commit SHA (if available)
get_git_commit_sha() {
    if git rev-parse --verify HEAD >/dev/null 2>&1; then
        git rev-parse --short HEAD
    else
        echo "unknown"
    fi
}

# Create deployment log file
create_deployment_log() {
    local operation="$1"
    local deployment_id="${2:-$(generate_deployment_id)}"
    local log_dir="${LOG_DIR:-/tmp}"
    
    local log_file="$log_dir/${operation}_${deployment_id}.log"
    touch "$log_file"
    echo "$log_file"
}

# Validate secret exists and has value
validate_secret() {
    local secret_name="$1"
    local project_id="${2:-$DEFAULT_PROJECT_ID}"
    
    # Check if secret exists
    if ! gcloud secrets describe "$secret_name" --project="$project_id" >/dev/null 2>&1; then
        util_log_error "Secret does not exist: $secret_name"
        return 1
    fi
    
    # Check if secret has value
    local secret_value
    secret_value=$(gcloud secrets versions access latest --secret="$secret_name" --project="$project_id" 2>/dev/null || echo "")
    
    if [[ -z "$secret_value" || "$secret_value" == "null" ]]; then
        util_log_error "Secret exists but has no value: $secret_name"
        return 1
    fi
    
    return 0
}

# Build environment variables string for different environments
build_env_vars() {
    local environment="$1"
    local deployment_id="${2:-$(generate_deployment_id)}"
    local commit_sha="${3:-$(get_git_commit_sha)}"
    
    local base_vars="ENVIRONMENT=$environment,LOG_LEVEL=INFO,DEPLOYMENT_ID=$deployment_id,COMMIT_SHA=$commit_sha"
    local transformer_vars="TRANSFORMER_OPTIMIZED=true,TRANSFORMER_BATCH_SIZE=1,TRANSFORMER_MAX_LENGTH=512,TORCH_COMPILE_MODE=reduce-overhead,TRANSFORMERS_CACHE=/app/models/cache,TOKENIZERS_PARALLELISM=false,OMP_NUM_THREADS=6,MKL_NUM_THREADS=6,PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,TRANSFORMERS_NO_ADVISORY_WARNINGS=1,TORCH_INFERENCE_MODE=1"
    
    case "$environment" in
        production)
            echo "$base_vars,ML_OPTIMIZED=true,RL_STORAGE_ENABLED=true,MODEL_PRESERVATION_ENABLED=true,TRADING_MODE=simulation,$transformer_vars"
            ;;
        staging)
            echo "$base_vars,ML_OPTIMIZED=false,RL_STORAGE_ENABLED=true,MODEL_PRESERVATION_ENABLED=false,TRADING_MODE=simulation,$transformer_vars,OMP_NUM_THREADS=4,MKL_NUM_THREADS=4"
            ;;
        *)
            echo "$base_vars,$transformer_vars,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2"
            ;;
    esac
}

# Get resource limits based on environment
get_resource_limits() {
    local environment="$1"
    
    case "$environment" in
        production)
            echo "--memory=8Gi --cpu=6 --concurrency=15 --timeout=4200 --max-instances=10 --min-instances=1"
            ;;
        staging)
            echo "--memory=6Gi --cpu=4 --concurrency=20 --timeout=2700 --max-instances=3 --min-instances=0"
            ;;
        *)
            echo "--memory=2Gi --cpu=1 --concurrency=100 --timeout=1800 --max-instances=1 --min-instances=0"
            ;;
    esac
}

# Print deployment summary
print_deployment_summary() {
    local environment="$1"
    local service="$2"
    local region="${3:-$DEFAULT_REGION}"
    local deployment_id="${4:-unknown}"
    
    local service_url
    service_url=$(get_service_url "$service" "$region")
    
    util_log_header "📋 Deployment Summary"
    echo "================================================"
    echo "Environment: $environment"
    echo "Service: $service"
    echo "Region: $region"
    echo "Deployment ID: $deployment_id"
    echo "Service URL: $service_url"
    echo "Health Check: $service_url/health"
    echo "Configuration: $service_url/config"
    echo "Timestamp: $(date)"
    echo "================================================"
}

# Cleanup function for deployment scripts
cleanup_deployment() {
    local exit_code=$?
    local operation="${1:-deployment}"
    
    if [[ $exit_code -ne 0 ]]; then
        util_log_error "$operation failed with exit code $exit_code"
    else
        util_log_success "$operation completed successfully"
    fi
    
    return $exit_code
}

# Export functions for use in other scripts
export -f validate_environment
export -f get_service_name
export -f get_cloud_sql_instance
export -f get_model_bucket
export -f check_required_tools
export -f check_gcp_auth
export -f check_docker
export -f get_current_revision
export -f get_service_url
export -f health_check_with_retry
export -f wait_for_service_ready
export -f check_image_exists
export -f generate_deployment_id
export -f get_git_commit_sha
export -f create_deployment_log
export -f validate_secret
export -f build_env_vars
export -f get_resource_limits
export -f print_deployment_summary
export -f cleanup_deployment

util_log_info "Deployment utilities loaded successfully"