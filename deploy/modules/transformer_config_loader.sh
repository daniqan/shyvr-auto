#!/bin/bash

# Transformer Configuration Loader Module for Deployment Pipeline
# Provides configuration loading functions for transformer model deployment
# Designed to be sourced by deploy.sh and blue_green_deployment.sh
# 
# Functions provided:
# - load_transformer_configurations: Main configuration loading entry point
# - parse_transformer_rollout_yaml: Parse transformer_rollout.yaml
# - parse_transformer_models_json: Parse transformer_models.json
# - export_transformer_env_vars: Export environment variables
# - set_transformer_resource_limits: Set resource limits
# - configure_transformer_health_checks: Configure health check parameters
# - determine_rollout_stage: Determine rollout stage logic
#
# Exit codes:
# 0 - Success
# 1 - General configuration failure
# 2 - YAML parsing failure
# 3 - JSON parsing failure
# 4 - Environment variable export failure
# 5 - Resource limit configuration failure
# 6 - Health check configuration failure

# Ensure script fails on any error
set -euo pipefail

# Global configuration variables
TRANSFORMER_ROLLOUT_CONFIG="${TRANSFORMER_ROLLOUT_CONFIG:-deploy/configs/transformer_rollout.yaml}"
TRANSFORMER_MODELS_CONFIG="${TRANSFORMER_MODELS_CONFIG:-deploy/configs/transformer_models.json}"

# Configuration results
ROLLOUT_STAGE=""
TRAFFIC_PERCENTAGE=""
MODEL_TYPE=""
MODEL_MEMORY=""
MODEL_CPU=""
HEALTH_ENDPOINT=""
HEALTH_CHECK_ATTEMPTS=""
HEALTH_CHECK_INTERVAL=""
TRAFFIC_MIGRATION_DELAY=""

# Colors for consistent logging with deployment scripts
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions consistent with deployment pipeline
log_config_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_config_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_config_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_config_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_config_header() { echo -e "${PURPLE}[CONFIG-LOADER]${NC} $1"; }

# Check if required tools are available
check_config_dependencies() {
    local missing_tools=()
    
    # Check for yq (YAML parser)
    if ! command -v yq >/dev/null 2>&1; then
        missing_tools+=("yq")
    fi
    
    # Check for jq (JSON parser)
    if ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_config_error "Missing required tools: ${missing_tools[*]}"
        log_config_info "Install missing tools: apt-get install -y yq jq (Ubuntu) or brew install yq jq (macOS)"
        return 1
    fi
    
    return 0
}

# Parse transformer rollout YAML configuration
parse_transformer_rollout_yaml() {
    local config_file="${1:-$TRANSFORMER_ROLLOUT_CONFIG}"
    
    log_config_info "Parsing transformer rollout configuration: $config_file"
    
    if [[ ! -f "$config_file" ]]; then
        log_config_error "Rollout config file not found: $config_file"
        return 2
    fi
    
    # Validate YAML syntax
    if ! yq eval . "$config_file" >/dev/null 2>&1; then
        log_config_error "Invalid YAML in rollout config file: $config_file"
        return 2
    fi
    
    # Parse key configuration values
    HEALTH_CHECK_ATTEMPTS=$(yq eval '.validation.health_check_attempts // 15' "$config_file")
    HEALTH_CHECK_INTERVAL=$(yq eval '.validation.health_check_interval // 20' "$config_file")
    TRAFFIC_MIGRATION_DELAY=$(yq eval '.canary_deployment.traffic_ramp_delay_seconds // 180' "$config_file")
    
    # Export for deployment script compatibility
    export HEALTH_CHECK_ATTEMPTS
    export HEALTH_CHECK_INTERVAL
    export TRAFFIC_MIGRATION_DELAY
    
    log_config_success "Rollout configuration parsed successfully"
    log_config_info "Health check attempts: $HEALTH_CHECK_ATTEMPTS"
    log_config_info "Health check interval: $HEALTH_CHECK_INTERVAL"
    log_config_info "Traffic migration delay: $TRAFFIC_MIGRATION_DELAY"
    
    return 0
}

# Parse transformer models JSON configuration
parse_transformer_models_json() {
    local model_type="${1:-}"
    local config_file="${2:-$TRANSFORMER_MODELS_CONFIG}"
    
    if [[ -z "$model_type" ]]; then
        log_config_error "Model type is required for JSON parsing"
        return 3
    fi
    
    log_config_info "Parsing transformer models configuration for $model_type: $config_file"
    
    if [[ ! -f "$config_file" ]]; then
        log_config_error "Models config file not found: $config_file"
        return 3
    fi
    
    # Validate JSON syntax
    if ! jq . "$config_file" >/dev/null 2>&1; then
        log_config_error "Invalid JSON in models config file: $config_file"
        return 3
    fi
    
    # Check if model type exists in config
    if ! jq -e ".[\"$model_type\"]" "$config_file" >/dev/null 2>&1; then
        log_config_error "Model type '$model_type' not found in configuration"
        return 3
    fi
    
    # Parse model-specific configuration
    MODEL_MEMORY=$(jq -r ".[\"$model_type\"].resources.memory" "$config_file")
    MODEL_CPU=$(jq -r ".[\"$model_type\"].resources.cpu" "$config_file")
    HEALTH_ENDPOINT=$(jq -r ".[\"$model_type\"].health_checks.endpoint" "$config_file")
    local health_timeout=$(jq -r ".[\"$model_type\"].health_checks.timeout_seconds" "$config_file")
    local health_interval=$(jq -r ".[\"$model_type\"].health_checks.interval_seconds" "$config_file")
    local max_failures=$(jq -r ".[\"$model_type\"].health_checks.max_failures" "$config_file")
    
    # Export for deployment script compatibility
    export MODEL_MEMORY
    export MODEL_CPU
    export HEALTH_ENDPOINT
    
    log_config_success "Models configuration parsed successfully for $model_type"
    log_config_info "Model memory: $MODEL_MEMORY"
    log_config_info "Model CPU: $MODEL_CPU"
    log_config_info "Health endpoint: $HEALTH_ENDPOINT"
    
    return 0
}

# Export transformer environment variables for deployment scripts
export_transformer_env_vars() {
    local model_type="${1:-}"
    local memory="${2:-}"
    local cpu="${3:-}"
    local health_endpoint="${4:-}"
    local health_timeout="${5:-30}"
    
    if [[ -z "$model_type" ]]; then
        log_config_error "Model type is required for environment variable export"
        return 4
    fi
    
    log_config_info "Exporting transformer environment variables for $model_type"
    
    # Set core transformer variables
    export MODEL_TYPE="$model_type"
    export TRANSFORMER_MEMORY="${memory:-4Gi}"
    export TRANSFORMER_CPU="${cpu:-2}"
    export TRANSFORMER_HEALTH_ENDPOINT="${health_endpoint:-/health}"
    export TRANSFORMER_HEALTH_TIMEOUT="$health_timeout"
    
    # Set deployment-specific variables
    export TRANSFORMER_OPTIMIZED="true"
    export ROLLOUT_MODE="progressive"
    export MODEL_VALIDATION_ENABLED="true"
    
    log_config_success "Environment variables exported successfully"
    
    return 0
}

# Set transformer resource limits based on model requirements
set_transformer_resource_limits() {
    local model_type="${1:-}"
    local memory="${2:-4Gi}"
    local cpu="${3:-2}"
    local max_instances="${4:-10}"
    
    log_config_info "Setting resource limits for $model_type"
    
    # Validate memory format
    if [[ ! "$memory" =~ ^[0-9]+Gi$ ]]; then
        log_config_error "Invalid memory format: $memory (expected format: [number]Gi)"
        return 5
    fi
    
    # Validate CPU format
    if [[ ! "$cpu" =~ ^[0-9]+$ ]]; then
        log_config_error "Invalid CPU format: $cpu (expected format: [number])"
        return 5
    fi
    
    # Set resource limit variables
    export MEMORY_LIMIT="$memory"
    export CPU_LIMIT="$cpu"
    export MAX_INSTANCES="$max_instances"
    
    # Set Cloud Run specific variables
    export CLOUD_RUN_MEMORY="$memory"
    export CLOUD_RUN_CPU="$cpu"
    export CLOUD_RUN_TIMEOUT="4200"  # 70 minutes for transformer loading
    
    log_config_success "Resource limits configured: Memory=$memory, CPU=$cpu, MaxInstances=$max_instances"
    
    return 0
}

# Configure transformer health check parameters
configure_transformer_health_checks() {
    local health_endpoint="${1:-/health}"
    local timeout="${2:-30}"
    local interval="${3:-15}"
    local max_failures="${4:-3}"
    
    log_config_info "Configuring health check parameters"
    
    # Set health check variables
    export HEALTH_CHECK_ENDPOINT="$health_endpoint"
    export HEALTH_CHECK_TIMEOUT="$timeout"
    export HEALTH_CHECK_INTERVAL="$interval"
    export HEALTH_CHECK_MAX_FAILURES="$max_failures"
    
    log_config_success "Health checks configured: Endpoint=$health_endpoint, Timeout=${timeout}s, Interval=${interval}s"
    
    return 0
}

# Determine rollout stage based on environment and traffic
determine_rollout_stage() {
    local environment="${1:-staging}"
    local target_traffic="${2:-100}"
    
    log_config_info "Determining rollout stage for $environment with $target_traffic% traffic"
    
    case "$environment" in
        "staging")
            # Staging uses simplified rollout
            if [[ $target_traffic -le 50 ]]; then
                ROLLOUT_STAGE="canary"
                TRAFFIC_PERCENTAGE="50"
            else
                ROLLOUT_STAGE="full"
                TRAFFIC_PERCENTAGE="100"
            fi
            ;;
        "production")
            # Production uses full progressive rollout
            if [[ $target_traffic -le 10 ]]; then
                ROLLOUT_STAGE="canary"
                TRAFFIC_PERCENTAGE="10"
            elif [[ $target_traffic -le 25 ]]; then
                ROLLOUT_STAGE="limited"
                TRAFFIC_PERCENTAGE="25"
            elif [[ $target_traffic -le 50 ]]; then
                ROLLOUT_STAGE="expanded"
                TRAFFIC_PERCENTAGE="50"
            else
                ROLLOUT_STAGE="full"
                TRAFFIC_PERCENTAGE="100"
            fi
            ;;
        *)
            log_config_warning "Unknown environment: $environment, defaulting to staging behavior"
            ROLLOUT_STAGE="full"
            TRAFFIC_PERCENTAGE="100"
            ;;
    esac
    
    # Export for deployment script compatibility
    export ROLLOUT_STAGE
    export TRAFFIC_PERCENTAGE
    
    log_config_success "Rollout stage determined: $ROLLOUT_STAGE ($TRAFFIC_PERCENTAGE%)"
    
    return 0
}

# Configure for blue-green deployment compatibility
configure_for_blue_green_deployment() {
    local model_type="${1:-}"
    local environment="${2:-staging}"
    
    log_config_info "Configuring for blue-green deployment: $model_type in $environment"
    
    # Load configurations
    if ! parse_transformer_rollout_yaml; then
        return 1
    fi
    
    if ! parse_transformer_models_json "$model_type"; then
        return 1
    fi
    
    # Set blue-green specific variables
    export BLUE_GREEN_ENABLED="true"
    export GRADUAL_ROLLOUT="true"
    
    # Override deployment script parameters based on environment
    case "$environment" in
        "production")
            export INITIAL_TRAFFIC_PERCENT="10"
            export INTERMEDIATE_TRAFFIC_PERCENT="50"
            export HEALTH_CHECK_ATTEMPTS="15"
            export HEALTH_CHECK_INTERVAL="20"
            export TRAFFIC_MIGRATION_DELAY="180"
            ;;
        "staging")
            export INITIAL_TRAFFIC_PERCENT="50"
            export INTERMEDIATE_TRAFFIC_PERCENT="100"
            export HEALTH_CHECK_ATTEMPTS="10"
            export HEALTH_CHECK_INTERVAL="15"
            export TRAFFIC_MIGRATION_DELAY="120"
            ;;
    esac
    
    log_config_success "Blue-green deployment configuration completed"
    
    return 0
}

# Configure for Cloud Run deployment
configure_for_cloud_run() {
    local model_type="${1:-}"
    local environment="${2:-staging}"
    
    log_config_info "Configuring for Cloud Run deployment: $model_type in $environment"
    
    # Load model configuration
    if ! parse_transformer_models_json "$model_type"; then
        return 1
    fi
    
    # Set Cloud Run specific timeouts and scaling
    case "$model_type" in
        "TimesFM")
            export CLOUD_RUN_MEMORY="6Gi"
            export CLOUD_RUN_CPU="4"
            export CLOUD_RUN_TIMEOUT="4800"  # 80 minutes for large models
            ;;
        "TimesMixer")
            export CLOUD_RUN_MEMORY="5Gi"
            export CLOUD_RUN_CPU="3"
            export CLOUD_RUN_TIMEOUT="4200"  # 70 minutes
            ;;
        "iTransformer"|"PatchTST")
            export CLOUD_RUN_MEMORY="$MODEL_MEMORY"
            export CLOUD_RUN_CPU="$MODEL_CPU"
            export CLOUD_RUN_TIMEOUT="3600"  # 60 minutes
            ;;
        *)
            # Default values
            export CLOUD_RUN_MEMORY="4Gi"
            export CLOUD_RUN_CPU="2"
            export CLOUD_RUN_TIMEOUT="3600"
            ;;
    esac
    
    # Environment-specific scaling
    if [[ "$environment" == "production" ]]; then
        export CLOUD_RUN_MIN_INSTANCES="1"
        export CLOUD_RUN_MAX_INSTANCES="10"
        export CLOUD_RUN_CONCURRENCY="15"
    else
        export CLOUD_RUN_MIN_INSTANCES="0"
        export CLOUD_RUN_MAX_INSTANCES="3"
        export CLOUD_RUN_CONCURRENCY="10"
    fi
    
    log_config_success "Cloud Run configuration completed"
    
    return 0
}

# Main configuration loading function - entry point for deployment scripts
load_transformer_configurations() {
    local model_type="${1:-}"
    local environment="${2:-staging}"
    
    if [[ -z "$model_type" ]]; then
        log_config_error "Model type is required for configuration loading"
        return 1
    fi
    
    log_config_header "🔧 Loading transformer configurations for $model_type in $environment"
    
    # Check dependencies
    if ! check_config_dependencies; then
        log_config_error "Configuration dependencies check failed"
        return 1
    fi
    
    # Load rollout configuration
    if ! parse_transformer_rollout_yaml; then
        log_config_error "Failed to parse rollout configuration"
        return 1
    fi
    
    # Load model-specific configuration
    if ! parse_transformer_models_json "$model_type"; then
        log_config_error "Failed to parse model configuration"
        return 1
    fi
    
    # Export environment variables
    if ! export_transformer_env_vars "$model_type" "$MODEL_MEMORY" "$MODEL_CPU" "$HEALTH_ENDPOINT"; then
        log_config_error "Failed to export environment variables"
        return 1
    fi
    
    # Set resource limits
    if ! set_transformer_resource_limits "$model_type" "$MODEL_MEMORY" "$MODEL_CPU"; then
        log_config_error "Failed to set resource limits"
        return 1
    fi
    
    # Configure health checks
    if ! configure_transformer_health_checks "$HEALTH_ENDPOINT"; then
        log_config_error "Failed to configure health checks"
        return 1
    fi
    
    # Determine rollout stage
    if ! determine_rollout_stage "$environment"; then
        log_config_error "Failed to determine rollout stage"
        return 1
    fi
    
    log_config_success "✅ Transformer configurations loaded successfully for $model_type"
    
    return 0
}

# Export all configuration for deployment script integration
export_all_config() {
    # Export all key variables that deployment scripts expect
    export TRANSFORMER_ROLLOUT_CONFIG
    export TRANSFORMER_MODELS_CONFIG
    export MODEL_TYPE
    export MODEL_MEMORY
    export MODEL_CPU
    export HEALTH_ENDPOINT
    export ROLLOUT_STAGE
    export TRAFFIC_PERCENTAGE
    export HEALTH_CHECK_ATTEMPTS
    export HEALTH_CHECK_INTERVAL
    export TRAFFIC_MIGRATION_DELAY
    export TRANSFORMER_OPTIMIZED
    export ROLLOUT_MODE
    export MODEL_VALIDATION_ENABLED
}

# Main execution guard - only run if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Direct execution for testing
    MODEL_TYPE="${1:-iTransformer}"
    ENVIRONMENT="${2:-staging}"
    
    log_config_info "Direct execution mode - testing configuration loading"
    load_transformer_configurations "$MODEL_TYPE" "$ENVIRONMENT"
    export_all_config
    
    # Show loaded configuration
    log_config_info "Loaded configuration:"
    log_config_info "  Model: $MODEL_TYPE"
    log_config_info "  Memory: $MODEL_MEMORY"
    log_config_info "  CPU: $MODEL_CPU"
    log_config_info "  Health: $HEALTH_ENDPOINT"
    log_config_info "  Stage: $ROLLOUT_STAGE"
    log_config_info "  Traffic: $TRAFFIC_PERCENTAGE%"
fi