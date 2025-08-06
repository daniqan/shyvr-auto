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
ENVIRONMENTS_CONFIG="${ENVIRONMENTS_CONFIG:-deploy/configs/environments.json}"
TRANSFORMER_ROLLOUT_CONFIG="${TRANSFORMER_ROLLOUT_CONFIG:-deploy/configs/transformer_rollout.yaml}"

# Configuration results
ENVIRONMENT="${ENVIRONMENT:-production}"
ROLLOUT_STAGE=""
TRAFFIC_PERCENTAGE=""
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

# Load environment-based configuration from environments.json
load_environment_config() {
    local environment="${1:-${ENVIRONMENT}}"
    local config_file="${2:-$ENVIRONMENTS_CONFIG}"
    
    log_config_info "Loading environment configuration for: $environment"
    
    if [[ ! -f "$config_file" ]]; then
        log_config_error "Environment config file not found: $config_file"
        return 1
    fi
    
    # Validate JSON syntax
    if ! jq . "$config_file" >/dev/null 2>&1; then
        log_config_error "Invalid JSON in environment config file: $config_file"
        return 1
    fi
    
    # Check if environment exists in config
    if ! jq -e ".[\"$environment\"]" "$config_file" >/dev/null 2>&1; then
        log_config_error "Environment '$environment' not found in configuration"
        return 1
    fi
    
    # Parse environment-specific configuration
    MODEL_MEMORY=$(jq -r ".[\"$environment\"].resources.memory" "$config_file")
    MODEL_CPU=$(jq -r ".[\"$environment\"].resources.cpu" "$config_file")
    HEALTH_ENDPOINT=$(jq -r ".[\"$environment\"].health_checks.endpoint" "$config_file")
    local timeout_seconds=$(jq -r ".[\"$environment\"].resources.timeout_seconds" "$config_file")
    
    # Export environment-specific variables
    export ENVIRONMENT="$environment"
    export MODEL_MEMORY
    export MODEL_CPU
    export HEALTH_ENDPOINT
    export TRANSFORMER_TIMEOUT="$timeout_seconds"
    
    log_config_success "Environment configuration loaded: $environment"
    log_config_info "Memory: $MODEL_MEMORY, CPU: $MODEL_CPU, Health: $HEALTH_ENDPOINT"
    
    return 0
}

# Check if required tools are available
check_config_dependencies() {
    local missing_tools=()
    
    # Check for jq (JSON parser)
    if ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi
    
    # Check for yq (YAML parser) - still needed for rollout config
    if ! command -v yq >/dev/null 2>&1; then
        missing_tools+=("yq")
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

# Set resource limits based on environment configuration
set_environment_resource_limits() {
    local environment="${1:-${ENVIRONMENT}}"
    local memory="${2:-$MODEL_MEMORY}"
    local cpu="${3:-$MODEL_CPU}"
    
    log_config_info "Setting resource limits for $environment environment"
    
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
    
    # Set Cloud Run specific variables based on environment
    export CLOUD_RUN_MEMORY="$memory"
    export CLOUD_RUN_CPU="$cpu"
    export CLOUD_RUN_TIMEOUT="${TRANSFORMER_TIMEOUT:-4200}"
    
    # Environment-specific scaling
    case "$environment" in
        "development")
            export CLOUD_RUN_MIN_INSTANCES="0"
            export CLOUD_RUN_MAX_INSTANCES="1"
            export CLOUD_RUN_CONCURRENCY="3"
            ;;
        "staging")
            export CLOUD_RUN_MIN_INSTANCES="0"
            export CLOUD_RUN_MAX_INSTANCES="3"
            export CLOUD_RUN_CONCURRENCY="6"
            ;;
        "production")
            export CLOUD_RUN_MIN_INSTANCES="1"
            export CLOUD_RUN_MAX_INSTANCES="10"
            export CLOUD_RUN_CONCURRENCY="15"
            ;;
    esac
    
    log_config_success "Resource limits configured: Memory=$memory, CPU=$cpu"
    log_config_info "Environment: $environment, Min/Max instances: ${CLOUD_RUN_MIN_INSTANCES}/${CLOUD_RUN_MAX_INSTANCES}"
    
    return 0
}

# Export environment variables for deployment scripts
export_environment_vars() {
    local environment="${1:-${ENVIRONMENT}}"
    local memory="${2:-$MODEL_MEMORY}"
    local cpu="${3:-$MODEL_CPU}"
    local health_endpoint="${4:-$HEALTH_ENDPOINT}"
    
    log_config_info "Exporting environment variables for $environment"
    
    # Set core environment variables
    export ENVIRONMENT="$environment"
    export TRANSFORMER_MEMORY="$memory"
    export TRANSFORMER_CPU="$cpu"
    export TRANSFORMER_HEALTH_ENDPOINT="$health_endpoint"
    export TRANSFORMER_TIMEOUT="${TRANSFORMER_TIMEOUT:-4200}"
    
    # Set deployment-specific variables
    export TRANSFORMER_OPTIMIZED="true"
    export ROLLOUT_MODE="progressive"
    export MODEL_VALIDATION_ENABLED="true"
    
    # Environment-specific model configuration
    case "$environment" in
        "development")
            export MODEL_TYPE="lstm"
            export MOCK_TRANSFORMERS="true"
            export DEVELOPMENT_MODE="true"
            ;;
        "staging"|"production")
            export MODEL_TYPE="ensemble"
            export MOCK_TRANSFORMERS="false"
            export ENSEMBLE_MODELS="lstm,iTransformer,PatchTST,TimesMixer,TimesFM"
            if [[ "$environment" == "production" ]]; then
                export PRODUCTION_MODE="true"
                export ENSEMBLE_OPTIMIZATION="true"
            else
                export STAGING_MODE="true"
            fi
            ;;
    esac
    
    log_config_success "Environment variables exported successfully for $environment"
    
    return 0
}

# Configure health check parameters for environment
configure_environment_health_checks() {
    local environment="${1:-${ENVIRONMENT}}"
    local health_endpoint="${2:-$HEALTH_ENDPOINT}"
    
    log_config_info "Configuring health check parameters for $environment"
    
    # Set health check variables based on environment
    case "$environment" in
        "development")
            export HEALTH_CHECK_TIMEOUT="30"
            export HEALTH_CHECK_INTERVAL="15"
            export HEALTH_CHECK_MAX_FAILURES="3"
            ;;
        "staging")
            export HEALTH_CHECK_TIMEOUT="40"
            export HEALTH_CHECK_INTERVAL="20"
            export HEALTH_CHECK_MAX_FAILURES="3"
            ;;
        "production")
            export HEALTH_CHECK_TIMEOUT="45"
            export HEALTH_CHECK_INTERVAL="20"
            export HEALTH_CHECK_MAX_FAILURES="3"
            ;;
    esac
    
    export HEALTH_CHECK_ENDPOINT="$health_endpoint"
    
    log_config_success "Health checks configured: Endpoint=$health_endpoint, Timeout=${HEALTH_CHECK_TIMEOUT}s"
    
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
load_environment_configurations() {
    local environment="${1:-${ENVIRONMENT:-production}}"
    
    log_config_header "🔧 Loading ensemble configurations for $environment environment"
    
    # Check dependencies
    if ! check_config_dependencies; then
        log_config_error "Configuration dependencies check failed"
        return 1
    fi
    
    # Load environment-specific configuration
    if ! load_environment_config "$environment"; then
        log_config_error "Failed to load environment configuration"
        return 1
    fi
    
    # Load rollout configuration
    if ! parse_transformer_rollout_yaml; then
        log_config_error "Failed to parse rollout configuration"
        return 1
    fi
    
    # Set resource limits based on environment
    if ! set_environment_resource_limits "$environment"; then
        log_config_error "Failed to set resource limits"
        return 1
    fi
    
    # Export environment variables
    if ! export_environment_vars "$environment"; then
        log_config_error "Failed to export environment variables"
        return 1
    fi
    
    # Configure health checks
    if ! configure_environment_health_checks "$environment"; then
        log_config_error "Failed to configure health checks"
        return 1
    fi
    
    # Determine rollout stage
    if ! determine_rollout_stage "$environment"; then
        log_config_error "Failed to determine rollout stage"
        return 1
    fi
    
    log_config_success "✅ Ensemble configurations loaded successfully for $environment"
    
    return 0
}

# Backwards compatibility function - deprecated
load_transformer_configurations() {
    local model_type="${1:-}"
    local environment="${2:-staging}"
    
    log_config_warning "load_transformer_configurations is deprecated - use load_environment_configurations"
    
    # Map old model type to environment if needed
    if [[ -n "$model_type" && -z "${ENVIRONMENT:-}" ]]; then
        case "$model_type" in
            "lstm")
                environment="development"
                ;;
            *)
                environment="production"
                ;;
        esac
    fi
    
    load_environment_configurations "$environment"
}

# Export all configuration for deployment script integration
export_all_config() {
    # Export all key variables that deployment scripts expect
    export ENVIRONMENTS_CONFIG
    export TRANSFORMER_ROLLOUT_CONFIG
    export ENVIRONMENT
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
    export CLOUD_RUN_MEMORY
    export CLOUD_RUN_CPU
    export CLOUD_RUN_TIMEOUT
    export CLOUD_RUN_MIN_INSTANCES
    export CLOUD_RUN_MAX_INSTANCES
    export CLOUD_RUN_CONCURRENCY
}

# Main execution guard - only run if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Direct execution for testing
    ENVIRONMENT="${1:-${ENVIRONMENT:-production}}"
    
    log_config_info "Direct execution mode - testing ensemble configuration loading"
    load_environment_configurations "$ENVIRONMENT"
    export_all_config
    
    # Show loaded configuration
    log_config_info "Loaded configuration:"
    log_config_info "  Environment: $ENVIRONMENT"
    log_config_info "  Model Type: $MODEL_TYPE"
    log_config_info "  Memory: $MODEL_MEMORY"
    log_config_info "  CPU: $MODEL_CPU"
    log_config_info "  Health: $HEALTH_ENDPOINT"
    log_config_info "  Stage: $ROLLOUT_STAGE"
    log_config_info "  Traffic: $TRAFFIC_PERCENTAGE%"
fi