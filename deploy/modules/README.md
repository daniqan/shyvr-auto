# Deploy Modules

This directory contains modular deployment components that can be sourced by various deployment scripts.

## Purpose

The modules directory contains reusable deployment components that enhance the existing deployment pipeline with transformer-specific functionality.

## Available Modules

### transformer_validation.sh

Provides comprehensive validation functions for transformer model deployment.

**Functions:**
- `validate_transformer_models`: Main validation entry point
- `validate_transformer_resources`: Check resource requirements against available resources
- `validate_transformer_health_endpoints`: Test health endpoints
- `test_attention_computation`: Validate attention mechanism computation
- `validate_model_loading_performance`: Check model loading performance
- `check_nan_inf_handling`: Verify NaN/Inf handling capabilities

**Usage:**
```bash
# Source the module
source deploy/modules/transformer_validation.sh

# Run validation
validate_transformer_models "iTransformer" "staging"
echo "Validation result: $TRANSFORMER_VALIDATION_RESULT"
```

**Integration:**
- Automatically sourced by `automated_deployment_pipeline.sh` in Stage 2.5
- Returns proper exit codes for pipeline integration
- Exports `TRANSFORMER_VALIDATION_RESULT` and `TRANSFORMER_VALIDATION_MESSAGE`

### transformer_config_loader.sh

Provides configuration loading and environment variable export for transformer deployments.

**Functions:**
- `load_transformer_configurations`: Main configuration loading entry point
- `parse_transformer_rollout_yaml`: Parse transformer_rollout.yaml
- `parse_transformer_models_json`: Parse transformer_models.json  
- `export_transformer_env_vars`: Export environment variables for deployment
- `set_transformer_resource_limits`: Configure resource limits based on model requirements
- `configure_transformer_health_checks`: Set health check parameters
- `determine_rollout_stage`: Determine rollout stage based on environment
- `configure_for_blue_green_deployment`: Configure for blue-green deployment compatibility
- `configure_for_cloud_run`: Configure for Cloud Run deployment

**Usage:**
```bash
# Source the module
source deploy/modules/transformer_config_loader.sh

# Load configuration for a specific model and environment
load_transformer_configurations "TimesFM" "production"

# Export all configuration
export_all_config
```

**Integration:**
- Automatically sourced by `blue_green_deployment.sh` when `TRANSFORMER_MODEL_TYPE` is set
- Sourced by `deploy.sh` for blue-green deployment mode
- Compatible with existing deployment script parameters
- Requires `yq` and `jq` tools for YAML/JSON parsing

## Configuration Files

The modules work with these configuration files:
- `deploy/configs/transformer_rollout.yaml`: Rollout strategy configuration
- `deploy/configs/transformer_models.json`: Model-specific resource requirements

## Environment Variables

### Input Variables
- `TRANSFORMER_MODEL_TYPE`: Type of transformer model to deploy (e.g., "iTransformer", "TimesFM")
- `TRANSFORMER_ROLLOUT_CONFIG`: Path to rollout configuration file
- `TRANSFORMER_MODELS_CONFIG`: Path to models configuration file
- `AVAILABLE_MEMORY`: Available memory for validation
- `AVAILABLE_CPU`: Available CPU for validation

### Output Variables
- `TRANSFORMER_VALIDATION_RESULT`: "PASS" or "FAIL"
- `TRANSFORMER_VALIDATION_MESSAGE`: Validation status message
- `MODEL_TYPE`: Configured model type
- `TRANSFORMER_MEMORY`: Required memory
- `TRANSFORMER_CPU`: Required CPU
- `CLOUD_RUN_MEMORY`, `CLOUD_RUN_CPU`, `CLOUD_RUN_TIMEOUT`: Cloud Run settings
- `HEALTH_CHECK_ATTEMPTS`, `HEALTH_CHECK_INTERVAL`: Health check parameters

## Integration Points

### automated_deployment_pipeline.sh
- Stage 2.5: Sources `transformer_validation.sh` for comprehensive validation
- Minimal changes: Only 5-10 lines added to existing pipeline

### blue_green_deployment.sh
- Sources `transformer_config_loader.sh` when `TRANSFORMER_MODEL_TYPE` is set
- Automatically configures resource limits based on model requirements
- Overrides health check parameters for transformer-specific needs

### deploy.sh
- Sources `transformer_config_loader.sh` for blue-green deployment mode
- Provides configuration compatibility across deployment methods

## Error Handling

Both modules provide robust error handling with specific exit codes:
- 0: Success
- 1: General failure
- 2: Resource/parsing failure
- 3: Health endpoint failure
- 4-6: Specific validation failures

## Testing

Comprehensive test suites are available:
- `tests/unit/deploy/test_transformer_validation_module.py`
- `tests/unit/deploy/test_transformer_config_loader_module.py`

Run tests with:
```bash
uv run pytest tests/unit/deploy/test_transformer_*_module.py -v
```