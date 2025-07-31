# Model Preservation System - User Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Overview

The Model Preservation System provides comprehensive version control, backup, and management capabilities for machine learning models in the Shyvr RLTE system. It offers automatic model versioning, cloud storage integration, caching for performance, and rollback capabilities.

### Key Features

- **Automated Versioning**: Semantic versioning (1.0.0, 1.1.0, 2.0.0) with automatic version detection
- **Cloud Storage**: Google Cloud Storage integration for distributed model storage
- **Model Tagging**: Tag models with custom labels for easy identification
- **Model Branching**: Create and manage model branches for different experiments
- **Performance Monitoring**: Track model performance metrics over time
- **Rollback Capability**: Quickly revert to previous model versions
- **Caching System**: Multi-level caching for fast model loading
- **REST API**: Complete API for programmatic access

## Quick Start

### Prerequisites

1. Python 3.11+ with uv package manager
2. Google Cloud Storage bucket configured
3. PostgreSQL database for metadata
4. Required environment variables set

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd shyvrai-rlte

# Install dependencies
uv sync

# Set up environment variables
cp env.template .env
# Edit .env with your configuration
```

### Basic Setup

```python
from src.model_preservation.manager import ModelPreservationManager

# Initialize the manager
manager = ModelPreservationManager()

# Save a model
model_data = {"weights": [...], "config": {...}}
version_info = manager.save_model(
    model_data=model_data,
    model_type="dqn_agent",
    description="Initial model training"
)
print(f"Model saved as version {version_info['version']}")
```

## Basic Usage

### Saving Models

```python
# Save with automatic versioning
version_info = manager.save_model(
    model_data=model_data,
    model_type="lstm_predictor",
    description="Improved accuracy model",
    metadata={
        "accuracy": 0.95,
        "training_epochs": 100,
        "dataset_size": 10000
    }
)

# Save with custom tag
version_info = manager.save_model(
    model_data=model_data,
    model_type="dqn_agent",
    description="Production candidate",
    tags=["production", "stable", "v2.1"]
)
```

### Loading Models

```python
# Load latest version
model_data = manager.load_model(
    model_type="dqn_agent"
)

# Load specific version
model_data = manager.load_model(
    model_type="dqn_agent",
    version="1.2.3"
)

# Load by tag
model_data = manager.load_model(
    model_type="dqn_agent",
    tag="production"
)
```

### Model History and Rollback

```python
# Get model history
history = manager.get_model_history(
    model_type="dqn_agent",
    limit=10
)

for version in history:
    print(f"Version {version['version']}: {version['description']}")

# Rollback to previous version
success = manager.rollback_model(
    model_type="dqn_agent",
    target_version="1.1.0"
)
```

## Advanced Features

### Model Tagging

Tags provide flexible labeling for models:

```python
# Add tags to existing model
manager.tag_model(
    model_type="dqn_agent",
    version="1.2.0",
    tags=["stable", "production-ready"]
)

# Remove tags
manager.remove_tags(
    model_type="dqn_agent",
    version="1.2.0",
    tags=["experimental"]
)

# Find models by tag
models = manager.find_models_by_tag("production")
```

### Model Branching

Create separate development branches:

```python
# Create a new branch
branch_info = manager.create_branch(
    source_model_type="dqn_agent",
    source_version="1.0.0",
    branch_name="experimental-rewards",
    description="Testing new reward function"
)

# Save to branch
version_info = manager.save_model(
    model_data=model_data,
    model_type="dqn_agent",
    branch="experimental-rewards",
    description="Updated reward weights"
)

# Merge branch
merge_result = manager.merge_branch(
    source_branch="experimental-rewards",
    target_branch="main",
    model_type="dqn_agent"
)
```

### Performance Monitoring

Track model performance over time:

```python
# Record performance metrics
manager.record_performance(
    model_type="dqn_agent",
    version="1.2.0",
    metrics={
        "accuracy": 0.94,
        "precision": 0.92,
        "recall": 0.96,
        "f1_score": 0.94,
        "training_time": 3600,
        "inference_time": 0.05
    }
)

# Get performance history
performance_data = manager.get_performance_history(
    model_type="dqn_agent",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### Caching Configuration

The system uses multi-level caching:

```python
# Configure cache settings
cache_config = {
    "memory_cache_size": 100,  # MB
    "disk_cache_size": 1000,   # MB
    "cache_ttl": 3600,         # seconds
    "enable_compression": True
}

manager = ModelPreservationManager(cache_config=cache_config)

# Clear cache when needed
manager.clear_cache(model_type="dqn_agent")
```

## Configuration

### Environment Variables

```bash
# Google Cloud Storage
GOOGLE_CLOUD_PROJECT=your-project-id
MODEL_PRESERVATION_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=shyvr_rlte
POSTGRES_USER=your-username
POSTGRES_PASSWORD=your-password

# Model Preservation Settings
MODEL_PRESERVATION_ENABLED=true
MODEL_CACHE_SIZE_MB=500
MODEL_COMPRESSION_ENABLED=true
MODEL_VERSIONING_STRATEGY=semantic
```

### Configuration File

Create `config/model_preservation.yaml`:

```yaml
storage:
  provider: gcs
  bucket: your-bucket-name
  prefix: models/
  compression: true

database:
  track_metadata: true
  enable_performance_tracking: true
  cleanup_old_versions: true
  retention_days: 90

caching:
  memory_cache_mb: 500
  disk_cache_mb: 2000
  cache_ttl_seconds: 3600
  
versioning:
  strategy: semantic
  auto_increment: true
  branch_support: true

monitoring:
  track_usage: true
  performance_metrics: true
  alert_on_failures: true
```

## Best Practices

### Model Organization

1. **Use Descriptive Model Types**: Use clear, consistent naming
   ```python
   # Good
   model_type = "dqn_trading_agent_v2"
   
   # Avoid
   model_type = "model1"
   ```

2. **Meaningful Descriptions**: Always include informative descriptions
   ```python
   description = "DQN agent with improved reward function and 95% accuracy"
   ```

3. **Consistent Metadata**: Use standardized metadata fields
   ```python
   metadata = {
       "accuracy": 0.95,
       "training_epochs": 100,
       "dataset_version": "2024-01-15",
       "hyperparameters": {...}
   }
   ```

### Version Management

1. **Use Semantic Versioning**: Follow semantic versioning principles
   - Major version: Breaking changes
   - Minor version: New features, backward compatible
   - Patch version: Bug fixes

2. **Tag Important Versions**: Tag significant models
   ```python
   tags = ["production", "milestone", "stable"]
   ```

3. **Regular Cleanup**: Remove old, unused versions
   ```python
   manager.cleanup_old_versions(
       model_type="dqn_agent",
       keep_versions=10
   )
   ```

### Performance Optimization

1. **Use Caching Wisely**: Configure appropriate cache sizes
2. **Compress Large Models**: Enable compression for storage efficiency
3. **Monitor Storage Usage**: Regularly check storage consumption
4. **Batch Operations**: Use bulk operations when possible

### Security and Backup

1. **Regular Backups**: Implement backup strategies
2. **Access Control**: Use proper IAM for cloud storage
3. **Encryption**: Enable encryption for sensitive models
4. **Audit Logging**: Monitor model access and changes

### Development Workflow

1. **Use Branches**: Create branches for experimental work
2. **Test Before Production**: Validate models before production deployment
3. **Document Changes**: Maintain detailed change logs
4. **Performance Monitoring**: Continuously monitor model performance

## Examples

### Complete Workflow Example

```python
import asyncio
from src.model_preservation.manager import ModelPreservationManager

async def complete_workflow():
    manager = ModelPreservationManager()
    
    # 1. Save initial model
    initial_model = {"weights": [...], "config": {...}}
    version_info = await manager.save_model(
        model_data=initial_model,
        model_type="trading_dqn",
        description="Initial trading model",
        metadata={"accuracy": 0.87}
    )
    print(f"Saved version: {version_info['version']}")
    
    # 2. Create experimental branch
    branch = await manager.create_branch(
        source_model_type="trading_dqn",
        source_version=version_info['version'],
        branch_name="improved-rewards",
        description="Testing new reward structure"
    )
    
    # 3. Save improved model to branch
    improved_model = {"weights": [...], "config": {...}}
    improved_version = await manager.save_model(
        model_data=improved_model,
        model_type="trading_dqn",
        branch="improved-rewards",
        description="Enhanced reward function",
        metadata={"accuracy": 0.93}
    )
    
    # 4. Test and validate
    test_model = await manager.load_model(
        model_type="trading_dqn",
        branch="improved-rewards"
    )
    
    # 5. Merge to main if successful
    if validate_model(test_model):
        merge_result = await manager.merge_branch(
            source_branch="improved-rewards",
            target_branch="main",
            model_type="trading_dqn"
        )
        
        # 6. Tag as production ready
        await manager.tag_model(
            model_type="trading_dqn",
            version=merge_result['version'],
            tags=["production", "tested", "v2.0"]
        )

if __name__ == "__main__":
    asyncio.run(complete_workflow())
```

This user guide provides a comprehensive introduction to the Model Preservation System. For detailed API documentation, see the [API Reference](api_reference.md). For troubleshooting, consult the [Troubleshooting Guide](troubleshooting.md).