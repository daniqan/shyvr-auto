# Model Preservation Architecture

## Overview

The Shyvr RLTE Model Preservation System provides comprehensive model storage, versioning, and recovery capabilities designed to handle all operational scenarios:

- **Between Sessions/Restarts**: Automatic model persistence
- **Between Deployments**: Version control and rollback
- **During Emergency Stops**: Graceful shutdown with critical backups
- **Across Mode Changes**: Mode-specific model isolation (Analysis/Simulation/Live)

## Key Features

### 1. Graceful Shutdown Handling
- Signal handlers for SIGTERM/SIGINT
- Emergency backup with CRITICAL priority
- Automatic preservation of all active models
- Guaranteed model state persistence

### 2. Automated Backups
- Configurable backup intervals (default: 5 minutes)
- Background backup tasks
- Priority-based retention policies
- Automatic cleanup of old models

### 3. Model Versioning
- Semantic versioning (v1.0.0 format)
- Automatic version incrementing
- Version history tracking
- Rollback to any previous version

### 4. Rollback Capabilities
- Rollback by version number
- Rollback by date/time
- Rollback with performance validation
- Audit trail of all rollback operations

### 5. Mode-Specific Isolation
- Separate model storage per mode
- Automatic backup on mode transitions
- Cross-mode model migration support
- Mode-specific performance tracking

## Architecture Components

### Storage Backend (GCS)
```python
# GCS bucket structure:
models/
├── analysis/
│   ├── lstm/
│   │   ├── v1.0.0/
│   │   └── v1.0.1/
│   ├── dqn/
│   └── ensemble/
├── simulation/
│   ├── lstm/
│   ├── dqn/
│   └── ensemble/
└── live/
    ├── lstm/
    ├── dqn/
    └── ensemble/
```

### Database Schema (Cloud SQL)
- `model_preservation_metadata`: Core metadata storage
- `model_version_history`: Version tracking
- `model_preservation_events`: Audit trail
- `model_performance_tracking`: Performance metrics

## Configuration

### Basic Configuration
```python
from src.model_preservation import PreservationConfig

config = PreservationConfig(
    # GCS settings
    gcs_bucket="shyvr-models",
    gcs_prefix="models",
    
    # Backup settings
    max_backups_per_model=10,
    backup_retention_days=30,
    emergency_backup_retention_days=90,
    
    # Versioning settings
    versioning_enabled=True,
    version_format="v{major}.{minor}.{patch}",
    auto_increment_version=True,
    
    # Performance settings
    compression_enabled=True,
    compression_level=6,
    chunk_size_mb=50,
    
    # Mode isolation
    isolate_by_mode=True,
    
    # Emergency settings
    emergency_backup_enabled=True,
    emergency_backup_interval_minutes=5
)
```

## Usage Examples

### 1. Basic Model Preservation
```python
from src.model_preservation import get_preservation_manager, ModelType

# Get preservation manager
manager = get_preservation_manager()
await manager.start()

# Save a model
preservation_id = await manager.save_ml_model(
    model_type=ModelType.LSTM,
    model_data=lstm_model,
    performance_metrics={"accuracy": 0.95},
    reason="checkpoint",
    priority=PreservationPriority.NORMAL
)

# Load latest model
model, metadata = await manager.load_ml_model(
    model_type=ModelType.LSTM,
    mode="simulation"
)
```

### 2. Mode Change Handling
```python
# Set initial mode
await manager.set_mode("analysis")

# Save models in analysis mode
await manager.save_ml_model(...)

# Change mode (triggers automatic backup)
await manager.set_mode("simulation")

# Models are now isolated by mode
```

### 3. Emergency Shutdown
```python
# Automatic handling via signal handlers
# Or manual emergency backup:
await manager._perform_shutdown_backup()
```

### 4. Version Rollback
```python
# Rollback to specific version
model, metadata = await manager.rollback_model(
    model_type=ModelType.DQN,
    target_version="v1.2.0"
)

# Rollback by date
model, metadata = await manager.rollback_model(
    model_type=ModelType.DQN,
    target_date=datetime(2024, 1, 1)
)
```

### 5. Integration with Existing Components
```python
from src.model_preservation.integration import PreservationIntegration

# Create integration
integration = PreservationIntegration(manager)

# Integrate with ML models
integration.integrate_with_model_manager(model_manager)

# Integrate with RL agents
integration.integrate_with_rl_agent("agent_001", rl_agent)

# Now all saves/loads are automatically preserved
```

## API Endpoints

### Health Check
```
GET /api/preservation/health
```
Returns preservation system health status including GCS accessibility, database status, and storage metrics.

### List Models
```
GET /api/preservation/models?model_type=lstm&mode=simulation&limit=50
```
Lists available preserved models with filtering options.

### Rollback Model
```
POST /api/preservation/rollback/{model_type}
{
    "target_version": "v1.2.0",
    "target_date": "2024-01-01T00:00:00"
}
```
Performs model rollback to specified version or date.

## Database Migration

Run the preservation schema migration:
```bash
psql -d shyvr_rlte -f database/migrations/005_create_model_preservation_schema.sql
```

## Monitoring and Maintenance

### Retention Policy
Models are automatically cleaned up based on:
- Priority level (CRITICAL models kept longer)
- Age (configurable retention days)
- Count (max backups per model type)

### Performance Metrics
The system tracks:
- Model accuracy over time
- Inference latency
- Storage usage
- Preservation operation durations

### Audit Trail
All preservation operations are logged:
- Model saves/loads
- Version changes
- Rollback operations
- Deletion events

## Best Practices

1. **Regular Backups**: Keep emergency backup interval reasonable (5-15 minutes)
2. **Version Management**: Use semantic versioning for clear history
3. **Mode Isolation**: Always set correct mode before operations
4. **Priority Usage**:
   - CRITICAL: Emergency stops, deployments
   - HIGH: Mode changes, important checkpoints
   - NORMAL: Regular saves
   - LOW: Background/experimental models

5. **Storage Optimization**: Enable compression for large models

## Troubleshooting

### Common Issues

1. **GCS Access Errors**
   - Verify bucket permissions
   - Check service account credentials
   - Ensure bucket exists

2. **Version Conflicts**
   - Check version history table
   - Use rollback to resolve
   - Manual version reset if needed

3. **Storage Limits**
   - Monitor storage usage
   - Adjust retention policies
   - Clean up archived models

### Debug Commands
```python
# Check preservation health
health = await manager.get_preservation_health()

# List all models
models = await manager.list_available_models()

# Force cleanup
await manager.handler._apply_retention_policy(metadata)
```

## Security Considerations

1. **Access Control**: GCS bucket should have proper IAM policies
2. **Encryption**: Models are encrypted at rest in GCS
3. **Audit Logging**: All operations logged for compliance
4. **Soft Deletes**: Models marked as deleted but retained for recovery

## Future Enhancements

1. **Model Diffing**: Store only deltas between versions
2. **Distributed Preservation**: Multi-region replication
3. **Hot Swapping**: Zero-downtime model updates
4. **A/B Testing**: Simultaneous multi-version deployment
5. **Performance Profiling**: Automatic model performance regression detection