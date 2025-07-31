"""
Model Preservation Module

Provides comprehensive model preservation architecture for:
- Between sessions/restarts
- Between deployments
- During emergency stops
- Across mode changes (Analysis/Simulation/Live)
"""

from .base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    StorageError,
    ChecksumError,
    VersionError,
    StorageHandlerBase,
    generate_model_id,
    calculate_checksum,
    format_file_size,
    validate_version_format
)

# from .gcs_handler import GCSModelPreservationHandler
# from .manager import ModelPreservationManager

__all__ = [
    # Base classes and types
    'ModelMetadata',
    'PreservationPriority',
    'ModelState',
    
    # Exceptions
    'PreservationError',
    'StorageError',
    'ChecksumError',
    'VersionError',
    
    # Interfaces and handlers
    'StorageHandlerBase',
    # 'GCSModelPreservationHandler',
    
    # Manager
    # 'ModelPreservationManager',
    
    # Helper functions
    'generate_model_id',
    'calculate_checksum',
    'format_file_size',
    'validate_version_format'
]

# Global preservation manager instance
_preservation_manager = None


def get_preservation_manager():
    """Get or create the global preservation manager instance"""
    global _preservation_manager
    return _preservation_manager


def set_preservation_manager(manager):
    """Set the global preservation manager instance"""
    global _preservation_manager
    _preservation_manager = manager


async def initialize_preservation_system(config=None):
    """Initialize the model preservation system"""
    from .manager import PreservationManager, PreservationConfig
    from src.utils.config import get_config
    
    # Get configuration
    app_config = get_config()
    
    # Create preservation config
    preservation_config = PreservationConfig(
        gcs_bucket=getattr(app_config.model_preservation, 'gcs_bucket', 'shyvr-models-prod'),
        backup_interval_hours=getattr(app_config.model_preservation, 'backup_interval_hours', 6.0),
        max_versions_per_model=getattr(app_config.model_preservation, 'max_versions_per_model', 10),
        enable_compression=getattr(app_config.model_preservation, 'enable_compression', True),
        mode_isolation=getattr(app_config.model_preservation, 'mode_isolation', True),
        enable_caching=getattr(app_config.model_preservation, 'enable_caching', True),
        enable_metrics=getattr(app_config.model_preservation, 'enable_metrics', True)
    )
    
    # Create and initialize manager
    manager = PreservationManager(preservation_config)
    await manager.initialize()
    await manager.start()
    
    # Set global instance
    set_preservation_manager(manager)
    
    # Set up API integration
    from .api import set_preservation_manager as set_api_manager
    set_api_manager(manager)
    
    return manager


async def shutdown_preservation_system():
    """Shutdown the model preservation system gracefully"""
    if _preservation_manager:
        await _preservation_manager.stop()