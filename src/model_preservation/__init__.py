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
    ModelType,
    PreservationConfig,
    PreservationPriority,
    ModelState,
    PreservationError,
    BackupError,
    RestoreError,
    VersionError,
    ModelPreservationInterface,
    BasePreservationHandler
)

from .gcs_handler import GCSModelPreservationHandler
from .manager import ModelPreservationManager

__all__ = [
    # Base classes and types
    'ModelMetadata',
    'ModelType',
    'PreservationConfig',
    'PreservationPriority',
    'ModelState',
    
    # Exceptions
    'PreservationError',
    'BackupError',
    'RestoreError',
    'VersionError',
    
    # Interfaces and handlers
    'ModelPreservationInterface',
    'BasePreservationHandler',
    'GCSModelPreservationHandler',
    
    # Manager
    'ModelPreservationManager'
]

# Global preservation manager instance
_preservation_manager = None


def get_preservation_manager() -> ModelPreservationManager:
    """Get or create the global preservation manager instance"""
    global _preservation_manager
    
    if _preservation_manager is None:
        _preservation_manager = ModelPreservationManager()
    
    return _preservation_manager


async def initialize_preservation_system(config: PreservationConfig = None):
    """Initialize the model preservation system"""
    manager = get_preservation_manager()
    
    if config:
        manager.config = config
        manager.handler = GCSModelPreservationHandler(config)
    
    await manager.start()
    return manager


async def shutdown_preservation_system():
    """Shutdown the model preservation system gracefully"""
    if _preservation_manager:
        await _preservation_manager.stop()