"""
Model Preservation Base Classes and Interfaces

Provides base classes, enums, and utilities for model preservation.
"""

import re
import uuid
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod


# Enums
class PreservationPriority(Enum):
    """Priority levels for model preservation"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ModelState(Enum):
    """State of preserved models"""
    ACTIVE = "active"
    PRESERVED = "preserved"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


# Dataclasses
@dataclass
class ModelMetadata:
    """Metadata for preserved models with semantic versioning support"""
    model_id: str
    model_type: str  # e.g., "lstm", "dqn", "transformer"
    version: str
    created_at: datetime
    preserved_at: Optional[datetime] = None
    file_path: str = ""
    file_size_bytes: int = 0
    checksum: str = ""
    compression_type: Optional[str] = None
    preservation_priority: PreservationPriority = PreservationPriority.NORMAL
    state: ModelState = ModelState.ACTIVE
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    mode: Optional[str] = None  # simulation, analysis, live
    description: Optional[str] = None
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate version format on initialization"""
        if not validate_semantic_version_format(self.version):
            raise VersionError(f"Invalid semantic version format: {self.version}")
    
    @property
    def semantic_version(self) -> 'SemanticVersion':
        """
        Get semantic version object for advanced version operations.
        
        Returns:
            SemanticVersion object for this model's version
            
        Raises:
            VersionError: If version string is invalid
        """
        # Import here to avoid circular imports
        from .versioning import SemanticVersion
        return SemanticVersion(self.version)
    
    def is_prerelease(self) -> bool:
        """Check if this model version is a prerelease"""
        return self.semantic_version.is_prerelease()
    
    def is_stable(self) -> bool:
        """Check if this model version is stable"""
        return self.semantic_version.is_stable()
    
    def compare_version(self, other: Union[str, 'ModelMetadata']) -> int:
        """
        Compare this model's version with another version.
        
        Args:
            other: Another version string or ModelMetadata object
            
        Returns:
            -1 if this version is older, 0 if equal, 1 if newer
        """
        if isinstance(other, str):
            other_version = other
        elif isinstance(other, ModelMetadata):
            other_version = other.version
        else:
            raise TypeError(f"Cannot compare version with {type(other)}")
        
        # Import here to avoid circular imports
        from .versioning import SemanticVersion
        
        this_version = self.semantic_version
        other_semantic = SemanticVersion(other_version)
        
        if this_version < other_semantic:
            return -1
        elif this_version > other_semantic:
            return 1
        else:
            return 0


# Abstract Base Classes
class StorageHandlerBase(ABC):
    """Abstract base class for model storage handlers"""
    
    @abstractmethod
    async def save(self, model_data: bytes, metadata: ModelMetadata) -> str:
        """Save model data and return storage path"""
        pass
    
    @abstractmethod
    async def load(self, storage_path: str) -> bytes:
        """Load model data from storage"""
        pass
    
    @abstractmethod
    async def delete(self, storage_path: str) -> bool:
        """Delete model from storage"""
        pass
    
    @abstractmethod
    async def exists(self, storage_path: str) -> bool:
        """Check if model exists in storage"""
        pass
    
    @abstractmethod
    async def list_models(self, prefix: str = "", limit: int = 100) -> List[str]:
        """List models in storage with optional prefix filter"""
        pass
    
    @abstractmethod
    async def get_metadata(self, storage_path: str) -> Dict[str, Any]:
        """Get storage-specific metadata"""
        pass


# Exception Classes
class PreservationError(Exception):
    """Base exception for preservation errors"""
    pass


class StorageError(PreservationError):
    """Storage operation failed"""
    pass


class ChecksumError(PreservationError):
    """Checksum verification failed"""
    pass


class VersionError(PreservationError):
    """Version conflict or invalid version"""
    pass


# Helper Functions
def generate_model_id(model_type: str, version: str) -> str:
    """Generate unique model ID"""
    return f"{model_type}-{version}-{uuid.uuid4().hex[:8]}"


def calculate_checksum(data: bytes) -> str:
    """Calculate SHA256 checksum"""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def format_file_size(size_bytes: int) -> str:
    """Format file size for display"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def validate_version_format(version: str) -> bool:
    """
    Validate semantic version format (legacy function for backward compatibility).
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid semantic version format
        
    Note:
        This function is deprecated. Use validate_semantic_version_format instead.
    """
    return validate_semantic_version_format(version)


def validate_semantic_version_format(version: str) -> bool:
    """
    Validate semantic version format using the new semantic versioning system.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid semantic version format
    """
    try:
        from .versioning import is_valid_semantic_version
        return is_valid_semantic_version(version)
    except ImportError:
        # Fallback to regex if versioning module not available
        pattern = r'^[vV]?\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?(\+[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)*)?$'
        return bool(re.match(pattern, version))