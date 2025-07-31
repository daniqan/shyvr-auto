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
from typing import Dict, Any, Optional, List
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
    """Metadata for preserved models"""
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
    """Validate semantic version format"""
    # Updated regex to support formats like v1.0.0-alpha.1
    pattern = r'^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$'
    return bool(re.match(pattern, version))