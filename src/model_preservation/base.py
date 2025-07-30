"""
Model Preservation Base Classes and Interfaces

Provides comprehensive model preservation architecture for handling:
- Between sessions/restarts
- Between deployments  
- During emergency stops
- Across mode changes (Analysis/Simulation/Live)
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import structlog
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from src.utils.database import get_database_connection, DatabaseError


logger = structlog.get_logger()


class PreservationError(Exception):
    """Base exception for model preservation errors"""
    pass


class BackupError(PreservationError):
    """Error during backup operations"""
    pass


class RestoreError(PreservationError):
    """Error during restore operations"""
    pass


class VersionError(PreservationError):
    """Error with model versioning"""
    pass


class ModelType(Enum):
    """Types of models in the system"""
    LSTM = "lstm"
    DQN = "dqn"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class PreservationPriority(Enum):
    """Priority levels for preservation operations"""
    CRITICAL = "critical"  # Emergency stops, mode changes
    HIGH = "high"         # Deployments, planned restarts
    NORMAL = "normal"     # Regular checkpoints
    LOW = "low"           # Background backups


class ModelState(Enum):
    """Model states in preservation lifecycle"""
    ACTIVE = "active"
    PRESERVED = "preserved"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


@dataclass
class ModelMetadata:
    """Metadata for preserved models"""
    model_id: str
    model_type: ModelType
    version: str
    mode: str  # analysis, simulation, live
    created_at: datetime
    preserved_at: datetime
    checksum: str
    size_bytes: int
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    training_info: Dict[str, Any] = field(default_factory=dict)
    preservation_reason: str = ""
    priority: PreservationPriority = PreservationPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreservationConfig:
    """Configuration for model preservation"""
    # GCS settings
    gcs_bucket: str
    gcs_prefix: str = "models"
    
    # Backup settings
    max_backups_per_model: int = 10
    backup_retention_days: int = 30
    emergency_backup_retention_days: int = 90
    
    # Versioning settings
    versioning_enabled: bool = True
    version_format: str = "v{major}.{minor}.{patch}"
    auto_increment_version: bool = True
    
    # Performance settings
    compression_enabled: bool = True
    compression_level: int = 6
    chunk_size_mb: int = 50
    max_concurrent_operations: int = 3
    
    # Mode isolation settings
    isolate_by_mode: bool = True
    mode_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "analysis": "analysis",
        "simulation": "simulation", 
        "live": "live"
    })
    
    # Emergency settings
    emergency_backup_enabled: bool = True
    emergency_backup_interval_minutes: int = 5
    
    # Database settings
    metadata_table: str = "model_preservation_metadata"


class ModelPreservationInterface(ABC):
    """Interface for model preservation operations"""
    
    @abstractmethod
    async def save_model(
        self,
        model_data: bytes,
        metadata: ModelMetadata,
        priority: PreservationPriority = PreservationPriority.NORMAL
    ) -> str:
        """Save model with metadata and return preservation ID"""
        pass
    
    @abstractmethod
    async def load_model(
        self,
        preservation_id: str
    ) -> Tuple[bytes, ModelMetadata]:
        """Load model by preservation ID"""
        pass
    
    @abstractmethod
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        mode: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """List available models with filters"""
        pass
    
    @abstractmethod
    async def delete_model(
        self,
        preservation_id: str,
        soft_delete: bool = True
    ) -> bool:
        """Delete or archive a preserved model"""
        pass
    
    @abstractmethod
    async def rollback_model(
        self,
        model_type: ModelType,
        target_version: Optional[str] = None,
        target_date: Optional[datetime] = None
    ) -> Tuple[bytes, ModelMetadata]:
        """Rollback to a previous model version"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check preservation system health"""
        pass


class BasePreservationHandler(ModelPreservationInterface):
    """Base implementation of model preservation"""
    
    def __init__(self, config: PreservationConfig):
        self.config = config
        self.logger = structlog.get_logger().bind(component="ModelPreservation")
        self._shutdown_event = asyncio.Event()
        self._emergency_backup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start preservation services"""
        if self.config.emergency_backup_enabled:
            self._emergency_backup_task = asyncio.create_task(
                self._emergency_backup_loop()
            )
        self.logger.info("Model preservation started", config=self.config)
    
    async def stop(self):
        """Stop preservation services gracefully"""
        self._shutdown_event.set()
        if self._emergency_backup_task:
            await self._emergency_backup_task
        self.logger.info("Model preservation stopped")
    
    async def _emergency_backup_loop(self):
        """Background task for emergency backups"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(
                    self.config.emergency_backup_interval_minutes * 60
                )
                if not self._shutdown_event.is_set():
                    await self._perform_emergency_backup()
            except Exception as e:
                self.logger.error("Emergency backup failed", error=str(e))
    
    async def _perform_emergency_backup(self):
        """Perform emergency backup of active models"""
        # Implementation in derived classes
        pass
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of model data"""
        return hashlib.sha256(data).hexdigest()
    
    def _generate_preservation_id(
        self,
        metadata: ModelMetadata
    ) -> str:
        """Generate unique preservation ID"""
        components = [
            metadata.model_type.value,
            metadata.mode,
            metadata.version,
            metadata.preserved_at.isoformat()
        ]
        return "_".join(components).replace(":", "-")
    
    @asynccontextmanager
    async def _preservation_transaction(self):
        """Context manager for preservation transactions"""
        async with get_database_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def _save_metadata_to_db(
        self,
        metadata: ModelMetadata,
        preservation_id: str,
        conn=None
    ):
        """Save model metadata to database"""
        query = f"""
            INSERT INTO {self.config.metadata_table} (
                preservation_id, model_id, model_type, version, mode,
                created_at, preserved_at, checksum, size_bytes,
                performance_metrics, training_info, preservation_reason,
                priority, tags, metadata, state
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
        """
        
        await conn.execute(
            query,
            preservation_id,
            metadata.model_id,
            metadata.model_type.value,
            metadata.version,
            metadata.mode,
            metadata.created_at,
            metadata.preserved_at,
            metadata.checksum,
            metadata.size_bytes,
            json.dumps(metadata.performance_metrics),
            json.dumps(metadata.training_info),
            metadata.preservation_reason,
            metadata.priority.value,
            metadata.tags,
            json.dumps(metadata.metadata),
            ModelState.PRESERVED.value
        )
    
    async def _get_metadata_from_db(
        self,
        preservation_id: str,
        conn=None
    ) -> Optional[ModelMetadata]:
        """Retrieve model metadata from database"""
        query = f"""
            SELECT model_id, model_type, version, mode, created_at,
                   preserved_at, checksum, size_bytes, performance_metrics,
                   training_info, preservation_reason, priority, tags, metadata
            FROM {self.config.metadata_table}
            WHERE preservation_id = $1 AND state != $2
        """
        
        row = await conn.fetchrow(query, preservation_id, ModelState.DELETED.value)
        
        if not row:
            return None
        
        return ModelMetadata(
            model_id=row['model_id'],
            model_type=ModelType(row['model_type']),
            version=row['version'],
            mode=row['mode'],
            created_at=row['created_at'],
            preserved_at=row['preserved_at'],
            checksum=row['checksum'],
            size_bytes=row['size_bytes'],
            performance_metrics=json.loads(row['performance_metrics']),
            training_info=json.loads(row['training_info']),
            preservation_reason=row['preservation_reason'],
            priority=PreservationPriority(row['priority']),
            tags=row['tags'],
            metadata=json.loads(row['metadata'])
        )
    
    def _get_mode_prefix(self, mode: str) -> str:
        """Get storage prefix for mode isolation"""
        if not self.config.isolate_by_mode:
            return ""
        return self.config.mode_prefixes.get(mode, mode)