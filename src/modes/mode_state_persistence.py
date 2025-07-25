"""
Mode State Persistence System

This module provides comprehensive save/load functionality for mode states
and configurations. It handles serialization, validation, migration, and
recovery of mode state data with production-grade reliability.

Key Features:
- Mode state serialization and deserialization
- Configuration persistence and migration
- State validation and integrity checks
- Recovery from corrupted state files
- Atomic file operations and backup management

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import gzip
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import structlog

from src.modes.base import ModeType, ModeStatus, ModeConfig


logger = structlog.get_logger()


class StateVersion(Enum):
    """State file format versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    CURRENT = "1.1"


class StateValidationLevel(Enum):
    """Validation levels for state files."""
    BASIC = "basic"          # Basic structure validation
    COMPREHENSIVE = "comprehensive"  # Full validation including checksums
    STRICT = "strict"        # Strict validation with all constraints


@dataclass
class StateMetadata:
    """Metadata for state files."""
    version: str
    created_at: datetime
    mode_id: UUID
    mode_type: str
    checksum: str
    file_size: int
    compressed: bool = False
    migration_required: bool = False
    validation_level: StateValidationLevel = StateValidationLevel.COMPREHENSIVE


@dataclass
class StateValidationResult:
    """Result of state validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[StateMetadata] = None
    validation_time: datetime = field(default_factory=datetime.now)


@dataclass
class PersistenceConfig:
    """Configuration for state persistence system."""
    state_directory: str = "data/mode_states"
    backup_directory: str = "data/mode_states/backups"
    enable_compression: bool = True
    enable_checksums: bool = True
    enable_atomic_writes: bool = True
    max_backup_files: int = 10
    backup_retention_days: int = 30
    validation_level: StateValidationLevel = StateValidationLevel.COMPREHENSIVE
    auto_migrate: bool = True
    create_directories: bool = True
    file_permissions: int = 0o600  # Read/write for owner only
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_backup_files <= 0:
            raise ValueError("max_backup_files must be positive")
        if self.backup_retention_days <= 0:
            raise ValueError("backup_retention_days must be positive")


class ModeStatePersistence:
    """
    Comprehensive state persistence system for trading modes.
    
    Handles saving, loading, validation, and migration of mode states
    with production-grade reliability and error handling.
    """
    
    def __init__(self, config: PersistenceConfig):
        """Initialize state persistence system."""
        self.config = config
        
        # Setup directories
        self.state_dir = Path(config.state_directory)
        self.backup_dir = Path(config.backup_directory)
        
        if config.create_directories:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.loaded_states: Dict[UUID, Dict[str, Any]] = {}
        self.state_metadata: Dict[UUID, StateMetadata] = {}
        
        # Configure logger
        self.logger = logger.bind(
            component="mode_state_persistence",
            state_directory=str(self.state_dir),
            backup_directory=str(self.backup_dir)
        )
        
        self.logger.info("Mode state persistence initialized")
    
    async def save_mode_state(
        self,
        mode_id: UUID,
        state_data: Dict[str, Any],
        create_backup: bool = True
    ) -> bool:
        """Save mode state to file with optional backup."""
        try:
            # Validate state data
            validation_result = await self._validate_state_data(state_data)
            if not validation_result.valid:
                raise StateValidationError(f"State validation failed: {validation_result.errors}")
            
            # Create backup if requested and file exists
            state_file = self._get_state_file_path(mode_id)
            if create_backup and state_file.exists():
                await self._create_backup(mode_id)
            
            # Prepare state data for saving
            enhanced_state = await self._prepare_state_for_saving(mode_id, state_data)
            
            # Save state using atomic write
            if self.config.enable_atomic_writes:
                success = await self._atomic_write_state(state_file, enhanced_state)
            else:
                success = await self._write_state_file(state_file, enhanced_state)
            
            if success:
                # Update tracking
                self.loaded_states[mode_id] = state_data.copy()
                
                # Create metadata
                metadata = await self._create_state_metadata(mode_id, state_file, enhanced_state)
                self.state_metadata[mode_id] = metadata
                
                self.logger.info(
                    "Mode state saved successfully",
                    mode_id=str(mode_id),
                    file_path=str(state_file),
                    compressed=metadata.compressed,
                    file_size=metadata.file_size
                )
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to save mode state", mode_id=str(mode_id), error=str(e))
            return False
    
    async def load_mode_state(
        self,
        mode_id: UUID,
        validate: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load mode state from file with validation."""
        state_file = self._get_state_file_path(mode_id)
        
        if not state_file.exists():
            self.logger.warning("State file not found", mode_id=str(mode_id), file_path=str(state_file))
            return None
        
        try:
            # Load state data
            raw_data = await self._read_state_file(state_file)
            if not raw_data:
                return None
            
            # Validate if requested
            if validate:
                validation_result = await self._validate_loaded_state(raw_data)
                if not validation_result.valid:
                    # Try to recover from backup
                    self.logger.warning(
                        "State validation failed, attempting backup recovery",
                        mode_id=str(mode_id),
                        errors=validation_result.errors
                    )
                    return await self._recover_from_backup(mode_id)
            
            # Extract state data
            state_data = await self._extract_state_data(raw_data)
            
            # Handle migration if needed
            if self._requires_migration(raw_data):
                if self.config.auto_migrate:
                    state_data = await self._migrate_state_data(state_data, raw_data)
                else:
                    raise StateMigrationError("State migration required but auto_migrate is disabled")
            
            # Update tracking
            self.loaded_states[mode_id] = state_data.copy()
            
            self.logger.info(
                "Mode state loaded successfully",
                mode_id=str(mode_id),
                file_path=str(state_file)
            )
            
            return state_data
            
        except Exception as e:
            self.logger.error("Failed to load mode state", mode_id=str(mode_id), error=str(e))
            
            # Try to recover from backup
            try:
                return await self._recover_from_backup(mode_id)
            except Exception as backup_error:
                self.logger.error("Backup recovery failed", error=str(backup_error))
                return None
    
    async def delete_mode_state(
        self,
        mode_id: UUID,
        create_backup: bool = True
    ) -> bool:
        """Delete mode state file with optional backup."""
        state_file = self._get_state_file_path(mode_id)
        
        if not state_file.exists():
            self.logger.warning("State file not found for deletion", mode_id=str(mode_id))
            return True
        
        try:
            # Create backup before deletion
            if create_backup:
                await self._create_backup(mode_id)
            
            # Delete state file
            state_file.unlink()
            
            # Clean up tracking
            if mode_id in self.loaded_states:
                del self.loaded_states[mode_id]
            if mode_id in self.state_metadata:
                del self.state_metadata[mode_id]
            
            self.logger.info("Mode state deleted", mode_id=str(mode_id))
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete mode state", mode_id=str(mode_id), error=str(e))
            return False
    
    async def list_saved_states(self) -> List[Dict[str, Any]]:
        """List all saved state files with metadata."""
        state_files = []
        
        try:
            for file_path in self.state_dir.glob("state_*.json*"):
                try:
                    # Extract mode ID from filename
                    mode_id = self._extract_mode_id_from_filename(file_path.name)
                    if not mode_id:
                        continue
                    
                    # Get file metadata
                    stat = file_path.stat()
                    
                    state_info = {
                        "mode_id": str(mode_id),
                        "file_path": str(file_path),
                        "file_size": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime),
                        "compressed": file_path.suffix == ".gz"
                    }
                    
                    # Try to load metadata if available
                    if mode_id in self.state_metadata:
                        metadata = self.state_metadata[mode_id]
                        state_info.update({
                            "version": metadata.version,
                            "mode_type": metadata.mode_type,
                            "created_at": metadata.created_at,
                            "checksum": metadata.checksum
                        })
                    
                    state_files.append(state_info)
                    
                except Exception as e:
                    self.logger.warning("Error processing state file", file_path=str(file_path), error=str(e))
                    continue
            
            return sorted(state_files, key=lambda x: x["modified_time"], reverse=True)
            
        except Exception as e:
            self.logger.error("Failed to list saved states", error=str(e))
            return []
    
    async def validate_state_file(
        self,
        mode_id: UUID,
        validation_level: Optional[StateValidationLevel] = None
    ) -> StateValidationResult:
        """Validate a specific state file."""
        validation_level = validation_level or self.config.validation_level
        state_file = self._get_state_file_path(mode_id)
        
        if not state_file.exists():
            return StateValidationResult(
                valid=False,
                errors=[f"State file not found: {state_file}"]
            )
        
        try:
            # Read and validate state data
            raw_data = await self._read_state_file(state_file)
            if not raw_data:
                return StateValidationResult(
                    valid=False,
                    errors=["Failed to read state file"]
                )
            
            return await self._validate_loaded_state(raw_data, validation_level)
            
        except Exception as e:
            return StateValidationResult(
                valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def cleanup_old_backups(self) -> int:
        """Clean up old backup files based on retention policy."""
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        
        try:
            for backup_file in self.backup_dir.glob("backup_*.json*"):
                try:
                    # Check file modification time
                    stat = backup_file.stat()
                    file_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    if file_time < cutoff_date:
                        backup_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    self.logger.warning("Error cleaning backup file", file_path=str(backup_file), error=str(e))
                    continue
            
            self.logger.info("Cleaned up old backups", cleaned_count=cleaned_count)
            return cleaned_count
            
        except Exception as e:
            self.logger.error("Failed to cleanup old backups", error=str(e))
            return 0
    
    async def create_state_snapshot(
        self,
        mode_id: UUID,
        description: str = ""
    ) -> Optional[str]:
        """Create a named snapshot of current state."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"snapshot_{mode_id}_{timestamp}"
            if description:
                snapshot_name += f"_{description}"
            
            state_file = self._get_state_file_path(mode_id)
            if not state_file.exists():
                self.logger.error("Cannot create snapshot: state file not found", mode_id=str(mode_id))
                return None
            
            snapshot_file = self.backup_dir / f"{snapshot_name}.json"
            if self.config.enable_compression:
                snapshot_file = snapshot_file.with_suffix(".json.gz")
            
            # Copy state file to snapshot
            if self.config.enable_compression and not str(state_file).endswith(".gz"):
                # Compress during copy
                with open(state_file, 'rb') as f_in:
                    with gzip.open(snapshot_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(state_file, snapshot_file)
            
            self.logger.info(
                "State snapshot created",
                mode_id=str(mode_id),
                snapshot_file=str(snapshot_file),
                description=description
            )
            
            return str(snapshot_file)
            
        except Exception as e:
            self.logger.error("Failed to create state snapshot", mode_id=str(mode_id), error=str(e))
            return None
    
    # Private methods
    
    def _get_state_file_path(self, mode_id: UUID) -> Path:
        """Get file path for mode state."""
        filename = f"state_{mode_id}.json"
        if self.config.enable_compression:
            filename += ".gz"
        return self.state_dir / filename
    
    def _get_backup_file_path(self, mode_id: UUID) -> Path:
        """Get backup file path for mode state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup_{mode_id}_{timestamp}.json"
        if self.config.enable_compression:
            filename += ".gz"
        return self.backup_dir / filename
    
    async def _prepare_state_for_saving(
        self,
        mode_id: UUID,
        state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare state data for saving with metadata."""
        enhanced_state = {
            "metadata": {
                "version": StateVersion.CURRENT.value,
                "created_at": datetime.now().isoformat(),
                "mode_id": str(mode_id),
                "mode_type": state_data.get("mode_type", "unknown"),
                "serialization_format": "json"
            },
            "state": await self._serialize_state_data(state_data)
        }
        
        # Add checksum if enabled
        if self.config.enable_checksums:
            state_json = json.dumps(enhanced_state["state"], sort_keys=True, default=str)
            checksum = hashlib.sha256(state_json.encode()).hexdigest()
            enhanced_state["metadata"]["checksum"] = checksum
        
        return enhanced_state
    
    async def _serialize_state_data(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state data handling special types."""
        serialized = {}
        
        for key, value in state_data.items():
            if isinstance(value, Decimal):
                serialized[key] = {"_type": "Decimal", "_value": str(value)}
            elif isinstance(value, UUID):
                serialized[key] = {"_type": "UUID", "_value": str(value)}
            elif isinstance(value, datetime):
                serialized[key] = {"_type": "datetime", "_value": value.isoformat()}
            elif isinstance(value, (list, dict)):
                # Recursively serialize complex types
                serialized[key] = await self._serialize_complex_type(value)
            else:
                serialized[key] = value
        
        return serialized
    
    async def _serialize_complex_type(self, obj: Union[List, Dict]) -> Union[List, Dict]:
        """Recursively serialize complex types."""
        if isinstance(obj, dict):
            return {k: await self._serialize_complex_type(v) if isinstance(v, (dict, list)) else v for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._serialize_complex_type(item) if isinstance(item, (dict, list)) else item for item in obj]
        else:
            return obj
    
    async def _atomic_write_state(self, file_path: Path, state_data: Dict[str, Any]) -> bool:
        """Write state file atomically using temporary file."""
        temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
        
        try:
            # Write to temporary file
            await self._write_state_file(temp_file, state_data)
            
            # Atomic move to final location
            temp_file.replace(file_path)
            
            # Set file permissions
            file_path.chmod(self.config.file_permissions)
            
            return True
            
        except Exception as e:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
            
            self.logger.error("Atomic write failed", file_path=str(file_path), error=str(e))
            return False
    
    async def _write_state_file(self, file_path: Path, state_data: Dict[str, Any]) -> bool:
        """Write state data to file."""
        try:
            state_json = json.dumps(state_data, indent=2, default=str)
            
            if self.config.enable_compression:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(state_json)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(state_json)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to write state file", file_path=str(file_path), error=str(e))
            return False
    
    async def _read_state_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read state data from file."""
        try:
            if file_path.suffix == ".gz" or self.config.enable_compression:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            self.logger.error("Failed to read state file", file_path=str(file_path), error=str(e))
            return None
    
    async def _validate_state_data(self, state_data: Dict[str, Any]) -> StateValidationResult:
        """Validate state data structure."""
        errors = []
        warnings = []
        
        # Basic structure validation
        if not isinstance(state_data, dict):
            errors.append("State data must be a dictionary")
            return StateValidationResult(valid=False, errors=errors)
        
        # Check required fields
        required_fields = ["mode_id", "mode_type", "status"]
        for field in required_fields:
            if field not in state_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate field types
        if "mode_id" in state_data:
            try:
                UUID(str(state_data["mode_id"]))
            except ValueError:
                errors.append("Invalid mode_id format")
        
        if "mode_type" in state_data:
            if state_data["mode_type"] not in [mt.value for mt in ModeType]:
                warnings.append(f"Unknown mode_type: {state_data['mode_type']}")
        
        return StateValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_loaded_state(
        self,
        raw_data: Dict[str, Any],
        validation_level: StateValidationLevel = StateValidationLevel.COMPREHENSIVE
    ) -> StateValidationResult:
        """Validate loaded state data."""
        errors = []
        warnings = []
        
        # Check structure
        if "metadata" not in raw_data or "state" not in raw_data:
            errors.append("Invalid state file structure")
            return StateValidationResult(valid=False, errors=errors)
        
        metadata = raw_data["metadata"]
        state_data = raw_data["state"]
        
        # Validate metadata
        if "version" not in metadata:
            errors.append("Missing version in metadata")
        
        # Checksum validation
        if validation_level in [StateValidationLevel.COMPREHENSIVE, StateValidationLevel.STRICT]:
            if self.config.enable_checksums and "checksum" in metadata:
                state_json = json.dumps(state_data, sort_keys=True, default=str)
                expected_checksum = hashlib.sha256(state_json.encode()).hexdigest()
                if metadata["checksum"] != expected_checksum:
                    errors.append("Checksum validation failed")
        
        # Validate state data
        deserialized_state = await self._deserialize_state_data(state_data)
        state_validation = await self._validate_state_data(deserialized_state)
        errors.extend(state_validation.errors)
        warnings.extend(state_validation.warnings)
        
        return StateValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=StateMetadata(
                version=metadata.get("version", "unknown"),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                mode_id=UUID(metadata.get("mode_id", str(uuid4()))),
                mode_type=metadata.get("mode_type", "unknown"),
                checksum=metadata.get("checksum", ""),
                file_size=0,  # Will be set by caller
                compressed=False  # Will be set by caller
            )
        )
    
    async def _extract_state_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and deserialize state data."""
        state_data = raw_data.get("state", {})
        return await self._deserialize_state_data(state_data)
    
    async def _deserialize_state_data(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state data handling special types."""
        deserialized = {}
        
        for key, value in state_data.items():
            if isinstance(value, dict) and "_type" in value and "_value" in value:
                # Handle special types
                type_name = value["_type"]
                type_value = value["_value"]
                
                if type_name == "Decimal":
                    deserialized[key] = Decimal(type_value)
                elif type_name == "UUID":
                    deserialized[key] = UUID(type_value)
                elif type_name == "datetime":
                    deserialized[key] = datetime.fromisoformat(type_value)
                else:
                    deserialized[key] = type_value
            elif isinstance(value, (list, dict)):
                # Recursively deserialize complex types
                deserialized[key] = await self._deserialize_complex_type(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    async def _deserialize_complex_type(self, obj: Union[List, Dict]) -> Union[List, Dict]:
        """Recursively deserialize complex types."""
        if isinstance(obj, dict):
            return {k: await self._deserialize_complex_type(v) if isinstance(v, (dict, list)) else v for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._deserialize_complex_type(item) if isinstance(item, (dict, list)) else item for item in obj]
        else:
            return obj
    
    def _requires_migration(self, raw_data: Dict[str, Any]) -> bool:
        """Check if state data requires migration."""
        metadata = raw_data.get("metadata", {})
        version = metadata.get("version", "1.0")
        return version != StateVersion.CURRENT.value
    
    async def _migrate_state_data(
        self,
        state_data: Dict[str, Any],
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate state data to current version."""
        metadata = raw_data.get("metadata", {})
        from_version = metadata.get("version", "1.0")
        
        self.logger.info(
            "Migrating state data",
            from_version=from_version,
            to_version=StateVersion.CURRENT.value
        )
        
        # Perform version-specific migrations
        if from_version == "1.0":
            state_data = await self._migrate_from_v1_0(state_data)
        
        return state_data
    
    async def _migrate_from_v1_0(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0 to current version."""
        # Add any fields that were added in newer versions
        if "metadata" not in state_data:
            state_data["metadata"] = {}
        
        return state_data
    
    async def _create_backup(self, mode_id: UUID) -> bool:
        """Create backup of existing state file."""
        state_file = self._get_state_file_path(mode_id)
        if not state_file.exists():
            return True
        
        try:
            backup_file = self._get_backup_file_path(mode_id)
            shutil.copy2(state_file, backup_file)
            
            # Clean up old backups
            await self._cleanup_mode_backups(mode_id)
            
            self.logger.debug("State backup created", mode_id=str(mode_id), backup_file=str(backup_file))
            return True
            
        except Exception as e:
            self.logger.error("Failed to create backup", mode_id=str(mode_id), error=str(e))
            return False
    
    async def _cleanup_mode_backups(self, mode_id: UUID) -> None:
        """Clean up old backups for a specific mode."""
        try:
            # Find all backup files for this mode
            backup_pattern = f"backup_{mode_id}_*.json*"
            backup_files = list(self.backup_dir.glob(backup_pattern))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            if len(backup_files) > self.config.max_backup_files:
                for backup_file in backup_files[self.config.max_backup_files:]:
                    try:
                        backup_file.unlink()
                    except Exception as e:
                        self.logger.warning("Failed to remove old backup", file=str(backup_file), error=str(e))
        
        except Exception as e:
            self.logger.error("Failed to cleanup mode backups", mode_id=str(mode_id), error=str(e))
    
    async def _recover_from_backup(self, mode_id: UUID) -> Optional[Dict[str, Any]]:
        """Recover state from backup files."""
        try:
            # Find most recent backup
            backup_pattern = f"backup_{mode_id}_*.json*"
            backup_files = list(self.backup_dir.glob(backup_pattern))
            
            if not backup_files:
                self.logger.error("No backup files found for recovery", mode_id=str(mode_id))
                return None
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Try to load from most recent backup
            for backup_file in backup_files:
                try:
                    raw_data = await self._read_state_file(backup_file)
                    if raw_data:
                        validation_result = await self._validate_loaded_state(raw_data)
                        if validation_result.valid:
                            state_data = await self._extract_state_data(raw_data)
                            
                            self.logger.info(
                                "Successfully recovered from backup",
                                mode_id=str(mode_id),
                                backup_file=str(backup_file)
                            )
                            
                            return state_data
                except Exception as e:
                    self.logger.warning("Backup recovery attempt failed", backup_file=str(backup_file), error=str(e))
                    continue
            
            self.logger.error("All backup recovery attempts failed", mode_id=str(mode_id))
            return None
            
        except Exception as e:
            self.logger.error("Backup recovery failed", mode_id=str(mode_id), error=str(e))
            return None
    
    async def _create_state_metadata(
        self,
        mode_id: UUID,
        file_path: Path,
        state_data: Dict[str, Any]
    ) -> StateMetadata:
        """Create metadata for saved state."""
        metadata = state_data.get("metadata", {})
        stat = file_path.stat()
        
        return StateMetadata(
            version=metadata.get("version", StateVersion.CURRENT.value),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            mode_id=mode_id,
            mode_type=metadata.get("mode_type", "unknown"),
            checksum=metadata.get("checksum", ""),
            file_size=stat.st_size,
            compressed=str(file_path).endswith(".gz")
        )
    
    def _extract_mode_id_from_filename(self, filename: str) -> Optional[UUID]:
        """Extract mode ID from state filename."""
        try:
            # Remove prefix and suffixes
            if filename.startswith("state_"):
                id_part = filename[6:]  # Remove "state_"
                if id_part.endswith(".json.gz"):
                    id_part = id_part[:-8]  # Remove ".json.gz"
                elif id_part.endswith(".json"):
                    id_part = id_part[:-5]  # Remove ".json"
                
                return UUID(id_part)
        except ValueError:
            pass
        
        return None


# Exception Classes
class StatePersistenceError(Exception):
    """Base state persistence error."""
    pass


class StateValidationError(StatePersistenceError):
    """Error during state validation."""
    pass


class StateCorruptionError(StatePersistenceError):
    """Error indicating state corruption."""
    pass


class StateMigrationError(StatePersistenceError):
    """Error during state migration."""
    pass