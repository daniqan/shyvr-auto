"""
Model Preservation Manager

Coordinates between GCS storage and database metadata for comprehensive model preservation.
Implements Phase 1.4 requirements from TODO_CHECKLIST.md.
"""

import asyncio
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re

from .base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    VersionError,
    generate_model_id,
    generate_cache_key,
    calculate_checksum
)
from .versioning import SemanticVersion, find_latest_version
from .gcs_handler import GCSHandler
from .db_handler import DatabaseHandler
from .caching import (
    CacheConfig,
    CacheManager,
    CacheWarmer,
    LRUCache,
    DiskCache,
    GCSCache
)
from .monitoring import PreservationMetricsCollector, StructuredLogger


@dataclass
class PreservationConfig:
    """Configuration for preservation manager"""
    gcs_bucket: str
    backup_interval_hours: float = 6.0
    max_versions_per_model: int = 10
    enable_compression: bool = True
    mode_isolation: bool = True
    auto_backup: bool = True
    emergency_backup: bool = True
    
    # Caching configuration
    enable_caching: bool = True
    
    # Monitoring configuration
    enable_metrics: bool = True
    enable_structured_logging: bool = True
    cache_memory_limit_gb: float = 2.0
    cache_disk_dir: str = "/tmp/models"
    cache_warmup_enabled: bool = True
    cache_prefetch_enabled: bool = True
    cache_metrics_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.backup_interval_hours <= 0:
            raise ValueError("Backup interval must be positive")
        if self.max_versions_per_model < 1:
            raise ValueError("Max versions must be at least 1")
        if self.enable_caching and self.cache_memory_limit_gb <= 0:
            raise ValueError("Cache memory limit must be positive")


@dataclass
class PreservationStats:
    """Statistics for preservation system"""
    total_models: int = 0
    active_models: int = 0
    total_size_mb: float = 0.0
    models_by_type: Dict[str, int] = field(default_factory=dict)
    models_by_mode: Dict[str, int] = field(default_factory=dict)
    last_backup: Optional[datetime] = None
    backup_success_rate: float = 100.0


class PreservationManager:
    """
    Manager for model preservation
    
    Coordinates between GCS storage and database operations to provide:
    - Auto-versioning support
    - Graceful shutdown handling
    - Emergency backup functionality
    - Background backup tasks
    - Rollback functionality
    - Model migration between modes
    """
    
    def __init__(self, config: PreservationConfig):
        self.config = config
        
        # Initialize storage handler
        self.storage_handler = GCSHandler(bucket_name=config.gcs_bucket)
        
        # Initialize database handler
        self.db_handler = DatabaseHandler()
        
        # Initialize caching system if enabled
        self.cache_manager = None
        self.cache_warmer = None
        if config.enable_caching:
            self._initialize_caching()
        
        # Initialize monitoring if enabled
        self.metrics_collector = None
        self.logger = None
        if config.enable_metrics:
            self.metrics_collector = PreservationMetricsCollector()
        if config.enable_structured_logging:
            self.logger = StructuredLogger(component="model_preservation")
        
        # Internal state
        self._shutdown_event = asyncio.Event()
        self._background_task = None
        self._is_running = False
        self._version_cache: Dict[str, List[str]] = {}  # model_type -> versions
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _initialize_caching(self):
        """Initialize the caching system"""
        try:
            # Create cache configuration
            cache_config = CacheConfig(
                memory_limit_gb=self.config.cache_memory_limit_gb,
                disk_cache_dir=self.config.cache_disk_dir,
                gcs_bucket=self.config.gcs_bucket,
                cache_warmup_enabled=self.config.cache_warmup_enabled,
                prefetch_enabled=self.config.cache_prefetch_enabled,
                metrics_enabled=self.config.cache_metrics_enabled,
                compression_enabled=self.config.enable_compression
            )
            
            # Initialize cache layers
            memory_cache = LRUCache(
                max_size_bytes=int(cache_config.memory_limit_gb * 1024 * 1024 * 1024)
            )
            
            disk_cache = DiskCache(
                cache_dir=cache_config.disk_cache_dir,
                compression_enabled=cache_config.compression_enabled
            )
            
            # GCS cache uses the same client as storage handler
            gcs_cache = None
            if hasattr(self.storage_handler, 'client'):
                gcs_cache = GCSCache(
                    gcs_client=self.storage_handler.client,
                    bucket_name=cache_config.gcs_bucket,
                    cache_prefix="cache/"
                )
            
            # Create cache manager
            self.cache_manager = CacheManager(
                config=cache_config,
                memory_cache=memory_cache,
                disk_cache=disk_cache,
                gcs_cache=gcs_cache
            )
            
            # Create cache warmer
            if cache_config.cache_warmup_enabled:
                self.cache_warmer = CacheWarmer(
                    cache_manager=self.cache_manager,
                    preservation_manager=self
                )
                
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Failed to initialize caching system: {e}")
            self.cache_manager = None
            self.cache_warmer = None
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        asyncio.create_task(self._handle_shutdown())
    
    async def _handle_shutdown(self):
        """Perform emergency backup on shutdown"""
        try:
            if self.config.emergency_backup:
                await self.emergency_backup()
        finally:
            self._shutdown_event.set()
    
    async def initialize(self):
        """Initialize handlers and start services"""
        # Initialize storage handler
        await self.storage_handler.initialize()
        
        # Initialize database handler when available
        if self.db_handler:
            await self.db_handler.initialize()
        
        self._is_running = True
    
    async def start(self):
        """Start background services"""
        await self.initialize()
        
        # Warm cache if enabled
        if self.cache_warmer and self.config.cache_warmup_enabled:
            try:
                await self.cache_warmer.warm_latest_models(count=10)
            except Exception as e:
                print(f"Warning: Cache warming failed: {e}")
        
        # Start background backup task if enabled
        if self.config.auto_backup:
            self._background_task = asyncio.create_task(self._background_backup_loop())
    
    async def stop(self):
        """Stop services gracefully"""
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel background task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Perform final backup if configured
        if self.config.emergency_backup:
            await self.emergency_backup()
    
    async def save_model(
        self,
        model_data: bytes,
        model_type: str,
        version: Optional[str] = None,
        mode: str = "analysis",
        branch: str = "main",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: PreservationPriority = PreservationPriority.NORMAL,
        increment_type: str = "patch"
    ) -> str:
        """
        Save a model with auto-versioning support using semantic versioning.
        
        Args:
            model_data: Serialized model data
            model_type: Type of model (e.g., "lstm", "dqn")
            version: Optional version string, auto-generated if not provided
            mode: Operational mode (analysis, simulation, live)
            branch: Branch to save to (default: main)
            tags: Optional tags for the model
            metadata: Optional additional metadata
            priority: Preservation priority
            increment_type: Type of version increment ("patch", "minor", "major")
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        import time
        start_time = time.time()
        success = False
        actual_version = version
        
        try:
            # Validate branch exists
            if self.db_handler and not await self.db_handler.branch_exists(branch):
                raise ValueError(f"Branch '{branch}' does not exist")
            
            # Generate version if not provided
            if not version:
                version = await self._generate_next_version(model_type, increment_type, branch=branch)
            
            # Create metadata object
            model_metadata = ModelMetadata(
                model_id=generate_model_id(model_type, version),
                model_type=model_type,
                version=version,
                created_at=datetime.now(),
                preserved_at=datetime.now(),
                file_size_bytes=len(model_data),
                checksum=calculate_checksum(model_data),
                preservation_priority=priority,
                state=ModelState.ACTIVE,
                mode=mode,
                branch=branch,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Save to storage
            storage_path = await self.storage_handler.save(model_data, model_metadata)
            model_metadata.file_path = storage_path
            
            # Save metadata to database if available
            if self.db_handler:
                model_id = await self.db_handler.save_metadata(model_metadata)
                
                # Record save event
                await self.db_handler.record_event(
                    model_id=model_id,
                    event_type="saved",
                    details={"priority": priority.value}
                )
                
                # Update standard tags automatically
                await self.db_handler.update_standard_tags(
                    model_type=model_type,
                    new_version=version,
                    preservation_id=model_metadata.preservation_id if hasattr(model_metadata, 'preservation_id') else model_id,
                    mode=mode
                )
            else:
                model_id = model_metadata.model_id
            
            # Enforce version limit
            await self._enforce_version_limit(model_type, mode)
            
            # Mark success for monitoring
            success = True
            actual_version = version
            
            return model_id
            
        except Exception as e:
            # Record error for monitoring
            if self.metrics_collector:
                self.metrics_collector.record_error(
                    operation="save",
                    model_type=model_type,
                    error_type=type(e).__name__
                )
            
            if self.logger:
                self.logger.log_error(
                    operation="save",
                    model_type=model_type,
                    version=actual_version or "unknown",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            
            if isinstance(e, ValueError):
                raise
            raise PreservationError(f"Failed to save model: {str(e)}")
        
        finally:
            # Record metrics and logging
            duration = time.time() - start_time
            
            if self.metrics_collector and success:
                self.metrics_collector.record_save_duration(
                    model_type=model_type,
                    duration=duration,
                    model_size=len(model_data),
                    success=success,
                    priority=priority.value,
                    mode=mode
                )
            
            if self.logger and success:
                self.logger.log_save_operation(
                    model_type=model_type,
                    version=actual_version,
                    duration=duration,
                    size_bytes=len(model_data),
                    success=success,
                    storage_path=getattr(model_metadata, 'file_path', 'unknown') if 'model_metadata' in locals() else 'unknown',
                    mode=mode,
                    priority=priority.value
                )
    
    async def load_model(
        self,
        model_type: str,
        version: Optional[str] = None,
        mode: str = "analysis",
        branch: str = "main",
        fallback: bool = False
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Load a model from storage with caching support
        
        Args:
            model_type: Type of model to load
            version: Specific version to load (latest if not specified, can be a tag name)
            mode: Operational mode
            branch: Branch to load from (default: main)
            fallback: Whether to fallback to previous version if not found
            
        Returns:
            Tuple of (model_data, metadata)
        """
        import time
        start_time = time.time()
        success = False
        cache_hit = False
        actual_version = version
        
        try:
            # Validate branch exists
            if self.db_handler and not await self.db_handler.branch_exists(branch):
                raise ValueError(f"Branch '{branch}' does not exist")
            
            # Resolve tag to version if version looks like a tag
            resolved_version = version
            if version and self.db_handler:
                # Check if version is a tag name
                tag_version = await self.db_handler.resolve_tag_to_version(model_type, version, branch=branch)
                if tag_version:
                    resolved_version = tag_version
            
            # Try cache first if caching is enabled
            if self.cache_manager and resolved_version:
                cache_key = generate_cache_key(model_type, resolved_version, mode, branch)
                cached_entry = await self.cache_manager.get(cache_key)
                if cached_entry:
                    cache_hit = True
                    success = True
                    actual_version = resolved_version
                    return cached_entry.data, cached_entry.metadata
            
            # Get metadata from database if available
            if self.db_handler:
                metadata = await self.db_handler.get_metadata(
                    model_type=model_type,
                    version=resolved_version,
                    mode=mode,
                    branch=branch
                )
                
                if not metadata and fallback:
                    # Try to find previous version
                    versions = await self.db_handler.get_versions(
                        model_type=model_type,
                        mode=mode,
                        branch=branch
                    )
                    if versions:
                        # Use previous version
                        for prev_version in versions:
                            metadata = await self.db_handler.get_metadata(
                                model_type=model_type,
                                version=prev_version["version"],
                                mode=mode,
                                branch=branch
                            )
                            if metadata:
                                resolved_version = prev_version["version"]
                                break
                
                if not metadata:
                    raise FileNotFoundError(f"Model not found: {model_type} {version or 'latest'}")
                
                # Load from storage
                model_data = await self.storage_handler.load(metadata["storage_path"])
                
                # Cache the loaded model if caching is enabled
                if self.cache_manager and resolved_version:
                    cache_key = generate_cache_key(model_type, resolved_version, mode, branch)
                    await self.cache_manager.put(cache_key, model_data, metadata)
                
                # Record load event
                event_details = {"source": "storage"}
                if version:
                    event_details["requested_version"] = version
                    
                # Check if we're using a fallback version
                if fallback and version and metadata.get("version") != version:
                    event_details["source"] = "fallback"
                    
                await self.db_handler.record_event(
                    model_id=metadata["model_id"],
                    event_type="loaded",
                    details=event_details
                )
                
                success = True
                actual_version = metadata.get("version", resolved_version)
                return model_data, metadata
            else:
                # Direct storage load without database
                # This is a simplified implementation for when db_handler is not available
                raise FileNotFoundError("Model not found: database handler not available")
                
        except Exception as e:
            # Record error for monitoring
            if self.metrics_collector:
                self.metrics_collector.record_error(
                    operation="load",
                    model_type=model_type,
                    error_type=type(e).__name__
                )
            
            if self.logger:
                self.logger.log_error(
                    operation="load",
                    model_type=model_type,
                    version=actual_version or "unknown",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise PreservationError(f"Failed to load model: {str(e)}")
        
        finally:
            # Record metrics and logging for load operation
            duration = time.time() - start_time
            
            if self.metrics_collector:
                # Record cache metrics
                if cache_hit:
                    self.metrics_collector.record_cache_hit(model_type)
                elif not success:  # Only record miss if we actually tried (success False but no error)
                    self.metrics_collector.record_cache_miss(model_type)
                
                # Record load duration
                self.metrics_collector.record_load_duration(
                    model_type=model_type,
                    duration=duration,
                    cache_hit=cache_hit,
                    success=success
                )
            
            if self.logger and success:
                self.logger.log_load_operation(
                    model_type=model_type,
                    version=actual_version or "unknown",
                    duration=duration,
                    cache_hit=cache_hit,
                    success=success,
                    mode=mode
                )
    
    async def rollback_model(
        self,
        model_type: str,
        target_version: str,
        mode: str = "analysis"
    ) -> Dict[str, Any]:
        """
        Rollback to a previous model version
        
        Args:
            model_type: Type of model to rollback
            target_version: Version to rollback to
            mode: Operational mode
            
        Returns:
            Rollback result information
        """
        try:
            # Get current version metadata
            current_metadata = await self.db_handler.get_metadata(
                model_type=model_type,
                mode=mode
            )
            
            # Get target version metadata
            target_metadata = await self.db_handler.get_metadata(
                model_type=model_type,
                version=target_version,
                mode=mode
            )
            
            if not target_metadata:
                raise VersionError(f"Target version not found: {target_version}")
            
            # Load target model
            model_data = await self.storage_handler.load(target_metadata["storage_path"])
            
            # Save as new version
            new_version = await self._generate_next_version(model_type)
            new_model_id = await self.save_model(
                model_data=model_data,
                model_type=model_type,
                version=new_version,
                mode=mode,
                tags=["rollback", f"from_{current_metadata['version']}"],
                metadata={"rollback_from": current_metadata["version"], "rollback_to": target_version}
            )
            
            # Record rollback event
            await self.db_handler.record_event(
                model_id=new_model_id,
                event_type="rollback",
                details={
                    "from_version": current_metadata["version"],
                    "to_version": target_version,
                    "new_version": new_version
                }
            )
            
            return {
                "rolled_back_from": current_metadata["version"],
                "rolled_back_to": target_version,
                "new_version": new_version,
                "model_id": new_model_id
            }
            
        except Exception as e:
            raise PreservationError(f"Failed to rollback model: {str(e)}")
    
    async def migrate_model(
        self,
        model_type: str,
        version: str,
        from_mode: str,
        to_mode: str
    ) -> str:
        """
        Migrate a model between operational modes
        
        Args:
            model_type: Type of model to migrate
            version: Version to migrate
            from_mode: Source mode
            to_mode: Target mode
            
        Returns:
            New model ID in target mode
        """
        try:
            # Get source model metadata
            source_metadata = await self.db_handler.get_metadata(
                model_type=model_type,
                version=version,
                mode=from_mode
            )
            
            if not source_metadata:
                raise FileNotFoundError(f"Source model not found: {model_type} {version} in {from_mode}")
            
            # Load model data
            model_data = await self.storage_handler.load(source_metadata["storage_path"])
            
            # Save to new mode
            new_model_id = await self.save_model(
                model_data=model_data,
                model_type=model_type,
                version=version,
                mode=to_mode,
                tags=[f"migrated_from_{from_mode}"],
                metadata={"migrated_from": from_mode, "original_model_id": source_metadata["model_id"]}
            )
            
            return new_model_id
            
        except Exception as e:
            raise PreservationError(f"Failed to migrate model: {str(e)}")
    
    async def emergency_backup(self) -> List[str]:
        """
        Perform emergency backup of all active models
        
        Returns:
            List of backed up model IDs
        """
        backed_up = []
        
        try:
            if self.db_handler:
                # Get all active models
                active_models = await self.db_handler.get_active_models()
                
                for model in active_models:
                    try:
                        # Load model data
                        model_data = await self.storage_handler.load(model["storage_path"])
                        
                        # Save with emergency tag
                        model_id = await self.save_model(
                            model_data=model_data,
                            model_type=model["model_type"],
                            version=f"{model['version']}-emergency-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            mode=model["mode"],
                            tags=["emergency_backup"],
                            priority=PreservationPriority.CRITICAL
                        )
                        
                        backed_up.append(model_id)
                    except Exception as e:
                        # Log error but continue with other models
                        pass
            
            return backed_up
            
        except Exception as e:
            raise PreservationError(f"Emergency backup failed: {str(e)}")
    
    async def get_stats(self) -> PreservationStats:
        """Get preservation system statistics"""
        stats = PreservationStats()
        
        if self.db_handler:
            db_stats = await self.db_handler.get_preservation_stats()
            stats.total_models = db_stats.get("total_models", 0)
            stats.active_models = db_stats.get("active_models", 0)
            stats.total_size_mb = db_stats.get("total_size_bytes", 0) / 1024 / 1024
            stats.models_by_type = db_stats.get("models_by_type", {})
            stats.models_by_mode = db_stats.get("models_by_mode", {})
        
        return stats
    
    async def cleanup_old_versions(self, days: int = 30) -> int:
        """
        Clean up model versions older than specified days
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of versions deleted
        """
        deleted_count = 0
        
        if self.db_handler and hasattr(self.db_handler, 'get_old_versions'):
            old_versions = await self.db_handler.get_old_versions(days=days)
            
            for version in old_versions:
                try:
                    # Delete from storage
                    await self.storage_handler.delete(version["storage_path"])
                    
                    # Update state in database
                    await self.db_handler.update_state(
                        model_type=version.get("model_type", "lstm"),
                        version=version["version"],
                        state=ModelState.DELETED
                    )
                    
                    deleted_count += 1
                except Exception:
                    # Log error but continue
                    pass
        
        return deleted_count
    
    async def track_performance(self, model_id: str, metrics: Dict[str, Any]):
        """Track model performance metrics"""
        if self.db_handler:
            await self.db_handler.track_performance(model_id=model_id, metrics=metrics)
    
    async def save_model_with_version_bump(
        self,
        model_data: bytes,
        model_type: str,
        bump_type: str,
        mode: str = "analysis",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: PreservationPriority = PreservationPriority.NORMAL
    ) -> str:
        """
        Save a model with explicit version bump type.
        
        Args:
            model_data: Serialized model data
            model_type: Type of model
            bump_type: Version bump type ("patch", "minor", "major")
            mode: Operational mode
            tags: Optional tags
            metadata: Optional metadata
            priority: Preservation priority
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        return await self.save_model(
            model_data=model_data,
            model_type=model_type,
            mode=mode,
            tags=tags,
            metadata=metadata,
            priority=priority,
            increment_type=bump_type
        )
    
    async def get_latest_version(
        self,
        model_type: str,
        mode: Optional[str] = None,
        branch: str = "main",
        include_prerelease: bool = False
    ) -> Optional[str]:
        """
        Get the latest version for a model type.
        
        Args:
            model_type: Type of model
            mode: Optional mode filter
            branch: Branch to get latest version from
            include_prerelease: Whether to include prerelease versions
            
        Returns:
            Latest version string or None if no versions exist
        """
        if not self.db_handler:
            return None
        
        try:
            versions = await self.db_handler.get_versions(model_type=model_type, branch=branch)
            
            # Filter by mode if specified
            if mode:
                versions = [v for v in versions if v.get("mode") == mode]
            
            version_strings = [v["version"] for v in versions]
            
            if version_strings:
                latest = find_latest_version(version_strings, include_prerelease=include_prerelease)
                return str(latest) if latest else None
            
            return None
            
        except Exception:
            return None
    
    async def get_stable_versions(
        self,
        model_type: str,
        mode: Optional[str] = None
    ) -> List[str]:
        """
        Get all stable versions for a model type.
        
        Args:
            model_type: Type of model
            mode: Optional mode filter
            
        Returns:
            List of stable version strings sorted in descending order
        """
        if not self.db_handler:
            return []
        
        try:
            versions = await self.db_handler.get_versions(model_type=model_type)
            
            # Filter by mode if specified
            if mode:
                versions = [v for v in versions if v.get("mode") == mode]
            
            stable_versions = []
            for v in versions:
                try:
                    semantic_ver = SemanticVersion(v["version"])
                    if semantic_ver.is_stable():
                        stable_versions.append(semantic_ver)
                except Exception:
                    continue  # Skip invalid versions
            
            # Sort in descending order (newest first)
            stable_versions.sort(reverse=True)
            return [str(v) for v in stable_versions]
            
        except Exception:
            return []
    
    async def compare_model_versions(
        self,
        model_type: str,
        version1: str,
        version2: str
    ) -> int:
        """
        Compare two model versions.
        
        Args:
            model_type: Type of model (for context)
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        try:
            v1 = SemanticVersion(version1)
            v2 = SemanticVersion(version2)
            
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
                
        except Exception as e:
            raise VersionError(f"Failed to compare versions {version1} and {version2}: {str(e)}")
    
    async def get_version_history(
        self,
        model_type: str,
        mode: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a model type.
        
        Args:
            model_type: Type of model
            mode: Optional mode filter
            limit: Maximum number of versions to return
            
        Returns:
            List of version information sorted by version (newest first)
        """
        if not self.db_handler:
            return []
        
        try:
            versions = await self.db_handler.get_versions(model_type=model_type)
            
            # Filter by mode if specified
            if mode:
                versions = [v for v in versions if v.get("mode") == mode]
            
            # Add semantic version objects for sorting
            version_data = []
            for v in versions:
                try:
                    semantic_ver = SemanticVersion(v["version"])
                    version_data.append({
                        **v,
                        "_semantic_version": semantic_ver
                    })
                except Exception:
                    continue  # Skip invalid versions
            
            # Sort by semantic version (newest first)
            version_data.sort(key=lambda x: x["_semantic_version"], reverse=True)
            
            # Remove internal semantic version and apply limit
            result = []
            for v in version_data[:limit]:
                v_copy = dict(v)
                del v_copy["_semantic_version"]
                result.append(v_copy)
            
            return result
            
        except Exception:
            return []
    
    async def _generate_next_version(
        self, 
        model_type: str, 
        increment_type: str = "patch",
        branch: str = "main"
    ) -> str:
        """
        Generate next semantic version number for a model type.
        
        Args:
            model_type: Type of model to generate version for
            increment_type: Type of increment ("patch", "minor", "major")
            branch: Branch to generate version for
            
        Returns:
            Next semantic version string
        """
        if self.db_handler:
            # Get existing versions from database for the specific branch
            versions = await self.db_handler.get_versions(model_type=model_type, branch=branch)
            version_strings = [v["version"] for v in versions]
            
            if version_strings:
                # Find latest version using semantic versioning
                latest_version = find_latest_version(version_strings, include_prerelease=False)
                
                if latest_version:
                    # Increment based on type
                    if increment_type == "major":
                        next_version = latest_version.bump_major()
                    elif increment_type == "minor":
                        next_version = latest_version.bump_minor()
                    else:  # patch (default)
                        next_version = latest_version.bump_patch()
                    
                    return f"v{str(next_version)}"
                else:
                    # No valid semantic versions found, start fresh
                    return "v1.0.0"
            else:
                # No versions exist, start with 1.0.0
                return "v1.0.0"
        else:
            # Simple versioning without database
            return "v1.0.0"
    
    async def _enforce_version_limit(self, model_type: str, mode: str):
        """Enforce maximum version limit per model"""
        if self.db_handler:
            # Get all versions for this model type and mode
            versions = await self.db_handler.get_versions(model_type=model_type)
            
            # Filter by mode if isolation is enabled
            if self.config.mode_isolation:
                versions = [v for v in versions if v.get("mode") == mode]
            
            if len(versions) >= self.config.max_versions_per_model:
                # Sort by creation date and delete oldest
                versions.sort(key=lambda v: v["created_at"])
                versions_to_delete = versions[:len(versions) - self.config.max_versions_per_model + 1]
                
                for version in versions_to_delete:
                    try:
                        # Get full metadata to find storage path
                        metadata = await self.db_handler.get_metadata(
                            model_type=model_type,
                            version=version["version"],
                            mode=mode if self.config.mode_isolation else None
                        )
                        if metadata and "storage_path" in metadata:
                            # Delete from storage
                            await self.storage_handler.delete(metadata["storage_path"])
                            
                            # Update state in database
                            await self.db_handler.update_state(
                                model_type=model_type,
                                version=version["version"],
                                state=ModelState.DELETED
                            )
                    except Exception:
                        # Log error but continue
                        pass
    
    async def _background_backup_loop(self):
        """Background task for automatic backups"""
        while self._is_running:
            try:
                # Wait for backup interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.backup_interval_hours * 3600
                )
                
                if self._shutdown_event.is_set():
                    break
                    
            except asyncio.TimeoutError:
                # Perform backup
                if self._is_running and self.db_handler:
                    active_models = await self.db_handler.get_active_models()
                    
                    for model in active_models:
                        try:
                            # Load and re-save model
                            model_data = await self.storage_handler.load(model["storage_path"])
                            
                            await self.save_model(
                                model_data=model_data,
                                model_type=model["model_type"],
                                version=f"{model['version']}-backup-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                mode=model["mode"],
                                tags=["auto_backup"],
                                priority=PreservationPriority.NORMAL
                            )
                        except Exception:
                            # Log error but continue
                            pass
    
    # =============================================================================
    # TAG OPERATIONS
    # =============================================================================
    
    async def tag_model(
        self,
        model_type: str,
        version: str,
        tag_name: str,
        branch: str = "main",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Tag a model version with a friendly name
        
        Args:
            model_type: Type of model to tag
            version: Version to tag
            tag_name: Name of the tag
            branch: Branch where the model exists
            description: Optional description
            metadata: Optional additional metadata
            
        Returns:
            Tag ID of created/updated tag
            
        Raises:
            PreservationError: If tagging fails
            ValueError: If tag name is invalid or version doesn't exist
        """
        try:
            if not self.db_handler:
                raise PreservationError("Database handler required for tagging")
            
            # Import here to avoid circular imports
            from .base import ModelTag, is_valid_tag_name
            
            # Validate tag name
            if not is_valid_tag_name(tag_name):
                raise ValueError(f"Invalid tag name: {tag_name}")
            
            # Create ModelTag object
            tag = ModelTag(
                tag_name=tag_name,
                model_type=model_type,
                version=version,
                branch=branch,
                created_at=datetime.now(),
                description=description,
                metadata=metadata or {}
            )
            
            # Save tag to database
            tag_id = await self.db_handler.save_tag(tag)
            
            # Record tagging event
            await self.db_handler.record_event(
                model_id=f"{model_type}-{version}",
                event_type="tagged",
                details={
                    "tag_name": tag_name,
                    "description": description
                }
            )
            
            return tag_id
            
        except Exception as e:
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Failed to tag model: {str(e)}")
    
    async def resolve_tag(
        self,
        model_type: str,
        tag_name: str,
        branch: str = "main"
    ) -> Optional[str]:
        """
        Resolve a tag name to its version
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to resolve
            branch: Branch where tag exists
            
        Returns:
            Version string or None if tag not found
        """
        if not self.db_handler:
            return None
        
        try:
            return await self.db_handler.resolve_tag_to_version(model_type, tag_name, branch=branch)
        except Exception:
            return None
    
    async def move_tag(
        self,
        model_type: str,
        tag_name: str,
        new_version: str,
        description: Optional[str] = None
    ) -> None:
        """
        Move a tag to point to a different version
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to move
            new_version: New version to point to
            description: Optional new description
            
        Raises:
            PreservationError: If move fails
            ValueError: If tag or version doesn't exist
        """
        try:
            if not self.db_handler:
                raise PreservationError("Database handler required for tag operations")
            
            # Update tag in database
            await self.db_handler.update_tag(
                model_type=model_type,
                tag_name=tag_name,
                new_version=new_version,
                description=description
            )
            
            # Record tag move event
            await self.db_handler.record_event(
                model_id=f"{model_type}-{new_version}",
                event_type="tag_moved",
                details={
                    "tag_name": tag_name,
                    "new_version": new_version,
                    "description": description
                }
            )
            
        except Exception as e:
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Failed to move tag: {str(e)}")
    
    async def delete_tag(
        self,
        model_type: str,
        tag_name: str
    ) -> None:
        """
        Delete a tag
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to delete
            
        Raises:
            PreservationError: If deletion fails
            ValueError: If tag doesn't exist
        """
        try:
            if not self.db_handler:
                raise PreservationError("Database handler required for tag operations")
            
            # Import here to avoid circular imports
            from .base import StandardTags
            
            # Prevent deletion of standard tags
            if StandardTags.is_standard_tag(tag_name):
                raise ValueError(f"Cannot delete standard tag: {tag_name}")
            
            # Delete tag from database
            await self.db_handler.delete_tag(model_type, tag_name)
            
            # Record tag deletion event
            await self.db_handler.record_event(
                model_id=f"{model_type}-unknown",
                event_type="tag_deleted",
                details={"tag_name": tag_name}
            )
            
        except Exception as e:
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Failed to delete tag: {str(e)}")
    
    async def list_model_tags(
        self,
        model_type: str
    ) -> List[Dict[str, Any]]:
        """
        List all tags for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            List of tag information dictionaries
        """
        if not self.db_handler:
            return []
        
        try:
            return await self.db_handler.list_tags(model_type)
        except Exception:
            return []
    
    async def get_tag_info(
        self,
        model_type: str,
        tag_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tag
        
        Args:
            model_type: Type of model
            tag_name: Name of tag
            
        Returns:
            Tag information dictionary or None if not found
        """
        if not self.db_handler:
            return None
        
        try:
            return await self.db_handler.get_tag(model_type, tag_name)
        except Exception:
            return None
    
    async def get_tag_history(
        self,
        model_type: str,
        tag_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get tag change history
        
        Args:
            model_type: Type of model
            tag_name: Optional specific tag name
            limit: Maximum number of history entries
            
        Returns:
            List of tag history entries
        """
        if not self.db_handler:
            return []
        
        try:
            return await self.db_handler.get_tag_history(
                model_type=model_type,
                tag_name=tag_name,
                limit=limit
            )
        except Exception:
            return []
    
    async def update_stable_tag(
        self,
        model_type: str,
        mode: str = "analysis"
    ) -> Optional[str]:
        """
        Update the 'stable' tag to point to the latest stable version
        
        Args:
            model_type: Type of model
            mode: Operational mode
            
        Returns:
            Version that stable tag now points to, or None if no stable versions
        """
        if not self.db_handler:
            return None
        
        try:
            # Get all stable versions
            stable_versions = await self.get_stable_versions(model_type, mode)
            
            if not stable_versions:
                return None
            
            # Latest stable is first in the list (sorted descending)
            latest_stable = stable_versions[0]
            
            # Import here to avoid circular imports
            from .base import StandardTags
            
            # Update stable tag
            await self.move_tag(
                model_type=model_type,
                tag_name=StandardTags.STABLE,
                new_version=latest_stable,
                description=f"Latest stable version of {model_type}"
            )
            
            return latest_stable
            
        except Exception:
            return None
    
    # =============================================================================
    # BRANCH SUPPORT METHODS
    # =============================================================================
    
    async def create_branch(
        self,
        branch_name: str,
        source_branch: str = "main",
        description: Optional[str] = None
    ) -> str:
        """
        Create a new branch for experimental model development
        
        Args:
            branch_name: Name of the new branch
            source_branch: Source branch to copy from (default: main)
            description: Optional description of the branch
            
        Returns:
            Branch ID from database
            
        Raises:
            ValueError: If branch name is invalid or already exists
        """
        from .base import validate_branch_name
        
        # Validate branch name
        if not validate_branch_name(branch_name):
            raise ValueError(f"Invalid branch name: {branch_name}")
        
        if not self.db_handler:
            raise PreservationError("Database handler not initialized")
        
        try:
            # Check if branch already exists
            if await self.db_handler.branch_exists(branch_name):
                raise ValueError(f"Branch '{branch_name}' already exists")
            
            # Create branch in database
            branch_id = await self.db_handler.create_branch(
                branch_name=branch_name,
                source_branch=source_branch,
                description=description
            )
            
            # Record branch creation event
            await self.db_handler.record_event(
                model_id=f"branch-{branch_name}",
                event_type="branch_created",
                details={
                    "branch_name": branch_name,
                    "source_branch": source_branch,
                    "description": description
                }
            )
            
            return branch_id
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise PreservationError(f"Failed to create branch: {str(e)}")
    
    async def list_branches(self) -> List[Dict[str, Any]]:
        """
        List all branches with their metadata
        
        Returns:
            List of branch information dictionaries
        """
        if not self.db_handler:
            return []
        
        try:
            return await self.db_handler.list_branches()
        except Exception:
            return []
    
    async def delete_branch(
        self,
        branch_name: str,
        force: bool = False
    ) -> None:
        """
        Delete a branch
        
        Args:
            branch_name: Name of branch to delete
            force: Force deletion even if branch contains models
            
        Raises:
            ValueError: If branch doesn't exist or cannot be deleted
        """
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")
        
        if not self.db_handler:
            raise PreservationError("Database handler not initialized")
        
        try:
            # Check if branch exists
            if not await self.db_handler.branch_exists(branch_name):
                raise ValueError(f"Branch '{branch_name}' does not exist")
            
            # Check if branch has models
            model_count = await self.db_handler.get_branch_model_count(branch_name)
            if model_count > 0 and not force:
                raise ValueError(f"Cannot delete branch '{branch_name}' containing {model_count} models. Use force=True to override.")
            
            # Delete branch
            await self.db_handler.delete_branch(branch_name)
            
            # Record deletion event
            await self.db_handler.record_event(
                model_id=f"branch-{branch_name}",
                event_type="branch_deleted",
                details={
                    "branch_name": branch_name,
                    "forced": force,
                    "model_count": model_count
                }
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise PreservationError(f"Failed to delete branch: {str(e)}")
    
    async def promote_model(
        self,
        model_type: str,
        version: str,
        source_branch: str,
        target_branch: str,
        mode: Optional[str] = None
    ) -> str:
        """
        Promote a model from one branch to another
        
        Args:
            model_type: Type of model
            version: Version to promote
            source_branch: Source branch
            target_branch: Target branch
            mode: Optional mode context
            
        Returns:
            New model ID in target branch
            
        Raises:
            ValueError: If model or branches don't exist
        """
        if not self.db_handler:
            raise PreservationError("Database handler not initialized")
        
        try:
            # Validate branches exist
            if not await self.db_handler.branch_exists(source_branch):
                raise ValueError(f"Source branch '{source_branch}' does not exist")
            
            if not await self.db_handler.branch_exists(target_branch):
                raise ValueError(f"Target branch '{target_branch}' does not exist")
            
            # Load model from source branch
            source_metadata = await self.db_handler.get_metadata(
                model_type=model_type,
                version=version,
                branch=source_branch,
                mode=mode
            )
            
            if not source_metadata:
                raise ValueError(f"Model {model_type} {version} not found in branch {source_branch}")
            
            # Load model data
            model_data = await self.storage_handler.load(source_metadata["storage_path"])
            
            # Save to target branch with same version
            new_model_id = await self.save_model(
                model_data=model_data,
                model_type=model_type,
                version=version,
                branch=target_branch,
                mode=mode,
                tags=source_metadata.get("tags", []),
                metadata=source_metadata.get("metadata", {})
            )
            
            # Record promotion event
            await self.db_handler.record_event(
                model_id=new_model_id,
                event_type="model_promoted",
                details={
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "original_model_id": source_metadata["model_id"]
                }
            )
            
            return new_model_id
            
        except Exception as e:
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Failed to promote model: {str(e)}")
    
    async def list_models(self, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models, optionally filtered by branch
        
        Args:
            branch: Optional branch to filter by
            
        Returns:
            List of model metadata
        """
        if not self.db_handler:
            return []
        
        try:
            return await self.db_handler.list_models(branch=branch)
        except Exception:
            return []