"""
Database Handler for Model Preservation

Handles all database operations for model preservation metadata, events, and performance tracking.
Integrates with the existing database schema created in migration 006_model_preservation_schema.sql.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
import json

from ..utils.database import (
    get_database_connection,
    execute_query,
    DatabaseError,
    DatabaseQueryError
)
from .base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    validate_version_format
)

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Database handler for model preservation operations
    
    Provides comprehensive database operations for:
    - Model metadata storage and retrieval
    - Version history tracking
    - Performance metrics tracking
    - Event logging
    - Statistics and monitoring
    """
    
    def __init__(self):
        """Initialize database handler"""
        self._initialized = False
        self._connection_tested = False
    
    async def initialize(self) -> None:
        """Initialize database handler and verify schema"""
        if self._initialized:
            return
        
        try:
            # Test database connection
            async with get_database_connection() as conn:
                await conn.fetchval("SELECT 1")
                self._connection_tested = True
            
            # Verify preservation schema exists
            await self._verify_schema()
            
            self._initialized = True
            logger.info("Database handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database handler: {e}")
            raise PreservationError(f"Database initialization failed: {e}")
    
    async def _verify_schema(self) -> None:
        """Verify that required database schema exists"""
        required_tables = [
            'model_preservation_metadata',
            'model_version_history',
            'model_preservation_events',
            'model_performance_tracking'
        ]
        
        try:
            async with get_database_connection() as conn:
                for table in required_tables:
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = $1
                        )
                    """, table)
                    
                    if not exists:
                        raise PreservationError(f"Required table {table} does not exist")
                        
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            raise PreservationError(f"Schema verification failed: {e}")
    
    async def save_metadata(self, metadata: ModelMetadata) -> str:
        """
        Save model metadata to database
        
        Args:
            metadata: ModelMetadata object to save
            
        Returns:
            Model ID of saved metadata
            
        Raises:
            PreservationError: If save operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate inputs
            if not validate_version_format(metadata.version):
                raise ValueError(f"Invalid version format: {metadata.version}")
            
            if metadata.model_type not in ['lstm', 'dqn', 'ensemble', 'custom']:
                raise ValueError(f"Invalid model type: {metadata.model_type}")
            
            if metadata.mode not in ['analysis', 'simulation', 'live']:
                raise ValueError(f"Invalid mode: {metadata.mode}")
            
            # Generate preservation ID
            preservation_id = f"{metadata.model_type}-{metadata.version}-{metadata.mode}-{uuid.uuid4().hex[:8]}"
            
            async with get_database_connection() as conn:
                # Insert metadata
                query = """
                    INSERT INTO model_preservation_metadata (
                        preservation_id, model_id, model_type, version, mode,
                        created_at, preserved_at, checksum, size_bytes, gcs_path,
                        performance_metrics, training_info, preservation_reason,
                        priority, tags, metadata, state
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                    ) RETURNING preservation_id
                """
                
                result = await conn.fetchval(
                    query,
                    preservation_id,
                    metadata.model_id,
                    metadata.model_type,
                    metadata.version,
                    metadata.mode,
                    metadata.created_at,
                    metadata.preserved_at or datetime.now(timezone.utc),
                    metadata.checksum,
                    metadata.file_size_bytes,
                    metadata.file_path,
                    json.dumps(metadata.performance_metrics) if metadata.performance_metrics else None,
                    json.dumps({"compression_type": metadata.compression_type}) if metadata.compression_type else None,
                    metadata.description,
                    metadata.preservation_priority.value,
                    metadata.tags,
                    json.dumps(metadata.metadata) if metadata.metadata else None,
                    metadata.state.value
                )
                
                # Update version history
                await self._update_version_history(
                    conn,
                    metadata.model_type,
                    metadata.mode,
                    metadata.version,
                    preservation_id
                )
                
                logger.info(f"Saved model metadata: {preservation_id}")
                return metadata.model_id
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Database save failed: {e}")
    
    async def get_metadata(
        self,
        model_type: str,
        version: Optional[str] = None,
        mode: str = "analysis",
        branch: str = "main"
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata from database
        
        Args:
            model_type: Type of model to retrieve
            version: Specific version (latest if not specified)
            mode: Operational mode
            branch: Branch to get model from
            
        Returns:
            Metadata dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                if version:
                    # Get specific version
                    query = """
                        SELECT 
                            preservation_id, model_id, model_type, version, mode, branch,
                            created_at, preserved_at, checksum, size_bytes, gcs_path,
                            performance_metrics, training_info, preservation_reason,
                            priority, tags, metadata, state, updated_at
                        FROM model_preservation_metadata
                        WHERE model_type = $1 AND version = $2 AND mode = $3 AND branch = $4
                        AND state != 'deleted'
                        ORDER BY preserved_at DESC
                        LIMIT 1
                    """
                    row = await conn.fetchrow(query, model_type, version, mode, branch)
                else:
                    # Get latest version
                    query = """
                        SELECT 
                            preservation_id, model_id, model_type, version, mode, branch,
                            created_at, preserved_at, checksum, size_bytes, gcs_path,
                            performance_metrics, training_info, preservation_reason,
                            priority, tags, metadata, state, updated_at
                        FROM model_preservation_metadata
                        WHERE model_type = $1 AND mode = $2 AND branch = $3
                        AND state != 'deleted'
                        ORDER BY preserved_at DESC
                        LIMIT 1
                    """
                    row = await conn.fetchrow(query, model_type, mode, branch)
                
                if not row:
                    return None
                
                # Convert row to dictionary
                return {
                    "preservation_id": row["preservation_id"],
                    "model_id": row["model_id"],
                    "model_type": row["model_type"],
                    "version": row["version"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "preserved_at": row["preserved_at"],
                    "checksum": row["checksum"],
                    "file_size_bytes": row["size_bytes"],
                    "storage_path": row["gcs_path"],
                    "performance_metrics": json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
                    "training_info": json.loads(row["training_info"]) if row["training_info"] else {},
                    "preservation_reason": row["preservation_reason"],
                    "priority": row["priority"],
                    "tags": row["tags"] or [],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "state": row["state"],
                    "updated_at": row["updated_at"]
                }
                
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            raise PreservationError(f"Database query failed: {e}")
    
    async def get_versions(
        self,
        model_type: str,
        mode: Optional[str] = None,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Get all versions for a model type
        
        Args:
            model_type: Type of model
            mode: Optional mode filter
            branch: Branch to get versions from
            
        Returns:
            List of version information dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                if mode:
                    query = """
                        SELECT version, mode, branch, preserved_at, state, priority
                        FROM model_preservation_metadata
                        WHERE model_type = $1 AND mode = $2 AND branch = $3
                        AND state != 'deleted'
                        ORDER BY preserved_at DESC
                    """
                    rows = await conn.fetch(query, model_type, mode, branch)
                else:
                    query = """
                        SELECT version, mode, branch, preserved_at, state, priority
                        FROM model_preservation_metadata
                        WHERE model_type = $1 AND branch = $2
                        AND state != 'deleted'
                        ORDER BY preserved_at DESC
                    """
                    rows = await conn.fetch(query, model_type, branch)
                
                return [
                    {
                        "version": row["version"],
                        "mode": row["mode"],
                        "branch": row["branch"],
                        "created_at": row["preserved_at"],
                        "state": row["state"],
                        "priority": row["priority"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get versions: {e}")
            raise PreservationError(f"Database query failed: {e}")
    
    async def record_event(
        self,
        model_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> str:
        """
        Record a preservation event
        
        Args:
            model_id: Model ID associated with event
            event_type: Type of event (saved, loaded, etc.)
            details: Optional event details
            success: Whether the event was successful
            
        Returns:
            Event ID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get preservation info from model_id
            async with get_database_connection() as conn:
                # Extract model info from model_id
                parts = model_id.split('-')
                if len(parts) >= 3:
                    model_type = parts[0]
                    version = f"{parts[1]}-{parts[2]}-{parts[3]}"  # Handle v1.0.0 format
                    mode = "analysis"  # Default
                else:
                    model_type = "unknown"
                    version = "unknown"
                    mode = "analysis"
                
                # Find preservation_id
                preservation_id = await conn.fetchval("""
                    SELECT preservation_id FROM model_preservation_metadata
                    WHERE model_id = $1
                    ORDER BY preserved_at DESC
                    LIMIT 1
                """, model_id)
                
                if not preservation_id:
                    # Create a placeholder preservation_id for the event
                    preservation_id = f"{model_type}-{version}-{uuid.uuid4().hex[:8]}"
                
                event_id = uuid.uuid4()
                query = """
                    INSERT INTO model_preservation_events (
                        event_id, event_type, preservation_id, model_type, mode,
                        event_data, success, occurred_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    ) RETURNING event_id
                """
                
                result = await conn.fetchval(
                    query,
                    event_id,
                    event_type,
                    preservation_id,
                    model_type,
                    mode,
                    json.dumps(details) if details else None,
                    success,
                    datetime.now(timezone.utc)
                )
                
                logger.debug(f"Recorded event: {event_type} for model {model_id}")
                return str(result)
                
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            raise PreservationError(f"Event recording failed: {e}")
    
    async def get_active_models(self) -> List[Dict[str, Any]]:
        """
        Get all active models
        
        Returns:
            List of active model dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                query = """
                    SELECT 
                        model_id, model_type, version, mode, gcs_path,
                        preserved_at, size_bytes, priority, state
                    FROM model_preservation_metadata
                    WHERE state = 'active'
                    ORDER BY preserved_at DESC
                """
                
                rows = await conn.fetch(query)
                
                return [
                    {
                        "model_id": row["model_id"],
                        "model_type": row["model_type"],
                        "version": row["version"],
                        "mode": row["mode"],
                        "storage_path": row["gcs_path"],
                        "preserved_at": row["preserved_at"],
                        "size_bytes": row["size_bytes"],
                        "priority": row["priority"],
                        "state": row["state"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get active models: {e}")
            raise PreservationError(f"Database query failed: {e}")
    
    async def get_preservation_stats(self) -> Dict[str, Any]:
        """
        Get preservation system statistics
        
        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                # Get basic counts
                total_models = await conn.fetchval("""
                    SELECT COUNT(*) FROM model_preservation_metadata
                    WHERE state != 'deleted'
                """) or 0
                
                active_models = await conn.fetchval("""
                    SELECT COUNT(*) FROM model_preservation_metadata
                    WHERE state = 'active'
                """) or 0
                
                total_size_bytes = await conn.fetchval("""
                    SELECT COALESCE(SUM(size_bytes), 0) FROM model_preservation_metadata
                    WHERE state != 'deleted'
                """) or 0
                
                # Get models by type
                type_stats = await conn.fetch("""
                    SELECT model_type, COUNT(*) as count
                    FROM model_preservation_metadata
                    WHERE state != 'deleted'
                    GROUP BY model_type
                """)
                
                models_by_type = {row["model_type"]: row["count"] for row in type_stats}
                
                # Get models by mode
                mode_stats = await conn.fetch("""
                    SELECT mode, COUNT(*) as count
                    FROM model_preservation_metadata
                    WHERE state != 'deleted'
                    GROUP BY mode
                """)
                
                models_by_mode = {row["mode"]: row["count"] for row in mode_stats}
                
                return {
                    "total_models": total_models,
                    "active_models": active_models,
                    "total_size_bytes": total_size_bytes,
                    "models_by_type": models_by_type,
                    "models_by_mode": models_by_mode
                }
                
        except Exception as e:
            logger.error(f"Failed to get preservation stats: {e}")
            raise PreservationError(f"Statistics query failed: {e}")
    
    async def track_performance(
        self,
        model_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Track model performance metrics
        
        Args:
            model_id: Model ID to track performance for
            metrics: Performance metrics dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                # Get preservation_id from model_id
                preservation_id = await conn.fetchval("""
                    SELECT preservation_id FROM model_preservation_metadata
                    WHERE model_id = $1
                    ORDER BY preserved_at DESC
                    LIMIT 1
                """, model_id)
                
                if not preservation_id:
                    logger.warning(f"No preservation record found for model {model_id}")
                    return
                
                # Insert performance tracking record
                query = """
                    INSERT INTO model_performance_tracking (
                        preservation_id, accuracy, loss, prediction_count,
                        successful_predictions, total_return, sharpe_ratio,
                        max_drawdown, win_rate, inference_time_ms, memory_usage_mb,
                        period_start, period_end, custom_metrics
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    )
                """
                
                now = datetime.now(timezone.utc)
                
                await conn.execute(
                    query,
                    preservation_id,
                    metrics.get("accuracy"),
                    metrics.get("loss"),
                    metrics.get("prediction_count"),
                    metrics.get("successful_predictions"),
                    metrics.get("total_return"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("win_rate"),
                    metrics.get("inference_time_ms"),
                    metrics.get("memory_usage_mb"),
                    now - timedelta(hours=1),  # Default period start
                    now,  # Period end
                    json.dumps(metrics) if metrics else None
                )
                
                logger.debug(f"Tracked performance for model {model_id}")
                
        except Exception as e:
            logger.error(f"Failed to track performance: {e}")
            raise PreservationError(f"Performance tracking failed: {e}")
    
    async def update_state(
        self,
        model_type: str,
        version: str,
        state: ModelState,
        mode: Optional[str] = None
    ) -> None:
        """
        Update model state
        
        Args:
            model_type: Type of model
            version: Version to update
            state: New state
            mode: Optional mode filter
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                if mode:
                    query = """
                        UPDATE model_preservation_metadata
                        SET state = $1, updated_at = NOW()
                        WHERE model_type = $2 AND version = $3 AND mode = $4
                    """
                    await conn.execute(query, state.value, model_type, version, mode)
                else:
                    query = """
                        UPDATE model_preservation_metadata
                        SET state = $1, updated_at = NOW()
                        WHERE model_type = $2 AND version = $3
                    """
                    await conn.execute(query, state.value, model_type, version)
                
                logger.debug(f"Updated state for {model_type} {version} to {state.value}")
                
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            raise PreservationError(f"State update failed: {e}")
    
    async def get_old_versions(self, days: int) -> List[Dict[str, Any]]:
        """
        Get old model versions for cleanup
        
        Args:
            days: Age threshold in days
            
        Returns:
            List of old version dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                query = """
                    SELECT 
                        model_id, model_type, version, mode, gcs_path,
                        preserved_at, size_bytes, state
                    FROM model_preservation_metadata
                    WHERE preserved_at < $1
                    AND state != 'deleted'
                    AND priority != 'critical'
                    ORDER BY preserved_at ASC
                """
                
                rows = await conn.fetch(query, cutoff_date)
                
                return [
                    {
                        "model_id": row["model_id"],
                        "model_type": row["model_type"],
                        "version": row["version"],
                        "mode": row["mode"],
                        "storage_path": row["gcs_path"],
                        "created_at": row["preserved_at"],
                        "size_bytes": row["size_bytes"],
                        "state": row["state"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get old versions: {e}")
            raise PreservationError(f"Old versions query failed: {e}")
    
    async def _update_version_history(
        self,
        conn,
        model_type: str,
        mode: str,
        version: str,
        preservation_id: str
    ) -> None:
        """Update version history table"""
        try:
            # Parse version components
            version_parts = version.replace('v', '').split('.')
            major = int(version_parts[0]) if len(version_parts) > 0 else 1
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 else 0
            
            # Get current version history
            current_row = await conn.fetchrow("""
                SELECT current_preservation_id, current_version
                FROM model_version_history
                WHERE model_type = $1 AND mode = $2
            """, model_type, mode)
            
            if current_row:
                # Update existing record
                await conn.execute("""
                    UPDATE model_version_history
                    SET 
                        previous_version = current_version,
                        previous_preservation_id = current_preservation_id,
                        current_version = $1,
                        current_preservation_id = $2,
                        major_version = $3,
                        minor_version = $4,
                        patch_version = $5,
                        changed_at = NOW()
                    WHERE model_type = $6 AND mode = $7
                """, version, preservation_id, major, minor, patch, model_type, mode)
            else:
                # Insert new record
                await conn.execute("""
                    INSERT INTO model_version_history (
                        model_type, mode, current_version, current_preservation_id,
                        major_version, minor_version, patch_version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, model_type, mode, version, preservation_id, major, minor, patch)
                
        except Exception as e:
            logger.error(f"Failed to update version history: {e}")
            # Don't raise error - this is not critical for the main operation
    
    # =============================================================================
    # TAG OPERATIONS
    # =============================================================================
    
    async def save_tag(self, tag: 'ModelTag') -> str:
        """
        Save a model tag to database
        
        Args:
            tag: ModelTag object to save
            
        Returns:
            Tag ID of saved tag
            
        Raises:
            PreservationError: If save operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Import here to avoid circular imports
            from .base import ModelTag
            
            if not isinstance(tag, ModelTag):
                raise ValueError(f"Expected ModelTag object, got {type(tag)}")
            
            async with get_database_connection() as conn:
                # Check if preservation_id exists
                preservation_exists = await conn.fetchval("""
                    SELECT preservation_id FROM model_preservation_metadata
                    WHERE model_type = $1 AND version = $2
                    LIMIT 1
                """, tag.model_type, tag.version)
                
                if not preservation_exists:
                    raise ValueError(f"No preserved model found for {tag.model_type} version {tag.version}")
                
                # Insert or update tag
                query = """
                    INSERT INTO model_tags (
                        tag_id, tag_name, model_type, version, preservation_id,
                        created_at, updated_at, description, metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    )
                    ON CONFLICT (tag_name, model_type)
                    DO UPDATE SET
                        version = EXCLUDED.version,
                        preservation_id = EXCLUDED.preservation_id,
                        updated_at = EXCLUDED.updated_at,
                        description = EXCLUDED.description,
                        metadata = EXCLUDED.metadata
                    RETURNING tag_id
                """
                
                result = await conn.fetchval(
                    query,
                    tag.tag_id,
                    tag.tag_name,
                    tag.model_type,
                    tag.version,
                    preservation_exists,
                    tag.created_at,
                    tag.updated_at or tag.created_at,
                    tag.description,
                    json.dumps(tag.metadata) if tag.metadata else None
                )
                
                logger.info(f"Saved tag: {tag.tag_name} -> {tag.model_type} {tag.version}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to save tag: {e}")
            if isinstance(e, (ValueError, PreservationError)):
                raise
            raise PreservationError(f"Tag save failed: {e}")
    
    async def get_tag(
        self,
        model_type: str,
        tag_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get tag information from database
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to retrieve
            
        Returns:
            Tag information dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                query = """
                    SELECT 
                        tag_id, tag_name, model_type, version, preservation_id,
                        created_at, updated_at, description, metadata
                    FROM model_tags
                    WHERE model_type = $1 AND tag_name = $2
                """
                
                row = await conn.fetchrow(query, model_type, tag_name)
                
                if not row:
                    return None
                
                return {
                    "tag_id": row["tag_id"],
                    "tag_name": row["tag_name"],
                    "model_type": row["model_type"],
                    "version": row["version"],
                    "preservation_id": row["preservation_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "description": row["description"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
                
        except Exception as e:
            logger.error(f"Failed to get tag: {e}")
            raise PreservationError(f"Tag query failed: {e}")
    
    async def update_tag(
        self,
        model_type: str,
        tag_name: str,
        new_version: str,
        description: Optional[str] = None
    ) -> None:
        """
        Update a tag to point to a new version
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to update
            new_version: New version to point to
            description: Optional new description
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                # Get preservation_id for new version
                preservation_id = await conn.fetchval("""
                    SELECT preservation_id FROM model_preservation_metadata
                    WHERE model_type = $1 AND version = $2
                    LIMIT 1
                """, model_type, new_version)
                
                if not preservation_id:
                    raise ValueError(f"No preserved model found for {model_type} version {new_version}")
                
                # Update tag
                query = """
                    UPDATE model_tags
                    SET version = $1, preservation_id = $2, updated_at = NOW()
                """
                params = [new_version, preservation_id]
                
                if description is not None:
                    query += ", description = $3"
                    params.append(description)
                
                query += " WHERE model_type = $" + str(len(params) + 1) + " AND tag_name = $" + str(len(params) + 2)
                params.extend([model_type, tag_name])
                
                result = await conn.execute(query, *params)
                
                if result == "UPDATE 0":
                    raise ValueError(f"Tag {tag_name} not found for model type {model_type}")
                
                logger.info(f"Updated tag: {tag_name} -> {model_type} {new_version}")
                
        except Exception as e:
            logger.error(f"Failed to update tag: {e}")
            if isinstance(e, ValueError):
                raise
            raise PreservationError(f"Tag update failed: {e}")
    
    async def delete_tag(
        self,
        model_type: str,
        tag_name: str
    ) -> None:
        """
        Delete a tag from database
        
        Args:
            model_type: Type of model
            tag_name: Name of tag to delete
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                result = await conn.execute("""
                    DELETE FROM model_tags
                    WHERE model_type = $1 AND tag_name = $2
                """, model_type, tag_name)
                
                if result == "DELETE 0":
                    raise ValueError(f"Tag {tag_name} not found for model type {model_type}")
                
                logger.info(f"Deleted tag: {tag_name} for {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
            if isinstance(e, ValueError):
                raise
            raise PreservationError(f"Tag deletion failed: {e}")
    
    async def list_tags(
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
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                query = """
                    SELECT 
                        tag_id, tag_name, model_type, version, preservation_id,
                        created_at, updated_at, description, metadata
                    FROM model_tags
                    WHERE model_type = $1
                    ORDER BY created_at DESC
                """
                
                rows = await conn.fetch(query, model_type)
                
                return [
                    {
                        "tag_id": row["tag_id"],
                        "tag_name": row["tag_name"],
                        "model_type": row["model_type"],
                        "version": row["version"],
                        "preservation_id": row["preservation_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "description": row["description"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to list tags: {e}")
            raise PreservationError(f"Tag listing failed: {e}")
    
    async def resolve_tag_to_version(
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
            branch: Branch where tag exists (currently not used - tags are per model_type only)
            
        Returns:
            Version string or None if tag not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                version = await conn.fetchval("""
                    SELECT version FROM model_tags
                    WHERE model_type = $1 AND tag_name = $2
                """, model_type, tag_name)
                
                return version
                
        except Exception as e:
            logger.error(f"Failed to resolve tag: {e}")
            raise PreservationError(f"Tag resolution failed: {e}")
    
    async def get_tag_history(
        self,
        model_type: str,
        tag_name: Optional[str] = None,
        limit: int = 100
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
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                if tag_name:
                    query = """
                        SELECT 
                            tag_id, tag_name, model_type, old_version, new_version,
                            old_preservation_id, new_preservation_id, action, reason,
                            changed_by, changed_at
                        FROM model_tag_history
                        WHERE model_type = $1 AND tag_name = $2
                        ORDER BY changed_at DESC
                        LIMIT $3
                    """
                    rows = await conn.fetch(query, model_type, tag_name, limit)
                else:
                    query = """
                        SELECT 
                            tag_id, tag_name, model_type, old_version, new_version,
                            old_preservation_id, new_preservation_id, action, reason,
                            changed_by, changed_at
                        FROM model_tag_history
                        WHERE model_type = $1
                        ORDER BY changed_at DESC
                        LIMIT $2
                    """
                    rows = await conn.fetch(query, model_type, limit)
                
                return [
                    {
                        "tag_id": row["tag_id"],
                        "tag_name": row["tag_name"],
                        "model_type": row["model_type"],
                        "old_version": row["old_version"],
                        "new_version": row["new_version"],
                        "old_preservation_id": row["old_preservation_id"],
                        "new_preservation_id": row["new_preservation_id"],
                        "action": row["action"],
                        "reason": row["reason"],
                        "changed_by": row["changed_by"],
                        "changed_at": row["changed_at"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get tag history: {e}")
            raise PreservationError(f"Tag history query failed: {e}")
    
    async def update_standard_tags(
        self,
        model_type: str,
        new_version: str,
        preservation_id: str,
        mode: str = "analysis"
    ) -> None:
        """
        Update standard tags (latest, stable) automatically
        
        Args:
            model_type: Type of model
            new_version: New version that was saved
            preservation_id: Preservation ID of the new version
            mode: Operational mode
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                # Call the database function to update standard tags
                await conn.execute("""
                    SELECT update_standard_tags($1, $2, $3, $4)
                """, model_type, new_version, preservation_id, mode)
                
                logger.info(f"Updated standard tags for {model_type} {new_version}")
                
        except Exception as e:
            logger.error(f"Failed to update standard tags: {e}")
            # Don't raise error - this is not critical for the main operation
    
    # =============================================================================
    # BRANCH OPERATIONS
    # =============================================================================
    
    async def branch_exists(self, branch_name: str) -> bool:
        """
        Check if a branch exists
        
        Args:
            branch_name: Name of the branch to check
            
        Returns:
            True if branch exists and is active
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                result = await conn.fetchval("""
                    SELECT branch_exists($1)
                """, branch_name)
                
                return bool(result)
                
        except Exception as e:
            logger.error(f"Failed to check branch existence: {e}")
            # Default to true for main branch
            return branch_name == "main"
    
    async def create_branch(
        self,
        branch_name: str,
        source_branch: str = "main",
        description: Optional[str] = None
    ) -> str:
        """
        Create a new branch
        
        Args:
            branch_name: Name of the new branch
            source_branch: Source branch to copy from
            description: Optional description
            
        Returns:
            Branch ID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                branch_id = await conn.fetchval("""
                    SELECT create_branch($1, $2, $3, $4)
                """, branch_name, source_branch, description, "system")
                
                logger.info(f"Created branch '{branch_name}' from '{source_branch}'")
                return str(branch_id)
                
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            raise PreservationError(f"Failed to create branch: {str(e)}")
    
    async def list_branches(self) -> List[Dict[str, Any]]:
        """
        List all active branches
        
        Returns:
            List of branch information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        branch_name as name,
                        description,
                        created_at,
                        created_by,
                        model_count
                    FROM model_branches
                    WHERE is_active = true
                    ORDER BY 
                        CASE WHEN branch_name = 'main' THEN 0 ELSE 1 END,
                        created_at DESC
                """)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list branches: {e}")
            return []
    
    async def get_branch_model_count(self, branch_name: str) -> int:
        """
        Get count of models in a branch
        
        Args:
            branch_name: Name of the branch
            
        Returns:
            Number of active models in the branch
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                count = await conn.fetchval("""
                    SELECT get_branch_model_count($1)
                """, branch_name)
                
                return int(count or 0)
                
        except Exception as e:
            logger.error(f"Failed to get branch model count: {e}")
            return 0
    
    async def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """
        Delete a branch
        
        Args:
            branch_name: Name of branch to delete
            force: Force deletion even if branch has models
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                await conn.execute("""
                    SELECT delete_branch($1, $2)
                """, branch_name, force)
                
                logger.info(f"Deleted branch '{branch_name}' (force={force})")
                
        except Exception as e:
            logger.error(f"Failed to delete branch: {e}")
            raise PreservationError(f"Failed to delete branch: {str(e)}")
    
    async def list_models(self, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models, optionally filtered by branch
        
        Args:
            branch: Optional branch to filter by
            
        Returns:
            List of model metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with get_database_connection() as conn:
                if branch:
                    query = """
                        SELECT 
                            model_id,
                            model_type,
                            version,
                            branch,
                            mode,
                            state,
                            created_at,
                            file_size_bytes,
                            tags
                        FROM model_preservation_metadata
                        WHERE branch = $1 AND state != 'deleted'
                        ORDER BY created_at DESC
                    """
                    rows = await conn.fetch(query, branch)
                else:
                    query = """
                        SELECT 
                            model_id,
                            model_type,
                            version,
                            branch,
                            mode,
                            state,
                            created_at,
                            file_size_bytes,
                            tags
                        FROM model_preservation_metadata
                        WHERE state != 'deleted'
                        ORDER BY created_at DESC
                    """
                    rows = await conn.fetch(query)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []