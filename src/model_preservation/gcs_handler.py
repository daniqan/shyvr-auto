"""
Google Cloud Storage Handler for Model Preservation

Implements model storage and retrieval using GCS with:
- Versioning and metadata management
- Mode-specific isolation
- Compression and chunking
- Concurrent operations
"""

import asyncio
import gzip
import io
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import structlog
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from concurrent.futures import ThreadPoolExecutor

from .base import (
    BasePreservationHandler, ModelMetadata, ModelType, PreservationConfig,
    PreservationPriority, ModelState, BackupError, RestoreError, VersionError
)
from src.utils.database import get_database_connection, DatabaseError


logger = structlog.get_logger()


class GCSModelPreservationHandler(BasePreservationHandler):
    """Google Cloud Storage implementation of model preservation"""
    
    def __init__(self, config: PreservationConfig):
        super().__init__(config)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(config.gcs_bucket)
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_operations
        )
        
    async def save_model(
        self,
        model_data: bytes,
        metadata: ModelMetadata,
        priority: PreservationPriority = PreservationPriority.NORMAL
    ) -> str:
        """Save model to GCS with metadata"""
        try:
            # Update metadata
            metadata.preserved_at = datetime.utcnow()
            metadata.priority = priority
            metadata.checksum = self._calculate_checksum(model_data)
            metadata.size_bytes = len(model_data)
            
            # Generate preservation ID
            preservation_id = self._generate_preservation_id(metadata)
            
            # Compress if enabled
            if self.config.compression_enabled:
                model_data = await self._compress_data(model_data)
                metadata.metadata['compressed'] = True
                metadata.metadata['compression_level'] = self.config.compression_level
            
            # Determine GCS path
            gcs_path = self._build_gcs_path(metadata, preservation_id)
            
            # Save to GCS
            await self._upload_to_gcs(gcs_path, model_data, metadata)
            
            # Save metadata to database
            async with get_database_connection() as conn:
                await self._save_metadata_to_db(metadata, preservation_id, conn)
            
            # Handle retention policy
            await self._apply_retention_policy(metadata)
            
            self.logger.info(
                "Model saved successfully",
                preservation_id=preservation_id,
                model_type=metadata.model_type.value,
                mode=metadata.mode,
                version=metadata.version,
                size_mb=metadata.size_bytes / 1024 / 1024,
                priority=priority.value
            )
            
            return preservation_id
            
        except Exception as e:
            self.logger.error(
                "Failed to save model",
                error=str(e),
                model_type=metadata.model_type.value
            )
            raise BackupError(f"Failed to save model: {str(e)}")
    
    async def load_model(
        self,
        preservation_id: str
    ) -> Tuple[bytes, ModelMetadata]:
        """Load model from GCS by preservation ID"""
        try:
            # Get metadata from database
            async with get_database_connection() as conn:
                metadata = await self._get_metadata_from_db(preservation_id, conn)
            
            if not metadata:
                raise RestoreError(f"Model not found: {preservation_id}")
            
            # Build GCS path
            gcs_path = self._build_gcs_path(metadata, preservation_id)
            
            # Download from GCS
            model_data = await self._download_from_gcs(gcs_path)
            
            # Decompress if needed
            if metadata.metadata.get('compressed', False):
                model_data = await self._decompress_data(model_data)
            
            # Verify checksum
            calculated_checksum = self._calculate_checksum(model_data)
            if calculated_checksum != metadata.checksum:
                self.logger.error(
                    "Checksum mismatch",
                    preservation_id=preservation_id,
                    expected=metadata.checksum,
                    calculated=calculated_checksum
                )
                raise RestoreError("Model data corrupted: checksum mismatch")
            
            self.logger.info(
                "Model loaded successfully",
                preservation_id=preservation_id,
                model_type=metadata.model_type.value,
                mode=metadata.mode,
                version=metadata.version
            )
            
            return model_data, metadata
            
        except Exception as e:
            self.logger.error(
                "Failed to load model",
                preservation_id=preservation_id,
                error=str(e)
            )
            raise RestoreError(f"Failed to load model: {str(e)}")
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        mode: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """List available models with filters"""
        try:
            query = f"""
                SELECT preservation_id, model_id, model_type, version, mode,
                       created_at, preserved_at, checksum, size_bytes,
                       performance_metrics, training_info, preservation_reason,
                       priority, tags, metadata
                FROM {self.config.metadata_table}
                WHERE state = $1
            """
            params = [ModelState.PRESERVED.value]
            param_count = 1
            
            if model_type:
                param_count += 1
                query += f" AND model_type = ${param_count}"
                params.append(model_type.value)
            
            if mode:
                param_count += 1
                query += f" AND mode = ${param_count}"
                params.append(mode)
            
            if version:
                param_count += 1
                query += f" AND version = ${param_count}"
                params.append(version)
            
            if tags:
                param_count += 1
                query += f" AND tags && ${param_count}"
                params.append(tags)
            
            query += " ORDER BY preserved_at DESC"
            query += f" LIMIT {limit}"
            
            async with get_database_connection() as conn:
                rows = await conn.fetch(query, *params)
            
            models = []
            for row in rows:
                models.append(ModelMetadata(
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
                ))
            
            return models
            
        except Exception as e:
            self.logger.error("Failed to list models", error=str(e))
            raise RestoreError(f"Failed to list models: {str(e)}")
    
    async def delete_model(
        self,
        preservation_id: str,
        soft_delete: bool = True
    ) -> bool:
        """Delete or archive a preserved model"""
        try:
            async with get_database_connection() as conn:
                if soft_delete:
                    # Soft delete - mark as deleted in database
                    query = f"""
                        UPDATE {self.config.metadata_table}
                        SET state = $1, updated_at = $2
                        WHERE preservation_id = $3
                    """
                    await conn.execute(
                        query,
                        ModelState.DELETED.value,
                        datetime.utcnow(),
                        preservation_id
                    )
                else:
                    # Hard delete - remove from GCS and database
                    metadata = await self._get_metadata_from_db(preservation_id, conn)
                    if metadata:
                        gcs_path = self._build_gcs_path(metadata, preservation_id)
                        blob = self.bucket.blob(gcs_path)
                        
                        # Delete from GCS
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            self._executor,
                            blob.delete
                        )
                    
                    # Delete from database
                    query = f"""
                        DELETE FROM {self.config.metadata_table}
                        WHERE preservation_id = $1
                    """
                    await conn.execute(query, preservation_id)
            
            self.logger.info(
                "Model deleted",
                preservation_id=preservation_id,
                soft_delete=soft_delete
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete model",
                preservation_id=preservation_id,
                error=str(e)
            )
            return False
    
    async def rollback_model(
        self,
        model_type: ModelType,
        target_version: Optional[str] = None,
        target_date: Optional[datetime] = None
    ) -> Tuple[bytes, ModelMetadata]:
        """Rollback to a previous model version"""
        try:
            query = f"""
                SELECT preservation_id
                FROM {self.config.metadata_table}
                WHERE model_type = $1 AND state = $2
            """
            params = [model_type.value, ModelState.PRESERVED.value]
            param_count = 2
            
            if target_version:
                param_count += 1
                query += f" AND version = ${param_count}"
                params.append(target_version)
            
            if target_date:
                param_count += 1
                query += f" AND preserved_at <= ${param_count}"
                params.append(target_date)
            
            query += " ORDER BY preserved_at DESC LIMIT 1"
            
            async with get_database_connection() as conn:
                row = await conn.fetchrow(query, *params)
            
            if not row:
                raise VersionError(
                    f"No suitable model found for rollback: {model_type.value}"
                )
            
            preservation_id = row['preservation_id']
            model_data, metadata = await self.load_model(preservation_id)
            
            self.logger.info(
                "Model rollback successful",
                model_type=model_type.value,
                target_version=target_version,
                target_date=target_date,
                rolled_back_to=preservation_id
            )
            
            return model_data, metadata
            
        except Exception as e:
            self.logger.error(
                "Failed to rollback model",
                model_type=model_type.value,
                error=str(e)
            )
            raise VersionError(f"Failed to rollback model: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check preservation system health"""
        try:
            health_status = {
                "status": "healthy",
                "gcs_accessible": False,
                "database_accessible": False,
                "total_models": 0,
                "storage_used_mb": 0,
                "errors": []
            }
            
            # Check GCS accessibility
            try:
                loop = asyncio.get_event_loop()
                exists = await loop.run_in_executor(
                    self._executor,
                    self.bucket.exists
                )
                health_status["gcs_accessible"] = exists
            except Exception as e:
                health_status["errors"].append(f"GCS error: {str(e)}")
            
            # Check database and get stats
            try:
                async with get_database_connection() as conn:
                    health_status["database_accessible"] = True
                    
                    # Get model statistics
                    query = f"""
                        SELECT 
                            COUNT(*) as total_models,
                            SUM(size_bytes) as total_size,
                            COUNT(DISTINCT model_type) as model_types,
                            COUNT(DISTINCT mode) as modes
                        FROM {self.config.metadata_table}
                        WHERE state = $1
                    """
                    row = await conn.fetchrow(query, ModelState.PRESERVED.value)
                    
                    if row:
                        health_status["total_models"] = row['total_models']
                        health_status["storage_used_mb"] = (
                            row['total_size'] / 1024 / 1024 if row['total_size'] else 0
                        )
                        health_status["model_types"] = row['model_types']
                        health_status["modes"] = row['modes']
                        
            except Exception as e:
                health_status["database_accessible"] = False
                health_status["errors"].append(f"Database error: {str(e)}")
            
            # Determine overall status
            if not health_status["gcs_accessible"] or not health_status["database_accessible"]:
                health_status["status"] = "unhealthy"
            elif health_status["errors"]:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _compress_data(self, data: bytes) -> bytes:
        """Compress model data using gzip"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: gzip.compress(data, compresslevel=self.config.compression_level)
        )
    
    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress model data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            gzip.decompress,
            data
        )
    
    async def _upload_to_gcs(
        self,
        gcs_path: str,
        data: bytes,
        metadata: ModelMetadata
    ):
        """Upload data to GCS with metadata"""
        blob = self.bucket.blob(gcs_path)
        
        # Set blob metadata
        blob.metadata = {
            'model_type': metadata.model_type.value,
            'version': metadata.version,
            'mode': metadata.mode,
            'checksum': metadata.checksum,
            'preserved_at': metadata.preserved_at.isoformat()
        }
        
        # Upload in chunks if large
        loop = asyncio.get_event_loop()
        if len(data) > self.config.chunk_size_mb * 1024 * 1024:
            # Use resumable upload for large files
            await loop.run_in_executor(
                self._executor,
                lambda: blob.upload_from_string(data, checksum='md5')
            )
        else:
            # Regular upload for smaller files
            await loop.run_in_executor(
                self._executor,
                blob.upload_from_string,
                data
            )
    
    async def _download_from_gcs(self, gcs_path: str) -> bytes:
        """Download data from GCS"""
        blob = self.bucket.blob(gcs_path)
        
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            self._executor,
            blob.download_as_bytes
        )
        
        return data
    
    def _build_gcs_path(
        self,
        metadata: ModelMetadata,
        preservation_id: str
    ) -> str:
        """Build GCS path with mode isolation"""
        parts = [self.config.gcs_prefix]
        
        # Add mode prefix if isolation enabled
        if self.config.isolate_by_mode:
            mode_prefix = self._get_mode_prefix(metadata.mode)
            if mode_prefix:
                parts.append(mode_prefix)
        
        # Add model type
        parts.append(metadata.model_type.value)
        
        # Add version directory
        parts.append(metadata.version)
        
        # Add preservation ID as filename
        parts.append(f"{preservation_id}.model")
        
        return "/".join(parts)
    
    async def _apply_retention_policy(self, metadata: ModelMetadata):
        """Apply retention policy to remove old backups"""
        try:
            # Determine retention days based on priority
            if metadata.priority == PreservationPriority.CRITICAL:
                retention_days = self.config.emergency_backup_retention_days
            else:
                retention_days = self.config.backup_retention_days
            
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Get old models to delete
            query = f"""
                SELECT preservation_id
                FROM {self.config.metadata_table}
                WHERE model_type = $1 
                AND mode = $2
                AND preserved_at < $3
                AND state = $4
                ORDER BY preserved_at DESC
                OFFSET $5
            """
            
            async with get_database_connection() as conn:
                rows = await conn.fetch(
                    query,
                    metadata.model_type.value,
                    metadata.mode,
                    cutoff_date,
                    ModelState.PRESERVED.value,
                    self.config.max_backups_per_model
                )
            
            # Delete old models
            for row in rows:
                await self.delete_model(row['preservation_id'], soft_delete=True)
            
            if rows:
                self.logger.info(
                    "Retention policy applied",
                    model_type=metadata.model_type.value,
                    mode=metadata.mode,
                    deleted_count=len(rows)
                )
                
        except Exception as e:
            self.logger.error("Failed to apply retention policy", error=str(e))
    
    async def _perform_emergency_backup(self):
        """Perform emergency backup of active models"""
        try:
            # This would be implemented to backup currently active models
            # For now, log the intention
            self.logger.info("Performing emergency backup check")
            
            # In a real implementation, this would:
            # 1. Query for active models from model managers
            # 2. Save them with CRITICAL priority
            # 3. Tag them as emergency backups
            
        except Exception as e:
            self.logger.error("Emergency backup failed", error=str(e))