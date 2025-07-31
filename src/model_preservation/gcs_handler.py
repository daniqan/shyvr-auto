"""
Google Cloud Storage (GCS) handler for model preservation

Implements the StorageHandlerBase interface for GCS operations.
"""

import asyncio
import gzip
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from concurrent.futures import ThreadPoolExecutor

from .base import StorageHandlerBase, StorageError, ChecksumError


class GCSHandler(StorageHandlerBase):
    """Google Cloud Storage handler for model preservation"""
    
    def __init__(self, bucket_name: str, project_id: str = None,
                 enable_compression: bool = True, compression_level: int = 6,
                 cache_size_mb: int = 100, max_workers: int = 4):
        """
        Initialize GCS handler
        
        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID (optional)
            enable_compression: Enable zlib compression
            compression_level: Compression level (1-9)
            cache_size_mb: Cache size limit in MB
            max_workers: Max concurrent workers for operations
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.cache_size_mb = cache_size_mb
        self.max_workers = max_workers
        
        # Initialize later
        self.client = None
        self.bucket = None
        self._cache = {}
        self._cache_size = 0
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def initialize(self):
        """Initialize GCS client and bucket"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._init_client)
        
    def _init_client(self):
        """Initialize GCS client (sync)"""
        if self.project_id:
            self.client = storage.Client(project=self.project_id)
        else:
            self.client = storage.Client()
        
        try:
            self.bucket = self.client.bucket(self.bucket_name)
            # Check if bucket exists
            self.bucket.reload()
        except NotFound:
            # Create bucket if it doesn't exist
            self.bucket = self.client.create_bucket(self.bucket_name)
            
    async def save(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        """Save model to GCS with optional compression"""
        # Calculate checksum
        checksum = self._calculate_checksum(model_data)
        
        # Compress if enabled
        if self.enable_compression:
            model_data = self._compress_data(model_data)
            
        # Construct storage path
        storage_path = self._construct_path(metadata)
        
        # Save to GCS
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._save_sync,
            storage_path,
            model_data,
            metadata,
            checksum
        )
        
        return storage_path
        
    def _save_sync(self, storage_path: str, data: bytes, 
                   metadata: Dict[str, Any], checksum: str):
        """Save to GCS (sync)"""
        blob = self.bucket.blob(storage_path)
        
        # Add metadata
        blob.metadata = metadata.copy()
        blob.metadata['checksum'] = checksum
        blob.metadata['compressed'] = self.enable_compression
        blob.metadata['compression_level'] = self.compression_level
        
        # Upload
        blob.upload_from_string(data)
        
    async def load(self, storage_path: str, use_cache: bool = True) -> bytes:
        """Load model from GCS with caching support"""
        # Check cache first
        if use_cache and storage_path in self._cache:
            return self._cache[storage_path]
            
        # Load from GCS
        loop = asyncio.get_event_loop()
        data, metadata = await loop.run_in_executor(
            self._executor,
            self._load_sync,
            storage_path
        )
        
        # Decompress if needed
        if storage_path.endswith('.gz') or metadata.get('compressed', False):
            data = self._decompress_data(data)
            
        # Verify checksum if available
        if 'checksum' in metadata:
            calculated = self._calculate_checksum(data)
            if calculated != metadata['checksum']:
                raise ChecksumError(
                    f"Checksum mismatch: expected {metadata['checksum']}, "
                    f"got {calculated}"
                )
                
        # Cache if enabled
        if use_cache:
            self._add_to_cache(storage_path, data)
            
        return data
        
    def _load_sync(self, storage_path: str):
        """Load from GCS (sync)"""
        blob = self.bucket.blob(storage_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"Model not found: {storage_path}")
            
        # Download data
        data = blob.download_as_bytes()
        
        # Get metadata
        blob.reload()  # Refresh metadata
        metadata = blob.metadata or {}
        
        return data, metadata
        
    async def delete(self, storage_path: str) -> bool:
        """Delete model from GCS"""
        # Remove from cache
        if storage_path in self._cache:
            del self._cache[storage_path]
            
        # Delete from GCS
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._delete_sync,
            storage_path
        )
        
    def _delete_sync(self, storage_path: str) -> bool:
        """Delete from GCS (sync)"""
        blob = self.bucket.blob(storage_path)
        
        if blob.exists():
            blob.delete()
            return True
        return False
        
    async def exists(self, storage_path: str) -> bool:
        """Check if model exists in GCS"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._exists_sync,
            storage_path
        )
        
    def _exists_sync(self, storage_path: str) -> bool:
        """Check existence in GCS (sync)"""
        blob = self.bucket.blob(storage_path)
        return blob.exists()
        
    async def list_models(self, prefix: str = "", limit: int = 100) -> List[str]:
        """List models in GCS bucket"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._list_sync,
            prefix,
            limit
        )
        
    def _list_sync(self, prefix: str, limit: int) -> List[str]:
        """List models in GCS (sync)"""
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=limit)
        return [blob.name for blob in blobs]
        
    async def get_metadata(self, storage_path: str) -> Dict[str, Any]:
        """Get GCS-specific metadata"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_metadata_sync,
            storage_path
        )
        
    def _get_metadata_sync(self, storage_path: str) -> Dict[str, Any]:
        """Get metadata from GCS (sync)"""
        blob = self.bucket.blob(storage_path)
        blob.reload()  # Refresh metadata
        
        return {
            "size": blob.size,
            "md5_hash": blob.md5_hash,
            "created": blob.time_created.isoformat() if blob.time_created else None,
            "custom_metadata": blob.metadata or {}
        }
        
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum"""
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data, compresslevel=self.compression_level)
        
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data"""
        return gzip.decompress(data)
        
    def _construct_path(self, metadata: Dict[str, Any]) -> str:
        """Construct storage path from metadata"""
        model_type = metadata.get('model_type', 'unknown')
        version = metadata.get('version', 'v1.0.0')
        mode = metadata.get('mode', '')
        
        # Build path components
        path_parts = ['models']
        if mode:
            path_parts.append(mode)
        path_parts.extend([model_type, version])
        
        # Add file name
        if self.enable_compression:
            path_parts.append('model.pkl.gz')
        else:
            path_parts.append('model.pkl')
            
        return '/'.join(path_parts)
        
    def _add_to_cache(self, key: str, data: bytes):
        """Add to cache with size management"""
        data_size = len(data)
        max_cache_bytes = self.cache_size_mb * 1024 * 1024
        
        # Remove items if cache would exceed limit
        while self._cache_size + data_size > max_cache_bytes and self._cache:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self._cache))
            oldest_size = len(self._cache[oldest_key])
            del self._cache[oldest_key]
            self._cache_size -= oldest_size
            
        # Add new item
        self._cache[key] = data
        self._cache_size += data_size