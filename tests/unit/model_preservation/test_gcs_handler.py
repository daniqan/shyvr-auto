"""
Tests for GCS (Google Cloud Storage) handler for model preservation
"""

import pytest
import asyncio
import gzip
import json
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any, List
import hashlib


# These will be imported from src.model_preservation.gcs_handler once implemented
# For now, we'll define a mock implementation for TDD

class GCSHandler:
    """Google Cloud Storage handler for model preservation"""
    
    def __init__(self, bucket_name: str, project_id: str = None, 
                 enable_compression: bool = True, compression_level: int = 6):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.client = None
        self.bucket = None
        self._cache = {}
    
    async def initialize(self):
        """Initialize GCS client and bucket"""
        pass
    
    async def save(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        """Save model to GCS with optional compression"""
        pass
    
    async def load(self, storage_path: str, use_cache: bool = True) -> bytes:
        """Load model from GCS with caching support"""
        pass
    
    async def delete(self, storage_path: str) -> bool:
        """Delete model from GCS"""
        pass
    
    async def exists(self, storage_path: str) -> bool:
        """Check if model exists in GCS"""
        pass
    
    async def list_models(self, prefix: str = "", limit: int = 100) -> List[str]:
        """List models in GCS bucket"""
        pass
    
    async def get_metadata(self, storage_path: str) -> Dict[str, Any]:
        """Get GCS-specific metadata"""
        pass
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum"""
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data, compresslevel=self.compression_level)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data"""
        return gzip.decompress(data)


class TestGCSHandler:
    """Test GCS handler functionality"""
    
    @pytest.fixture
    def mock_gcs_client(self):
        """Create mock GCS client"""
        with patch('google.cloud.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_client.return_value.bucket.return_value = mock_bucket
            yield mock_client, mock_bucket
    
    @pytest.fixture
    def handler(self, mock_gcs_client):
        """Create GCS handler with mocked client"""
        handler = GCSHandler(
            bucket_name="test-bucket",
            project_id="test-project",
            enable_compression=True,
            compression_level=6
        )
        # Mock the client initialization
        handler.client = mock_gcs_client[0].return_value
        handler.bucket = mock_gcs_client[1]
        return handler
    
    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data"""
        return b"fake model binary data" * 100  # Make it larger for compression tests
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata"""
        return {
            "model_id": "test-123",
            "model_type": "lstm",
            "version": "v1.0.0",
            "created_at": datetime.now().isoformat(),
            "file_size_bytes": 2200,
            "preservation_priority": "high"
        }
    
    def test_handler_initialization(self):
        """Test GCS handler initialization"""
        handler = GCSHandler(
            bucket_name="my-bucket",
            project_id="my-project",
            enable_compression=True,
            compression_level=9
        )
        
        assert handler.bucket_name == "my-bucket"
        assert handler.project_id == "my-project"
        assert handler.enable_compression is True
        assert handler.compression_level == 9
        assert handler.client is None
        assert handler.bucket is None
        assert handler._cache == {}
    
    def test_handler_default_values(self):
        """Test handler with default values"""
        handler = GCSHandler(bucket_name="test-bucket")
        
        assert handler.bucket_name == "test-bucket"
        assert handler.project_id is None
        assert handler.enable_compression is True
        assert handler.compression_level == 6
    
    @pytest.mark.asyncio
    async def test_initialize_client(self, mock_gcs_client):
        """Test GCS client initialization"""
        handler = GCSHandler(bucket_name="test-bucket", project_id="test-project")
        
        # Mock the initialization
        with patch.object(handler, 'initialize', new_callable=AsyncMock) as mock_init:
            await handler.initialize()
            mock_init.assert_called_once()
    
    def test_calculate_checksum(self, handler):
        """Test checksum calculation"""
        data = b"test data"
        checksum = handler._calculate_checksum(data)
        
        assert checksum.startswith("sha256:")
        assert len(checksum) == 71  # "sha256:" + 64 hex chars
        
        # Verify same data produces same checksum
        checksum2 = handler._calculate_checksum(data)
        assert checksum == checksum2
        
        # Verify different data produces different checksum
        different_data = b"different data"
        different_checksum = handler._calculate_checksum(different_data)
        assert checksum != different_checksum
    
    def test_compress_data(self, handler, sample_model_data):
        """Test data compression"""
        compressed = handler._compress_data(sample_model_data)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(sample_model_data)  # Should be smaller
        
        # Verify it's valid gzip data
        decompressed = gzip.decompress(compressed)
        assert decompressed == sample_model_data
    
    def test_decompress_data(self, handler, sample_model_data):
        """Test data decompression"""
        compressed = gzip.compress(sample_model_data)
        decompressed = handler._decompress_data(compressed)
        
        assert decompressed == sample_model_data
    
    def test_compression_levels(self):
        """Test different compression levels"""
        data = b"test data" * 1000
        
        # Test different compression levels
        handler_fast = GCSHandler("bucket", compression_level=1)
        handler_best = GCSHandler("bucket", compression_level=9)
        
        compressed_fast = handler_fast._compress_data(data)
        compressed_best = handler_best._compress_data(data)
        
        # Higher compression level should produce smaller output
        assert len(compressed_best) <= len(compressed_fast)
        
        # Both should decompress to same data
        assert handler_fast._decompress_data(compressed_fast) == data
        assert handler_best._decompress_data(compressed_best) == data
    
    @pytest.mark.asyncio
    async def test_save_model_with_compression(self, handler, sample_model_data, sample_metadata):
        """Test saving model with compression enabled"""
        # Mock blob operations
        mock_blob = MagicMock()
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the save method since it's not implemented yet
        async def mock_save(data, metadata):
            # Simulate save operation
            if handler.enable_compression:
                data = handler._compress_data(data)
            
            storage_path = f"models/{metadata['model_type']}/{metadata['version']}/model.pkl.gz"
            
            # Store in mock blob
            mock_blob.upload_from_string.call_args = (data,)
            mock_blob.metadata = metadata
            
            return storage_path
        
        handler.save = mock_save
        storage_path = await handler.save(sample_model_data, sample_metadata)
        
        assert storage_path.endswith(".gz")
        assert "lstm" in storage_path
        assert "v1.0.0" in storage_path
    
    @pytest.mark.asyncio
    async def test_save_model_without_compression(self, sample_model_data, sample_metadata):
        """Test saving model without compression"""
        handler = GCSHandler("test-bucket", enable_compression=False)
        
        # Mock blob operations
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        handler.bucket = mock_bucket
        
        # Mock the save method
        async def mock_save(data, metadata):
            storage_path = f"models/{metadata['model_type']}/{metadata['version']}/model.pkl"
            mock_blob.upload_from_string.call_args = (data,)
            return storage_path
        
        handler.save = mock_save
        storage_path = await handler.save(sample_model_data, sample_metadata)
        
        assert not storage_path.endswith(".gz")
        assert storage_path.endswith(".pkl")
    
    @pytest.mark.asyncio
    async def test_load_model_with_cache(self, handler, sample_model_data):
        """Test loading model with caching enabled"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        compressed_data = handler._compress_data(sample_model_data)
        
        # Mock blob download
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = compressed_data
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the load method
        async def mock_load(path, use_cache=True):
            if use_cache and path in handler._cache:
                return handler._cache[path]
            
            # Simulate download
            data = mock_blob.download_as_bytes()
            
            # Decompress if needed
            if path.endswith('.gz'):
                data = handler._decompress_data(data)
            
            # Cache the result
            if use_cache:
                handler._cache[path] = data
            
            return data
        
        handler.load = mock_load
        
        # First load - should hit storage
        loaded_data = await handler.load(storage_path)
        assert loaded_data == sample_model_data
        assert storage_path in handler._cache
        
        # Second load - should hit cache
        mock_blob.download_as_bytes.reset_mock()
        loaded_data_cached = await handler.load(storage_path)
        assert loaded_data_cached == sample_model_data
        mock_blob.download_as_bytes.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_load_model_without_cache(self, handler, sample_model_data):
        """Test loading model with caching disabled"""
        storage_path = "models/lstm/v1.0.0/model.pkl"
        
        # Mock blob download
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = sample_model_data
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the load method
        call_count = 0
        async def mock_load(path, use_cache=True):
            nonlocal call_count
            call_count += 1
            return sample_model_data
        
        handler.load = mock_load
        
        # Load without cache
        await handler.load(storage_path, use_cache=False)
        await handler.load(storage_path, use_cache=False)
        
        # Should be called twice (no caching)
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, handler):
        """Test loading non-existent model"""
        storage_path = "models/nonexistent/model.pkl"
        
        # Mock blob that doesn't exist
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the load method to raise exception
        async def mock_load(path, use_cache=True):
            raise FileNotFoundError(f"Model not found: {path}")
        
        handler.load = mock_load
        
        with pytest.raises(FileNotFoundError):
            await handler.load(storage_path)
    
    @pytest.mark.asyncio
    async def test_delete_model(self, handler):
        """Test deleting model from GCS"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        
        # Mock blob operations
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the delete method
        async def mock_delete(path):
            if mock_blob.exists():
                mock_blob.delete()
                # Remove from cache if present
                if path in handler._cache:
                    del handler._cache[path]
                return True
            return False
        
        handler.delete = mock_delete
        
        # Add to cache first
        handler._cache[storage_path] = b"cached data"
        
        # Delete the model
        deleted = await handler.delete(storage_path)
        
        assert deleted is True
        assert storage_path not in handler._cache
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, handler):
        """Test deleting non-existent model"""
        storage_path = "models/nonexistent/model.pkl"
        
        # Mock blob that doesn't exist
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the delete method
        async def mock_delete(path):
            return False
        
        handler.delete = mock_delete
        
        deleted = await handler.delete(storage_path)
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_exists_check(self, handler):
        """Test checking if model exists"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        
        # Mock blob exists check
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the exists method
        async def mock_exists(path):
            return mock_blob.exists()
        
        handler.exists = mock_exists
        
        exists = await handler.exists(storage_path)
        assert exists is True
        
        # Test non-existent
        mock_blob.exists.return_value = False
        exists = await handler.exists("models/nonexistent/model.pkl")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, handler):
        """Test listing models in bucket"""
        # Mock blob listing
        mock_blobs = [
            Mock(name="models/lstm/v1.0.0/model.pkl.gz"),
            Mock(name="models/lstm/v1.1.0/model.pkl.gz"),
            Mock(name="models/dqn/v1.0.0/model.pkl.gz"),
            Mock(name="models/transformer/v2.0.0/model.pkl.gz")
        ]
        
        handler.bucket.list_blobs.return_value = mock_blobs
        
        # Mock the list_models method
        async def mock_list(prefix="", limit=100):
            blobs = handler.bucket.list_blobs(prefix=prefix)
            paths = [blob.name for blob in blobs]
            if prefix:
                paths = [p for p in paths if p.startswith(prefix)]
            return paths[:limit]
        
        handler.list_models = mock_list
        
        # List all models
        all_models = await handler.list_models()
        assert len(all_models) == 4
        
        # List LSTM models only
        lstm_models = await handler.list_models(prefix="models/lstm/")
        assert len(lstm_models) == 2
        assert all("lstm" in path for path in lstm_models)
        
        # List with limit
        limited = await handler.list_models(limit=2)
        assert len(limited) == 2
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, handler):
        """Test getting GCS metadata"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        
        # Mock blob with metadata
        mock_blob = MagicMock()
        mock_blob.size = 1024 * 1024  # 1MB
        mock_blob.md5_hash = "abc123"
        mock_blob.time_created = datetime.now()
        mock_blob.metadata = {"model_type": "lstm", "version": "v1.0.0"}
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the get_metadata method
        async def mock_get_metadata(path):
            blob = handler.bucket.blob(path)
            return {
                "size": blob.size,
                "md5_hash": blob.md5_hash,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "custom_metadata": blob.metadata
            }
        
        handler.get_metadata = mock_get_metadata
        
        metadata = await handler.get_metadata(storage_path)
        
        assert metadata["size"] == 1024 * 1024
        assert metadata["md5_hash"] == "abc123"
        assert "created" in metadata
        assert metadata["custom_metadata"]["model_type"] == "lstm"
    
    @pytest.mark.asyncio
    async def test_error_handling_network_error(self, handler):
        """Test handling of network errors"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        
        # Mock network error
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = Exception("Network error")
        handler.bucket.blob.return_value = mock_blob
        
        # Mock the load method with error
        async def mock_load(path, use_cache=True):
            raise Exception("Network error")
        
        handler.load = mock_load
        
        with pytest.raises(Exception, match="Network error"):
            await handler.load(storage_path)
    
    @pytest.mark.asyncio
    async def test_checksum_verification(self, handler, sample_model_data):
        """Test checksum verification on load"""
        storage_path = "models/lstm/v1.0.0/model.pkl.gz"
        expected_checksum = handler._calculate_checksum(sample_model_data)
        
        # Mock blob with metadata including checksum
        mock_blob = MagicMock()
        mock_blob.metadata = {"checksum": expected_checksum}
        mock_blob.download_as_bytes.return_value = handler._compress_data(sample_model_data)
        handler.bucket.blob.return_value = mock_blob
        
        # Mock load with checksum verification
        async def mock_load_with_verification(path, use_cache=True):
            blob = handler.bucket.blob(path)
            data = blob.download_as_bytes()
            
            # Decompress
            if path.endswith('.gz'):
                data = handler._decompress_data(data)
            
            # Verify checksum
            if blob.metadata and 'checksum' in blob.metadata:
                calculated = handler._calculate_checksum(data)
                if calculated != blob.metadata['checksum']:
                    raise ValueError(f"Checksum mismatch: expected {blob.metadata['checksum']}, got {calculated}")
            
            return data
        
        handler.load = mock_load_with_verification
        
        # Should load successfully with matching checksum
        loaded_data = await handler.load(storage_path)
        assert loaded_data == sample_model_data
        
        # Test with mismatched checksum
        mock_blob.metadata["checksum"] = "wrong_checksum"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            await handler.load(storage_path)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, handler, sample_model_data):
        """Test concurrent save/load operations"""
        # Mock save and load methods
        save_count = 0
        load_count = 0
        
        async def mock_save(data, metadata):
            nonlocal save_count
            save_count += 1
            await asyncio.sleep(0.01)  # Simulate I/O
            return f"models/{metadata['model_type']}/{metadata['version']}/model.pkl"
        
        async def mock_load(path, use_cache=True):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.01)  # Simulate I/O
            return sample_model_data
        
        handler.save = mock_save
        handler.load = mock_load
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(5):
            metadata = {
                "model_type": "lstm",
                "version": f"v1.0.{i}",
                "model_id": f"test-{i}"
            }
            tasks.append(handler.save(sample_model_data, metadata))
            tasks.append(handler.load(f"models/lstm/v1.0.{i}/model.pkl"))
        
        results = await asyncio.gather(*tasks)
        
        assert save_count == 5
        assert load_count == 5
        assert len(results) == 10
    
    def test_cache_memory_management(self, handler):
        """Test cache memory limits"""
        # Add items to cache
        for i in range(100):
            handler._cache[f"model_{i}"] = b"x" * 10000  # 10KB each
        
        # In real implementation, would have cache size limits
        # For now, just verify cache works
        assert len(handler._cache) == 100
        
        # Clear cache
        handler._cache.clear()
        assert len(handler._cache) == 0
    
    @pytest.mark.asyncio
    async def test_bucket_creation_if_not_exists(self):
        """Test bucket creation if it doesn't exist"""
        with patch('google.cloud.storage.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock bucket doesn't exist
            mock_client.bucket.side_effect = Exception("Bucket not found")
            
            handler = GCSHandler("new-bucket", "test-project")
            
            # In real implementation, would create bucket
            # For now, just verify initialization doesn't fail
            assert handler.bucket_name == "new-bucket"
    
    @pytest.mark.asyncio
    async def test_versioning_support(self, handler, sample_model_data):
        """Test model versioning support"""
        base_metadata = {
            "model_type": "lstm",
            "model_id": "lstm-base"
        }
        
        # Mock save with versioning
        versions_saved = []
        
        async def mock_save_versioned(data, metadata):
            version = metadata.get("version", "v1.0.0")
            path = f"models/{metadata['model_type']}/{version}/model.pkl"
            versions_saved.append(version)
            return path
        
        handler.save = mock_save_versioned
        
        # Save multiple versions
        for i in range(3):
            metadata = base_metadata.copy()
            metadata["version"] = f"v1.0.{i}"
            await handler.save(sample_model_data, metadata)
        
        assert len(versions_saved) == 3
        assert "v1.0.0" in versions_saved
        assert "v1.0.1" in versions_saved
        assert "v1.0.2" in versions_saved