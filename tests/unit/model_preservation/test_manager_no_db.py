"""
Tests for Model Preservation Manager without Database Handler

Tests the PreservationManager class functionality when db_handler is None.
This covers edge cases where the database is not available.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

from src.model_preservation.manager import (
    PreservationManager,
    PreservationConfig
)
from src.model_preservation.base import (
    PreservationError
)


class TestPreservationManagerNoDatabase:
    """Test preservation manager without database handler"""
    
    @pytest.fixture
    def mock_storage_handler(self):
        """Create mock storage handler"""
        handler = AsyncMock()
        handler.save = AsyncMock(return_value="models/lstm/v1.0.0/model.pkl.gz")
        handler.load = AsyncMock(return_value=b"model_data")
        handler.delete = AsyncMock(return_value=True)
        handler.exists = AsyncMock(return_value=True)
        handler.list_models = AsyncMock(return_value=[])
        handler.initialize = AsyncMock()
        return handler
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=6,
            max_versions_per_model=10,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=True,
            emergency_backup=True
        )
    
    @pytest.fixture
    def manager(self, config, mock_storage_handler):
        """Create preservation manager without db_handler"""
        manager = PreservationManager(config)
        manager.storage_handler = mock_storage_handler
        # Explicitly set db_handler to None
        manager.db_handler = None
        return manager
    
    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data"""
        return b"fake model binary data" * 100
    
    @pytest.mark.asyncio
    async def test_save_model_without_db(self, manager, sample_model_data):
        """Test saving model when db_handler is None"""
        # Save model
        model_id = await manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.0",
            mode="analysis"
        )
        
        # Should use the generated model_id since no db_handler
        assert model_id.startswith("lstm-v1.0.0-")
        
        # Verify storage handler called
        manager.storage_handler.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_without_db(self, manager):
        """Test loading model when db_handler is None"""
        # Should raise FileNotFoundError since db_handler is needed for metadata
        with pytest.raises(FileNotFoundError, match="database handler not available"):
            await manager.load_model(
                model_type="lstm",
                version="v1.0.0",
                mode="analysis"
            )
    
    @pytest.mark.asyncio
    async def test_emergency_backup_without_db(self, manager):
        """Test emergency backup when db_handler is None"""
        # Should return empty list since no active models can be retrieved
        backed_up = await manager.emergency_backup()
        assert backed_up == []
    
    @pytest.mark.asyncio
    async def test_get_stats_without_db(self, manager):
        """Test getting stats when db_handler is None"""
        stats = await manager.get_stats()
        
        # Should return default stats
        assert stats.total_models == 0
        assert stats.active_models == 0
        assert stats.total_size_mb == 0.0
        assert stats.models_by_type == {}
        assert stats.models_by_mode == {}
    
    @pytest.mark.asyncio
    async def test_cleanup_without_db(self, manager):
        """Test cleanup when db_handler is None"""
        deleted = await manager.cleanup_old_versions(days=30)
        
        # Should return 0 since no db_handler
        assert deleted == 0
    
    @pytest.mark.asyncio
    async def test_track_performance_without_db(self, manager):
        """Test performance tracking when db_handler is None"""
        # Should not raise any error, just do nothing
        await manager.track_performance(
            model_id="test-123",
            metrics={"accuracy": 0.95}
        )
    
    @pytest.mark.asyncio
    async def test_version_generation_without_db(self, manager):
        """Test version generation when db_handler is None"""
        version = await manager._generate_next_version("lstm")
        
        # Should return default version
        assert version == "v1.0.0"
    
    @pytest.mark.asyncio
    async def test_initialize_without_db(self, manager):
        """Test initialization when db_handler is None"""
        await manager.initialize()
        
        # Should only initialize storage handler
        manager.storage_handler.initialize.assert_called_once()
        assert manager._is_running is True
    
    @pytest.mark.asyncio
    async def test_background_task_without_db(self, manager):
        """Test background backup task when db_handler is None"""
        manager.config.backup_interval_hours = 0.001  # Very short for testing
        
        # Start background task
        await manager.start()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Stop
        await manager.stop()
        
        # Should not have done any backups since no db_handler
        assert manager.storage_handler.save.call_count == 0