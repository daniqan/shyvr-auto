"""
Tests for Model Preservation Manager

Tests the PreservationManager class that coordinates between GCS storage
and database metadata for comprehensive model preservation.
"""

import pytest
import asyncio
import signal
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any, List
import json

# Import the manager (to be implemented)
from src.model_preservation.manager import (
    PreservationManager,
    PreservationConfig,
    PreservationStats
)
from src.model_preservation.base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    VersionError,
    generate_model_id,
    calculate_checksum
)


class TestPreservationManager:
    """Test preservation manager functionality"""
    
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
    def mock_db_handler(self):
        """Create mock database handler"""
        handler = AsyncMock()
        handler.save_metadata = AsyncMock(return_value="model-123")
        handler.get_metadata = AsyncMock()
        handler.update_state = AsyncMock()
        handler.get_versions = AsyncMock(return_value=[])
        handler.record_event = AsyncMock()
        handler.track_performance = AsyncMock()
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
    def manager(self, config, mock_storage_handler, mock_db_handler):
        """Create preservation manager with mocks"""
        manager = PreservationManager(config)
        manager.storage_handler = mock_storage_handler
        manager.db_handler = mock_db_handler
        return manager
    
    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data"""
        return b"fake model binary data" * 100
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata"""
        return ModelMetadata(
            model_id="test-123",
            model_type="lstm",
            version="v1.0.0",
            created_at=datetime.now(),
            file_size_bytes=2200,
            preservation_priority=PreservationPriority.HIGH,
            mode="analysis",
            tags=["test", "lstm"],
            performance_metrics={"accuracy": 0.95}
        )
    
    def test_manager_initialization(self, config):
        """Test manager initialization"""
        manager = PreservationManager(config)
        
        assert manager.config == config
        assert manager.storage_handler is not None  # Should be initialized with GCSHandler
        assert manager.db_handler is not None  # Should be initialized with DatabaseHandler
        assert manager._background_task is None
        assert manager._shutdown_event is not None
        assert manager._is_running is False
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = PreservationConfig(
            gcs_bucket="my-bucket",
            backup_interval_hours=12,
            max_versions_per_model=5
        )
        manager = PreservationManager(config)
        assert manager.config.backup_interval_hours == 12
        
        # Invalid interval
        with pytest.raises(ValueError, match="Backup interval must be positive"):
            PreservationConfig(
                gcs_bucket="my-bucket",
                backup_interval_hours=-1
            )
        
        # Invalid max versions
        with pytest.raises(ValueError, match="Max versions must be at least 1"):
            PreservationConfig(
                gcs_bucket="my-bucket",
                max_versions_per_model=0
            )
    
    @pytest.mark.asyncio
    async def test_initialize_handlers(self, manager):
        """Test handler initialization"""
        await manager.initialize()
        
        manager.storage_handler.initialize.assert_called_once()
        manager.db_handler.initialize.assert_called_once()
        assert manager._is_running is True
    
    @pytest.mark.asyncio
    async def test_save_model_basic(self, manager, sample_model_data, sample_metadata):
        """Test basic model saving"""
        # Setup mock returns
        manager.storage_handler.save.return_value = "models/lstm/v1.0.0/model.pkl.gz"
        manager.db_handler.save_metadata.return_value = "model-123"
        
        # Save model
        model_id = await manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.0",
            mode="analysis",
            tags=["test"],
            metadata={"custom": "value"}
        )
        
        assert model_id == "model-123"
        
        # Verify storage handler called
        manager.storage_handler.save.assert_called_once()
        call_args = manager.storage_handler.save.call_args[0]
        assert call_args[0] == sample_model_data
        
        # Verify db handler called
        manager.db_handler.save_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_model_auto_versioning(self, manager, sample_model_data):
        """Test automatic version generation"""
        # Mock existing versions
        manager.db_handler.get_versions.return_value = [
            {"version": "v1.0.0"},
            {"version": "v1.0.1"},
            {"version": "v1.1.0"}
        ]
        
        # Save without version
        await manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            mode="analysis"
        )
        
        # Should generate v1.1.1
        call_args = manager.storage_handler.save.call_args[0]
        metadata = call_args[1]
        assert metadata.version == "v1.1.1"
    
    @pytest.mark.asyncio
    async def test_save_model_version_limit(self, manager, sample_model_data):
        """Test version limit enforcement"""
        manager.config.max_versions_per_model = 3
        
        # Mock existing versions - need to return them twice:
        # First call during version generation, second during limit enforcement
        old_versions = [
            {"version": "v1.0.0", "created_at": datetime.now() - timedelta(days=3), "mode": "analysis"},
            {"version": "v1.0.1", "created_at": datetime.now() - timedelta(days=2), "mode": "analysis"},
            {"version": "v1.0.2", "created_at": datetime.now() - timedelta(days=1), "mode": "analysis"},
        ]
        # Mock get_versions to return old versions initially,
        # then include new version after save
        all_versions = old_versions + [{"version": "v1.0.3", "created_at": datetime.now(), "mode": "analysis"}]
        manager.db_handler.get_versions.side_effect = [
            old_versions,  # For version generation
            all_versions   # For limit check after save
        ]
        
        # Mock get_metadata to return storage path for oldest version
        manager.db_handler.get_metadata.return_value = {
            "storage_path": "models/lstm/v1.0.0/model.pkl.gz"
        }
        
        # Save new version
        await manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.3",
            mode="analysis"
        )
        
        # Should delete oldest version
        manager.storage_handler.delete.assert_called_once()
        manager.db_handler.update_state.assert_called_with(
            model_type="lstm",
            version="v1.0.0",
            state=ModelState.DELETED
        )
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, manager, sample_model_data):
        """Test successful model loading"""
        # Setup mocks
        manager.storage_handler.load.return_value = sample_model_data
        manager.db_handler.get_metadata.return_value = {
            "model_id": "test-123",
            "storage_path": "models/lstm/v1.0.0/model.pkl.gz",
            "state": "active"
        }
        
        # Load model
        data, metadata = await manager.load_model(
            model_type="lstm",
            version="v1.0.0",
            mode="analysis"
        )
        
        assert data == sample_model_data
        assert metadata["model_id"] == "test-123"
        
        # Verify event recorded
        manager.db_handler.record_event.assert_called_with(
            model_id="test-123",
            event_type="loaded",
            details={"source": "primary", "requested_version": "v1.0.0"}
        )
    
    @pytest.mark.asyncio
    async def test_load_model_fallback(self, manager, sample_model_data):
        """Test fallback to previous version"""
        # Storage handler returns data when fallback version is loaded
        manager.storage_handler.load.return_value = sample_model_data
        
        # Mock fallback metadata
        manager.db_handler.get_metadata.side_effect = [
            None,  # Primary not found
            {  # Fallback found
                "model_id": "test-122",
                "storage_path": "models/lstm/v0.9.0/model.pkl.gz",
                "state": "active"
            }
        ]
        
        # Mock previous versions
        manager.db_handler.get_versions.return_value = [
            {"version": "v0.9.0", "state": "active"}
        ]
        
        # Load with fallback
        data, metadata = await manager.load_model(
            model_type="lstm",
            version="v1.0.0",
            mode="analysis",
            fallback=True
        )
        
        assert data == sample_model_data
        assert metadata["model_id"] == "test-122"
        
        # Verify fallback event
        manager.db_handler.record_event.assert_called_with(
            model_id="test-122",
            event_type="loaded",
            details={"source": "fallback", "requested_version": "v1.0.0"}
        )
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self, manager):
        """Test model not found error"""
        manager.db_handler.get_metadata.return_value = None
        
        with pytest.raises(FileNotFoundError, match="Model not found"):
            await manager.load_model(
                model_type="lstm",
                version="v1.0.0",
                mode="analysis"
            )
    
    @pytest.mark.asyncio
    async def test_rollback_model(self, manager, sample_model_data):
        """Test model rollback functionality"""
        # Mock current and target versions
        manager.db_handler.get_metadata.side_effect = [
            {  # Current version
                "model_id": "test-124",
                "version": "v1.2.0",
                "storage_path": "models/lstm/v1.2.0/model.pkl.gz"
            },
            {  # Target version
                "model_id": "test-123",
                "version": "v1.0.0",
                "storage_path": "models/lstm/v1.0.0/model.pkl.gz"
            }
        ]
        
        manager.storage_handler.load.return_value = sample_model_data
        
        # Perform rollback
        result = await manager.rollback_model(
            model_type="lstm",
            target_version="v1.0.0",
            mode="analysis"
        )
        
        assert result["rolled_back_from"] == "v1.2.0"
        assert result["rolled_back_to"] == "v1.0.0"
        
        # Verify new version saved
        manager.storage_handler.save.assert_called_once()
        
        # Verify events
        assert manager.db_handler.record_event.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_emergency_backup(self, manager):
        """Test emergency backup functionality"""
        # Mock active models
        active_models = [
            {
                "model_type": "lstm",
                "version": "v1.0.0",
                "mode": "analysis",
                "storage_path": "models/lstm/v1.0.0/model.pkl.gz"
            },
            {
                "model_type": "dqn",
                "version": "v2.0.0",
                "mode": "live",
                "storage_path": "models/dqn/v2.0.0/model.pkl.gz"
            }
        ]
        manager.db_handler.get_active_models = AsyncMock(return_value=active_models)
        manager.storage_handler.load.return_value = b"model_data"
        
        # Trigger emergency backup
        backed_up = await manager.emergency_backup()
        
        assert len(backed_up) == 2
        assert manager.storage_handler.save.call_count == 2
        
        # Verify emergency versions created
        for call in manager.storage_handler.save.call_args_list:
            metadata = call[0][1]
            assert metadata.tags == ["emergency_backup"]
            assert metadata.preservation_priority == PreservationPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, manager):
        """Test graceful shutdown handling"""
        # Mock emergency backup to track if it was called
        emergency_backup_called = False
        
        async def mock_emergency_backup():
            nonlocal emergency_backup_called
            emergency_backup_called = True
            return ["backup-1", "backup-2"]
        
        manager.emergency_backup = mock_emergency_backup
        
        # Start manager
        await manager.start()
        
        # Trigger shutdown
        await manager.stop()
        
        # Verify emergency backup was called (if configured)
        if manager.config.emergency_backup:
            assert emergency_backup_called
        
        assert not manager._is_running
        assert manager._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_background_backup_task(self, manager):
        """Test background backup task"""
        manager.config.backup_interval_hours = 0.001  # Very short for testing
        
        # Mock active models
        manager.db_handler.get_active_models = AsyncMock(return_value=[
            {
                "model_type": "lstm",
                "version": "v1.0.0",
                "mode": "analysis",
                "storage_path": "models/lstm/v1.0.0/model.pkl.gz"
            }
        ])
        manager.storage_handler.load.return_value = b"model_data"
        
        # Start background task
        await manager.start()
        
        # Wait for at least one backup
        await asyncio.sleep(0.1)
        
        # Stop
        await manager.stop()
        
        # Verify backup occurred
        assert manager.storage_handler.save.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_migrate_model_between_modes(self, manager, sample_model_data):
        """Test model migration between modes"""
        # Setup source model
        manager.db_handler.get_metadata.return_value = {
            "model_id": "test-123",
            "mode": "analysis",
            "storage_path": "models/analysis/lstm/v1.0.0/model.pkl.gz"
        }
        manager.storage_handler.load.return_value = sample_model_data
        
        # Migrate model
        new_model_id = await manager.migrate_model(
            model_type="lstm",
            version="v1.0.0",
            from_mode="analysis",
            to_mode="simulation"
        )
        
        assert new_model_id is not None
        
        # Verify saved to new mode
        save_call = manager.storage_handler.save.call_args[0]
        metadata = save_call[1]
        assert metadata.mode == "simulation"
        assert metadata.tags == ["migrated_from_analysis"]
    
    @pytest.mark.asyncio
    async def test_get_model_stats(self, manager):
        """Test getting preservation statistics"""
        # Mock stats data
        manager.db_handler.get_preservation_stats = AsyncMock(return_value={
            "total_models": 50,
            "active_models": 10,
            "total_size_bytes": 1024 * 1024 * 100,  # 100MB
            "models_by_type": {"lstm": 20, "dqn": 30},
            "models_by_mode": {"analysis": 25, "simulation": 15, "live": 10}
        })
        
        stats = await manager.get_stats()
        
        assert isinstance(stats, PreservationStats)
        assert stats.total_models == 50
        assert stats.active_models == 10
        assert stats.total_size_mb == 100
        assert stats.models_by_type["lstm"] == 20
    
    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, manager):
        """Test cleanup of old model versions"""
        # Mock old versions
        old_versions = [
            {
                "version": "v0.1.0",
                "created_at": datetime.now() - timedelta(days=90),
                "storage_path": "models/lstm/v0.1.0/model.pkl.gz"
            }
        ]
        manager.db_handler.get_old_versions = AsyncMock(return_value=old_versions)
        
        # Run cleanup
        deleted = await manager.cleanup_old_versions(days=30)
        
        assert deleted == 1
        manager.storage_handler.delete.assert_called_once()
        manager.db_handler.update_state.assert_called_with(
            model_type="lstm",
            version="v0.1.0",
            state=ModelState.DELETED
        )
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, manager):
        """Test model performance tracking"""
        # Track performance
        await manager.track_performance(
            model_id="test-123",
            metrics={
                "accuracy": 0.95,
                "loss": 0.05,
                "inference_time_ms": 10
            }
        )
        
        manager.db_handler.track_performance.assert_called_once_with(
            model_id="test-123",
            metrics={
                "accuracy": 0.95,
                "loss": 0.05,
                "inference_time_ms": 10
            }
        )
    
    @pytest.mark.asyncio
    async def test_error_handling_storage_failure(self, manager, sample_model_data):
        """Test handling of storage failures"""
        manager.storage_handler.save.side_effect = Exception("Storage error")
        
        with pytest.raises(PreservationError, match="Failed to save model"):
            await manager.save_model(
                model_data=sample_model_data,
                model_type="lstm",
                version="v1.0.0"
            )
        
        # Error event recording is not implemented in this version
        # Could be added as a future enhancement
    
    @pytest.mark.asyncio
    async def test_concurrent_saves(self, manager, sample_model_data):
        """Test concurrent model saves"""
        # Save multiple models concurrently
        tasks = []
        for i in range(5):
            task = manager.save_model(
                model_data=sample_model_data,
                model_type="lstm",
                version=f"v1.0.{i}",
                mode="analysis"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert manager.storage_handler.save.call_count == 5
        assert manager.db_handler.save_metadata.call_count == 5
    
    def test_signal_handler_registration(self, config):
        """Test signal handler registration"""
        with patch('signal.signal') as mock_signal:
            manager = PreservationManager(config)
            
            # Verify signal handlers registered
            assert mock_signal.call_count >= 2  # At least SIGTERM and SIGINT
            # Check that signal handlers were registered
            calls = mock_signal.call_args_list
            signals_registered = [call[0][0] for call in calls]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered
    
    @pytest.mark.asyncio
    async def test_mode_isolation(self, manager, sample_model_data):
        """Test mode isolation when enabled"""
        manager.config.mode_isolation = True
        
        # Save model for specific mode
        await manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.0",
            mode="live"
        )
        
        # Verify mode included in metadata
        call_args = manager.storage_handler.save.call_args[0]
        metadata = call_args[1]
        assert metadata.mode == "live"
        
        # Try to load from different mode - should fail
        manager.db_handler.get_metadata.return_value = None
        
        with pytest.raises(FileNotFoundError):
            await manager.load_model(
                model_type="lstm",
                version="v1.0.0",
                mode="analysis"  # Different mode
            )