"""
Integration Tests for Model Preservation System

Tests complete integration flows for model preservation including:
- Full save/load cycles with GCS and database
- Graceful shutdown scenarios
- Emergency stop handling
- Mode change preservation flows

Following TDD methodology with comprehensive test coverage.
"""

import asyncio
import os
import pytest
import signal
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.model_preservation.manager import PreservationManager, PreservationConfig
from src.model_preservation.base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError
)


@pytest.fixture
def integration_config():
    """Integration test configuration"""
    return PreservationConfig(
        gcs_bucket="test-integration-bucket",
        backup_interval_hours=0.001,  # Very short for testing
        max_versions_per_model=5,
        enable_compression=True,
        mode_isolation=True,
        auto_backup=True,
        emergency_backup=True
    )


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client for integration tests"""
    client = AsyncMock()
    
    # Mock bucket and blob operations
    bucket = MagicMock()
    blob = MagicMock()
    
    bucket.blob.return_value = blob
    client.bucket.return_value = bucket
    
    # Mock blob operations
    blob.upload_from_string = AsyncMock()
    blob.download_as_bytes = AsyncMock(return_value=b"mock_model_data")
    blob.exists = AsyncMock(return_value=True)
    blob.delete = AsyncMock()
    blob.size = 1024
    blob.time_created = datetime.now()
    blob.md5_hash = "mock_hash"
    
    return client


@pytest.fixture
def mock_db_connection():
    """Mock database connection for integration tests"""
    conn = AsyncMock()
    
    # Mock basic database operations
    conn.execute.return_value = None
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = 1
    
    return conn


@pytest.fixture
async def integration_manager(integration_config, mock_gcs_client, mock_db_connection):
    """Integration test preservation manager with mocked dependencies"""
    manager = PreservationManager(integration_config)
    
    # Mock the GCS client in storage handler
    with patch.object(manager.storage_handler, '_get_client', return_value=mock_gcs_client):
        # Mock database handler
        db_handler = AsyncMock()
        db_handler.save_metadata = AsyncMock(return_value="model-123")
        db_handler.get_metadata = AsyncMock()
        db_handler.update_state = AsyncMock()
        db_handler.get_versions = AsyncMock(return_value=[])
        db_handler.record_event = AsyncMock()
        db_handler.track_performance = AsyncMock()
        db_handler.get_active_models = AsyncMock(return_value=[])
        db_handler.get_preservation_stats = AsyncMock(return_value={
            "total_models": 10,
            "active_models": 5,
            "total_size_bytes": 1024 * 1024 * 50,
            "models_by_type": {"lstm": 5, "dqn": 5},
            "models_by_mode": {"analysis": 3, "simulation": 4, "live": 3}
        })
        db_handler.initialize = AsyncMock()
        
        manager.db_handler = db_handler
        
        # Initialize the manager
        await manager.initialize()
        
        yield manager
        
        # Cleanup
        if manager._is_running:
            await manager.stop()


@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return b"sample_model_binary_data" * 50  # Make it reasonably sized


class TestPreservationIntegration:
    """Integration tests for model preservation system"""
    
    @pytest.mark.asyncio
    async def test_complete_save_load_cycle(self, integration_manager, sample_model_data):
        """Test complete save/load cycle with all components"""
        # Setup mock returns for successful save
        integration_manager.db_handler.save_metadata.return_value = "model-save-load-123"
        integration_manager.db_handler.get_metadata.return_value = {
            "model_id": "model-save-load-123",
            "model_type": "lstm",
            "version": "v1.0.0",
            "storage_path": "models/analysis/lstm/v1.0.0/model.pkl.gz",
            "state": "active",
            "created_at": datetime.now(),
            "mode": "analysis"
        }
        
        # Test save operation
        model_id = await integration_manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.0",
            mode="analysis",
            tags=["integration_test", "lstm"],
            metadata={"test": "save_load_cycle"},
            priority=PreservationPriority.HIGH
        )
        
        assert model_id == "model-save-load-123"
        
        # Verify storage handler was called
        integration_manager.storage_handler.save.assert_called_once()
        
        # Verify database handler was called
        integration_manager.db_handler.save_metadata.assert_called_once()
        save_call_args = integration_manager.db_handler.save_metadata.call_args[0][0]
        assert save_call_args.model_type == "lstm"
        assert save_call_args.version == "v1.0.0"
        assert save_call_args.mode == "analysis"
        assert "integration_test" in save_call_args.tags
        
        # Test load operation
        loaded_data, loaded_metadata = await integration_manager.load_model(
            model_type="lstm",
            version="v1.0.0",
            mode="analysis"
        )
        
        # Verify loaded data matches saved data
        assert loaded_data == sample_model_data
        assert loaded_metadata["model_id"] == "model-save-load-123"
        assert loaded_metadata["model_type"] == "lstm"
        assert loaded_metadata["version"] == "v1.0.0"
        
        # Verify load event was recorded
        integration_manager.db_handler.record_event.assert_called_with(
            model_id="model-save-load-123",
            event_type="loaded",
            details={"source": "primary", "requested_version": "v1.0.0"}
        )
    
    @pytest.mark.asyncio
    async def test_save_load_with_fallback_integration(self, integration_manager, sample_model_data):
        """Test save/load cycle with fallback to previous version"""
        # Setup mock for initial save
        integration_manager.db_handler.save_metadata.return_value = "model-fallback-123"
        
        # Save initial version
        await integration_manager.save_model(
            model_data=sample_model_data,
            model_type="dqn",
            version="v1.0.0",
            mode="simulation"
        )
        
        # Setup mock for fallback scenario
        integration_manager.db_handler.get_metadata.side_effect = [
            None,  # Primary version not found
            {  # Fallback version found
                "model_id": "model-fallback-123",
                "model_type": "dqn",
                "version": "v1.0.0",
                "storage_path": "models/simulation/dqn/v1.0.0/model.pkl.gz",
                "state": "active",
                "mode": "simulation"
            }
        ]
        
        # Mock versions for fallback
        integration_manager.db_handler.get_versions.return_value = [
            {"version": "v1.0.0", "state": "active"}
        ]
        
        # Test load with fallback
        loaded_data, loaded_metadata = await integration_manager.load_model(
            model_type="dqn",
            version="v1.1.0",  # Request non-existent version
            mode="simulation",
            fallback=True
        )
        
        # Verify fallback worked
        assert loaded_data == sample_model_data
        assert loaded_metadata["model_id"] == "model-fallback-123"
        assert loaded_metadata["version"] == "v1.0.0"
        
        # Verify fallback event was recorded
        integration_manager.db_handler.record_event.assert_called_with(
            model_id="model-fallback-123",
            event_type="loaded",
            details={"source": "fallback", "requested_version": "v1.1.0"}
        )
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_integration(self, integration_manager, sample_model_data):
        """Test graceful shutdown with model preservation"""
        # Setup active models for emergency backup
        active_models = [
            {
                "model_id": "model-shutdown-1",
                "model_type": "lstm",
                "version": "v1.0.0",
                "mode": "analysis",
                "storage_path": "models/analysis/lstm/v1.0.0/model.pkl.gz"
            },
            {
                "model_id": "model-shutdown-2", 
                "model_type": "dqn",
                "version": "v2.0.1",
                "mode": "live",
                "storage_path": "models/live/dqn/v2.0.1/model.pkl.gz"
            }
        ]
        integration_manager.db_handler.get_active_models.return_value = active_models
        
        # Mock storage load for emergency backup
        integration_manager.storage_handler.load.return_value = sample_model_data
        
        # Start the manager
        await integration_manager.start()
        assert integration_manager._is_running is True
        
        # Trigger graceful shutdown
        await integration_manager.stop()
        
        # Verify manager stopped
        assert integration_manager._is_running is False
        assert integration_manager._shutdown_event.is_set()
        
        # Verify emergency backup was triggered (if configured)
        if integration_manager.config.emergency_backup:
            # Should have created emergency backups for both models
            assert integration_manager.storage_handler.save.call_count >= 2
            
            # Check that emergency backup versions were created
            save_calls = integration_manager.storage_handler.save.call_args_list
            for call in save_calls:
                metadata = call[0][1]
                assert "emergency_backup" in metadata.tags
                assert metadata.preservation_priority == PreservationPriority.CRITICAL
    
    @pytest.mark.asyncio 
    async def test_emergency_stop_handling(self, integration_manager, sample_model_data):
        """Test emergency stop handling with immediate model backup"""
        # Setup critical model that needs emergency backup
        critical_model = {
            "model_id": "critical-model-123",
            "model_type": "transformer",
            "version": "v3.1.0",
            "mode": "live",
            "storage_path": "models/live/transformer/v3.1.0/model.pkl.gz",
            "preservation_priority": "critical"
        }
        integration_manager.db_handler.get_active_models.return_value = [critical_model]
        
        # Mock storage operations
        integration_manager.storage_handler.load.return_value = sample_model_data
        integration_manager.db_handler.save_metadata.return_value = "emergency-backup-123"
        
        # Trigger emergency backup directly
        backed_up_models = await integration_manager.emergency_backup()
        
        # Verify emergency backup completed
        assert len(backed_up_models) == 1
        assert backed_up_models[0] == "emergency-backup-123"
        
        # Verify storage operations
        integration_manager.storage_handler.load.assert_called_once_with(
            "models/live/transformer/v3.1.0/model.pkl.gz"
        )
        
        # Verify emergency backup was saved with correct metadata
        save_call = integration_manager.storage_handler.save.call_args
        model_data, metadata = save_call[0]
        
        assert model_data == sample_model_data
        assert metadata.model_type == "transformer"
        assert "emergency_backup" in metadata.tags
        assert metadata.preservation_priority == PreservationPriority.CRITICAL
        assert metadata.mode == "live"
        assert "emergency" in metadata.version
    
    @pytest.mark.asyncio
    async def test_mode_change_preservation_flow(self, integration_manager, sample_model_data):
        """Test model preservation during mode changes"""
        # Setup source model in analysis mode
        source_metadata = {
            "model_id": "mode-change-source-123",
            "model_type": "lstm",
            "version": "v1.5.0", 
            "mode": "analysis",
            "storage_path": "models/analysis/lstm/v1.5.0/model.pkl.gz",
            "state": "active"
        }
        integration_manager.db_handler.get_metadata.return_value = source_metadata
        integration_manager.storage_handler.load.return_value = sample_model_data
        integration_manager.db_handler.save_metadata.return_value = "mode-change-target-123"
        
        # Test model migration from analysis to simulation mode
        migrated_model_id = await integration_manager.migrate_model(
            model_type="lstm",
            version="v1.5.0",
            from_mode="analysis", 
            to_mode="simulation"
        )
        
        assert migrated_model_id == "mode-change-target-123"
        
        # Verify source model was loaded
        integration_manager.db_handler.get_metadata.assert_called_with(
            model_type="lstm",
            version="v1.5.0",
            mode="analysis"
        )
        integration_manager.storage_handler.load.assert_called_with(
            "models/analysis/lstm/v1.5.0/model.pkl.gz"
        )
        
        # Verify model was saved to new mode
        save_call = integration_manager.storage_handler.save.call_args
        model_data, metadata = save_call[0]
        
        assert model_data == sample_model_data
        assert metadata.model_type == "lstm"
        assert metadata.version == "v1.5.0"
        assert metadata.mode == "simulation"
        assert "migrated_from_analysis" in metadata.tags
        assert metadata.metadata["migrated_from"] == "analysis"
        assert metadata.metadata["original_model_id"] == "mode-change-source-123"
    
    @pytest.mark.asyncio
    async def test_concurrent_save_operations_integration(self, integration_manager, sample_model_data):
        """Test concurrent model save operations don't interfere"""
        # Setup unique return values for each save
        model_ids = [f"concurrent-model-{i}" for i in range(5)]
        integration_manager.db_handler.save_metadata.side_effect = model_ids
        
        # Save multiple models concurrently
        tasks = []
        for i in range(5):
            task = integration_manager.save_model(
                model_data=sample_model_data,
                model_type="lstm",
                version=f"v1.0.{i}",
                mode="analysis",
                tags=[f"concurrent_test_{i}"]
            )
            tasks.append(task)
        
        # Wait for all saves to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all saves completed successfully
        assert len(results) == 5
        assert all(model_id in model_ids for model_id in results)
        
        # Verify storage handler was called for each save
        assert integration_manager.storage_handler.save.call_count == 5
        
        # Verify database handler was called for each save
        assert integration_manager.db_handler.save_metadata.call_count == 5
    
    @pytest.mark.asyncio
    async def test_version_limit_enforcement_integration(self, integration_manager, sample_model_data):
        """Test version limit enforcement in integration scenario"""
        integration_manager.config.max_versions_per_model = 3
        
        # Mock existing versions that exceed limit
        existing_versions = [
            {"version": "v1.0.0", "created_at": datetime.now() - timedelta(days=3), "mode": "analysis"},
            {"version": "v1.0.1", "created_at": datetime.now() - timedelta(days=2), "mode": "analysis"},
            {"version": "v1.0.2", "created_at": datetime.now() - timedelta(days=1), "mode": "analysis"},
        ]
        
        # Mock version retrieval for limit enforcement
        all_versions_after_save = existing_versions + [
            {"version": "v1.0.3", "created_at": datetime.now(), "mode": "analysis"}
        ]
        
        integration_manager.db_handler.get_versions.side_effect = [
            existing_versions,  # For version generation
            all_versions_after_save  # For limit check after save
        ]
        
        # Mock metadata retrieval for oldest version deletion
        integration_manager.db_handler.get_metadata.return_value = {
            "storage_path": "models/analysis/lstm/v1.0.0/model.pkl.gz"
        }
        
        integration_manager.db_handler.save_metadata.return_value = "version-limit-test-123"
        
        # Save new model that should trigger limit enforcement
        model_id = await integration_manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.3",
            mode="analysis"
        )
        
        assert model_id == "version-limit-test-123"
        
        # Verify oldest version was deleted from storage
        integration_manager.storage_handler.delete.assert_called_with(
            "models/analysis/lstm/v1.0.0/model.pkl.gz"
        )
        
        # Verify oldest version state was updated in database
        integration_manager.db_handler.update_state.assert_called_with(
            model_type="lstm",
            version="v1.0.0",
            state=ModelState.DELETED
        )
    
    @pytest.mark.asyncio
    async def test_background_backup_task_integration(self, integration_manager, sample_model_data):
        """Test background backup task execution"""
        # Configure very short backup interval for testing
        integration_manager.config.backup_interval_hours = 0.001  # ~3.6 seconds
        
        # Setup active models for backup
        active_models = [
            {
                "model_id": "background-backup-1",
                "model_type": "lstm",
                "version": "v1.0.0",
                "mode": "analysis",
                "storage_path": "models/analysis/lstm/v1.0.0/model.pkl.gz"
            }
        ]
        integration_manager.db_handler.get_active_models.return_value = active_models
        integration_manager.storage_handler.load.return_value = sample_model_data
        integration_manager.db_handler.save_metadata.return_value = "background-backup-result"
        
        # Start background backup task
        await integration_manager.start()
        
        # Wait for at least one backup cycle
        await asyncio.sleep(0.1)
        
        # Stop background task
        await integration_manager.stop()
        
        # Verify backup occurred
        assert integration_manager.storage_handler.save.call_count >= 1
        
        # Verify backup was tagged correctly
        save_calls = integration_manager.storage_handler.save.call_args_list
        backup_found = False
        for call in save_calls:
            metadata = call[0][1]
            if "auto_backup" in metadata.tags:
                backup_found = True
                break
        
        assert backup_found, "Auto backup should have been created"
    
    @pytest.mark.asyncio
    async def test_rollback_integration_flow(self, integration_manager, sample_model_data):
        """Test complete rollback integration flow"""
        # Setup current version metadata
        current_metadata = {
            "model_id": "rollback-current-123",
            "version": "v2.0.0",
            "storage_path": "models/analysis/lstm/v2.0.0/model.pkl.gz"
        }
        
        # Setup target version metadata
        target_metadata = {
            "model_id": "rollback-target-123", 
            "version": "v1.5.0",
            "storage_path": "models/analysis/lstm/v1.5.0/model.pkl.gz"
        }
        
        integration_manager.db_handler.get_metadata.side_effect = [
            current_metadata,  # Current version
            target_metadata    # Target version
        ]
        
        # Mock existing versions for new version generation
        integration_manager.db_handler.get_versions.return_value = [
            {"version": "v1.5.0"},
            {"version": "v2.0.0"}
        ]
        
        integration_manager.storage_handler.load.return_value = sample_model_data
        integration_manager.db_handler.save_metadata.return_value = "rollback-new-123"
        
        # Perform rollback
        result = await integration_manager.rollback_model(
            model_type="lstm",
            target_version="v1.5.0",
            mode="analysis"
        )
        
        # Verify rollback result
        assert result["rolled_back_from"] == "v2.0.0"
        assert result["rolled_back_to"] == "v1.5.0"  
        assert result["model_id"] == "rollback-new-123"
        assert "new_version" in result
        
        # Verify target version was loaded
        integration_manager.storage_handler.load.assert_called_with(
            "models/analysis/lstm/v1.5.0/model.pkl.gz"
        )
        
        # Verify new version was saved
        save_call = integration_manager.storage_handler.save.call_args
        model_data, metadata = save_call[0]
        
        assert model_data == sample_model_data
        assert "rollback" in metadata.tags
        assert f"from_v2.0.0" in metadata.tags
        assert metadata.metadata["rollback_from"] == "v2.0.0"
        assert metadata.metadata["rollback_to"] == "v1.5.0"
        
        # Verify rollback event was recorded
        integration_manager.db_handler.record_event.assert_called_with(
            model_id="rollback-new-123",
            event_type="rollback",
            details={
                "from_version": "v2.0.0",
                "to_version": "v1.5.0", 
                "new_version": result["new_version"]
            }
        )
    
    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, integration_manager):
        """Test performance tracking integration"""
        model_id = "performance-test-123"
        metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "inference_time_ms": 150,
            "memory_usage_mb": 512,
            "training_time_minutes": 45
        }
        
        # Track performance
        await integration_manager.track_performance(model_id, metrics)
        
        # Verify database handler was called
        integration_manager.db_handler.track_performance.assert_called_once_with(
            model_id=model_id,
            metrics=metrics
        )
    
    @pytest.mark.asyncio
    async def test_preservation_stats_integration(self, integration_manager):
        """Test preservation statistics retrieval integration"""
        # Get system stats
        stats = await integration_manager.get_stats()
        
        # Verify stats are properly calculated
        assert stats.total_models == 10
        assert stats.active_models == 5
        assert stats.total_size_mb == 50.0  # 50MB
        assert stats.models_by_type["lstm"] == 5
        assert stats.models_by_type["dqn"] == 5
        assert stats.models_by_mode["analysis"] == 3
        assert stats.models_by_mode["simulation"] == 4
        assert stats.models_by_mode["live"] == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integration_manager, sample_model_data):
        """Test error handling in integration scenarios"""
        # Test storage failure during save
        integration_manager.storage_handler.save.side_effect = Exception("Storage service unavailable")
        
        with pytest.raises(PreservationError, match="Failed to save model"):
            await integration_manager.save_model(
                model_data=sample_model_data,
                model_type="lstm",
                version="v1.0.0"
            )
        
        # Reset storage handler
        integration_manager.storage_handler.save.side_effect = None
        integration_manager.storage_handler.save.return_value = "models/lstm/v1.0.0/model.pkl.gz"
        
        # Test database failure during metadata save
        integration_manager.db_handler.save_metadata.side_effect = Exception("Database connection lost")
        
        with pytest.raises(PreservationError, match="Failed to save model"):
            await integration_manager.save_model(
                model_data=sample_model_data,
                model_type="lstm",
                version="v1.0.0"
            )
    
    @pytest.mark.asyncio
    async def test_mode_isolation_integration(self, integration_manager, sample_model_data):
        """Test mode isolation in integration scenarios"""
        assert integration_manager.config.mode_isolation is True
        
        # Save model in live mode
        integration_manager.db_handler.save_metadata.return_value = "mode-isolation-live-123"
        
        await integration_manager.save_model(
            model_data=sample_model_data,
            model_type="lstm",
            version="v1.0.0",
            mode="live"
        )
        
        # Verify mode was set in metadata
        save_call = integration_manager.storage_handler.save.call_args
        metadata = save_call[0][1]
        assert metadata.mode == "live"
        
        # Try to load from different mode (should fail)
        integration_manager.db_handler.get_metadata.return_value = None
        
        with pytest.raises(FileNotFoundError, match="Model not found"):
            await integration_manager.load_model(
                model_type="lstm",
                version="v1.0.0", 
                mode="analysis"  # Different mode
            )
        
        # Verify metadata was queried with correct mode
        integration_manager.db_handler.get_metadata.assert_called_with(
            model_type="lstm",
            version="v1.0.0",
            mode="analysis"
        )


# Additional integration test scenarios
class TestAdvancedIntegrationScenarios:
    """Advanced integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_signal_handling_integration(self, integration_config):
        """Test signal handling integration with real signal registration"""
        with patch('signal.signal') as mock_signal:
            manager = PreservationManager(integration_config)
            
            # Verify signal handlers were registered
            assert mock_signal.call_count >= 2
            
            # Check specific signals
            calls = mock_signal.call_args_list
            signals_registered = [call[0][0] for call in calls]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered
    
    @pytest.mark.asyncio 
    async def test_cleanup_integration(self, integration_manager):
        """Test cleanup operations integration"""
        # Mock old versions for cleanup
        old_versions = [
            {
                "version": "v0.1.0",
                "created_at": datetime.now() - timedelta(days=45),
                "storage_path": "models/lstm/v0.1.0/model.pkl.gz",
                "model_type": "lstm"
            },
            {
                "version": "v0.2.0", 
                "created_at": datetime.now() - timedelta(days=35),
                "storage_path": "models/lstm/v0.2.0/model.pkl.gz",
                "model_type": "lstm"
            }
        ]
        
        integration_manager.db_handler.get_old_versions = AsyncMock(return_value=old_versions)
        
        # Run cleanup
        deleted_count = await integration_manager.cleanup_old_versions(days=30)
        
        # Verify cleanup results
        assert deleted_count == 2
        
        # Verify storage deletions
        assert integration_manager.storage_handler.delete.call_count == 2
        expected_calls = [
            "models/lstm/v0.1.0/model.pkl.gz",
            "models/lstm/v0.2.0/model.pkl.gz"
        ]
        actual_calls = [call[0][0] for call in integration_manager.storage_handler.delete.call_args_list]
        assert all(path in actual_calls for path in expected_calls)
        
        # Verify database state updates
        assert integration_manager.db_handler.update_state.call_count == 2