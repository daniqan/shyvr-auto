"""
Comprehensive tests for version tagging functionality in model preservation system

Tests the tagging system that supports tags like "latest", "stable", "experimental", etc.
Following TDD methodology - these tests are written first before implementation.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from src.model_preservation.base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    VersionError
)
from src.model_preservation.versioning import SemanticVersion


class TestModelTag:
    """Test ModelTag dataclass and basic functionality"""
    
    def test_model_tag_creation(self):
        """Test creating a ModelTag with valid data"""
        from src.model_preservation.base import ModelTag
        
        tag = ModelTag(
            tag_name="latest",
            model_type="dqn",
            version="v1.2.3",
            created_at=datetime.now(timezone.utc),
            description="Latest stable version"
        )
        assert tag.tag_name == "latest"
        assert tag.model_type == "dqn"
        assert tag.version == "v1.2.3"
        assert tag.is_standard_tag is True
        assert tag.tag_id == "dqn-latest"
    
    def test_model_tag_validation(self):
        """Test ModelTag validation for invalid inputs"""
        from src.model_preservation.base import ModelTag
        
        # Test invalid tag name
        with pytest.raises(ValueError, match="Invalid tag name"):
            ModelTag(
                tag_name="invalid tag name",  # Spaces not allowed
                model_type="dqn",
                version="v1.2.3",
                created_at=datetime.now(timezone.utc)
            )
        
        # Test invalid version
        with pytest.raises(ValueError, match="Invalid version format"):
            ModelTag(
                tag_name="latest",
                model_type="dqn",
                version="invalid_version",
                created_at=datetime.now(timezone.utc)
            )
    
    def test_standard_tag_constants(self):
        """Test that standard tag constants are defined"""
        from src.model_preservation.base import StandardTags
        
        assert StandardTags.LATEST == "latest"
        assert StandardTags.STABLE == "stable"
        assert StandardTags.EXPERIMENTAL == "experimental"
        
        all_tags = StandardTags.get_all()
        assert "latest" in all_tags
        assert "stable" in all_tags
        assert "experimental" in all_tags
        
        assert StandardTags.is_standard_tag("latest") is True
        assert StandardTags.is_standard_tag("custom") is False


class TestDatabaseTagOperations:
    """Test database operations for version tagging"""
    
    @pytest.fixture
    def mock_db_handler(self):
        """Mock database handler for testing"""
        from src.model_preservation.db_handler import DatabaseHandler
        handler = DatabaseHandler()
        handler._initialized = True
        return handler
    
    @pytest.mark.asyncio
    async def test_save_tag(self, mock_db_handler):
        """Test saving a tag to database"""
        from src.model_preservation.base import ModelTag
        
        tag = ModelTag(
            tag_name="latest",
            model_type="dqn", 
            version="v1.2.3",
            created_at=datetime.now(timezone.utc)
        )
        
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            mock_conn.return_value.__aenter__.return_value.fetchval.side_effect = [
                "dqn-v1.2.3-preservation-id",  # preservation_exists check
                "tag_id"  # save_tag result
            ]
            
            result = await mock_db_handler.save_tag(tag)
            assert result == "tag_id"
    
    @pytest.mark.asyncio
    async def test_get_tag(self, mock_db_handler):
        """Test retrieving a tag from database"""
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            mock_row = {
                'tag_id': 'dqn-latest',
                'tag_name': 'latest',
                'model_type': 'dqn',
                'version': 'v1.2.3',
                'preservation_id': 'dqn-v1.2.3-12345678',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'description': 'Latest version',
                'metadata': None
            }
            mock_conn.return_value.__aenter__.return_value.fetchrow.return_value = mock_row
            
            tag = await mock_db_handler.get_tag("dqn", "latest")
            assert tag["tag_name"] == "latest"
            assert tag["version"] == "v1.2.3"
            assert tag["model_type"] == "dqn"
    
    @pytest.mark.asyncio
    async def test_update_tag(self, mock_db_handler):
        """Test updating a tag to point to new version"""
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            # Mock preservation_id lookup and update
            mock_conn.return_value.__aenter__.return_value.fetchval.return_value = "dqn-v1.3.0-preservation-id"
            mock_conn.return_value.__aenter__.return_value.execute.return_value = "UPDATE 1"
            
            await mock_db_handler.update_tag("dqn", "latest", "v1.3.0")
            
            # Verify database update was called
            mock_conn.return_value.__aenter__.return_value.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_delete_tag(self, mock_db_handler):
        """Test deleting a tag from database"""
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            mock_conn.return_value.__aenter__.return_value.execute.return_value = "DELETE 1"
            
            await mock_db_handler.delete_tag("dqn", "custom-tag")
            
            # Verify database deletion was called
            mock_conn.return_value.__aenter__.return_value.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_list_tags_for_model_type(self, mock_db_handler):
        """Test listing all tags for a model type"""
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            mock_rows = [
                {
                    'tag_id': 'dqn-latest',
                    'tag_name': 'latest', 
                    'model_type': 'dqn',
                    'version': 'v1.2.3',
                    'preservation_id': 'dqn-v1.2.3-12345678',
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'description': 'Latest version',
                    'metadata': None
                },
                {
                    'tag_id': 'dqn-stable',
                    'tag_name': 'stable',
                    'model_type': 'dqn', 
                    'version': 'v1.2.0',
                    'preservation_id': 'dqn-v1.2.0-12345678',
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'description': 'Stable version',
                    'metadata': None
                }
            ]
            mock_conn.return_value.__aenter__.return_value.fetch.return_value = mock_rows
            
            tags = await mock_db_handler.list_tags("dqn")
            assert len(tags) == 2
            assert tags[0]["tag_name"] == "latest"
            assert tags[1]["tag_name"] == "stable"
    
    @pytest.mark.asyncio
    async def test_resolve_tag_to_version(self, mock_db_handler):
        """Test resolving a tag name to its version"""
        with patch('src.model_preservation.db_handler.get_database_connection') as mock_conn:
            mock_conn.return_value.__aenter__.return_value.fetchval.return_value = "v1.2.3"
            
            version = await mock_db_handler.resolve_tag_to_version("dqn", "latest")
            assert version == "v1.2.3"


class TestPreservationManagerTagging:
    """Test tagging functionality in PreservationManager"""
    
    @pytest.fixture
    def mock_preservation_manager(self):
        """Mock preservation manager for testing"""
        from src.model_preservation.manager import PreservationManager, PreservationConfig
        config = PreservationConfig(gcs_bucket="test-bucket")
        manager = PreservationManager(config)
        manager._is_running = True
        return manager
    
    @pytest.mark.asyncio
    async def test_tag_model(self, mock_preservation_manager):
        """Test tagging a model with a specific tag"""
        # Mock the db_handler
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.save_tag.return_value = "tag_id"
        mock_preservation_manager.db_handler.record_event.return_value = None
        
        result = await mock_preservation_manager.tag_model(
            model_type="dqn",
            version="v1.2.3",
            tag_name="stable",
            description="Stable release"
        )
        assert result == "tag_id"
        
        # Verify tag was saved
        mock_preservation_manager.db_handler.save_tag.assert_called_once()
        mock_preservation_manager.db_handler.record_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resolve_tag(self, mock_preservation_manager):
        """Test resolving a tag to get version"""
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.resolve_tag_to_version.return_value = "v1.2.3"
        
        version = await mock_preservation_manager.resolve_tag("dqn", "latest")
        assert version == "v1.2.3"
        
        mock_preservation_manager.db_handler.resolve_tag_to_version.assert_called_once_with("dqn", "latest", branch="main")
    
    @pytest.mark.asyncio
    async def test_load_model_by_tag(self, mock_preservation_manager):
        """Test loading a model using a tag instead of version"""
        # Mock db_handler for tag resolution and metadata retrieval
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.branch_exists.return_value = True
        mock_preservation_manager.db_handler.resolve_tag_to_version.return_value = "v1.2.3"
        mock_preservation_manager.db_handler.get_metadata.return_value = {
            "model_id": "dqn-v1.2.3-12345678",
            "model_type": "dqn",
            "version": "v1.2.3",
            "storage_path": "gs://bucket/dqn-v1.2.3.pkl"
        }
        mock_preservation_manager.db_handler.record_event.return_value = None
        
        # Mock storage handler
        mock_preservation_manager.storage_handler = AsyncMock()
        mock_preservation_manager.storage_handler.load.return_value = b"model_data"
        
        # Mock cache manager (disable caching for this test)
        mock_preservation_manager.cache_manager = None
        
        data, metadata = await mock_preservation_manager.load_model(
            model_type="dqn",
            version="latest"  # This should resolve to v1.2.3
        )
        
        assert data == b"model_data"
        assert metadata["version"] == "v1.2.3"
        assert metadata["model_type"] == "dqn"
        
        # Verify tag resolution was called
        mock_preservation_manager.db_handler.resolve_tag_to_version.assert_called_once_with("dqn", "latest", branch="main")
    
    @pytest.mark.asyncio
    async def test_move_tag(self, mock_preservation_manager):
        """Test moving a tag from one version to another"""
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.update_tag.return_value = None
        mock_preservation_manager.db_handler.record_event.return_value = None
        
        await mock_preservation_manager.move_tag("dqn", "latest", "v1.3.0", description="Updated to latest")
        
        mock_preservation_manager.db_handler.update_tag.assert_called_once_with(
            model_type="dqn", 
            tag_name="latest", 
            new_version="v1.3.0", 
            description="Updated to latest"
        )
        mock_preservation_manager.db_handler.record_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_model_tags(self, mock_preservation_manager):
        """Test listing all tags for a model type"""
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.list_tags.return_value = [
            {"tag_name": "latest", "version": "v1.2.3", "description": "Latest version"},
            {"tag_name": "stable", "version": "v1.2.0", "description": "Stable version"}
        ]
        
        tags = await mock_preservation_manager.list_model_tags("dqn")
        assert len(tags) == 2
        assert tags[0]["tag_name"] == "latest"
        assert tags[1]["tag_name"] == "stable"
        
        mock_preservation_manager.db_handler.list_tags.assert_called_once_with("dqn")


class TestAutomaticTagging:
    """Test automatic tagging logic"""
    
    @pytest.fixture
    def mock_preservation_manager(self):
        """Mock preservation manager for testing"""
        from src.model_preservation.manager import PreservationManager, PreservationConfig
        config = PreservationConfig(gcs_bucket="test-bucket")
        manager = PreservationManager(config)
        manager._is_running = True
        return manager
    
    @pytest.mark.asyncio
    async def test_auto_tag_latest_on_save(self, mock_preservation_manager):
        """Test that 'latest' tag is automatically updated when saving a new model"""
        # Mock all required dependencies
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.branch_exists.return_value = True
        mock_preservation_manager.db_handler.save_metadata.return_value = "dqn-v1.3.0-12345678"
        mock_preservation_manager.db_handler.record_event.return_value = None
        mock_preservation_manager.db_handler.update_standard_tags.return_value = None
        
        mock_preservation_manager.storage_handler = AsyncMock()
        mock_preservation_manager.storage_handler.save.return_value = "gs://bucket/dqn-v1.3.0.pkl"
        
        model_id = await mock_preservation_manager.save_model(
            model_data=b"test_data",
            model_type="dqn",
            version="v1.3.0"
        )
        
        # Verify update_standard_tags was called (this updates latest and stable automatically)
        mock_preservation_manager.db_handler.update_standard_tags.assert_called_once()
        
        # Get the call args to verify the parameters
        call_args = mock_preservation_manager.db_handler.update_standard_tags.call_args
        assert call_args[1]["model_type"] == "dqn"
        assert call_args[1]["new_version"] == "v1.3.0"
    
    @pytest.mark.asyncio
    async def test_auto_tag_stable(self, mock_preservation_manager):
        """Test that 'stable' tag points to newest non-prerelease version"""
        # Mock db_handler
        mock_preservation_manager.db_handler = AsyncMock()
        
        # Mock get_stable_versions to return stable versions in descending order
        mock_preservation_manager.get_stable_versions = AsyncMock()
        mock_preservation_manager.get_stable_versions.return_value = [
            "v1.2.1",  # Latest stable (will be used for stable tag)
            "v1.2.0"   # Older stable
        ]
        
        # Mock move_tag method
        mock_preservation_manager.db_handler.update_tag.return_value = None
        mock_preservation_manager.db_handler.record_event.return_value = None
        
        latest_stable = await mock_preservation_manager.update_stable_tag("dqn")
        
        # Verify stable tag was updated to latest stable version
        assert latest_stable == "v1.2.1"
        mock_preservation_manager.get_stable_versions.assert_called_once_with("dqn", "analysis")
    
    @pytest.mark.asyncio
    async def test_experimental_tag_manual_only(self, mock_preservation_manager):
        """Test that 'experimental' tag is only set manually, not automatically"""
        # Mock all required dependencies for save_model
        mock_preservation_manager.db_handler = AsyncMock()
        mock_preservation_manager.db_handler.branch_exists.return_value = True
        mock_preservation_manager.db_handler.save_metadata.return_value = "dqn-v1.3.0-alpha-12345678"
        mock_preservation_manager.db_handler.record_event.return_value = None
        mock_preservation_manager.db_handler.update_standard_tags.return_value = None
        
        mock_preservation_manager.storage_handler = AsyncMock()
        mock_preservation_manager.storage_handler.save.return_value = "gs://bucket/dqn-v1.3.0-alpha.pkl"
        
        # Mock tag_model to track calls
        mock_preservation_manager.tag_model = AsyncMock()
        
        model_id = await mock_preservation_manager.save_model(
            model_data=b"test_data",
            model_type="dqn",
            version="v1.3.0-alpha"  # Prerelease version
        )
        
        # Verify update_standard_tags was called (handles latest/stable automatically)
        mock_preservation_manager.db_handler.update_standard_tags.assert_called_once()
        
        # Verify tag_model was NOT called for experimental tag
        # (experimental tags must be set manually)
        mock_preservation_manager.tag_model.assert_not_called()


class TestTagValidation:
    """Test tag validation and error handling"""
    
    def test_tag_name_validation(self):
        """Test tag name validation rules"""
        from src.model_preservation.base import is_valid_tag_name
        
        # Valid tag names
        valid_names = ["latest", "stable", "experimental", "release-1", "feature_branch", "v1-prod", "a", "test123"]
        for name in valid_names:
            assert is_valid_tag_name(name), f"'{name}' should be valid"
        
        # Invalid tag names
        invalid_names = ["", " ", "latest ", " stable", "tag with spaces", "tag@special", "tag!invalid", "--invalid", "a" * 51]
        for name in invalid_names:
            assert not is_valid_tag_name(name), f"'{name}' should be invalid"
    
    def test_reserved_tag_names(self):
        """Test that reserved tag names are properly handled"""
        from src.model_preservation.base import is_reserved_tag_name
        
        reserved_names = ["latest", "stable", "experimental"]
        for name in reserved_names:
            assert is_reserved_tag_name(name), f"'{name}' should be reserved"
        
        custom_names = ["feature-1", "release-candidate", "custom"]
        for name in custom_names:
            assert not is_reserved_tag_name(name), f"'{name}' should not be reserved"
    
    @pytest.mark.asyncio
    async def test_duplicate_tag_handling(self):
        """Test handling of duplicate tag names"""
        from src.model_preservation.manager import PreservationManager, PreservationConfig
        from src.model_preservation.base import ModelTag
        
        config = PreservationConfig(gcs_bucket="test-bucket")
        manager = PreservationManager(config)
        manager.db_handler = AsyncMock()
        
        # Mock successful tag saves - the database schema handles uniqueness via UPSERT
        manager.db_handler.save_tag.return_value = "tag_id"
        manager.db_handler.record_event.return_value = None
        
        # Create tag first time
        tag_id_1 = await manager.tag_model("dqn", "v1.0.0", "latest", description="First latest")
        assert tag_id_1 == "tag_id"
        
        # Create same tag name with different version (should update existing)  
        tag_id_2 = await manager.tag_model("dqn", "v1.1.0", "latest", description="Updated latest")
        assert tag_id_2 == "tag_id"
        
        # Verify save_tag was called twice (one for each tag operation)
        assert manager.db_handler.save_tag.call_count == 2
    
    @pytest.mark.asyncio
    async def test_tag_to_nonexistent_version(self):
        """Test error handling when tagging a nonexistent version"""
        from src.model_preservation.manager import PreservationManager, PreservationConfig
        from src.model_preservation.base import PreservationError
        
        config = PreservationConfig(gcs_bucket="test-bucket")
        manager = PreservationManager(config)
        manager.db_handler = AsyncMock()
        
        # Mock save_tag to raise error for nonexistent version
        manager.db_handler.save_tag.side_effect = ValueError("No preserved model found for dqn version v99.99.99")
        
        with pytest.raises(ValueError, match="No preserved model found"):
            await manager.tag_model("dqn", "v99.99.99", "latest")


class TestTagSpecialCases:
    """Test special cases and edge conditions for tagging"""
    
    @pytest.mark.asyncio
    async def test_tag_with_same_version(self):
        """Test creating multiple tags pointing to same version"""
        # TODO: Test multiple tags per version once implemented
        pytest.skip("Multiple tags per version not yet implemented")
    
    @pytest.mark.asyncio
    async def test_tag_across_modes(self):
        """Test tagging behavior across different operational modes"""
        # TODO: Test cross-mode tagging once implemented
        # Tags should be mode-specific if mode_isolation is enabled
        pytest.skip("Cross-mode tagging not yet implemented")
    
    @pytest.mark.asyncio
    async def test_tag_cleanup_on_version_deletion(self):
        """Test that tags are cleaned up when versions are deleted"""
        # TODO: Test tag cleanup once implemented
        pytest.skip("Tag cleanup not yet implemented")
    
    @pytest.mark.asyncio
    async def test_tag_history_tracking(self):
        """Test tracking of tag changes over time"""
        # TODO: Test tag history once implemented
        # This might be a future enhancement
        pytest.skip("Tag history not yet implemented")


class TestTagPerformance:
    """Test performance characteristics of tagging operations"""
    
    @pytest.mark.asyncio
    async def test_tag_resolution_performance(self):
        """Test that tag resolution is fast enough"""
        # TODO: Test performance once implemented
        # Resolution should be <10ms for cached tags
        pytest.skip("Performance testing not yet implemented")
    
    @pytest.mark.asyncio
    async def test_batch_tag_operations(self):
        """Test batch operations for multiple tags"""
        # TODO: Test batch operations once implemented
        # Useful for updating multiple tags efficiently
        pytest.skip("Batch operations not yet implemented")


# Integration test fixtures
@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return b"fake_model_data_for_testing"


@pytest.fixture
def sample_versions():
    """Sample version list for testing"""
    return [
        "v1.0.0",
        "v1.1.0", 
        "v1.1.1",
        "v1.2.0-alpha",
        "v1.2.0-beta.1",
        "v1.2.0",
        "v2.0.0-rc.1"
    ]


# Test utilities
def assert_tag_points_to_version(tag_data: Dict[str, Any], expected_version: str):
    """Utility to assert tag points to expected version"""
    assert tag_data["version"] == expected_version, f"Tag should point to {expected_version}, got {tag_data['version']}"


def assert_semantic_version_order(versions: List[str]):
    """Utility to assert versions are in semantic order"""
    semantic_versions = [SemanticVersion(v) for v in versions]
    sorted_versions = sorted(semantic_versions)
    assert semantic_versions == sorted_versions, "Versions should be in semantic order"