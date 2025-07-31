"""
Tests for model preservation branch support functionality

This module tests Phase 2.1.4 - Branch Support for Experimental Models.
Tests branch creation, validation, saving, loading, merging, and branch isolation.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from src.model_preservation.base import (
    ModelMetadata,
    PreservationPriority,
    ModelState,
    PreservationError,
    VersionError,
    generate_model_id,
    calculate_checksum
)
from src.model_preservation.manager import PreservationManager, PreservationConfig


# =============================================================================
# BRANCH VALIDATION TESTS
# =============================================================================

class TestBranchValidation:
    """Test branch name validation and constants"""
    
    def test_default_branch_constant(self):
        """Test that DEFAULT_BRANCH constant exists and has correct value"""
        from src.model_preservation.base import DEFAULT_BRANCH
        assert DEFAULT_BRANCH == "main"
    
    def test_valid_branch_names(self):
        """Test validation of valid branch names"""
        from src.model_preservation.base import validate_branch_name
        
        valid_names = [
            "main",
            "develop", 
            "feature-123",
            "experimental",
            "hotfix_2024",
            "release-v1.0",
            "test_branch",
            "model-experiments",
            "a",  # single character
            "a" * 63  # max length
        ]
        
        for name in valid_names:
            assert validate_branch_name(name), f"'{name}' should be valid"
    
    def test_invalid_branch_names(self):
        """Test validation of invalid branch names"""
        from src.model_preservation.base import validate_branch_name
        
        invalid_names = [
            "",  # empty
            " ",  # whitespace only
            "branch with spaces",
            "branch@symbol",
            "branch#hash",
            "branch$dollar",
            "branch%percent",
            "branch^caret",
            "branch&ampersand",
            "branch*asterisk",
            "branch(paren",
            "branch)paren",
            "branch+plus",
            "branch=equals",
            "branch[bracket",
            "branch]bracket",
            "branch{brace",
            "branch}brace",
            "branch\\backslash",
            "branch|pipe",
            "branch;semicolon",
            "branch:colon",
            "branch\"quote",
            "branch'quote",
            "branch<less",
            "branch>greater",
            "branch,comma",
            "branch.dot.dot",
            "branch/slash",
            "branch?question",
            "-startswithdash",
            "_startswithunderscore",
            "endswithdasd-",
            "endswithunderscore_",
            "a" * 64,  # too long
            "BRANCH",  # uppercase not allowed in this system
        ]
        
        for name in invalid_names:
            assert not validate_branch_name(name), f"'{name}' should be invalid"
    
    def test_branch_uniqueness_per_model_type(self):
        """Test that branches are unique per model type"""
        # This will be tested in integration tests with the database
        pass


# =============================================================================
# MODEL METADATA BRANCH EXTENSION TESTS
# =============================================================================

class TestModelMetadataBranchExtension:
    """Test ModelMetadata extension with branch support"""
    
    def test_model_metadata_with_branch(self):
        """Test creating ModelMetadata with branch field"""
        metadata = ModelMetadata(
            model_id="test-model-123",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now(),
            branch="experimental"
        )
        
        assert metadata.branch == "experimental"
        assert metadata.model_type == "dqn"
        assert metadata.version == "v1.0.0"
    
    def test_model_metadata_default_branch(self):
        """Test ModelMetadata defaults to main branch"""
        metadata = ModelMetadata(
            model_id="test-model-123",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now()
        )
        
        from src.model_preservation.base import DEFAULT_BRANCH
        assert metadata.branch == DEFAULT_BRANCH
    
    def test_model_metadata_branch_validation(self):
        """Test ModelMetadata validates branch names"""
        # Valid branch should work
        metadata = ModelMetadata(
            model_id="test-model-123",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now(),
            branch="valid-branch-name"
        )
        assert metadata.branch == "valid-branch-name"
        
        # Invalid branch should raise error
        with pytest.raises(ValueError, match="Invalid branch name"):
            ModelMetadata(
                model_id="test-model-123",
                model_type="dqn",
                version="v1.0.0",
                created_at=datetime.now(),
                branch="invalid branch name"
            )
    
    def test_model_metadata_branch_uniqueness_key(self):
        """Test that branch is included in uniqueness calculation"""
        # Same model in different branches should be considered different
        metadata1 = ModelMetadata(
            model_id="test-model-123",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now(),
            branch="main"
        )
        
        metadata2 = ModelMetadata(
            model_id="test-model-124",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now(),
            branch="experimental"
        )
        
        # They should have different uniqueness keys
        assert metadata1.get_uniqueness_key() != metadata2.get_uniqueness_key()
        
        # Same branch should have same uniqueness key for same model/version
        metadata3 = ModelMetadata(
            model_id="test-model-125",
            model_type="dqn",
            version="v1.0.0",
            created_at=datetime.now(),
            branch="main"
        )
        
        assert metadata1.get_uniqueness_key() == metadata3.get_uniqueness_key()


# =============================================================================
# BRANCH OPERATIONS TESTS
# =============================================================================

class TestBranchOperations:
    """Test branch creation, listing, and management operations"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock preservation config"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=1.0,
            max_versions_per_model=5,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=False,
            emergency_backup=False
        )
    
    @pytest.fixture
    def mock_manager(self, mock_config):
        """Mock preservation manager with mocked handlers"""
        with patch('src.model_preservation.manager.GCSHandler') as mock_gcs, \
             patch('src.model_preservation.manager.DatabaseHandler') as mock_db:
            
            manager = PreservationManager(mock_config)
            manager.storage_handler = AsyncMock()
            manager.db_handler = AsyncMock()
            manager.db_handler.initialize = AsyncMock()
            manager.storage_handler.initialize = AsyncMock()
            
            return manager
    
    @pytest.mark.asyncio
    async def test_create_branch(self, mock_manager):
        """Test creating a new branch"""
        # Mock database operations
        mock_manager.db_handler.create_branch = AsyncMock(return_value="branch-id-123")
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=False)
        mock_manager.db_handler.record_event = AsyncMock()
        
        branch_id = await mock_manager.create_branch(
            branch_name="experimental",
            source_branch="main",
            description="Experimental model development"
        )
        
        assert branch_id == "branch-id-123"
        mock_manager.db_handler.create_branch.assert_called_once_with(
            branch_name="experimental",
            source_branch="main", 
            description="Experimental model development"
        )
        mock_manager.db_handler.record_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_branch_already_exists(self, mock_manager):
        """Test creating a branch that already exists"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        
        with pytest.raises(ValueError, match="Branch 'experimental' already exists"):
            await mock_manager.create_branch(
                branch_name="experimental",
                source_branch="main"
            )
    
    @pytest.mark.asyncio
    async def test_create_branch_invalid_name(self, mock_manager):
        """Test creating a branch with invalid name"""
        with pytest.raises(ValueError, match="Invalid branch name"):
            await mock_manager.create_branch(
                branch_name="invalid branch name",
                source_branch="main"
            )
    
    @pytest.mark.asyncio
    async def test_list_branches(self, mock_manager):
        """Test listing all branches"""
        expected_branches = [
            {"name": "main", "created_at": datetime.now(), "model_count": 5},
            {"name": "experimental", "created_at": datetime.now(), "model_count": 2},
            {"name": "hotfix", "created_at": datetime.now(), "model_count": 1}
        ]
        
        mock_manager.db_handler.list_branches = AsyncMock(return_value=expected_branches)
        
        branches = await mock_manager.list_branches()
        
        assert len(branches) == 3
        assert branches[0]["name"] == "main"
        assert branches[1]["name"] == "experimental"
        assert branches[2]["name"] == "hotfix"
        mock_manager.db_handler.list_branches.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_branch(self, mock_manager):
        """Test deleting a branch"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_branch_model_count = AsyncMock(return_value=0)
        mock_manager.db_handler.delete_branch = AsyncMock()
        mock_manager.db_handler.record_event = AsyncMock()
        
        await mock_manager.delete_branch("experimental")
        
        mock_manager.db_handler.delete_branch.assert_called_once_with("experimental")
        mock_manager.db_handler.record_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_branch_not_exists(self, mock_manager):
        """Test deleting a branch that doesn't exist"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=False)
        
        with pytest.raises(ValueError, match="Branch 'nonexistent' does not exist"):
            await mock_manager.delete_branch("nonexistent")
    
    @pytest.mark.asyncio
    async def test_delete_main_branch(self, mock_manager):
        """Test that main branch cannot be deleted"""
        with pytest.raises(ValueError, match="Cannot delete main branch"):
            await mock_manager.delete_branch("main")
    
    @pytest.mark.asyncio
    async def test_delete_branch_with_models(self, mock_manager):
        """Test deleting a branch that contains models"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_branch_model_count = AsyncMock(return_value=3)
        
        with pytest.raises(ValueError, match="Cannot delete branch 'experimental' containing 3 models"):
            await mock_manager.delete_branch("experimental", force=False)
    
    @pytest.mark.asyncio
    async def test_delete_branch_with_models_force(self, mock_manager):
        """Test force deleting a branch that contains models"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_branch_model_count = AsyncMock(return_value=3)
        mock_manager.db_handler.delete_branch = AsyncMock()
        mock_manager.db_handler.record_event = AsyncMock()
        
        await mock_manager.delete_branch("experimental", force=True)
        
        mock_manager.db_handler.delete_branch.assert_called_once_with("experimental")


# =============================================================================
# BRANCH-AWARE MODEL OPERATIONS TESTS  
# =============================================================================

class TestBranchAwareModelOperations:
    """Test saving and loading models with branch support"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock preservation config"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=1.0,
            max_versions_per_model=5,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=False,
            emergency_backup=False
        )
    
    @pytest.fixture
    def mock_manager(self, mock_config):
        """Mock preservation manager with mocked handlers"""
        with patch('src.model_preservation.manager.GCSHandler') as mock_gcs, \
             patch('src.model_preservation.manager.DatabaseHandler') as mock_db:
            
            manager = PreservationManager(mock_config)
            manager.storage_handler = AsyncMock()
            manager.db_handler = AsyncMock()
            manager.db_handler.initialize = AsyncMock()
            manager.storage_handler.initialize = AsyncMock()
            
            return manager
    
    @pytest.mark.asyncio
    async def test_save_model_with_branch(self, mock_manager):
        """Test saving a model to a specific branch"""
        test_data = b"model data"
        
        # Mock the necessary operations
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_versions = AsyncMock(return_value=[])
        mock_manager.storage_handler.save = AsyncMock(return_value="gcs://test-bucket/model.pkl")
        mock_manager.db_handler.save_metadata = AsyncMock(return_value="model-id-123")
        mock_manager.db_handler.record_event = AsyncMock()
        mock_manager.db_handler.update_standard_tags = AsyncMock()
        
        model_id = await mock_manager.save_model(
            model_data=test_data,
            model_type="dqn",
            branch="experimental",
            mode="analysis",
            tags=["test"],
            metadata={"experiment": "branch_test"}
        )
        
        assert model_id == "model-id-123"
        
        # Verify branch existence was checked
        mock_manager.db_handler.branch_exists.assert_called_once_with("experimental")
        
        # Verify model was saved with branch information
        call_args = mock_manager.db_handler.save_metadata.call_args[0][0]
        assert call_args.branch == "experimental"
        assert call_args.model_type == "dqn"
        assert call_args.mode == "analysis"
    
    @pytest.mark.asyncio 
    async def test_save_model_to_nonexistent_branch(self, mock_manager):
        """Test saving a model to a non-existent branch"""
        test_data = b"model data"
        
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=False)
        
        with pytest.raises(ValueError, match="Branch 'nonexistent' does not exist"):
            await mock_manager.save_model(
                model_data=test_data,
                model_type="dqn",
                branch="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_save_model_default_branch(self, mock_manager):
        """Test saving a model to default (main) branch"""
        test_data = b"model data"
        
        # Mock the necessary operations  
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_versions = AsyncMock(return_value=[])
        mock_manager.storage_handler.save = AsyncMock(return_value="gcs://test-bucket/model.pkl")
        mock_manager.db_handler.save_metadata = AsyncMock(return_value="model-id-123")
        mock_manager.db_handler.record_event = AsyncMock()
        mock_manager.db_handler.update_standard_tags = AsyncMock()
        
        model_id = await mock_manager.save_model(
            model_data=test_data,
            model_type="dqn",
            mode="analysis"
            # No branch specified, should default to main
        )
        
        assert model_id == "model-id-123"
        
        # Verify it used the main branch
        call_args = mock_manager.db_handler.save_metadata.call_args[0][0]
        assert call_args.branch == "main"
    
    @pytest.mark.asyncio
    async def test_load_model_from_branch(self, mock_manager):
        """Test loading a model from a specific branch"""
        expected_data = b"model data"
        expected_metadata = {
            "model_id": "model-123",
            "model_type": "dqn", 
            "version": "v1.0.0",
            "branch": "experimental",
            "storage_path": "gcs://test-bucket/model.pkl"
        }
        
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_metadata = AsyncMock(return_value=expected_metadata)
        mock_manager.storage_handler.load = AsyncMock(return_value=expected_data)
        mock_manager.db_handler.record_event = AsyncMock()
        
        model_data, metadata = await mock_manager.load_model(
            model_type="dqn",
            version="v1.0.0",
            branch="experimental",
            mode="analysis"
        )
        
        assert model_data == expected_data
        assert metadata == expected_metadata
        
        # Verify branch existence was checked
        mock_manager.db_handler.branch_exists.assert_called_once_with("experimental")
        
        # Verify metadata query included branch
        mock_manager.db_handler.get_metadata.assert_called_once_with(
            model_type="dqn",
            version="v1.0.0", 
            mode="analysis",
            branch="experimental"
        )
    
    @pytest.mark.asyncio
    async def test_load_model_from_nonexistent_branch(self, mock_manager):
        """Test loading a model from a non-existent branch"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=False)
        
        with pytest.raises(ValueError, match="Branch 'nonexistent' does not exist"):
            await mock_manager.load_model(
                model_type="dqn",
                branch="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_load_model_default_branch(self, mock_manager):
        """Test loading a model from default (main) branch"""
        expected_data = b"model data"
        expected_metadata = {
            "model_id": "model-123",
            "model_type": "dqn",
            "version": "v1.0.0", 
            "branch": "main",
            "storage_path": "gcs://test-bucket/model.pkl"
        }
        
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_metadata = AsyncMock(return_value=expected_metadata)
        mock_manager.storage_handler.load = AsyncMock(return_value=expected_data)
        mock_manager.db_handler.record_event = AsyncMock()
        
        model_data, metadata = await mock_manager.load_model(
            model_type="dqn",
            version="v1.0.0",
            mode="analysis"
            # No branch specified, should default to main
        )
        
        assert model_data == expected_data
        assert metadata == expected_metadata
        
        # Verify it used the main branch
        mock_manager.db_handler.get_metadata.assert_called_once_with(
            model_type="dqn",
            version="v1.0.0",
            mode="analysis", 
            branch="main"
        )


# =============================================================================
# BRANCH ISOLATION TESTS
# =============================================================================

class TestBranchIsolation:
    """Test that branches properly isolate models from each other"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock preservation config"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=1.0,
            max_versions_per_model=5,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=False,
            emergency_backup=False
        )
        
    @pytest.fixture
    def mock_manager(self, mock_config):
        """Mock preservation manager with mocked handlers"""
        with patch('src.model_preservation.manager.GCSHandler') as mock_gcs, \
             patch('src.model_preservation.manager.DatabaseHandler') as mock_db:
            
            manager = PreservationManager(mock_config)
            manager.storage_handler = AsyncMock()
            manager.db_handler = AsyncMock()  
            manager.db_handler.initialize = AsyncMock()
            manager.storage_handler.initialize = AsyncMock()
            
            return manager
    
    @pytest.mark.asyncio
    async def test_branch_version_isolation(self, mock_manager):
        """Test that versions are independent per branch"""
        # Mock different version lists for different branches
        def mock_get_versions(model_type, branch=None):
            if branch == "main":
                return [{"version": "v1.0.0"}, {"version": "v1.1.0"}]
            elif branch == "experimental":
                return [{"version": "v1.0.0"}]  # Same version but different branch
            else:
                return []
        
        mock_manager.db_handler.get_versions = AsyncMock(side_effect=mock_get_versions)
        
        # Get latest version for main branch
        main_latest = await mock_manager.get_latest_version("dqn", branch="main")
        
        # Get latest version for experimental branch  
        exp_latest = await mock_manager.get_latest_version("dqn", branch="experimental")
        
        # Both should be able to retrieve their own versions independently
        mock_manager.db_handler.get_versions.assert_any_call(model_type="dqn", branch="main")
        mock_manager.db_handler.get_versions.assert_any_call(model_type="dqn", branch="experimental")
    
    @pytest.mark.asyncio
    async def test_branch_tag_isolation(self, mock_manager):
        """Test that tags are isolated per branch"""
        # Mock tag operations that include branch context
        mock_manager.db_handler.save_tag = AsyncMock(return_value="tag-123")
        mock_manager.db_handler.record_event = AsyncMock()
        
        # Tag same version in different branches
        await mock_manager.tag_model(
            model_type="dqn",
            version="v1.0.0",
            tag_name="stable",
            branch="main"
        )
        
        await mock_manager.tag_model(
            model_type="dqn", 
            version="v1.0.0",
            tag_name="stable",
            branch="experimental"
        )
        
        # Both operations should succeed (no conflict)
        assert mock_manager.db_handler.save_tag.call_count == 2
    
    @pytest.mark.asyncio
    async def test_branch_model_listing_isolation(self, mock_manager):
        """Test that model listings are isolated per branch"""
        # Mock different model lists for different branches
        main_models = [
            {"model_type": "dqn", "version": "v1.0.0", "branch": "main"},
            {"model_type": "dqn", "version": "v1.1.0", "branch": "main"}
        ]
        
        exp_models = [
            {"model_type": "dqn", "version": "v1.0.0", "branch": "experimental"}
        ]
        
        def mock_list_models(branch=None):
            if branch == "main":
                return main_models
            elif branch == "experimental":
                return exp_models
            else:
                return []
        
        mock_manager.db_handler.list_models = AsyncMock(side_effect=mock_list_models)
        
        # List models in each branch
        main_list = await mock_manager.list_models(branch="main")
        exp_list = await mock_manager.list_models(branch="experimental")
        
        assert len(main_list) == 2
        assert len(exp_list) == 1
        assert all(m["branch"] == "main" for m in main_list)
        assert all(m["branch"] == "experimental" for m in exp_list)


# =============================================================================
# BRANCH MERGING/PROMOTION TESTS
# =============================================================================

class TestBranchMerging:
    """Test branch merging and promotion operations"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock preservation config"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=1.0,
            max_versions_per_model=5,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=False,
            emergency_backup=False
        )
    
    @pytest.fixture  
    def mock_manager(self, mock_config):
        """Mock preservation manager with mocked handlers"""
        with patch('src.model_preservation.manager.GCSHandler') as mock_gcs, \
             patch('src.model_preservation.manager.DatabaseHandler') as mock_db:
            
            manager = PreservationManager(mock_config)
            manager.storage_handler = AsyncMock()
            manager.db_handler = AsyncMock()
            manager.db_handler.initialize = AsyncMock()
            manager.storage_handler.initialize = AsyncMock()
            
            return manager
    
    @pytest.mark.asyncio
    async def test_promote_model_to_main(self, mock_manager):
        """Test promoting a model from experimental branch to main"""
        # Mock source model metadata
        source_metadata = {
            "model_id": "exp-model-123",
            "model_type": "dqn",
            "version": "v1.2.0",
            "branch": "experimental",
            "storage_path": "gcs://test-bucket/exp-model.pkl",
            "mode": "analysis"
        }
        
        model_data = b"experimental model data"
        
        # Mock database operations
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_metadata = AsyncMock(return_value=source_metadata)
        mock_manager.storage_handler.load = AsyncMock(return_value=model_data)
        mock_manager.db_handler.get_versions = AsyncMock(return_value=[{"version": "v1.1.0"}])
        mock_manager.storage_handler.save = AsyncMock(return_value="gcs://test-bucket/main-model.pkl")
        mock_manager.db_handler.save_metadata = AsyncMock(return_value="main-model-123")
        mock_manager.db_handler.record_event = AsyncMock()
        mock_manager.db_handler.update_standard_tags = AsyncMock()
        
        new_model_id = await mock_manager.promote_model(
            model_type="dqn",
            version="v1.2.0",
            source_branch="experimental",
            target_branch="main",
            mode="analysis"
        )
        
        assert new_model_id == "main-model-123"
        
        # Verify the promotion workflow
        mock_manager.db_handler.get_metadata.assert_called_once_with(
            model_type="dqn",
            version="v1.2.0",
            branch="experimental",
            mode="analysis"
        )
        
        mock_manager.storage_handler.load.assert_called_once_with(source_metadata["storage_path"])
        
        # Verify new model was saved with target branch
        save_call_args = mock_manager.db_handler.save_metadata.call_args[0][0]
        assert save_call_args.branch == "main"
        assert save_call_args.model_type == "dqn"
        assert save_call_args.version == "v1.2.0"
    
    @pytest.mark.asyncio
    async def test_promote_nonexistent_model(self, mock_manager):
        """Test promoting a non-existent model"""
        mock_manager.db_handler.branch_exists = AsyncMock(return_value=True)
        mock_manager.db_handler.get_metadata = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="Model dqn v1.2.0 not found in branch experimental"):
            await mock_manager.promote_model(
                model_type="dqn",
                version="v1.2.0",
                source_branch="experimental",
                target_branch="main"
            )
    
    @pytest.mark.asyncio
    async def test_promote_to_nonexistent_branch(self, mock_manager):
        """Test promoting to a non-existent branch"""
        def mock_branch_exists(branch):
            return branch != "nonexistent"
        
        mock_manager.db_handler.branch_exists = AsyncMock(side_effect=mock_branch_exists)
        
        with pytest.raises(ValueError, match="Target branch 'nonexistent' does not exist"):
            await mock_manager.promote_model(
                model_type="dqn",
                version="v1.2.0", 
                source_branch="experimental",
                target_branch="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_promote_from_nonexistent_branch(self, mock_manager):
        """Test promoting from a non-existent branch"""
        def mock_branch_exists(branch):
            return branch != "nonexistent"
        
        mock_manager.db_handler.branch_exists = AsyncMock(side_effect=mock_branch_exists)
        
        with pytest.raises(ValueError, match="Source branch 'nonexistent' does not exist"):
            await mock_manager.promote_model(
                model_type="dqn",
                version="v1.2.0",
                source_branch="nonexistent", 
                target_branch="main"
            )


# =============================================================================
# BRANCH-AWARE VERSIONING TESTS
# =============================================================================

class TestBranchAwareVersioning:
    """Test that versioning works correctly with branches"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock preservation config"""
        return PreservationConfig(
            gcs_bucket="test-bucket",
            backup_interval_hours=1.0,
            max_versions_per_model=5,
            enable_compression=True,
            mode_isolation=True,
            auto_backup=False,
            emergency_backup=False
        )
    
    @pytest.fixture
    def mock_manager(self, mock_config):
        """Mock preservation manager with mocked handlers"""
        with patch('src.model_preservation.manager.GCSHandler') as mock_gcs, \
             patch('src.model_preservation.manager.DatabaseHandler') as mock_db:
            
            manager = PreservationManager(mock_config)
            manager.storage_handler = AsyncMock()
            manager.db_handler = AsyncMock()
            manager.db_handler.initialize = AsyncMock()
            manager.storage_handler.initialize = AsyncMock()
            
            return manager
    
    @pytest.mark.asyncio
    async def test_independent_versioning_per_branch(self, mock_manager):
        """Test that each branch maintains independent version sequences"""
        # Mock different version histories for different branches
        def mock_get_versions(model_type, branch=None):
            if branch == "main":
                return [{"version": "v1.0.0"}, {"version": "v1.1.0"}, {"version": "v1.2.0"}]
            elif branch == "experimental":
                return [{"version": "v1.0.0"}, {"version": "v1.1.0"}]
            elif branch == "hotfix":
                return [{"version": "v1.0.0"}]
            else:
                return []
        
        mock_manager.db_handler.get_versions = AsyncMock(side_effect=mock_get_versions)
        
        # Generate next version for each branch
        main_next = await mock_manager._generate_next_version("dqn", branch="main")
        exp_next = await mock_manager._generate_next_version("dqn", branch="experimental")  
        hotfix_next = await mock_manager._generate_next_version("dqn", branch="hotfix")
        
        # Each branch should have its own version sequence
        assert main_next == "v1.3.0"  # Next after v1.2.0
        assert exp_next == "v1.2.0"   # Next after v1.1.0
        assert hotfix_next == "v1.1.0"  # Next after v1.0.0
    
    @pytest.mark.asyncio
    async def test_branch_specific_tags(self, mock_manager):
        """Test that tags are specific to branches"""
        # Mock tag resolution that includes branch context
        def mock_resolve_tag(model_type, tag_name, branch=None):
            if branch == "main" and tag_name == "stable":
                return "v1.2.0"
            elif branch == "experimental" and tag_name == "stable":
                return "v1.1.0"
            elif branch == "main" and tag_name == "latest":
                return "v1.3.0"
            elif branch == "experimental" and tag_name == "latest":
                return "v1.2.0"
            else:
                return None
        
        mock_manager.db_handler.resolve_tag_to_version = AsyncMock(side_effect=mock_resolve_tag)
        
        # Resolve same tag in different branches
        main_stable = await mock_manager.resolve_tag("dqn", "stable", branch="main")
        exp_stable = await mock_manager.resolve_tag("dqn", "stable", branch="experimental")
        
        assert main_stable == "v1.2.0"
        assert exp_stable == "v1.1.0"
        
        # Verify calls included branch context
        mock_manager.db_handler.resolve_tag_to_version.assert_any_call("dqn", "stable", branch="main")
        mock_manager.db_handler.resolve_tag_to_version.assert_any_call("dqn", "stable", branch="experimental")
    
    @pytest.mark.asyncio
    async def test_get_latest_version_per_branch(self, mock_manager):
        """Test getting latest version for specific branch"""
        # Mock version data with branch information
        def mock_get_versions(model_type, branch=None):
            all_versions = [
                {"version": "v1.0.0", "branch": "main"},
                {"version": "v1.1.0", "branch": "main"},
                {"version": "v1.2.0", "branch": "main"},
                {"version": "v1.0.0", "branch": "experimental"},
                {"version": "v1.1.0", "branch": "experimental"},
                {"version": "v2.0.0-alpha", "branch": "experimental"}
            ]
            
            if branch:
                return [v for v in all_versions if v["branch"] == branch]
            return all_versions
        
        mock_manager.db_handler.get_versions = AsyncMock(side_effect=mock_get_versions)
        
        # Get latest for each branch
        main_latest = await mock_manager.get_latest_version("dqn", branch="main")
        exp_latest = await mock_manager.get_latest_version("dqn", branch="experimental", include_prerelease=True)
        
        # Should return latest for each branch
        mock_manager.db_handler.get_versions.assert_any_call(model_type="dqn", branch="main")
        mock_manager.db_handler.get_versions.assert_any_call(model_type="dqn", branch="experimental")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBranchIntegration:
    """Integration tests for branch functionality"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_branch_workflow(self):
        """Test complete branch workflow: create, save, load, promote"""
        # This test would require actual database connection
        # For now, it's marked as integration test and skipped in unit tests
        pytest.skip("Integration test - requires database connection")
    
    @pytest.mark.integration  
    @pytest.mark.asyncio
    async def test_branch_performance_with_many_models(self):
        """Test branch performance with large number of models"""
        # This test would benchmark branch operations with large datasets
        pytest.skip("Integration test - requires database connection")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_branch_concurrent_operations(self):
        """Test concurrent branch operations"""
        # This test would verify thread safety of branch operations
        pytest.skip("Integration test - requires database connection")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestBranchErrorHandling:
    """Test error handling for branch operations"""
    
    def test_branch_name_validation_edge_cases(self):
        """Test edge cases in branch name validation"""
        from src.model_preservation.base import validate_branch_name
        
        # Test minimum length
        assert validate_branch_name("a")
        
        # Test maximum length
        assert validate_branch_name("a" * 63)
        assert not validate_branch_name("a" * 64)
        
        # Test special characters
        assert not validate_branch_name("branch.name")  # dots not allowed
        assert validate_branch_name("branch-name")      # hyphens allowed
        assert validate_branch_name("branch_name")      # underscores allowed
        
        # Test Unicode characters
        assert not validate_branch_name("branch-ñame")
        assert not validate_branch_name("branch-名前")
    
    def test_reserved_branch_names(self):
        """Test that reserved branch names are handled correctly"""
        from src.model_preservation.base import is_reserved_branch_name
        
        # Standard reserved names
        assert is_reserved_branch_name("HEAD")
        assert is_reserved_branch_name("ORIG_HEAD") 
        assert is_reserved_branch_name("FETCH_HEAD")
        assert is_reserved_branch_name("MERGE_HEAD")
        
        # Case sensitivity
        assert is_reserved_branch_name("head")
        assert is_reserved_branch_name("Head")
        
        # Non-reserved names
        assert not is_reserved_branch_name("main")
        assert not is_reserved_branch_name("experimental")
        assert not is_reserved_branch_name("feature-123")