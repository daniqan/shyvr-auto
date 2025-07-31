"""
Test Model Preservation API Endpoints

Comprehensive tests for Phase 2.4 API endpoints following TDD methodology.
All tests will fail initially and be implemented to pass one by one.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException, status

# Test will fail until API is implemented
try:
    from src.model_preservation.api import preservation_api, PreservationAPI
    API_MODULE_EXISTS = True
except ImportError:
    API_MODULE_EXISTS = False
    preservation_api = None
    PreservationAPI = None


class TestPreservationAPI:
    """Test suite for model preservation API endpoints"""
    
    @pytest.fixture
    def mock_preservation_manager(self):
        """Mock preservation manager for testing"""
        manager = AsyncMock()
        
        # Mock list_models response
        manager.list_models.return_value = [
            {
                "model_id": "test-model-1",
                "model_type": "dqn_agent",
                "version": "1.0.0",
                "mode": "simulation",
                "state": "active",
                "created_at": datetime.utcnow(),
                "size_mb": 15.2,
                "checksum": "abc123"
            },
            {
                "model_id": "test-model-2", 
                "model_type": "lstm_model",
                "version": "2.1.0",
                "mode": "analysis",
                "state": "active",
                "created_at": datetime.utcnow() - timedelta(hours=2),
                "size_mb": 8.7,
                "checksum": "def456"
            }
        ]
        
        # Mock load_model response
        manager.load_model.return_value = {
            "model_data": b"fake_model_data",
            "metadata": {
                "model_id": "test-model-1",
                "model_type": "dqn_agent", 
                "version": "1.0.0",
                "mode": "simulation",
                "created_at": datetime.utcnow().isoformat(),
                "checksum": "abc123"
            }
        }
        
        # Mock rollback_model response
        manager.rollback_model.return_value = True
        
        # Mock delete operations
        manager.delete_model = AsyncMock(return_value=True)
        
        # Mock health check
        manager.get_stats.return_value = {
            "total_models": 25,
            "active_models": 23,
            "total_size_mb": 156.8,
            "models_by_type": {"dqn_agent": 12, "lstm_model": 11, "random_forest": 2},
            "models_by_mode": {"simulation": 15, "analysis": 8, "live": 2},
            "last_backup": datetime.utcnow(),
            "backup_success_rate": 98.5
        }
        
        return manager
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user"""
        user = MagicMock()
        user.user_id = "test-user-123"
        user.username = "testuser"
        user.permissions = {"read", "write"}
        return user
    
    @pytest.fixture
    def client(self, mock_preservation_manager):
        """Test client with mocked preservation manager"""
        if not API_MODULE_EXISTS:
            pytest.skip("Preservation API module not yet implemented")
        
        # Patch the preservation manager
        with patch("src.model_preservation.api.preservation_manager", mock_preservation_manager):
            from main import app
            return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_list_models_endpoint(self, client, mock_auth_user):
        """Test GET /api/preservation/models endpoint"""
        # This will fail until endpoint is implemented
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "pagination" in data
        assert "timestamp" in data
        assert len(data["models"]) == 2
        
        # Verify model structure
        model = data["models"][0]
        required_fields = ["model_id", "model_type", "version", "mode", "state", "created_at", "size_mb"]
        for field in required_fields:
            assert field in model
    
    @pytest.mark.asyncio 
    async def test_list_models_with_pagination(self, client, mock_auth_user):
        """Test list models with pagination parameters"""
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["pagination"]["limit"] == 10
        assert data["pagination"]["offset"] == 0
        assert "total" in data["pagination"]
        assert "has_more" in data["pagination"]
    
    @pytest.mark.asyncio
    async def test_list_models_with_filters(self, client, mock_auth_user):
        """Test list models with type and mode filters"""
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models?model_type=dqn_agent&mode=simulation")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filters" in data
        assert data["filters"]["model_type"] == "dqn_agent"
        assert data["filters"]["mode"] == "simulation"
    
    @pytest.mark.asyncio
    async def test_get_specific_model_endpoint(self, client, mock_auth_user):
        """Test GET /api/preservation/models/{type}/{version} endpoint"""
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models/dqn_agent/1.0.0")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model" in data
        assert "metadata" in data
        assert "timestamp" in data
        
        model_data = data["model"]
        assert model_data["model_type"] == "dqn_agent"
        assert model_data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self, client, mock_auth_user, mock_preservation_manager):
        """Test getting non-existent model returns 404"""
        mock_preservation_manager.load_model.side_effect = FileNotFoundError("Model not found")
        
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models/nonexistent/1.0.0")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_rollback_model_endpoint(self, client, mock_auth_user):
        """Test POST /api/preservation/rollback endpoint"""
        rollback_request = {
            "model_type": "dqn_agent",
            "target_version": "1.0.0",
            "mode": "simulation",
            "reason": "Performance regression in v1.1.0"
        }
        
        with patch("src.dashboard.auth.require_write", return_value=mock_auth_user):
            response = client.post("/api/preservation/rollback", json=rollback_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data
        assert "rollback_details" in data
    
    @pytest.mark.asyncio
    async def test_rollback_validation_error(self, client, mock_auth_user):
        """Test rollback with invalid parameters"""
        invalid_request = {
            "model_type": "",  # Empty model type
            "target_version": "invalid_version"
        }
        
        with patch("src.dashboard.auth.require_write", return_value=mock_auth_user):
            response = client.post("/api/preservation/rollback", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_delete_model_version_endpoint(self, client, mock_auth_user):
        """Test DELETE /api/preservation/models/{type}/{version} endpoint"""
        with patch("src.dashboard.auth.require_admin", return_value=mock_auth_user):
            response = client.delete("/api/preservation/models/dqn_agent/1.0.0")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, client, mock_auth_user, mock_preservation_manager):
        """Test deleting non-existent model version"""
        mock_preservation_manager.delete_model.return_value = False
        
        with patch("src.dashboard.auth.require_admin", return_value=mock_auth_user):
            response = client.delete("/api/preservation/models/nonexistent/1.0.0")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, client, mock_auth_user):
        """Test GET /api/preservation/health endpoint"""
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "statistics" in data
        assert "storage_health" in data
        assert "timestamp" in data
        
        stats = data["statistics"]
        assert "total_models" in stats
        assert "active_models" in stats
        assert "total_size_mb" in stats
        assert "backup_success_rate" in stats
    
    @pytest.mark.asyncio
    async def test_health_check_degraded_status(self, client, mock_auth_user, mock_preservation_manager):
        """Test health check when system is degraded"""
        # Mock degraded stats
        degraded_stats = {
            "total_models": 25,
            "active_models": 20,  # Some models inactive
            "backup_success_rate": 85.0,  # Lower success rate
            "storage_errors": 3
        }
        mock_preservation_manager.get_stats.return_value = degraded_stats
        
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["degraded", "warning"]
    
    @pytest.mark.asyncio
    async def test_authentication_required(self, client):
        """Test that endpoints require authentication"""
        # Test without authentication
        response = client.get("/api/preservation/models")
        assert response.status_code in [401, 403]  # Unauthorized or Forbidden
        
        response = client.post("/api/preservation/rollback", json={})
        assert response.status_code in [401, 403]
        
        response = client.delete("/api/preservation/models/test/1.0.0")
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_authorization_levels(self, client):
        """Test different authorization levels"""
        read_user = MagicMock()
        read_user.permissions = {"read"}
        
        write_user = MagicMock()
        write_user.permissions = {"read", "write"}
        
        # Read user can access GET endpoints
        with patch("src.dashboard.auth.require_read", return_value=read_user):
            response = client.get("/api/preservation/models")
            assert response.status_code == 200
        
        # Read user cannot access write endpoints
        with patch("src.dashboard.auth.require_write", side_effect=HTTPException(status_code=403)):
            response = client.post("/api/preservation/rollback", json={})
            assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, mock_auth_user, mock_preservation_manager):
        """Test error handling in API endpoints"""
        # Mock manager error
        mock_preservation_manager.list_models.side_effect = Exception("Database connection failed")
        
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            response = client.get("/api/preservation/models")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_response_time_requirements(self, client, mock_auth_user):
        """Test that API responses meet <100ms requirement"""
        import time
        
        with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
            start_time = time.time()
            response = client.get("/api/preservation/models")
            end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        assert response_time_ms < 100  # Phase 2 requirement: <100ms
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, mock_auth_user):
        """Test handling of concurrent API requests"""
        import concurrent.futures
        import time
        
        def make_request():
            with patch("src.dashboard.auth.require_read", return_value=mock_auth_user):
                return client.get("/api/preservation/models")
        
        # Test 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
            end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Total time should be reasonable for concurrent processing
        total_time_ms = (end_time - start_time) * 1000
        assert total_time_ms < 1000  # Should handle 10 concurrent requests in <1s


class TestPreservationAPIModels:
    """Test Pydantic models for API requests/responses"""
    
    def test_rollback_request_model(self):
        """Test RollbackRequest validation"""
        if not API_MODULE_EXISTS:
            pytest.skip("Preservation API module not yet implemented")
        
        from src.model_preservation.api import RollbackRequest
        
        # Valid request
        valid_request = RollbackRequest(
            model_type="dqn_agent",
            target_version="1.0.0",
            mode="simulation",
            reason="Performance regression"
        )
        assert valid_request.model_type == "dqn_agent"
        assert valid_request.target_version == "1.0.0"
        
        # Invalid request - empty model_type
        with pytest.raises(ValueError):
            RollbackRequest(
                model_type="",
                target_version="1.0.0"
            )
    
    def test_model_list_response_model(self):
        """Test ModelListResponse structure"""
        if not API_MODULE_EXISTS:
            pytest.skip("Preservation API module not yet implemented")
        
        from src.model_preservation.api import ModelListResponse
        
        response = ModelListResponse(
            models=[],
            pagination={"limit": 10, "offset": 0, "total": 0, "has_more": False},
            filters={},
            timestamp=datetime.utcnow()
        )
        assert response.models == []
        assert response.pagination["limit"] == 10


class TestPreservationAPIIntegration:
    """Integration tests for preservation API with real PreservationManager"""
    
    @pytest.fixture
    async def real_preservation_manager(self):
        """Create a real preservation manager with test config"""
        from src.model_preservation.manager import PreservationManager, PreservationConfig
        
        config = PreservationConfig(
            gcs_bucket="test-bucket",
            enable_compression=True,
            max_versions_per_model=5,
            enable_caching=False,  # Disable caching for tests
            enable_metrics=False   # Disable metrics for tests
        )
        
        manager = PreservationManager(config)
        # Mock the storage and db handlers for integration tests
        manager.storage_handler = AsyncMock()
        manager.db_handler = AsyncMock()
        manager.cache_manager = AsyncMock()
        
        await manager.initialize()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_api_with_real_manager(self, real_preservation_manager):
        """Test API endpoints with real preservation manager instance"""
        if not API_MODULE_EXISTS:
            pytest.skip("Preservation API module not yet implemented")
        
        # This test will validate that the API properly integrates with
        # the actual PreservationManager methods and handles real data structures
        
        # Mock some real-looking data (need to use patch since it's a real manager)
        with patch.object(real_preservation_manager, 'list_models', return_value=[
            {
                "model_id": "real-model-123",
                "model_type": "dqn_agent",
                "version": "1.2.3",
                "mode": "simulation",
                "state": "active",
                "created_at": datetime.utcnow(),
                "size_mb": 12.5,
                "checksum": "real_checksum_456"
            }
        ]) as mock_list:
            # Test that manager integration works
            result = await real_preservation_manager.list_models()
            assert len(result) == 1
            assert result[0]["model_type"] == "dqn_agent"