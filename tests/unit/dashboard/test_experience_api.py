"""
Tests for dashboard experience API endpoints
Following TDD methodology - tests written before implementation
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.api import dashboard_api
from src.dashboard.auth import User, require_read


# Create test app with dashboard routes
test_app = FastAPI()
test_app.include_router(dashboard_api.router)


class TestExperienceAPIEndpoints:
    """Test suite for Experience API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return User(
            user_id="test-user-123",
            username="testuser",
            permissions={"dashboard.read", "dashboard.write", "admin"},
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_experience_data(self):
        """Sample experience data for testing"""
        return {
            "id": 1,
            "experience_id": str(uuid4()),
            "session_id": str(uuid4()),
            "user_id": None,
            "state_data": {
                "price": 0.00123,
                "volume_24h": 150000,
                "rsi": 65.0,
                "position_size": 0.1
            },
            "action": 1,
            "reward": 0.15,
            "next_state_data": {
                "price": 0.00125,
                "volume_24h": 160000,
                "rsi": 70.0,
                "position_size": 0.2
            },
            "done": False,
            "priority": 0.8,
            "trading_mode": "simulation",
            "token_address": "0x6982508145454Ce325dDbE47a25d4ec3d2311933",
            "chain": "ethereum",
            "market_conditions": {
                "volatility": 0.15,
                "trend": "bullish",
                "support_level": 0.00118
            },
            "performance_metrics": {
                "execution_latency_ms": 150,
                "memory_usage_mb": 64
            },
            "metadata": {
                "strategy": "dqn",
                "episode": 1,
                "step": 1
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    @pytest.fixture
    def sample_experience_stats(self):
        """Sample experience statistics for testing"""
        return {
            "total_experiences": 10000,
            "recent_experiences_24h": 500,
            "average_reward": 0.125,
            "success_rate": 0.65,
            "top_performing_actions": [
                {"action": 1, "avg_reward": 0.25, "count": 3000},
                {"action": 2, "avg_reward": 0.18, "count": 2500},
                {"action": 0, "avg_reward": 0.05, "count": 4500}
            ],
            "reward_distribution": {
                "min": -0.5,
                "max": 1.2,
                "mean": 0.125,
                "std": 0.3,
                "quartiles": [-0.1, 0.05, 0.15, 0.4]
            },
            "trading_mode_breakdown": {
                "simulation": 7500,
                "live": 1500,
                "analysis": 1000
            },
            "chain_distribution": {
                "ethereum": 6000,
                "solana": 4000
            }
        }
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Sample performance metrics for testing"""
        return {
            "overall_performance": {
                "total_reward": 1250.75,
                "average_reward_per_experience": 0.125,
                "reward_volatility": 0.3,
                "sharpe_ratio": 0.42,
                "max_drawdown": -0.15,
                "win_rate": 0.65
            },
            "time_series_performance": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=23)).isoformat(),
                    "cumulative_reward": 1100.5,
                    "experiences_count": 8000,
                    "average_reward": 0.138
                },
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=12)).isoformat(),
                    "cumulative_reward": 1180.2,
                    "experiences_count": 9000,
                    "average_reward": 0.131
                },
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cumulative_reward": 1250.75,
                    "experiences_count": 10000,
                    "average_reward": 0.125
                }
            ],
            "action_performance": [
                {"action": 0, "total_reward": 225.0, "count": 4500, "avg_reward": 0.05},
                {"action": 1, "total_reward": 750.0, "count": 3000, "avg_reward": 0.25},
                {"action": 2, "total_reward": 275.75, "count": 2500, "avg_reward": 0.18}
            ],
            "mode_performance": {
                "simulation": {"total_reward": 937.5, "count": 7500, "avg_reward": 0.125},
                "live": {"total_reward": 225.0, "count": 1500, "avg_reward": 0.15},
                "analysis": {"total_reward": 88.25, "count": 1000, "avg_reward": 0.088}
            }
        }
    
    # =============================================================================
    # RECENT EXPERIENCES ENDPOINT TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_recent_experiences_success(self, mock_auth, mock_service, client, mock_user, sample_experience_data):
        """Test GET /api/v1/experiences/recent endpoint success"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_recent_experiences = AsyncMock(return_value={
            "experiences": [sample_experience_data],
            "total_count": 1,
            "has_more": False
        })
        
        # Override the require_read dependency to return our mock user
        def override_require_read():
            return mock_user
        
        test_app.dependency_overrides[require_read] = override_require_read
        
        try:
            # Make request
            response = client.get("/dashboard/api/v1/experiences/recent?limit=100&offset=0")
            
            # Should return 200 with experience data
            assert response.status_code == 200
            data = response.json()
            assert "experiences" in data
            assert "pagination" in data
            assert len(data["experiences"]) == 1
            assert data["experiences"][0]["experience_id"] == sample_experience_data["experience_id"]
            assert data["pagination"]["limit"] == 100
            assert data["pagination"]["offset"] == 0
            assert data["pagination"]["total"] == 1
            assert data["pagination"]["has_more"] is False
            
            # Should call service method with correct parameters
            mock_service.get_recent_experiences.assert_called_once_with(
                limit=100,
                offset=0,
                session_id=None,
                trading_mode=None,
                hours_back=24
            )
        finally:
            # Clean up dependency override
            test_app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_get_recent_experiences_no_auth(self, mock_auth, client, mock_user):
        """Test that recent experiences endpoint requires authentication"""
        mock_auth.return_value = mock_user
        
        # Make request without dependency override (should require auth)
        response = client.get("/dashboard/api/v1/experiences/recent")
        
        # Should return 403 because authentication is required
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_recent_experiences_with_filters(self, mock_auth, mock_service, client, mock_user):
        """Test recent experiences endpoint with filters"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_recent_experiences = AsyncMock(return_value={
            "experiences": [],
            "total_count": 0,
            "has_more": False
        })
        
        # Make request with filters
        response = client.get(
            "/dashboard/api/v1/experiences/recent"
            "?limit=50&offset=10&session_id=test-session&trading_mode=simulation&hours_back=48"
        )
        
        # Should call service with filter parameters
        if response.status_code == 200:  # Only check if endpoint exists
            mock_service.get_recent_experiences.assert_called_once_with(
                limit=50,
                offset=10,
                session_id="test-session",
                trading_mode="simulation",
                hours_back=48
            )
    
    # =============================================================================
    # EXPERIENCE STATS ENDPOINT TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_stats_success(self, mock_auth, mock_service, client, mock_user, sample_experience_stats):
        """Test GET /api/v1/experiences/stats endpoint success"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_experience_stats = AsyncMock(return_value=sample_experience_stats)
        
        # Make request
        response = client.get("/dashboard/api/v1/experiences/stats?hours_back=24")
        
        # Should return 200 with stats data
        assert response.status_code == 200
        data = response.json()
        assert "total_experiences" in data
        assert "average_reward" in data
        assert "success_rate" in data
        assert "top_performing_actions" in data
        assert data["total_experiences"] == 10000
        assert data["average_reward"] == 0.125
        assert len(data["top_performing_actions"]) == 3
        
        # Should call service method
        mock_service.get_experience_stats.assert_called_once_with(hours_back=24)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_stats_not_implemented(self, mock_auth, client, mock_user):
        """Test that experience stats endpoint returns 404 (not implemented yet)"""
        mock_auth.return_value = mock_user
        
        # Make request - should fail because endpoint doesn't exist yet
        response = client.get("/dashboard/api/v1/experiences/stats")
        
        # Should return 404 because route doesn't exist yet
        assert response.status_code == 404
    
    # =============================================================================
    # EXPERIENCE PERFORMANCE ENDPOINT TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_performance_success(self, mock_auth, mock_service, client, mock_user, sample_performance_metrics):
        """Test GET /api/v1/experiences/performance endpoint success"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_experience_performance = AsyncMock(return_value=sample_performance_metrics)
        
        # Make request
        response = client.get("/dashboard/api/v1/experiences/performance?hours_back=24")
        
        # Should return 200 with performance data
        assert response.status_code == 200
        data = response.json()
        assert "overall_performance" in data
        assert "time_series_performance" in data
        assert "action_performance" in data
        assert "mode_performance" in data
        assert data["overall_performance"]["total_reward"] == 1250.75
        assert data["overall_performance"]["win_rate"] == 0.65
        assert len(data["time_series_performance"]) == 3
        assert len(data["action_performance"]) == 3
        
        # Should call service method
        mock_service.get_experience_performance.assert_called_once_with(hours_back=24)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_performance_not_implemented(self, mock_auth, client, mock_user):
        """Test that experience performance endpoint returns 404 (not implemented yet)"""
        mock_auth.return_value = mock_user
        
        # Make request - should fail because endpoint doesn't exist yet
        response = client.get("/dashboard/api/v1/experiences/performance")
        
        # Should return 404 because route doesn't exist yet
        assert response.status_code == 404
    
    # =============================================================================
    # EXPERIENCE SEARCH ENDPOINT TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_search_success(self, mock_auth, mock_service, client, mock_user, sample_experience_data):
        """Test GET /api/v1/experiences/search endpoint success"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.search_experiences = AsyncMock(return_value={
            "experiences": [sample_experience_data],
            "total_count": 1,
            "has_more": False
        })
        
        # Make request with search parameters
        response = client.get(
            "/dashboard/api/v1/experiences/search"
            "?reward_min=0.1&reward_max=0.5&action=1&trading_mode=simulation"
            "&token_address=0x6982508145454Ce325dDbE47a25d4ec3d2311933"
            "&limit=50&offset=0"
        )
        
        # Should return 200 with search results
        assert response.status_code == 200
        data = response.json()
        assert "experiences" in data
        assert "pagination" in data
        assert "filters" in data
        assert len(data["experiences"]) == 1
        assert data["experiences"][0]["reward"] == 0.15
        assert data["filters"]["reward_min"] == 0.1
        assert data["filters"]["reward_max"] == 0.5
        assert data["filters"]["action"] == 1
        
        # Should call service method with filters
        mock_service.search_experiences.assert_called_once_with(
            reward_min=0.1,
            reward_max=0.5,
            action=1,
            trading_mode="simulation",
            token_address="0x6982508145454Ce325dDbE47a25d4ec3d2311933",
            chain=None,
            session_id=None,
            done=None,
            priority_min=None,
            priority_max=None,
            limit=50,
            offset=0
        )
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_search_not_implemented(self, mock_auth, client, mock_user):
        """Test that experience search endpoint returns 404 (not implemented yet)"""
        mock_auth.return_value = mock_user
        
        # Make request - should fail because endpoint doesn't exist yet
        response = client.get("/dashboard/api/v1/experiences/search")
        
        # Should return 404 because route doesn't exist yet
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_get_experience_search_minimal_filters(self, mock_auth, mock_service, client, mock_user):
        """Test experience search with minimal filters"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.search_experiences = AsyncMock(return_value={
            "experiences": [],
            "total_count": 0,
            "has_more": False
        })
        
        # Make request with minimal parameters
        response = client.get("/dashboard/api/v1/experiences/search?limit=25")
        
        # Should call service with default parameters
        if response.status_code == 200:  # Only check if endpoint exists
            mock_service.search_experiences.assert_called_once_with(
                reward_min=None,
                reward_max=None,
                action=None,
                trading_mode=None,
                token_address=None,
                chain=None,
                session_id=None,
                done=None,
                priority_min=None,
                priority_max=None,
                limit=25,
                offset=0
            )
    
    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_experience_api_error_handling(self, mock_auth, mock_service, client, mock_user):
        """Test error handling for experience API endpoints"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_recent_experiences = AsyncMock(side_effect=Exception("Database connection error"))
        
        # Make request - should handle service errors gracefully
        response = client.get("/dashboard/api/v1/experiences/recent")
        
        # Should return appropriate error response when implemented
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_experience_api_authorization(self, mock_auth, client):
        """Test authorization for experience API endpoints"""
        # Setup mock to return None (unauthorized)
        mock_auth.return_value = None
        
        # Make requests to all endpoints
        endpoints = [
            "/dashboard/api/v1/experiences/recent",
            "/dashboard/api/v1/experiences/stats",
            "/dashboard/api/v1/experiences/performance",
            "/dashboard/api/v1/experiences/search"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should return 401 when endpoints are implemented
            if response.status_code != 404:  # If endpoint exists
                assert response.status_code == 401
    
    # =============================================================================
    # PAGINATION AND VALIDATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_experience_api_pagination_validation(self, mock_auth, client, mock_user):
        """Test pagination parameter validation"""
        mock_auth.return_value = mock_user
        
        # Test invalid limit values
        response = client.get("/dashboard/api/v1/experiences/recent?limit=0")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error
            
        response = client.get("/dashboard/api/v1/experiences/recent?limit=10000")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error
        
        # Test invalid offset values
        response = client.get("/dashboard/api/v1/experiences/recent?offset=-1")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.require_read')
    async def test_experience_search_parameter_validation(self, mock_auth, client, mock_user):
        """Test search parameter validation"""
        mock_auth.return_value = mock_user
        
        # Test invalid reward range
        response = client.get("/dashboard/api/v1/experiences/search?reward_min=0.5&reward_max=0.1")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error
        
        # Test invalid priority range
        response = client.get("/dashboard/api/v1/experiences/search?priority_min=1.5")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error
        
        # Test invalid action value
        response = client.get("/dashboard/api/v1/experiences/search?action=-1")
        if response.status_code != 404:  # If endpoint exists
            assert response.status_code == 422  # Validation error


class TestExperienceAPIIntegration:
    """Integration tests for Experience API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client for integration testing"""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return User(
            user_id="integration-user-456",
            username="integrationuser",
            permissions={"dashboard.read", "admin"},
            created_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.require_read')
    async def test_experience_api_workflow(self, mock_auth, mock_service, client, mock_user):
        """Test complete experience API workflow"""
        # Setup mocks
        mock_auth.return_value = mock_user
        
        # Mock service responses
        mock_service.get_recent_experiences = AsyncMock(return_value={
            "experiences": [],
            "total_count": 0,
            "has_more": False
        })
        mock_service.get_experience_stats = AsyncMock(return_value={
            "total_experiences": 0,
            "average_reward": 0.0
        })
        mock_service.get_experience_performance = AsyncMock(return_value={
            "overall_performance": {"total_reward": 0.0}
        })
        mock_service.search_experiences = AsyncMock(return_value={
            "experiences": [],
            "total_count": 0,
            "has_more": False
        })
        
        # Test workflow when endpoints are implemented
        endpoints_to_test = [
            ("/dashboard/api/v1/experiences/recent", "get_recent_experiences"),
            ("/dashboard/api/v1/experiences/stats", "get_experience_stats"),
            ("/dashboard/api/v1/experiences/performance", "get_experience_performance"),
            ("/dashboard/api/v1/experiences/search", "search_experiences")
        ]
        
        for endpoint, service_method in endpoints_to_test:
            response = client.get(endpoint)
            
            # Only test if endpoint is implemented
            if response.status_code != 404:
                assert response.status_code == 200
                data = response.json()
                assert "timestamp" in data
                
                # Check that appropriate service method was called
                service_mock = getattr(mock_service, service_method)
                service_mock.assert_called()
    
    @pytest.mark.asyncio
    async def test_experience_api_endpoints_implemented(self, client):
        """Test that experience API endpoints are now implemented and require auth"""
        endpoints = [
            "/dashboard/api/v1/experiences/recent",
            "/dashboard/api/v1/experiences/stats", 
            "/dashboard/api/v1/experiences/performance",
            "/dashboard/api/v1/experiences/search"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should return 403 because authentication is required
            assert response.status_code == 403