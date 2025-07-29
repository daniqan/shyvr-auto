"""
Tests for main.py application functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app, get_ml_model_health, get_rl_agent_health


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_api_root_endpoint(self, client):
        """Test /api endpoint"""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Shyvr RLTE API"
        assert data["status"] == "running"
    
    @patch('main.get_config')
    @patch('src.utils.database.check_database_health')
    @patch('main.get_ml_model_health')
    @patch('main.get_rl_agent_health')
    def test_health_check_success(self, mock_rl_health, mock_ml_health, 
                                 mock_db_health, mock_config, client):
        """Test /health endpoint with all systems healthy"""
        # Setup mocks
        mock_config.return_value = MagicMock(
            app=MagicMock(version="0.1.0", environment="test")
        )
        mock_db_health.return_value = {
            "status": "healthy",
            "connectivity": True,
            "database_size": "10MB",
            "active_connections": 2,
            "tables_exist": True,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        mock_ml_health.return_value = {
            "status": "healthy",
            "models_loaded": 3,
            "ensemble_available": True
        }
        mock_rl_health.return_value = {
            "status": "healthy",
            "is_trained": True,
            "meets_targets": True
        }
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["environment"] == "test"
        assert "components" in data
        assert data["components"]["database"]["status"] == "healthy"

    @patch('main.get_config')
    @patch('src.utils.database.check_database_health')
    def test_health_check_database_unhealthy(self, mock_db_health, mock_config, client):
        """Test /health endpoint with database unhealthy"""
        mock_config.return_value = MagicMock(
            app=MagicMock(version="0.1.0", environment="test")
        )
        mock_db_health.return_value = {
            "status": "unhealthy",
            "connectivity": False,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["database"]["status"] == "unhealthy"

    @patch('src.utils.database.check_database_health')
    def test_health_check_error_handling(self, mock_db_health, client):
        """Test /health endpoint error handling"""
        # Mock database health to avoid side effects
        mock_db_health.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        with patch('main.get_config', side_effect=Exception("Config error")):
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data


class TestMLHealthFunction:
    """Test ML model health check function"""
    
    @patch('src.ml_analysis.model_manager.ModelManager')
    @patch('main.get_config')
    @pytest.mark.asyncio
    async def test_ml_health_success(self, mock_config, mock_model_manager):
        """Test successful ML health check"""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_manager_instance = AsyncMock()
        mock_manager_instance.health_check.return_value = {
            "overall_healthy": True,
            "models": {"model1": "healthy", "model2": "healthy"},
            "ensemble_available": True,
            "cache_size": 5,
            "performance": {"accuracy": 0.95}
        }
        mock_model_manager.return_value = mock_manager_instance
        
        result = await get_ml_model_health()
        
        assert result["status"] == "healthy"
        assert result["models_loaded"] == 2
        assert result["ensemble_available"] is True
        assert result["cache_size"] == 5
        assert "details" in result

    @patch('src.ml_analysis.model_manager.ModelManager')
    @patch('main.get_config')
    @pytest.mark.asyncio
    async def test_ml_health_unhealthy(self, mock_config, mock_model_manager):
        """Test ML health check with unhealthy models"""
        mock_config.return_value = MagicMock()
        mock_manager_instance = AsyncMock()
        mock_manager_instance.health_check.return_value = {
            "overall_healthy": False,
            "models": {"model1": "unhealthy"},
            "ensemble_available": False
        }
        mock_model_manager.return_value = mock_manager_instance
        
        result = await get_ml_model_health()
        
        assert result["status"] == "unhealthy"
        assert result["ensemble_available"] is False

    @patch('src.ml_analysis.model_manager.ModelManager', side_effect=Exception("Model error"))
    @pytest.mark.asyncio
    async def test_ml_health_error(self, mock_model_manager):
        """Test ML health check error handling"""
        result = await get_ml_model_health()
        
        assert result["status"] == "error"
        assert "error" in result
        assert result["error"] == "Model error"


class TestRLHealthFunction:
    """Test RL agent health check function"""
    
    @patch('src.rl_agent.dqn_agent.DQNTradingAgent')
    @patch('main.get_config')
    @pytest.mark.asyncio
    async def test_rl_health_success(self, mock_config, mock_agent):
        """Test successful RL health check"""
        mock_config.return_value = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_instance.health_check.return_value = {
            "is_trained": True,
            "training_episodes": 1000,
            "meets_targets": True,
            "performance_metrics": {
                "sharpe_ratio": 1.5,
                "win_rate": 0.65,
                "max_drawdown": 0.1
            }
        }
        mock_agent.return_value = mock_agent_instance
        
        result = await get_rl_agent_health()
        
        assert result["status"] == "healthy"
        assert result["is_trained"] is True
        assert result["meets_targets"] is True
        assert result["training_episodes"] == 1000
        assert "performance" in result

    @patch('src.rl_agent.dqn_agent.DQNTradingAgent')
    @patch('main.get_config')
    @pytest.mark.asyncio
    async def test_rl_health_training(self, mock_config, mock_agent):
        """Test RL health check with agent still training"""
        mock_config.return_value = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_instance.health_check.return_value = {
            "is_trained": False,
            "training_episodes": 500,
            "meets_targets": False,
            "performance_metrics": {}
        }
        mock_agent.return_value = mock_agent_instance
        
        result = await get_rl_agent_health()
        
        assert result["status"] == "training"
        assert result["is_trained"] is False
        assert result["meets_targets"] is False

    @patch('src.rl_agent.dqn_agent.DQNTradingAgent', side_effect=Exception("Agent error"))
    @pytest.mark.asyncio
    async def test_rl_health_error(self, mock_agent):
        """Test RL health check error handling"""
        result = await get_rl_agent_health()
        
        assert result["status"] == "error"
        assert "error" in result
        assert result["error"] == "Agent error"


class TestConfigEndpoint:
    """Test configuration endpoint"""
    
    @patch('main.get_config')
    def test_config_endpoint_success(self, mock_config, client):
        """Test /config endpoint success"""
        # Create mock with explicit property setting
        mock_obj = MagicMock()
        mock_obj.app.name = "test-rlte"
        mock_obj.app.version = "0.1.0"
        mock_obj.app.environment = "test"
        mock_obj.app.debug = True
        mock_obj.trading.modes = ["analysis", "simulation"]
        mock_obj.trading.risk_management.max_position_size_pct = 10.0
        mock_obj.trading.risk_management.max_daily_loss_pct = 5.0
        mock_obj.trading.risk_management.max_drawdown_pct = 15.0
        mock_obj.agent.model_type = "DQN"
        mock_obj.agent.max_active_rules = 50
        mock_config.return_value = mock_obj
        
        response = client.get("/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["app"]["name"] == "test-rlte"
        assert data["app"]["version"] == "0.1.0"
        assert data["trading"]["modes"] == ["analysis", "simulation"]
        assert "agent" in data

    def test_config_endpoint_error(self, client):
        """Test /config endpoint error handling"""
        with patch('main.get_config', side_effect=Exception("Config error")):
            response = client.get("/config")
            assert response.status_code == 200
            
            data = response.json()
            assert "error" in data
            assert data["error"] == "Config error"


class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    @patch('main.metrics_registry')
    @patch('main.trading_metrics')
    @patch('main.safety_metrics')
    @patch('main.analysis_metrics')
    def test_metrics_endpoint_success(self, mock_analysis, mock_safety, 
                                    mock_trading, mock_registry, client):
        """Test /metrics endpoint success"""
        mock_registry.generate_output.return_value = "# Prometheus metrics output"
        
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "Prometheus metrics output" in response.text
        
        # Verify metrics collection was called
        mock_trading.collect_metrics.assert_called_once()
        mock_safety.collect_metrics.assert_called_once()
        mock_analysis.collect_metrics.assert_called_once()

    @patch('main.metrics_registry')
    def test_metrics_endpoint_error(self, mock_registry, client):
        """Test /metrics endpoint error handling"""
        mock_registry.generate_output.side_effect = Exception("Metrics error")
        
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "Error generating metrics" in response.text


class TestAPIKeyEndpoint:
    """Test API key endpoint"""
    
    @patch('src.dashboard.auth.dashboard_auth')
    def test_api_key_success(self, mock_auth, client):
        """Test /api/auth/key endpoint success"""
        # Setup mock auth data
        mock_user = MagicMock()
        mock_user.username = "admin"
        mock_user.permissions = ["admin"]
        mock_user.user_id = "admin-123"
        
        mock_auth._users = {"admin-123": mock_user}
        mock_auth._api_keys = {"test-api-key": "admin-123"}
        
        response = client.get("/api/auth/key")
        assert response.status_code == 200
        
        data = response.json()
        assert data["api_key"] == "test-api-key"

    @patch('src.dashboard.auth.dashboard_auth')
    def test_api_key_no_admin(self, mock_auth, client):
        """Test /api/auth/key endpoint with no admin user"""
        mock_auth._users = {}
        mock_auth._api_keys = {}
        
        response = client.get("/api/auth/key")
        assert response.status_code == 200
        
        data = response.json()
        assert "error" in data
        assert "Admin user not found" in data["error"]

    def test_api_key_error(self, client):
        """Test /api/auth/key endpoint error handling"""
        # Mock the dashboard_auth._users.items() method to raise exception when called
        mock_dashboard_auth = MagicMock()
        mock_dashboard_auth._users.items.side_effect = Exception("Auth error")
        
        with patch('src.dashboard.auth.dashboard_auth', mock_dashboard_auth):
            response = client.get("/api/auth/key")
            assert response.status_code == 200
            
            data = response.json()
            assert "error" in data
            assert data["error"] == "Auth error"