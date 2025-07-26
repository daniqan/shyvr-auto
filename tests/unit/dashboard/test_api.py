"""
Tests for dashboard API routes
"""

import pytest
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from src.dashboard.api import dashboard_api
from src.dashboard.base import DashboardData, SystemStatus, TradingMode
from src.dashboard.auth import User, Session


# Create test app with dashboard routes
test_app = FastAPI()
test_app.include_router(dashboard_api.router)


class TestDashboardAPI:
    """Test suite for Dashboard API routes"""
    
    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_dashboard_service(self):
        """Mock dashboard service"""
        return MagicMock()
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service"""
        auth_service = MagicMock()
        auth_service.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions={"dashboard.read", "dashboard.write", "trading.control", "admin"},
            created_at=datetime.utcnow()
        ))
        auth_service.verify_jwt_token = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser", 
            permissions=["dashboard.read", "dashboard.write", "trading.control"],
            created_at=datetime.utcnow()
        ))
        return auth_service
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_dashboard_data_endpoint(self, mock_service, client):
        """Test GET /dashboard/data endpoint"""
        # Setup mock data
        dashboard_data = DashboardData.create_default()
        dashboard_data.active_alerts = 5
        mock_service.get_dashboard_data = AsyncMock(return_value=dashboard_data)
        mock_service.record_request = MagicMock()
        
        # Make request
        response = client.get("/dashboard/data")
        
        # Should return 200 with dashboard data
        assert response.status_code == 200
        data = response.json()
        assert "system_metrics" in data
        assert "portfolio_status" in data
        assert "trading_status" in data
        assert "ml_rl_status" in data
        assert data["active_alerts"] == 5
        
        # Should record request
        mock_service.record_request.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_dashboard_data_error(self, mock_service, client):
        """Test dashboard data endpoint error handling"""
        # Setup mock to raise error
        mock_service.get_dashboard_data = AsyncMock(side_effect=Exception("Service error"))
        mock_service.record_request = MagicMock()
        
        # Make request
        response = client.get("/dashboard/data")
        
        # Should return 500
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Service error" in data["error"]
        
        # Should record error
        mock_service.record_request.assert_called_once()
        call_args = mock_service.record_request.call_args
        assert call_args[1]["error"] is True
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_switch_trading_mode_endpoint(self, mock_auth, mock_service, client):
        """Test POST /dashboard/trading/mode endpoint"""
        # Setup mocks
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["trading.control"],
            created_at=datetime.utcnow()
        ))
        mock_service.switch_trading_mode = AsyncMock(return_value=True)
        
        # Make request
        response = client.post(
            "/dashboard/trading/mode",
            json={"mode": "simulation"},
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "simulation" in data["message"]
        
        # Should call service method
        mock_service.switch_trading_mode.assert_called_once_with("simulation", "test-user")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.auth_service')
    async def test_switch_trading_mode_unauthorized(self, mock_auth, client):
        """Test trading mode switch without proper authorization"""
        # Setup mock to return None (unauthorized)
        mock_auth.verify_api_key = AsyncMock(return_value=None)
        
        # Make request
        response = client.post(
            "/dashboard/trading/mode",
            json={"mode": "simulation"},
            headers={"X-API-Key": "invalid-key"}
        )
        
        # Should return 401
        assert response.status_code == 401
        data = response.json()
        assert "Unauthorized" in data["detail"]
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_switch_trading_mode_insufficient_permissions(self, mock_auth, mock_service, client):
        """Test trading mode switch with insufficient permissions"""
        # Setup mock with limited permissions
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions={"dashboard.read"},  # No trading.control
            created_at=datetime.utcnow()
        ))
        
        # Make request
        response = client.post(
            "/dashboard/trading/mode",
            json={"mode": "simulation"},
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 403
        assert response.status_code == 403
        data = response.json()
        assert "Insufficient permissions" in data["detail"]
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_emergency_stop_endpoint(self, mock_auth, mock_service, client):
        """Test POST /dashboard/trading/emergency-stop endpoint"""
        # Setup mocks
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["trading.control"],
            created_at=datetime.utcnow()
        ))
        mock_service.emergency_stop = AsyncMock(return_value=True)
        
        # Make request
        response = client.post(
            "/dashboard/trading/emergency-stop",
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Emergency stop activated" in data["message"]
        
        # Should call service method
        mock_service.emergency_stop.assert_called_once_with("test-user")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_update_risk_limits_endpoint(self, mock_auth, mock_service, client):
        """Test PUT /dashboard/trading/risk-limits endpoint"""
        # Setup mocks
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["trading.control"],
            created_at=datetime.utcnow()
        ))
        mock_service.update_risk_limits = AsyncMock(return_value=True)
        
        risk_limits = {
            "max_position_size_pct": 0.15,
            "max_daily_loss_pct": 0.03,
            "stop_loss_pct": 0.1
        }
        
        # Make request
        response = client.put(
            "/dashboard/trading/risk-limits",
            json=risk_limits,
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Risk limits updated" in data["message"]
        
        # Should call service method
        mock_service.update_risk_limits.assert_called_once_with(risk_limits, "test-user")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_trading_history_endpoint(self, mock_service, client):
        """Test GET /dashboard/trading/history endpoint"""
        # Setup mock data
        mock_trades = [
            {
                "id": "trade-1",
                "symbol": "BTC/USDC",
                "side": "buy",
                "size": "0.1",
                "price": "50000",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        mock_service.get_trading_history = AsyncMock(return_value=mock_trades)
        
        # Make request
        response = client.get("/dashboard/trading/history?limit=50")
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "BTC/USDC"
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_system_logs_endpoint(self, mock_service, client):
        """Test GET /dashboard/system/logs endpoint"""
        # Setup mock data
        mock_logs = [
            "System started successfully",
            "Dashboard service initialized",
            "ML models loaded"
        ]
        mock_service.get_system_logs = AsyncMock(return_value=mock_logs)
        
        # Make request
        response = client.get("/dashboard/system/logs?limit=10")
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert "System started" in data[0]
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_system_restart_endpoint(self, mock_auth, mock_service, client):
        """Test POST /dashboard/system/restart endpoint"""
        # Setup mocks
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["system.control"],
            created_at=datetime.utcnow()
        ))
        
        # Make request
        response = client.post(
            "/dashboard/system/restart",
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "System restart initiated" in data["message"]
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.auth_service')
    async def test_system_restart_insufficient_permissions(self, mock_auth, client):
        """Test system restart with insufficient permissions"""
        # Setup mock with limited permissions
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions={"dashboard.read", "trading.control"},  # No system.control
            created_at=datetime.utcnow()
        ))
        
        # Make request
        response = client.post(
            "/dashboard/system/restart",
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return 403
        assert response.status_code == 403
        data = response.json()
        assert "Insufficient permissions" in data["detail"]


class TestWebSocketEndpoint:
    """Test suite for WebSocket endpoint"""
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Mock WebSocket manager"""
        manager = MagicMock()
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.handle_message = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.websocket_manager')
    @patch('src.dashboard.api.auth_service')
    async def test_websocket_connection_success(self, mock_auth, mock_ws_manager):
        """Test successful WebSocket connection"""
        # Setup mocks
        mock_auth.verify_jwt_token = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["dashboard.read"],
            created_at=datetime.utcnow()
        ))
        
        # Create mock WebSocket
        mock_websocket = MagicMock()
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "subscribe", "topic": "dashboard"}),
            json.dumps({"type": "ping"})
        ])
        mock_websocket.close = AsyncMock()
        
        # Import and call websocket endpoint function
        from src.dashboard.api import websocket_endpoint
        
        # Should handle connection without errors
        try:
            await websocket_endpoint(mock_websocket)
        except Exception:
            # Expected when receive_text runs out of messages
            pass
        
        # Should accept connection
        mock_websocket.accept.assert_called_once()
        
        # Should connect to manager
        mock_ws_manager.connect.assert_called_once()
        
        # Should handle messages
        assert mock_ws_manager.handle_message.call_count >= 1
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.auth_service')
    async def test_websocket_connection_unauthorized(self, mock_auth):
        """Test WebSocket connection with invalid token"""
        # Setup mock to return None (unauthorized)
        mock_auth.verify_jwt_token = AsyncMock(return_value=None)
        
        # Create mock WebSocket
        mock_websocket = MagicMock()
        mock_websocket.query_params = {"token": "invalid-token"}
        mock_websocket.close = AsyncMock()
        
        # Import and call websocket endpoint function
        from src.dashboard.api import websocket_endpoint
        
        # Should close connection
        await websocket_endpoint(mock_websocket)
        mock_websocket.close.assert_called_once_with(code=1008, reason="Unauthorized")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.websocket_manager')
    @patch('src.dashboard.api.auth_service')
    async def test_websocket_connection_error_handling(self, mock_auth, mock_ws_manager):
        """Test WebSocket error handling"""
        # Setup mocks
        mock_auth.verify_jwt_token = AsyncMock(return_value=User(
            user_id="test-user",
            username="testuser",
            permissions=["dashboard.read"],
            created_at=datetime.utcnow()
        ))
        
        # Make receive_text raise an exception
        mock_websocket = MagicMock()
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("Connection error"))
        mock_websocket.close = AsyncMock()
        
        # Import and call websocket endpoint function
        from src.dashboard.api import websocket_endpoint
        
        # Should handle error gracefully
        await websocket_endpoint(mock_websocket)
        
        # Should still disconnect from manager
        mock_ws_manager.disconnect.assert_called_once()


class TestAPIIntegration:
    """Integration tests for Dashboard API"""
    
    @pytest.fixture
    def client(self):
        """Test client for integration testing"""
        return TestClient(test_app)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    @patch('src.dashboard.api.auth_service')
    async def test_full_api_workflow(self, mock_auth, mock_service, client):
        """Test complete API workflow"""
        # Setup mocks
        mock_auth.verify_api_key = AsyncMock(return_value=User(
            user_id="admin-user",
            username="admin",
            permissions=["dashboard.read", "dashboard.write", "trading.control", "system.control"],
            created_at=datetime.utcnow()
        ))
        
        dashboard_data = DashboardData.create_default()
        dashboard_data.trading_status.mode = TradingMode.ANALYSIS
        mock_service.get_dashboard_data = AsyncMock(return_value=dashboard_data)
        mock_service.switch_trading_mode = AsyncMock(return_value=True)
        mock_service.emergency_stop = AsyncMock(return_value=True)
        mock_service.update_risk_limits = AsyncMock(return_value=True)
        mock_service.get_trading_history = AsyncMock(return_value=[])
        mock_service.get_system_logs = AsyncMock(return_value=["System ready"])
        mock_service.record_request = MagicMock()
        
        headers = {"X-API-Key": "admin-api-key"}
        
        # 1. Get dashboard data
        response = client.get("/dashboard/data")
        assert response.status_code == 200
        data = response.json()
        assert data["trading_status"]["mode"] == "analysis"
        
        # 2. Switch to simulation mode
        response = client.post(
            "/dashboard/trading/mode",
            json={"mode": "simulation"},
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # 3. Update risk limits
        response = client.put(
            "/dashboard/trading/risk-limits",
            json={"max_position_size_pct": 0.1, "max_daily_loss_pct": 0.02},
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # 4. Get trading history
        response = client.get("/dashboard/trading/history")
        assert response.status_code == 200
        
        # 5. Get system logs
        response = client.get("/dashboard/system/logs")
        assert response.status_code == 200
        
        # 6. Emergency stop
        response = client.post("/dashboard/trading/emergency-stop", headers=headers)
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # All service methods should have been called
        mock_service.get_dashboard_data.assert_called()
        mock_service.switch_trading_mode.assert_called_once_with("simulation", "admin-user")
        mock_service.update_risk_limits.assert_called_once()
        mock_service.emergency_stop.assert_called_once_with("admin-user")
        mock_service.get_trading_history.assert_called_once()
        mock_service.get_system_logs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_error_responses(self, client):
        """Test API error response formats"""
        # Test endpoint that doesn't exist
        response = client.get("/dashboard/nonexistent")
        assert response.status_code == 404
        
        # Test invalid JSON
        response = client.post(
            "/dashboard/trading/mode",
            data="invalid json",
            headers={"Content-Type": "application/json", "X-API-Key": "test"}
        )
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_request_recording(self, mock_service, client):
        """Test that requests are properly recorded for metrics"""
        # Setup mock
        dashboard_data = DashboardData.create_default()
        mock_service.get_dashboard_data = AsyncMock(return_value=dashboard_data)
        mock_service.record_request = MagicMock()
        
        # Make multiple requests
        for _ in range(3):
            response = client.get("/dashboard/data")
            assert response.status_code == 200
        
        # Should record all requests
        assert mock_service.record_request.call_count == 3
        
        # Check call arguments
        for call_args in mock_service.record_request.call_args_list:
            # Should record response time
            assert len(call_args[0]) == 1  # response_time argument
            assert isinstance(call_args[0][0], float)
            # Should record no error
            assert call_args[1]["error"] is False