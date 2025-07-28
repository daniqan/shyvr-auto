"""
Tests for enhanced dashboard API endpoints (Phase 4.3)
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from src.dashboard.api import dashboard_api
from src.dashboard.base import DashboardData, SystemStatus, TradingMode, Position
from src.dashboard.auth import User
from src.xai.trading_integration import TradingExplanation
from src.xai.data_models import ExplanationData


# Create test app with dashboard routes
test_app = FastAPI()
test_app.include_router(dashboard_api.router)


class TestEnhancedDashboardAPI:
    """Test suite for Enhanced Dashboard API endpoints (Phase 4.3)"""
    
    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return User(
            user_id="test-user-12345",
            username="testuser", 
            permissions={"dashboard.read", "dashboard.write", "trading.control", "admin"},
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def mock_explanation_data(self):
        """Mock XAI explanation data"""
        return ExplanationData(
            feature_importance={
                "price_change": 0.35,
                "volume_ratio": 0.25,
                "rsi": 0.20,
                "ma_crossover": 0.15,
                "volatility": 0.05
            },
            explanation_type="permutation",
            instance_data=[1.0, 2.0, 3.0, 4.0, 5.0],
            model_prediction=0.75,
            confidence_score=0.82,
            explanation_metadata={"model_version": "1.2.0"}
        )
    
    @pytest.fixture
    def mock_trading_explanation(self, mock_explanation_data):
        """Mock trading explanation"""
        return TradingExplanation(
            decision_id="trade-12345",
            timestamp="2024-01-15T10:30:00Z",
            decision_type="buy",
            symbol="SOL/USDC",
            explanation_data=mock_explanation_data,
            model_type="ml_model",
            confidence=0.82,
            metadata={"strategy": "momentum"}
        )
    
    # =============================================================================
    # XAI EXPLANATION ENDPOINT TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_xai_explanations_success(self, mock_service, mock_auth, client, mock_user, mock_trading_explanation):
        """Test successful XAI explanations retrieval"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_xai_explanations = AsyncMock(return_value=[
            {
                "decision_id": mock_trading_explanation.decision_id,
                "timestamp": mock_trading_explanation.timestamp,
                "decision_type": mock_trading_explanation.decision_type,
                "symbol": mock_trading_explanation.symbol,
                "confidence": mock_trading_explanation.confidence,
                "explanation_data": mock_trading_explanation.explanation_data.to_dict()
            }
        ])
        
        # Make request
        response = client.get("/dashboard/xai/explanations?symbol=SOL/USDC&limit=50")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "explanations" in data
        assert "count" in data
        assert "timestamp" in data
        assert len(data["explanations"]) == 1
        assert data["explanations"][0]["decision_id"] == "trade-12345"
        assert data["explanations"][0]["symbol"] == "SOL/USDC"
        
        # Verify service was called with correct parameters
        mock_service.get_xai_explanations.assert_called_once_with(
            symbol="SOL/USDC",
            decision_type=None,
            limit=50
        )
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_xai_explanation_by_id_success(self, mock_service, mock_auth, client, mock_user, mock_trading_explanation):
        """Test successful single XAI explanation retrieval"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_xai_explanation = AsyncMock(return_value={
            "decision_id": mock_trading_explanation.decision_id,
            "explanation_data": mock_trading_explanation.explanation_data.to_dict()
        })
        
        # Make request
        decision_id = "trade-12345"
        response = client.get(f"/dashboard/xai/explanations/{decision_id}")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert data["explanation"]["decision_id"] == decision_id
        
        # Verify service was called
        mock_service.get_xai_explanation.assert_called_once_with(decision_id)
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_xai_explanation_not_found(self, mock_service, mock_auth, client, mock_user):
        """Test XAI explanation not found"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_xai_explanation = AsyncMock(return_value=None)
        
        # Make request
        response = client.get("/dashboard/xai/explanations/nonexistent-id")
        
        # Assertions
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_feature_importance_summary(self, mock_service, mock_auth, client, mock_user):
        """Test feature importance summary endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_feature_importance_summary = AsyncMock(return_value={
            "price_change": 0.35,
            "volume_ratio": 0.25,
            "rsi": 0.20
        })
        
        # Make request
        response = client.get("/dashboard/xai/feature-importance?symbol=BTC/USDC&hours_back=48")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "feature_importance" in data
        assert "time_range" in data
        assert len(data["feature_importance"]) == 3
        assert data["feature_importance"]["price_change"] == 0.35
        
        # Verify service was called with correct parameters
        mock_service.get_feature_importance_summary.assert_called_once_with(
            symbol="BTC/USDC",
            hours_back=48
        )
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_xai_cache_stats(self, mock_service, mock_auth, client, mock_user):
        """Test XAI cache statistics endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_xai_cache_stats = AsyncMock(return_value={
            "explanation_cache_size": 150,
            "explainer_cache_size": 5,
            "cache_limit": 1000,
            "enabled": True
        })
        
        # Make request
        response = client.get("/dashboard/xai/cache-stats")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data
        assert data["cache_stats"]["explanation_cache_size"] == 150
        assert data["cache_stats"]["enabled"] is True
    
    # =============================================================================
    # INTERACTIVE CHARTING ENDPOINT TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_price_chart_data(self, mock_service, mock_auth, client, mock_user):
        """Test price chart data endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_price_chart_data = AsyncMock(return_value=[
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1500.0
            },
            {
                "timestamp": "2024-01-15T11:00:00Z", 
                "open": 103.0,
                "high": 107.0,
                "low": 101.0,
                "close": 106.0,
                "volume": 1800.0
            }
        ])
        
        # Make request
        response = client.get("/dashboard/charts/price-data?symbol=ETH/USDC&timeframe=1h&limit=100")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "timeframe" in data
        assert "data" in data
        assert "count" in data
        assert data["symbol"] == "ETH/USDC"
        assert data["timeframe"] == "1h"
        assert len(data["data"]) == 2
        assert data["data"][0]["open"] == 100.0
        
        # Verify service was called
        mock_service.get_price_chart_data.assert_called_once_with(
            symbol="ETH/USDC",
            timeframe="1h",
            limit=100
        )
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_performance_chart_data(self, mock_service, mock_auth, client, mock_user):
        """Test performance chart data endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_performance_chart_data = AsyncMock(return_value=[
            {
                "timestamp": "2024-01-15T00:00:00Z",
                "portfolio_value": 10000.0,
                "daily_return": 0.0,
                "cumulative_return": 0.0
            },
            {
                "timestamp": "2024-01-16T00:00:00Z",
                "portfolio_value": 10250.0,
                "daily_return": 2.5,
                "cumulative_return": 2.5
            }
        ])
        
        # Make request
        response = client.get("/dashboard/charts/performance-data?timeframe=1d&days_back=7")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "timeframe" in data
        assert "days_back" in data
        assert "data" in data
        assert data["timeframe"] == "1d"
        assert data["days_back"] == 7
        assert len(data["data"]) == 2
        
        # Verify service was called
        mock_service.get_performance_chart_data.assert_called_once_with(
            timeframe="1d",
            days_back=7
        )
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_trading_volume_chart_data(self, mock_service, mock_auth, client, mock_user):
        """Test trading volume chart data endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_trading_volume_chart_data = AsyncMock(return_value=[
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "volume_usd": 2500.0,
                "trade_count": 15,
                "avg_trade_size": 166.67
            }
        ])
        
        # Make request
        response = client.get("/dashboard/charts/trading-volume-data?timeframe=1h&hours_back=12")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "timeframe" in data
        assert "hours_back" in data
        assert "data" in data
        assert len(data["data"]) == 1
        
        # Verify service was called
        mock_service.get_trading_volume_chart_data.assert_called_once_with(
            timeframe="1h",
            hours_back=12
        )
    
    # =============================================================================
    # PERFORMANCE ATTRIBUTION ENDPOINT TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_performance_attribution(self, mock_service, mock_auth, client, mock_user):
        """Test performance attribution endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_performance_attribution = AsyncMock(return_value={
            "timeframe": "1d",
            "days_analyzed": 30,
            "strategy_attribution": {
                "momentum": {
                    "return_contribution": 1.2,
                    "risk_contribution": 0.3,
                    "sharpe_ratio": 1.8,
                    "trade_count": 45,
                    "win_rate": 0.67
                }
            },
            "total_return": 2.5,
            "risk_adjusted_return": 1.4
        })
        
        # Make request
        response = client.get("/dashboard/performance/attribution?timeframe=1d&days_back=30")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "attribution" in data
        assert "timeframe" in data
        assert "days_back" in data
        assert data["attribution"]["total_return"] == 2.5
        
        # Verify service was called
        mock_service.get_performance_attribution.assert_called_once_with(
            timeframe="1d",
            days_back=30
        )
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_risk_metrics(self, mock_service, mock_auth, client, mock_user):
        """Test risk metrics endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_risk_metrics = AsyncMock(return_value={
            "timeframe": "1d",
            "days_analyzed": 30,
            "value_at_risk": {
                "var_95": 250.0,
                "var_99": 400.0,
                "cvar_95": 350.0
            },
            "volatility_metrics": {
                "daily_volatility": 0.025,
                "annualized_volatility": 0.25
            }
        })
        
        # Make request
        response = client.get("/dashboard/performance/risk-metrics?timeframe=1d&days_back=30")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "risk_metrics" in data
        assert data["risk_metrics"]["value_at_risk"]["var_95"] == 250.0
        
        # Verify service was called
        mock_service.get_risk_metrics.assert_called_once_with(
            timeframe="1d",
            days_back=30
        )
    
    # =============================================================================
    # MANUAL OVERRIDE CONTROL ENDPOINT TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_trading')
    @patch('src.dashboard.service.dashboard_service')
    def test_execute_manual_trade_success(self, mock_service, mock_auth, client, mock_user):
        """Test successful manual trade execution"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.execute_manual_trade = AsyncMock(return_value=True)
        
        # Make request
        trade_request = {
            "symbol": "BTC/USDC",
            "side": "buy",
            "amount": 0.1,
            "order_type": "market"
        }
        response = client.post("/dashboard/controls/manual-trade", json=trade_request)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "trade executed" in data["message"].lower()
        
        # Verify service was called
        mock_service.execute_manual_trade.assert_called_once_with(
            symbol="BTC/USDC",
            side="buy",
            amount=0.1,
            order_type="market",
            price=None,
            user_id=mock_user.user_id
        )
    
    @patch('src.dashboard.auth.require_trading')
    @patch('src.dashboard.service.dashboard_service')
    def test_execute_manual_trade_validation_error(self, mock_service, mock_auth, client, mock_user):
        """Test manual trade with validation error"""
        # Setup mocks
        mock_auth.return_value = mock_user
        
        # Make request with invalid side
        trade_request = {
            "symbol": "BTC/USDC",
            "side": "invalid_side",  # Invalid side
            "amount": 0.1,
            "order_type": "market"
        }
        response = client.post("/dashboard/controls/manual-trade", json=trade_request)
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    @patch('src.dashboard.auth.require_trading')
    @patch('src.dashboard.service.dashboard_service')
    def test_override_trading_signal(self, mock_service, mock_auth, client, mock_user):
        """Test trading signal override"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.override_trading_signal = AsyncMock(return_value=True)
        
        # Make request
        override_request = {
            "signal_id": "signal-12345",
            "action": "override",
            "reason": "Manual intervention required"
        }
        response = client.post("/dashboard/controls/override-signal", json=override_request)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify service was called
        mock_service.override_trading_signal.assert_called_once_with(
            signal_id="signal-12345",
            action="override",
            reason="Manual intervention required",
            user_id=mock_user.user_id
        )
    
    @patch('src.dashboard.auth.require_trading')
    @patch('src.dashboard.service.dashboard_service')
    def test_pause_trading_strategy(self, mock_service, mock_auth, client, mock_user):
        """Test trading strategy pause"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.pause_trading_strategy = AsyncMock(return_value=True)
        
        # Make request
        pause_request = {
            "strategy_name": "momentum_strategy",
            "duration_minutes": 60,
            "reason": "Market volatility too high"
        }
        response = client.post("/dashboard/controls/pause-strategy", json=pause_request)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "paused" in data["message"].lower()
        
        # Verify service was called
        mock_service.pause_trading_strategy.assert_called_once_with(
            strategy_name="momentum_strategy",
            duration_minutes=60,
            reason="Market volatility too high",
            user_id=mock_user.user_id
        )
    
    # =============================================================================
    # REAL-TIME METRICS ENDPOINT TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_realtime_metrics(self, mock_service, mock_auth, client, mock_user):
        """Test real-time metrics endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_realtime_metrics = AsyncMock(return_value={
            "system": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "active_connections": 12,
                "requests_per_minute": 45.0,
                "error_rate": 0.5,
                "response_time_ms": 120.0
            },
            "trading": {
                "mode": "simulation",
                "is_active": True,
                "trades_today": 8,
                "volume_today": 2500.0,
                "win_rate": 65.5,
                "signals_count": 5
            },
            "portfolio": {
                "total_value": 10250.0,
                "daily_pnl": 75.25,
                "daily_pnl_pct": 0.75,
                "positions_count": 3,
                "available_balance": 8500.0
            },
            "ml_rl": {
                "ml_prediction_accuracy": 77.5,
                "rl_action_success_rate": 64.2,
                "integration_active": True,
                "decision_latency_ms": 9.8
            }
        })
        
        # Make request
        response = client.get("/dashboard/realtime/metrics")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "system" in metrics
        assert "trading" in metrics
        assert "portfolio" in metrics
        assert "ml_rl" in metrics
        
        assert metrics["system"]["cpu_usage"] == 45.2
        assert metrics["trading"]["mode"] == "simulation"
        assert metrics["portfolio"]["total_value"] == 10250.0
        assert metrics["ml_rl"]["integration_active"] is True
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_get_live_positions(self, mock_service, mock_auth, client, mock_user):
        """Test live positions endpoint"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_position = Position(
            symbol="BTC/USDC",
            chain="solana",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            unrealized_pnl=Decimal("100.0"),
            unrealized_pnl_pct=Decimal("2.0"),
            margin_used=Decimal("1000.0"),
            leverage=5.0,
            entry_time=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            mode=TradingMode.SIMULATION,
            is_simulated=True
        )
        mock_service.get_live_positions = AsyncMock(return_value=[mock_position])
        
        # Make request
        response = client.get("/dashboard/realtime/live-positions")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert "count" in data
        assert data["count"] == 1
        assert len(data["positions"]) == 1
        
        position = data["positions"][0]
        assert position["symbol"] == "BTC/USDC"
        assert position["side"] == "long"
        assert float(position["size"]) == 0.1
    
    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_xai_explanations_service_error(self, mock_service, mock_auth, client, mock_user):
        """Test XAI explanations endpoint with service error"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_xai_explanations = AsyncMock(side_effect=Exception("Service error"))
        
        # Make request
        response = client.get("/dashboard/xai/explanations")
        
        # Assertions
        assert response.status_code == 500
        assert "Failed to retrieve XAI explanations" in response.json()["detail"]
    
    @patch('src.dashboard.auth.require_trading')
    @patch('src.dashboard.service.dashboard_service')
    def test_manual_trade_service_failure(self, mock_service, mock_auth, client, mock_user):
        """Test manual trade with service failure"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.execute_manual_trade = AsyncMock(return_value=False)
        
        # Make request
        trade_request = {
            "symbol": "ETH/USDC",
            "side": "sell",
            "amount": 1.0,
            "order_type": "market"
        }
        response = client.post("/dashboard/controls/manual-trade", json=trade_request)
        
        # Assertions
        assert response.status_code == 400
        assert "Failed to execute manual trade" in response.json()["detail"]
    
    @patch('src.dashboard.auth.require_read')
    @patch('src.dashboard.service.dashboard_service')
    def test_chart_data_service_error(self, mock_service, mock_auth, client, mock_user):
        """Test chart data endpoint with service error"""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_service.get_price_chart_data = AsyncMock(side_effect=Exception("Chart service error"))
        
        # Make request
        response = client.get("/dashboard/charts/price-data?symbol=BTC/USDC")
        
        # Assertions
        assert response.status_code == 500
        assert "Failed to retrieve price chart data" in response.json()["detail"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEnhancedDashboardAPIIntegration:
    """Integration tests for enhanced dashboard API endpoints"""
    
    @pytest.fixture
    def mock_xai_manager(self):
        """Mock XAI explanation manager"""
        manager = MagicMock()
        manager.get_recent_explanations = MagicMock(return_value=[])
        manager.get_explanation = MagicMock(return_value=None)
        manager.get_feature_importance_summary = MagicMock(return_value={})
        manager.get_cache_stats = MagicMock(return_value={})
        return manager
    
    @patch('src.dashboard.service.dashboard_service')
    def test_xai_explanation_flow_integration(self, mock_service):
        """Test complete XAI explanation flow"""
        # This would test the complete flow from API endpoint to XAI system
        # In a real integration test, we would verify:
        # 1. API endpoint receives request
        # 2. Service layer processes request
        # 3. XAI system generates explanation
        # 4. Response is properly formatted and returned
        pass  # Placeholder for integration test
    
    @patch('src.dashboard.service.dashboard_service')
    def test_manual_trading_control_integration(self, mock_service):
        """Test complete manual trading control integration"""
        # This would test the complete flow from manual trade request
        # to actual trade execution in simulation/live modes
        pass  # Placeholder for integration test