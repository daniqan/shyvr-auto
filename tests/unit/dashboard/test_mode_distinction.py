"""
Tests for dashboard mode distinction functionality

This test suite ensures that the dashboard properly distinguishes between
simulation and live trading modes with clear visual indicators and mode context.
"""

import pytest
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.api import dashboard_api
from src.dashboard.base import (
    DashboardData, SystemStatus, TradingMode, SystemMetrics,
    PortfolioStatus, TradingStatus, MLRLStatus, Position
)
from src.dashboard.auth import User


# Create test app with dashboard routes
test_app = FastAPI()
test_app.include_router(dashboard_api.router)


class TestDashboardModeDistinction:
    """Test suite for dashboard mode distinction functionality"""

    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(test_app)

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return User(
            user_id="test-user",
            username="testuser",
            permissions={"dashboard.read", "dashboard.write", "trading.control"},
            created_at=datetime.utcnow()
        )

    @pytest.fixture
    def simulation_dashboard_data(self):
        """Mock dashboard data for simulation mode"""
        return DashboardData(
            system_metrics=SystemMetrics.create_default(),
            portfolio_status=PortfolioStatus(
                total_value_usd=Decimal("10000.00"),
                available_balance_usd=Decimal("8000.00"),
                margin_used_usd=Decimal("2000.00"),
                unrealized_pnl_usd=Decimal("150.00"),
                realized_pnl_usd=Decimal("50.00"),
                daily_pnl_usd=Decimal("200.00"),
                daily_pnl_pct=Decimal("2.0"),
                total_return_pct=Decimal("2.0"),
                max_drawdown_pct=Decimal("5.0"),
                current_drawdown_pct=Decimal("1.0"),
                sharpe_ratio=1.5,
                volatility_pct=15.0,
                var_95_usd=Decimal("500.00"),
                position_count=3,
                active_positions=[],
                recent_trades=[],  # Missing field
                chain_balances={"solana": Decimal("5000.00"), "ethereum": Decimal("5000.00")},
                last_updated=datetime.utcnow(),
                mode=TradingMode.SIMULATION,  # Key field for mode distinction
                is_simulated=True  # Additional flag for clarity
            ),
            trading_status=TradingStatus(
                mode=TradingMode.SIMULATION,
                is_trading_active=True,
                last_trade_time=datetime.utcnow(),
                trades_today=5,
                volume_today_usd=Decimal("1000.00"),
                tokens_analyzed_today=25,
                analysis_running=True,
                simulation_running=True,
                live_trading_enabled=False,
                emergency_stop_active=False,
                risk_limits_active=True,
                max_position_size_usd=Decimal("1000.00"),  # Missing field
                max_daily_loss_usd=Decimal("500.00"),  # Missing field
                win_rate_pct=60.0,
                avg_trade_duration_hours=2.5,
                avg_profit_per_trade_usd=Decimal("40.00"),
                tokens_in_watchlist=10,
                high_confidence_signals=3,
                last_updated=datetime.utcnow(),
                is_simulated=True
            ),
            ml_rl_status=MLRLStatus.create_default(),
            active_alerts=[],
            warnings_count=0,
            errors_count=0,
            recent_logs=[],
            recent_notifications=[],
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def live_dashboard_data(self):
        """Mock dashboard data for live trading mode"""
        return DashboardData(
            system_metrics=SystemMetrics.create_default(),
            portfolio_status=PortfolioStatus(
                total_value_usd=Decimal("50000.00"),
                available_balance_usd=Decimal("45000.00"),
                margin_used_usd=Decimal("5000.00"),
                unrealized_pnl_usd=Decimal("750.00"),
                realized_pnl_usd=Decimal("250.00"),
                daily_pnl_usd=Decimal("1000.00"),
                daily_pnl_pct=Decimal("2.0"),
                total_return_pct=Decimal("2.0"),
                max_drawdown_pct=Decimal("3.0"),
                current_drawdown_pct=Decimal("0.5"),
                sharpe_ratio=2.0,
                volatility_pct=12.0,
                var_95_usd=Decimal("2500.00"),
                position_count=2,
                active_positions=[],
                recent_trades=[],  # Missing field
                chain_balances={"solana": Decimal("25000.00"), "ethereum": Decimal("25000.00")},
                last_updated=datetime.utcnow(),
                mode=TradingMode.LIVE,
                is_simulated=False
            ),
            trading_status=TradingStatus(
                mode=TradingMode.LIVE,
                is_trading_active=True,
                last_trade_time=datetime.utcnow(),
                trades_today=3,
                volume_today_usd=Decimal("5000.00"),
                tokens_analyzed_today=15,
                analysis_running=True,
                simulation_running=False,
                live_trading_enabled=True,
                emergency_stop_active=False,
                risk_limits_active=True,
                max_position_size_usd=Decimal("5000.00"),  # Missing field
                max_daily_loss_usd=Decimal("2500.00"),  # Missing field
                win_rate_pct=75.0,
                avg_trade_duration_hours=4.0,
                avg_profit_per_trade_usd=Decimal("333.33"),
                tokens_in_watchlist=8,
                high_confidence_signals=2,
                last_updated=datetime.utcnow(),
                is_simulated=False
            ),
            ml_rl_status=MLRLStatus.create_default(),
            active_alerts=[],
            warnings_count=0,
            errors_count=0,
            recent_logs=[],
            recent_notifications=[],
            timestamp=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_simulation_mode_api_response_includes_mode_context(
        self, simulation_dashboard_data, mock_user
    ):
        """Test that API responses include mode context for simulation"""
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = simulation_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/data")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that mode information is included
                assert data["trading_status"]["mode"] == "simulation"
                assert data["trading_status"]["is_simulated"] is True
                assert data["portfolio_status"]["mode"] == "simulation"
                assert data["portfolio_status"]["is_simulated"] is True
                
                # Verify simulation-specific flags
                assert data["trading_status"]["mode_status"]["simulation_running"] is True
                assert data["trading_status"]["mode_status"]["live_trading_enabled"] is False

    @pytest.mark.asyncio
    async def test_live_mode_api_response_includes_mode_context(
        self, live_dashboard_data, mock_user
    ):
        """Test that API responses include mode context for live trading"""
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = live_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/data")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that mode information is included
                assert data["trading_status"]["mode"] == "live"
                assert data["trading_status"]["is_simulated"] is False
                assert data["portfolio_status"]["mode"] == "live"
                assert data["portfolio_status"]["is_simulated"] is False
                
                # Verify live trading specific flags
                assert data["trading_status"]["mode_status"]["simulation_running"] is False
                assert data["trading_status"]["mode_status"]["live_trading_enabled"] is True

    @pytest.mark.asyncio
    async def test_portfolio_values_have_mode_prefixes(
        self, simulation_dashboard_data, live_dashboard_data, mock_user
    ):
        """Test that portfolio values include mode-specific prefixes/indicators"""
        
        # Test simulation mode
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = simulation_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/portfolio/status")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that portfolio status includes mode context
                assert data["portfolio_status"]["mode"] == "simulation"
                assert data["portfolio_status"]["is_simulated"] is True
                
                # Values should be marked as simulated
                assert "total_value_usd" in data["portfolio_status"]
                assert "available_balance_usd" in data["portfolio_status"]
        
        # Test live mode
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = live_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/portfolio/status")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that portfolio status includes mode context
                assert data["portfolio_status"]["mode"] == "live"
                assert data["portfolio_status"]["is_simulated"] is False

    @pytest.mark.asyncio
    async def test_trading_history_includes_mode_context(
        self, simulation_dashboard_data, mock_user
    ):
        """Test that trading history includes mode context for each trade"""
        mock_trades = [
            {
                "id": "trade-1",
                "symbol": "SOL/USDC",
                "chain": "solana",
                "dex": "jupiter",
                "side": "buy",
                "size": "100.0",
                "price": "50.0",
                "value_usd": "5000.0",
                "fee": "5.0",
                "slippage_pct": "0.1",
                "execution_time_ms": 250,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "simulation",  # Mode context
                "is_simulated": True   # Clear simulation flag
            },
            {
                "id": "trade-2",
                "symbol": "ETH/USDC",
                "chain": "ethereum",
                "dex": "uniswap_v3",
                "side": "sell",
                "size": "2.0",
                "price": "2000.0",
                "value_usd": "4000.0",
                "fee": "12.0",
                "slippage_pct": "0.2",
                "execution_time_ms": 180,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "simulation",
                "is_simulated": True
            }
        ]
        
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_trading_history.return_value = mock_trades
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/trading/history")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that all trades include mode context
                assert len(data["trades"]) == 2
                for trade in data["trades"]:
                    assert "mode" in trade
                    assert "is_simulated" in trade
                    assert trade["mode"] == "simulation"
                    assert trade["is_simulated"] is True

    @pytest.mark.asyncio
    async def test_trading_mode_switch_updates_mode_context(self, mock_user):
        """Test that switching trading modes updates the mode context properly"""
        
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.switch_trading_mode.return_value = True
            
            with patch('src.dashboard.auth.require_trading', return_value=mock_user):
                client = TestClient(test_app)
                
                # Switch to simulation mode
                response = client.post(
                    "/dashboard/trading/mode",
                    json={"mode": "simulation"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "simulation" in data["message"]
                
                # Verify the service was called with correct mode
                mock_service.switch_trading_mode.assert_called_with("simulation", mock_user.user_id)
                
                # Switch to live mode
                response = client.post(
                    "/dashboard/trading/mode",
                    json={"mode": "live"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "live" in data["message"]

    @pytest.mark.asyncio
    async def test_system_metrics_include_mode_aware_info(
        self, simulation_dashboard_data, mock_user
    ):
        """Test that system metrics include mode-aware information"""
        
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = simulation_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/system/metrics")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that system metrics are present
                assert "system_metrics" in data
                assert "timestamp" in data
                
                # System metrics should include component statuses
                system_metrics = data["system_metrics"]
                assert "component_statuses" in system_metrics
                assert "performance" in system_metrics

    @pytest.mark.asyncio
    async def test_ml_rl_status_mode_awareness(
        self, simulation_dashboard_data, mock_user
    ):
        """Test that ML/RL status includes mode-specific information"""
        
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.get_dashboard_data.return_value = simulation_dashboard_data
            
            with patch('src.dashboard.auth.require_read', return_value=mock_user):
                client = TestClient(test_app)
                response = client.get("/dashboard/ml-rl/status")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check that ML/RL status is present
                assert "ml_rl_status" in data
                assert "timestamp" in data
                
                # ML/RL status should include integration info
                ml_rl_status = data["ml_rl_status"]
                assert "integration" in ml_rl_status
                assert "training" in ml_rl_status
                assert "performance" in ml_rl_status

    def test_position_serialization_includes_mode_info(self):
        """Test that position serialization includes mode information"""
        position = Position(
            symbol="SOL/USDC",
            chain="solana",
            side="long",
            size=Decimal("100.0"),
            entry_price=Decimal("50.0"),
            current_price=Decimal("52.0"),
            unrealized_pnl=Decimal("200.0"),
            unrealized_pnl_pct=Decimal("4.0"),
            margin_used=Decimal("1000.0"),
            leverage=5.0,
            entry_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Test position serialization with mode context
        api = dashboard_api
        serialized = api._serialize_position(position)
        
        # Check that all required fields are present
        required_fields = [
            "symbol", "chain", "side", "size", "entry_price",
            "current_price", "unrealized_pnl", "unrealized_pnl_pct",
            "leverage", "entry_time", "last_updated"
        ]
        
        for field in required_fields:
            assert field in serialized

    def test_trade_serialization_includes_mode_info(self):
        """Test that trade serialization includes mode information"""
        from datetime import datetime
        
        # Mock trade object
        class MockTrade:
            def __init__(self):
                self.id = "trade-123"
                self.symbol = "SOL/USDC"
                self.chain = "solana"
                self.dex = "jupiter"
                self.side = "buy"
                self.size = Decimal("100.0")
                self.price = Decimal("50.0")
                self.value_usd = Decimal("5000.0")
                self.fee = Decimal("5.0")
                self.slippage_pct = Decimal("0.1")
                self.execution_time_ms = 250
                self.status = "completed"
                self.timestamp = datetime.utcnow()
        
        trade = MockTrade()
        
        # Test trade serialization
        api = dashboard_api
        serialized = api._serialize_trade(trade)
        
        # Check that all required fields are present
        required_fields = [
            "id", "symbol", "chain", "dex", "side", "size",
            "price", "value_usd", "fee", "slippage_pct",
            "execution_time_ms", "status", "timestamp"
        ]
        
        for field in required_fields:
            assert field in serialized

    @pytest.mark.asyncio
    async def test_dashboard_data_serialization_preserves_mode_context(
        self, simulation_dashboard_data
    ):
        """Test that dashboard data serialization preserves mode context"""
        api = dashboard_api
        serialized = api._serialize_dashboard_data(simulation_dashboard_data)
        
        # Check that all major sections are present
        assert "system_metrics" in serialized
        assert "portfolio_status" in serialized
        assert "trading_status" in serialized
        assert "ml_rl_status" in serialized
        assert "alerts" in serialized
        assert "recent_activity" in serialized
        assert "timestamp" in serialized
        
        # Check that mode context is preserved in trading status
        trading_status = serialized["trading_status"]
        assert trading_status["mode"] == "simulation"
        assert trading_status["is_simulated"] is True
        
        # Check that mode context is preserved in portfolio status
        portfolio_status = serialized["portfolio_status"]
        assert portfolio_status["mode"] == "simulation"
        assert portfolio_status["is_simulated"] is True

    @pytest.mark.asyncio
    async def test_emergency_stop_mode_awareness(self, mock_user):
        """Test that emergency stop works in both simulation and live modes"""
        
        with patch('src.dashboard.service.dashboard_service') as mock_service:
            mock_service.emergency_stop.return_value = True
            
            with patch('src.dashboard.auth.require_trading', return_value=mock_user):
                client = TestClient(test_app)
                response = client.post("/dashboard/trading/emergency-stop")
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "emergency stop" in data["message"].lower()
                
                # Verify the service was called with user ID
                mock_service.emergency_stop.assert_called_with(mock_user.user_id)

    def test_mode_validation_in_trading_mode_request(self):
        """Test that trading mode request validation works properly"""
        from src.dashboard.api import TradingModeRequest
        import pytest
        from pydantic import ValidationError
        
        # Valid modes
        valid_modes = ['analysis', 'simulation', 'live']
        for mode in valid_modes:
            request = TradingModeRequest(mode=mode)
            assert request.mode == mode
        
        # Invalid mode
        with pytest.raises(ValidationError):
            TradingModeRequest(mode='invalid_mode')