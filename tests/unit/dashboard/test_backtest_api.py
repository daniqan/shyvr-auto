"""
Tests for dashboard backtest results API endpoint
"""

import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.dashboard.api import DashboardAPI
from src.dashboard.service import DashboardService
from src.modes.analysis_mode import AnalysisMode, BacktestEngine


class TestBacktestResultsAPI:
    """Test suite for backtest results API endpoint"""

    @pytest.fixture
    def dashboard_api(self):
        """Dashboard API instance for testing"""
        return DashboardAPI()

    @pytest.fixture
    def mock_user(self):
        """Mock user for authentication"""
        user = MagicMock()
        user.user_id = "test-user-123"
        user.username = "testuser"
        user.permissions = {"read", "trading"}
        return user

    @pytest.fixture
    def sample_backtest_result(self):
        """Sample backtest result data"""
        return {
            "strategy_name": "LSTM_DQN_Strategy",
            "total_return": 0.125,  # 12.5%
            "annual_return": 0.15,   # 15%
            "max_drawdown": -0.08,   # -8%
            "sharpe_ratio": 1.42,
            "volatility": 0.18,
            "win_rate": 0.65,
            "total_trades": 156,
            "profitable_trades": 101,
            "losing_trades": 55,
            "avg_trade_duration_hours": 4.2,
            "avg_profit_per_trade": 12.45,
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "initial_balance": 10000.0,
            "final_balance": 11250.0,
            "equity_curve": [
                {"timestamp": "2024-01-01T00:00:00Z", "value": 10000.0},
                {"timestamp": "2024-01-05T00:00:00Z", "value": 10150.0},
                {"timestamp": "2024-01-10T00:00:00Z", "value": 10320.0},
                {"timestamp": "2024-01-15T00:00:00Z", "value": 10180.0},
                {"timestamp": "2024-01-20T00:00:00Z", "value": 10650.0},
                {"timestamp": "2024-01-25T00:00:00Z", "value": 10890.0},
                {"timestamp": "2024-01-31T23:59:59Z", "value": 11250.0}
            ],
            "trade_history": [
                {
                    "timestamp": "2024-01-02T10:30:00Z",
                    "symbol": "SOL/USDC",
                    "side": "BUY",
                    "quantity": 15.5,
                    "price": 95.40,
                    "value": 1478.70,
                    "pnl": 45.20,
                    "duration_hours": 6.5
                },
                {
                    "timestamp": "2024-01-03T14:15:00Z",
                    "symbol": "ETH/USDC",
                    "side": "SELL",
                    "quantity": 2.1,
                    "price": 2340.50,
                    "value": 4915.05,
                    "pnl": -22.10,
                    "duration_hours": 3.2
                }
            ],
            "performance_metrics": {
                "sortino_ratio": 1.88,
                "calmar_ratio": 1.875,
                "maximum_consecutive_wins": 8,
                "maximum_consecutive_losses": 3,
                "profit_factor": 2.1,
                "recovery_factor": 1.95
            }
        }

    @pytest.mark.asyncio
    async def test_get_backtest_results_endpoint_exists(self, dashboard_api):
        """Test that backtest results endpoint is properly registered"""
        # Check if the route exists in the router
        routes = dashboard_api.router.routes
        backtest_route = None
        
        for route in routes:
            if hasattr(route, 'path') and '/backtest/results' in route.path:
                backtest_route = route
                break
        
        # For now, this will fail since we haven't implemented the endpoint yet
        # This is the first failing test in our TDD approach
        assert backtest_route is not None, "Backtest results endpoint should be registered"

    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_backtest_results_success(self, mock_dashboard_service, dashboard_api, mock_user, sample_backtest_result):
        """Test successful backtest results retrieval"""
        # Mock the dashboard service to return backtest results
        mock_dashboard_service.get_backtest_results = AsyncMock(return_value=sample_backtest_result)
        
        # This test will fail until we implement the endpoint
        # Call the endpoint method directly (once implemented)
        try:
            result = await dashboard_api.get_backtest_results(user=mock_user)
            
            # Verify response structure
            assert "backtest_results" in result
            assert "timestamp" in result
            
            backtest_data = result["backtest_results"]
            assert backtest_data["strategy_name"] == "LSTM_DQN_Strategy"
            assert backtest_data["total_return"] == 0.125
            assert backtest_data["sharpe_ratio"] == 1.42
            assert len(backtest_data["equity_curve"]) == 7
            assert len(backtest_data["trade_history"]) == 2
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results method not implemented yet")

    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_backtest_results_with_filters(self, mock_dashboard_service, dashboard_api, mock_user, sample_backtest_result):
        """Test backtest results retrieval with filters"""
        mock_dashboard_service.get_backtest_results = AsyncMock(return_value=sample_backtest_result)
        
        # Test with various filters
        filters = {
            "strategy_name": "LSTM_DQN_Strategy",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "min_sharpe_ratio": 1.0
        }
        
        try:
            result = await dashboard_api.get_backtest_results(
                strategy_name=filters["strategy_name"],
                start_date=filters["start_date"],
                end_date=filters["end_date"],
                min_sharpe_ratio=filters["min_sharpe_ratio"],
                user=mock_user
            )
            
            # Verify filters were applied
            mock_dashboard_service.get_backtest_results.assert_called_once_with(
                strategy_name=filters["strategy_name"],
                start_date=filters["start_date"],
                end_date=filters["end_date"],
                min_sharpe_ratio=filters["min_sharpe_ratio"]
            )
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results method with filters not implemented yet")

    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_backtest_results_no_data(self, mock_dashboard_service, dashboard_api, mock_user):
        """Test backtest results when no data is available"""
        # Mock empty results
        mock_dashboard_service.get_backtest_results = AsyncMock(return_value=None)
        
        try:
            result = await dashboard_api.get_backtest_results(user=mock_user)
            
            assert "backtest_results" in result
            assert result["backtest_results"] is None
            assert "message" in result
            assert "No backtest results available" in result["message"]
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results method not implemented yet")

    @pytest.mark.asyncio
    @patch('src.dashboard.api.dashboard_service')
    async def test_get_backtest_results_error_handling(self, mock_dashboard_service, dashboard_api, mock_user):
        """Test backtest results error handling"""
        # Mock service to raise an exception
        mock_dashboard_service.get_backtest_results = AsyncMock(side_effect=Exception("Database connection failed"))
        
        try:
            with pytest.raises(Exception) as excinfo:
                await dashboard_api.get_backtest_results(user=mock_user)
            
            assert "Failed to retrieve backtest results" in str(excinfo.value)
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results method not implemented yet")

    @pytest.mark.asyncio
    async def test_get_backtest_results_authentication(self, dashboard_api):
        """Test that backtest results endpoint requires authentication"""
        # This should test that the endpoint requires proper authentication
        # Will fail until implemented
        try:
            # Try to call without authentication
            result = await dashboard_api.get_backtest_results()
            pytest.fail("Should require authentication")
        except TypeError:
            # Expected - missing required user parameter
            pass
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results method not implemented yet")


class TestBacktestResultsService:
    """Test suite for backtest results service methods"""

    @pytest.fixture  
    def dashboard_service(self):
        """Dashboard service instance for testing"""
        return DashboardService()

    @pytest.fixture
    def mock_analysis_mode(self):
        """Mock analysis mode with backtest engine"""
        analysis_mode = MagicMock(spec=AnalysisMode)
        analysis_mode.backtest_engine = MagicMock(spec=BacktestEngine)
        return analysis_mode

    @pytest.fixture
    def sample_raw_backtest_data(self):
        """Sample raw backtest data from analysis mode"""
        return {
            "strategy_name": "LSTM_DQN_Strategy",
            "total_return": 0.125,
            "annual_return": 0.15,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.42,
            "trade_history": [
                {"timestamp": datetime(2024, 1, 2, 10, 30), "symbol": "SOL/USDC", "side": "BUY", "pnl": 45.20},
                {"timestamp": datetime(2024, 1, 3, 14, 15), "symbol": "ETH/USDC", "side": "SELL", "pnl": -22.10}
            ],
            "equity_curve": [
                {"timestamp": datetime(2024, 1, 1), "value": Decimal("10000.0")},
                {"timestamp": datetime(2024, 1, 31), "value": Decimal("11250.0")}
            ]
        }

    @pytest.mark.asyncio
    @patch('src.dashboard.service.get_mode_manager')
    async def test_get_backtest_results_from_analysis_mode(self, mock_get_mode_manager, dashboard_service, mock_analysis_mode, sample_raw_backtest_data):
        """Test getting backtest results from analysis mode"""
        # Mock mode manager to return analysis mode
        mock_mode_manager = MagicMock()
        mock_mode_manager.get_active_mode.return_value = mock_analysis_mode
        mock_mode_manager.get_mode_by_type.return_value = mock_analysis_mode
        mock_get_mode_manager.return_value = mock_mode_manager
        
        # Mock backtest engine to return sample data
        mock_analysis_mode.backtest_engine.get_latest_results.return_value = sample_raw_backtest_data
        
        # This will fail until we implement the service method
        try:
            result = await dashboard_service.get_backtest_results()
            
            # Verify data transformation
            assert result["strategy_name"] == "LSTM_DQN_Strategy"
            assert result["total_return"] == 0.125
            assert len(result["trade_history"]) == 2
            assert len(result["equity_curve"]) == 2
            
            # Verify datetime serialization
            for trade in result["trade_history"]:
                assert isinstance(trade["timestamp"], str)
            
            for point in result["equity_curve"]:
                assert isinstance(point["timestamp"], str)
                assert isinstance(point["value"], (int, float))
                
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results service method not implemented yet")

    @pytest.mark.asyncio
    @patch('src.dashboard.service.get_mode_manager')
    async def test_get_backtest_results_no_analysis_mode(self, mock_get_mode_manager, dashboard_service):
        """Test getting backtest results when no analysis mode is active"""
        # Mock mode manager to return None (no analysis mode active)
        mock_mode_manager = MagicMock()
        mock_mode_manager.get_mode_by_type.return_value = None
        mock_get_mode_manager.return_value = mock_mode_manager
        
        try:
            result = await dashboard_service.get_backtest_results()
            
            assert result is None
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results service method not implemented yet")

    @pytest.mark.asyncio
    @patch('src.dashboard.service.get_mode_manager')
    async def test_get_backtest_results_no_backtest_engine(self, mock_get_mode_manager, dashboard_service, mock_analysis_mode):
        """Test getting backtest results when analysis mode has no backtest engine"""
        # Mock analysis mode without backtest engine
        mock_analysis_mode.backtest_engine = None
        
        mock_mode_manager = MagicMock()
        mock_mode_manager.get_mode_by_type.return_value = mock_analysis_mode
        mock_get_mode_manager.return_value = mock_mode_manager
        
        try:
            result = await dashboard_service.get_backtest_results()
            
            assert result is None
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results service method not implemented yet")

    @pytest.mark.asyncio 
    @patch('src.dashboard.service.get_mode_manager')
    async def test_get_backtest_results_with_strategy_filter(self, mock_get_mode_manager, dashboard_service, mock_analysis_mode, sample_raw_backtest_data):
        """Test filtering backtest results by strategy name"""
        mock_mode_manager = MagicMock()
        mock_mode_manager.get_mode_by_type.return_value = mock_analysis_mode
        mock_get_mode_manager.return_value = mock_mode_manager
        
        # Mock backtest engine to return filtered results
        mock_analysis_mode.backtest_engine.get_results_by_strategy.return_value = sample_raw_backtest_data
        
        try:
            result = await dashboard_service.get_backtest_results(strategy_name="LSTM_DQN_Strategy")
            
            # Verify the filter was applied
            mock_analysis_mode.backtest_engine.get_results_by_strategy.assert_called_once_with("LSTM_DQN_Strategy")
            assert result["strategy_name"] == "LSTM_DQN_Strategy"
            
        except AttributeError:
            # Expected to fail until we implement the method
            pytest.fail("get_backtest_results service method with filters not implemented yet")