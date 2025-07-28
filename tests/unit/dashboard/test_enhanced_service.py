"""
Tests for enhanced dashboard service methods (Phase 4.3)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.dashboard.service import DashboardService
from src.dashboard.base import DashboardData, SystemStatus, TradingMode, Position
from src.xai.trading_integration import TradingExplanationManager, TradingExplanation
from src.xai.data_models import ExplanationData


class TestEnhancedDashboardService:
    """Test suite for Enhanced Dashboard Service methods (Phase 4.3)"""
    
    @pytest.fixture
    def dashboard_service(self):
        """Dashboard service instance"""
        service = DashboardService()
        # Mock the config to avoid initialization issues
        service.config = MagicMock()
        return service
    
    @pytest.fixture
    def mock_xai_manager(self):
        """Mock XAI explanation manager"""
        manager = MagicMock(spec=TradingExplanationManager)
        return manager
    
    @pytest.fixture
    def mock_explanation_data(self):
        """Mock explanation data"""
        return ExplanationData(
            feature_importance={
                "price_momentum": 0.4,
                "volume_spike": 0.3,
                "rsi_oversold": 0.2,
                "support_level": 0.1
            },
            explanation_type="permutation",
            instance_data=[1.0, 2.0, 3.0, 4.0],
            model_prediction=0.85,
            confidence_score=0.92
        )
    
    @pytest.fixture
    def mock_trading_explanation(self, mock_explanation_data):
        """Mock trading explanation"""
        return TradingExplanation(
            decision_id="decision-123",
            timestamp=datetime.utcnow().isoformat() + 'Z',
            decision_type="buy",
            symbol="ETH/USDC",
            explanation_data=mock_explanation_data,
            model_type="ml_model",
            confidence=0.92,
            metadata={"strategy": "momentum", "confidence_threshold": 0.8}
        )
    
    # =============================================================================
    # XAI EXPLANATION SERVICE TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_xai_explanations_success(self, dashboard_service, mock_xai_manager, mock_trading_explanation):
        """Test successful XAI explanations retrieval"""
        # Setup
        dashboard_service.xai_explanation_manager = mock_xai_manager
        mock_xai_manager.get_recent_explanations.return_value = [mock_trading_explanation]
        mock_xai_manager.to_dict.return_value = {
            "decision_id": mock_trading_explanation.decision_id,
            "symbol": mock_trading_explanation.symbol,
            "decision_type": mock_trading_explanation.decision_type
        }
        
        # Execute
        result = await dashboard_service.get_xai_explanations(
            symbol="ETH/USDC",
            decision_type="buy",
            limit=50
        )
        
        # Assertions
        assert len(result) == 1
        assert result[0]["decision_id"] == "decision-123"
        mock_xai_manager.get_recent_explanations.assert_called_once_with(
            symbol="ETH/USDC",
            decision_type="buy",
            limit=50
        )
        mock_xai_manager.to_dict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_xai_explanations_no_manager(self, dashboard_service):
        """Test XAI explanations retrieval when manager not initialized"""
        # Setup - no XAI manager
        dashboard_service.xai_explanation_manager = None
        
        # Execute
        result = await dashboard_service.get_xai_explanations()
        
        # Assertions
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_xai_explanation_by_id(self, dashboard_service, mock_xai_manager, mock_trading_explanation):
        """Test single XAI explanation retrieval by ID"""
        # Setup
        dashboard_service.xai_explanation_manager = mock_xai_manager
        mock_xai_manager.get_explanation.return_value = mock_trading_explanation
        mock_xai_manager.to_dict.return_value = {"decision_id": "decision-123"}
        
        # Execute
        result = await dashboard_service.get_xai_explanation("decision-123")
        
        # Assertions
        assert result["decision_id"] == "decision-123"
        mock_xai_manager.get_explanation.assert_called_once_with("decision-123")
    
    @pytest.mark.asyncio
    async def test_get_xai_explanation_not_found(self, dashboard_service, mock_xai_manager):
        """Test XAI explanation retrieval when not found"""
        # Setup
        dashboard_service.xai_explanation_manager = mock_xai_manager
        mock_xai_manager.get_explanation.return_value = None
        
        # Execute
        result = await dashboard_service.get_xai_explanation("nonexistent-id")
        
        # Assertions
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_feature_importance_summary(self, dashboard_service, mock_xai_manager):
        """Test feature importance summary retrieval"""
        # Setup
        dashboard_service.xai_explanation_manager = mock_xai_manager
        expected_importance = {
            "price_momentum": 0.35,
            "volume_ratio": 0.25,
            "rsi": 0.20,
            "ma_crossover": 0.15,
            "volatility": 0.05
        }
        mock_xai_manager.get_feature_importance_summary.return_value = expected_importance
        
        # Execute
        result = await dashboard_service.get_feature_importance_summary(
            symbol="BTC/USDC",
            hours_back=24
        )
        
        # Assertions
        assert result == expected_importance
        mock_xai_manager.get_feature_importance_summary.assert_called_once_with(
            symbol="BTC/USDC",
            hours_back=24
        )
    
    @pytest.mark.asyncio
    async def test_get_xai_cache_stats(self, dashboard_service, mock_xai_manager):
        """Test XAI cache statistics retrieval"""
        # Setup
        dashboard_service.xai_explanation_manager = mock_xai_manager
        expected_stats = {
            "explanation_cache_size": 150,
            "explainer_cache_size": 5,
            "cache_limit": 1000,
            "enabled": True
        }
        mock_xai_manager.get_cache_stats.return_value = expected_stats
        
        # Execute
        result = await dashboard_service.get_xai_cache_stats()
        
        # Assertions
        assert result == expected_stats
        mock_xai_manager.get_cache_stats.assert_called_once()
    
    # =============================================================================
    # INTERACTIVE CHARTING DATA TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_price_chart_data(self, dashboard_service):
        """Test price chart data generation"""
        # Execute
        result = await dashboard_service.get_price_chart_data(
            symbol="BTC/USDC",
            timeframe="1h",
            limit=5
        )
        
        # Assertions
        assert len(result) == 5
        for candle in result:
            assert "timestamp" in candle
            assert "open" in candle
            assert "high" in candle
            assert "low" in candle
            assert "close" in candle
            assert "volume" in candle
            assert candle["high"] >= candle["low"]
            assert candle["volume"] > 0
    
    @pytest.mark.asyncio
    async def test_get_performance_chart_data(self, dashboard_service):
        """Test performance chart data generation"""
        # Execute
        result = await dashboard_service.get_performance_chart_data(
            timeframe="1d",
            days_back=3
        )
        
        # Assertions
        assert len(result) == 3
        for point in result:
            assert "timestamp" in point
            assert "portfolio_value" in point
            assert "daily_return" in point
            assert "cumulative_return" in point
            assert point["portfolio_value"] > 0
    
    @pytest.mark.asyncio
    async def test_get_trading_volume_chart_data(self, dashboard_service):
        """Test trading volume chart data generation"""
        # Execute
        result = await dashboard_service.get_trading_volume_chart_data(
            timeframe="1h",
            hours_back=2
        )
        
        # Assertions
        assert len(result) == 2
        for point in result:
            assert "timestamp" in point
            assert "volume_usd" in point
            assert "trade_count" in point
            assert "avg_trade_size" in point
            assert point["volume_usd"] > 0
            assert point["trade_count"] > 0
    
    # =============================================================================
    # PERFORMANCE ATTRIBUTION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_performance_attribution(self, dashboard_service):
        """Test performance attribution analysis"""
        # Execute
        result = await dashboard_service.get_performance_attribution(
            timeframe="1d",
            days_back=30
        )
        
        # Assertions
        assert "timeframe" in result
        assert "days_analyzed" in result
        assert "strategy_attribution" in result
        assert "total_return" in result
        assert "risk_adjusted_return" in result
        assert result["timeframe"] == "1d"
        assert result["days_analyzed"] == 30
        assert isinstance(result["strategy_attribution"], dict)
        
        # Check strategy attribution structure
        for strategy, metrics in result["strategy_attribution"].items():
            assert "return_contribution" in metrics
            assert "risk_contribution" in metrics
            assert "sharpe_ratio" in metrics
            assert "trade_count" in metrics
            assert "win_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, dashboard_service):
        """Test risk metrics calculation"""
        # Execute
        result = await dashboard_service.get_risk_metrics(
            timeframe="1d",
            days_back=30
        )
        
        # Assertions
        assert "timeframe" in result
        assert "days_analyzed" in result
        assert "value_at_risk" in result
        assert "volatility_metrics" in result
        assert "drawdown_metrics" in result
        assert "correlation_metrics" in result
        
        # Check VaR structure
        var_metrics = result["value_at_risk"]
        assert "var_95" in var_metrics
        assert "var_99" in var_metrics
        assert "cvar_95" in var_metrics
        assert var_metrics["var_99"] >= var_metrics["var_95"]
        
        # Check volatility structure
        vol_metrics = result["volatility_metrics"]
        assert "daily_volatility" in vol_metrics
        assert "annualized_volatility" in vol_metrics
        assert vol_metrics["annualized_volatility"] > vol_metrics["daily_volatility"]
    
    # =============================================================================
    # MANUAL OVERRIDE CONTROL TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.activity_logger')
    @patch('src.dashboard.service.websocket_manager')
    async def test_execute_manual_trade(self, mock_websocket, mock_activity_logger, dashboard_service):
        """Test manual trade execution"""
        # Setup mocks
        mock_activity_logger.log_activity = AsyncMock()
        mock_websocket.send_system_alert = AsyncMock()
        
        # Execute
        result = await dashboard_service.execute_manual_trade(
            symbol="ETH/USDC",
            side="buy",
            amount=1.0,
            order_type="market",
            price=None,
            user_id="user-12345"
        )
        
        # Assertions
        assert result is True
        mock_activity_logger.log_activity.assert_called_once()
        mock_websocket.send_system_alert.assert_called_once()
        
        # Check activity log call
        activity_call = mock_activity_logger.log_activity.call_args
        assert activity_call[1]["event_type"] == "manual_trade_executed"
        assert activity_call[1]["metadata"]["symbol"] == "ETH/USDC"
        assert activity_call[1]["metadata"]["side"] == "buy"
        assert activity_call[1]["metadata"]["amount"] == 1.0
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.activity_logger')
    @patch('src.dashboard.service.websocket_manager')
    async def test_override_trading_signal(self, mock_websocket, mock_activity_logger, dashboard_service):
        """Test trading signal override"""
        # Setup mocks
        mock_activity_logger.log_activity = AsyncMock()
        mock_websocket.send_system_alert = AsyncMock()
        
        # Execute
        result = await dashboard_service.override_trading_signal(
            signal_id="signal-123",
            action="pause",
            reason="Market conditions changed",
            user_id="user-12345"
        )
        
        # Assertions
        assert result is True
        mock_activity_logger.log_activity.assert_called_once()
        mock_websocket.send_system_alert.assert_called_once()
        
        # Check activity log call
        activity_call = mock_activity_logger.log_activity.call_args
        assert activity_call[1]["event_type"] == "signal_pause"
        assert activity_call[1]["metadata"]["signal_id"] == "signal-123"
        assert activity_call[1]["metadata"]["reason"] == "Market conditions changed"
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.activity_logger')
    @patch('src.dashboard.service.websocket_manager')
    async def test_pause_trading_strategy(self, mock_websocket, mock_activity_logger, dashboard_service):
        """Test trading strategy pause"""
        # Setup mocks
        mock_activity_logger.log_activity = AsyncMock()
        mock_websocket.send_system_alert = AsyncMock()
        
        # Execute
        result = await dashboard_service.pause_trading_strategy(
            strategy_name="momentum_strategy",
            duration_minutes=60,
            reason="High volatility detected",
            user_id="user-12345"
        )
        
        # Assertions
        assert result is True
        mock_activity_logger.log_activity.assert_called_once()
        mock_websocket.send_system_alert.assert_called_once()
        
        # Check activity log call
        activity_call = mock_activity_logger.log_activity.call_args
        assert activity_call[1]["event_type"] == "strategy_paused"
        assert activity_call[1]["metadata"]["strategy_name"] == "momentum_strategy"
        assert activity_call[1]["metadata"]["duration_minutes"] == 60
    
    # =============================================================================
    # REAL-TIME METRICS TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_realtime_metrics(self, dashboard_service):
        """Test real-time metrics compilation"""
        # Mock get_dashboard_data
        dashboard_service.get_dashboard_data = AsyncMock()
        mock_dashboard_data = MagicMock()
        
        # Setup mock system metrics
        mock_dashboard_data.system_metrics.cpu_usage_pct = 45.2
        mock_dashboard_data.system_metrics.memory_usage_pct = 67.8
        mock_dashboard_data.system_metrics.active_connections = 12
        mock_dashboard_data.system_metrics.requests_per_minute = 45.0
        mock_dashboard_data.system_metrics.error_rate_pct = 0.5
        mock_dashboard_data.system_metrics.response_time_ms = 120.0
        
        # Setup mock trading status
        mock_dashboard_data.trading_status.mode.value = "simulation"
        mock_dashboard_data.trading_status.is_trading_active = True
        mock_dashboard_data.trading_status.trades_today = 8
        mock_dashboard_data.trading_status.volume_today_usd = Decimal("2500.0")
        mock_dashboard_data.trading_status.win_rate_pct = 65.5
        mock_dashboard_data.trading_status.high_confidence_signals = 5
        
        # Setup mock portfolio status
        mock_dashboard_data.portfolio_status.total_value_usd = Decimal("10250.0")
        mock_dashboard_data.portfolio_status.daily_pnl_usd = Decimal("75.25")
        mock_dashboard_data.portfolio_status.daily_pnl_pct = Decimal("0.75")
        mock_dashboard_data.portfolio_status.position_count = 3
        mock_dashboard_data.portfolio_status.available_balance_usd = Decimal("8500.0")
        
        # Setup mock ML/RL status
        mock_dashboard_data.ml_rl_status.ml_prediction_accuracy_pct = 77.5
        mock_dashboard_data.ml_rl_status.rl_action_success_rate_pct = 64.2
        mock_dashboard_data.ml_rl_status.ml_rl_integration_active = True
        mock_dashboard_data.ml_rl_status.ml_rl_decision_latency_ms = 9.8
        
        dashboard_service.get_dashboard_data.return_value = mock_dashboard_data
        
        # Execute
        result = await dashboard_service.get_realtime_metrics()
        
        # Assertions
        assert "system" in result
        assert "trading" in result
        assert "portfolio" in result
        assert "ml_rl" in result
        
        # Check system metrics
        system_metrics = result["system"]
        assert system_metrics["cpu_usage"] == 45.2
        assert system_metrics["memory_usage"] == 67.8
        assert system_metrics["active_connections"] == 12
        
        # Check trading metrics
        trading_metrics = result["trading"]
        assert trading_metrics["mode"] == "simulation"
        assert trading_metrics["is_active"] is True
        assert trading_metrics["trades_today"] == 8
        
        # Check portfolio metrics
        portfolio_metrics = result["portfolio"]
        assert portfolio_metrics["total_value"] == 10250.0
        assert portfolio_metrics["daily_pnl"] == 75.25
        
        # Check ML/RL metrics
        ml_rl_metrics = result["ml_rl"]
        assert ml_rl_metrics["ml_prediction_accuracy"] == 77.5
        assert ml_rl_metrics["integration_active"] is True
    
    @pytest.mark.asyncio
    async def test_get_live_positions(self, dashboard_service):
        """Test live positions retrieval"""
        # Mock get_dashboard_data
        dashboard_service.get_dashboard_data = AsyncMock()
        mock_dashboard_data = MagicMock()
        
        # Create mock positions
        mock_position = Position(
            symbol="BTC/USDC",
            chain="solana",
            side="long",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            unrealized_pnl=Decimal("500.0"),
            unrealized_pnl_pct=Decimal("2.0"),
            margin_used=Decimal("2500.0"),
            leverage=4.0,
            entry_time=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            mode=TradingMode.SIMULATION,
            is_simulated=True
        )
        
        mock_dashboard_data.portfolio_status.active_positions = [mock_position]
        dashboard_service.get_dashboard_data.return_value = mock_dashboard_data
        
        # Execute
        result = await dashboard_service.get_live_positions()
        
        # Assertions
        assert len(result) == 1
        position = result[0]
        assert position.symbol == "BTC/USDC"
        assert position.side == "long"
        assert position.size == Decimal("0.5")
        assert position.is_simulated is True
    
    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_xai_explanations_error_handling(self, dashboard_service, mock_xai_manager):
        """Test XAI explanations error handling"""
        # Setup - manager raises exception
        dashboard_service.xai_explanation_manager = mock_xai_manager
        mock_xai_manager.get_recent_explanations.side_effect = Exception("XAI error")
        
        # Execute
        result = await dashboard_service.get_xai_explanations()
        
        # Assertions
        assert result == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_manual_trade_error_handling(self, dashboard_service):
        """Test manual trade error handling"""
        # Mock activity_logger to raise exception
        with patch('src.dashboard.service.activity_logger') as mock_activity_logger:
            mock_activity_logger.log_activity = AsyncMock(side_effect=Exception("Logging error"))
            
            # Execute
            result = await dashboard_service.execute_manual_trade(
                symbol="BTC/USDC",
                side="buy",
                amount=1.0,
                user_id="user-123"
            )
            
            # Assertions
            assert result is False  # Should return False on error
    
    @pytest.mark.asyncio
    async def test_chart_data_error_handling(self, dashboard_service):
        """Test chart data error handling"""
        # Mock datetime to raise exception
        with patch('src.dashboard.service.datetime') as mock_datetime:
            mock_datetime.utcnow.side_effect = Exception("Time error")
            
            # Execute
            result = await dashboard_service.get_price_chart_data("BTC/USDC")
            
            # Assertions
            assert result == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_realtime_metrics_error_handling(self, dashboard_service):
        """Test real-time metrics error handling"""
        # Mock get_dashboard_data to raise exception
        dashboard_service.get_dashboard_data = AsyncMock(side_effect=Exception("Dashboard error"))
        
        # Execute
        result = await dashboard_service.get_realtime_metrics()
        
        # Assertions
        assert result == {}  # Should return empty dict on error


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestEnhancedDashboardServicePerformance:
    """Performance tests for enhanced dashboard service"""
    
    @pytest.fixture
    def dashboard_service(self):
        """Dashboard service instance with mocked components"""
        service = DashboardService()
        service.config = MagicMock()
        service.xai_explanation_manager = MagicMock()
        return service
    
    @pytest.mark.asyncio
    async def test_chart_data_generation_performance(self, dashboard_service):
        """Test chart data generation performance with large datasets"""
        # Execute with large limit
        start_time = asyncio.get_event_loop().time()
        result = await dashboard_service.get_price_chart_data(
            symbol="BTC/USDC",
            timeframe="1m",
            limit=1000
        )
        end_time = asyncio.get_event_loop().time()
        
        # Assertions
        assert len(result) == 1000
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_concurrent_xai_requests(self, dashboard_service):
        """Test concurrent XAI explanation requests"""
        # Setup
        dashboard_service.xai_explanation_manager.get_recent_explanations.return_value = []
        dashboard_service.xai_explanation_manager.to_dict.return_value = {}
        
        # Execute concurrent requests
        tasks = [
            dashboard_service.get_xai_explanations(limit=100)
            for _ in range(10)
        ]
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Assertions
        assert len(results) == 10
        assert all(isinstance(result, list) for result in results)
        assert (end_time - start_time) < 2.0  # Should handle concurrency efficiently