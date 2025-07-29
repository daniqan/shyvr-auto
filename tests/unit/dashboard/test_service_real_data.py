"""
Tests for dashboard service real data integration
Following TDD approach - tests written first to define expected behavior
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from uuid import uuid4

from src.dashboard.service import DashboardService
from src.dashboard.base import (
    DashboardData, SystemMetrics, TradingStatus, PortfolioStatus, MLRLStatus,
    SystemStatus, TradingMode, Position, Trade, MLModel, RLAgent
)
from src.portfolio.base import Portfolio, PortfolioConfig
from src.modes.base import ModeType, ModeStatus


class TestDashboardServiceRealData:
    """Test suite for DashboardService real data integration"""
    
    @pytest.fixture
    def dashboard_service_instance(self):
        """Dashboard service instance for testing"""
        return DashboardService()
    
    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio with real data structure"""
        portfolio_config = PortfolioConfig(
            initial_balance=Decimal("10000.00"),
            base_currency="USDC"
        )
        return Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=portfolio_config,
            cash_balance=Decimal("8500.00"),
            total_value=Decimal("12500.00")
        )
    
    @pytest.fixture
    def mock_portfolio_manager(self, mock_portfolio):
        """Mock portfolio manager with real-like methods"""
        manager = MagicMock()
        manager.get_portfolio_summary = AsyncMock(return_value={
            'total_value': Decimal("12500.00"),
            'available_balance': Decimal("8500.00"),
            'margin_used': Decimal("1500.00"),
            'unrealized_pnl': Decimal("350.75"),
            'realized_pnl': Decimal("1250.25"),
            'daily_pnl': Decimal("125.50"),
            'daily_pnl_pct': Decimal("1.02"),
            'total_return_pct': Decimal("25.00"),
            'max_drawdown_pct': Decimal("3.5"),
            'current_drawdown_pct': Decimal("0.8"),
            'sharpe_ratio': 2.15,
            'volatility_pct': 12.8,
            'var_95': Decimal("225.00"),
            'position_count': 5,
            'chain_balances': {
                "solana": Decimal("6000.00"),
                "ethereum": Decimal("4000.00"),
                "base": Decimal("2500.00")
            }
        })
        manager.get_active_positions = AsyncMock(return_value=[
            Position(
                symbol="SOL/USDC",
                chain="solana",
                side="long",
                size=Decimal("10.5"),
                entry_price=Decimal("95.50"),
                current_price=Decimal("98.75"),
                unrealized_pnl=Decimal("34.13"),
                unrealized_pnl_pct=Decimal("3.4"),
                margin_used=Decimal("500.00"),
                leverage=2.0,
                entry_time=datetime.utcnow() - timedelta(hours=2),
                last_updated=datetime.utcnow()
            )
        ])
        manager.get_recent_trades = AsyncMock(return_value=[
            Trade(
                id=str(uuid4()),
                symbol="ETH/USDC",
                chain="ethereum",
                dex="uniswap_v3",
                side="buy",
                size=Decimal("0.5"),
                price=Decimal("2350.00"),
                value_usd=Decimal("1175.00"),
                fee=Decimal("3.50"),
                slippage_pct=Decimal("0.12"),
                execution_time_ms=250.0,
                timestamp=datetime.utcnow() - timedelta(minutes=30),
                status="completed"
            )
        ])
        return manager
    
    @pytest.fixture
    def mock_mode_manager(self):
        """Mock mode manager with real-like methods"""
        manager = MagicMock()
        
        # Mock active modes
        mock_mode = MagicMock()
        mock_mode.status.value = "active"
        mock_mode.get_status = MagicMock(return_value={
            'is_active': True,
            'last_activity': datetime.utcnow(),
            'trades_today': 12,
            'volume_today': Decimal("3500.00"),
            'win_rate': 68.5,
            'avg_trade_duration': 2.8,
            'avg_profit_per_trade': Decimal("52.30"),
            'tokens_analyzed': 185,
            'tokens_in_watchlist': 28,
            'high_confidence_signals': 7
        })
        
        manager.list_active_modes = MagicMock(return_value={
            uuid4(): mock_mode
        })
        manager._mode_type_registry = {ModeType.SIMULATION: uuid4()}
        manager.get_current_mode = MagicMock(return_value=TradingMode.SIMULATION)
        return manager
    
    @pytest.fixture
    def mock_model_manager(self):
        """Mock model manager with real-like methods"""
        manager = MagicMock()
        manager.get_model_status = AsyncMock(return_value=[
            {
                'name': 'LSTM Price Predictor',
                'type': 'LSTM',
                'status': 'healthy',
                'accuracy': 0.82,
                'last_training': datetime.utcnow() - timedelta(hours=1),
                'predictions_today': 324,
                'avg_prediction_time_ms': 0.8,
                'model_size_mb': 15.2,
                'version': '1.3.1'
            },
            {
                'name': 'Random Forest Classifier',
                'type': 'RandomForest',
                'status': 'healthy',
                'accuracy': 0.76,
                'last_training': datetime.utcnow() - timedelta(hours=3),
                'predictions_today': 298,
                'avg_prediction_time_ms': 1.2,
                'model_size_mb': 8.7,
                'version': '2.1.0'
            }
        ])
        manager.get_performance_metrics = AsyncMock(return_value={
            'ml_prediction_accuracy': 78.5,
            'ml_confidence_threshold': 0.72,
            'ml_training_active': False,
            'continuous_learning_active': True
        })
        return manager
    
    @pytest.fixture
    def mock_rl_agent(self):
        """Mock RL agent with real-like methods"""
        agent = MagicMock()
        agent.get_status = AsyncMock(return_value={
            'name': 'DQN Trading Agent',
            'algorithm': 'DQN',
            'status': 'healthy',
            'episode': 18942,
            'epsilon': 0.03,
            'avg_reward': 3.25,
            'win_rate': 71.2,
            'experience_buffer_size': 52000,
            'last_training': datetime.utcnow() - timedelta(minutes=45),
            'actions_today': 156,
            'avg_decision_time_ms': 6.8,
            'action_success_rate': 69.8
        })
        return agent
    
    @pytest.fixture
    def mock_activity_logger(self):
        """Mock activity logger with real database data"""
        logger = MagicMock()
        logger.get_recent_activities = AsyncMock(return_value=[
            {
                'id': 1,
                'timestamp': datetime.utcnow() - timedelta(minutes=5),
                'category': 'trading',
                'action': 'execute',
                'title': 'Trade executed: SOL/USDC buy',
                'severity': 'info',
                'metadata': {'symbol': 'SOL/USDC', 'side': 'buy'}
            },
            {
                'id': 2,
                'timestamp': datetime.utcnow() - timedelta(minutes=10),
                'category': 'system',
                'action': 'alert',
                'title': 'High volatility detected',
                'severity': 'warning',
                'metadata': {'volatility': 0.15}
            }
        ])
        logger.get_trading_history = AsyncMock(return_value=[
            {
                'trade_id': str(uuid4()),
                'symbol': 'ETH/USDC',
                'side': 'sell',
                'size': Decimal("0.8"),
                'price': Decimal("2380.00"),
                'timestamp': datetime.utcnow() - timedelta(hours=2),
                'realized_pnl': Decimal("45.60"),
                'fees': Decimal("2.30")
            }
        ])
        return logger

    # Tests for _get_portfolio_status() with real data
    @pytest.mark.asyncio
    async def test_get_portfolio_status_with_real_data(self, dashboard_service_instance, mock_portfolio_manager):
        """Test _get_portfolio_status() fetches real portfolio data"""
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        
        portfolio_status = await service._get_portfolio_status()
        
        # Verify real data was fetched
        mock_portfolio_manager.get_portfolio_summary.assert_called_once()
        mock_portfolio_manager.get_active_positions.assert_called_once()
        mock_portfolio_manager.get_recent_trades.assert_called_once()
        
        # Verify returned data structure
        assert isinstance(portfolio_status, PortfolioStatus)
        assert portfolio_status.total_value_usd == Decimal("12500.00")
        assert portfolio_status.available_balance_usd == Decimal("8500.00")
        assert portfolio_status.daily_pnl_usd == Decimal("125.50")
        assert portfolio_status.position_count == 5
        assert len(portfolio_status.chain_balances) == 3
        assert portfolio_status.chain_balances["solana"] == Decimal("6000.00")
    
    @pytest.mark.asyncio
    async def test_get_portfolio_status_fallback_when_no_manager(self, dashboard_service_instance):
        """Test _get_portfolio_status() falls back to default when no portfolio manager"""
        service = dashboard_service_instance
        service.portfolio_manager = None
        
        portfolio_status = await service._get_portfolio_status()
        
        # Should return default/fallback data
        assert isinstance(portfolio_status, PortfolioStatus)
        assert portfolio_status.total_value_usd > Decimal("0")
    
    @pytest.mark.asyncio
    async def test_get_portfolio_status_handles_manager_errors(self, dashboard_service_instance, mock_portfolio_manager):
        """Test _get_portfolio_status() handles portfolio manager errors gracefully"""
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        
        # Make portfolio manager raise an error
        mock_portfolio_manager.get_portfolio_summary.side_effect = Exception("Portfolio service error")
        
        portfolio_status = await service._get_portfolio_status()
        
        # Should return default data, not raise error
        assert isinstance(portfolio_status, PortfolioStatus)

    # Tests for _get_trading_status() with real data
    @pytest.mark.asyncio
    async def test_get_trading_status_with_real_data(self, dashboard_service_instance, mock_mode_manager):
        """Test _get_trading_status() fetches real trading status"""
        service = dashboard_service_instance
        service.mode_manager = mock_mode_manager
        
        trading_status = await service._get_trading_status()
        
        # Verify real data was fetched
        mock_mode_manager.list_active_modes.assert_called_once()
        
        # Verify returned data structure
        assert isinstance(trading_status, TradingStatus)
        assert trading_status.mode == TradingMode.SIMULATION
        assert trading_status.trades_today == 12
        assert trading_status.volume_today_usd == Decimal("3500.00")
        assert trading_status.win_rate_pct == 68.5
        assert trading_status.tokens_analyzed_today == 185
        assert trading_status.high_confidence_signals == 7
    
    @pytest.mark.asyncio
    async def test_get_trading_status_maps_mode_types_correctly(self, dashboard_service_instance):
        """Test _get_trading_status() correctly maps mode types"""
        service = dashboard_service_instance
        
        # Mock mode manager with different mode types
        mock_manager = MagicMock()
        mock_mode = MagicMock()
        mock_mode.status.value = "active"
        mock_mode.get_status = MagicMock(return_value={
            'trades_today': 5,
            'volume_today': Decimal("1500.00"),
            'win_rate': 55.0
        })
        
        # Test analysis mode
        mock_manager.list_active_modes = MagicMock(return_value={uuid4(): mock_mode})
        mock_manager._mode_type_registry = {ModeType.ANALYSIS: list(mock_manager.list_active_modes().keys())[0]}
        service.mode_manager = mock_manager
        
        trading_status = await service._get_trading_status()
        assert trading_status.mode == TradingMode.ANALYSIS
        assert trading_status.analysis_running is True
        assert trading_status.simulation_running is False
        assert trading_status.live_trading_enabled is False

    # Tests for _get_ml_rl_status() with real data
    @pytest.mark.asyncio
    async def test_get_ml_rl_status_with_real_data(self, dashboard_service_instance, mock_model_manager, mock_rl_agent):
        """Test _get_ml_rl_status() fetches real ML/RL status"""
        service = dashboard_service_instance
        service.model_manager = mock_model_manager
        service.rl_agent = mock_rl_agent
        
        ml_rl_status = await service._get_ml_rl_status()
        
        # Verify real data was fetched
        mock_model_manager.get_model_status.assert_called_once()
        mock_model_manager.get_performance_metrics.assert_called_once()
        mock_rl_agent.get_status.assert_called_once()
        
        # Verify returned data structure
        assert isinstance(ml_rl_status, MLRLStatus)
        assert len(ml_rl_status.ml_models) == 2
        assert len(ml_rl_status.rl_agents) == 1
        
        # Check ML model data
        lstm_model = ml_rl_status.ml_models[0]
        assert lstm_model.name == "LSTM Price Predictor"
        assert lstm_model.accuracy == 0.82
        assert lstm_model.predictions_today == 324
        
        # Check RL agent data
        rl_agent = ml_rl_status.rl_agents[0]
        assert rl_agent.name == "DQN Trading Agent"
        assert rl_agent.episode == 18942
        assert rl_agent.win_rate_pct == 71.2
        assert rl_agent.actions_today == 156
    
    @pytest.mark.asyncio
    async def test_get_ml_rl_status_handles_missing_components(self, dashboard_service_instance):
        """Test _get_ml_rl_status() handles missing ML/RL components"""
        service = dashboard_service_instance
        service.model_manager = None
        service.rl_agent = None
        
        ml_rl_status = await service._get_ml_rl_status()
        
        # Should return default data, not crash
        assert isinstance(ml_rl_status, MLRLStatus)
        assert ml_rl_status.ml_rl_integration_active is False

    # Tests for chart data methods with real data sources
    @pytest.mark.asyncio
    async def test_get_price_chart_data_with_real_source(self, dashboard_service_instance):
        """Test get_price_chart_data() uses real data sources"""
        service = dashboard_service_instance
        
        # Mock market data service
        mock_market_data = MagicMock()
        mock_market_data.get_historical_ohlcv = AsyncMock(return_value=[
            {
                'timestamp': datetime.utcnow() - timedelta(hours=2),
                'open': 98.50,
                'high': 99.75,
                'low': 97.80,
                'close': 99.25,
                'volume': 15420.50
            },
            {
                'timestamp': datetime.utcnow() - timedelta(hours=1),
                'open': 99.25,
                'high': 101.20,
                'low': 99.00,
                'close': 100.85,
                'volume': 18750.25
            }
        ])
        service.market_data_service = mock_market_data
        
        chart_data = await service.get_price_chart_data("SOL/USDC", "1h", 48)
        
        # Verify real data source was called
        mock_market_data.get_historical_ohlcv.assert_called_once_with("SOL/USDC", "1h", 48)
        
        # Verify data structure
        assert isinstance(chart_data, list)
        assert len(chart_data) == 2
        assert chart_data[0]['open'] == 98.50
        assert chart_data[1]['close'] == 100.85
    
    @pytest.mark.asyncio
    async def test_get_performance_chart_data_with_real_source(self, dashboard_service_instance, mock_portfolio_manager):
        """Test get_performance_chart_data() uses real portfolio data"""
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        
        # Mock historical performance data
        mock_portfolio_manager.get_historical_performance = AsyncMock(return_value=[
            {
                'timestamp': datetime.utcnow() - timedelta(days=2),
                'portfolio_value': Decimal("12000.00"),
                'daily_return': Decimal("0.015"),
                'cumulative_return': Decimal("0.20")
            },
            {
                'timestamp': datetime.utcnow() - timedelta(days=1),
                'portfolio_value': Decimal("12350.00"),
                'daily_return': Decimal("0.029"),
                'cumulative_return': Decimal("0.235")
            }
        ])
        
        performance_data = await service.get_performance_chart_data("1d", 30)
        
        # Verify real data source was called
        mock_portfolio_manager.get_historical_performance.assert_called_once_with(days=30)
        
        # Verify data structure
        assert isinstance(performance_data, list)
        assert len(performance_data) == 2
        assert performance_data[0]['portfolio_value'] == 12000.00
        assert performance_data[1]['daily_return'] == 2.9  # Converted to percentage

    # Tests for get_trading_history() with real database data
    @pytest.mark.asyncio
    async def test_get_trading_history_with_real_database(self, dashboard_service_instance, mock_activity_logger):
        """Test get_trading_history() fetches from activity logger database"""
        service = dashboard_service_instance
        service.activity_logger = mock_activity_logger
        
        trading_history = await service.get_trading_history(limit=50)
        
        # Verify real database was queried
        mock_activity_logger.get_trading_history.assert_called_once_with(limit=50)
        
        # Verify returned data structure
        assert isinstance(trading_history, list)
        assert len(trading_history) == 1  # Mock returns 1 trade
        
        trade = trading_history[0]
        assert 'trade_id' in trade
        assert 'symbol' in trade
        assert trade['symbol'] == 'ETH/USDC'
        assert trade['realized_pnl'] == Decimal("45.60")
    
    @pytest.mark.asyncio
    async def test_get_trading_history_handles_database_errors(self, dashboard_service_instance, mock_activity_logger):
        """Test get_trading_history() handles database errors gracefully"""
        service = dashboard_service_instance
        service.activity_logger = mock_activity_logger
        
        # Make activity logger raise database error
        mock_activity_logger.get_trading_history.side_effect = Exception("Database connection error")
        
        trading_history = await service.get_trading_history(limit=50)
        
        # Should return empty list, not raise error
        assert isinstance(trading_history, list)
        assert len(trading_history) == 0

    # Tests for integrated dashboard data with all real components
    @pytest.mark.asyncio
    async def test_get_dashboard_data_with_all_real_components(
        self, dashboard_service_instance, mock_portfolio_manager, mock_mode_manager, 
        mock_model_manager, mock_rl_agent, mock_activity_logger
    ):
        """Test get_dashboard_data() integrates all real data sources"""
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        service.mode_manager = mock_mode_manager
        service.model_manager = mock_model_manager
        service.rl_agent = mock_rl_agent
        service.activity_logger = mock_activity_logger
        
        # Mock recent activity for dashboard
        service._get_recent_activity = AsyncMock(return_value=[
            {
                'title': 'Trade executed: SOL/USDC buy',
                'severity': 'info'
            },
            {
                'title': 'High volatility detected',
                'severity': 'warning'
            },
            {
                'title': 'System startup complete',
                'severity': 'critical'
            }
        ])
        
        dashboard_data = await service.get_dashboard_data()
        
        # Verify all real data sources were called
        mock_portfolio_manager.get_portfolio_summary.assert_called_once()
        mock_mode_manager.list_active_modes.assert_called_once()
        mock_model_manager.get_model_status.assert_called_once()
        mock_rl_agent.get_status.assert_called_once()
        
        # Verify integrated data
        assert isinstance(dashboard_data, DashboardData)
        
        # Portfolio data
        assert dashboard_data.portfolio_status.total_value_usd == Decimal("12500.00")
        assert dashboard_data.portfolio_status.position_count == 5
        
        # Trading data
        assert dashboard_data.trading_status.trades_today == 12
        assert dashboard_data.trading_status.volume_today_usd == Decimal("3500.00")
        
        # ML/RL data
        assert len(dashboard_data.ml_rl_status.ml_models) == 2
        assert len(dashboard_data.ml_rl_status.rl_agents) == 1
        
        # Activity data
        assert dashboard_data.active_alerts == 1  # One critical
        assert dashboard_data.warnings_count == 1  # One warning
        assert len(dashboard_data.recent_logs) == 3

    # Tests for error handling and fallbacks
    @pytest.mark.asyncio
    async def test_dashboard_graceful_degradation(self, dashboard_service_instance):
        """Test dashboard service gracefully degrades when real components fail"""
        service = dashboard_service_instance
        
        # Set up components that will fail
        failing_portfolio_manager = MagicMock()
        failing_portfolio_manager.get_portfolio_summary.side_effect = Exception("Portfolio service down")
        service.portfolio_manager = failing_portfolio_manager
        
        failing_mode_manager = MagicMock()
        failing_mode_manager.list_active_modes.side_effect = Exception("Mode service down")
        service.mode_manager = failing_mode_manager
        
        # Should still return valid dashboard data
        dashboard_data = await service.get_dashboard_data()
        
        assert isinstance(dashboard_data, DashboardData)
        # Should have fallback/default values
        assert dashboard_data.portfolio_status.total_value_usd > Decimal("0")
        assert dashboard_data.trading_status.mode in [TradingMode.ANALYSIS, TradingMode.SIMULATION, TradingMode.LIVE]

    # Performance and concurrency tests
    @pytest.mark.asyncio
    async def test_concurrent_real_data_requests(self, dashboard_service_instance, mock_portfolio_manager, mock_mode_manager):
        """Test concurrent requests to real data sources don't cause issues"""
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        service.mode_manager = mock_mode_manager
        
        # Make concurrent requests
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(service._get_portfolio_status()))
            tasks.append(asyncio.create_task(service._get_trading_status()))
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        for i, result in enumerate(results):
            if i % 2 == 0:  # Portfolio status
                assert isinstance(result, PortfolioStatus)
            else:  # Trading status
                assert isinstance(result, TradingStatus)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.psutil')
    async def test_dashboard_data_caching_behavior(self, mock_psutil, dashboard_service_instance, mock_portfolio_manager):
        """Test dashboard data caching works with real data sources"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 35.0
        mock_memory = MagicMock()
        mock_memory.used = 768 * 1024 * 1024
        mock_memory.percent = 55.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        service = dashboard_service_instance
        service.portfolio_manager = mock_portfolio_manager
        
        # First request
        data1 = await service.get_dashboard_data()
        
        # Second request (should potentially use cached data for expensive operations)
        data2 = await service.get_dashboard_data()
        
        # Both should be valid
        assert isinstance(data1, DashboardData)
        assert isinstance(data2, DashboardData)
        
        # Portfolio manager should have been called for both (no caching implemented yet)
        assert mock_portfolio_manager.get_portfolio_summary.call_count >= 1


class TestDashboardServiceRealDataErrorHandling:
    """Test error handling for real data integration"""
    
    @pytest.fixture
    def dashboard_service_instance(self):
        """Dashboard service instance for testing"""
        return DashboardService()
    
    @pytest.mark.asyncio
    async def test_portfolio_manager_timeout_handling(self, dashboard_service_instance):
        """Test handling of portfolio manager timeouts"""
        service = dashboard_service_instance
        
        # Mock portfolio manager that times out
        timeout_manager = MagicMock()
        timeout_manager.get_portfolio_summary = AsyncMock(side_effect=asyncio.TimeoutError("Request timeout"))
        service.portfolio_manager = timeout_manager
        
        portfolio_status = await service._get_portfolio_status()
        
        # Should return default data, not raise error
        assert isinstance(portfolio_status, PortfolioStatus)
    
    @pytest.mark.asyncio
    async def test_model_manager_connection_error_handling(self, dashboard_service_instance):
        """Test handling of ML model manager connection errors"""
        service = dashboard_service_instance
        
        # Mock model manager with connection error
        error_manager = MagicMock()
        error_manager.get_model_status = AsyncMock(side_effect=ConnectionError("Cannot connect to ML service"))
        service.model_manager = error_manager
        
        ml_rl_status = await service._get_ml_rl_status()
        
        # Should return default data, not raise error
        assert isinstance(ml_rl_status, MLRLStatus)
    
    @pytest.mark.asyncio
    async def test_activity_logger_database_error_handling(self, dashboard_service_instance):
        """Test handling of activity logger database errors"""
        service = dashboard_service_instance
        
        # Mock activity logger with database error
        error_logger = MagicMock()
        error_logger.get_recent_activities = AsyncMock(side_effect=Exception("Database connection failed"))
        service.activity_logger = error_logger
        
        # Should not crash when getting recent activity
        recent_activity = await service._get_recent_activity()
        
        assert isinstance(recent_activity, list)
        assert len(recent_activity) == 0  # Empty on error


class TestDashboardServiceDataValidation:
    """Test data validation for real data integration"""
    
    @pytest.fixture
    def dashboard_service_instance(self):
        """Dashboard service instance for testing"""
        return DashboardService()
    
    @pytest.mark.asyncio
    async def test_portfolio_data_validation(self, dashboard_service_instance):
        """Test validation of portfolio data from real sources"""
        service = dashboard_service_instance
        
        # Mock portfolio manager with invalid data
        invalid_manager = MagicMock()
        invalid_manager.get_portfolio_summary = AsyncMock(return_value={
            'total_value': "invalid",  # Should be Decimal
            'available_balance': None,  # Should be Decimal
            'position_count': -1  # Should be non-negative
        })
        invalid_manager.get_active_positions = AsyncMock(return_value=[])
        invalid_manager.get_recent_trades = AsyncMock(return_value=[])
        service.portfolio_manager = invalid_manager
        
        portfolio_status = await service._get_portfolio_status()
        
        # Should handle invalid data gracefully
        assert isinstance(portfolio_status, PortfolioStatus)
        assert isinstance(portfolio_status.total_value_usd, Decimal)
        assert portfolio_status.position_count >= 0
    
    @pytest.mark.asyncio
    async def test_trading_data_validation(self, dashboard_service_instance):
        """Test validation of trading data from real sources"""
        service = dashboard_service_instance
        
        # Mock mode manager with invalid data
        invalid_manager = MagicMock()
        mock_mode = MagicMock()
        mock_mode.status.value = "invalid_status"
        mock_mode.get_status = MagicMock(return_value={
            'trades_today': "not_a_number",
            'win_rate': 150.0,  # > 100%
            'volume_today': None
        })
        invalid_manager.list_active_modes = MagicMock(return_value={uuid4(): mock_mode})
        invalid_manager._mode_type_registry = {}
        service.mode_manager = invalid_manager
        
        trading_status = await service._get_trading_status()
        
        # Should handle invalid data gracefully
        assert isinstance(trading_status, TradingStatus)
        assert isinstance(trading_status.trades_today, int)
        assert 0 <= trading_status.win_rate_pct <= 100
        assert isinstance(trading_status.volume_today_usd, Decimal)