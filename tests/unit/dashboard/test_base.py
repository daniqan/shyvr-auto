"""
Tests for dashboard base models and data structures
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from src.dashboard.base import (
    DashboardData, SystemMetrics, TradingStatus, PortfolioStatus, MLRLStatus,
    SystemStatus, TradingMode, Position, Trade, MLModel, RLAgent,
    DashboardDataCache, dashboard_cache, DashboardError
)


class TestSystemMetrics:
    """Test suite for SystemMetrics data structure"""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation with valid data"""
        now = datetime.utcnow()
        
        metrics = SystemMetrics(
            status=SystemStatus.HEALTHY,
            uptime_seconds=3600.0,
            cpu_usage_pct=25.5,
            memory_usage_mb=512.0,
            memory_usage_pct=45.0,
            active_connections=10,
            requests_per_minute=150.0,
            error_rate_pct=0.5,
            response_time_ms=25.0,
            last_updated=now,
            database_status=SystemStatus.HEALTHY,
            redis_status=SystemStatus.HEALTHY,
            ml_models_status=SystemStatus.HEALTHY,
            rl_agent_status=SystemStatus.HEALTHY,
            dex_connections_status=SystemStatus.HEALTHY,
            total_requests=1000,
            total_errors=5,
            cache_hit_rate_pct=85.0
        )
        
        assert metrics.status == SystemStatus.HEALTHY
        assert metrics.uptime_seconds == 3600.0
        assert metrics.cpu_usage_pct == 25.5
        assert metrics.memory_usage_mb == 512.0
        assert metrics.active_connections == 10
        assert metrics.total_requests == 1000
        assert metrics.cache_hit_rate_pct == 85.0
    
    def test_system_metrics_default_creation(self):
        """Test SystemMetrics default factory method"""
        metrics = SystemMetrics.create_default()
        
        assert metrics.status == SystemStatus.HEALTHY
        assert metrics.uptime_seconds == 0.0
        assert metrics.cpu_usage_pct == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.active_connections == 0
        assert metrics.total_requests == 0
        assert metrics.total_errors == 0
        assert isinstance(metrics.last_updated, datetime)


class TestPortfolioStatus:
    """Test suite for PortfolioStatus data structure"""
    
    def test_portfolio_status_creation(self):
        """Test PortfolioStatus creation with valid data"""
        positions = [
            Position(
                symbol="BTC/USDC",
                chain="solana",
                side="long",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("52000"),
                unrealized_pnl=Decimal("200"),
                unrealized_pnl_pct=Decimal("0.04"),
                margin_used=Decimal("5000"),
                leverage=1.0,
                entry_time=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
        ]
        
        trades = [
            Trade(
                id="trade-123",
                symbol="ETH/USDC",
                chain="ethereum",
                dex="uniswap_v3",
                side="buy",
                size=Decimal("1.0"),
                price=Decimal("3000"),
                value_usd=Decimal("3000"),
                fee=Decimal("9.0"),
                slippage_pct=Decimal("0.1"),
                execution_time_ms=150.0,
                timestamp=datetime.utcnow(),
                status="completed"
            )
        ]
        
        portfolio = PortfolioStatus(
            total_value_usd=Decimal("10000"),
            available_balance_usd=Decimal("5000"),
            margin_used_usd=Decimal("3000"),
            unrealized_pnl_usd=Decimal("200"),
            realized_pnl_usd=Decimal("500"),
            daily_pnl_usd=Decimal("100"),
            daily_pnl_pct=Decimal("1.0"),
            total_return_pct=Decimal("7.0"),
            max_drawdown_pct=Decimal("5.0"),
            current_drawdown_pct=Decimal("2.0"),
            sharpe_ratio=1.5,
            volatility_pct=15.0,
            var_95_usd=Decimal("200"),
            active_positions=positions,
            position_count=1,
            recent_trades=trades,
            chain_balances={"solana": Decimal("5000"), "ethereum": Decimal("5000")},
            last_updated=datetime.utcnow()
        )
        
        assert portfolio.total_value_usd == Decimal("10000")
        assert portfolio.position_count == 1
        assert len(portfolio.active_positions) == 1
        assert len(portfolio.recent_trades) == 1
        assert portfolio.chain_balances["solana"] == Decimal("5000")
        assert portfolio.sharpe_ratio == 1.5
    
    def test_portfolio_status_default_creation(self):
        """Test PortfolioStatus default factory method"""
        portfolio = PortfolioStatus.create_default()
        
        assert portfolio.total_value_usd == Decimal("0")
        assert portfolio.position_count == 0
        assert len(portfolio.active_positions) == 0
        assert len(portfolio.recent_trades) == 0
        assert len(portfolio.chain_balances) == 0


class TestTradingStatus:
    """Test suite for TradingStatus data structure"""
    
    def test_trading_status_creation(self):
        """Test TradingStatus creation with valid data"""
        trading = TradingStatus(
            mode=TradingMode.SIMULATION,
            is_trading_active=True,
            last_trade_time=datetime.utcnow() - timedelta(minutes=5),
            trades_today=15,
            volume_today_usd=Decimal("25000"),
            analysis_running=True,
            simulation_running=True,
            live_trading_enabled=False,
            emergency_stop_active=False,
            risk_limits_active=True,
            max_position_size_usd=Decimal("1000"),
            max_daily_loss_usd=Decimal("500"),
            win_rate_pct=65.5,
            avg_trade_duration_hours=2.5,
            avg_profit_per_trade_usd=Decimal("45.50"),
            tokens_analyzed_today=100,
            tokens_in_watchlist=25,
            high_confidence_signals=8,
            last_updated=datetime.utcnow()
        )
        
        assert trading.mode == TradingMode.SIMULATION
        assert trading.is_trading_active is True
        assert trading.trades_today == 15
        assert trading.volume_today_usd == Decimal("25000")
        assert trading.win_rate_pct == 65.5
        assert trading.tokens_analyzed_today == 100
    
    def test_trading_status_default_creation(self):
        """Test TradingStatus default factory method"""
        trading = TradingStatus.create_default()
        
        assert trading.mode == TradingMode.ANALYSIS
        assert trading.is_trading_active is False
        assert trading.trades_today == 0
        assert trading.emergency_stop_active is False
        assert trading.risk_limits_active is True


class TestMLRLStatus:
    """Test suite for MLRLStatus data structure"""
    
    def test_ml_rl_status_creation(self):
        """Test MLRLStatus creation with valid data"""
        ml_models = [
            MLModel(
                name="LSTM Price Predictor",
                type="LSTM",
                status=SystemStatus.HEALTHY,
                accuracy=0.78,
                last_training_time=datetime.utcnow() - timedelta(hours=2),
                predictions_today=150,
                avg_prediction_time_ms=1.2,
                model_size_mb=15.5,
                version="1.2.0"
            )
        ]
        
        rl_agents = [
            RLAgent(
                name="DQN Trading Agent",
                algorithm="DQN",
                status=SystemStatus.HEALTHY,
                episode=5000,
                epsilon=0.1,
                avg_reward=2.5,
                win_rate_pct=62.0,
                experience_buffer_size=50000,
                last_training_time=datetime.utcnow() - timedelta(minutes=30),
                actions_today=75,
                avg_decision_time_ms=8.5
            )
        ]
        
        mlrl = MLRLStatus(
            ml_models=ml_models,
            rl_agents=rl_agents,
            ml_rl_integration_active=True,
            ml_rl_decision_latency_ms=12.5,
            ml_confidence_threshold=0.7,
            rl_action_confidence=0.8,
            ml_training_active=False,
            rl_training_active=True,
            continuous_learning_active=True,
            ml_prediction_accuracy_pct=77.5,
            rl_action_success_rate_pct=64.2,
            ensemble_agreement_pct=81.0,
            last_updated=datetime.utcnow()
        )
        
        assert len(mlrl.ml_models) == 1
        assert len(mlrl.rl_agents) == 1
        assert mlrl.ml_rl_integration_active is True
        assert mlrl.ml_confidence_threshold == 0.7
        assert mlrl.ml_models[0].name == "LSTM Price Predictor"
        assert mlrl.rl_agents[0].algorithm == "DQN"
    
    def test_ml_rl_status_default_creation(self):
        """Test MLRLStatus default factory method"""
        mlrl = MLRLStatus.create_default()
        
        assert len(mlrl.ml_models) == 0
        assert len(mlrl.rl_agents) == 0
        assert mlrl.ml_rl_integration_active is True
        assert mlrl.continuous_learning_active is True


class TestDashboardData:
    """Test suite for DashboardData data structure"""
    
    def test_dashboard_data_creation(self):
        """Test DashboardData creation with all components"""
        system_metrics = SystemMetrics.create_default()
        portfolio_status = PortfolioStatus.create_default()
        trading_status = TradingStatus.create_default()
        ml_rl_status = MLRLStatus.create_default()
        
        dashboard_data = DashboardData(
            system_metrics=system_metrics,
            portfolio_status=portfolio_status,
            trading_status=trading_status,
            ml_rl_status=ml_rl_status,
            active_alerts=2,
            warnings_count=1,
            errors_count=0,
            recent_logs=["System started", "Dashboard initialized"],
            recent_notifications=["Welcome to Shyvr RLTE"],
            timestamp=datetime.utcnow()
        )
        
        assert dashboard_data.active_alerts == 2
        assert dashboard_data.warnings_count == 1
        assert dashboard_data.errors_count == 0
        assert len(dashboard_data.recent_logs) == 2
        assert dashboard_data.system_metrics.status == SystemStatus.HEALTHY
        assert dashboard_data.trading_status.mode == TradingMode.ANALYSIS
    
    def test_dashboard_data_default_creation(self):
        """Test DashboardData default factory method"""
        dashboard_data = DashboardData.create_default()
        
        assert dashboard_data.active_alerts == 0
        assert dashboard_data.warnings_count == 0
        assert dashboard_data.errors_count == 0
        assert len(dashboard_data.recent_logs) == 0
        assert isinstance(dashboard_data.timestamp, datetime)


class TestDashboardDataCache:
    """Test suite for DashboardDataCache"""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test cache initialization"""
        cache = DashboardDataCache()
        
        # Initial state
        last_update = await cache.get_last_update()
        assert last_update is None
        
        # Get data creates default data
        data = await cache.get_data()
        assert isinstance(data, DashboardData)
        assert data.active_alerts == 0
        
        # Should have update time now
        last_update = await cache.get_last_update()
        assert isinstance(last_update, datetime)
    
    @pytest.mark.asyncio
    async def test_cache_update_data(self):
        """Test updating cache with new data"""
        cache = DashboardDataCache()
        
        # Create custom data
        custom_data = DashboardData.create_default()
        custom_data.active_alerts = 5
        custom_data.warnings_count = 2
        
        # Update cache
        await cache.update_data(custom_data)
        
        # Retrieve data
        retrieved = await cache.get_data()
        assert retrieved.active_alerts == 5
        assert retrieved.warnings_count == 2
        
        # Should have update time
        last_update = await cache.get_last_update()
        assert isinstance(last_update, datetime)
    
    @pytest.mark.asyncio
    async def test_cache_partial_update(self):
        """Test partial cache updates"""
        cache = DashboardDataCache()
        
        # Get initial data
        initial_data = await cache.get_data()
        assert initial_data.active_alerts == 0
        
        # Partial update
        await cache.partial_update(active_alerts=3, errors_count=1)
        
        # Check updated data
        updated_data = await cache.get_data()
        assert updated_data.active_alerts == 3
        assert updated_data.errors_count == 1
        assert isinstance(updated_data.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        """Test concurrent cache access"""
        cache = DashboardDataCache()
        
        async def update_cache(alert_count):
            await cache.partial_update(active_alerts=alert_count)
        
        # Concurrent updates
        await asyncio.gather(
            update_cache(1),
            update_cache(2),
            update_cache(3)
        )
        
        # Final state should be consistent
        data = await cache.get_data()
        assert data.active_alerts in [1, 2, 3]  # Any of the updates could be final


class TestPosition:
    """Test suite for Position data structure"""
    
    def test_position_creation(self):
        """Test Position creation with valid data"""
        position = Position(
            symbol="BTC/USDC",
            chain="solana",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000"),
            unrealized_pnl=Decimal("200"),
            unrealized_pnl_pct=Decimal("0.04"),
            margin_used=Decimal("5000"),
            leverage=1.0,
            entry_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        assert position.symbol == "BTC/USDC"
        assert position.chain == "solana"
        assert position.side == "long"
        assert position.size == Decimal("0.1")
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("52000")
        assert position.unrealized_pnl == Decimal("200")
        assert position.leverage == 1.0


class TestTrade:
    """Test suite for Trade data structure"""
    
    def test_trade_creation(self):
        """Test Trade creation with valid data"""
        trade = Trade(
            id="trade-123",
            symbol="ETH/USDC",
            chain="ethereum",
            dex="uniswap_v3",
            side="buy",
            size=Decimal("1.0"),
            price=Decimal("3000"),
            value_usd=Decimal("3000"),
            fee=Decimal("9.0"),
            slippage_pct=Decimal("0.1"),
            execution_time_ms=150.0,
            timestamp=datetime.utcnow(),
            status="completed"
        )
        
        assert trade.id == "trade-123"
        assert trade.symbol == "ETH/USDC"
        assert trade.chain == "ethereum"
        assert trade.dex == "uniswap_v3"
        assert trade.side == "buy"
        assert trade.size == Decimal("1.0")
        assert trade.price == Decimal("3000")
        assert trade.value_usd == Decimal("3000")
        assert trade.status == "completed"


class TestMLModel:
    """Test suite for MLModel data structure"""
    
    def test_ml_model_creation(self):
        """Test MLModel creation with valid data"""
        model = MLModel(
            name="LSTM Price Predictor",
            type="LSTM",
            status=SystemStatus.HEALTHY,
            accuracy=0.78,
            last_training_time=datetime.utcnow() - timedelta(hours=2),
            predictions_today=150,
            avg_prediction_time_ms=1.2,
            model_size_mb=15.5,
            version="1.2.0"
        )
        
        assert model.name == "LSTM Price Predictor"
        assert model.type == "LSTM"
        assert model.status == SystemStatus.HEALTHY
        assert model.accuracy == 0.78
        assert model.predictions_today == 150
        assert model.avg_prediction_time_ms == 1.2
        assert model.model_size_mb == 15.5
        assert model.version == "1.2.0"


class TestRLAgent:
    """Test suite for RLAgent data structure"""
    
    def test_rl_agent_creation(self):
        """Test RLAgent creation with valid data"""
        agent = RLAgent(
            name="DQN Trading Agent",
            algorithm="DQN",
            status=SystemStatus.HEALTHY,
            episode=5000,
            epsilon=0.1,
            avg_reward=2.5,
            win_rate_pct=62.0,
            experience_buffer_size=50000,
            last_training_time=datetime.utcnow() - timedelta(minutes=30),
            actions_today=75,
            avg_decision_time_ms=8.5
        )
        
        assert agent.name == "DQN Trading Agent"
        assert agent.algorithm == "DQN"
        assert agent.status == SystemStatus.HEALTHY
        assert agent.episode == 5000
        assert agent.epsilon == 0.1
        assert agent.avg_reward == 2.5
        assert agent.win_rate_pct == 62.0
        assert agent.experience_buffer_size == 50000
        assert agent.actions_today == 75
        assert agent.avg_decision_time_ms == 8.5


class TestDashboardError:
    """Test suite for DashboardError exception"""
    
    def test_dashboard_error_creation(self):
        """Test DashboardError exception creation"""
        error = DashboardError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_dashboard_error_raising(self):
        """Test raising DashboardError exception"""
        with pytest.raises(DashboardError) as exc_info:
            raise DashboardError("Test error for raising")
        
        assert str(exc_info.value) == "Test error for raising"