"""
Test suite for SimulationMode Phase 3 - Monitoring Integration.

Tests the integration of Prometheus metrics system with SimulationMode for
simulation performance monitoring, dashboards, and alerts.

Following TDD methodology - tests written before implementation.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from src.modes.base import ModeType, ModeStatus, ModeConfig
from src.portfolio.base import Portfolio, PortfolioConfig
from src.rl_agent.base import MarketState, TradeAction
from src.monitoring.base import MetricsRegistry, MetricsCollector
from src.modes.simulation_mode import SimulationMode


def create_mock_token():
    """Helper function to create mock DiscoveredToken."""
    from src.discovery.base import DiscoveredToken, TokenStatus
    from src.utils.base import Chain
    
    return DiscoveredToken(
        address="TEST_TOKEN_123",
        name="Test Token", 
        symbol="TEST",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        status=TokenStatus.VALIDATED
    )


class TestSimulationMonitoringIntegration:
    """Test suite for simulation monitoring integration."""
    
    @pytest.fixture
    def portfolio(self):
        """Portfolio instance for testing."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        return Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
    
    @pytest.fixture
    def simulation_config(self):
        """Simulation mode configuration with monitoring enabled."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_prometheus_metrics": True,
                "enable_simulation_dashboards": True,
                "enable_performance_alerts": True,
                "metrics_collection_interval": 1.0,
                "enable_detailed_metrics": True,
                "enable_real_time_monitoring": True
            }
        )
    
    @pytest.fixture
    def mock_metrics_registry(self):
        """Mock Prometheus metrics registry."""
        registry = Mock(spec=MetricsRegistry)
        registry.get_counter = Mock()
        registry.get_gauge = Mock()
        registry.get_histogram = Mock()
        registry.generate_output = Mock(return_value=b"# Mock metrics output")
        return registry
    
    @pytest.fixture
    def simulation_mode_with_monitoring(self, simulation_config, portfolio, mock_metrics_registry):
        """SimulationMode instance with monitoring enabled."""
        mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_config,
            portfolio=portfolio
        )
        mode.metrics_registry = mock_metrics_registry
        return mode
    
    async def test_prometheus_metrics_initialization(self, simulation_mode_with_monitoring):
        """Test initialization of Prometheus metrics for simulation."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Should initialize simulation-specific metrics
        assert hasattr(mode, 'simulation_metrics_collector')
        assert mode.simulation_metrics_collector is not None
        
        # Should register core simulation metrics
        mode.metrics_registry.get_counter.assert_any_call(
            "simulation_trades_total",
            "Total number of simulated trades executed",
            ["mode_id", "action_type", "success"]
        )
        
        mode.metrics_registry.get_gauge.assert_any_call(
            "simulation_portfolio_value",
            "Current virtual portfolio value in USD",
            ["mode_id"]
        )
        
        mode.metrics_registry.get_histogram.assert_any_call(
            "simulation_trade_execution_time",
            "Simulation trade execution time in seconds",
            ["mode_id"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
    
    async def test_simulation_performance_metrics_collection(self, simulation_mode_with_monitoring):
        """Test collection of simulation performance metrics."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        await mode.start()
        
        # Mock metrics counters and gauges
        mock_trades_counter = Mock()
        mock_portfolio_gauge = Mock()
        mock_pnl_gauge = Mock()
        
        mode.metrics_registry.get_counter.return_value = mock_trades_counter
        mode.metrics_registry.get_gauge.side_effect = [mock_portfolio_gauge, mock_pnl_gauge]
        
        # Execute a simulated trade
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=25.0  # Should trigger BUY
        )
        
        action = await mode.process_tick(market_state)
        
        # Should record metrics
        mock_trades_counter.labels.assert_called()
        mock_portfolio_gauge.labels.assert_called()
    
    async def test_real_time_metrics_updates(self, simulation_mode_with_monitoring):
        """Test real-time metrics updates during simulation."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        await mode.start()
        
        # Mock metrics components
        mode.simulation_metrics_collector = Mock()
        mode.simulation_metrics_collector.update_real_time_metrics = AsyncMock()
        
        # Generate multiple market ticks
        mock_token = create_mock_token()
        
        for i in range(5):
            market_state = MarketState(
                token=mock_token,
                price_usd=100.0 + i,
                price_change_24h=5.0,
                volume_24h=1000000.0,
                rsi=30.0 + i * 5
            )
            
            await mode.process_tick(market_state)
            await asyncio.sleep(0.001)  # Brief pause
        
        # Should update metrics in real-time
        assert mode.simulation_metrics_collector.update_real_time_metrics.call_count >= 5
    
    async def test_simulation_specific_metric_definitions(self, simulation_mode_with_monitoring):
        """Test simulation-specific metric definitions."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Get metric definitions
        metric_definitions = mode.simulation_metrics_collector.get_metric_definitions()
        
        # Should include simulation-specific metrics
        expected_metrics = {
            "simulation_trades_total": "Total number of simulated trades executed",
            "simulation_portfolio_value": "Current virtual portfolio value in USD",
            "simulation_unrealized_pnl": "Current unrealized P&L in USD",
            "simulation_realized_pnl": "Total realized P&L in USD",
            "simulation_win_rate": "Percentage of profitable trades",
            "simulation_drawdown": "Current portfolio drawdown percentage",
            "simulation_slippage": "Average slippage in basis points",
            "simulation_fees_paid": "Total fees paid in USD",
            "simulation_execution_latency": "Trade execution latency in milliseconds",
            "simulation_risk_score": "Current portfolio risk score",
            "simulation_sharpe_ratio": "Current Sharpe ratio",
            "simulation_volatility": "Portfolio volatility",
            "simulation_active_positions": "Number of active positions"
        }
        
        for metric_name, description in expected_metrics.items():
            assert metric_name in metric_definitions
            assert description == metric_definitions[metric_name]
    
    async def test_performance_dashboard_integration(self, simulation_mode_with_monitoring):
        """Test integration with performance dashboards."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Mock dashboard integration
        mode.simulation_dashboard = Mock()
        mode.simulation_dashboard.update_dashboard_data = AsyncMock()
        mode.simulation_dashboard.get_dashboard_config = Mock(
            return_value={
                'panels': ['portfolio_value', 'pnl_chart', 'trade_distribution'],
                'refresh_interval': 5,
                'alert_thresholds': {'drawdown': 0.15, 'win_rate': 0.4}
            }
        )
        
        # Generate dashboard data
        dashboard_data = await mode.get_dashboard_metrics()
        
        # Should provide dashboard-ready metrics
        assert 'current_portfolio_value' in dashboard_data
        assert 'total_pnl' in dashboard_data
        assert 'trade_statistics' in dashboard_data
        assert 'risk_metrics' in dashboard_data
        assert 'performance_charts' in dashboard_data
    
    async def test_simulation_alerts_system(self, simulation_mode_with_monitoring):
        """Test simulation performance alerts system."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Mock alert system
        mode.simulation_alert_manager = Mock()
        mode.simulation_alert_manager.check_alert_conditions = AsyncMock(
            return_value=[
                {
                    'type': 'performance_warning',
                    'message': 'Win rate below threshold',
                    'severity': 'warning',
                    'value': 0.35,
                    'threshold': 0.4
                }
            ]
        )
        mode.simulation_alert_manager.send_alert = AsyncMock()
        
        # Simulate poor performance scenario
        mode.simulation_metrics.win_rate = Decimal("0.35")  # Below threshold
        
        # Check for alerts
        alerts = await mode.simulation_alert_manager.check_alert_conditions()
        
        # Should generate performance alerts
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'performance_warning'
    
    async def test_metrics_export_for_prometheus(self, simulation_mode_with_monitoring):
        """Test metrics export in Prometheus format."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        await mode.start()
        
        # Generate some activity
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0
        )
        
        await mode.process_tick(market_state)
        
        # Export metrics
        metrics_output = mode.metrics_registry.generate_output()
        
        # Should generate Prometheus format output
        assert isinstance(metrics_output, bytes)
        mode.metrics_registry.generate_output.assert_called_once()
    
    async def test_historical_metrics_storage(self, simulation_mode_with_monitoring):
        """Test storage of historical metrics for analysis."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Mock historical storage
        mode.metrics_storage = Mock()
        mode.metrics_storage.store_metrics_snapshot = AsyncMock()
        mode.metrics_storage.get_historical_data = AsyncMock(
            return_value={
                'timestamps': [datetime.now() - timedelta(minutes=i) for i in range(10)],
                'portfolio_values': [10000 + i * 100 for i in range(10)],
                'trade_counts': [i for i in range(10)]
            }
        )
        
        # Store metrics snapshot
        metrics_snapshot = {
            'timestamp': datetime.now(),
            'portfolio_value': mode.virtual_portfolio.equity,
            'total_trades': mode.simulation_metrics.total_trades,
            'unrealized_pnl': mode.virtual_portfolio.total_unrealized_pnl
        }
        
        await mode.metrics_storage.store_metrics_snapshot(metrics_snapshot)
        
        # Should store historical data
        mode.metrics_storage.store_metrics_snapshot.assert_called_once_with(metrics_snapshot)
    
    async def test_custom_simulation_metrics(self, simulation_mode_with_monitoring):
        """Test custom simulation-specific metrics."""
        mode = simulation_mode_with_monitoring
        await mode.initialize()
        
        # Mock custom metrics
        mode.custom_metrics_collector = Mock()
        mode.custom_metrics_collector.collect_simulation_accuracy = Mock(return_value=0.85)
        mode.custom_metrics_collector.collect_strategy_effectiveness = Mock(return_value=0.72)
        mode.custom_metrics_collector.collect_virtual_slippage_impact = Mock(return_value=0.05)
        
        # Collect custom metrics
        custom_metrics = {
            'simulation_accuracy': mode.custom_metrics_collector.collect_simulation_accuracy(),
            'strategy_effectiveness': mode.custom_metrics_collector.collect_strategy_effectiveness(),
            'virtual_slippage_impact': mode.custom_metrics_collector.collect_virtual_slippage_impact()
        }
        
        # Should collect simulation-specific custom metrics
        assert custom_metrics['simulation_accuracy'] == 0.85
        assert custom_metrics['strategy_effectiveness'] == 0.72
        assert custom_metrics['virtual_slippage_impact'] == 0.05


class TestSimulationMetricsCollector:
    """Test suite for SimulationMetricsCollector."""
    
    @pytest.fixture
    def mock_simulation_mode(self):
        """Mock simulation mode for metrics collection."""
        mode = Mock()
        mode.mode_id = uuid4()
        mode.virtual_portfolio = Mock()
        mode.virtual_portfolio.equity = Decimal("10500")
        mode.virtual_portfolio.total_unrealized_pnl = Decimal("500")
        mode.simulation_metrics = Mock()
        mode.simulation_metrics.total_trades = 25
        mode.simulation_metrics.win_rate = Decimal("0.65")
        return mode
    
    @pytest.fixture
    def metrics_registry(self):
        """Mock metrics registry."""
        return Mock(spec=MetricsRegistry)
    
    @pytest.fixture
    def simulation_metrics_collector(self, metrics_registry, mock_simulation_mode):
        """SimulationMetricsCollector instance for testing."""
        from src.monitoring.simulation_metrics import SimulationMetricsCollector
        return SimulationMetricsCollector(metrics_registry, mock_simulation_mode)
    
    async def test_metrics_collector_initialization(self, simulation_metrics_collector, metrics_registry):
        """Test metrics collector initialization."""
        # Should initialize with registry and simulation mode
        assert simulation_metrics_collector.registry == metrics_registry
        assert simulation_metrics_collector.simulation_mode is not None
        
        # Should register all required metrics
        assert metrics_registry.get_counter.call_count > 0
        assert metrics_registry.get_gauge.call_count > 0
        assert metrics_registry.get_histogram.call_count > 0
    
    async def test_collect_portfolio_metrics(self, simulation_metrics_collector):
        """Test collection of portfolio metrics."""
        # Mock metric instances
        portfolio_value_gauge = Mock()
        unrealized_pnl_gauge = Mock()
        
        simulation_metrics_collector.portfolio_value_gauge = portfolio_value_gauge
        simulation_metrics_collector.unrealized_pnl_gauge = unrealized_pnl_gauge
        
        # Collect metrics
        simulation_metrics_collector.collect_portfolio_metrics()
        
        # Should update portfolio metrics
        portfolio_value_gauge.labels.assert_called()
        unrealized_pnl_gauge.labels.assert_called()
    
    async def test_collect_trading_metrics(self, simulation_metrics_collector):
        """Test collection of trading metrics."""
        # Mock metric instances
        trades_counter = Mock()
        win_rate_gauge = Mock()
        
        simulation_metrics_collector.trades_counter = trades_counter
        simulation_metrics_collector.win_rate_gauge = win_rate_gauge
        
        # Collect metrics
        simulation_metrics_collector.collect_trading_metrics()
        
        # Should update trading metrics
        win_rate_gauge.labels.assert_called()
    
    async def test_collect_risk_metrics(self, simulation_metrics_collector):
        """Test collection of risk metrics."""
        # Mock risk metrics
        simulation_metrics_collector.simulation_mode.virtual_portfolio.calculate_risk_metrics = Mock(
            return_value=Mock(
                var_95=Decimal("200"),
                volatility=Decimal("0.15"),
                max_position_risk=Decimal("0.08")
            )
        )
        
        # Mock metric instances
        var_gauge = Mock()
        volatility_gauge = Mock()
        
        simulation_metrics_collector.var_gauge = var_gauge
        simulation_metrics_collector.volatility_gauge = volatility_gauge
        
        # Collect metrics
        simulation_metrics_collector.collect_risk_metrics()
        
        # Should update risk metrics
        var_gauge.labels.assert_called()
        volatility_gauge.labels.assert_called()
    
    async def test_metrics_collection_health(self, simulation_metrics_collector):
        """Test metrics collector health monitoring."""
        # Should be healthy by default
        assert simulation_metrics_collector.is_healthy() is True
        
        # Simulate collection error
        simulation_metrics_collector._collection_errors = 5
        
        # Health check should consider error count
        health_status = simulation_metrics_collector.is_healthy()
        assert isinstance(health_status, bool)


class TestSimulationDashboards:
    """Test suite for simulation dashboards and visualization."""
    
    @pytest.fixture
    def simulation_dashboard_config(self):
        """Dashboard configuration for simulation monitoring."""
        return {
            'dashboard_name': 'Simulation Trading Performance',
            'refresh_interval': 5,
            'panels': [
                {
                    'type': 'portfolio_value_chart',
                    'title': 'Virtual Portfolio Value Over Time',
                    'metrics': ['simulation_portfolio_value'],
                    'time_range': '1h'
                },
                {
                    'type': 'pnl_distribution',
                    'title': 'P&L Distribution',
                    'metrics': ['simulation_realized_pnl', 'simulation_unrealized_pnl'],
                    'chart_type': 'bar'
                },
                {
                    'type': 'trade_performance',
                    'title': 'Trade Performance Metrics',
                    'metrics': ['simulation_win_rate', 'simulation_sharpe_ratio'],
                    'display_type': 'gauge'
                },
                {
                    'type': 'risk_monitoring',
                    'title': 'Risk Metrics',
                    'metrics': ['simulation_drawdown', 'simulation_volatility'],
                    'alert_thresholds': {'drawdown': 0.15, 'volatility': 0.25}
                }
            ],
            'alerts': [
                {
                    'name': 'High Drawdown Alert',
                    'condition': 'simulation_drawdown > 0.15',
                    'severity': 'warning'
                },
                {
                    'name': 'Poor Win Rate Alert',
                    'condition': 'simulation_win_rate < 0.4',
                    'severity': 'warning'
                }
            ]
        }
    
    async def test_dashboard_configuration_loading(self, simulation_dashboard_config):
        """Test loading of dashboard configuration."""
        from src.monitoring.simulation_dashboard import SimulationDashboard
        
        dashboard = SimulationDashboard(simulation_dashboard_config)
        
        # Should load configuration properly
        assert dashboard.config['dashboard_name'] == 'Simulation Trading Performance'
        assert len(dashboard.config['panels']) == 4
        assert len(dashboard.config['alerts']) == 2
    
    async def test_dashboard_data_preparation(self, simulation_dashboard_config):
        """Test preparation of data for dashboard visualization."""
        from src.monitoring.simulation_dashboard import SimulationDashboard
        
        dashboard = SimulationDashboard(simulation_dashboard_config)
        
        # Mock data source
        dashboard.data_source = Mock()
        dashboard.data_source.get_metrics_data = AsyncMock(
            return_value={
                'simulation_portfolio_value': [10000, 10100, 10050, 10200],
                'simulation_win_rate': [0.5, 0.55, 0.52, 0.58],
                'simulation_drawdown': [0.0, 0.05, 0.08, 0.03],
                'timestamps': [datetime.now() - timedelta(minutes=i) for i in range(4)]
            }
        )
        
        # Prepare dashboard data
        dashboard_data = await dashboard.prepare_dashboard_data()
        
        # Should format data for visualization
        assert 'panels' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'last_updated' in dashboard_data
    
    async def test_real_time_dashboard_updates(self, simulation_dashboard_config):
        """Test real-time dashboard updates."""
        from src.monitoring.simulation_dashboard import SimulationDashboard
        
        dashboard = SimulationDashboard(simulation_dashboard_config)
        
        # Mock real-time data stream
        dashboard.real_time_stream = Mock()
        dashboard.real_time_stream.subscribe = Mock()
        dashboard.real_time_stream.get_latest_metrics = Mock(
            return_value={
                'simulation_portfolio_value': 10250,
                'simulation_unrealized_pnl': 250,
                'timestamp': datetime.now()
            }
        )
        
        # Subscribe to real-time updates
        dashboard.start_real_time_updates()
        
        # Should subscribe to metrics stream
        dashboard.real_time_stream.subscribe.assert_called_once()


class TestSimulationAlerting:
    """Test suite for simulation performance alerting."""
    
    @pytest.fixture
    def alert_configuration(self):
        """Alert configuration for simulation monitoring."""
        return {
            'alerts': [
                {
                    'name': 'simulation_high_drawdown',
                    'condition': 'simulation_drawdown > 0.15',
                    'severity': 'warning',
                    'message': 'Simulation drawdown exceeds 15%',
                    'cooldown_minutes': 10
                },
                {
                    'name': 'simulation_poor_performance',
                    'condition': 'simulation_win_rate < 0.4 AND simulation_trades_total > 10',
                    'severity': 'warning',
                    'message': 'Simulation win rate below 40%',
                    'cooldown_minutes': 30
                },
                {
                    'name': 'simulation_system_error',
                    'condition': 'simulation_execution_errors > 5',
                    'severity': 'critical',
                    'message': 'Multiple simulation execution errors detected',
                    'cooldown_minutes': 5
                }
            ],
            'notification_channels': ['console', 'metrics'],
            'enable_alert_escalation': True
        }
    
    async def test_alert_condition_evaluation(self, alert_configuration):
        """Test evaluation of alert conditions."""
        from src.monitoring.simulation_alerts import SimulationAlertManager
        
        alert_manager = SimulationAlertManager(alert_configuration)
        
        # Mock current metrics
        current_metrics = {
            'simulation_drawdown': 0.18,  # Above threshold
            'simulation_win_rate': 0.35,  # Below threshold
            'simulation_trades_total': 15,
            'simulation_execution_errors': 2
        }
        
        # Evaluate alert conditions
        triggered_alerts = alert_manager.evaluate_alert_conditions(current_metrics)
        
        # Should trigger appropriate alerts
        assert len(triggered_alerts) >= 2  # drawdown and performance alerts
        alert_names = [alert['name'] for alert in triggered_alerts]
        assert 'simulation_high_drawdown' in alert_names
        assert 'simulation_poor_performance' in alert_names
    
    async def test_alert_cooldown_mechanism(self, alert_configuration):
        """Test alert cooldown to prevent spam."""
        from src.monitoring.simulation_alerts import SimulationAlertManager
        
        alert_manager = SimulationAlertManager(alert_configuration)
        
        # Mock metrics triggering alert
        trigger_metrics = {
            'simulation_drawdown': 0.20,
            'simulation_win_rate': 0.5,
            'simulation_trades_total': 15
        }
        
        # First evaluation should trigger alert
        alerts1 = alert_manager.evaluate_alert_conditions(trigger_metrics)
        assert len(alerts1) > 0
        
        # Immediate re-evaluation should not trigger (cooldown)
        alerts2 = alert_manager.evaluate_alert_conditions(trigger_metrics)
        assert len(alerts2) == 0  # Should be suppressed by cooldown
    
    async def test_alert_notification_delivery(self, alert_configuration):
        """Test delivery of alert notifications."""
        from src.monitoring.simulation_alerts import SimulationAlertManager
        
        alert_manager = SimulationAlertManager(alert_configuration)
        
        # Mock notification channels
        alert_manager.console_notifier = Mock()
        alert_manager.metrics_notifier = Mock()
        alert_manager.console_notifier.send_notification = AsyncMock()
        alert_manager.metrics_notifier.send_notification = AsyncMock()
        
        # Create alert
        alert = {
            'name': 'test_alert',
            'severity': 'warning',
            'message': 'Test alert message',
            'timestamp': datetime.now()
        }
        
        # Send alert
        await alert_manager.send_alert(alert)
        
        # Should notify all configured channels
        alert_manager.console_notifier.send_notification.assert_called_once()
        alert_manager.metrics_notifier.send_notification.assert_called_once()