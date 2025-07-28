"""
Integration tests for BacktestResultIntegrator with AnalysisMode

Following TDD methodology - tests define requirements for Phase 3 integration
between BacktestResultIntegrator and AnalysisMode with optional configuration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from src.modes.analysis_mode import AnalysisMode, BacktestResult
from src.modes.base import ModeConfig, ModeType
from src.modes.backtest_integration import (
    BacktestResultIntegrator,
    BacktestIntegrationConfig,
    ValidationDataPoint,
    IntegrationStatus
)
from src.portfolio.base import Portfolio
from src.rl_agent.base import TradeAction, MarketState
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestBacktestAnalysisModeIntegration:
    """Test integration between BacktestResultIntegrator and AnalysisMode"""
    
    @pytest.fixture
    def portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = uuid4()
        portfolio.cash_balance = Decimal("10000")
        portfolio.total_value = Decimal("10000")
        portfolio.open_positions = {}
        return portfolio
    
    @pytest.fixture
    def analysis_config_with_integration(self):
        """Create analysis mode configuration with backtest integration enabled."""
        return ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "comprehensive",
                "include_backtesting": True,
                "backtest_integration": {
                    "enabled": True,
                    "validation_enabled": True,
                    "min_backtest_trades": 5,
                    "validation_sample_ratio": 0.2
                }
            }
        )
    
    @pytest.fixture
    def analysis_config_without_integration(self):
        """Create analysis mode configuration with backtest integration disabled."""
        return ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "comprehensive",
                "include_backtesting": True,
                "backtest_integration": {
                    "enabled": False
                }
            }
        )
    
    @pytest.fixture
    def sample_backtest_result(self):
        """Create sample backtest result for testing."""
        trade_history = [
            {
                "timestamp": datetime.now() - timedelta(hours=i),
                "action": "BUY" if i % 2 == 0 else "SELL",
                "price": 100.0 + i,
                "return": 0.02 if i % 2 == 0 else -0.01,
                "volume": 1000.0
            }
            for i in range(10)
        ]
        
        return BacktestResult(
            strategy_name="TEST_STRATEGY",
            total_return=0.15,
            annual_return=0.18,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            win_rate=0.6,
            total_trades=10,
            profit_factor=1.5,
            trade_history=trade_history,
            performance_metrics={
                "avg_trade_duration": 2.5,
                "avg_win": 0.03,
                "avg_loss": -0.015
            }
        )
    
    def test_analysis_mode_initializes_backtest_integrator_when_enabled(
        self, analysis_config_with_integration, portfolio
    ):
        """Test that AnalysisMode initializes BacktestResultIntegrator when enabled."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Should have backtest integrator initialized
        assert hasattr(mode, 'backtest_integrator')
        assert isinstance(mode.backtest_integrator, BacktestResultIntegrator)
        assert mode.backtest_integrator.is_enabled()
    
    def test_analysis_mode_does_not_initialize_integrator_when_disabled(
        self, analysis_config_without_integration, portfolio
    ):
        """Test that AnalysisMode does not initialize BacktestResultIntegrator when disabled."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_without_integration,
            portfolio=portfolio
        )
        
        # Should not have backtest integrator or it should be disabled
        if hasattr(mode, 'backtest_integrator') and mode.backtest_integrator is not None:
            assert not mode.backtest_integrator.is_enabled()
        else:
            # Integrator is not initialized at all when disabled (or is None)
            assert mode.backtest_integrator is None
    
    def test_analysis_mode_uses_default_disabled_integration(self, portfolio):
        """Test that AnalysisMode defaults to disabled integration."""
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "comprehensive",
                "include_backtesting": True
                # No backtest_integration specified - should default to disabled
            }
        )
        
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=config,
            portfolio=portfolio
        )
        
        # Should not have integration enabled by default
        if hasattr(mode, 'backtest_integrator') and mode.backtest_integrator is not None:
            assert not mode.backtest_integrator.is_enabled()
        else:
            # No integrator at all when disabled by default (or is None)
            assert mode.backtest_integrator is None
    
    @pytest.mark.asyncio
    async def test_analysis_mode_feeds_backtest_results_to_integrator(
        self, analysis_config_with_integration, portfolio, sample_backtest_result
    ):
        """Test that AnalysisMode feeds backtest results to integrator after completion."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock the integrator
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        mock_integrator.process_backtest_result = AsyncMock(return_value=True)
        mode.backtest_integrator = mock_integrator
        
        # Feed backtest result
        await mode.process_backtest_result(sample_backtest_result)
        
        # Should call integrator
        mock_integrator.process_backtest_result.assert_called_once_with(sample_backtest_result)
    
    @pytest.mark.asyncio
    async def test_analysis_mode_skips_integration_when_disabled(
        self, analysis_config_without_integration, portfolio, sample_backtest_result
    ):
        """Test that AnalysisMode skips integration when disabled."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_without_integration,
            portfolio=portfolio
        )
        
        # Should not call integrator when disabled
        result = await mode.process_backtest_result(sample_backtest_result)
        
        # Should still return successfully but without integration
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_integration_handles_learning_engine_gracefully(
        self, analysis_config_with_integration, portfolio, sample_backtest_result
    ):
        """Test that integration handles missing learning engine gracefully."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock integrator without learning engine
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        mock_integrator.can_feed_to_learning_system = Mock(return_value=False)
        mock_integrator.process_backtest_result = AsyncMock(return_value=True)
        mode.backtest_integrator = mock_integrator
        
        # Should handle gracefully without learning engine
        result = await mode.process_backtest_result(sample_backtest_result)
        assert result is not None
        mock_integrator.process_backtest_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_error_handling(
        self, analysis_config_with_integration, portfolio, sample_backtest_result
    ):
        """Test that integration errors are handled without breaking analysis."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock integrator that raises an error
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        mock_integrator.process_backtest_result = AsyncMock(side_effect=Exception("Integration error"))
        mode.backtest_integrator = mock_integrator
        
        # Should handle error gracefully and not break analysis
        try:
            await mode.process_backtest_result(sample_backtest_result)
            # If we get here, error was handled gracefully
            assert True
        except Exception as e:
            # Should not propagate integration errors
            assert "Integration error" not in str(e)
    
    @pytest.mark.asyncio
    async def test_model_validation_performance_tracking(
        self, analysis_config_with_integration, portfolio
    ):
        """Test that model validation performance is tracked against historical data."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock integrator with performance tracking
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        
        # Create a proper ModelPerformanceComparison object
        from src.modes.backtest_integration import ModelPerformanceComparison
        mock_comparison = ModelPerformanceComparison(
            backtest_metrics={"sharpe_ratio": 1.5},
            live_metrics={"sharpe_ratio": 1.3},
            comparison_window=30,
            created_at=datetime.now()
        )
        mock_integrator.get_performance_tracking_history = Mock(return_value=[mock_comparison])
        mode.backtest_integrator = mock_integrator
        
        # Get validation performance history
        history = mode.get_validation_performance_history()
        
        assert len(history) == 1
        assert "backtest_metrics" in history[0]
        assert "live_metrics" in history[0]
        mock_integrator.get_performance_tracking_history.assert_called_once()
    
    def test_integration_configuration_parsing(self, portfolio):
        """Test that integration configuration is properly parsed."""
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "backtest_integration": {
                    "enabled": True,
                    "validation_enabled": False,
                    "min_backtest_trades": 15,
                    "validation_sample_ratio": 0.3,
                    "max_validation_age_days": 45
                }
            }
        )
        
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=config,
            portfolio=portfolio
        )
        
        # Should parse custom configuration
        if hasattr(mode, 'backtest_integrator'):
            integration_config = mode.backtest_integrator.config
            assert integration_config.enabled == True
            assert integration_config.validation_enabled == False
            assert integration_config.min_backtest_trades == 15
            assert integration_config.validation_sample_ratio == 0.3
            assert integration_config.max_validation_age_days == 45
    
    @pytest.mark.asyncio
    async def test_integration_metrics_collection(
        self, analysis_config_with_integration, portfolio, sample_backtest_result
    ):
        """Test that integration metrics are collected properly."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock integrator with metrics
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        mock_integrator.process_backtest_result = AsyncMock(return_value=True)
        mock_integrator.get_integration_metrics = Mock(return_value={
            "status": "active",
            "validation_data_points_processed": 50,
            "performance_comparisons_created": 3,
            "last_integration_time": datetime.now().isoformat()
        })
        mode.backtest_integrator = mock_integrator
        
        await mode.process_backtest_result(sample_backtest_result)
        
        # Get integration metrics
        metrics = mode.get_backtest_integration_metrics()
        
        assert metrics["status"] == "active"
        assert metrics["validation_data_points_processed"] == 50
        assert metrics["performance_comparisons_created"] == 3
        assert "last_integration_time" in metrics
    
    @pytest.mark.asyncio
    async def test_selective_integration_based_on_strategy(
        self, analysis_config_with_integration, portfolio
    ):
        """Test that integration can be selective based on strategy performance."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Create backtest results with different quality
        good_backtest = BacktestResult(
            strategy_name="GOOD_STRATEGY",
            total_return=0.25,
            annual_return=0.30,
            max_drawdown=-0.03,
            sharpe_ratio=2.1,
            sortino_ratio=2.5,
            win_rate=0.7,
            total_trades=20,
            profit_factor=2.0,
            trade_history=[
                {"timestamp": datetime.now(), "action": "BUY", "price": 100.0, "return": 0.05}
                for _ in range(20)
            ],
            performance_metrics={}
        )
        
        poor_backtest = BacktestResult(
            strategy_name="POOR_STRATEGY",
            total_return=-0.1,
            annual_return=-0.12,
            max_drawdown=-0.25,
            sharpe_ratio=-0.5,
            sortino_ratio=-0.8,
            win_rate=0.3,
            total_trades=5,  # Below minimum
            profit_factor=0.5,
            trade_history=[
                {"timestamp": datetime.now(), "action": "SELL", "price": 100.0, "return": -0.02}
                for _ in range(5)
            ],
            performance_metrics={}
        )
        
        # Mock integrator
        mock_integrator = Mock(spec=BacktestResultIntegrator)
        mock_integrator.is_enabled = Mock(return_value=True)
        mock_integrator.process_backtest_result = AsyncMock(return_value=True)
        mode.backtest_integrator = mock_integrator
        
        # Process both results
        await mode.process_backtest_result(good_backtest)
        await mode.process_backtest_result(poor_backtest)
        
        # Both should be processed (validation logic is in integrator)
        assert mock_integrator.process_backtest_result.call_count == 2
    
    @pytest.mark.asyncio
    async def test_integration_with_continuous_learning_engine(
        self, analysis_config_with_integration, portfolio, sample_backtest_result
    ):
        """Test integration with actual continuous learning engine."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=analysis_config_with_integration,
            portfolio=portfolio
        )
        
        # Mock continuous learning engine
        mock_learning_engine = Mock()
        mock_learning_engine.add_validation_data = AsyncMock(return_value=True)
        mock_learning_engine.get_performance_metrics = AsyncMock(return_value={
            "total_return": 0.12,
            "sharpe_ratio": 1.3,
            "win_rate": 0.58
        })
        
        # Create integrator with learning engine
        integration_config = BacktestIntegrationConfig(enabled=True)
        integrator = BacktestResultIntegrator(integration_config, mock_learning_engine)
        mode.backtest_integrator = integrator
        
        # Process backtest result
        await mode.process_backtest_result(sample_backtest_result)
        
        # Should have called learning engine
        mock_learning_engine.add_validation_data.assert_called_once()
        mock_learning_engine.get_performance_metrics.assert_called_once()
    
    def test_configuration_validation_errors(self, portfolio):
        """Test that invalid integration configuration raises appropriate errors."""
        invalid_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "backtest_integration": {
                    "enabled": True,
                    "validation_sample_ratio": 1.5  # Invalid - > 1.0
                }
            }
        )
        
        # Should raise configuration error
        with pytest.raises(ValueError):
            mode = AnalysisMode(
                mode_id=uuid4(),
                config=invalid_config,
                portfolio=portfolio
            )


class TestIntegrationOptionalBehavior:
    """Test that integration is completely optional and doesn't affect existing functionality"""
    
    @pytest.fixture
    def portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = uuid4()
        portfolio.cash_balance = Decimal("10000")
        portfolio.total_value = Decimal("10000")
        portfolio.open_positions = {}
        return portfolio
    
    @pytest.fixture
    def minimal_analysis_config(self):
        """Create minimal analysis mode configuration without integration."""
        return ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "basic"
            }
        )
    
    @pytest.mark.asyncio
    async def test_existing_functionality_unaffected(self, minimal_analysis_config, portfolio):
        """Test that existing AnalysisMode functionality is unaffected by integration."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=minimal_analysis_config,
            portfolio=portfolio
        )
        
        # Basic functionality should work as before
        await mode.initialize()
        assert mode.status.name in ["INACTIVE", "INITIALIZED"]
        
        await mode.start()
        assert mode.status.name == "ACTIVE"
        
        # Market processing should work
        token = DiscoveredToken(
            address="test_address",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000.0,
            timestamp=datetime.now()
        )
        
        # Should process market tick without issues
        action = await mode.process_tick(market_state)
        assert action == TradeAction.HOLD or action is None
        
        await mode.stop()
        assert mode.status.name in ["INACTIVE", "STOPPED"]
    
    def test_no_integration_dependencies_loaded_when_disabled(self, minimal_analysis_config, portfolio):
        """Test that integration dependencies are not loaded when disabled."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=minimal_analysis_config,
            portfolio=portfolio
        )
        
        # Should not have integration-related attributes when disabled
        if hasattr(mode, 'backtest_integrator') and mode.backtest_integrator is not None:
            # If it exists, it should be disabled
            assert not mode.backtest_integrator.is_enabled()
        else:
            # Or it might not exist at all or be None - both are acceptable
            assert mode.backtest_integrator is None
    
    @pytest.mark.asyncio 
    async def test_performance_impact_minimal_when_disabled(self, minimal_analysis_config, portfolio):
        """Test that disabled integration has minimal performance impact."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=minimal_analysis_config,
            portfolio=portfolio
        )
        
        # Time the initialization
        start_time = datetime.now()
        await mode.initialize()
        await mode.start()
        init_time = (datetime.now() - start_time).total_seconds()
        
        # Should initialize quickly without integration
        assert init_time < 1.0  # Should be very fast
        
        # Market processing should be fast
        token = DiscoveredToken(
            address="test_address",
            symbol="TEST", 
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000.0,
            timestamp=datetime.now()
        )
        
        start_time = datetime.now()
        await mode.process_tick(market_state)
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Market processing should be fast
        assert process_time < 0.1
        
        await mode.stop()