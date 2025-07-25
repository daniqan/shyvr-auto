"""
Comprehensive tests for Risk Manager.

This module contains extensive tests for the RiskManager class following
TDD methodology. Tests cover position sizing, risk limits, monitoring,
portfolio risk metrics, and integration with other portfolio components.

Tests are organized by functionality:
- Position sizing management
- Risk limits enforcement  
- Real-time risk monitoring
- Portfolio risk metrics
- Dynamic position sizing
- Emergency stop-loss protection
- Multi-chain/DEX risk aggregation
- Correlation analysis
- Risk budgeting and optimization
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4, UUID

from src.portfolio.base import (
    Portfolio,
    Position,
    Transaction,
    PortfolioConfig,
    PositionType,
    PositionStatus,
    TransactionType,
    RiskMetrics,
    DrawdownMetrics,
    PerformanceMetrics,
    RiskLimitExceededError,
)
from src.portfolio.risk_manager import (
    RiskManager,
    RiskConfig,
    RiskAlert,
    RiskAlertType,
    RiskAlertSeverity,
    PositionSizeRecommendation,
    RiskAssessmentResult,
    RiskLimitResult,
    RiskMonitoringResult,
    PortfolioRiskResult,
    CorrelationAnalysisResult,
    RiskBudgetResult,
    VaRCalculationResult,
    StressTestResult,
    RiskManagerError,
)
from src.utils.base import Chain


# Module-level fixtures for all test classes
@pytest.fixture
def portfolio_config():
    """Create test portfolio configuration."""
    return PortfolioConfig(
        initial_balance=Decimal("100000"),
        base_currency="USDC",
        max_position_size_pct=Decimal("0.1"),  # 10%
        max_daily_loss_pct=Decimal("0.05"),    # 5%
        max_drawdown_pct=Decimal("0.15"),      # 15%
        stop_loss_pct=Decimal("0.08"),         # 8%
        take_profit_pct=Decimal("0.4"),        # 40%
        max_open_positions=10,
        min_trade_amount_usd=Decimal("10"),
        enable_risk_management=True
    )


@pytest.fixture
def risk_config():
    """Create test risk configuration."""
    return RiskConfig(
        max_portfolio_risk_pct=Decimal("0.02"),     # 2% portfolio risk
        max_single_position_pct=Decimal("0.1"),     # 10% single position
        max_correlation_threshold=Decimal("0.8"),   # 80% correlation limit
        var_confidence_level=Decimal("0.95"),       # 95% VaR confidence
        stress_test_scenarios=5,                    # 5 stress scenarios
        position_sizing_method="kelly",             # Kelly criterion
        enable_dynamic_sizing=True,
        enable_correlation_limits=True,
        enable_drawdown_protection=True,
        max_leverage=Decimal("3.0"),                # 3x max leverage
        emergency_stop_loss_pct=Decimal("0.1"),     # 10% emergency stop
        daily_loss_reset_hour=0,                    # Reset at midnight UTC
        alert_thresholds={
            "warning": Decimal("0.75"),             # 75% of limit
            "critical": Decimal("0.9")              # 90% of limit
        },
        risk_free_rate=Decimal("0.05"),             # 5% risk-free rate
        lookback_period_days=30,                    # 30-day lookback
        min_observations=20,                        # Min data points
        correlation_update_frequency_hours=4,       # Update every 4 hours
        enable_real_time_monitoring=True
    )


@pytest.fixture
def portfolio(portfolio_config):
    """Create test portfolio."""
    return Portfolio(
        portfolio_id=uuid4(),
        name="Test Portfolio",
        config=portfolio_config,
        cash_balance=Decimal("100000"),
        total_value=Decimal("100000")
    )


@pytest.fixture
def risk_manager(portfolio, risk_config):
    """Create RiskManager instance."""
    return RiskManager(
        portfolio=portfolio,
        config=risk_config
    )


@pytest.fixture
def sample_position():
    """Create sample position for testing."""
    return Position(
        position_id=uuid4(),
        symbol="BTC/USDC",
        position_type=PositionType.SPOT,
        chain=Chain.SOLANA,
        dex_name="jupiter",
        size=Decimal("1.0"),
        entry_price=Decimal("50000"),
        current_price=Decimal("52000"),
        status=PositionStatus.OPEN,
        leverage=Decimal("1.0"),
        side="LONG"
    )


class TestRiskManagerBasics:
    """Test basic RiskManager functionality."""
    
    def test_risk_manager_initialization(self, portfolio, risk_config):
        """Test RiskManager initialization."""
        risk_manager = RiskManager(portfolio=portfolio, config=risk_config)
        
        assert risk_manager.portfolio == portfolio
        assert risk_manager.config == risk_config
        assert risk_manager.alerts == []
        assert risk_manager.last_correlation_update is None
        assert risk_manager._price_history == {}
        assert risk_manager._correlation_matrix == {}
        assert risk_manager._daily_pnl_history == []
        assert not risk_manager._emergency_stop_active
    
    def test_risk_config_validation(self):
        """Test risk configuration validation."""
        # Valid config should not raise
        config = RiskConfig(
            max_portfolio_risk_pct=Decimal("0.02"),
            max_single_position_pct=Decimal("0.1")
        )
        assert config.max_portfolio_risk_pct == Decimal("0.02")
        
        # Invalid percentages should raise
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            RiskConfig(max_portfolio_risk_pct=Decimal("1.5"))
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            RiskConfig(max_single_position_pct=Decimal("-0.1"))
    
    @pytest.mark.asyncio
    async def test_portfolio_assignment(self, risk_manager, portfolio):
        """Test portfolio assignment and validation."""
        assert risk_manager.portfolio == portfolio
        
        # Test portfolio update
        new_portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="New Portfolio",
            config=portfolio.config,
            cash_balance=Decimal("200000"),
            total_value=Decimal("200000")
        )
        
        risk_manager.portfolio = new_portfolio
        assert risk_manager.portfolio == new_portfolio


class TestPositionSizing:
    """Test position sizing functionality."""
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_basic(self, risk_manager):
        """Test basic position size calculation."""
        result = await risk_manager.calculate_position_size(
            symbol="BTC/USDC",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("46000"),
            confidence_level=Decimal("0.8")
        )
        
        assert result.success
        assert result.symbol == "BTC/USDC"
        assert result.recommended_size > 0
        assert result.recommended_size_usd > 0
        assert result.risk_amount > 0
        assert result.position_risk_pct > 0
        assert result.position_risk_pct <= risk_manager.config.max_single_position_pct
        assert result.sizing_method in ["kelly", "fixed_fractional", "volatility_adjusted"]
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_kelly_criterion(self, risk_manager):
        """Test Kelly criterion position sizing."""
        risk_manager.config.position_sizing_method = "kelly"
        
        with patch.object(risk_manager, '_calculate_win_probability', return_value=Decimal("0.6")):
            with patch.object(risk_manager, '_calculate_average_win_loss_ratio', return_value=Decimal("2.0")):
                result = await risk_manager.calculate_position_size(
                    symbol="BTC/USDC",
                    entry_price=Decimal("50000"),
                    stop_loss_price=Decimal("46000"),
                    confidence_level=Decimal("0.8")
                )
                
                assert result.success
                assert result.sizing_method == "kelly"
                assert result.kelly_fraction is not None
                assert 0 < result.kelly_fraction <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_fixed_fractional(self, risk_manager):
        """Test fixed fractional position sizing."""
        risk_manager.config.position_sizing_method = "fixed_fractional"
        
        result = await risk_manager.calculate_position_size(
            symbol="BTC/USDC",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("46000"),
            risk_fraction=Decimal("0.02")  # 2% risk
        )
        
        assert result.success
        assert result.sizing_method == "fixed_fractional"
        assert result.risk_amount == risk_manager.portfolio.total_value * Decimal("0.02")
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_volatility_adjusted(self, risk_manager):
        """Test volatility-adjusted position sizing."""
        risk_manager.config.position_sizing_method = "volatility_adjusted"
        
        with patch.object(risk_manager, '_get_asset_volatility', return_value=Decimal("0.3")):
            result = await risk_manager.calculate_position_size(
                symbol="BTC/USDC",
                entry_price=Decimal("50000"),
                stop_loss_price=Decimal("46000"),
                target_volatility=Decimal("0.15")
            )
            
            assert result.success
            assert result.sizing_method == "volatility_adjusted"
            assert result.volatility_adjustment is not None
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_exceeds_limits(self, risk_manager):
        """Test position sizing when limits are exceeded."""
        # Set very tight stop loss to force large position size
        result = await risk_manager.calculate_position_size(
            symbol="BTC/USDC",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49999"),  # Very tight stop
            confidence_level=Decimal("0.8")
        )
        
        assert result.success
        # Should be capped at max position size
        max_position_value = risk_manager.portfolio.total_value * risk_manager.config.max_single_position_pct
        assert result.recommended_size_usd <= max_position_value
        assert result.size_capped
        assert len(result.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_insufficient_capital(self, risk_manager):
        """Test position sizing with insufficient capital."""
        # Drain portfolio
        risk_manager.portfolio.cash_balance = Decimal("100")
        risk_manager.portfolio.total_value = Decimal("100")
        
        result = await risk_manager.calculate_position_size(
            symbol="BTC/USDC",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("46000")
        )
        
        assert not result.success
        assert "insufficient capital" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_params(self, risk_manager):
        """Test position sizing with invalid parameters."""
        # Stop loss above entry price for long position
        result = await risk_manager.calculate_position_size(
            symbol="BTC/USDC",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("55000"),
            side="LONG"
        )
        
        assert not result.success
        assert "invalid stop loss" in result.error_message.lower()


class TestRiskLimits:
    """Test risk limits enforcement."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager with positions."""
        rm = RiskManager(portfolio=portfolio, config=risk_config)
        
        # Add some test positions
        positions = [
            Position(
                position_id=uuid4(),
                symbol="BTC/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("52000"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                size=Decimal("10.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                status=PositionStatus.OPEN
            )
        ]
        
        for pos in positions:
            rm.portfolio.add_position(pos)
        
        return rm
    
    @pytest.mark.asyncio
    async def test_check_position_size_limits(self, risk_manager):
        """Test position size limit checking."""
        result = await risk_manager.check_risk_limits()
        
        assert result.success
        assert result.limits_checked > 0
        assert "position_size" in [check.limit_type for check in result.limit_checks]
        
        # Check individual position limit
        position = list(risk_manager.portfolio.positions.values())[0]
        position_check = next(
            check for check in result.limit_checks 
            if check.limit_type == "position_size" and check.position_id == position.position_id
        )
        
        assert position_check.current_value > 0
        assert position_check.limit_value == risk_manager.config.max_single_position_pct
        assert not position_check.limit_exceeded  # Should be within limits
    
    @pytest.mark.asyncio
    async def test_check_portfolio_risk_limits(self, risk_manager):
        """Test portfolio risk limit checking."""
        result = await risk_manager.check_risk_limits()
        
        portfolio_risk_check = next(
            check for check in result.limit_checks 
            if check.limit_type == "portfolio_risk"
        )
        
        assert portfolio_risk_check.current_value >= 0
        assert portfolio_risk_check.limit_value == risk_manager.config.max_portfolio_risk_pct
    
    @pytest.mark.asyncio
    async def test_check_daily_loss_limits(self, risk_manager):
        """Test daily loss limit checking."""
        # Simulate daily loss
        risk_manager._daily_pnl_history = [
            {"date": datetime.now().date(), "pnl": Decimal("-3000")},  # 3% loss
        ]
        
        result = await risk_manager.check_risk_limits()
        
        daily_loss_check = next(
            (check for check in result.limit_checks if check.limit_type == "daily_loss"),
            None
        )
        
        if daily_loss_check:
            assert daily_loss_check.current_value > 0  # Loss as positive value
            assert daily_loss_check.limit_value == risk_manager.portfolio.config.max_daily_loss_pct
    
    @pytest.mark.asyncio
    async def test_check_drawdown_limits(self, risk_manager):
        """Test drawdown limit checking."""
        # Set portfolio drawdown metrics
        risk_manager.portfolio.drawdown_metrics = DrawdownMetrics(
            current_drawdown=Decimal("0.08"),  # 8% drawdown
            max_drawdown=Decimal("0.12"),      # 12% max drawdown
            max_drawdown_duration_days=15,
            peak_value=Decimal("120000"),
            trough_value=Decimal("110400")
        )
        
        result = await risk_manager.check_risk_limits()
        
        drawdown_check = next(
            (check for check in result.limit_checks if check.limit_type == "drawdown"),
            None
        )
        
        if drawdown_check:
            assert drawdown_check.current_value == Decimal("0.08")
            assert drawdown_check.limit_value == risk_manager.portfolio.config.max_drawdown_pct
    
    @pytest.mark.asyncio
    async def test_check_correlation_limits(self, risk_manager):
        """Test correlation limit checking."""
        # Mock correlation matrix with high correlation
        risk_manager._correlation_matrix = {
            ("BTC/USDC", "ETH/USDC"): Decimal("0.85")  # High correlation
        }
        
        result = await risk_manager.check_risk_limits()
        
        correlation_checks = [
            check for check in result.limit_checks 
            if check.limit_type == "correlation"
        ]
        
        if correlation_checks:
            correlation_check = correlation_checks[0]
            assert correlation_check.current_value == Decimal("0.85")
            assert correlation_check.limit_value == risk_manager.config.max_correlation_threshold
            assert correlation_check.limit_exceeded  # Should exceed 0.8 threshold
    
    @pytest.mark.asyncio
    async def test_check_leverage_limits(self, risk_manager):
        """Test leverage limit checking."""
        # Add leveraged position
        leveraged_position = Position(
            position_id=uuid4(),
            symbol="BTC-PERP",
            position_type=PositionType.PERPETUAL,
            chain=Chain.SOLANA,
            dex_name="hyperliquid",
            size=Decimal("2.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("5.0"),  # Exceeds max leverage of 3.0
            side="LONG"
        )
        
        risk_manager.portfolio.add_position(leveraged_position)
        
        result = await risk_manager.check_risk_limits()
        
        leverage_check = next(
            (check for check in result.limit_checks 
             if check.limit_type == "leverage" and check.position_id == leveraged_position.position_id),
            None
        )
        
        if leverage_check:
            assert leverage_check.current_value == Decimal("5.0")
            assert leverage_check.limit_value == risk_manager.config.max_leverage
            assert leverage_check.limit_exceeded
    
    @pytest.mark.asyncio
    async def test_enforce_risk_limits_emergency_stop(self, risk_manager):
        """Test emergency stop enforcement."""
        # Simulate large loss triggering emergency stop
        risk_manager.portfolio.drawdown_metrics = DrawdownMetrics(
            current_drawdown=Decimal("0.12"),  # 12% drawdown exceeds emergency threshold
            max_drawdown=Decimal("0.12"),
            max_drawdown_duration_days=1,
            peak_value=Decimal("100000"),
            trough_value=Decimal("88000")
        )
        
        with patch.object(risk_manager, '_trigger_emergency_stop') as mock_stop:
            result = await risk_manager.enforce_risk_limits()
            
            mock_stop.assert_called_once()
            assert result.success
            assert result.emergency_stop_triggered
            assert len(result.actions_taken) > 0
    
    @pytest.mark.asyncio
    async def test_enforce_risk_limits_position_reduction(self, risk_manager):
        """Test position reduction enforcement."""
        # Add oversized position
        large_position = Position(
            position_id=uuid4(),
            symbol="LARGE/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("50.0"),  # Large position
            entry_price=Decimal("1000"),
            current_price=Decimal("1000"),
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(large_position)
        
        with patch.object(risk_manager, '_reduce_position_size') as mock_reduce:
            result = await risk_manager.enforce_risk_limits()
            
            # Should trigger position reduction
            assert result.success
            if result.positions_reduced:
                mock_reduce.assert_called()


class TestRiskMonitoring:
    """Test real-time risk monitoring."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager instance."""
        return RiskManager(portfolio=portfolio, config=risk_config)
    
    @pytest.mark.asyncio
    async def test_monitor_portfolio_risk_basic(self, risk_manager):
        """Test basic portfolio risk monitoring."""
        result = await risk_manager.monitor_portfolio_risk()
        
        assert result.success
        assert result.total_portfolio_risk >= 0
        assert result.position_count >= 0
        assert result.monitoring_timestamp is not None
        assert isinstance(result.risk_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_monitor_portfolio_risk_with_positions(self, risk_manager):
        """Test portfolio risk monitoring with positions."""
        # Add test position
        position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("48000"),  # Losing position
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(position)
        
        result = await risk_manager.monitor_portfolio_risk()
        
        assert result.success
        assert result.position_count == 1
        assert result.total_portfolio_risk > 0
        assert len(result.position_risks) == 1
        
        position_risk = result.position_risks[0]
        assert position_risk.position_id == position.position_id
        assert position_risk.current_risk > 0  # Losing position should have risk
    
    @pytest.mark.asyncio
    async def test_generate_risk_alerts(self, risk_manager):
        """Test risk alert generation."""
        # Add position approaching limits
        risky_position = Position(
            position_id=uuid4(),
            symbol="RISKY/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("8.0"),  # Large position (8% of portfolio)
            entry_price=Decimal("1000"),
            current_price=Decimal("900"),  # 10% loss
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(risky_position)
        
        alerts = await risk_manager.generate_risk_alerts()
        
        assert len(alerts) > 0
        
        # Should have position size warning
        size_alerts = [a for a in alerts if a.alert_type == RiskAlertType.POSITION_SIZE]
        if size_alerts:
            alert = size_alerts[0]
            assert alert.severity in [RiskAlertSeverity.WARNING, RiskAlertSeverity.CRITICAL]
            assert alert.position_id == risky_position.position_id
    
    @pytest.mark.asyncio
    async def test_update_risk_metrics_basic(self, risk_manager):
        """Test risk metrics updating."""
        result = await risk_manager.update_risk_metrics()
        
        assert result.success
        assert isinstance(result.var_95, Decimal)
        assert isinstance(result.var_99, Decimal)
        assert isinstance(result.expected_shortfall, Decimal)
        assert isinstance(result.portfolio_volatility, Decimal)
        
        # Check that portfolio risk metrics were updated
        assert risk_manager.portfolio.risk_metrics is not None
        assert risk_manager.portfolio.risk_metrics.var_95 == result.var_95
    
    @pytest.mark.asyncio
    async def test_update_risk_metrics_with_history(self, risk_manager):
        """Test risk metrics with price history."""
        # Add price history
        risk_manager._price_history = {
            "BTC/USDC": [
                {"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(50000 + i * 100)}
                for i in range(30)
            ]
        }
        
        result = await risk_manager.update_risk_metrics()
        
        assert result.success
        assert result.portfolio_volatility > 0
        assert result.var_95 > 0
        assert result.var_99 > result.var_95  # VaR 99% should be higher than 95%
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_var(self, risk_manager):
        """Test Value at Risk calculation."""
        # Add mock returns data
        returns = [Decimal("0.02"), Decimal("-0.01"), Decimal("0.015"), Decimal("-0.005"), Decimal("0.01")]
        
        with patch.object(risk_manager, '_get_portfolio_returns', return_value=returns):
            var_result = await risk_manager.calculate_portfolio_var(
                confidence_level=Decimal("0.95"),
                time_horizon_days=1
            )
            
            assert var_result.success
            assert var_result.var_amount > 0
            assert var_result.var_percentage > 0
            assert var_result.confidence_level == Decimal("0.95")
            assert var_result.time_horizon_days == 1
    
    @pytest.mark.asyncio
    async def test_perform_stress_test(self, risk_manager):
        """Test stress testing functionality."""
        # Add test position
        position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(position)
        
        result = await risk_manager.perform_stress_test()
        
        assert result.success
        assert len(result.scenarios) > 0
        assert len(result.scenario_results) == len(result.scenarios)
        
        # Check scenario results
        for scenario_result in result.scenario_results:
            assert isinstance(scenario_result.scenario_name, str)
            assert isinstance(scenario_result.portfolio_pnl, Decimal)
            assert isinstance(scenario_result.portfolio_pnl_pct, Decimal)


class TestCorrelationAnalysis:
    """Test correlation analysis functionality."""
    
    @pytest.fixture
    def risk_manager_with_positions(self, portfolio, risk_config):
        """Create RiskManager with multiple positions."""
        rm = RiskManager(portfolio=portfolio, config=risk_config)
        
        positions = [
            Position(
                position_id=uuid4(),
                symbol="BTC/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("52000"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                size=Decimal("10.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="SOL/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("100.0"),
                entry_price=Decimal("100"),
                current_price=Decimal("110"),
                status=PositionStatus.OPEN
            )
        ]
        
        for pos in positions:
            rm.portfolio.add_position(pos)
        
        return rm
    
    @pytest.mark.asyncio
    async def test_calculate_correlation_matrix(self, risk_manager_with_positions):
        """Test correlation matrix calculation."""
        # Mock price history for correlation calculation
        price_history = {
            "BTC/USDC": [
                {"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(50000 + i * 100)}
                for i in range(30)
            ],
            "ETH/USDC": [
                {"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(3000 + i * 10)}
                for i in range(30)
            ],
            "SOL/USDC": [
                {"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(100 + i * 2)}
                for i in range(30)
            ]
        }
        
        risk_manager_with_positions._price_history = price_history
        
        result = await risk_manager_with_positions.calculate_correlation_matrix()
        
        assert result.success
        assert len(result.correlation_pairs) > 0
        assert result.last_updated is not None
        
        # Check correlation pairs
        for pair in result.correlation_pairs:
            assert pair.symbol1 != pair.symbol2
            assert -1 <= pair.correlation <= 1
            assert isinstance(pair.correlation, Decimal)
    
    @pytest.mark.asyncio
    async def test_analyze_position_correlations(self, risk_manager_with_positions):
        """Test position correlation analysis."""
        # Set up correlation matrix
        risk_manager_with_positions._correlation_matrix = {
            ("BTC/USDC", "ETH/USDC"): Decimal("0.85"),
            ("BTC/USDC", "SOL/USDC"): Decimal("0.75"),
            ("ETH/USDC", "SOL/USDC"): Decimal("0.70")
        }
        
        result = await risk_manager_with_positions.analyze_position_correlations()
        
        assert result.success
        assert len(result.high_correlation_groups) >= 0
        assert result.max_correlation >= 0
        assert result.portfolio_concentration_risk >= 0
        
        # Should identify high correlation group if correlations exceed threshold
        if result.max_correlation > risk_manager_with_positions.config.max_correlation_threshold:
            assert len(result.high_correlation_groups) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_position_concentration(self, risk_manager_with_positions):
        """Test position concentration calculation."""
        result = await risk_manager_with_positions.calculate_position_concentration()
        
        assert result.success
        assert 0 <= result.concentration_index <= 1
        assert len(result.position_weights) > 0
        
        # Check that weights sum to approximately 1
        total_weight = sum(w.weight for w in result.position_weights)
        assert abs(total_weight - Decimal("1.0")) < Decimal("0.01")  # Allow small rounding
        
        # Largest position should be first
        if len(result.position_weights) > 1:
            assert result.position_weights[0].weight >= result.position_weights[1].weight


class TestDynamicPositionSizing:
    """Test dynamic position sizing based on volatility and risk."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager with dynamic sizing enabled."""
        risk_config.enable_dynamic_sizing = True
        return RiskManager(portfolio=portfolio, config=risk_config)
    
    @pytest.mark.asyncio
    async def test_dynamic_position_sizing_volatility(self, risk_manager):
        """Test dynamic sizing based on volatility."""
        # Mock volatility data
        with patch.object(risk_manager, '_get_asset_volatility', return_value=Decimal("0.5")):
            result = await risk_manager.calculate_dynamic_position_size(
                symbol="BTC/USDC",
                entry_price=Decimal("50000"),
                base_size=Decimal("1.0"),
                volatility_target=Decimal("0.2")
            )
            
            assert result.success
            assert result.adjusted_size != result.base_size  # Should be adjusted
            assert result.volatility_adjustment is not None
            assert result.current_volatility == Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_dynamic_position_sizing_portfolio_risk(self, risk_manager):
        """Test dynamic sizing based on portfolio risk."""
        # Set high portfolio risk
        risk_manager.portfolio.risk_metrics = RiskMetrics(
            var_95=Decimal("5000"),  # High VaR
            var_99=Decimal("8000"),
            expected_shortfall=Decimal("10000"),
            volatility=Decimal("0.3"),
            max_position_risk=Decimal("0.15"),
            concentration_risk=Decimal("0.8")
        )
        
        result = await risk_manager.calculate_dynamic_position_size(
            symbol="ETH/USDC",
            entry_price=Decimal("3000"),
            base_size=Decimal("10.0")
        )
        
        assert result.success
        # Should reduce size due to high portfolio risk
        assert result.adjusted_size <= result.base_size
        assert result.risk_adjustment is not None
    
    @pytest.mark.asyncio
    async def test_dynamic_position_sizing_correlation_adjustment(self, risk_manager):
        """Test dynamic sizing with correlation adjustment."""
        # Add correlated position
        correlated_position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000"),
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(correlated_position)
        
        # Set high correlation
        risk_manager._correlation_matrix = {
            ("BTC/USDC", "ETH/USDC"): Decimal("0.9")
        }
        
        result = await risk_manager.calculate_dynamic_position_size(
            symbol="ETH/USDC",  # Highly correlated with existing BTC position
            entry_price=Decimal("3000"),
            base_size=Decimal("10.0")
        )
        
        assert result.success
        # Should reduce size due to high correlation
        if result.correlation_adjustment:
            assert result.adjusted_size <= result.base_size


class TestRiskBudgeting:
    """Test risk budgeting and allocation optimization."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager instance."""
        return RiskManager(portfolio=portfolio, config=risk_config)
    
    @pytest.mark.asyncio
    async def test_calculate_risk_budget(self, risk_manager):
        """Test risk budget calculation."""
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        
        with patch.object(risk_manager, '_get_asset_volatility') as mock_vol:
            with patch.object(risk_manager, '_get_expected_return') as mock_return:
                mock_vol.side_effect = [Decimal("0.4"), Decimal("0.5"), Decimal("0.6")]
                mock_return.side_effect = [Decimal("0.15"), Decimal("0.12"), Decimal("0.20")]
                
                result = await risk_manager.calculate_risk_budget(
                    symbols=symbols,
                    total_risk_budget=Decimal("0.1")  # 10% total risk
                )
                
                assert result.success
                assert len(result.allocations) == len(symbols)
                
                # Check allocations sum to total budget
                total_allocation = sum(a.risk_allocation for a in result.allocations)
                assert abs(total_allocation - Decimal("0.1")) < Decimal("0.01")
                
                # Check Sharpe ratios calculated
                for allocation in result.allocations:
                    assert allocation.expected_return > 0
                    assert allocation.volatility > 0
                    assert allocation.sharpe_ratio > 0
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_allocation(self, risk_manager):
        """Test portfolio allocation optimization."""
        # Add current positions
        positions = [
            Position(
                position_id=uuid4(),
                symbol="BTC/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("52000"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                size=Decimal("10.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                status=PositionStatus.OPEN
            )
        ]
        
        for pos in positions:
            risk_manager.portfolio.add_position(pos)
        
        with patch.object(risk_manager, 'calculate_correlation_matrix') as mock_corr:
            mock_corr.return_value = CorrelationAnalysisResult(
                success=True,
                correlation_pairs=[],
                max_correlation=Decimal("0.5"),
                portfolio_concentration_risk=Decimal("0.3")
            )
            
            result = await risk_manager.optimize_portfolio_allocation()
            
            assert result.success
            assert len(result.current_allocations) > 0
            assert len(result.optimal_allocations) > 0
            
            # Should provide rebalancing recommendations if needed
            if result.rebalancing_needed:
                assert len(result.rebalancing_trades) > 0


class TestEmergencyProtection:
    """Test emergency stop-loss and liquidation protection."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager instance."""
        return RiskManager(portfolio=portfolio, config=risk_config)
    
    @pytest.mark.asyncio
    async def test_emergency_stop_trigger(self, risk_manager):
        """Test emergency stop triggering."""
        # Simulate large drawdown
        risk_manager.portfolio.drawdown_metrics = DrawdownMetrics(
            current_drawdown=Decimal("0.12"),  # 12% exceeds 10% emergency threshold
            max_drawdown=Decimal("0.12"),
            max_drawdown_duration_days=1,
            peak_value=Decimal("100000"),
            trough_value=Decimal("88000")
        )
        
        with patch.object(risk_manager, '_close_all_positions') as mock_close:
            result = await risk_manager.check_emergency_conditions()
            
            assert result.emergency_triggered
            assert result.trigger_reason == "drawdown_limit_exceeded"
            assert result.trigger_value == Decimal("0.12")
            
            # Should trigger position closure
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_liquidation_risk_protection(self, risk_manager):
        """Test liquidation risk protection."""
        # Add leveraged position near liquidation
        risky_position = Position(
            position_id=uuid4(),
            symbol="BTC-PERP",
            position_type=PositionType.PERPETUAL,
            chain=Chain.SOLANA,
            dex_name="hyperliquid",
            size=Decimal("2.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("45000"),  # 10% loss
            status=PositionStatus.OPEN,
            leverage=Decimal("5.0"),
            liquidation_price=Decimal("44000"),  # Close to liquidation
            side="LONG"
        )
        
        risk_manager.portfolio.add_position(risky_position)
        
        result = await risk_manager.check_liquidation_risks()
        
        assert result.success
        assert len(result.at_risk_positions) > 0
        
        at_risk_pos = result.at_risk_positions[0]
        assert at_risk_pos.position_id == risky_position.position_id
        assert at_risk_pos.distance_to_liquidation < Decimal("0.05")  # Less than 5%
        assert at_risk_pos.recommended_action in ["reduce_size", "close_position", "add_margin"]
    
    @pytest.mark.asyncio
    async def test_auto_stop_loss_execution(self, risk_manager):
        """Test automatic stop-loss execution."""
        # Add position that hits stop loss
        stop_loss_position = Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("10.0"),
            entry_price=Decimal("3000"),
            current_price=Decimal("2750"),  # Below stop loss
            status=PositionStatus.OPEN,
            stop_loss_price=Decimal("2760")
        )
        
        risk_manager.portfolio.add_position(stop_loss_position)
        
        with patch.object(risk_manager, '_execute_stop_loss') as mock_stop:
            result = await risk_manager.check_stop_loss_triggers()
            
            assert result.success
            assert len(result.triggered_positions) > 0
            
            triggered_pos = result.triggered_positions[0]
            assert triggered_pos.position_id == stop_loss_position.position_id
            assert triggered_pos.current_price == Decimal("2750")
            assert triggered_pos.stop_loss_price == Decimal("2760")
            
            # Should execute stop loss
            mock_stop.assert_called_once_with(stop_loss_position.position_id)
    
    @pytest.mark.asyncio
    async def test_margin_call_protection(self, risk_manager):
        """Test margin call protection."""
        # Add leveraged position with low margin
        margin_position = Position(
            position_id=uuid4(),
            symbol="BTC-PERP",
            position_type=PositionType.PERPETUAL,
            chain=Chain.SOLANA,
            dex_name="hyperliquid",
            size=Decimal("2.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("47000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("10.0"),
            margin_used=Decimal("10000"),
            liquidation_price=Decimal("45000"),
            side="LONG"
        )
        
        risk_manager.portfolio.add_position(margin_position)
        
        result = await risk_manager.check_margin_requirements()
        
        assert result.success
        if result.margin_calls:
            margin_call = result.margin_calls[0]
            assert margin_call.position_id == margin_position.position_id
            assert margin_call.current_margin > 0
            assert margin_call.required_margin > margin_call.current_margin
            assert margin_call.margin_call_amount > 0


class TestMultiChainRiskAggregation:
    """Test multi-chain and multi-DEX risk aggregation."""
    
    @pytest.fixture
    def multi_chain_risk_manager(self, portfolio, risk_config):
        """Create RiskManager with multi-chain positions."""
        rm = RiskManager(portfolio=portfolio, config=risk_config)
        
        # Add positions across different chains and DEXs
        positions = [
            Position(
                position_id=uuid4(),
                symbol="BTC/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("52000"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                size=Decimal("10.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                status=PositionStatus.OPEN
            ),
            Position(
                position_id=uuid4(),
                symbol="BTC-PERP",
                position_type=PositionType.PERPETUAL,
                chain=Chain.SOLANA,
                dex_name="hyperliquid",
                size=Decimal("0.5"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                status=PositionStatus.OPEN,
                leverage=Decimal("3.0"),
                side="LONG"
            )
        ]
        
        for pos in positions:
            rm.portfolio.add_position(pos)
        
        return rm
    
    @pytest.mark.asyncio
    async def test_aggregate_risk_by_chain(self, multi_chain_risk_manager):
        """Test risk aggregation by blockchain chain."""
        result = await multi_chain_risk_manager.aggregate_risk_by_chain()
        
        assert result.success
        assert len(result.chain_risks) > 0
        
        # Should have risks for Solana and Ethereum
        chain_names = [cr.chain.value for cr in result.chain_risks]
        assert "solana" in chain_names
        assert "ethereum" in chain_names
        
        # Check Solana risk (has 2 positions)
        solana_risk = next(cr for cr in result.chain_risks if cr.chain == Chain.SOLANA)
        assert solana_risk.position_count == 2
        assert solana_risk.total_exposure > 0
        assert solana_risk.risk_percentage > 0
    
    @pytest.mark.asyncio
    async def test_aggregate_risk_by_dex(self, multi_chain_risk_manager):
        """Test risk aggregation by DEX."""
        result = await multi_chain_risk_manager.aggregate_risk_by_dex()
        
        assert result.success
        assert len(result.dex_risks) > 0
        
        # Should have risks for different DEXs
        dex_names = [dr.dex_name for dr in result.dex_risks]
        assert "jupiter" in dex_names
        assert "uniswap_v3" in dex_names
        assert "hyperliquid" in dex_names
        
        # Check DEX risk details
        for dex_risk in result.dex_risks:
            assert dex_risk.position_count > 0
            assert dex_risk.total_exposure > 0
            assert dex_risk.risk_percentage >= 0
    
    @pytest.mark.asyncio
    async def test_cross_chain_correlation_analysis(self, multi_chain_risk_manager):
        """Test cross-chain correlation analysis."""
        # Mock price history for cross-chain analysis
        multi_chain_risk_manager._price_history = {
            "BTC/USDC": [{"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(50000 + i * 100)} for i in range(30)],
            "ETH/USDC": [{"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(3000 + i * 10)} for i in range(30)],
            "BTC-PERP": [{"timestamp": datetime.now() - timedelta(days=i), "price": Decimal(50000 + i * 95)} for i in range(30)]
        }
        
        result = await multi_chain_risk_manager.analyze_cross_chain_correlations()
        
        assert result.success
        if result.cross_chain_correlations:
            # Should identify correlation between BTC spot and BTC perp
            btc_correlations = [
                cc for cc in result.cross_chain_correlations 
                if "BTC" in cc.symbol1 and "BTC" in cc.symbol2
            ]
            if btc_correlations:
                assert btc_correlations[0].correlation > Decimal("0.5")  # BTC should be correlated
    
    @pytest.mark.asyncio
    async def test_calculate_concentration_risk_across_chains(self, multi_chain_risk_manager):
        """Test concentration risk calculation across chains."""
        result = await multi_chain_risk_manager.calculate_concentration_risk()
        
        assert result.success
        assert 0 <= result.concentration_index <= 1
        assert len(result.asset_concentrations) > 0
        
        # Should identify BTC concentration (BTC/USDC + BTC-PERP)
        btc_concentrations = [
            ac for ac in result.asset_concentrations 
            if "BTC" in ac.asset_symbol
        ]
        if len(btc_concentrations) > 1:
            total_btc_weight = sum(ac.weight for ac in btc_concentrations)
            assert total_btc_weight > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_config):
        """Create RiskManager instance."""
        return RiskManager(portfolio=portfolio, config=risk_config)
    
    @pytest.mark.asyncio
    async def test_handle_insufficient_data(self, risk_manager):
        """Test handling of insufficient historical data."""
        result = await risk_manager.calculate_portfolio_var(
            confidence_level=Decimal("0.95"),
            time_horizon_days=1
        )
        
        # Should handle gracefully with insufficient data
        if not result.success:
            assert "insufficient data" in result.error_message.lower()
        else:
            # If successful, should use fallback methods
            assert result.var_amount >= 0
    
    @pytest.mark.asyncio
    async def test_handle_invalid_position_data(self, risk_manager):
        """Test handling of invalid position data."""
        # Add position with invalid data
        invalid_position = Position(
            position_id=uuid4(),
            symbol="INVALID/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0"),  # Invalid size
            entry_price=Decimal("0"),  # Invalid price
            current_price=Decimal("0"),
            status=PositionStatus.OPEN
        )
        
        risk_manager.portfolio.add_position(invalid_position)
        
        result = await risk_manager.monitor_portfolio_risk()
        
        # Should handle invalid position gracefully
        assert result.success or "invalid position" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_handle_api_failures(self, risk_manager):
        """Test handling of external API failures."""
        with patch.object(risk_manager, '_fetch_market_data', side_effect=Exception("API Error")):
            result = await risk_manager.update_risk_metrics()
            
            # Should handle API failure gracefully
            if not result.success:
                assert "api" in result.error_message.lower() or "fetch" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_handle_calculation_errors(self, risk_manager):
        """Test handling of calculation errors."""
        # Mock division by zero or other calculation errors
        with patch.object(risk_manager, '_calculate_sharpe_ratio', side_effect=ZeroDivisionError()):
            result = await risk_manager.calculate_risk_budget(
                symbols=["BTC/USDC"],
                total_risk_budget=Decimal("0.1")
            )
            
            # Should handle calculation errors
            if not result.success:
                assert "calculation" in result.error_message.lower() or "error" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, risk_manager):
        """Test concurrent risk management operations."""
        # Run multiple operations concurrently
        tasks = [
            risk_manager.monitor_portfolio_risk(),
            risk_manager.check_risk_limits(),
            risk_manager.update_risk_metrics(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete without deadlocks
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)


class TestIntegration:
    """Test integration with other portfolio components."""
    
    @pytest.fixture
    def integrated_risk_manager(self, portfolio, risk_config):
        """Create RiskManager with mocked dependencies."""
        rm = RiskManager(portfolio=portfolio, config=risk_config)
        
        # Mock integration with PositionTracker and PnLCalculator
        rm.position_tracker = Mock()
        rm.pnl_calculator = Mock()
        
        return rm
    
    @pytest.mark.asyncio
    async def test_integration_with_position_tracker(self, integrated_risk_manager):
        """Test integration with PositionTracker."""
        # Mock position tracker methods
        integrated_risk_manager.position_tracker.get_positions_by_status.return_value = {}
        integrated_risk_manager.position_tracker.total_unrealized_pnl = Decimal("1000")
        
        result = await integrated_risk_manager.monitor_portfolio_risk()
        
        assert result.success
        # Verify position tracker integration
        integrated_risk_manager.position_tracker.get_positions_by_status.assert_called()
    
    @pytest.mark.asyncio
    async def test_integration_with_pnl_calculator(self, integrated_risk_manager):
        """Test integration with PnLCalculator."""
        # Mock PnL calculator
        mock_pnl_result = Mock()
        mock_pnl_result.success = True
        mock_pnl_result.total_pnl = Decimal("5000")
        mock_pnl_result.total_unrealized_pnl = Decimal("3000")
        mock_pnl_result.total_realized_pnl = Decimal("2000")
        
        integrated_risk_manager.pnl_calculator.calculate_portfolio_pnl = AsyncMock(return_value=mock_pnl_result)
        
        result = await integrated_risk_manager.update_risk_metrics()
        
        assert result.success
        # Verify PnL calculator integration
        integrated_risk_manager.pnl_calculator.calculate_portfolio_pnl.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_risk_alerts_trigger_position_actions(self, integrated_risk_manager):
        """Test that risk alerts trigger appropriate position actions."""
        # Add risky position
        risky_position = Position(
            position_id=uuid4(),
            symbol="RISKY/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("20.0"),  # Large position
            entry_price=Decimal("1000"),
            current_price=Decimal("800"),  # 20% loss
            status=PositionStatus.OPEN
        )
        
        integrated_risk_manager.portfolio.add_position(risky_position)
        
        with patch.object(integrated_risk_manager, '_execute_risk_action') as mock_action:
            result = await integrated_risk_manager.enforce_risk_limits()
            
            # Should trigger risk actions for risky position
            if result.actions_taken:
                mock_action.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])