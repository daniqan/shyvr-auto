"""
Comprehensive TDD tests for Enhanced LiveRiskManager.

This module contains failing tests that define the requirements for a production-grade
live risk manager with comprehensive position limits, exposure controls, real-time
risk monitoring, and advanced risk assessment capabilities.

Following TDD methodology - these tests are written first and will fail until
the implementation is complete.

Key Risk Management Requirements Tested:
- Position limits and exposure controls
- Real-time risk monitoring and alerts
- Concentration risk management
- Correlation-based risk assessment
- Dynamic position sizing
- Sector and token exposure limits
- Leverage and margin controls
- Risk-adjusted position sizing
- Portfolio risk budgeting
- Stress testing and scenario analysis
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import List, Dict, Any, Optional

from src.modes.base import ModeStatus
from src.portfolio.base import (
    Portfolio, PortfolioConfig, Position, PositionStatus, PositionType,
    PerformanceMetrics, RiskMetrics
)
from src.modes.live_mode import (
    LiveRiskManager, LiveModeConfig, RiskValidationResult, SafetyCheckResult
)
from src.rl_agent.base import MarketState


@pytest.fixture
def enhanced_live_config():
    """Create enhanced live mode configuration for risk testing."""
    return LiveModeConfig(
        initial_balance=Decimal("100000"),
        max_position_size_pct=Decimal("0.10"),        # 10% max single position
        max_daily_loss_pct=Decimal("0.05"),           # 5% daily loss limit
        max_drawdown_pct=Decimal("0.15"),             # 15% max drawdown
        max_open_positions=15,                        # 15 concurrent positions
        
        # Enhanced risk parameters
        max_sector_exposure_pct=Decimal("0.30"),      # 30% max sector exposure
        max_token_concentration_pct=Decimal("0.15"),  # 15% max single token
        max_correlated_exposure_pct=Decimal("0.25"),  # 25% max correlated exposure
        correlation_threshold=Decimal("0.70"),        # 70% correlation threshold
        max_leverage_ratio=Decimal("2.0"),            # 2x max leverage
        min_liquidity_requirement=Decimal("1000000"), # $1M minimum liquidity
        
        # Risk monitoring
        risk_check_frequency_seconds=5,               # Check every 5 seconds
        position_rebalance_threshold=Decimal("0.20"), # 20% rebalance threshold
        volatility_adjustment_factor=Decimal("0.5"),  # 50% volatility adjustment
        
        # Alert thresholds
        risk_warning_threshold=Decimal("0.75"),       # 75% of limit
        risk_critical_threshold=Decimal("0.90"),      # 90% of limit
        
        # Advanced risk features
        enable_dynamic_position_sizing=True,
        enable_correlation_monitoring=True,
        enable_sector_limits=True,
        enable_stress_testing=True,
        stress_test_scenarios=["flash_crash", "market_dump", "high_volatility"],
    )


@pytest.fixture
def mock_portfolio_with_positions():
    """Create mock portfolio with realistic positions for testing."""
    portfolio = Mock(spec=Portfolio)
    portfolio.total_value = Decimal("95000")
    
    # Create mock positions
    positions = {
        "pos_1": Mock(
            position_id=uuid4(),
            symbol="SOL/USDC",
            size=Decimal("1000"),
            entry_price=Decimal("100"),
            current_price=Decimal("95"),
            market_value=Decimal("9500"),
            unrealized_pnl=Decimal("-500"),
            status=PositionStatus.OPEN,
            token_address="SOL_ADDRESS",
            sector="L1_BLOCKCHAIN",
            correlation_group="MAJOR_ALTS"
        ),
        "pos_2": Mock(
            position_id=uuid4(),
            symbol="ETH/USDC", 
            size=Decimal("500"),
            entry_price=Decimal("2000"),
            current_price=Decimal("1950"),
            market_value=Decimal("9750"),
            unrealized_pnl=Decimal("-250"),
            status=PositionStatus.OPEN,
            token_address="ETH_ADDRESS",
            sector="L1_BLOCKCHAIN",
            correlation_group="MAJOR_ALTS"
        ),
        "pos_3": Mock(
            position_id=uuid4(),
            symbol="LINK/USDC",
            size=Decimal("2000"),
            entry_price=Decimal("15"),
            current_price=Decimal("14"),
            market_value=Decimal("2800"),
            unrealized_pnl=Decimal("-200"),
            status=PositionStatus.OPEN,
            token_address="LINK_ADDRESS",
            sector="ORACLE",
            correlation_group="DEFI_INFRASTRUCTURE"
        )
    }
    
    portfolio.positions = positions
    
    # Mock performance metrics
    performance = Mock(spec=PerformanceMetrics)
    performance.current_balance = Decimal("95000")
    performance.initial_balance = Decimal("100000")
    performance.total_pnl = Decimal("-5000")
    performance.max_drawdown = Decimal("0.05")
    performance.total_exposure = Decimal("22050")  # Sum of position values
    
    portfolio.get_performance_metrics.return_value = performance
    portfolio.performance_metrics = performance
    
    return portfolio


@pytest.fixture
def live_risk_manager(enhanced_live_config, mock_portfolio_with_positions):
    """Create live risk manager for testing."""
    return LiveRiskManager(
        portfolio=mock_portfolio_with_positions,
        config=enhanced_live_config,
        enable_real_time_monitoring=True
    )


class TestLiveRiskManagerInitialization:
    """Test live risk manager initialization and configuration."""
    
    def test_initialization_with_enhanced_config(self, enhanced_live_config, mock_portfolio_with_positions):
        """Test risk manager initializes with enhanced configuration."""
        risk_manager = LiveRiskManager(
            portfolio=mock_portfolio_with_positions,
            config=enhanced_live_config,
            enable_real_time_monitoring=True
        )
        
        assert risk_manager.config == enhanced_live_config
        assert risk_manager.portfolio == mock_portfolio_with_positions
        assert risk_manager.enable_real_time_monitoring is True
        assert risk_manager.max_position_size_pct == Decimal("0.10")
    
    def test_risk_tracking_initialization(self, live_risk_manager):
        """Test risk tracking variables are properly initialized."""
        assert live_risk_manager.daily_trades == 0
        assert live_risk_manager.daily_volume == Decimal("0")
        assert live_risk_manager.session_start_value == Decimal("0")
        assert live_risk_manager.last_risk_check is not None


class TestPositionSizeValidation:
    """Test position size validation with enhanced controls."""
    
    @pytest.mark.asyncio
    async def test_position_size_within_limits(self, live_risk_manager):
        """Test position size validation when within normal limits."""
        # 5% position size (within 10% limit)
        position_size = Decimal("5000")
        token_address = "NEW_TOKEN_ADDRESS"
        
        result = await live_risk_manager.validate_position_size(token_address, position_size)
        
        assert result.is_valid
        assert result.risk_level == SafetyCheckResult.SAFE
    
    @pytest.mark.asyncio
    async def test_position_size_exceeds_single_limit(self, live_risk_manager):
        """Test rejection when position size exceeds single position limit."""
        # 12% position size (exceeds 10% limit)
        position_size = Decimal("12000")
        token_address = "NEW_TOKEN_ADDRESS"
        
        result = await live_risk_manager.validate_position_size(token_address, position_size)
        
        assert not result.is_valid
        assert result.risk_level == SafetyCheckResult.DANGER
        assert "exceeds maximum" in result.reason
        assert "Reduce position size" in result.recommended_action
    
    @pytest.mark.asyncio
    async def test_concentration_risk_validation(self, live_risk_manager):
        """Test concentration risk validation for existing token exposure."""
        # Try to add more SOL when already have exposure
        existing_sol_exposure = Decimal("9500")  # From existing position
        new_position_size = Decimal("8000")      # Additional exposure
        total_exposure = existing_sol_exposure + new_position_size  # 17500 = 18.4%
        
        result = await live_risk_manager.validate_position_size("SOL_ADDRESS", new_position_size)
        
        assert not result.is_valid
        assert result.risk_level == SafetyCheckResult.WARNING
        assert "concentration" in result.reason.lower() or "exposure" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_sector_exposure_limits(self, live_risk_manager):
        """Test sector exposure limit enforcement."""
        # Current L1_BLOCKCHAIN exposure: SOL (9500) + ETH (9750) = 19250 (20.26%)
        # Adding more L1 exposure that would exceed 30% sector limit
        new_l1_position = Decimal("15000")  # Would make total L1 exposure 36.05%
        
        result = await live_risk_manager.validate_sector_exposure("L1_BLOCKCHAIN", new_l1_position)
        
        assert not result.is_valid
        assert result.risk_level == SafetyCheckResult.WARNING
        assert "sector exposure" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_correlation_based_sizing(self, live_risk_manager):
        """Test position sizing adjustment based on correlation."""
        # Try to add position highly correlated with existing MAJOR_ALTS group
        new_token_address = "ADA_ADDRESS"
        position_size = Decimal("8000")
        correlation_group = "MAJOR_ALTS"  # Same as SOL and ETH
        
        result = await live_risk_manager.validate_correlated_exposure(
            new_token_address, position_size, correlation_group
        )
        
        # Should flag high correlation risk
        assert result.risk_level in [SafetyCheckResult.WARNING, SafetyCheckResult.DANGER]
        assert "correlation" in result.reason.lower()


class TestRealTimeRiskMonitoring:
    """Test real-time risk monitoring and alert generation."""
    
    @pytest.mark.asyncio
    async def test_continuous_risk_monitoring(self, live_risk_manager):
        """Test continuous risk monitoring functionality."""
        # Start risk monitoring
        await live_risk_manager.start_risk_monitoring()
        
        assert live_risk_manager.is_monitoring_active
        assert live_risk_manager.monitoring_task is not None
    
    @pytest.mark.asyncio
    async def test_risk_threshold_alerts(self, live_risk_manager):
        """Test generation of risk alerts at different threshold levels."""
        # Simulate portfolio approaching risk limits
        portfolio_risk_pct = Decimal("0.76")  # 76% of limit (above warning threshold)
        
        alert = await live_risk_manager.check_risk_thresholds(portfolio_risk_pct)
        
        assert alert is not None
        assert alert.severity == "WARNING"
        assert "risk threshold" in alert.message.lower()
    
    @pytest.mark.asyncio
    async def test_critical_risk_alert(self, live_risk_manager):
        """Test critical risk alert generation."""
        # Simulate portfolio at critical risk level
        portfolio_risk_pct = Decimal("0.92")  # 92% of limit (above critical threshold)
        
        alert = await live_risk_manager.check_risk_thresholds(portfolio_risk_pct)
        
        assert alert is not None
        assert alert.severity == "CRITICAL"
        assert "critical" in alert.message.lower()
    
    @pytest.mark.asyncio
    async def test_risk_monitoring_frequency(self, live_risk_manager):
        """Test risk monitoring respects configured frequency."""
        live_risk_manager.last_risk_check = datetime.now()
        
        # Immediate check should be skipped due to frequency limit
        should_check = live_risk_manager.should_perform_risk_check()
        assert not should_check
        
        # Check after sufficient time should proceed
        live_risk_manager.last_risk_check = datetime.now() - timedelta(seconds=10)
        should_check = live_risk_manager.should_perform_risk_check()
        assert should_check


class TestDynamicPositionSizing:
    """Test dynamic position sizing based on market conditions and risk."""
    
    @pytest.mark.asyncio
    async def test_volatility_adjusted_sizing(self, live_risk_manager):
        """Test position sizing adjustment based on volatility."""
        market_state = Mock()
        market_state.volatility = 0.8  # High volatility
        base_position_size = Decimal("10000")
        
        adjusted_size = await live_risk_manager.calculate_volatility_adjusted_size(
            base_position_size, market_state
        )
        
        # Should reduce position size due to high volatility
        assert adjusted_size < base_position_size
        reduction_factor = adjusted_size / base_position_size
        assert reduction_factor == live_risk_manager.config.volatility_adjustment_factor
    
    @pytest.mark.asyncio
    async def test_kelly_criterion_sizing(self, live_risk_manager):
        """Test Kelly criterion-based position sizing."""
        # Mock historical performance data
        win_rate = Decimal("0.6")      # 60% win rate
        avg_win = Decimal("0.15")      # 15% average win
        avg_loss = Decimal("0.08")     # 8% average loss
        
        kelly_size = await live_risk_manager.calculate_kelly_position_size(
            win_rate, avg_win, avg_loss
        )
        
        # Kelly formula: f = (bp - q) / b where b=avg_win/avg_loss, p=win_rate, q=1-win_rate
        expected_kelly = (win_rate * (avg_win / avg_loss) - (1 - win_rate)) / (avg_win / avg_loss)
        
        assert abs(kelly_size - expected_kelly) < Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_risk_parity_sizing(self, live_risk_manager):
        """Test risk parity position sizing."""
        # Mock position volatilities
        position_volatilities = {
            "SOL_ADDRESS": Decimal("0.6"),    # 60% volatility
            "ETH_ADDRESS": Decimal("0.4"),    # 40% volatility
            "LINK_ADDRESS": Decimal("0.8"),   # 80% volatility
        }
        
        risk_parity_weights = await live_risk_manager.calculate_risk_parity_weights(
            position_volatilities
        )
        
        # Lower volatility assets should get higher weights
        assert risk_parity_weights["ETH_ADDRESS"] > risk_parity_weights["LINK_ADDRESS"]
        
        # Weights should sum to 1
        total_weight = sum(risk_parity_weights.values())
        assert abs(total_weight - Decimal("1.0")) < Decimal("0.01")


class TestAdvancedRiskAssessment:
    """Test advanced risk assessment and portfolio analysis."""
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_analysis(self, live_risk_manager):
        """Test correlation matrix analysis for portfolio risk."""
        # Mock correlation data
        correlation_matrix = {
            ("SOL_ADDRESS", "ETH_ADDRESS"): Decimal("0.75"),   # High correlation
            ("SOL_ADDRESS", "LINK_ADDRESS"): Decimal("0.45"),  # Moderate correlation
            ("ETH_ADDRESS", "LINK_ADDRESS"): Decimal("0.35"),  # Low correlation
        }
        
        portfolio_correlation_risk = await live_risk_manager.analyze_portfolio_correlations(
            correlation_matrix
        )
        
        assert portfolio_correlation_risk.max_correlation == Decimal("0.75")
        assert portfolio_correlation_risk.average_correlation > Decimal("0")
        assert portfolio_correlation_risk.high_correlation_pairs == 1  # SOL-ETH pair
    
    @pytest.mark.asyncio
    async def test_value_at_risk_calculation(self, live_risk_manager):
        """Test Value at Risk (VaR) calculation."""
        # Mock historical returns
        portfolio_returns = [
            Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"), 
            Decimal("-0.02"), Decimal("0.01"), Decimal("-0.04"),
            Decimal("0.02"), Decimal("-0.01"), Decimal("-0.03")
        ]
        confidence_level = Decimal("0.95")  # 95% confidence
        
        var_result = await live_risk_manager.calculate_portfolio_var(
            portfolio_returns, confidence_level
        )
        
        assert var_result.var_amount > Decimal("0")
        assert var_result.confidence_level == confidence_level
        assert var_result.time_horizon == 1  # 1-day VaR
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, live_risk_manager):
        """Test portfolio stress testing under extreme scenarios."""
        # Define stress test scenarios
        scenarios = {
            "flash_crash": {"market_drop": Decimal("0.30"), "correlation_increase": Decimal("0.9")},
            "market_dump": {"market_drop": Decimal("0.50"), "correlation_increase": Decimal("0.95")},
            "high_volatility": {"volatility_multiplier": Decimal("3.0")}
        }
        
        stress_results = await live_risk_manager.run_stress_tests(scenarios)
        
        assert len(stress_results) == 3
        assert "flash_crash" in stress_results
        assert stress_results["flash_crash"].portfolio_loss > Decimal("0")
        assert stress_results["market_dump"].portfolio_loss > stress_results["flash_crash"].portfolio_loss


class TestLeverageAndMarginControls:
    """Test leverage and margin control mechanisms."""
    
    @pytest.mark.asyncio
    async def test_leverage_limit_enforcement(self, live_risk_manager):
        """Test enforcement of leverage limits."""
        # Calculate current leverage
        total_position_value = Decimal("22050")  # Sum of all positions
        portfolio_value = Decimal("95000")
        current_leverage = total_position_value / portfolio_value  # ~0.23x
        
        # Try to add position that would exceed 2x leverage limit
        new_position_value = Decimal("180000")  # Would make total leverage > 2x
        
        leverage_check = await live_risk_manager.validate_leverage_limit(new_position_value)
        
        assert not leverage_check.is_valid
        assert "leverage" in leverage_check.reason.lower()
        assert leverage_check.risk_level == SafetyCheckResult.DANGER
    
    @pytest.mark.asyncio
    async def test_margin_requirement_calculation(self, live_risk_manager):
        """Test margin requirement calculation for positions."""
        position_value = Decimal("10000")
        leverage_ratio = Decimal("1.5")  # 1.5x leverage
        
        margin_required = await live_risk_manager.calculate_margin_requirement(
            position_value, leverage_ratio
        )
        
        # Margin should be position_value / leverage_ratio
        expected_margin = position_value / leverage_ratio
        assert margin_required == expected_margin
    
    @pytest.mark.asyncio
    async def test_margin_call_detection(self, live_risk_manager):
        """Test margin call detection when positions move against trader."""
        # Simulate positions with significant unrealized losses
        total_unrealized_loss = Decimal("8000")  # Large unrealized loss
        available_margin = Decimal("15000")
        margin_call_threshold = Decimal("0.20")  # 20% margin threshold
        
        margin_utilization = total_unrealized_loss / available_margin
        
        should_margin_call = margin_utilization > margin_call_threshold
        assert should_margin_call
        
        margin_call_result = await live_risk_manager.check_margin_requirements()
        assert margin_call_result.requires_margin_call


class TestLiquidityRiskManagement:
    """Test liquidity risk assessment and management."""
    
    @pytest.mark.asyncio
    async def test_liquidity_requirement_validation(self, live_risk_manager):
        """Test validation of minimum liquidity requirements."""
        token_address = "LOW_LIQUIDITY_TOKEN"
        position_size = Decimal("5000")
        available_liquidity = Decimal("500000")  # Below 1M requirement
        
        liquidity_check = await live_risk_manager.validate_liquidity_requirements(
            token_address, position_size, available_liquidity
        )
        
        assert not liquidity_check.is_valid
        assert "liquidity" in liquidity_check.reason.lower()
        assert liquidity_check.risk_level == SafetyCheckResult.WARNING
    
    @pytest.mark.asyncio
    async def test_position_impact_assessment(self, live_risk_manager):
        """Test assessment of position impact on market liquidity."""
        position_size = Decimal("50000")
        market_depth = Decimal("200000")  # Position is 25% of market depth
        
        impact_assessment = await live_risk_manager.assess_market_impact(
            position_size, market_depth
        )
        
        impact_percentage = position_size / market_depth
        assert impact_assessment.market_impact_pct == impact_percentage
        assert impact_assessment.is_high_impact  # 25% is high impact
    
    @pytest.mark.asyncio
    async def test_slippage_estimation(self, live_risk_manager):
        """Test slippage estimation for large positions."""
        order_size = Decimal("25000")
        order_book_data = Mock()
        order_book_data.total_liquidity = Decimal("100000")
        order_book_data.price_levels = 10
        
        estimated_slippage = await live_risk_manager.estimate_execution_slippage(
            order_size, order_book_data
        )
        
        assert estimated_slippage.slippage_pct > Decimal("0")
        assert estimated_slippage.execution_cost > Decimal("0")


class TestRiskBudgetingAndOptimization:
    """Test risk budgeting and portfolio optimization features."""
    
    @pytest.mark.asyncio
    async def test_risk_budget_allocation(self, live_risk_manager):
        """Test allocation of risk budget across positions."""
        total_risk_budget = Decimal("0.02")  # 2% portfolio risk budget
        position_count = 3
        
        risk_allocation = await live_risk_manager.allocate_risk_budget(
            total_risk_budget, position_count
        )
        
        # Should allocate risk approximately equally
        expected_per_position = total_risk_budget / position_count
        for position_risk in risk_allocation.values():
            assert abs(position_risk - expected_per_position) < Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_triggers(self, live_risk_manager):
        """Test triggers for portfolio rebalancing."""
        # Simulate portfolio drift from target allocation
        current_weights = {
            "SOL_ADDRESS": Decimal("0.45"),  # Target: 0.33, drift: +0.12
            "ETH_ADDRESS": Decimal("0.35"),  # Target: 0.33, drift: +0.02
            "LINK_ADDRESS": Decimal("0.20"),  # Target: 0.33, drift: -0.13
        }
        
        target_weights = {
            "SOL_ADDRESS": Decimal("0.33"),
            "ETH_ADDRESS": Decimal("0.33"),
            "LINK_ADDRESS": Decimal("0.33"),
        }
        
        rebalance_needed = await live_risk_manager.check_rebalancing_requirements(
            current_weights, target_weights
        )
        
        assert rebalance_needed.requires_rebalancing
        assert len(rebalance_needed.positions_to_adjust) > 0
        # SOL and LINK should need adjustment due to >20% drift
    
    @pytest.mark.asyncio
    async def test_risk_adjusted_returns_optimization(self, live_risk_manager):
        """Test optimization for risk-adjusted returns."""
        # Mock expected returns and risks for assets
        asset_data = {
            "SOL_ADDRESS": {"expected_return": Decimal("0.15"), "volatility": Decimal("0.6")},
            "ETH_ADDRESS": {"expected_return": Decimal("0.12"), "volatility": Decimal("0.4")},
            "LINK_ADDRESS": {"expected_return": Decimal("0.18"), "volatility": Decimal("0.8")},
        }
        
        optimization_result = await live_risk_manager.optimize_portfolio_allocation(asset_data)
        
        # Should prefer higher Sharpe ratio assets (return/risk)
        eth_sharpe = asset_data["ETH_ADDRESS"]["expected_return"] / asset_data["ETH_ADDRESS"]["volatility"]
        link_sharpe = asset_data["LINK_ADDRESS"]["expected_return"] / asset_data["LINK_ADDRESS"]["volatility"]
        
        # ETH has better Sharpe ratio (0.30 vs 0.225), should get higher weight
        assert optimization_result.optimal_weights["ETH_ADDRESS"] > optimization_result.optimal_weights["LINK_ADDRESS"]


class TestEmergencyRiskResponse:
    """Test emergency risk response mechanisms."""
    
    @pytest.mark.asyncio
    async def test_risk_based_position_closure(self, live_risk_manager):
        """Test automatic position closure based on risk thresholds."""
        # Simulate high-risk position
        risky_position = Mock()
        risky_position.risk_score = Decimal("0.95")  # Very high risk
        risky_position.position_id = uuid4()
        
        closure_recommendation = await live_risk_manager.assess_position_closure_need(risky_position)
        
        assert closure_recommendation.should_close
        assert closure_recommendation.urgency == "HIGH"
        assert "risk score" in closure_recommendation.reason.lower()
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_circuit_breaker(self, live_risk_manager):
        """Test portfolio-level risk circuit breaker."""
        # Simulate portfolio risk exceeding safe limits
        portfolio_risk_metrics = Mock()
        portfolio_risk_metrics.total_var = Decimal("0.12")  # 12% VaR
        portfolio_risk_metrics.stress_test_loss = Decimal("0.35")  # 35% stress loss
        
        circuit_breaker_result = await live_risk_manager.check_portfolio_circuit_breaker(
            portfolio_risk_metrics
        )
        
        assert circuit_breaker_result.should_halt_trading
        assert circuit_breaker_result.reason == "EXCESSIVE_PORTFOLIO_RISK"
    
    @pytest.mark.asyncio
    async def test_gradual_risk_reduction(self, live_risk_manager):
        """Test gradual risk reduction strategy."""
        current_risk_level = Decimal("0.85")  # 85% of risk limit
        target_risk_level = Decimal("0.60")   # Target: 60% of risk limit
        
        reduction_plan = await live_risk_manager.create_risk_reduction_plan(
            current_risk_level, target_risk_level
        )
        
        assert len(reduction_plan.steps) > 0
        assert reduction_plan.total_risk_reduction > Decimal("0")
        assert reduction_plan.estimated_duration > 0  # Should take some time


class TestRiskReportingAndMetrics:
    """Test risk reporting and metrics collection."""
    
    def test_comprehensive_risk_report(self, live_risk_manager):
        """Test generation of comprehensive risk reports."""
        report = live_risk_manager.generate_risk_report()
        
        assert "portfolio_risk_summary" in report
        assert "position_risk_breakdown" in report
        assert "concentration_analysis" in report
        assert "correlation_analysis" in report
        assert "liquidity_assessment" in report
        assert "stress_test_results" in report
        assert "risk_limit_utilization" in report
    
    def test_real_time_risk_metrics(self, live_risk_manager):
        """Test collection of real-time risk metrics."""
        metrics = live_risk_manager.get_real_time_risk_metrics()
        
        assert "current_portfolio_risk" in metrics
        assert "var_95" in metrics
        assert "max_position_concentration" in metrics
        assert "sector_exposures" in metrics
        assert "correlation_risk_score" in metrics
        assert "liquidity_score" in metrics
    
    def test_risk_alert_history(self, live_risk_manager):
        """Test tracking of risk alert history."""
        # Simulate some risk alerts
        live_risk_manager.risk_alerts.extend([
            Mock(timestamp=datetime.now() - timedelta(hours=2), severity="WARNING"),
            Mock(timestamp=datetime.now() - timedelta(hours=1), severity="CRITICAL"),
            Mock(timestamp=datetime.now() - timedelta(minutes=30), severity="WARNING"),
        ])
        
        alert_summary = live_risk_manager.get_risk_alert_summary()
        
        assert alert_summary.total_alerts == 3
        assert alert_summary.critical_alerts == 1
        assert alert_summary.warning_alerts == 2
        assert alert_summary.recent_alerts_1h == 2  # Last hour


class TestRiskManagerIntegration:
    """Test integration with other system components."""
    
    @pytest.mark.asyncio
    async def test_integration_with_emergency_stop(self, live_risk_manager):
        """Test integration with emergency stop system."""
        # Mock emergency conditions
        mock_emergency_system = Mock()
        live_risk_manager.emergency_system = mock_emergency_system
        
        # Trigger risk condition that should notify emergency system
        extreme_risk_result = await live_risk_manager.check_emergency_conditions()
        
        assert extreme_risk_result.should_stop
        assert extreme_risk_result.reason == "EXTREME_PORTFOLIO_RISK"
    
    @pytest.mark.asyncio
    async def test_integration_with_portfolio_manager(self, live_risk_manager):
        """Test integration with portfolio management."""
        # Test that risk manager can access portfolio data
        portfolio_summary = await live_risk_manager.get_portfolio_risk_summary()
        
        assert portfolio_summary.total_positions > 0
        assert portfolio_summary.total_exposure > Decimal("0")
        assert portfolio_summary.risk_score >= Decimal("0")
    
    @pytest.mark.asyncio
    async def test_risk_based_trade_approval(self, live_risk_manager):
        """Test risk-based trade approval process."""
        # Mock trade request
        trade_request = Mock()
        trade_request.token_address = "NEW_TOKEN"
        trade_request.position_size = Decimal("7500")
        trade_request.action = "BUY"
        
        approval_result = await live_risk_manager.approve_trade_request(trade_request)
        
        assert hasattr(approval_result, 'is_approved')
        assert hasattr(approval_result, 'risk_assessment')
        assert hasattr(approval_result, 'recommended_adjustments')


# Performance and Load Testing
class TestRiskManagerPerformance:
    """Test risk manager performance under load."""
    
    @pytest.mark.asyncio
    async def test_high_frequency_risk_checks(self, live_risk_manager):
        """Test performance of high-frequency risk checks."""
        start_time = datetime.now()
        
        # Perform 100 rapid risk checks
        for _ in range(100):
            await live_risk_manager.perform_quick_risk_check()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete 100 checks in under 1 second
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_risk_assessments(self, live_risk_manager):
        """Test concurrent risk assessments don't interfere."""
        # Run multiple risk assessments concurrently
        tasks = [
            live_risk_manager.assess_portfolio_risk(),
            live_risk_manager.calculate_position_correlations(),
            live_risk_manager.run_stress_tests({}),
            live_risk_manager.update_risk_metrics()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete without errors
        for result in results:
            assert not isinstance(result, Exception)