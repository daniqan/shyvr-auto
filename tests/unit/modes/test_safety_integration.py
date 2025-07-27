"""
Comprehensive TDD tests for Safety System Integration and Position Liquidation.

This module contains failing tests that define the requirements for integrating
all safety systems into the LiveMode.process_tick loop and implementing graceful
position liquidation when emergency conditions are triggered.

Following TDD methodology - these tests are written first and will fail until
the implementation is complete.

Key Safety Integration Requirements Tested:
- Pre-trade safety checks in process_tick loop
- Emergency stop integration with trade execution
- Graceful position liquidation triggers and execution
- Safety system coordination and prioritization
- Real-time safety monitoring during trading
- Recovery and resumption mechanisms
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import uuid4
from typing import List, Dict, Any, Optional

from src.modes.base import ModeStatus
from src.portfolio.base import (
    Portfolio, PortfolioConfig, Position, PositionStatus, PositionType,
    PerformanceMetrics, RiskMetrics
)
from src.modes.live_mode import (
    LiveMode, LiveModeConfig, EmergencyStopSystem, LiveRiskManager,
    SafetyCheckResult, EmergencyStopReason, EmergencyStopResult
)
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.dex.base import SwapResult, SwapStatus


@pytest.fixture
def mock_market_state():
    """Create mock market state for testing."""
    token = Mock()
    token.address = "TEST_TOKEN_ADDRESS"
    token.symbol = "TEST"
    
    market_state = Mock(spec=MarketState)
    market_state.token = token
    market_state.price_usd = 1.5
    market_state.volume_24h = 1000000.0
    market_state.rsi = 45.0
    market_state.volatility = 0.3
    
    return market_state


@pytest.fixture
def mock_portfolio_for_integration():
    """Create mock portfolio for integration testing."""
    portfolio = Mock(spec=Portfolio)
    portfolio.total_value = Decimal("85000")  # 15% down from initial 100k
    
    # Create positions for liquidation testing
    positions = {
        "pos_1": Mock(
            position_id=uuid4(),
            symbol="SOL/USDC",
            size=Decimal("1000"),
            entry_price=Decimal("100"),
            current_price=Decimal("80"),
            market_value=Decimal("8000"),
            unrealized_pnl=Decimal("-2000"),
            status=PositionStatus.OPEN,
            priority_score=Decimal("0.8")  # High priority for liquidation
        ),
        "pos_2": Mock(
            position_id=uuid4(),
            symbol="ETH/USDC",
            size=Decimal("500"), 
            entry_price=Decimal("2000"),
            current_price=Decimal("1800"),
            market_value=Decimal("9000"),
            unrealized_pnl=Decimal("-1000"),
            status=PositionStatus.OPEN,
            priority_score=Decimal("0.6")  # Medium priority
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
            priority_score=Decimal("0.3")  # Low priority
        )
    }
    
    portfolio.positions = positions
    
    # Mock performance metrics
    performance = Mock(spec=PerformanceMetrics)
    performance.current_balance = Decimal("85000")
    performance.initial_balance = Decimal("100000") 
    performance.total_pnl = Decimal("-15000")
    performance.max_drawdown = Decimal("0.15")  # 15% drawdown
    performance.unrealized_pnl = Decimal("-3200")
    
    portfolio.get_performance_metrics.return_value = performance
    portfolio.performance_metrics = performance
    
    return portfolio


@pytest.fixture
def enhanced_live_mode_config():
    """Create enhanced live mode configuration for safety testing."""
    return LiveModeConfig(
        initial_balance=Decimal("100000"),
        max_daily_loss_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.15"),
        emergency_drawdown_pct=Decimal("0.20"),
        
        # Enhanced safety parameters
        enable_pre_trade_safety_checks=True,
        safety_check_frequency_seconds=1,  # Very frequent for testing
        enable_emergency_liquidation=True,
        liquidation_trigger_threshold=Decimal("0.12"),  # 12% drawdown triggers liquidation
        partial_liquidation_percentage=Decimal("0.50"),  # Liquidate 50% initially
        full_liquidation_threshold=Decimal("0.18"),     # 18% drawdown triggers full liquidation
        
        # Trading controls
        max_position_size_pct=Decimal("0.10"),
        max_open_positions=15,
        enable_real_trading=False,  # Use simulation for testing
        
        # Safety coordination
        safety_system_priority_order=["emergency_stop", "risk_manager", "liquidity_check"],
        enable_safety_coordination=True,
        safety_override_threshold=Decimal("0.95"),  # 95% risk threshold for override
    )


class TestPreTradeSafetyChecks:
    """Test pre-trade safety checks in process_tick loop."""
    
    @pytest.mark.asyncio
    async def test_safety_checks_before_trade_decision(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that safety checks are performed before making trade decisions."""
        # Create live mode with mocked components
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        live_mode.status = ModeStatus.ACTIVE
        
        # Mock safety systems
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
            should_stop=False, reason=EmergencyStopReason.MANUAL_STOP
        ))
        
        risk_manager = Mock(spec=LiveRiskManager)
        risk_manager.check_daily_loss_limit = AsyncMock(return_value=Mock(is_valid=True))
        
        safety_interlocks = Mock()
        safety_interlocks.check_trading_allowed = AsyncMock(return_value=True)
        
        session_manager = Mock()
        session_manager.can_trade_now = Mock(return_value=True)
        
        live_mode.emergency_system = emergency_system
        live_mode.risk_manager = risk_manager
        live_mode.safety_interlocks = safety_interlocks
        live_mode.session_manager = session_manager
        
        # Mock trading decision and execution
        live_mode._make_trading_decision = AsyncMock(return_value=TradeAction.BUY)
        live_mode._execute_live_trade = AsyncMock(return_value=Mock(success=True))
        
        market_state = Mock()
        
        # Process tick should call safety checks first
        action = await live_mode.process_tick(market_state)
        
        # Verify safety checks were called before trading decision
        emergency_system.check_emergency_conditions.assert_called_once()
        safety_interlocks.check_trading_allowed.assert_called_once()
        risk_manager.check_daily_loss_limit.assert_called_once()
        session_manager.can_trade_now.assert_called_once()
        
        # Trading decision should have been made
        live_mode._make_trading_decision.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trade_blocked_by_emergency_stop(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that trades are blocked when emergency stop is triggered."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        live_mode.status = ModeStatus.ACTIVE
        
        # Mock emergency system triggering stop
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
            should_stop=True, 
            reason=EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED,
            message="Emergency drawdown exceeded"
        ))
        emergency_system.trigger_emergency_stop = AsyncMock()
        
        live_mode.emergency_system = emergency_system
        live_mode._make_trading_decision = AsyncMock()
        
        market_state = Mock()
        
        # Process tick should detect emergency and not make trading decision
        action = await live_mode.process_tick(market_state)
        
        assert action is None
        emergency_system.trigger_emergency_stop.assert_called_once()
        live_mode._make_trading_decision.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_trade_blocked_by_risk_limits(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that trades are blocked when risk limits are exceeded."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        live_mode.status = ModeStatus.ACTIVE
        
        # Mock systems
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
            should_stop=False, reason=EmergencyStopReason.MANUAL_STOP
        ))
        
        # Risk manager blocks trade
        risk_manager = Mock(spec=LiveRiskManager)
        risk_manager.check_daily_loss_limit = AsyncMock(return_value=Mock(
            is_valid=False, 
            reason="Daily loss limit exceeded",
            risk_level=SafetyCheckResult.DANGER
        ))
        
        safety_interlocks = Mock()
        safety_interlocks.check_trading_allowed = AsyncMock(return_value=True)
        
        live_mode.emergency_system = emergency_system
        live_mode.risk_manager = risk_manager
        live_mode.safety_interlocks = safety_interlocks
        live_mode._make_trading_decision = AsyncMock()
        
        market_state = Mock()
        
        # Process tick should detect risk violation and not trade
        action = await live_mode.process_tick(market_state)
        
        assert action is None
        live_mode._make_trading_decision.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_safety_check_frequency_respected(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that safety checks respect configured frequency limits."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        live_mode.last_safety_check = datetime.now()  # Just performed check
        
        # Safety check should be skipped due to frequency limit
        safety_result = await live_mode._perform_safety_checks()
        
        assert safety_result == SafetyCheckResult.SAFE  # Returns cached result


class TestEmergencyStopIntegration:
    """Test emergency stop integration with trade execution."""
    
    @pytest.mark.asyncio
    async def test_emergency_stop_halts_all_trading(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that emergency stop completely halts all trading activity."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        live_mode.status = ModeStatus.ACTIVE
        
        # Trigger emergency stop
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.is_emergency_stopped = True
        emergency_system.stop_reason = EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED
        emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
            should_stop=True,
            reason=EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED,
            triggered_at=datetime.now()
        ))
        
        live_mode.emergency_system = emergency_system
        
        market_state = Mock()
        
        # Multiple process_tick calls should all return None
        for _ in range(5):
            action = await live_mode.process_tick(market_state)
            assert action is None
    
    @pytest.mark.asyncio
    async def test_emergency_stop_triggers_immediate_assessment(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that emergency stop triggers immediate portfolio assessment."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.assess_liquidation_needs = AsyncMock(return_value=Mock(
            requires_liquidation=True,
            liquidation_priority_list=["pos_1", "pos_2"]
        ))
        
        live_mode.emergency_system = emergency_system
        
        # Trigger emergency stop
        await emergency_system.trigger_emergency_stop(
            EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED,
            "Test emergency",
            Decimal("75000")
        )
        
        # Should trigger immediate assessment
        emergency_system.assess_liquidation_needs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_emergency_stop_state_persistence(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that emergency stop state persists until manual reset."""
        emergency_system = EmergencyStopSystem(enhanced_live_mode_config)
        
        # Trigger emergency stop
        await emergency_system.trigger_emergency_stop(
            EmergencyStopReason.DAILY_LOSS_LIMIT,
            "Daily loss exceeded",
            Decimal("80000")
        )
        
        assert emergency_system.is_emergency_stopped
        
        # Multiple checks should maintain emergency state
        for _ in range(10):
            result = await emergency_system.check_emergency_conditions(mock_portfolio_for_integration)
            assert result.should_stop
            assert result.reason == EmergencyStopReason.DAILY_LOSS_LIMIT
        
        # State should persist until manual reset
        assert emergency_system.is_emergency_stopped


class TestGracefulPositionLiquidation:
    """Test graceful position liquidation when emergency conditions are triggered."""
    
    @pytest.mark.asyncio
    async def test_liquidation_trigger_conditions(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test conditions that trigger graceful position liquidation."""
        liquidation_system = Mock()
        
        # Test drawdown trigger
        drawdown_trigger = await liquidation_system.check_liquidation_triggers(
            portfolio_drawdown=Decimal("0.13"),  # Above 12% trigger
            daily_loss=Decimal("0.04"),
            portfolio_value=Decimal("87000")
        )
        
        assert drawdown_trigger.should_liquidate
        assert drawdown_trigger.trigger_reason == "DRAWDOWN_THRESHOLD"
        assert drawdown_trigger.liquidation_percentage == Decimal("0.50")  # Partial liquidation
    
    @pytest.mark.asyncio
    async def test_position_liquidation_priority_calculation(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test calculation of position liquidation priority."""
        liquidation_system = Mock()
        
        positions = list(mock_portfolio_for_integration.positions.values())
        
        # Calculate liquidation priority
        priority_list = await liquidation_system.calculate_liquidation_priority(positions)
        
        # Should prioritize positions with highest loss/risk
        # pos_1 has -2000 unrealized PnL and 0.8 priority score
        # pos_2 has -1000 unrealized PnL and 0.6 priority score  
        # pos_3 has -200 unrealized PnL and 0.3 priority score
        
        assert len(priority_list) == 3
        assert priority_list[0].position_id == positions[0].position_id  # Worst position first
    
    @pytest.mark.asyncio
    async def test_partial_liquidation_execution(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test execution of partial position liquidation."""
        liquidation_system = Mock()
        
        # Setup liquidation plan for 50% of positions
        liquidation_plan = Mock()
        liquidation_plan.total_positions = 3
        liquidation_plan.positions_to_liquidate = 2  # Liquidate worst 2 positions
        liquidation_plan.liquidation_percentage = Decimal("0.50")
        
        liquidation_system.create_liquidation_plan = AsyncMock(return_value=liquidation_plan)
        liquidation_system.execute_liquidation_plan = AsyncMock(return_value=Mock(
            liquidated_positions=2,
            total_proceeds=Decimal("8500"),
            execution_success=True
        ))
        
        # Execute partial liquidation
        result = await liquidation_system.execute_partial_liquidation(
            mock_portfolio_for_integration, liquidation_plan
        )
        
        assert result.execution_success
        assert result.liquidated_positions == 2
        liquidation_system.execute_liquidation_plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_liquidation_execution(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test execution of full portfolio liquidation."""
        liquidation_system = Mock()
        
        # Setup for full liquidation
        full_liquidation_plan = Mock()
        full_liquidation_plan.liquidate_all_positions = True
        full_liquidation_plan.emergency_liquidation = True
        
        liquidation_system.execute_full_liquidation = AsyncMock(return_value=Mock(
            liquidated_positions=3,
            total_proceeds=Decimal("19800"),  # Sum of all position values
            execution_success=True,
            emergency_mode=True
        ))
        
        # Execute full liquidation
        result = await liquidation_system.execute_full_liquidation(
            mock_portfolio_for_integration, full_liquidation_plan
        )
        
        assert result.execution_success
        assert result.liquidated_positions == 3
        assert result.emergency_mode
    
    @pytest.mark.asyncio
    async def test_liquidation_slippage_minimization(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test liquidation strategies to minimize slippage."""
        liquidation_system = Mock()
        
        # Test TWAP (Time-Weighted Average Price) liquidation
        twap_strategy = Mock()
        twap_strategy.time_windows = 5  # Split across 5 time windows
        twap_strategy.max_order_size_pct = Decimal("0.20")  # Max 20% per order
        
        liquidation_system.execute_twap_liquidation = AsyncMock(return_value=Mock(
            average_slippage=Decimal("0.015"),  # 1.5% average slippage
            total_execution_time=300,  # 5 minutes
            strategy_effectiveness=Decimal("0.85")  # 85% effective vs market order
        ))
        
        result = await liquidation_system.execute_twap_liquidation(
            mock_portfolio_for_integration.positions["pos_1"], twap_strategy
        )
        
        assert result.average_slippage < Decimal("0.02")  # Less than 2% slippage
        assert result.strategy_effectiveness > Decimal("0.80")  # >80% effective
    
    @pytest.mark.asyncio
    async def test_liquidation_across_multiple_dexes(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test liquidation execution across multiple DEXes for better prices."""
        liquidation_system = Mock()
        
        # Mock DEX-specific liquidation results
        dex_results = {
            "jupiter": Mock(liquidity=Decimal("50000"), slippage=Decimal("0.012")),
            "uniswap_v3": Mock(liquidity=Decimal("75000"), slippage=Decimal("0.008")),
            "hyperliquid": Mock(liquidity=Decimal("30000"), slippage=Decimal("0.020"))
        }
        
        liquidation_system.optimize_multi_dex_liquidation = AsyncMock(return_value=Mock(
            optimal_allocation={
                "jupiter": Decimal("0.30"),     # 30% to Jupiter
                "uniswap_v3": Decimal("0.60"),  # 60% to Uniswap (best liquidity/slippage)
                "hyperliquid": Decimal("0.10")  # 10% to Hyperliquid
            },
            estimated_total_slippage=Decimal("0.010"),  # 1% total slippage
            execution_plan="PARALLEL"  # Execute in parallel
        ))
        
        result = await liquidation_system.optimize_multi_dex_liquidation(dex_results)
        
        assert result.estimated_total_slippage < Decimal("0.015")
        assert result.execution_plan == "PARALLEL"
        # Should allocate more to Uniswap due to better liquidity/slippage
        assert result.optimal_allocation["uniswap_v3"] > result.optimal_allocation["jupiter"]


class TestSafetySystemCoordination:
    """Test coordination between multiple safety systems."""
    
    @pytest.mark.asyncio
    async def test_safety_system_priority_order(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that safety systems are checked in correct priority order."""
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        
        # Track call order
        call_order = []
        
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.check_emergency_conditions = AsyncMock(
            side_effect=lambda p: (call_order.append("emergency_stop"), 
                                 EmergencyStopResult(should_stop=False, reason=EmergencyStopReason.MANUAL_STOP))[1]
        )
        
        risk_manager = Mock(spec=LiveRiskManager)
        risk_manager.check_daily_loss_limit = AsyncMock(
            side_effect=lambda: (call_order.append("risk_manager"), Mock(is_valid=True))[1]
        )
        
        liquidity_checker = Mock()
        liquidity_checker.check_liquidity = AsyncMock(
            side_effect=lambda: (call_order.append("liquidity_check"), Mock(sufficient=True))[1]
        )
        
        live_mode.emergency_system = emergency_system
        live_mode.risk_manager = risk_manager
        live_mode.liquidity_checker = liquidity_checker
        
        # Execute coordinated safety check
        await live_mode._perform_coordinated_safety_checks()
        
        # Verify priority order matches configuration
        expected_order = ["emergency_stop", "risk_manager", "liquidity_check"]
        assert call_order == expected_order
    
    @pytest.mark.asyncio
    async def test_safety_system_conflict_resolution(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test resolution when safety systems have conflicting recommendations."""
        coordination_system = Mock()
        
        # Simulate conflicting recommendations
        safety_results = {
            "emergency_stop": Mock(recommendation="HALT_TRADING", priority=1, confidence=0.95),
            "risk_manager": Mock(recommendation="REDUCE_POSITIONS", priority=2, confidence=0.80),
            "liquidity_checker": Mock(recommendation="PROCEED", priority=3, confidence=0.70)
        }
        
        coordination_system.resolve_safety_conflicts = AsyncMock(return_value=Mock(
            final_recommendation="HALT_TRADING",
            resolution_reason="HIGHEST_PRIORITY_HIGH_CONFIDENCE",
            overridden_systems=["risk_manager", "liquidity_checker"]
        ))
        
        result = await coordination_system.resolve_safety_conflicts(safety_results)
        
        assert result.final_recommendation == "HALT_TRADING"
        assert "emergency_stop" not in result.overridden_systems
        assert len(result.overridden_systems) == 2
    
    @pytest.mark.asyncio
    async def test_safety_override_mechanism(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test safety override mechanism for extreme conditions."""
        override_system = Mock()
        
        # Simulate extreme conditions requiring override
        extreme_conditions = Mock()
        extreme_conditions.portfolio_risk_score = Decimal("0.97")  # 97% risk (above 95% override threshold)
        extreme_conditions.emergency_indicators = ["FLASH_CRASH", "LIQUIDITY_CRISIS"]
        
        override_system.check_override_conditions = AsyncMock(return_value=Mock(
            should_override=True,
            override_reason="EXTREME_PORTFOLIO_RISK",
            immediate_actions=["EMERGENCY_LIQUIDATION", "HALT_ALL_TRADING"],
            manual_intervention_required=True
        ))
        
        result = await override_system.check_override_conditions(extreme_conditions)
        
        assert result.should_override
        assert "EMERGENCY_LIQUIDATION" in result.immediate_actions
        assert result.manual_intervention_required


class TestRealTimeSafetyMonitoring:
    """Test real-time safety monitoring during active trading."""
    
    @pytest.mark.asyncio
    async def test_continuous_safety_monitoring_loop(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test continuous safety monitoring background loop."""
        safety_monitor = Mock()
        safety_monitor.is_monitoring = True
        safety_monitor.monitoring_frequency = 1  # 1 second intervals
        
        # Simulate monitoring loop
        safety_monitor.run_monitoring_loop = AsyncMock()
        monitoring_task = asyncio.create_task(safety_monitor.run_monitoring_loop())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        safety_monitor.run_monitoring_loop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_real_time_risk_metric_updates(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test real-time updates of risk metrics during trading."""
        risk_monitor = Mock()
        
        # Simulate metric updates
        initial_metrics = Mock(portfolio_risk=Decimal("0.45"), var_95=Decimal("0.08"))
        updated_metrics = Mock(portfolio_risk=Decimal("0.52"), var_95=Decimal("0.09"))
        
        risk_monitor.update_risk_metrics = AsyncMock(side_effect=[initial_metrics, updated_metrics])
        
        # Multiple updates
        metrics1 = await risk_monitor.update_risk_metrics(mock_portfolio_for_integration)
        metrics2 = await risk_monitor.update_risk_metrics(mock_portfolio_for_integration)
        
        assert metrics1.portfolio_risk < metrics2.portfolio_risk  # Risk increased
        assert metrics1.var_95 < metrics2.var_95  # VaR increased
    
    @pytest.mark.asyncio
    async def test_safety_alert_escalation(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test escalation of safety alerts based on severity and persistence."""
        alert_system = Mock()
        
        # Simulate persistent warning that should escalate
        warning_alert = Mock(severity="WARNING", persistence_count=3, escalation_threshold=3)
        
        alert_system.check_alert_escalation = AsyncMock(return_value=Mock(
            should_escalate=True,
            new_severity="CRITICAL",
            escalation_reason="PERSISTENT_WARNING",
            requires_immediate_action=True
        ))
        
        escalation_result = await alert_system.check_alert_escalation(warning_alert)
        
        assert escalation_result.should_escalate
        assert escalation_result.new_severity == "CRITICAL"
        assert escalation_result.requires_immediate_action


class TestRecoveryAndResumption:
    """Test recovery and resumption mechanisms after safety events."""
    
    @pytest.mark.asyncio
    async def test_emergency_stop_recovery_validation(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test validation before allowing recovery from emergency stop."""
        recovery_system = Mock()
        
        # Define recovery validation criteria
        recovery_criteria = Mock()
        recovery_criteria.portfolio_stabilized = True
        recovery_criteria.risk_metrics_normalized = True
        recovery_criteria.manual_approval_received = True
        recovery_criteria.cooling_off_period_elapsed = True
        
        recovery_system.validate_recovery_conditions = AsyncMock(return_value=Mock(
            can_recover=True,
            validation_passed=True,
            remaining_restrictions=["REDUCED_POSITION_SIZES", "ENHANCED_MONITORING"],
            full_recovery_eta=timedelta(hours=1)
        ))
        
        validation_result = await recovery_system.validate_recovery_conditions(recovery_criteria)
        
        assert validation_result.can_recover
        assert validation_result.validation_passed
        assert len(validation_result.remaining_restrictions) > 0
    
    @pytest.mark.asyncio
    async def test_gradual_trading_resumption(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test gradual resumption of trading after safety events."""
        resumption_system = Mock()
        
        # Define gradual resumption plan
        resumption_phases = [
            Mock(phase="MONITORING_ONLY", duration=timedelta(minutes=15), trading_allowed=False),
            Mock(phase="LIMITED_TRADING", duration=timedelta(minutes=30), max_position_size=Decimal("0.05")),
            Mock(phase="NORMAL_TRADING", duration=None, max_position_size=Decimal("0.10"))
        ]
        
        resumption_system.execute_gradual_resumption = AsyncMock(return_value=Mock(
            current_phase="LIMITED_TRADING",
            phase_progress=Decimal("0.6"),  # 60% through phase
            next_phase_eta=timedelta(minutes=12),
            total_resumption_success=True
        ))
        
        result = await resumption_system.execute_gradual_resumption(resumption_phases)
        
        assert result.current_phase == "LIMITED_TRADING"
        assert result.phase_progress == Decimal("0.6")
        assert result.total_resumption_success
    
    @pytest.mark.asyncio
    async def test_post_incident_analysis(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test post-incident analysis and learning."""
        analysis_system = Mock()
        
        # Mock incident data
        incident_data = Mock()
        incident_data.trigger_reason = EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED
        incident_data.portfolio_impact = Decimal("-15000")
        incident_data.positions_liquidated = 2
        incident_data.recovery_time = timedelta(hours=2)
        
        analysis_system.analyze_safety_incident = AsyncMock(return_value=Mock(
            root_causes=["EXCESSIVE_CORRELATION", "INSUFFICIENT_DIVERSIFICATION"],
            preventive_measures=["ENHANCED_CORRELATION_MONITORING", "STRICTER_POSITION_LIMITS"],
            system_improvements=["FASTER_LIQUIDATION", "BETTER_SLIPPAGE_ESTIMATION"],
            effectiveness_score=Decimal("0.75")  # 75% effective response
        ))
        
        analysis_result = await analysis_system.analyze_safety_incident(incident_data)
        
        assert len(analysis_result.root_causes) > 0
        assert len(analysis_result.preventive_measures) > 0
        assert analysis_result.effectiveness_score > Decimal("0.70")


class TestSafetySystemPerformance:
    """Test performance characteristics of safety systems."""
    
    @pytest.mark.asyncio
    async def test_safety_check_latency(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that safety checks complete within acceptable latency."""
        start_time = datetime.now()
        
        # Create live mode and perform safety checks
        live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
        live_mode.live_config = enhanced_live_mode_config
        
        # Mock fast safety systems
        emergency_system = Mock(spec=EmergencyStopSystem)
        emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
            should_stop=False, reason=EmergencyStopReason.MANUAL_STOP
        ))
        
        risk_manager = Mock(spec=LiveRiskManager)
        risk_manager.check_daily_loss_limit = AsyncMock(return_value=Mock(is_valid=True))
        
        live_mode.emergency_system = emergency_system
        live_mode.risk_manager = risk_manager
        
        # Perform multiple rapid safety checks
        for _ in range(100):
            await live_mode._perform_safety_checks()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete 100 safety checks in under 100ms
        assert duration < 0.1
    
    @pytest.mark.asyncio
    async def test_concurrent_safety_operations(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test that multiple safety operations can run concurrently without interference."""
        # Create multiple concurrent safety tasks
        tasks = []
        
        for i in range(10):
            live_mode = LiveMode(uuid4(), Mock(), mock_portfolio_for_integration)
            live_mode.live_config = enhanced_live_mode_config
            
            # Mock systems
            emergency_system = Mock(spec=EmergencyStopSystem)
            emergency_system.check_emergency_conditions = AsyncMock(return_value=EmergencyStopResult(
                should_stop=False, reason=EmergencyStopReason.MANUAL_STOP
            ))
            live_mode.emergency_system = emergency_system
            
            task = asyncio.create_task(live_mode._perform_safety_checks())
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert result == SafetyCheckResult.SAFE


# Integration test to verify all components work together
class TestComprehensiveSafetyIntegration:
    """Comprehensive integration tests for the complete safety system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_safety_workflow(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test complete end-to-end safety workflow from detection to recovery."""
        # This is a comprehensive test that would be implemented after all components are ready
        # It would test: Detection -> Emergency Stop -> Liquidation -> Recovery -> Resumption
        pass
    
    @pytest.mark.asyncio
    async def test_safety_system_stress_test(self, enhanced_live_mode_config, mock_portfolio_for_integration):
        """Test safety systems under extreme market stress conditions."""
        # This would test safety systems during simulated market crashes,
        # flash crashes, liquidity crises, etc.
        pass