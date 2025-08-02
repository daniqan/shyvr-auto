"""
Comprehensive TDD tests for EmergencyStopController.

This module contains failing tests that define the requirements for a production-grade
emergency stop controller with immediate trading halt capabilities, global and per-mode
emergency stops, automated triggers, and manual override controls.

Following TDD methodology - these tests are written first and will fail until
the implementation is complete.

Key Requirements Tested:
- Global emergency stop toggle affecting all trading modes
- Per-mode emergency stops for individual trading modes
- Automated stop triggers (drawdown, volatility, system errors)
- Manual override controls with authentication
- Integration with existing safety systems
- Stop reason tracking and comprehensive logging
- Recovery procedures and automated reset mechanisms
- Real-time monitoring and alerting
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import List, Dict, Any, Optional

from src.safety.emergency_stop_controller import (
    EmergencyStopController, EmergencyStopConfig, EmergencyStopReason,
    EmergencyStopResult, EmergencyStopLevel, EmergencyStopType,
    StopReasonCategory, ManualOverride, RecoveryProcedure
)
from src.safety.trading_safety_manager import TradingSafetyManager, TradingSafetyConfig
from src.modes.base import ModeStatus, TradingMode
from src.portfolio.base import Portfolio, PerformanceMetrics


@pytest.fixture
def emergency_stop_config():
    """Create emergency stop configuration for testing."""
    return EmergencyStopConfig(
        # Global thresholds
        global_max_drawdown_pct=Decimal("0.15"),  # 15% global max drawdown
        global_max_daily_loss_pct=Decimal("0.05"),  # 5% daily loss limit
        global_volatility_threshold_pct=Decimal("0.20"),  # 20% volatility threshold
        
        # Per-mode thresholds
        mode_max_drawdown_pct=Decimal("0.10"),  # 10% per-mode max drawdown
        mode_max_daily_loss_pct=Decimal("0.03"),  # 3% per-mode daily loss
        
        # System health thresholds
        max_consecutive_failures=3,
        max_system_error_rate=Decimal("0.10"),  # 10% error rate
        portfolio_sync_timeout_seconds=30,
        
        # Manual override settings
        require_manual_override_auth=True,
        override_session_timeout_minutes=30,
        
        # Recovery settings
        auto_recovery_enabled=True,
        recovery_check_interval_seconds=60,
        min_recovery_wait_minutes=15,
        
        # Monitoring settings
        real_time_monitoring_enabled=True,
        monitoring_interval_seconds=5,
        alert_escalation_enabled=True,
    )


@pytest.fixture
def mock_trading_safety_manager():
    """Create mock trading safety manager."""
    manager = Mock(spec=TradingSafetyManager)
    manager.is_emergency_stopped = False
    manager.activate_emergency_stop = AsyncMock()
    manager.deactivate_emergency_stop = AsyncMock()
    manager.get_safety_metrics = Mock(return_value={
        "total_validations": 100,
        "rejections": 5,
        "approval_rate": 0.95
    })
    return manager


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio for testing."""
    portfolio = Mock(spec=Portfolio)
    portfolio.total_value = Decimal("100000")
    portfolio.initial_value = Decimal("100000")
    
    # Mock performance metrics
    performance = Mock(spec=PerformanceMetrics)
    performance.current_balance = Decimal("100000")
    performance.initial_balance = Decimal("100000")
    performance.total_pnl = Decimal("0")
    performance.max_drawdown = Decimal("0.05")  # 5% drawdown
    performance.daily_pnl = Decimal("0")
    
    # Portfolio has performance_metrics property, not get_performance_metrics method
    portfolio.performance_metrics = performance
    
    # Mock get_performance_metrics method for compatibility
    def get_performance_metrics():
        return performance
    portfolio.get_performance_metrics = get_performance_metrics
    
    return portfolio


@pytest.fixture
def emergency_stop_controller(emergency_stop_config, mock_trading_safety_manager):
    """Create emergency stop controller for testing."""
    return EmergencyStopController(
        config=emergency_stop_config,
        trading_safety_manager=mock_trading_safety_manager
    )


class TestEmergencyStopControllerInitialization:
    """Test emergency stop controller initialization and configuration."""
    
    def test_initialization_with_config(self, emergency_stop_config, mock_trading_safety_manager):
        """Test controller initializes with proper configuration."""
        controller = EmergencyStopController(
            config=emergency_stop_config,
            trading_safety_manager=mock_trading_safety_manager
        )
        
        assert controller.config == emergency_stop_config
        assert controller.trading_safety_manager == mock_trading_safety_manager
        assert not controller.is_global_emergency_stopped
        assert controller.active_mode_stops == {}
        assert controller.stop_reasons == []
        assert controller.manual_overrides == {}
    
    def test_initialization_with_default_config(self, mock_trading_safety_manager):
        """Test controller initializes with default configuration when none provided."""
        controller = EmergencyStopController(
            trading_safety_manager=mock_trading_safety_manager
        )
        
        assert isinstance(controller.config, EmergencyStopConfig)
        assert controller.config.global_max_drawdown_pct == Decimal("0.20")  # Default 20%
    
    def test_initialization_validates_dependencies(self):
        """Test initialization validates required dependencies."""
        with pytest.raises(ValueError, match="TradingSafetyManager is required"):
            EmergencyStopController(trading_safety_manager=None)


class TestGlobalEmergencyStop:
    """Test global emergency stop functionality affecting all trading modes."""
    
    @pytest.mark.asyncio
    async def test_activate_global_emergency_stop(self, emergency_stop_controller):
        """Test activation of global emergency stop."""
        reason = EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED
        message = "Global portfolio drawdown exceeded 15%"
        portfolio_value = Decimal("85000")
        
        result = await emergency_stop_controller.activate_global_emergency_stop(
            reason=reason,
            message=message,
            portfolio_value=portfolio_value,
            triggered_by="automated_trigger"
        )
        
        assert result.is_stopped
        assert result.stop_type == EmergencyStopType.GLOBAL
        assert result.reason == reason
        assert result.message == message
        assert result.portfolio_value == portfolio_value
        assert emergency_stop_controller.is_global_emergency_stopped
        assert emergency_stop_controller.global_stop_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_global_stop_affects_all_modes(self, emergency_stop_controller):
        """Test global emergency stop affects all trading modes."""
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.SYSTEM_CRITICAL_ERROR,
            message="Critical system error detected"
        )
        
        # Check that all modes are affected
        assert await emergency_stop_controller.is_mode_stopped("live_mode")
        assert await emergency_stop_controller.is_mode_stopped("simulation_mode")
        assert await emergency_stop_controller.is_mode_stopped("analysis_mode")
    
    @pytest.mark.asyncio
    async def test_global_stop_prevents_trading_decisions(self, emergency_stop_controller):
        """Test global emergency stop prevents all trading decisions."""
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED,
            message="Global daily loss limit exceeded"
        )
        
        # Attempt to validate trading decision
        can_trade = await emergency_stop_controller.can_execute_trade(
            mode_name="live_mode",
            trade_amount=Decimal("1000")
        )
        
        assert not can_trade.is_allowed
        assert can_trade.stop_reason == EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED
        assert "global" in can_trade.message.lower()
    
    @pytest.mark.asyncio
    async def test_deactivate_global_emergency_stop(self, emergency_stop_controller):
        """Test deactivation of global emergency stop."""
        # First activate
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.MANUAL_STOP,
            message="Manual emergency stop"
        )
        
        assert emergency_stop_controller.is_global_emergency_stopped
        
        # Then deactivate
        result = await emergency_stop_controller.deactivate_global_emergency_stop(
            deactivated_by="admin_user",
            reason="Issue resolved"
        )
        
        assert result.was_stopped
        assert not result.is_stopped
        assert not emergency_stop_controller.is_global_emergency_stopped
        assert emergency_stop_controller.global_stop_timestamp is None


class TestPerModeEmergencyStops:
    """Test per-mode emergency stop functionality for individual trading modes."""
    
    @pytest.mark.asyncio
    async def test_activate_mode_emergency_stop(self, emergency_stop_controller):
        """Test activation of mode-specific emergency stop."""
        mode_name = "live_mode"
        reason = EmergencyStopReason.MODE_MAX_DRAWDOWN_EXCEEDED
        message = "Live mode drawdown exceeded 10%"
        
        result = await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name=mode_name,
            reason=reason,
            message=message,
            triggered_by="automated_trigger"
        )
        
        assert result.is_stopped
        assert result.stop_type == EmergencyStopType.MODE_SPECIFIC
        assert result.mode_name == mode_name
        assert result.reason == reason
        assert emergency_stop_controller.is_mode_stopped_sync(mode_name)
    
    @pytest.mark.asyncio
    async def test_mode_stop_only_affects_specific_mode(self, emergency_stop_controller):
        """Test mode-specific emergency stop only affects target mode."""
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="live_mode",
            reason=EmergencyStopReason.MODE_CONSECUTIVE_FAILURES,
            message="Live mode consecutive failures"
        )
        
        # Live mode should be stopped
        assert await emergency_stop_controller.is_mode_stopped("live_mode")
        
        # Other modes should not be stopped
        assert not await emergency_stop_controller.is_mode_stopped("simulation_mode")
        assert not await emergency_stop_controller.is_mode_stopped("analysis_mode")
    
    @pytest.mark.asyncio
    async def test_multiple_mode_stops(self, emergency_stop_controller):
        """Test multiple mode-specific emergency stops can be active simultaneously."""
        # Stop live mode
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="live_mode",
            reason=EmergencyStopReason.MODE_MAX_DRAWDOWN_EXCEEDED,
            message="Live mode drawdown exceeded"
        )
        
        # Stop simulation mode
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="simulation_mode",
            reason=EmergencyStopReason.MODE_SYSTEM_ERROR,
            message="Simulation mode system error"
        )
        
        assert await emergency_stop_controller.is_mode_stopped("live_mode")
        assert await emergency_stop_controller.is_mode_stopped("simulation_mode")
        assert not await emergency_stop_controller.is_mode_stopped("analysis_mode")
    
    @pytest.mark.asyncio
    async def test_deactivate_mode_emergency_stop(self, emergency_stop_controller):
        """Test deactivation of mode-specific emergency stop."""
        mode_name = "live_mode"
        
        # First activate
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name=mode_name,
            reason=EmergencyStopReason.MANUAL_STOP,
            message="Manual mode stop"
        )
        
        assert emergency_stop_controller.is_mode_stopped_sync(mode_name)
        
        # Then deactivate
        result = await emergency_stop_controller.deactivate_mode_emergency_stop(
            mode_name=mode_name,
            deactivated_by="admin_user",
            reason="Issue resolved"
        )
        
        assert result.was_stopped
        assert not result.is_stopped
        assert not emergency_stop_controller.is_mode_stopped_sync(mode_name)


class TestAutomatedStopTriggers:
    """Test automated stop triggers for various conditions."""
    
    @pytest.mark.asyncio
    async def test_drawdown_trigger_global(self, emergency_stop_controller, mock_portfolio):
        """Test automated trigger on global drawdown threshold."""
        # Set portfolio to exceed global drawdown threshold (15%)
        mock_portfolio.performance_metrics.max_drawdown = Decimal("0.16")  # 16%
        mock_portfolio.total_value = Decimal("84000")  # 16% loss
        
        result = await emergency_stop_controller.check_automated_triggers(mock_portfolio)
        
        assert result.should_trigger
        assert result.trigger_type == EmergencyStopType.GLOBAL
        assert result.reason == EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED
        assert "16.00%" in result.message
    
    @pytest.mark.asyncio
    async def test_daily_loss_trigger_global(self, emergency_stop_controller, mock_portfolio):
        """Test automated trigger on global daily loss threshold."""
        # Set portfolio to exceed global daily loss threshold (5%)
        mock_portfolio.performance_metrics.daily_pnl = Decimal("-6000")  # 6% loss
        mock_portfolio.total_value = Decimal("94000")
        
        result = await emergency_stop_controller.check_automated_triggers(mock_portfolio)
        
        assert result.should_trigger
        assert result.reason == EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_volatility_trigger(self, emergency_stop_controller, mock_portfolio):
        """Test automated trigger on extreme volatility."""
        # Set up volatility detection
        emergency_stop_controller.previous_portfolio_value = Decimal("100000")
        emergency_stop_controller.last_volatility_check = datetime.now() - timedelta(minutes=1)
        
        # Current value represents 25% drop (exceeds 20% threshold)
        mock_portfolio.total_value = Decimal("75000")
        
        result = await emergency_stop_controller.check_volatility_trigger(mock_portfolio)
        
        assert result.should_trigger
        assert result.reason == EmergencyStopReason.EXTREME_VOLATILITY
        assert "25.00%" in result.message
    
    @pytest.mark.asyncio
    async def test_system_error_trigger(self, emergency_stop_controller):
        """Test automated trigger on system error rate threshold."""
        # Record multiple system errors to exceed threshold
        for i in range(15):  # 15 errors out of 100 operations = 15% error rate
            await emergency_stop_controller.record_system_error(f"Error {i}")
        
        # Set total operations count
        emergency_stop_controller.total_operations = 100
        
        result = await emergency_stop_controller.check_system_health_triggers()
        
        assert result.should_trigger
        assert result.reason == EmergencyStopReason.HIGH_SYSTEM_ERROR_RATE
        assert "15.00%" in result.message
    
    @pytest.mark.asyncio
    async def test_consecutive_failures_trigger(self, emergency_stop_controller):
        """Test automated trigger on consecutive failures."""
        # Record consecutive failures exceeding threshold
        for i in range(4):  # Exceeds max of 3
            emergency_stop_controller.record_consecutive_failure(f"Failure {i}")
        
        result = await emergency_stop_controller.check_consecutive_failures_trigger()
        
        assert result.should_trigger
        assert result.reason == EmergencyStopReason.CONSECUTIVE_FAILURES
        assert emergency_stop_controller.consecutive_failures == 4
    
    @pytest.mark.asyncio
    async def test_portfolio_sync_timeout_trigger(self, emergency_stop_controller):
        """Test automated trigger on portfolio sync timeout."""
        # Set last sync time to exceed timeout threshold
        emergency_stop_controller.last_portfolio_sync = datetime.now() - timedelta(seconds=45)
        
        result = await emergency_stop_controller.check_portfolio_sync_trigger()
        
        assert result.should_trigger
        assert result.reason == EmergencyStopReason.PORTFOLIO_SYNC_TIMEOUT


class TestManualOverrideControls:
    """Test manual override controls with authentication and audit trail."""
    
    @pytest.mark.asyncio
    async def test_manual_override_requires_authentication(self, emergency_stop_controller):
        """Test manual override requires proper authentication."""
        # Attempt override without authentication
        with pytest.raises(ValueError, match="Authentication required"):
            await emergency_stop_controller.create_manual_override(
                override_type="global_stop",
                reason="Emergency testing",
                user_id=None,  # No authentication
                auth_token=None
            )
    
    @pytest.mark.asyncio
    async def test_create_manual_override(self, emergency_stop_controller):
        """Test creation of manual override with proper authentication."""
        user_id = "admin_user"
        auth_token = "valid_auth_token"
        
        with patch.object(emergency_stop_controller, '_validate_auth_token', return_value=True):
            override = await emergency_stop_controller.create_manual_override(
                override_type="global_stop",
                reason="Emergency testing",
                user_id=user_id,
                auth_token=auth_token
            )
        
        assert override.user_id == user_id
        assert override.override_type == "global_stop"
        assert override.reason == "Emergency testing"
        assert override.is_active
        assert override.created_at is not None
    
    @pytest.mark.asyncio
    async def test_manual_override_session_timeout(self, emergency_stop_controller):
        """Test manual override session timeout."""
        user_id = "admin_user"
        
        with patch.object(emergency_stop_controller, '_validate_auth_token', return_value=True):
            override = await emergency_stop_controller.create_manual_override(
                override_type="mode_stop",
                reason="Testing timeout",
                user_id=user_id,
                auth_token="token"
            )
        
        # Simulate timeout by setting creation time in the past
        override.created_at = datetime.now() - timedelta(minutes=35)  # Exceeds 30min timeout
        
        is_valid = await emergency_stop_controller.is_override_valid(override.override_id)
        
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_override_audit_trail(self, emergency_stop_controller):
        """Test comprehensive audit trail for manual overrides."""
        user_id = "admin_user"
        
        with patch.object(emergency_stop_controller, '_validate_auth_token', return_value=True):
            override = await emergency_stop_controller.create_manual_override(
                override_type="emergency_reset",
                reason="System recovery",
                user_id=user_id,
                auth_token="token"
            )
        
        # Deactivate override
        await emergency_stop_controller.deactivate_manual_override(
            override_id=override.override_id,
            user_id=user_id,
            reason="Testing complete"
        )
        
        # Check audit trail
        audit_trail = await emergency_stop_controller.get_override_audit_trail(override.override_id)
        
        assert len(audit_trail) == 2  # Creation and deactivation
        assert audit_trail[0].action == "created"
        assert audit_trail[1].action == "deactivated"
        assert all(entry.user_id == user_id for entry in audit_trail)
    
    @pytest.mark.asyncio
    async def test_emergency_reset_override(self, emergency_stop_controller):
        """Test emergency reset override functionality."""
        # First activate emergency stop
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.MANUAL_STOP,
            message="Test emergency"
        )
        
        assert emergency_stop_controller.is_global_emergency_stopped
        
        # Create emergency reset override
        with patch.object(emergency_stop_controller, '_validate_auth_token', return_value=True):
            override = await emergency_stop_controller.create_manual_override(
                override_type="emergency_reset",
                reason="Emergency resolved",
                user_id="admin_user",
                auth_token="token"
            )
        
        # Apply the override
        result = await emergency_stop_controller.apply_emergency_reset_override(override.override_id)
        
        assert result.was_successful
        assert not emergency_stop_controller.is_global_emergency_stopped


class TestStopReasonTracking:
    """Test comprehensive stop reason tracking and logging system."""
    
    @pytest.mark.asyncio
    async def test_stop_reason_classification(self, emergency_stop_controller):
        """Test proper classification of stop reasons."""
        reason = EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED
        
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=reason,
            message="Drawdown exceeded"
        )
        
        latest_reason = emergency_stop_controller.get_latest_stop_reason()
        
        assert latest_reason.reason == reason
        assert latest_reason.category == StopReasonCategory.PORTFOLIO_PROTECTION
        assert latest_reason.severity == EmergencyStopLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_stop_reason_history(self, emergency_stop_controller):
        """Test comprehensive stop reason history tracking."""
        # Trigger multiple stops
        reasons = [
            (EmergencyStopReason.MODE_MAX_DRAWDOWN_EXCEEDED, "Mode drawdown"),
            (EmergencyStopReason.CONSECUTIVE_FAILURES, "System failures"),
            (EmergencyStopReason.MANUAL_STOP, "Manual intervention")
        ]
        
        for reason, message in reasons:
            await emergency_stop_controller.activate_mode_emergency_stop(
                mode_name="test_mode",
                reason=reason,
                message=message
            )
            
            # Deactivate to trigger next
            await emergency_stop_controller.deactivate_mode_emergency_stop(
                mode_name="test_mode",
                deactivated_by="test",
                reason="Test sequence"
            )
        
        history = emergency_stop_controller.get_stop_reason_history()
        
        assert len(history) == 3
        assert all(entry.reason in [r[0] for r in reasons] for entry in history)
    
    def test_stop_reason_aggregation(self, emergency_stop_controller):
        """Test aggregation of stop reasons for analysis."""
        # Add multiple stop reasons to history
        emergency_stop_controller.stop_reasons = [
            Mock(reason=EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED, 
                 category=StopReasonCategory.PORTFOLIO_PROTECTION),
            Mock(reason=EmergencyStopReason.CONSECUTIVE_FAILURES, 
                 category=StopReasonCategory.SYSTEM_HEALTH),
            Mock(reason=EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED, 
                 category=StopReasonCategory.PORTFOLIO_PROTECTION),
        ]
        
        aggregation = emergency_stop_controller.get_stop_reason_aggregation()
        
        assert aggregation["by_reason"][EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED] == 2
        assert aggregation["by_category"][StopReasonCategory.PORTFOLIO_PROTECTION] == 2
        assert aggregation["by_category"][StopReasonCategory.SYSTEM_HEALTH] == 1
    
    def test_stop_reason_pattern_analysis(self, emergency_stop_controller):
        """Test pattern analysis of stop reasons."""
        # Create pattern of stops with timestamps
        pattern = emergency_stop_controller.analyze_stop_patterns()
        
        assert "frequency_analysis" in pattern
        assert "time_based_patterns" in pattern
        assert "correlation_analysis" in pattern


class TestRecoveryProcedures:
    """Test automated recovery procedures and manual reset mechanisms."""
    
    @pytest.mark.asyncio
    async def test_automated_recovery_check(self, emergency_stop_controller, mock_portfolio):
        """Test automated recovery condition checking."""
        # Activate emergency stop
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED,
            message="Drawdown exceeded"
        )
        
        # Set portfolio to recovery conditions
        mock_portfolio.performance_metrics.max_drawdown = Decimal("0.08")  # Below threshold
        mock_portfolio.total_value = Decimal("92000")  # Recovering
        
        # Wait minimum recovery time
        emergency_stop_controller.global_stop_timestamp = datetime.now() - timedelta(minutes=20)
        
        recovery_check = await emergency_stop_controller.check_recovery_conditions(mock_portfolio)
        
        assert recovery_check.can_recover
        assert recovery_check.recovery_reason == "Portfolio conditions improved"
        assert recovery_check.conditions_met
    
    @pytest.mark.asyncio
    async def test_recovery_procedure_execution(self, emergency_stop_controller, mock_portfolio):
        """Test execution of automated recovery procedure."""
        # Setup emergency stop
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED,
            message="Daily loss exceeded"
        )
        
        # Setup recovery conditions
        mock_portfolio.performance_metrics.daily_pnl = Decimal("-2000")  # Below threshold
        emergency_stop_controller.global_stop_timestamp = datetime.now() - timedelta(minutes=20)
        
        result = await emergency_stop_controller.execute_recovery_procedure(mock_portfolio)
        
        assert result.was_successful
        assert not emergency_stop_controller.is_global_emergency_stopped
        assert result.recovery_method == "automated"
    
    @pytest.mark.asyncio
    async def test_manual_reset_mechanism(self, emergency_stop_controller):
        """Test manual reset mechanism with proper authorization."""
        # Activate emergency stop
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.MANUAL_STOP,
            message="Manual test stop"
        )
        
        # Create manual reset
        with patch.object(emergency_stop_controller, '_validate_auth_token', return_value=True):
            reset_result = await emergency_stop_controller.manual_reset(
                reset_type="global",
                user_id="admin_user",
                auth_token="valid_token",
                reason="Testing complete"
            )
        
        assert reset_result.was_successful
        assert not emergency_stop_controller.is_global_emergency_stopped
        assert reset_result.reset_by == "admin_user"
    
    @pytest.mark.asyncio
    async def test_recovery_validation(self, emergency_stop_controller, mock_portfolio):
        """Test comprehensive recovery validation before reactivation."""
        # Setup stopped state
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.SYSTEM_CRITICAL_ERROR,
            message="System error"
        )
        
        # Recovery validation should check multiple conditions
        validation = await emergency_stop_controller.validate_recovery_readiness(mock_portfolio)
        
        assert hasattr(validation, 'portfolio_health_ok')
        assert hasattr(validation, 'system_health_ok')
        assert hasattr(validation, 'safety_systems_ok')
        assert hasattr(validation, 'min_wait_time_elapsed')
    
    @pytest.mark.asyncio
    async def test_partial_recovery_procedures(self, emergency_stop_controller):
        """Test partial recovery procedures for mode-specific stops."""
        # Stop multiple modes
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="live_mode",
            reason=EmergencyStopReason.MODE_MAX_DRAWDOWN_EXCEEDED,
            message="Live mode drawdown"
        )
        
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="simulation_mode",
            reason=EmergencyStopReason.MODE_SYSTEM_ERROR,
            message="Simulation error"
        )
        
        # Recover only live mode
        result = await emergency_stop_controller.execute_partial_recovery(
            mode_name="live_mode",
            recovery_reason="Drawdown conditions improved"
        )
        
        assert result.was_successful
        assert not emergency_stop_controller.is_mode_stopped_sync("live_mode")
        assert emergency_stop_controller.is_mode_stopped_sync("simulation_mode")  # Still stopped


class TestIntegrationWithSafetySystems:
    """Test integration with existing trading safety systems."""
    
    @pytest.mark.asyncio
    async def test_trading_safety_manager_integration(self, emergency_stop_controller, mock_trading_safety_manager):
        """Test integration with TradingSafetyManager."""
        await emergency_stop_controller.activate_global_emergency_stop(
            reason=EmergencyStopReason.MANUAL_STOP,
            message="Integration test"
        )
        
        # Should notify trading safety manager
        mock_trading_safety_manager.activate_emergency_stop.assert_called_once()
        
        # Deactivate
        await emergency_stop_controller.deactivate_global_emergency_stop(
            deactivated_by="test",
            reason="Test complete"
        )
        
        mock_trading_safety_manager.deactivate_emergency_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mode_integration(self, emergency_stop_controller):
        """Test integration with trading modes."""
        mock_live_mode = Mock()
        mock_live_mode.emergency_stop = AsyncMock()
        mock_live_mode.resume_operations = AsyncMock()
        
        # Register mode
        emergency_stop_controller.register_trading_mode("live_mode", mock_live_mode)
        
        # Activate mode stop
        await emergency_stop_controller.activate_mode_emergency_stop(
            mode_name="live_mode",
            reason=EmergencyStopReason.MODE_CONSECUTIVE_FAILURES,
            message="Mode failures"
        )
        
        # Should call mode's emergency stop
        mock_live_mode.emergency_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_portfolio_monitor_integration(self, emergency_stop_controller, mock_portfolio):
        """Test integration with portfolio monitoring systems."""
        # Setup monitoring
        await emergency_stop_controller.start_monitoring(mock_portfolio)
        
        # Simulate portfolio update
        await emergency_stop_controller.on_portfolio_update(mock_portfolio)
        
        # Should update internal tracking
        assert emergency_stop_controller.last_portfolio_check is not None
        assert emergency_stop_controller.portfolio_history is not None


class TestRealTimeMonitoringAndAlerting:
    """Test real-time monitoring and alerting capabilities."""
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring_start(self, emergency_stop_controller, mock_portfolio):
        """Test starting real-time monitoring."""
        await emergency_stop_controller.start_real_time_monitoring(mock_portfolio)
        
        assert emergency_stop_controller.monitoring_active
        assert emergency_stop_controller.monitoring_task is not None
    
    @pytest.mark.asyncio
    async def test_monitoring_interval_adherence(self, emergency_stop_controller, mock_portfolio):
        """Test monitoring adheres to configured intervals."""
        emergency_stop_controller.config.monitoring_interval_seconds = 1
        
        start_time = datetime.now()
        await emergency_stop_controller.start_real_time_monitoring(mock_portfolio)
        
        # Wait for a few monitoring cycles
        await asyncio.sleep(2.5)
        
        await emergency_stop_controller.stop_real_time_monitoring()
        
        # Should have performed at least 2 monitoring cycles
        assert len(emergency_stop_controller.monitoring_history) >= 2
    
    @pytest.mark.asyncio
    async def test_alert_generation_and_escalation(self, emergency_stop_controller, mock_portfolio):
        """Test alert generation and escalation."""
        # Set up conditions for alert
        mock_portfolio.performance_metrics.max_drawdown = Decimal("0.12")  # Warning level
        
        alert = await emergency_stop_controller.generate_alert(mock_portfolio)
        
        assert alert.level == "WARNING"
        assert alert.message is not None
        assert alert.timestamp is not None
        
        # Test escalation
        mock_portfolio.performance_metrics.max_drawdown = Decimal("0.16")  # Critical level
        
        critical_alert = await emergency_stop_controller.generate_alert(mock_portfolio)
        
        assert critical_alert.level == "CRITICAL"
        assert critical_alert.escalation_required
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, emergency_stop_controller, mock_portfolio):
        """Test stopping real-time monitoring."""
        await emergency_stop_controller.start_real_time_monitoring(mock_portfolio)
        assert emergency_stop_controller.monitoring_active
        
        await emergency_stop_controller.stop_real_time_monitoring()
        
        assert not emergency_stop_controller.monitoring_active
        assert emergency_stop_controller.monitoring_task is None


class TestEmergencyStopMetricsAndReporting:
    """Test metrics collection and reporting for emergency stop system."""
    
    def test_metrics_collection(self, emergency_stop_controller):
        """Test comprehensive metrics collection."""
        metrics = emergency_stop_controller.get_emergency_stop_metrics()
        
        expected_metrics = [
            "total_global_stops", "total_mode_stops", "stop_reasons_by_type",
            "average_stop_duration", "recovery_success_rate", "manual_overrides_count",
            "automated_recovery_count", "monitoring_uptime", "alert_count_by_level"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
    
    def test_performance_metrics(self, emergency_stop_controller):
        """Test performance metrics for emergency stop system."""
        performance = emergency_stop_controller.get_performance_metrics()
        
        assert "average_response_time_ms" in performance
        assert "system_availability_pct" in performance
        assert "false_positive_rate" in performance
        assert "recovery_time_avg_minutes" in performance
    
    def test_health_report_generation(self, emergency_stop_controller):
        """Test generation of emergency stop system health report."""
        report = emergency_stop_controller.generate_health_report()
        
        assert "system_status" in report
        assert "recent_stops" in report
        assert "configuration_summary" in report
        assert "performance_indicators" in report
        assert "recommendations" in report
    
    def test_audit_report_generation(self, emergency_stop_controller):
        """Test generation of audit report for compliance."""
        audit_report = emergency_stop_controller.generate_audit_report()
        
        assert "emergency_stop_events" in audit_report
        assert "manual_override_events" in audit_report
        assert "recovery_events" in audit_report
        assert "configuration_changes" in audit_report
        assert "compliance_status" in audit_report


class TestConfigurationAndValidation:
    """Test configuration validation and edge cases."""
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        with pytest.raises(ValueError):
            EmergencyStopConfig(
                global_max_drawdown_pct=Decimal("1.5")  # Invalid: > 100%
            )
        
        with pytest.raises(ValueError):
            EmergencyStopConfig(
                monitoring_interval_seconds=-1  # Invalid: negative
            )
    
    def test_configuration_validation(self, emergency_stop_config):
        """Test comprehensive configuration validation."""
        validation = emergency_stop_config.validate()
        
        assert validation.is_valid
        assert len(validation.errors) == 0
        assert len(validation.warnings) >= 0
    
    def test_production_configuration(self):
        """Test production-safe configuration settings."""
        prod_config = EmergencyStopConfig.production_config()
        
        # Production should have stricter thresholds
        assert prod_config.global_max_drawdown_pct <= Decimal("0.10")  # 10% max
        assert prod_config.max_consecutive_failures <= 3
        assert prod_config.require_manual_override_auth is True
        assert prod_config.auto_recovery_enabled is False  # Manual recovery in prod


# Error handling and edge cases
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_mode_name_handling(self, emergency_stop_controller):
        """Test handling of invalid mode names."""
        with pytest.raises(ValueError, match="Invalid mode name"):
            await emergency_stop_controller.activate_mode_emergency_stop(
                mode_name="",  # Empty mode name
                reason=EmergencyStopReason.MANUAL_STOP,
                message="Test"
            )
    
    @pytest.mark.asyncio
    async def test_portfolio_none_handling(self, emergency_stop_controller):
        """Test handling when portfolio is None."""
        result = await emergency_stop_controller.check_automated_triggers(None)
        
        assert not result.should_trigger
        assert "portfolio unavailable" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_stop_activation(self, emergency_stop_controller):
        """Test handling of concurrent stop activation attempts."""
        # Start multiple activations simultaneously
        tasks = [
            emergency_stop_controller.activate_global_emergency_stop(
                reason=EmergencyStopReason.MANUAL_STOP,
                message=f"Concurrent test {i}"
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed, others should be handled gracefully
        successful_results = [r for r in results if isinstance(r, EmergencyStopResult) and r.is_stopped]
        assert len(successful_results) == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_failure_handling(self, emergency_stop_controller, mock_portfolio):
        """Test handling of monitoring system failures."""
        # Mock portfolio to raise exception
        mock_portfolio.get_performance_metrics.side_effect = Exception("Portfolio error")
        
        # Monitoring should handle errors gracefully
        await emergency_stop_controller.start_real_time_monitoring(mock_portfolio)
        await asyncio.sleep(0.1)  # Let monitoring attempt to run
        
        # System should still be operational
        assert emergency_stop_controller.monitoring_active
        assert emergency_stop_controller.monitoring_errors > 0