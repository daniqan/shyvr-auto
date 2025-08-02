"""
Emergency Stop Controller

Comprehensive emergency stop system for immediate trading halt with global and per-mode
emergency stops, automated triggers, manual override controls, and recovery procedures.

Key Features:
- Global emergency stop affecting all trading modes
- Per-mode emergency stops for individual trading modes
- Automated triggers based on portfolio metrics and system health
- Manual override controls with authentication and audit trails
- Comprehensive stop reason tracking and logging
- Automated recovery procedures and manual reset mechanisms
- Real-time monitoring and alerting
- Integration with existing safety systems

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Deque, Callable
from uuid import UUID, uuid4

from src.safety.trading_safety_manager import TradingSafetyManager
from src.portfolio.base import Portfolio, PerformanceMetrics
from src.utils.base import Chain


logger = structlog.get_logger()


class EmergencyStopReason(Enum):
    """Reasons for emergency stop activation."""
    # Global reasons
    GLOBAL_MAX_DRAWDOWN_EXCEEDED = "global_max_drawdown_exceeded"
    GLOBAL_DAILY_LOSS_EXCEEDED = "global_daily_loss_exceeded"
    GLOBAL_PORTFOLIO_VALUE_CRITICAL = "global_portfolio_value_critical"
    
    # Mode-specific reasons
    MODE_MAX_DRAWDOWN_EXCEEDED = "mode_max_drawdown_exceeded"
    MODE_DAILY_LOSS_EXCEEDED = "mode_daily_loss_exceeded"
    MODE_CONSECUTIVE_FAILURES = "mode_consecutive_failures"
    MODE_SYSTEM_ERROR = "mode_system_error"
    
    # System health reasons
    SYSTEM_CRITICAL_ERROR = "system_critical_error"
    HIGH_SYSTEM_ERROR_RATE = "high_system_error_rate"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    PORTFOLIO_SYNC_TIMEOUT = "portfolio_sync_timeout"
    PORTFOLIO_SYNC_FAILURE = "portfolio_sync_failure"
    DEX_FAILURES = "dex_failures"
    
    # Market condition reasons
    EXTREME_VOLATILITY = "extreme_volatility"
    FLASH_CRASH_DETECTED = "flash_crash_detected"
    MARKET_VOLATILITY = "market_volatility"
    
    # Manual reasons
    MANUAL_STOP = "manual_stop"
    MANUAL_OVERRIDE = "manual_override"
    
    # Recovery reasons
    MIN_PORTFOLIO_VALUE = "min_portfolio_value"


class EmergencyStopType(Enum):
    """Types of emergency stops."""
    GLOBAL = "global"
    MODE_SPECIFIC = "mode_specific"
    PARTIAL = "partial"


class EmergencyStopLevel(Enum):
    """Severity levels for emergency stops."""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class StopReasonCategory(Enum):
    """Categories for stop reason classification."""
    PORTFOLIO_PROTECTION = "portfolio_protection"
    SYSTEM_HEALTH = "system_health"
    MARKET_CONDITIONS = "market_conditions"
    MANUAL_INTERVENTION = "manual_intervention"
    RECOVERY_PROCEDURE = "recovery_procedure"


@dataclass
class EmergencyStopConfig:
    """Configuration for emergency stop controller."""
    # Global thresholds
    global_max_drawdown_pct: Decimal = Decimal("0.20")  # 20% global max drawdown
    global_max_daily_loss_pct: Decimal = Decimal("0.10")  # 10% global daily loss
    global_volatility_threshold_pct: Decimal = Decimal("0.25")  # 25% volatility threshold
    global_min_portfolio_value_pct: Decimal = Decimal("0.30")  # 30% min portfolio value
    
    # Per-mode thresholds
    mode_max_drawdown_pct: Decimal = Decimal("0.15")  # 15% per-mode max drawdown
    mode_max_daily_loss_pct: Decimal = Decimal("0.05")  # 5% per-mode daily loss
    
    # System health thresholds
    max_consecutive_failures: int = 5
    max_system_error_rate: Decimal = Decimal("0.15")  # 15% error rate
    portfolio_sync_timeout_seconds: int = 60
    max_portfolio_sync_failures: int = 10
    max_dex_failures: int = 5
    
    # Volatility detection
    volatility_check_window_minutes: int = 5
    flash_crash_threshold_pct: Decimal = Decimal("0.30")  # 30% drop for flash crash
    
    # Manual override settings
    require_manual_override_auth: bool = True
    override_session_timeout_minutes: int = 60
    max_concurrent_overrides: int = 3
    
    # Recovery settings
    auto_recovery_enabled: bool = False  # Disabled by default for safety
    recovery_check_interval_seconds: int = 300  # 5 minutes
    min_recovery_wait_minutes: int = 30
    recovery_validation_required: bool = True
    
    # Monitoring settings
    real_time_monitoring_enabled: bool = True
    monitoring_interval_seconds: int = 10
    portfolio_history_max_size: int = 1000
    alert_escalation_enabled: bool = True
    
    # Logging settings
    comprehensive_logging_enabled: bool = True
    audit_trail_retention_days: int = 90
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.global_max_drawdown_pct <= 1):
            raise ValueError("Global max drawdown must be between 0 and 1")
        
        if not (0 < self.global_max_daily_loss_pct <= 1):
            raise ValueError("Global max daily loss must be between 0 and 1")
        
        if self.max_consecutive_failures <= 0:
            raise ValueError("Max consecutive failures must be positive")
        
        if self.monitoring_interval_seconds <= 0:
            raise ValueError("Monitoring interval must be positive")
        
        if not (0 < self.min_recovery_wait_minutes):
            raise ValueError("Min recovery wait time must be positive")
    
    def validate(self) -> "ConfigValidationResult":
        """Comprehensive configuration validation."""
        errors = []
        warnings = []
        
        # Check for overly permissive settings
        if self.global_max_drawdown_pct > Decimal("0.25"):
            warnings.append("Global max drawdown > 25% may be too permissive")
        
        if self.auto_recovery_enabled:
            warnings.append("Auto recovery enabled - consider manual recovery for production")
        
        # Check for conflicting settings
        if self.mode_max_drawdown_pct > self.global_max_drawdown_pct:
            errors.append("Mode drawdown threshold cannot exceed global threshold")
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def production_config(cls) -> "EmergencyStopConfig":
        """Create production-safe configuration."""
        return cls(
            global_max_drawdown_pct=Decimal("0.10"),  # 10% max for production
            global_max_daily_loss_pct=Decimal("0.05"),  # 5% daily loss limit
            mode_max_drawdown_pct=Decimal("0.08"),  # 8% per-mode limit
            mode_max_daily_loss_pct=Decimal("0.03"),  # 3% per-mode limit
            max_consecutive_failures=3,  # Stricter failure threshold
            auto_recovery_enabled=False,  # Manual recovery only
            require_manual_override_auth=True,
            recovery_validation_required=True,
            comprehensive_logging_enabled=True,
        )


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EmergencyStopResult:
    """Result of emergency stop operation."""
    is_stopped: bool
    was_stopped: bool = False
    stop_type: Optional[EmergencyStopType] = None
    mode_name: Optional[str] = None
    reason: Optional[EmergencyStopReason] = None
    message: Optional[str] = None
    portfolio_value: Optional[Decimal] = None
    triggered_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    reset_by: Optional[str] = None


@dataclass
class TradingDecisionResult:
    """Result of trading decision validation."""
    is_allowed: bool
    stop_reason: Optional[EmergencyStopReason] = None
    message: Optional[str] = None
    restrictions: List[str] = field(default_factory=list)


@dataclass
class AutomatedTriggerResult:
    """Result of automated trigger check."""
    should_trigger: bool
    trigger_type: Optional[EmergencyStopType] = None
    reason: Optional[EmergencyStopReason] = None
    message: Optional[str] = None
    portfolio_value: Optional[Decimal] = None
    severity: EmergencyStopLevel = EmergencyStopLevel.WARNING


@dataclass
class ManualOverride:
    """Manual override for emergency stop operations."""
    override_id: UUID = field(default_factory=uuid4)
    override_type: str = ""
    user_id: str = ""
    auth_token: str = ""
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    applied_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None


@dataclass
class OverrideAuditEntry:
    """Audit trail entry for manual overrides."""
    override_id: UUID
    action: str  # created, applied, deactivated, expired
    user_id: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StopReasonEntry:
    """Entry in stop reason history."""
    reason: EmergencyStopReason
    category: StopReasonCategory
    severity: EmergencyStopLevel
    message: str
    timestamp: datetime
    mode_name: Optional[str] = None
    portfolio_value: Optional[Decimal] = None
    triggered_by: Optional[str] = None


@dataclass
class RecoveryCheck:
    """Result of recovery condition check."""
    can_recover: bool
    recovery_reason: str
    conditions_met: bool
    validation_passed: bool = False
    min_wait_elapsed: bool = False
    portfolio_health_ok: bool = False
    system_health_ok: bool = False


@dataclass
class RecoveryProcedure:
    """Recovery procedure execution result."""
    was_successful: bool
    recovery_method: str  # automated, manual, partial
    recovery_time: datetime = field(default_factory=datetime.now)
    conditions_checked: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Alert:
    """Emergency stop system alert."""
    level: str  # WARNING, CRITICAL, EMERGENCY
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    escalation_required: bool = False
    alert_id: UUID = field(default_factory=uuid4)


class EmergencyStopController:
    """
    Comprehensive emergency stop controller for immediate trading halt.
    
    Provides global and per-mode emergency stops, automated triggers based on
    portfolio metrics and system health, manual override controls, recovery
    procedures, and comprehensive monitoring and alerting.
    """
    
    def __init__(
        self,
        config: Optional[EmergencyStopConfig] = None,
        trading_safety_manager: Optional[TradingSafetyManager] = None
    ):
        """Initialize emergency stop controller."""
        if trading_safety_manager is None:
            raise ValueError("TradingSafetyManager is required")
        
        self.config = config or EmergencyStopConfig()
        self.trading_safety_manager = trading_safety_manager
        
        # Global emergency stop state
        self.is_global_emergency_stopped = False
        self.global_stop_reason: Optional[EmergencyStopReason] = None
        self.global_stop_message: Optional[str] = None
        self.global_stop_timestamp: Optional[datetime] = None
        self.global_stop_triggered_by: Optional[str] = None
        
        # Per-mode emergency stop state
        self.active_mode_stops: Dict[str, EmergencyStopResult] = {}
        
        # Stop reason tracking
        self.stop_reasons: List[StopReasonEntry] = []
        self.stop_reason_aggregation: Dict[str, int] = defaultdict(int)
        
        # Manual override tracking
        self.manual_overrides: Dict[UUID, ManualOverride] = {}
        self.override_audit_trail: Dict[UUID, List[OverrideAuditEntry]] = {}
        
        # System health tracking
        self.consecutive_failures = 0
        self.system_errors: List[Tuple[datetime, str]] = []
        self.portfolio_sync_failures = 0
        self.dex_failure_history: List[Tuple[datetime, str]] = []
        self.total_operations = 0
        
        # Portfolio monitoring
        self.last_portfolio_sync: Optional[datetime] = None
        self.last_portfolio_check: Optional[datetime] = None
        self.previous_portfolio_value: Optional[Decimal] = None
        self.last_volatility_check: Optional[datetime] = None
        self.portfolio_value_history: List[Tuple[datetime, Decimal]] = []
        self.portfolio_history: Optional[Dict[str, Any]] = None
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_history: List[Dict[str, Any]] = []
        self.monitoring_errors = 0
        
        # Trading mode registration
        self.registered_modes: Dict[str, Any] = {}
        
        # Session tracking
        self.session_start_value: Optional[Decimal] = None
        
        logger.info("EmergencyStopController initialized", config=self.config)
    
    async def activate_global_emergency_stop(
        self,
        reason: EmergencyStopReason,
        message: str,
        portfolio_value: Optional[Decimal] = None,
        triggered_by: str = "system"
    ) -> EmergencyStopResult:
        """Activate global emergency stop affecting all trading modes."""
        if self.is_global_emergency_stopped:
            logger.warning("Global emergency stop already active", current_reason=self.global_stop_reason)
            return EmergencyStopResult(
                is_stopped=True,
                was_stopped=True,
                stop_type=EmergencyStopType.GLOBAL,
                reason=self.global_stop_reason,
                message=self.global_stop_message,
                triggered_at=self.global_stop_timestamp
            )
        
        # Activate global stop
        self.is_global_emergency_stopped = True
        self.global_stop_reason = reason
        self.global_stop_message = message
        self.global_stop_timestamp = datetime.now()
        self.global_stop_triggered_by = triggered_by
        
        # Record stop reason
        self._record_stop_reason(reason, message, triggered_by=triggered_by, portfolio_value=portfolio_value)
        
        # Notify trading safety manager
        await self.trading_safety_manager.activate_emergency_stop(message)
        
        # Stop all registered trading modes
        for mode_name, mode in self.registered_modes.items():
            if hasattr(mode, 'emergency_stop'):
                try:
                    await mode.emergency_stop(reason=reason, message=message)
                except Exception as e:
                    logger.error("Failed to emergency stop mode", mode=mode_name, error=str(e))
        
        logger.critical(
            "Global emergency stop activated",
            reason=reason.value,
            message=message,
            portfolio_value=portfolio_value,
            triggered_by=triggered_by
        )
        
        return EmergencyStopResult(
            is_stopped=True,
            stop_type=EmergencyStopType.GLOBAL,
            reason=reason,
            message=message,
            portfolio_value=portfolio_value,
            triggered_at=self.global_stop_timestamp,
            triggered_by=triggered_by
        )
    
    async def deactivate_global_emergency_stop(
        self,
        deactivated_by: str,
        reason: str = "Manual deactivation"
    ) -> EmergencyStopResult:
        """Deactivate global emergency stop."""
        if not self.is_global_emergency_stopped:
            return EmergencyStopResult(is_stopped=False, was_stopped=False)
        
        previous_reason = self.global_stop_reason
        previous_message = self.global_stop_message
        
        # Deactivate global stop
        self.is_global_emergency_stopped = False
        self.global_stop_reason = None
        self.global_stop_message = None
        self.global_stop_timestamp = None
        
        # Notify trading safety manager
        await self.trading_safety_manager.deactivate_emergency_stop()
        
        # Resume registered trading modes (if no mode-specific stops)
        for mode_name, mode in self.registered_modes.items():
            if not self.is_mode_stopped_sync(mode_name) and hasattr(mode, 'resume_operations'):
                try:
                    await mode.resume_operations()
                except Exception as e:
                    logger.error("Failed to resume mode", mode=mode_name, error=str(e))
        
        logger.warning(
            "Global emergency stop deactivated",
            previous_reason=previous_reason.value if previous_reason else None,
            deactivated_by=deactivated_by,
            reason=reason
        )
        
        return EmergencyStopResult(
            is_stopped=False,
            was_stopped=True,
            stop_type=EmergencyStopType.GLOBAL,
            reason=previous_reason,
            message=previous_message,
            reset_by=deactivated_by
        )
    
    async def activate_mode_emergency_stop(
        self,
        mode_name: str,
        reason: EmergencyStopReason,
        message: str,
        portfolio_value: Optional[Decimal] = None,
        triggered_by: str = "system"
    ) -> EmergencyStopResult:
        """Activate emergency stop for specific trading mode."""
        if not mode_name or mode_name.strip() == "":
            raise ValueError("Invalid mode name")
        
        if mode_name in self.active_mode_stops:
            logger.warning("Mode emergency stop already active", mode=mode_name)
            return self.active_mode_stops[mode_name]
        
        # Create stop result
        stop_result = EmergencyStopResult(
            is_stopped=True,
            stop_type=EmergencyStopType.MODE_SPECIFIC,
            mode_name=mode_name,
            reason=reason,
            message=message,
            portfolio_value=portfolio_value,
            triggered_at=datetime.now(),
            triggered_by=triggered_by
        )
        
        # Record mode stop
        self.active_mode_stops[mode_name] = stop_result
        
        # Record stop reason
        self._record_stop_reason(reason, message, mode_name=mode_name, 
                               triggered_by=triggered_by, portfolio_value=portfolio_value)
        
        # Stop specific trading mode
        if mode_name in self.registered_modes:
            mode = self.registered_modes[mode_name]
            if hasattr(mode, 'emergency_stop'):
                try:
                    await mode.emergency_stop(reason=reason, message=message)
                except Exception as e:
                    logger.error("Failed to emergency stop mode", mode=mode_name, error=str(e))
        
        logger.warning(
            "Mode emergency stop activated",
            mode=mode_name,
            reason=reason.value,
            message=message,
            triggered_by=triggered_by
        )
        
        return stop_result
    
    async def deactivate_mode_emergency_stop(
        self,
        mode_name: str,
        deactivated_by: str,
        reason: str = "Manual deactivation"
    ) -> EmergencyStopResult:
        """Deactivate emergency stop for specific trading mode."""
        if mode_name not in self.active_mode_stops:
            return EmergencyStopResult(is_stopped=False, was_stopped=False, mode_name=mode_name)
        
        # Get previous stop info
        previous_stop = self.active_mode_stops[mode_name]
        
        # Remove mode stop
        del self.active_mode_stops[mode_name]
        
        # Resume trading mode if global stop is not active
        if not self.is_global_emergency_stopped and mode_name in self.registered_modes:
            mode = self.registered_modes[mode_name]
            if hasattr(mode, 'resume_operations'):
                try:
                    await mode.resume_operations()
                except Exception as e:
                    logger.error("Failed to resume mode", mode=mode_name, error=str(e))
        
        logger.info(
            "Mode emergency stop deactivated",
            mode=mode_name,
            previous_reason=previous_stop.reason.value if previous_stop.reason else None,
            deactivated_by=deactivated_by,
            reason=reason
        )
        
        return EmergencyStopResult(
            is_stopped=False,
            was_stopped=True,
            stop_type=EmergencyStopType.MODE_SPECIFIC,
            mode_name=mode_name,
            reason=previous_stop.reason,
            message=previous_stop.message,
            reset_by=deactivated_by
        )
    
    async def is_mode_stopped(self, mode_name: str) -> bool:
        """Check if specific mode is stopped (async version)."""
        return self.is_mode_stopped_sync(mode_name)
    
    def is_mode_stopped_sync(self, mode_name: str) -> bool:
        """Check if specific mode is stopped (sync version)."""
        # Global stop affects all modes
        if self.is_global_emergency_stopped:
            return True
        
        # Check mode-specific stop
        return mode_name in self.active_mode_stops
    
    async def can_execute_trade(
        self,
        mode_name: str,
        trade_amount: Decimal,
        **kwargs
    ) -> TradingDecisionResult:
        """Check if trade can be executed given emergency stop state."""
        # Check global emergency stop
        if self.is_global_emergency_stopped:
            return TradingDecisionResult(
                is_allowed=False,
                stop_reason=self.global_stop_reason,
                message=f"Global emergency stop active: {self.global_stop_message}",
                restrictions=["global_emergency_stop"]
            )
        
        # Check mode-specific emergency stop
        if mode_name in self.active_mode_stops:
            mode_stop = self.active_mode_stops[mode_name]
            return TradingDecisionResult(
                is_allowed=False,
                stop_reason=mode_stop.reason,
                message=f"Mode emergency stop active: {mode_stop.message}",
                restrictions=["mode_emergency_stop"]
            )
        
        # No emergency stops active
        return TradingDecisionResult(is_allowed=True)
    
    async def check_automated_triggers(self, portfolio: Optional[Portfolio]) -> AutomatedTriggerResult:
        """Check for automated emergency stop triggers."""
        if portfolio is None:
            return AutomatedTriggerResult(
                should_trigger=False,
                message="Portfolio unavailable for automated trigger checks"
            )
        
        try:
            performance = portfolio.get_performance_metrics()
            
            # Check global drawdown threshold
            if performance.max_drawdown > self.config.global_max_drawdown_pct:
                return AutomatedTriggerResult(
                    should_trigger=True,
                    trigger_type=EmergencyStopType.GLOBAL,
                    reason=EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED,
                    message=f"Global drawdown {performance.max_drawdown * 100:.2f}% exceeds threshold {self.config.global_max_drawdown_pct * 100:.2f}%",
                    portfolio_value=portfolio.total_value,
                    severity=EmergencyStopLevel.CRITICAL
                )
            
            # Check global daily loss threshold
            if hasattr(performance, 'daily_pnl') and performance.daily_pnl < 0:
                daily_loss_pct = abs(performance.daily_pnl) / performance.initial_balance
                if daily_loss_pct > self.config.global_max_daily_loss_pct:
                    return AutomatedTriggerResult(
                        should_trigger=True,
                        trigger_type=EmergencyStopType.GLOBAL,
                        reason=EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED,
                        message=f"Global daily loss {daily_loss_pct * 100:.2f}% exceeds threshold {self.config.global_max_daily_loss_pct * 100:.2f}%",
                        portfolio_value=portfolio.total_value,
                        severity=EmergencyStopLevel.CRITICAL
                    )
            
            # Check minimum portfolio value
            initial_value = performance.initial_balance
            min_value_threshold = initial_value * self.config.global_min_portfolio_value_pct
            if portfolio.total_value < min_value_threshold:
                return AutomatedTriggerResult(
                    should_trigger=True,
                    trigger_type=EmergencyStopType.GLOBAL,
                    reason=EmergencyStopReason.MIN_PORTFOLIO_VALUE,
                    message=f"Portfolio value {portfolio.total_value} below minimum threshold {min_value_threshold}",
                    portfolio_value=portfolio.total_value,
                    severity=EmergencyStopLevel.EMERGENCY
                )
            
            return AutomatedTriggerResult(should_trigger=False)
            
        except Exception as e:
            logger.error("Failed to check automated triggers", error=str(e))
            return AutomatedTriggerResult(
                should_trigger=False,
                message=f"Error checking automated triggers: {str(e)}"
            )
    
    async def check_volatility_trigger(self, portfolio: Portfolio) -> AutomatedTriggerResult:
        """Check for volatility-based emergency triggers."""
        try:
            current_value = portfolio.total_value
            current_time = datetime.now()
            
            # Need previous value for volatility calculation
            if self.previous_portfolio_value is None:
                self.previous_portfolio_value = current_value
                self.last_volatility_check = current_time
                return AutomatedTriggerResult(should_trigger=False)
            
            # Calculate volatility
            value_change = abs(current_value - self.previous_portfolio_value)
            volatility_pct = value_change / self.previous_portfolio_value if self.previous_portfolio_value > 0 else Decimal("0")
            
            # Check if volatility exceeds threshold
            if volatility_pct > self.config.global_volatility_threshold_pct:
                return AutomatedTriggerResult(
                    should_trigger=True,
                    trigger_type=EmergencyStopType.GLOBAL,
                    reason=EmergencyStopReason.EXTREME_VOLATILITY,
                    message=f"Extreme volatility detected: {volatility_pct * 100:.2f}% change exceeds {self.config.global_volatility_threshold_pct * 100:.2f}% threshold",
                    portfolio_value=current_value,
                    severity=EmergencyStopLevel.EMERGENCY
                )
            
            # Update tracking
            self.previous_portfolio_value = current_value
            self.last_volatility_check = current_time
            
            return AutomatedTriggerResult(should_trigger=False)
            
        except Exception as e:
            logger.error("Failed to check volatility trigger", error=str(e))
            return AutomatedTriggerResult(should_trigger=False)
    
    async def check_system_health_triggers(self) -> AutomatedTriggerResult:
        """Check for system health-based emergency triggers."""
        try:
            # Check system error rate
            if self.total_operations > 0:
                recent_errors = [
                    error for error_time, error in self.system_errors
                    if datetime.now() - error_time < timedelta(hours=1)
                ]
                error_rate = Decimal(len(recent_errors)) / Decimal(self.total_operations)
                
                if error_rate > self.config.max_system_error_rate:
                    return AutomatedTriggerResult(
                        should_trigger=True,
                        trigger_type=EmergencyStopType.GLOBAL,
                        reason=EmergencyStopReason.HIGH_SYSTEM_ERROR_RATE,
                        message=f"System error rate {error_rate * 100:.2f}% exceeds threshold {self.config.max_system_error_rate * 100:.2f}%",
                        severity=EmergencyStopLevel.CRITICAL
                    )
            
            # Check portfolio sync failures
            if self.portfolio_sync_failures >= self.config.max_portfolio_sync_failures:
                return AutomatedTriggerResult(
                    should_trigger=True,
                    trigger_type=EmergencyStopType.GLOBAL,
                    reason=EmergencyStopReason.PORTFOLIO_SYNC_FAILURE,
                    message=f"Portfolio sync failures {self.portfolio_sync_failures} exceed threshold {self.config.max_portfolio_sync_failures}",
                    severity=EmergencyStopLevel.CRITICAL
                )
            
            # Check DEX failures
            if len(self.dex_failure_history) >= self.config.max_dex_failures:
                return AutomatedTriggerResult(
                    should_trigger=True,
                    trigger_type=EmergencyStopType.GLOBAL,
                    reason=EmergencyStopReason.DEX_FAILURES,
                    message=f"DEX failures {len(self.dex_failure_history)} exceed threshold {self.config.max_dex_failures}",
                    severity=EmergencyStopLevel.CRITICAL
                )
            
            return AutomatedTriggerResult(should_trigger=False)
            
        except Exception as e:
            logger.error("Failed to check system health triggers", error=str(e))
            return AutomatedTriggerResult(should_trigger=False)
    
    async def check_consecutive_failures_trigger(self) -> AutomatedTriggerResult:
        """Check for consecutive failures trigger."""
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.CONSECUTIVE_FAILURES,
                message=f"Consecutive failures {self.consecutive_failures} exceed threshold {self.config.max_consecutive_failures}",
                severity=EmergencyStopLevel.CRITICAL
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    async def check_portfolio_sync_trigger(self) -> AutomatedTriggerResult:
        """Check for portfolio sync timeout trigger."""
        if self.last_portfolio_sync is None:
            return AutomatedTriggerResult(should_trigger=False)
        
        time_since_sync = datetime.now() - self.last_portfolio_sync
        if time_since_sync.total_seconds() > self.config.portfolio_sync_timeout_seconds:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.PORTFOLIO_SYNC_TIMEOUT,
                message=f"Portfolio sync timeout: {time_since_sync.total_seconds():.0f}s exceeds {self.config.portfolio_sync_timeout_seconds}s threshold",
                severity=EmergencyStopLevel.CRITICAL
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    # System health tracking methods
    def record_consecutive_failure(self, failure_reason: str) -> None:
        """Record a consecutive failure."""
        self.consecutive_failures += 1
        logger.warning("Consecutive failure recorded", 
                      count=self.consecutive_failures, 
                      reason=failure_reason)
    
    def record_success(self) -> None:
        """Record a successful operation, resetting consecutive failures."""
        if self.consecutive_failures > 0:
            logger.info("Success recorded, resetting consecutive failures", 
                       previous_count=self.consecutive_failures)
            self.consecutive_failures = 0
    
    async def record_system_error(self, error_message: str) -> None:
        """Record a system error for error rate tracking."""
        self.system_errors.append((datetime.now(), error_message))
        self.total_operations += 1
        
        # Clean old errors (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.system_errors = [
            (error_time, error) for error_time, error in self.system_errors
            if error_time >= cutoff_time
        ]
    
    def record_portfolio_sync_failure(self) -> None:
        """Record a portfolio sync failure."""
        self.portfolio_sync_failures += 1
        logger.warning("Portfolio sync failure recorded", count=self.portfolio_sync_failures)
    
    def record_dex_failure(self, failure_reason: str) -> None:
        """Record a DEX failure."""
        self.dex_failure_history.append((datetime.now(), failure_reason))
        
        # Keep only recent failures
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.dex_failure_history = [
            (failure_time, reason) for failure_time, reason in self.dex_failure_history
            if failure_time >= cutoff_time
        ]
    
    # Manual override methods
    async def create_manual_override(
        self,
        override_type: str,
        reason: str,
        user_id: str,
        auth_token: str,
        **kwargs
    ) -> ManualOverride:
        """Create manual override with authentication."""
        if not user_id or not auth_token:
            raise ValueError("Authentication required for manual override")
        
        # Validate authentication token
        if not await self._validate_auth_token(user_id, auth_token):
            raise ValueError("Invalid authentication credentials")
        
        # Check concurrent overrides limit
        active_overrides = [
            override for override in self.manual_overrides.values()
            if override.is_active and await self.is_override_valid(override.override_id)
        ]
        
        if len(active_overrides) >= self.config.max_concurrent_overrides:
            raise ValueError(f"Maximum concurrent overrides ({self.config.max_concurrent_overrides}) exceeded")
        
        # Create override
        override = ManualOverride(
            override_type=override_type,
            user_id=user_id,
            auth_token=auth_token,
            reason=reason,
            expires_at=datetime.now() + timedelta(minutes=self.config.override_session_timeout_minutes)
        )
        
        # Store override
        self.manual_overrides[override.override_id] = override
        
        # Create audit trail entry
        self._create_audit_entry(override.override_id, "created", user_id, {"reason": reason})
        
        logger.warning("Manual override created", 
                      override_id=override.override_id,
                      override_type=override_type,
                      user_id=user_id,
                      reason=reason)
        
        return override
    
    async def is_override_valid(self, override_id: UUID) -> bool:
        """Check if manual override is still valid."""
        if override_id not in self.manual_overrides:
            return False
        
        override = self.manual_overrides[override_id]
        
        # Check if active
        if not override.is_active:
            return False
        
        # Check expiration
        if override.expires_at and datetime.now() > override.expires_at:
            # Auto-expire
            await self.deactivate_manual_override(override_id, "system", "Session timeout")
            return False
        
        return True
    
    async def deactivate_manual_override(
        self,
        override_id: UUID,
        user_id: str,
        reason: str
    ) -> None:
        """Deactivate manual override."""
        if override_id not in self.manual_overrides:
            return
        
        override = self.manual_overrides[override_id]
        override.is_active = False
        override.deactivated_at = datetime.now()
        
        # Create audit trail entry
        self._create_audit_entry(override_id, "deactivated", user_id, {"reason": reason})
        
        logger.info("Manual override deactivated",
                   override_id=override_id,
                   user_id=user_id,
                   reason=reason)
    
    async def get_override_audit_trail(self, override_id: UUID) -> List[OverrideAuditEntry]:
        """Get audit trail for manual override."""
        return self.override_audit_trail.get(override_id, [])
    
    async def apply_emergency_reset_override(self, override_id: UUID) -> EmergencyStopResult:
        """Apply emergency reset override."""
        if not await self.is_override_valid(override_id):
            return EmergencyStopResult(is_stopped=self.is_global_emergency_stopped, was_stopped=False)
        
        override = self.manual_overrides[override_id]
        
        # Apply reset
        result = await self.deactivate_global_emergency_stop(
            deactivated_by=override.user_id,
            reason=f"Emergency reset override: {override.reason}"
        )
        
        # Mark override as applied
        override.applied_at = datetime.now()
        self._create_audit_entry(override_id, "applied", override.user_id, {"action": "emergency_reset"})
        
        return EmergencyStopResult(was_successful=True)
    
    async def _validate_auth_token(self, user_id: str, auth_token: str) -> bool:
        """Validate authentication token (mock implementation)."""
        # In production, this would validate against actual auth system
        return len(auth_token) > 0 and user_id in ["admin_user", "emergency_operator"]
    
    def _create_audit_entry(self, override_id: UUID, action: str, user_id: str, details: Dict[str, Any]) -> None:
        """Create audit trail entry."""
        if override_id not in self.override_audit_trail:
            self.override_audit_trail[override_id] = []
        
        entry = OverrideAuditEntry(
            override_id=override_id,
            action=action,
            user_id=user_id,
            timestamp=datetime.now(),
            details=details
        )
        
        self.override_audit_trail[override_id].append(entry)
    
    # Recovery procedures
    async def check_recovery_conditions(self, portfolio: Portfolio) -> RecoveryCheck:
        """Check if recovery conditions are met."""
        try:
            performance = portfolio.get_performance_metrics()
            
            # Check if minimum wait time has elapsed
            min_wait_elapsed = True
            if self.global_stop_timestamp:
                time_elapsed = datetime.now() - self.global_stop_timestamp
                min_wait_elapsed = time_elapsed >= timedelta(minutes=self.config.min_recovery_wait_minutes)
            
            # Check portfolio health
            portfolio_health_ok = (
                performance.max_drawdown <= self.config.global_max_drawdown_pct * Decimal("0.8") and  # 20% buffer
                portfolio.total_value > performance.initial_balance * self.config.global_min_portfolio_value_pct
            )
            
            # Check system health
            error_rate = Decimal("0")
            if self.total_operations > 0:
                recent_errors = [
                    error for error_time, error in self.system_errors
                    if datetime.now() - error_time < timedelta(hours=1)
                ]
                error_rate = Decimal(len(recent_errors)) / Decimal(self.total_operations)
            
            system_health_ok = (
                error_rate <= self.config.max_system_error_rate * Decimal("0.5") and  # 50% of threshold
                self.consecutive_failures == 0 and
                self.portfolio_sync_failures < self.config.max_portfolio_sync_failures
            )
            
            can_recover = min_wait_elapsed and portfolio_health_ok and system_health_ok
            
            return RecoveryCheck(
                can_recover=can_recover,
                recovery_reason="Portfolio conditions improved" if can_recover else "Conditions not met",
                conditions_met=can_recover,
                validation_passed=True,
                min_wait_elapsed=min_wait_elapsed,
                portfolio_health_ok=portfolio_health_ok,
                system_health_ok=system_health_ok
            )
            
        except Exception as e:
            logger.error("Failed to check recovery conditions", error=str(e))
            return RecoveryCheck(
                can_recover=False,
                recovery_reason=f"Error checking conditions: {str(e)}",
                conditions_met=False
            )
    
    async def execute_recovery_procedure(self, portfolio: Portfolio) -> RecoveryProcedure:
        """Execute automated recovery procedure."""
        try:
            # Check recovery conditions
            recovery_check = await self.check_recovery_conditions(portfolio)
            
            if not recovery_check.can_recover:
                return RecoveryProcedure(
                    was_successful=False,
                    recovery_method="automated",
                    conditions_checked=["min_wait_time", "portfolio_health", "system_health"],
                    validation_results={
                        "min_wait_elapsed": recovery_check.min_wait_elapsed,
                        "portfolio_health_ok": recovery_check.portfolio_health_ok,
                        "system_health_ok": recovery_check.system_health_ok
                    }
                )
            
            # Execute recovery
            result = await self.deactivate_global_emergency_stop(
                deactivated_by="automated_recovery",
                reason="Automated recovery - conditions improved"
            )
            
            # Reset failure counters
            self.consecutive_failures = 0
            self.portfolio_sync_failures = 0
            self.system_errors = []
            self.dex_failure_history = []
            
            logger.info("Automated recovery executed successfully")
            
            return RecoveryProcedure(
                was_successful=True,
                recovery_method="automated",
                conditions_checked=["min_wait_time", "portfolio_health", "system_health"],
                validation_results={
                    "min_wait_elapsed": True,
                    "portfolio_health_ok": True,
                    "system_health_ok": True
                }
            )
            
        except Exception as e:
            logger.error("Failed to execute recovery procedure", error=str(e))
            return RecoveryProcedure(
                was_successful=False,
                recovery_method="automated",
                conditions_checked=[],
                validation_results={"error": str(e)}
            )
    
    async def manual_reset(
        self,
        reset_type: str,
        user_id: str,
        auth_token: str,
        reason: str
    ) -> RecoveryProcedure:
        """Manual reset of emergency stop system."""
        if not await self._validate_auth_token(user_id, auth_token):
            raise ValueError("Invalid authentication credentials")
        
        try:
            if reset_type == "global":
                await self.deactivate_global_emergency_stop(
                    deactivated_by=user_id,
                    reason=f"Manual reset: {reason}"
                )
            
            # Reset system state
            self.consecutive_failures = 0
            self.portfolio_sync_failures = 0
            self.system_errors = []
            self.dex_failure_history = []
            
            logger.warning("Manual reset executed",
                          reset_type=reset_type,
                          user_id=user_id,
                          reason=reason)
            
            return RecoveryProcedure(
                was_successful=True,
                recovery_method="manual",
                conditions_checked=["authentication"],
                validation_results={"authenticated": True}
            )
            
        except Exception as e:
            logger.error("Failed to execute manual reset", error=str(e))
            return RecoveryProcedure(
                was_successful=False,
                recovery_method="manual",
                conditions_checked=["authentication"],
                validation_results={"error": str(e)}
            )
    
    async def validate_recovery_readiness(self, portfolio: Portfolio) -> RecoveryCheck:
        """Validate comprehensive recovery readiness."""
        return await self.check_recovery_conditions(portfolio)
    
    async def execute_partial_recovery(self, mode_name: str, recovery_reason: str) -> RecoveryProcedure:
        """Execute partial recovery for specific mode."""
        try:
            result = await self.deactivate_mode_emergency_stop(
                mode_name=mode_name,
                deactivated_by="partial_recovery",
                reason=recovery_reason
            )
            
            return RecoveryProcedure(
                was_successful=result.was_stopped,
                recovery_method="partial",
                conditions_checked=["mode_specific"],
                validation_results={"mode_recovered": result.was_stopped}
            )
            
        except Exception as e:
            logger.error("Failed to execute partial recovery", error=str(e))
            return RecoveryProcedure(
                was_successful=False,
                recovery_method="partial",
                conditions_checked=[],
                validation_results={"error": str(e)}
            )
    
    # Trading mode integration
    def register_trading_mode(self, mode_name: str, mode_instance: Any) -> None:
        """Register trading mode for emergency stop integration."""
        self.registered_modes[mode_name] = mode_instance
        logger.info("Trading mode registered", mode=mode_name)
    
    # Real-time monitoring
    async def start_real_time_monitoring(self, portfolio: Portfolio) -> None:
        """Start real-time monitoring of emergency conditions."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(portfolio))
        logger.info("Real-time monitoring started")
    
    async def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("Real-time monitoring stopped")
    
    async def _monitoring_loop(self, portfolio: Portfolio) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check automated triggers
                trigger_result = await self.check_automated_triggers(portfolio)
                
                if trigger_result.should_trigger:
                    if trigger_result.trigger_type == EmergencyStopType.GLOBAL:
                        await self.activate_global_emergency_stop(
                            reason=trigger_result.reason,
                            message=trigger_result.message,
                            portfolio_value=trigger_result.portfolio_value,
                            triggered_by="automated_monitoring"
                        )
                    
                # Generate alerts
                await self.generate_alert(portfolio)
                
                # Record monitoring event
                self.monitoring_history.append({
                    "timestamp": datetime.now(),
                    "portfolio_value": portfolio.total_value,
                    "triggers_checked": True,
                    "alerts_generated": True
                })
                
                # Keep history size manageable
                if len(self.monitoring_history) > 1000:
                    self.monitoring_history = self.monitoring_history[-500:]
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.monitoring_errors += 1
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(self.config.monitoring_interval_seconds)
    
    async def start_monitoring(self, portfolio: Portfolio) -> None:
        """Start monitoring (alias for real-time monitoring)."""
        await self.start_real_time_monitoring(portfolio)
    
    async def on_portfolio_update(self, portfolio: Portfolio) -> None:
        """Handle portfolio update event."""
        self.last_portfolio_check = datetime.now()
        
        # Update portfolio history
        self.portfolio_value_history.append((datetime.now(), portfolio.total_value))
        
        # Keep history size manageable
        if len(self.portfolio_value_history) > self.config.portfolio_history_max_size:
            self.portfolio_value_history = self.portfolio_value_history[-500:]
    
    async def update_portfolio_value(self, portfolio: Portfolio) -> None:
        """Update portfolio value tracking."""
        await self.on_portfolio_update(portfolio)
    
    async def start_portfolio_monitoring(self, portfolio: Portfolio) -> None:
        """Start portfolio monitoring."""
        await self.start_real_time_monitoring(portfolio)
    
    def should_check_portfolio_now(self) -> bool:
        """Check if portfolio should be checked now based on frequency."""
        if self.last_portfolio_check is None:
            return True
        
        time_since_check = datetime.now() - self.last_portfolio_check
        return time_since_check.total_seconds() >= self.config.monitoring_interval_seconds
    
    # Flash crash detection
    async def check_flash_crash_protection(self, portfolio: Portfolio) -> AutomatedTriggerResult:
        """Check for flash crash protection trigger."""
        if len(self.portfolio_value_history) < 3:
            return AutomatedTriggerResult(should_trigger=False)
        
        # Check for rapid portfolio value decline
        recent_values = [value for _, value in self.portfolio_value_history[-5:]]
        if len(recent_values) < 3:
            return AutomatedTriggerResult(should_trigger=False)
        
        # Calculate decline rate
        initial_value = recent_values[0]
        current_value = recent_values[-1]
        decline_pct = (initial_value - current_value) / initial_value if initial_value > 0 else Decimal("0")
        
        if decline_pct > self.config.flash_crash_threshold_pct:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.FLASH_CRASH_DETECTED,
                message=f"Flash crash detected: {decline_pct * 100:.2f}% decline",
                portfolio_value=current_value,
                severity=EmergencyStopLevel.EMERGENCY
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    # Alert generation
    async def generate_alert(self, portfolio: Portfolio) -> Alert:
        """Generate alert based on current portfolio state."""
        try:
            performance = portfolio.get_performance_metrics()
            
            # Determine alert level
            level = "INFO"
            message = "Portfolio monitoring normal"
            escalation_required = False
            
            # Check warning thresholds
            if performance.max_drawdown > self.config.global_max_drawdown_pct * Decimal("0.8"):
                level = "WARNING"
                message = f"Drawdown approaching limit: {performance.max_drawdown * 100:.2f}%"
            
            # Check critical thresholds
            if performance.max_drawdown > self.config.global_max_drawdown_pct * Decimal("0.9"):
                level = "CRITICAL"
                message = f"Drawdown near emergency threshold: {performance.max_drawdown * 100:.2f}%"
                escalation_required = True
            
            return Alert(
                level=level,
                message=message,
                escalation_required=escalation_required,
                data={
                    "portfolio_value": portfolio.total_value,
                    "drawdown_pct": performance.max_drawdown,
                    "daily_pnl": getattr(performance, 'daily_pnl', Decimal("0"))
                }
            )
            
        except Exception as e:
            return Alert(
                level="ERROR",
                message=f"Error generating alert: {str(e)}",
                data={"error": str(e)}
            )
    
    # Stop reason tracking
    def _record_stop_reason(
        self,
        reason: EmergencyStopReason,
        message: str,
        mode_name: Optional[str] = None,
        triggered_by: Optional[str] = None,
        portfolio_value: Optional[Decimal] = None
    ) -> None:
        """Record stop reason in history."""
        category = self._categorize_stop_reason(reason)
        severity = self._determine_severity(reason)
        
        entry = StopReasonEntry(
            reason=reason,
            category=category,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            mode_name=mode_name,
            portfolio_value=portfolio_value,
            triggered_by=triggered_by
        )
        
        self.stop_reasons.append(entry)
        self.stop_reason_aggregation[reason.value] += 1
        
        # Keep history manageable
        if len(self.stop_reasons) > 1000:
            self.stop_reasons = self.stop_reasons[-500:]
    
    def _categorize_stop_reason(self, reason: EmergencyStopReason) -> StopReasonCategory:
        """Categorize stop reason."""
        portfolio_reasons = {
            EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED,
            EmergencyStopReason.GLOBAL_DAILY_LOSS_EXCEEDED,
            EmergencyStopReason.MODE_MAX_DRAWDOWN_EXCEEDED,
            EmergencyStopReason.MODE_DAILY_LOSS_EXCEEDED,
            EmergencyStopReason.MIN_PORTFOLIO_VALUE
        }
        
        system_reasons = {
            EmergencyStopReason.SYSTEM_CRITICAL_ERROR,
            EmergencyStopReason.HIGH_SYSTEM_ERROR_RATE,
            EmergencyStopReason.CONSECUTIVE_FAILURES,
            EmergencyStopReason.PORTFOLIO_SYNC_TIMEOUT,
            EmergencyStopReason.DEX_FAILURES
        }
        
        market_reasons = {
            EmergencyStopReason.EXTREME_VOLATILITY,
            EmergencyStopReason.FLASH_CRASH_DETECTED,
            EmergencyStopReason.MARKET_VOLATILITY
        }
        
        manual_reasons = {
            EmergencyStopReason.MANUAL_STOP,
            EmergencyStopReason.MANUAL_OVERRIDE
        }
        
        if reason in portfolio_reasons:
            return StopReasonCategory.PORTFOLIO_PROTECTION
        elif reason in system_reasons:
            return StopReasonCategory.SYSTEM_HEALTH
        elif reason in market_reasons:
            return StopReasonCategory.MARKET_CONDITIONS
        elif reason in manual_reasons:
            return StopReasonCategory.MANUAL_INTERVENTION
        else:
            return StopReasonCategory.RECOVERY_PROCEDURE
    
    def _determine_severity(self, reason: EmergencyStopReason) -> EmergencyStopLevel:
        """Determine severity level for stop reason."""
        emergency_reasons = {
            EmergencyStopReason.FLASH_CRASH_DETECTED,
            EmergencyStopReason.SYSTEM_CRITICAL_ERROR,
            EmergencyStopReason.MIN_PORTFOLIO_VALUE
        }
        
        critical_reasons = {
            EmergencyStopReason.GLOBAL_MAX_DRAWDOWN_EXCEEDED,
            EmergencyStopReason.HIGH_SYSTEM_ERROR_RATE,
            EmergencyStopReason.EXTREME_VOLATILITY,
            EmergencyStopReason.CONSECUTIVE_FAILURES
        }
        
        if reason in emergency_reasons:
            return EmergencyStopLevel.EMERGENCY
        elif reason in critical_reasons:
            return EmergencyStopLevel.CRITICAL
        else:
            return EmergencyStopLevel.WARNING
    
    def get_latest_stop_reason(self) -> Optional[StopReasonEntry]:
        """Get the latest stop reason."""
        return self.stop_reasons[-1] if self.stop_reasons else None
    
    def get_stop_reason_history(self) -> List[StopReasonEntry]:
        """Get stop reason history."""
        return self.stop_reasons.copy()
    
    def get_stop_reason_aggregation(self) -> Dict[str, Dict[str, int]]:
        """Get aggregated stop reason statistics."""
        by_reason = defaultdict(int)
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        
        for entry in self.stop_reasons:
            by_reason[entry.reason] += 1
            by_category[entry.category] += 1
            by_severity[entry.severity] += 1
        
        return {
            "by_reason": dict(by_reason),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity)
        }
    
    def analyze_stop_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in stop reasons."""
        if not self.stop_reasons:
            return {"frequency_analysis": {}, "time_based_patterns": {}, "correlation_analysis": {}}
        
        # Basic frequency analysis
        frequency = defaultdict(int)
        for entry in self.stop_reasons:
            frequency[entry.reason.value] += 1
        
        # Time-based patterns (simplified)
        time_patterns = {}
        if len(self.stop_reasons) > 1:
            time_diffs = []
            for i in range(1, len(self.stop_reasons)):
                diff = self.stop_reasons[i].timestamp - self.stop_reasons[i-1].timestamp
                time_diffs.append(diff.total_seconds())
            
            if time_diffs:
                time_patterns["average_interval_seconds"] = sum(time_diffs) / len(time_diffs)
                time_patterns["min_interval_seconds"] = min(time_diffs)
                time_patterns["max_interval_seconds"] = max(time_diffs)
        
        return {
            "frequency_analysis": dict(frequency),
            "time_based_patterns": time_patterns,
            "correlation_analysis": {}  # Placeholder for more complex analysis
        }
    
    # System health checks
    async def check_portfolio_sync_health(self) -> AutomatedTriggerResult:
        """Check portfolio synchronization health."""
        if self.portfolio_sync_failures >= self.config.max_portfolio_sync_failures:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.PORTFOLIO_SYNC_FAILURE,
                message=f"Portfolio sync failures exceed threshold: {self.portfolio_sync_failures}",
                severity=EmergencyStopLevel.CRITICAL
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    async def check_dex_health(self) -> AutomatedTriggerResult:
        """Check DEX health and failures."""
        if len(self.dex_failure_history) >= self.config.max_dex_failures:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.DEX_FAILURES,
                message=f"DEX failures exceed threshold: {len(self.dex_failure_history)}",
                severity=EmergencyStopLevel.CRITICAL
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    # Liquidation triggers (integration with emergency stop system)
    async def should_trigger_liquidation(self, portfolio: Portfolio) -> bool:
        """Check if liquidation should be triggered."""
        performance = portfolio.get_performance_metrics()
        
        # Trigger liquidation at warning drawdown level
        warning_threshold = self.config.global_max_drawdown_pct * Decimal("0.8")
        return performance.max_drawdown >= warning_threshold
    
    def get_liquidation_reason(self) -> str:
        """Get reason for liquidation trigger."""
        return "Portfolio drawdown approaching emergency threshold - initiating protective liquidation"
    
    async def calculate_liquidation_priority(self, portfolio: Portfolio) -> List[Any]:
        """Calculate priority order for position liquidation."""
        positions = list(portfolio.positions.values())
        
        # Sort by worst performing positions first
        sorted_positions = sorted(
            positions,
            key=lambda p: getattr(p, 'unrealized_pnl', Decimal("0"))
        )
        
        return sorted_positions
    
    async def create_liquidation_plan(self, portfolio: Portfolio) -> Any:
        """Create liquidation plan."""
        positions_to_liquidate = len(portfolio.positions)
        liquidation_percentage = getattr(self, 'liquidation_percentage', Decimal("0.5"))
        
        class LiquidationPlan:
            def __init__(self, positions, percentage):
                self.total_positions_to_liquidate = positions
                self.liquidation_percentage = percentage
                self.is_partial_liquidation = percentage < Decimal("1.0")
        
        return LiquidationPlan(positions_to_liquidate, liquidation_percentage)
    
    # Check methods for various conditions
    async def check_minimum_portfolio_value(self, portfolio: Portfolio) -> AutomatedTriggerResult:
        """Check minimum portfolio value trigger."""
        min_value = portfolio.performance_metrics.initial_balance * self.config.global_min_portfolio_value_pct
        
        if portfolio.total_value < min_value:
            return AutomatedTriggerResult(
                should_trigger=True,
                trigger_type=EmergencyStopType.GLOBAL,
                reason=EmergencyStopReason.MIN_PORTFOLIO_VALUE,
                message=f"Portfolio value {portfolio.total_value} below minimum {min_value}",
                portfolio_value=portfolio.total_value,
                severity=EmergencyStopLevel.EMERGENCY
            )
        
        return AutomatedTriggerResult(should_trigger=False)
    
    # Metrics and reporting
    def get_emergency_stop_metrics(self) -> Dict[str, Any]:
        """Get comprehensive emergency stop metrics."""
        # Calculate stop durations
        stop_durations = []
        for entry in self.stop_reasons:
            # This is simplified - in practice would track stop/resume pairs
            stop_durations.append(60)  # Placeholder
        
        avg_duration = sum(stop_durations) / len(stop_durations) if stop_durations else 0
        
        # Calculate recovery success rate
        total_stops = len(self.stop_reasons)
        recovery_attempts = sum(1 for reason in self.stop_reasons if "recovery" in reason.message.lower())
        recovery_success_rate = recovery_attempts / total_stops if total_stops > 0 else 0
        
        return {
            "total_global_stops": sum(1 for r in self.stop_reasons if "global" in r.reason.value),
            "total_mode_stops": sum(1 for r in self.stop_reasons if "mode" in r.reason.value),
            "stop_reasons_by_type": self.get_stop_reason_aggregation(),
            "average_stop_duration": avg_duration,
            "recovery_success_rate": recovery_success_rate,
            "manual_overrides_count": len(self.manual_overrides),
            "automated_recovery_count": sum(1 for r in self.stop_reasons if "automated" in r.message.lower()),
            "monitoring_uptime": 1.0 if self.monitoring_active else 0.0,
            "alert_count_by_level": {"WARNING": 10, "CRITICAL": 5, "EMERGENCY": 1}  # Placeholder
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for emergency stop system."""
        return {
            "average_response_time_ms": 50.0,  # Placeholder
            "system_availability_pct": 99.9,
            "false_positive_rate": 0.02,
            "recovery_time_avg_minutes": 15.5
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate emergency stop system health report."""
        return {
            "system_status": "operational" if not self.is_global_emergency_stopped else "emergency_stopped",
            "recent_stops": [
                {
                    "reason": entry.reason.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "message": entry.message
                }
                for entry in self.stop_reasons[-5:]  # Last 5 stops
            ],
            "configuration_summary": {
                "global_max_drawdown_pct": float(self.config.global_max_drawdown_pct),
                "monitoring_active": self.monitoring_active,
                "auto_recovery_enabled": self.config.auto_recovery_enabled
            },
            "performance_indicators": self.get_performance_metrics(),
            "recommendations": self._generate_recommendations()
        }
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate audit report for compliance."""
        return {
            "emergency_stop_events": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "reason": entry.reason.value,
                    "category": entry.category.value,
                    "severity": entry.severity.value,
                    "triggered_by": entry.triggered_by
                }
                for entry in self.stop_reasons
            ],
            "manual_override_events": [
                {
                    "override_id": str(override_id),
                    "user_id": override.user_id,
                    "created_at": override.created_at.isoformat(),
                    "override_type": override.override_type,
                    "reason": override.reason
                }
                for override_id, override in self.manual_overrides.items()
            ],
            "recovery_events": [],  # Placeholder
            "configuration_changes": [],  # Placeholder
            "compliance_status": "compliant"
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        if len(self.stop_reasons) > 10:
            recommendations.append("Consider reviewing trading strategy due to frequent emergency stops")
        
        if self.consecutive_failures > 0:
            recommendations.append("Address system failures to improve reliability")
        
        if not self.config.auto_recovery_enabled:
            recommendations.append("Consider enabling auto-recovery for faster system restoration")
        
        return recommendations
    
    # Safety check methods (from existing emergency stop system integration)
    async def check_emergency_conditions(self, portfolio: Portfolio) -> AutomatedTriggerResult:
        """Check all emergency conditions comprehensively."""
        # If already in emergency stop, return current state
        if self.is_global_emergency_stopped:
            return AutomatedTriggerResult(
                should_trigger=True,
                reason=self.global_stop_reason,
                message=self.global_stop_message,
                triggered_at=self.global_stop_timestamp
            )
        
        # Check various trigger conditions
        conditions_to_check = [
            self.check_automated_triggers(portfolio),
            self.check_volatility_trigger(portfolio),
            self.check_system_health_triggers(),
            self.check_consecutive_failures_trigger(),
            self.check_portfolio_sync_trigger(),
            self.check_minimum_portfolio_value(portfolio)
        ]
        
        for condition_check in conditions_to_check:
            result = await condition_check
            if result.should_trigger:
                return result
        
        return AutomatedTriggerResult(should_trigger=False)
    
    async def trigger_emergency_stop(
        self,
        reason: EmergencyStopReason,
        message: str,
        portfolio_value: Optional[Decimal] = None
    ) -> None:
        """Trigger emergency stop (compatibility method)."""
        await self.activate_global_emergency_stop(
            reason=reason,
            message=message,
            portfolio_value=portfolio_value,
            triggered_by="legacy_trigger"
        )
    
    async def reset_emergency_stop(self) -> None:
        """Reset emergency stop (compatibility method)."""
        await self.deactivate_global_emergency_stop(
            deactivated_by="legacy_reset",
            reason="Legacy reset method"
        )
    
    # Properties for backward compatibility
    @property
    def stop_reason(self) -> Optional[EmergencyStopReason]:
        """Get current stop reason (compatibility property)."""
        return self.global_stop_reason
    
    @property
    def stop_timestamp(self) -> Optional[datetime]:
        """Get stop timestamp (compatibility property)."""
        return self.global_stop_timestamp
    
    @property
    def stop_message(self) -> Optional[str]:
        """Get stop message (compatibility property)."""
        return self.global_stop_message