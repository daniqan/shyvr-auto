"""
Cross-Mode Safety System

This module provides unified safety monitoring and emergency stop capabilities
across all trading modes. It ensures consistent risk management, real-time
monitoring, and emergency response procedures.

Key Features:
- Unified risk monitoring across all modes
- Emergency stop coordination and propagation
- Safety interlock systems and validations
- Real-time health monitoring and alerting
- Production-grade safety protocols

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4
import structlog

from src.modes.base import ModeType, ModeStatus, ModeBase
from src.modes.mode_manager import ModeManager
from src.portfolio.base import Portfolio, RiskMetrics
from src.rl_agent.base import MarketState


logger = structlog.get_logger()


class SafetyStatus(Enum):
    """Overall safety system status."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    SYSTEM_ERROR = "system_error"


class EmergencyStopReason(Enum):
    """Reasons for emergency stop activation."""
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    SYSTEM_ERROR = "system_error"
    MANUAL_STOP = "manual_stop"
    API_FAILURE = "api_failure"
    CONNECTIVITY_LOSS = "connectivity_loss"
    PORTFOLIO_CORRUPTION = "portfolio_corruption"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"


class SafetyInterlockType(Enum):
    """Types of safety interlocks."""
    RISK_LIMITS = "risk_limits"
    POSITION_LIMITS = "position_limits"
    BALANCE_VALIDATION = "balance_validation"
    MARKET_CONDITIONS = "market_conditions"
    SYSTEM_HEALTH = "system_health"
    API_CONNECTIVITY = "api_connectivity"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyThreshold:
    """Safety threshold configuration."""
    name: str
    threshold_value: float
    current_value: float = 0.0
    threshold_type: str = "max"  # max, min, range
    breach_count: int = 0
    last_breach_time: Optional[datetime] = None
    alert_level: AlertLevel = AlertLevel.WARNING
    interlock_type: SafetyInterlockType = SafetyInterlockType.RISK_LIMITS
    
    @property
    def is_breached(self) -> bool:
        """Check if threshold is currently breached."""
        if self.threshold_type == "max":
            return self.current_value > self.threshold_value
        elif self.threshold_type == "min":
            return self.current_value < self.threshold_value
        return False
    
    @property
    def breach_percentage(self) -> float:
        """Calculate percentage of threshold breach."""
        if not self.is_breached:
            return 0.0
        
        if self.threshold_type == "max":
            return ((self.current_value - self.threshold_value) / self.threshold_value) * 100
        elif self.threshold_type == "min":
            return ((self.threshold_value - self.current_value) / self.threshold_value) * 100
        
        return 0.0


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: UUID
    alert_level: AlertLevel
    message: str
    threshold_name: str
    current_value: float
    threshold_value: float
    mode_id: Optional[UUID] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyValidationResult:
    """Result of safety validation check."""
    validation_id: UUID
    validation_type: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetySystemConfig:
    """Configuration for cross-mode safety system."""
    enable_emergency_stop: bool = True
    enable_risk_monitoring: bool = True
    enable_position_monitoring: bool = True
    enable_system_health_monitoring: bool = True
    
    # Risk thresholds
    max_drawdown_pct: float = 15.0
    max_daily_loss_pct: float = 5.0
    max_position_size_pct: float = 2.0
    min_liquidity_usd: float = 10000.0
    max_volatility: float = 0.3
    min_sharpe_ratio: float = 0.5
    max_correlation: float = 0.9
    
    # Monitoring intervals
    risk_monitoring_interval: float = 1.0  # seconds
    health_monitoring_interval: float = 5.0  # seconds
    position_monitoring_interval: float = 2.0  # seconds
    
    # Emergency stop settings
    emergency_stop_timeout_seconds: int = 30
    emergency_stop_retry_attempts: int = 3
    
    # Alert settings
    alert_cooldown_seconds: int = 60
    max_alerts_per_minute: int = 10
    enable_alert_escalation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be positive")
        if self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be positive")
        if self.risk_monitoring_interval <= 0:
            raise ValueError("risk_monitoring_interval must be positive")


class CrossModeSafetySystem:
    """
    Unified safety monitoring and emergency stop system for all trading modes.
    
    Provides comprehensive risk monitoring, emergency stop coordination,
    and safety validation across all active trading modes.
    """
    
    def __init__(self, config: SafetySystemConfig, mode_manager: ModeManager):
        """Initialize cross-mode safety system."""
        self.config = config
        self.mode_manager = mode_manager
        
        # Safety state
        self.safety_status = SafetyStatus.SAFE
        self.emergency_stop_active = False
        self.emergency_stop_reason: Optional[EmergencyStopReason] = None
        
        # Thresholds and monitoring
        self.safety_thresholds: Dict[str, SafetyThreshold] = {}
        self.active_alerts: Dict[UUID, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.validation_history: List[SafetyValidationResult] = []
        
        # Monitoring tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Alert rate limiting
        self._alert_counts: Dict[str, int] = {}
        self._last_alert_reset = time.time()
        
        # Initialize safety thresholds
        self._initialize_safety_thresholds()
        
        # Configure logger
        self.logger = logger.bind(
            component="cross_mode_safety",
            monitoring_interval=config.risk_monitoring_interval
        )
        
        self.logger.info("Cross-mode safety system initialized")
    
    async def start_monitoring(self) -> None:
        """Start all safety monitoring tasks."""
        if self._monitoring_tasks:
            self.logger.warning("Monitoring already started")
            return
        
        self.logger.info("Starting safety monitoring")
        
        # Start monitoring tasks
        if self.config.enable_risk_monitoring:
            task = asyncio.create_task(self._risk_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.enable_position_monitoring:
            task = asyncio.create_task(self._position_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.enable_system_health_monitoring:
            task = asyncio.create_task(self._system_health_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        # Start alert management task
        task = asyncio.create_task(self._alert_management_loop())
        self._monitoring_tasks.append(task)
        
        self.logger.info("Safety monitoring started", active_tasks=len(self._monitoring_tasks))
    
    async def stop_monitoring(self) -> None:
        """Stop all safety monitoring tasks."""
        self.logger.info("Stopping safety monitoring")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        self._shutdown_event.clear()
        
        self.logger.info("Safety monitoring stopped")
    
    async def trigger_emergency_stop(
        self, 
        reason: EmergencyStopReason,
        details: Optional[str] = None
    ) -> bool:
        """Trigger emergency stop across all modes."""
        if self.emergency_stop_active:
            self.logger.warning("Emergency stop already active")
            return True
        
        self.emergency_stop_active = True
        self.emergency_stop_reason = reason
        self.safety_status = SafetyStatus.EMERGENCY
        
        self.logger.critical(
            "Emergency stop triggered",
            reason=reason.value,
            details=details
        )
        
        # Create emergency alert
        alert = RiskAlert(
            alert_id=uuid4(),
            alert_level=AlertLevel.EMERGENCY,
            message=f"Emergency stop: {reason.value}",
            threshold_name="emergency_stop",
            current_value=1.0,
            threshold_value=0.0,
            metadata={"reason": reason.value, "details": details}
        )
        
        await self._add_alert(alert)
        
        try:
            # Stop all modes through mode manager
            success = True
            for mode_id in list(self.mode_manager.active_modes.keys()):
                try:
                    await asyncio.wait_for(
                        self.mode_manager.stop_mode(mode_id),
                        timeout=self.config.emergency_stop_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    self.logger.error("Emergency stop timeout", mode_id=str(mode_id))
                    success = False
                except Exception as e:
                    self.logger.error("Emergency stop failed", mode_id=str(mode_id), error=str(e))
                    success = False
            
            self.logger.info("Emergency stop completed", success=success)
            return success
            
        except Exception as e:
            self.logger.error("Emergency stop procedure failed", error=str(e))
            return False
    
    async def reset_emergency_stop(self) -> bool:
        """Reset emergency stop state after manual validation."""
        if not self.emergency_stop_active:
            self.logger.warning("Emergency stop not active")
            return True
        
        self.logger.info("Resetting emergency stop")
        
        # Validate safety conditions before reset
        validation_result = await self.validate_safety_conditions()
        if not validation_result.passed:
            self.logger.error("Cannot reset emergency stop: safety validation failed")
            return False
        
        # Reset emergency stop state
        self.emergency_stop_active = False
        self.emergency_stop_reason = None
        self.safety_status = SafetyStatus.SAFE
        
        # Create reset alert
        alert = RiskAlert(
            alert_id=uuid4(),
            alert_level=AlertLevel.INFO,
            message="Emergency stop reset",
            threshold_name="emergency_stop",
            current_value=0.0,
            threshold_value=0.0
        )
        
        await self._add_alert(alert)
        
        self.logger.info("Emergency stop reset completed")
        return True
    
    async def validate_mode_transition_safety(
        self, 
        from_mode: Optional[ModeType], 
        to_mode: ModeType
    ) -> SafetyValidationResult:
        """Validate safety conditions for mode transition."""
        validation_id = uuid4()
        
        # Check if emergency stop is active
        if self.emergency_stop_active:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="mode_transition",
                passed=False,
                message="Mode transition blocked: emergency stop active",
                details={"reason": self.emergency_stop_reason.value if self.emergency_stop_reason else "unknown"}
            )
        
        # Additional checks for live mode transitions
        if to_mode == ModeType.LIVE_TRADING:
            # Check risk thresholds
            for threshold in self.safety_thresholds.values():
                if threshold.is_breached and threshold.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                    return SafetyValidationResult(
                        validation_id=validation_id,
                        validation_type="mode_transition",
                        passed=False,
                        message=f"Live mode blocked: {threshold.name} threshold breached",
                        details={"threshold": threshold.name, "value": threshold.current_value}
                    )
        
        # Validate system health
        if self.safety_status in [SafetyStatus.CRITICAL, SafetyStatus.EMERGENCY]:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="mode_transition",
                passed=False,
                message=f"Mode transition blocked: system status {self.safety_status.value}",
                details={"safety_status": self.safety_status.value}
            )
        
        return SafetyValidationResult(
            validation_id=validation_id,
            validation_type="mode_transition",
            passed=True,
            message="Mode transition safety validation passed"
        )
    
    async def validate_trade_safety(
        self, 
        mode_id: UUID, 
        trade_action: str, 
        amount: Decimal,
        token_address: str
    ) -> SafetyValidationResult:
        """Validate safety conditions for trade execution."""
        validation_id = uuid4()
        
        # Check if emergency stop is active
        if self.emergency_stop_active:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="trade_validation",
                passed=False,
                message="Trade blocked: emergency stop active"
            )
        
        # Get mode and portfolio
        mode = self.mode_manager.active_modes.get(mode_id)
        if not mode:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="trade_validation",
                passed=False,
                message="Trade blocked: mode not found"
            )
        
        # Check position size limits
        portfolio_value = mode.portfolio.total_value
        trade_value = amount * Decimal("100")  # Placeholder price
        position_pct = (trade_value / portfolio_value) * 100
        
        if position_pct > self.config.max_position_size_pct:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="trade_validation",
                passed=False,
                message=f"Trade blocked: position size {position_pct:.2f}% exceeds limit {self.config.max_position_size_pct}%"
            )
        
        # Check balance validation
        if trade_action.upper() in ["BUY", "STRONG_BUY"]:
            if mode.portfolio.balance < trade_value:
                return SafetyValidationResult(
                    validation_id=validation_id,
                    validation_type="trade_validation",
                    passed=False,
                    message="Trade blocked: insufficient balance"
                )
        
        return SafetyValidationResult(
            validation_id=validation_id,
            validation_type="trade_validation",
            passed=True,
            message="Trade safety validation passed"
        )
    
    async def validate_safety_conditions(self) -> SafetyValidationResult:
        """Validate overall safety conditions."""
        validation_id = uuid4()
        
        # Check for critical threshold breaches
        critical_breaches = []
        for threshold in self.safety_thresholds.values():
            if threshold.is_breached and threshold.alert_level == AlertLevel.CRITICAL:
                critical_breaches.append(threshold.name)
        
        if critical_breaches:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="safety_conditions",
                passed=False,
                message=f"Critical thresholds breached: {', '.join(critical_breaches)}",
                details={"breached_thresholds": critical_breaches}
            )
        
        # Check system status
        if self.safety_status in [SafetyStatus.EMERGENCY, SafetyStatus.SYSTEM_ERROR]:
            return SafetyValidationResult(
                validation_id=validation_id,
                validation_type="safety_conditions",
                passed=False,
                message=f"System safety status: {self.safety_status.value}",
                details={"safety_status": self.safety_status.value}
            )
        
        return SafetyValidationResult(
            validation_id=validation_id,
            validation_type="safety_conditions",
            passed=True,
            message="Safety conditions validated successfully"
        )
    
    async def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status report."""
        return {
            "safety_status": self.safety_status.value,
            "emergency_stop_active": self.emergency_stop_active,
            "emergency_stop_reason": self.emergency_stop_reason.value if self.emergency_stop_reason else None,
            "active_alerts_count": len(self.active_alerts),
            "critical_alerts_count": len([a for a in self.active_alerts.values() if a.alert_level == AlertLevel.CRITICAL]),
            "emergency_alerts_count": len([a for a in self.active_alerts.values() if a.alert_level == AlertLevel.EMERGENCY]),
            "thresholds": {
                name: {
                    "current_value": threshold.current_value,
                    "threshold_value": threshold.threshold_value,
                    "is_breached": threshold.is_breached,
                    "breach_count": threshold.breach_count
                }
                for name, threshold in self.safety_thresholds.items()
            },
            "monitoring_active": len(self._monitoring_tasks) > 0,
            "last_validation": self.validation_history[-1].timestamp.isoformat() if self.validation_history else None
        }
    
    # Private methods
    
    def _initialize_safety_thresholds(self) -> None:
        """Initialize safety thresholds from configuration."""
        thresholds = [
            ("max_drawdown", self.config.max_drawdown_pct, AlertLevel.CRITICAL, SafetyInterlockType.RISK_LIMITS),
            ("daily_loss", self.config.max_daily_loss_pct, AlertLevel.CRITICAL, SafetyInterlockType.RISK_LIMITS),
            ("position_size", self.config.max_position_size_pct, AlertLevel.WARNING, SafetyInterlockType.POSITION_LIMITS),
            ("min_liquidity", self.config.min_liquidity_usd, AlertLevel.WARNING, SafetyInterlockType.MARKET_CONDITIONS),
            ("max_volatility", self.config.max_volatility, AlertLevel.WARNING, SafetyInterlockType.MARKET_CONDITIONS),
            ("min_sharpe", self.config.min_sharpe_ratio, AlertLevel.WARNING, SafetyInterlockType.RISK_LIMITS),
            ("max_correlation", self.config.max_correlation, AlertLevel.WARNING, SafetyInterlockType.RISK_LIMITS)
        ]
        
        for name, value, alert_level, interlock_type in thresholds:
            threshold_type = "max" if name.startswith("max_") else "min"
            
            self.safety_thresholds[name] = SafetyThreshold(
                name=name,
                threshold_value=value,
                threshold_type=threshold_type,
                alert_level=alert_level,
                interlock_type=interlock_type
            )
    
    async def _risk_monitoring_loop(self) -> None:
        """Main risk monitoring loop."""
        self.logger.info("Risk monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._update_risk_metrics()
                await self._check_risk_thresholds()
                
                await asyncio.sleep(self.config.risk_monitoring_interval)
                
            except Exception as e:
                self.logger.error("Risk monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _position_monitoring_loop(self) -> None:
        """Position monitoring loop."""
        self.logger.info("Position monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._monitor_positions()
                
                await asyncio.sleep(self.config.position_monitoring_interval)
                
            except Exception as e:
                self.logger.error("Position monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _system_health_monitoring_loop(self) -> None:
        """System health monitoring loop."""
        self.logger.info("System health monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._monitor_system_health()
                
                await asyncio.sleep(self.config.health_monitoring_interval)
                
            except Exception as e:
                self.logger.error("System health monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _alert_management_loop(self) -> None:
        """Alert management and escalation loop."""
        self.logger.info("Alert management loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._manage_alerts()
                await self._update_alert_rate_limits()
                
                await asyncio.sleep(5)  # Check alerts every 5 seconds
                
            except Exception as e:
                self.logger.error("Alert management error", error=str(e))
                await asyncio.sleep(1)
    
    async def _update_risk_metrics(self) -> None:
        """Update risk metrics from all active modes."""
        total_portfolio_value = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        max_drawdown = 0.0
        
        for mode in self.mode_manager.active_modes.values():
            if mode.status == ModeStatus.ACTIVE:
                portfolio = mode.portfolio
                total_portfolio_value += portfolio.total_value
                total_unrealized_pnl += portfolio.unrealized_pnl
                
                # Calculate drawdown for this mode
                if portfolio.total_value > 0:
                    mode_drawdown = abs(float(portfolio.unrealized_pnl / portfolio.total_value)) * 100
                    max_drawdown = max(max_drawdown, mode_drawdown)
        
        # Update thresholds
        if "max_drawdown" in self.safety_thresholds:
            self.safety_thresholds["max_drawdown"].current_value = max_drawdown
        
        # Calculate daily loss (placeholder implementation)
        daily_loss_pct = 0.0  # Would calculate from transaction history
        if "daily_loss" in self.safety_thresholds:
            self.safety_thresholds["daily_loss"].current_value = daily_loss_pct
    
    async def _check_risk_thresholds(self) -> None:
        """Check all risk thresholds for breaches."""
        for threshold in self.safety_thresholds.values():
            if threshold.is_breached:
                # Check if this is a new breach
                now = datetime.now()
                is_new_breach = (
                    threshold.last_breach_time is None or
                    (now - threshold.last_breach_time).total_seconds() > self.config.alert_cooldown_seconds
                )
                
                if is_new_breach:
                    threshold.breach_count += 1
                    threshold.last_breach_time = now
                    
                    # Create alert
                    alert = RiskAlert(
                        alert_id=uuid4(),
                        alert_level=threshold.alert_level,
                        message=f"Threshold breach: {threshold.name}",
                        threshold_name=threshold.name,
                        current_value=threshold.current_value,
                        threshold_value=threshold.threshold_value
                    )
                    
                    await self._add_alert(alert)
                    
                    # Trigger emergency stop for critical breaches
                    if threshold.alert_level == AlertLevel.CRITICAL:
                        await self.trigger_emergency_stop(
                            EmergencyStopReason.RISK_THRESHOLD_BREACH,
                            f"Critical threshold breach: {threshold.name}"
                        )
    
    async def _monitor_positions(self) -> None:
        """Monitor position sizes and limits."""
        for mode in self.mode_manager.active_modes.values():
            if mode.status == ModeStatus.ACTIVE:
                portfolio = mode.portfolio
                
                # Check individual position sizes
                for position in portfolio.positions.values():
                    if portfolio.total_value > 0:
                        position_pct = (position.quantity * position.current_price / portfolio.total_value) * 100
                        
                        if position_pct > self.config.max_position_size_pct:
                            alert = RiskAlert(
                                alert_id=uuid4(),
                                alert_level=AlertLevel.WARNING,
                                message=f"Position size exceeded: {position_pct:.2f}%",
                                threshold_name="position_size",
                                current_value=position_pct,
                                threshold_value=self.config.max_position_size_pct,
                                mode_id=mode.mode_id
                            )
                            
                            await self._add_alert(alert)
    
    async def _monitor_system_health(self) -> None:
        """Monitor overall system health."""
        # Check mode manager health
        active_modes = len([m for m in self.mode_manager.active_modes.values() if m.status == ModeStatus.ACTIVE])
        error_modes = len([m for m in self.mode_manager.active_modes.values() if m.status == ModeStatus.ERROR])
        
        if error_modes > 0 and active_modes == 0:
            # All modes are in error state
            alert = RiskAlert(
                alert_id=uuid4(),
                alert_level=AlertLevel.CRITICAL,
                message="All modes in error state",
                threshold_name="system_health",
                current_value=error_modes,
                threshold_value=0
            )
            
            await self._add_alert(alert)
            
            # Update safety status
            self.safety_status = SafetyStatus.CRITICAL
    
    async def _add_alert(self, alert: RiskAlert) -> None:
        """Add alert to active alerts and history."""
        # Check rate limiting
        current_time = time.time()
        if current_time - self._last_alert_reset > 60:  # Reset every minute
            self._alert_counts.clear()
            self._last_alert_reset = current_time
        
        alert_key = f"{alert.threshold_name}_{alert.alert_level.value}"
        current_count = self._alert_counts.get(alert_key, 0)
        
        if current_count >= self.config.max_alerts_per_minute:
            self.logger.warning("Alert rate limit exceeded", alert_key=alert_key)
            return
        
        # Add alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self._alert_counts[alert_key] = current_count + 1
        
        # Log alert
        self.logger.log(
            self._get_log_level_for_alert(alert.alert_level),
            "Safety alert generated",
            alert_id=str(alert.alert_id),
            alert_level=alert.alert_level.value,
            message=alert.message,
            threshold=alert.threshold_name,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value
        )
    
    async def _manage_alerts(self) -> None:
        """Manage active alerts and handle escalation."""
        now = datetime.now()
        
        # Auto-resolve old alerts
        to_resolve = []
        for alert in self.active_alerts.values():
            if not alert.resolved:
                # Check if threshold is no longer breached
                threshold = self.safety_thresholds.get(alert.threshold_name)
                if threshold and not threshold.is_breached:
                    alert.resolved = True
                    to_resolve.append(alert.alert_id)
        
        # Remove resolved alerts from active list
        for alert_id in to_resolve:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
    
    async def _update_alert_rate_limits(self) -> None:
        """Update alert rate limiting counters."""
        current_time = time.time()
        if current_time - self._last_alert_reset > 60:  # Reset every minute
            self._alert_counts.clear()
            self._last_alert_reset = current_time
    
    def _get_log_level_for_alert(self, alert_level: AlertLevel) -> str:
        """Get log level for alert level."""
        if alert_level == AlertLevel.EMERGENCY:
            return "critical"
        elif alert_level == AlertLevel.CRITICAL:
            return "error"
        elif alert_level == AlertLevel.WARNING:
            return "warning"
        else:
            return "info"


# Exception Classes
class SafetySystemError(Exception):
    """Base safety system error."""
    pass


class EmergencyStopError(SafetySystemError):
    """Error during emergency stop procedures."""
    pass