"""
Live Trading Safety Systems

This module contains the safety system components for live trading mode.
Extracted from live_mode.py for better maintainability.

Components:
- SafetyInterlocks: Trading permission checks
- EmergencyStopSystem: Emergency stop mechanisms  
- LiveRiskManager: Real-time rick management
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
import structlog
import torch

from src.portfolio.base import Portfolio, PositionStatus
from .config import (
    LiveModeConfig, EmergencyStopReason, SafetyCheckResult,
    RiskValidationResult, EmergencyStopResult
)

# Import attention monitoring components
try:
    from .attention_safety_monitors import (
        AttentionAnomalyDetector, AttentionAnomalyResult, AttentionAnomalyType,
        SafetyAction, ComprehensiveAnomalyResult
    )
    ATTENTION_MONITORING_AVAILABLE = True
except ImportError:
    # Graceful fallback if attention monitoring is not available
    ATTENTION_MONITORING_AVAILABLE = False
    AttentionAnomalyDetector = None

logger = structlog.get_logger()


class SafetyInterlocks:
    """Safety interlock mechanisms for live trading with attention monitoring."""
    
    def __init__(self, config: LiveModeConfig):
        self.config = config
        self.logger = logger.bind(component="SafetyInterlocks")
        
        # Attention monitoring components
        self.attention_detector: Optional[AttentionAnomalyDetector] = None
        self.attention_monitoring_enabled = False
        self.attention_trading_threshold = getattr(config, 'attention_anomaly_threshold', 0.7)
    
    async def check_trading_allowed(self) -> bool:
        """Check if trading is currently allowed based on safety rules."""
        current_time = datetime.now()
        
        # Check trading hours
        if not self._is_within_trading_hours(current_time):
            return False
        
        # Check if weekends are enabled
        if not self.config.enable_weekends and current_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return True
    
    async def check_position_limits(self, current_positions: int) -> bool:
        """Check if position limits allow new positions."""
        return current_positions < self.config.max_open_positions
    
    async def check_order_limits(self, pending_orders: int) -> bool:
        """Check if order limits allow new orders."""
        return pending_orders < self.config.max_concurrent_orders
    
    def _is_within_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours."""
        current_hour = current_time.hour
        
        if self.config.trading_hours_start <= self.config.trading_hours_end:
            # Normal hours (e.g., 9 AM to 5 PM)
            return self.config.trading_hours_start <= current_hour < self.config.trading_hours_end
        else:
            # Overnight hours (e.g., 10 PM to 6 AM)
            return current_hour >= self.config.trading_hours_start or current_hour < self.config.trading_hours_end
    
    async def add_attention_monitoring(self, detector: "AttentionAnomalyDetector") -> None:
        """Add attention monitoring to safety interlocks."""
        if not ATTENTION_MONITORING_AVAILABLE:
            self.logger.warning("Attention monitoring not available - detector not added")
            return
        
        self.attention_detector = detector
        self.attention_monitoring_enabled = True
        self.logger.info("Attention monitoring added to safety interlocks")
    
    async def initialize_attention_checks(self) -> Dict[str, Any]:
        """Initialize attention-based safety checks."""
        if not ATTENTION_MONITORING_AVAILABLE or not self.attention_detector:
            self.logger.warning("Cannot initialize attention checks - detector not available")
            return {"success": False, "reason": "detector_not_available"}
        
        try:
            if not self.attention_detector.is_initialized:
                await self.attention_detector.initialize()
            
            self.attention_monitoring_enabled = True
            self.logger.info("Attention safety checks initialized successfully")
            return {"success": True}
        except Exception as e:
            self.logger.error("Failed to initialize attention checks", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def check_trading_allowed_with_attention(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        attention_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if trading is allowed including attention anomaly checks."""
        # First check standard trading permissions
        standard_allowed = await self.check_trading_allowed()
        
        if not standard_allowed:
            return False
        
        # Check attention anomalies if monitoring is enabled
        if (self.attention_monitoring_enabled and 
            self.attention_detector and 
            attention_weights is not None):
            
            try:
                attention_result = await self.attention_detector.detect_all_anomalies(
                    attention_weights=attention_weights,
                    metadata=attention_metadata
                )
                
                # Block trading if severe attention anomalies detected
                if attention_result.overall_severity > self.attention_trading_threshold:
                    self.logger.warning(
                        "Trading blocked due to attention anomalies",
                        severity=attention_result.overall_severity,
                        threshold=self.attention_trading_threshold,
                        anomaly_count=len(attention_result.detected_anomalies)
                    )
                    return False
                
                # Block trading if emergency stop is recommended
                if attention_result.emergency_stop_required:
                    self.logger.warning("Trading blocked - attention anomaly emergency stop required")
                    return False
                
            except Exception as e:
                self.logger.error("Error during attention anomaly check for trading permission", error=str(e))
                # In case of error, be conservative and allow trading (safety systems will catch critical issues)
        
        return True


class EmergencyStopSystem:
    """Emergency stop system with comprehensive safety checks including attention anomaly detection."""
    
    def __init__(self, config: LiveModeConfig):
        self.config = config
        self.is_emergency_stopped = False
        self.stop_reason: Optional[EmergencyStopReason] = None
        self.stop_timestamp: Optional[datetime] = None
        self.stop_message = ""
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.logger = logger.bind(component="EmergencyStopSystem")
        
        # Attention monitoring components
        self.attention_detector: Optional[AttentionAnomalyDetector] = None
        self.attention_monitoring_enabled = False
        self.attention_emergency_threshold = getattr(config, 'attention_emergency_threshold', 0.9)
        self.attention_anomaly_threshold = getattr(config, 'attention_anomaly_threshold', 0.7)
    
    async def check_emergency_conditions(self, portfolio: Portfolio) -> EmergencyStopResult:
        """Check for emergency stop conditions."""
        if self.is_emergency_stopped:
            return EmergencyStopResult(
                should_stop=True,
                reason=self.stop_reason,
                message=self.stop_message,
                triggered_at=self.stop_timestamp
            )
        
        # Check maximum drawdown
        performance = portfolio.get_performance_metrics()
        current_drawdown = performance.max_drawdown
        
        if current_drawdown > self.config.emergency_drawdown_pct:
            return EmergencyStopResult(
                should_stop=True,
                reason=EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED,
                message=f"Emergency drawdown exceeded: {current_drawdown:.2%} > {self.config.emergency_drawdown_pct:.2%}",
                portfolio_value=performance.current_balance
            )
        
        # Check daily loss limit
        daily_loss_pct = abs(performance.total_pnl) / performance.initial_balance
        if daily_loss_pct > self.config.max_daily_loss_pct * 2:  # Emergency threshold at 2x normal
            return EmergencyStopResult(
                should_stop=True,
                reason=EmergencyStopReason.DAILY_LOSS_LIMIT,
                message=f"Emergency daily loss exceeded: {daily_loss_pct:.2%}",
                portfolio_value=performance.current_balance
            )
        
        return EmergencyStopResult(should_stop=False, reason=EmergencyStopReason.MANUAL_STOP)
    
    async def trigger_emergency_stop(self, reason: EmergencyStopReason, message: str = "", 
                                   portfolio_value: Decimal = Decimal("0")) -> None:
        """Trigger emergency stop with specified reason."""
        self.is_emergency_stopped = True
        self.stop_reason = reason
        self.stop_timestamp = datetime.now()
        self.stop_message = message
        
        self.logger.critical(
            "EMERGENCY STOP TRIGGERED",
            reason=reason.value,
            message=message,
            portfolio_value=str(portfolio_value),
            timestamp=self.stop_timestamp.isoformat()
        )
    
    async def reset_emergency_stop(self) -> None:
        """Reset emergency stop (requires manual intervention)."""
        self.is_emergency_stopped = False
        self.stop_reason = None
        self.stop_timestamp = None
        self.stop_message = ""
        self.consecutive_failures = 0
        
        self.logger.warning("Emergency stop reset - resuming operations")
    
    async def add_attention_detector(self, detector: "AttentionAnomalyDetector") -> None:
        """Add attention anomaly detector to the emergency stop system."""
        if not ATTENTION_MONITORING_AVAILABLE:
            self.logger.warning("Attention monitoring not available - detector not added")
            return
        
        self.attention_detector = detector
        self.attention_monitoring_enabled = True
        self.logger.info("Attention anomaly detector added to emergency stop system")
    
    async def initialize_attention_monitoring(self) -> Dict[str, Any]:
        """Initialize attention monitoring capabilities."""
        if not ATTENTION_MONITORING_AVAILABLE or not self.attention_detector:
            self.logger.warning("Cannot initialize attention monitoring - detector not available")
            return {"success": False, "reason": "detector_not_available"}
        
        try:
            result = await self.attention_detector.initialize()
            self.attention_monitoring_enabled = True
            self.logger.info("Attention monitoring initialized successfully")
            return {"success": True, "detector_result": result}
        except Exception as e:
            self.logger.error("Failed to initialize attention monitoring", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def check_emergency_conditions_with_attention(
        self,
        portfolio: Portfolio,
        attention_weights: Optional[torch.Tensor] = None,
        attention_metadata: Optional[Dict[str, Any]] = None
    ) -> EmergencyStopResult:
        """Check for emergency stop conditions including attention anomalies."""
        # First check standard emergency conditions
        standard_result = await self.check_emergency_conditions(portfolio)
        
        if standard_result.should_stop:
            return standard_result
        
        # Check attention anomalies if monitoring is enabled
        if (self.attention_monitoring_enabled and 
            self.attention_detector and 
            attention_weights is not None):
            
            try:
                attention_result = await self.attention_detector.detect_all_anomalies(
                    attention_weights=attention_weights,
                    metadata=attention_metadata,
                    portfolio=portfolio
                )
                
                # Check if emergency stop is required based on attention anomalies
                if attention_result.emergency_stop_required:
                    return EmergencyStopResult(
                        should_stop=True,
                        reason=EmergencyStopReason.ATTENTION_ANOMALY,
                        message=f"Critical attention anomaly detected: {len(attention_result.detected_anomalies)} anomalies, severity: {attention_result.overall_severity:.3f}",
                        portfolio_value=portfolio.get_performance_metrics().current_balance,
                        triggered_at=datetime.now()
                    )
                
                # Check overall severity threshold
                if attention_result.overall_severity > self.attention_emergency_threshold:
                    return EmergencyStopResult(
                        should_stop=True,
                        reason=EmergencyStopReason.ATTENTION_ANOMALY,
                        message=f"Attention anomaly severity threshold exceeded: {attention_result.overall_severity:.3f} > {self.attention_emergency_threshold}",
                        portfolio_value=portfolio.get_performance_metrics().current_balance,
                        triggered_at=datetime.now()
                    )
                
            except Exception as e:
                self.logger.error("Error during attention anomaly detection", error=str(e))
                # Continue with standard emergency check - don't fail due to attention monitoring issues
        
        return standard_result

    async def update_portfolio_value(self, portfolio: Portfolio) -> None:
        """Update portfolio value for monitoring."""
        # This is a placeholder for portfolio value monitoring
        # In a real implementation, this would track portfolio changes
        pass

    async def should_trigger_liquidation(self, portfolio: Portfolio) -> bool:
        """Check if liquidation should be triggered."""
        if not self.config.emergency_liquidation_enabled:
            return False
        
        performance = portfolio.get_performance_metrics()
        current_drawdown = performance.max_drawdown
        
        return current_drawdown > self.config.liquidation_trigger_threshold

    async def create_liquidation_plan(self, portfolio: Portfolio):
        """Create liquidation plan."""
        # This is a placeholder implementation
        # In reality, this would create a detailed liquidation plan
        
        class LiquidationPlan:
            def __init__(self, is_partial: bool, percentage: Decimal):
                self.is_partial_liquidation = is_partial
                self.liquidation_percentage = percentage
        
        performance = portfolio.get_performance_metrics()
        current_drawdown = performance.max_drawdown
        
        if current_drawdown > self.config.full_liquidation_threshold:
            return LiquidationPlan(False, Decimal("1.0"))  # Full liquidation
        else:
            return LiquidationPlan(True, self.config.partial_liquidation_percentage)


class LiveRiskManager:
    """Live risk management with production safety systems and attention anomaly monitoring."""
    
    def __init__(self, portfolio: Portfolio, config: LiveModeConfig, 
                 enable_real_time_monitoring: bool = True):
        self.portfolio = portfolio
        self.config = config
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.max_position_size_pct = config.max_position_size_pct
        self.is_monitoring_active = enable_real_time_monitoring
        self.logger = logger.bind(component="LiveRiskManager")
        
        # Risk tracking
        self.daily_trades = 0
        self.daily_volume = Decimal("0")
        self.session_start_value = Decimal("0")
        self.last_risk_check = datetime.now()
        
        # Attention monitoring components
        self.attention_detector: Optional[AttentionAnomalyDetector] = None
        self.attention_risk_monitoring_enabled = False
        self.attention_risk_threshold = getattr(config, 'attention_anomaly_threshold', 0.7)
    
    async def validate_position_size(self, token_address: str, amount_usd: Decimal) -> RiskValidationResult:
        """Validate position size against risk limits."""
        portfolio_value = self.portfolio.total_value
        max_position_value = portfolio_value * self.max_position_size_pct
        
        if amount_usd > max_position_value:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Position size {amount_usd} exceeds maximum {max_position_value} ({self.max_position_size_pct:.1%})",
                risk_level=SafetyCheckResult.DANGER,
                recommended_action="Reduce position size"
            )
        
        # Check concentration risk
        existing_positions = list(self.portfolio.positions.values())
        token_exposure = sum(
            pos.market_value for pos in existing_positions 
            if pos.symbol.startswith(token_address.split('_')[0])
        )
        
        total_exposure = token_exposure + amount_usd
        if total_exposure > max_position_value:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Total token exposure would exceed limits: {total_exposure} > {max_position_value}",
                risk_level=SafetyCheckResult.WARNING,
                recommended_action="Consider existing exposure"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)
    
    async def validate_new_position(self, token_address: str) -> RiskValidationResult:
        """Validate that a new position can be opened."""
        current_positions = len([p for p in self.portfolio.positions.values() if p.status == PositionStatus.OPEN])
        
        if current_positions >= self.config.max_open_positions:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Maximum positions reached: {current_positions}/{self.config.max_open_positions}",
                risk_level=SafetyCheckResult.WARNING,
                recommended_action="Close existing positions first"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)
    
    async def check_daily_loss_limit(self, daily_pnl: Decimal = None) -> RiskValidationResult:
        """Check daily loss limit enforcement."""
        if daily_pnl is None:
            performance = self.portfolio.performance_metrics
            daily_pnl = performance.total_pnl  # Simplified - would need proper daily calculation
        
        portfolio_value = self.portfolio.total_value
        daily_loss_limit = portfolio_value * self.config.max_daily_loss_pct
        
        if abs(daily_pnl) > daily_loss_limit and daily_pnl < 0:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Daily loss limit exceeded: {abs(daily_pnl)} > {daily_loss_limit}",
                risk_level=SafetyCheckResult.DANGER,
                recommended_action="Stop trading for today"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)
    
    async def check_emergency_conditions(self) -> EmergencyStopResult:
        """Check for emergency risk conditions."""
        performance = self.portfolio.performance_metrics
        
        # Check maximum drawdown
        if performance.max_drawdown > self.config.emergency_drawdown_pct:
            return EmergencyStopResult(
                should_stop=True,
                reason=EmergencyStopReason.MAX_DRAWDOWN_EXCEEDED,
                message=f"Emergency drawdown: {performance.max_drawdown:.2%}",
                portfolio_value=performance.current_balance
            )
        
        return EmergencyStopResult(should_stop=False, reason=EmergencyStopReason.MANUAL_STOP)

    async def validate_leverage_limit(self, position_size: Decimal) -> RiskValidationResult:
        """Validate leverage limits."""
        # Simplified leverage check
        portfolio_value = self.portfolio.total_value
        current_leverage = position_size / portfolio_value
        
        if current_leverage > self.config.max_leverage_ratio:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Leverage limit exceeded: {current_leverage:.2f}x > {self.config.max_leverage_ratio:.2f}x",
                risk_level=SafetyCheckResult.DANGER,
                recommended_action="Reduce position size to meet leverage limits"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)

    async def validate_sector_exposure(self, sector: str, position_size: Decimal) -> RiskValidationResult:
        """Validate sector exposure limits."""
        # Simplified sector exposure check
        portfolio_value = self.portfolio.total_value
        max_sector_exposure = portfolio_value * self.config.max_sector_exposure_pct
        
        # Would need to calculate current sector exposure in real implementation
        current_sector_exposure = Decimal("0")  # Placeholder
        total_exposure = current_sector_exposure + position_size
        
        if total_exposure > max_sector_exposure:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Sector exposure limit exceeded: {total_exposure} > {max_sector_exposure}",
                risk_level=SafetyCheckResult.WARNING,
                recommended_action="Reduce exposure to this sector"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)

    async def validate_correlated_exposure(self, token_address: str, position_size: Decimal, 
                                         correlation_group: str) -> RiskValidationResult:
        """Validate correlated exposure limits."""
        # Simplified correlation exposure check
        portfolio_value = self.portfolio.total_value
        max_correlated_exposure = portfolio_value * self.config.max_correlated_exposure_pct
        
        # Would need to calculate current correlated exposure in real implementation
        current_correlated_exposure = Decimal("0")  # Placeholder
        total_exposure = current_correlated_exposure + position_size
        
        if total_exposure > max_correlated_exposure:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Correlated exposure limit exceeded: {total_exposure} > {max_correlated_exposure}",
                risk_level=SafetyCheckResult.WARNING,
                recommended_action="Reduce exposure to correlated assets"
            )
        
        return RiskValidationResult(is_valid=True, risk_level=SafetyCheckResult.SAFE)

    async def check_risk_thresholds(self, portfolio_risk: Decimal):
        """Check portfolio risk thresholds."""
        # Simplified risk threshold check
        if portfolio_risk > self.config.risk_critical_threshold:
            return type('RiskAlert', (), {
                'severity': 'CRITICAL',
                'message': f'Portfolio risk critical: {portfolio_risk:.2%}'
            })()
        elif portfolio_risk > self.config.risk_warning_threshold:
            return type('RiskAlert', (), {
                'severity': 'WARNING', 
                'message': f'Portfolio risk warning: {portfolio_risk:.2%}'
            })()
        
        return None

    async def update_risk_metrics(self, portfolio: Portfolio) -> None:
        """Update risk metrics for monitoring."""
        # This is a placeholder for risk metrics updates
        # In a real implementation, this would update various risk metrics
        pass
    
    async def add_attention_risk_monitoring(self, detector: "AttentionAnomalyDetector") -> None:
        """Add attention-based risk monitoring to the risk manager."""
        if not ATTENTION_MONITORING_AVAILABLE:
            self.logger.warning("Attention monitoring not available - detector not added")
            return
        
        self.attention_detector = detector
        self.attention_risk_monitoring_enabled = True
        self.logger.info("Attention risk monitoring added to risk manager")
    
    async def initialize_attention_risk_checks(self) -> Dict[str, Any]:
        """Initialize attention-based risk checking capabilities."""
        if not ATTENTION_MONITORING_AVAILABLE or not self.attention_detector:
            self.logger.warning("Cannot initialize attention risk checks - detector not available")
            return {"success": False, "reason": "detector_not_available"}
        
        try:
            if not self.attention_detector.is_initialized:
                await self.attention_detector.initialize()
            
            self.attention_risk_monitoring_enabled = True
            self.logger.info("Attention risk checks initialized successfully")
            return {"success": True}
        except Exception as e:
            self.logger.error("Failed to initialize attention risk checks", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def validate_position_with_attention_risk(
        self,
        token_address: str,
        amount_usd: Decimal,
        attention_weights: Optional[torch.Tensor] = None,
        attention_metadata: Optional[Dict[str, Any]] = None
    ) -> RiskValidationResult:
        """Validate position considering both traditional and attention-based risks."""
        # First perform standard position validation
        standard_result = await self.validate_position_size(token_address, amount_usd)
        
        if not standard_result.is_valid:
            return standard_result
        
        # Check attention-based risks if monitoring is enabled
        if (self.attention_risk_monitoring_enabled and 
            self.attention_detector and 
            attention_weights is not None):
            
            try:
                attention_result = await self.attention_detector.detect_all_anomalies(
                    attention_weights=attention_weights,
                    metadata=attention_metadata,
                    portfolio=self.portfolio
                )
                
                # Determine risk level based on attention anomalies
                if attention_result.emergency_stop_required:
                    return RiskValidationResult(
                        is_valid=False,
                        reason=f"Position blocked due to critical attention anomalies: {len(attention_result.detected_anomalies)} anomalies with severity {attention_result.overall_severity:.3f}",
                        risk_level=SafetyCheckResult.DANGER,
                        recommended_action="emergency_stop"
                    )
                
                if attention_result.overall_severity > self.attention_risk_threshold:
                    # Determine action based on recommended action from attention analysis
                    if attention_result.recommended_action in [SafetyAction.EMERGENCY_STOP, SafetyAction.MODEL_FALLBACK]:
                        return RiskValidationResult(
                            is_valid=False,
                            reason=f"Position blocked due to attention anomalies: severity {attention_result.overall_severity:.3f} > {self.attention_risk_threshold}",
                            risk_level=SafetyCheckResult.DANGER,
                            recommended_action="block_position"
                        )
                    elif attention_result.recommended_action == SafetyAction.POSITION_REDUCTION:
                        return RiskValidationResult(
                            is_valid=False,
                            reason=f"Position size should be reduced due to attention anomalies: severity {attention_result.overall_severity:.3f}",
                            risk_level=SafetyCheckResult.WARNING,
                            recommended_action="reduce_position_size"
                        )
                
            except Exception as e:
                self.logger.error("Error during attention risk validation", error=str(e))
                # Continue with standard validation in case of attention monitoring errors
        
        return standard_result
    
    async def get_attention_risk_summary(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        attention_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get summary of attention-based risks for the portfolio."""
        if (not self.attention_risk_monitoring_enabled or 
            not self.attention_detector or 
            attention_weights is None):
            return {
                "attention_monitoring_enabled": False,
                "risk_summary": "Attention monitoring not available"
            }
        
        try:
            attention_result = await self.attention_detector.detect_all_anomalies(
                attention_weights=attention_weights,
                metadata=attention_metadata,
                portfolio=self.portfolio
            )
            
            return {
                "attention_monitoring_enabled": True,
                "overall_severity": attention_result.overall_severity,
                "emergency_stop_required": attention_result.emergency_stop_required,
                "recommended_action": attention_result.recommended_action.value,
                "detected_anomalies": [
                    {
                        "type": anomaly.anomaly_type.value,
                        "severity": anomaly.severity,
                        "confidence": anomaly.confidence,
                        "action": anomaly.recommended_action.value
                    }
                    for anomaly in attention_result.detected_anomalies
                ],
                "processing_time_ms": attention_result.processing_time_ms,
                "timestamp": attention_result.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error generating attention risk summary", error=str(e))
            return {
                "attention_monitoring_enabled": True,
                "error": str(e),
                "risk_summary": "Error retrieving attention risk data"
            }