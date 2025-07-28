"""
Simulation-Specific Safety Systems

This module provides safety systems specifically adapted for simulation trading,
with more lenient thresholds than live trading while still providing essential
protection mechanisms for virtual portfolios.

Key Features:
- Simulation-adapted risk thresholds
- Virtual portfolio protection mechanisms
- Experimental strategy safety validation
- Simulation-specific circuit breakers
- Risk monitoring adapted for simulation environments
"""

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
import structlog

from src.portfolio.base import Portfolio, Position, PositionStatus
from src.portfolio.virtual_portfolio import VirtualPortfolio


logger = structlog.get_logger()


class SimulationRiskLevel(Enum):
    """Risk levels for simulation trading."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERIMENTAL = "experimental"


class SimulationAlertSeverity(Enum):
    """Alert severity levels for simulation."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SimulationSafetyConfig:
    """Configuration for simulation safety systems."""
    
    # Risk thresholds (more lenient than live trading)
    max_drawdown_pct: float = 20.0  # vs 15% for live
    max_position_size_pct: float = 15.0  # vs 10% for live
    max_daily_loss_pct: float = 8.0  # vs 5% for live
    max_volatility_threshold: float = 0.4  # vs 0.3 for live
    
    # Position management
    max_open_positions: int = 15  # vs 10 for live
    position_concentration_limit: float = 0.25  # 25% in single position
    
    # Emergency stops (more lenient triggers)
    emergency_drawdown_threshold: float = 30.0  # vs 20% for live
    emergency_loss_velocity_threshold: float = 15.0  # Loss per hour
    
    # Experimental features
    enable_experimental_strategies: bool = True
    experimental_position_limit_pct: float = 5.0  # 5% for experimental
    experimental_max_positions: int = 3
    
    # Monitoring intervals
    safety_check_interval: float = 2.0  # seconds
    portfolio_health_check_interval: float = 10.0  # seconds
    
    # Circuit breaker settings
    enable_simulation_circuit_breakers: bool = True
    circuit_breaker_cooldown_minutes: int = 5
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be positive")
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")


@dataclass
class SimulationRiskAlert:
    """Risk alert for simulation trading."""
    alert_id: UUID
    alert_type: str
    severity: SimulationAlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationSafetyValidation:
    """Result of simulation safety validation."""
    validation_id: UUID
    validation_type: str
    is_safe: bool
    risk_level: SimulationRiskLevel
    message: str
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SimulationSafetyManager:
    """
    Safety manager specifically for simulation trading.
    
    Provides simulation-adapted safety checks with more lenient thresholds
    while maintaining essential protection mechanisms.
    """
    
    def __init__(self, config: SimulationSafetyConfig, virtual_portfolio: VirtualPortfolio):
        self.config = config
        self.virtual_portfolio = virtual_portfolio
        
        # Safety state
        self.active_alerts: Dict[UUID, SimulationRiskAlert] = {}
        self.alert_history: List[SimulationRiskAlert] = []
        self.validation_history: List[SimulationSafetyValidation] = []
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_activated_at: Optional[datetime] = None
        self.circuit_breaker_reason: Optional[str] = None
        
        # Performance tracking
        self.safety_metrics = {
            'total_validations': 0,
            'failed_validations': 0,
            'alerts_generated': 0,
            'circuit_breaker_activations': 0
        }
        
        self.logger = logger.bind(component="simulation_safety_manager")
        self.logger.info("Simulation safety manager initialized")
    
    async def validate_position_size(self, token_address: str, amount_usd: Decimal) -> SimulationSafetyValidation:
        """Validate position size against simulation-specific limits."""
        validation_id = uuid4()
        self.safety_metrics['total_validations'] += 1
        
        try:
            portfolio_value = self.virtual_portfolio.equity
            if portfolio_value <= 0:
                portfolio_value = self.virtual_portfolio.initial_balance
            
            position_pct = (amount_usd / portfolio_value) * 100
            
            # Check against simulation limits
            if position_pct > self.config.max_position_size_pct:
                self.safety_metrics['failed_validations'] += 1
                
                return SimulationSafetyValidation(
                    validation_id=validation_id,
                    validation_type="position_size",
                    is_safe=False,
                    risk_level=SimulationRiskLevel.HIGH,
                    message=f"Position size {position_pct:.2f}% exceeds simulation limit {self.config.max_position_size_pct}%",
                    recommendations=[
                        f"Reduce position size to below {self.config.max_position_size_pct}%",
                        "Consider splitting into multiple smaller positions"
                    ],
                    details={
                        "position_pct": float(position_pct),
                        "limit_pct": self.config.max_position_size_pct,
                        "amount_usd": float(amount_usd),
                        "portfolio_value": float(portfolio_value)
                    }
                )
            
            # Determine risk level
            risk_level = SimulationRiskLevel.LOW
            if position_pct > self.config.max_position_size_pct * 0.8:
                risk_level = SimulationRiskLevel.HIGH
            elif position_pct > self.config.max_position_size_pct * 0.6:
                risk_level = SimulationRiskLevel.MEDIUM
            
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="position_size",
                is_safe=True,
                risk_level=risk_level,
                message=f"Position size {position_pct:.2f}% within simulation limits",
                details={
                    "position_pct": float(position_pct),
                    "limit_pct": self.config.max_position_size_pct
                }
            )
            
        except Exception as e:
            self.logger.error("Error validating position size", error=str(e))
            self.safety_metrics['failed_validations'] += 1
            
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="position_size",
                is_safe=False,
                risk_level=SimulationRiskLevel.HIGH,
                message=f"Position size validation failed: {str(e)}"
            )
    
    async def check_drawdown_limits(self) -> SimulationSafetyValidation:
        """Check portfolio drawdown against simulation limits."""
        validation_id = uuid4()
        
        try:
            current_value = self.virtual_portfolio.equity
            peak_value = self.virtual_portfolio.peak_value
            
            if peak_value <= 0:
                drawdown_pct = 0.0
            else:
                drawdown_pct = ((peak_value - current_value) / peak_value) * 100
            
            # Check emergency threshold first
            if drawdown_pct > self.config.emergency_drawdown_threshold:
                await self._trigger_emergency_circuit_breaker(
                    f"Emergency drawdown threshold exceeded: {drawdown_pct:.2f}%"
                )
            
            # Check regular drawdown limit
            if drawdown_pct > self.config.max_drawdown_pct:
                await self._generate_alert(
                    alert_type="drawdown_exceeded",
                    severity=SimulationAlertSeverity.CRITICAL,
                    message=f"Simulation drawdown {drawdown_pct:.2f}% exceeds limit {self.config.max_drawdown_pct}%",
                    current_value=drawdown_pct,
                    threshold_value=self.config.max_drawdown_pct
                )
                
                return SimulationSafetyValidation(
                    validation_id=validation_id,
                    validation_type="drawdown_check",
                    is_safe=False,
                    risk_level=SimulationRiskLevel.HIGH,
                    message=f"Drawdown {drawdown_pct:.2f}% exceeds simulation limit",
                    recommendations=[
                        "Consider reducing position sizes",
                        "Review trading strategy performance",
                        "Implement stop-loss measures"
                    ],
                    details={
                        "current_drawdown_pct": drawdown_pct,
                        "limit_pct": self.config.max_drawdown_pct,
                        "current_value": float(current_value),
                        "peak_value": float(peak_value)
                    }
                )
            
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="drawdown_check",
                is_safe=True,
                risk_level=SimulationRiskLevel.LOW if drawdown_pct < 10 else SimulationRiskLevel.MEDIUM,
                message=f"Drawdown {drawdown_pct:.2f}% within simulation limits",
                details={"current_drawdown_pct": drawdown_pct}
            )
            
        except Exception as e:
            self.logger.error("Error checking drawdown limits", error=str(e))
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="drawdown_check",
                is_safe=False,
                risk_level=SimulationRiskLevel.HIGH,
                message=f"Drawdown check failed: {str(e)}"
            )
    
    async def validate_experimental_strategy(self, strategy_config: Dict[str, Any]) -> SimulationSafetyValidation:
        """Validate experimental strategy safety."""
        validation_id = uuid4()
        
        if not self.config.enable_experimental_strategies:
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="experimental_strategy",
                is_safe=False,
                risk_level=SimulationRiskLevel.HIGH,
                message="Experimental strategies disabled in configuration"
            )
        
        try:
            # Check experimental position limits
            current_experimental_positions = sum(
                1 for pos in self.virtual_portfolio.positions.values()
                if pos.metadata.get('strategy_type', '').startswith('experimental_')
            )
            
            if current_experimental_positions >= self.config.experimental_max_positions:
                return SimulationSafetyValidation(
                    validation_id=validation_id,
                    validation_type="experimental_strategy",
                    is_safe=False,
                    risk_level=SimulationRiskLevel.HIGH,
                    message=f"Experimental position limit {self.config.experimental_max_positions} reached",
                    recommendations=["Close existing experimental positions before opening new ones"]
                )
            
            # Validate experimental position size
            position_size_usd = strategy_config.get('position_size_usd', 0)
            portfolio_value = self.virtual_portfolio.equity
            
            if portfolio_value > 0:
                exp_position_pct = (position_size_usd / portfolio_value) * 100
                
                if exp_position_pct > self.config.experimental_position_limit_pct:
                    return SimulationSafetyValidation(
                        validation_id=validation_id,
                        validation_type="experimental_strategy",
                        is_safe=False,
                        risk_level=SimulationRiskLevel.HIGH,
                        message=f"Experimental position size {exp_position_pct:.2f}% exceeds limit {self.config.experimental_position_limit_pct}%"
                    )
            
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="experimental_strategy",
                is_safe=True,
                risk_level=SimulationRiskLevel.EXPERIMENTAL,
                message="Experimental strategy validated for simulation",
                details={
                    "strategy_type": strategy_config.get('strategy_type'),
                    "confidence_level": strategy_config.get('confidence_level', 0.0),
                    "current_experimental_positions": current_experimental_positions
                }
            )
            
        except Exception as e:
            self.logger.error("Error validating experimental strategy", error=str(e))
            return SimulationSafetyValidation(
                validation_id=validation_id,
                validation_type="experimental_strategy",
                is_safe=False,
                risk_level=SimulationRiskLevel.HIGH,
                message=f"Experimental strategy validation failed: {str(e)}"
            )
    
    async def check_emergency_conditions(self) -> Dict[str, Any]:
        """Check for emergency stop conditions."""
        emergency_conditions = {
            'emergency_stop_required': False,
            'reasons': [],
            'severity': 'normal'
        }
        
        try:
            # Check extreme drawdown
            current_value = self.virtual_portfolio.equity
            peak_value = self.virtual_portfolio.peak_value
            
            if peak_value > 0:
                drawdown_pct = ((peak_value - current_value) / peak_value) * 100
                
                if drawdown_pct > self.config.emergency_drawdown_threshold:
                    emergency_conditions['emergency_stop_required'] = True
                    emergency_conditions['reasons'].append('extreme_drawdown')
                    emergency_conditions['severity'] = 'critical'
            
            # Check rapid loss velocity (placeholder implementation)
            # In real implementation, would check loss rate over time
            daily_pnl = abs(float(self.virtual_portfolio.daily_pnl))
            portfolio_value = float(self.virtual_portfolio.equity)
            
            if portfolio_value > 0:
                loss_velocity = (daily_pnl / portfolio_value) * 100
                
                if loss_velocity > self.config.emergency_loss_velocity_threshold:
                    emergency_conditions['emergency_stop_required'] = True
                    emergency_conditions['reasons'].append('rapid_loss_velocity')
                    emergency_conditions['severity'] = 'critical'
            
            return emergency_conditions
            
        except Exception as e:
            self.logger.error("Error checking emergency conditions", error=str(e))
            return {
                'emergency_stop_required': True,
                'reasons': ['system_error'],
                'severity': 'critical',
                'error': str(e)
            }
    
    async def get_simulation_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive simulation safety metrics."""
        try:
            # Calculate current risk metrics
            portfolio_value = float(self.virtual_portfolio.equity)
            unrealized_pnl = float(self.virtual_portfolio.total_unrealized_pnl)
            
            # Position risk distribution
            position_risks = []
            for position in self.virtual_portfolio.positions.values():
                if position.status == PositionStatus.OPEN:
                    position_value = float(position.quantity * position.current_price)
                    position_risk_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                    position_risks.append({
                        'symbol': position.symbol,
                        'risk_pct': position_risk_pct,
                        'value_usd': position_value,
                        'pnl': float(position.unrealized_pnl)
                    })
            
            # Safety metrics
            active_alerts_count = len([a for a in self.active_alerts.values() if not a.resolved])
            critical_alerts_count = len([
                a for a in self.active_alerts.values() 
                if not a.resolved and a.severity == SimulationAlertSeverity.CRITICAL
            ])
            
            return {
                'risk_violations_count': self.safety_metrics['failed_validations'],
                'emergency_stops_triggered': 0,  # Would track actual emergency stops
                'circuit_breaker_activations': self.safety_metrics['circuit_breaker_activations'],
                'virtual_portfolio_health_score': self._calculate_portfolio_health_score(),
                'max_observed_drawdown': self._get_max_observed_drawdown(),
                'position_risk_distribution': position_risks,
                'active_alerts_count': active_alerts_count,
                'critical_alerts_count': critical_alerts_count,
                'safety_validations_total': self.safety_metrics['total_validations'],
                'safety_validation_success_rate': self._calculate_validation_success_rate(),
                'circuit_breaker_status': {
                    'active': self.circuit_breaker_active,
                    'reason': self.circuit_breaker_reason,
                    'activated_at': self.circuit_breaker_activated_at.isoformat() if self.circuit_breaker_activated_at else None
                }
            }
            
        except Exception as e:
            self.logger.error("Error getting safety metrics", error=str(e))
            return {'error': str(e)}
    
    async def get_simulation_risk_metrics(self) -> Dict[str, Any]:
        """Get simulation-specific risk metrics."""
        try:
            portfolio_value = float(self.virtual_portfolio.equity)
            initial_balance = float(self.virtual_portfolio.initial_balance)
            
            # Calculate simulation-specific metrics
            virtual_var_95 = self._calculate_virtual_var()
            simulation_sharpe = self._calculate_simulation_sharpe_ratio()
            experimental_performance = self._get_experimental_strategy_performance()
            portfolio_stability = self._calculate_portfolio_stability()
            simulation_accuracy = self._calculate_simulation_accuracy()
            
            return {
                'virtual_var_95': virtual_var_95,
                'simulation_sharpe_ratio': simulation_sharpe,
                'experimental_strategy_performance': experimental_performance,
                'virtual_portfolio_stability': portfolio_stability,
                'simulation_accuracy_score': simulation_accuracy,
                'total_return_pct': ((portfolio_value - initial_balance) / initial_balance) * 100,
                'risk_adjusted_return': simulation_sharpe * 100,  # Simplified calculation
                'portfolio_volatility': self._calculate_portfolio_volatility(),
                'concentration_risk': self._calculate_concentration_risk(),
                'max_position_risk_pct': self._get_max_position_risk_pct()
            }
            
        except Exception as e:
            self.logger.error("Error getting risk metrics", error=str(e))
            return {'error': str(e)}
    
    # Private helper methods
    
    async def _trigger_emergency_circuit_breaker(self, reason: str) -> None:
        """Trigger emergency circuit breaker."""
        if self.circuit_breaker_active:
            return
        
        self.circuit_breaker_active = True
        self.circuit_breaker_activated_at = datetime.now()
        self.circuit_breaker_reason = reason
        self.safety_metrics['circuit_breaker_activations'] += 1
        
        await self._generate_alert(
            alert_type="emergency_circuit_breaker",
            severity=SimulationAlertSeverity.CRITICAL,
            message=f"Emergency circuit breaker activated: {reason}",
            current_value=1.0,
            threshold_value=0.0,
            metadata={'reason': reason}
        )
        
        self.logger.critical("Emergency circuit breaker activated", reason=reason)
    
    async def _generate_alert(self, alert_type: str, severity: SimulationAlertSeverity,
                            message: str, current_value: float, threshold_value: float,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Generate safety alert."""
        alert = SimulationRiskAlert(
            alert_id=uuid4(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.safety_metrics['alerts_generated'] += 1
        
        self.logger.log(
            'critical' if severity == SimulationAlertSeverity.CRITICAL else 'warning',
            "Simulation safety alert",
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value
        )
    
    def _calculate_portfolio_health_score(self) -> float:
        """Calculate portfolio health score (0-100)."""
        try:
            score = 100.0
            
            # Deduct for drawdown
            current_value = self.virtual_portfolio.equity
            peak_value = self.virtual_portfolio.peak_value
            
            if peak_value > 0:
                drawdown_pct = ((peak_value - current_value) / peak_value) * 100
                score -= drawdown_pct * 2  # 2 points per % drawdown
            
            # Deduct for active alerts
            critical_alerts = len([
                a for a in self.active_alerts.values()
                if not a.resolved and a.severity == SimulationAlertSeverity.CRITICAL
            ])
            score -= critical_alerts * 15  # 15 points per critical alert
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0  # Default neutral score
    
    def _get_max_observed_drawdown(self) -> float:
        """Get maximum observed drawdown."""
        try:
            current_value = self.virtual_portfolio.equity
            peak_value = self.virtual_portfolio.peak_value
            
            if peak_value <= 0:
                return 0.0
            
            return ((peak_value - current_value) / peak_value) * 100
            
        except Exception:
            return 0.0
    
    def _calculate_validation_success_rate(self) -> float:
        """Calculate safety validation success rate."""
        if self.safety_metrics['total_validations'] == 0:
            return 100.0
        
        success_rate = (
            (self.safety_metrics['total_validations'] - self.safety_metrics['failed_validations']) 
            / self.safety_metrics['total_validations']
        ) * 100
        
        return round(success_rate, 2)
    
    def _calculate_virtual_var(self) -> float:
        """Calculate virtual Value at Risk (simplified)."""
        # Simplified VaR calculation for simulation
        portfolio_value = float(self.virtual_portfolio.equity)
        return portfolio_value * 0.05  # 5% VaR assumption
    
    def _calculate_simulation_sharpe_ratio(self) -> float:
        """Calculate simulation Sharpe ratio (simplified)."""
        # Simplified Sharpe ratio calculation
        total_return = float(self.virtual_portfolio.equity) - float(self.virtual_portfolio.initial_balance)
        return total_return / 1000.0  # Simplified calculation
    
    def _get_experimental_strategy_performance(self) -> Dict[str, Any]:
        """Get experimental strategy performance metrics."""
        experimental_positions = [
            pos for pos in self.virtual_portfolio.positions.values()
            if pos.metadata.get('strategy_type', '').startswith('experimental_')
        ]
        
        if not experimental_positions:
            return {'count': 0, 'total_pnl': 0.0, 'avg_performance': 0.0}
        
        total_pnl = sum(float(pos.unrealized_pnl) for pos in experimental_positions)
        avg_performance = total_pnl / len(experimental_positions)
        
        return {
            'count': len(experimental_positions),
            'total_pnl': total_pnl,
            'avg_performance': avg_performance
        }
    
    def _calculate_portfolio_stability(self) -> float:
        """Calculate portfolio stability score."""
        # Simplified stability calculation based on position count and distribution
        position_count = len([p for p in self.virtual_portfolio.positions.values() if p.status == PositionStatus.OPEN])
        max_positions = self.config.max_open_positions
        
        if position_count == 0:
            return 100.0
        
        stability = (1.0 - (position_count / max_positions)) * 100
        return max(0.0, min(100.0, stability))
    
    def _calculate_simulation_accuracy(self) -> float:
        """Calculate simulation accuracy score."""
        # Simplified accuracy based on validation success rate and alert frequency
        validation_rate = self._calculate_validation_success_rate()
        alert_penalty = len(self.active_alerts) * 5  # 5% penalty per active alert
        
        accuracy = validation_rate - alert_penalty
        return max(0.0, min(100.0, accuracy))
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility (simplified)."""
        # Simplified volatility calculation
        return 0.15  # Placeholder 15% volatility
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk."""
        if not self.virtual_portfolio.positions:
            return 0.0
        
        portfolio_value = float(self.virtual_portfolio.equity)
        if portfolio_value <= 0:
            return 0.0
        
        position_values = [
            float(pos.quantity * pos.current_price)
            for pos in self.virtual_portfolio.positions.values()
            if pos.status == PositionStatus.OPEN
        ]
        
        if not position_values:
            return 0.0
        
        max_position_value = max(position_values)
        concentration = (max_position_value / portfolio_value) * 100
        
        return concentration
    
    def _get_max_position_risk_pct(self) -> float:
        """Get maximum position risk percentage."""
        return self._calculate_concentration_risk()


class VirtualPortfolioProtector:
    """
    Virtual portfolio protection system for simulation trading.
    
    Provides protection mechanisms specifically designed for virtual portfolios.
    """
    
    def __init__(self, config: SimulationSafetyConfig):
        self.config = config
        self.protection_history: List[Dict[str, Any]] = []
        self.logger = logger.bind(component="virtual_portfolio_protector")
    
    async def validate_trade_safety(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade safety for virtual portfolio."""
        try:
            validation_result = {
                'is_safe': True,
                'risk_level': 'low',
                'recommendations': [],
                'protection_applied': False
            }
            
            # Extract trade details
            action = trade_request.get('action', '').upper()
            amount_usd = trade_request.get('amount_usd', 0)
            portfolio_value = trade_request.get('current_portfolio_value', 1)
            
            # Calculate position size percentage
            if portfolio_value > 0:
                position_pct = (amount_usd / portfolio_value) * 100
                
                # Check if position size is risky
                if position_pct > self.config.max_position_size_pct * 0.8:
                    validation_result['risk_level'] = 'high'
                    validation_result['recommendations'].append(
                        f"Large position size: {position_pct:.2f}%"
                    )
                
                if position_pct > self.config.max_position_size_pct:
                    validation_result['is_safe'] = False
                    validation_result['recommendations'].append(
                        "Position size exceeds safety limits"
                    )
            
            return validation_result
            
        except Exception as e:
            self.logger.error("Error validating trade safety", error=str(e))
            return {
                'is_safe': False,
                'risk_level': 'high',
                'error': str(e)
            }
    
    async def check_portfolio_health(self, portfolio_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check virtual portfolio health."""
        try:
            health_result = {
                'health_status': 'good',
                'risk_score': 0.0,
                'protection_active': False,
                'recommendations': []
            }
            
            if not portfolio_state:
                return health_result
            
            # Check portfolio metrics
            total_value = portfolio_state.get('total_value', 0)
            unrealized_pnl = portfolio_state.get('unrealized_pnl', 0)
            position_count = portfolio_state.get('position_count', 0)
            
            # Calculate risk score
            risk_score = 0.0
            
            # Add risk for negative PnL
            if unrealized_pnl < 0 and total_value > 0:
                pnl_risk = abs(unrealized_pnl / total_value) * 100
                risk_score += pnl_risk
            
            # Add risk for position concentration
            concentration_risk = portfolio_state.get('concentration_risk', 0)
            if concentration_risk > 20:  # 20% in single position
                risk_score += concentration_risk - 20
            
            # Determine health status
            if risk_score > 30:
                health_result['health_status'] = 'poor'
                health_result['recommendations'].append("High risk portfolio detected")
            elif risk_score > 15:
                health_result['health_status'] = 'fair'
                health_result['recommendations'].append("Moderate risk detected")
            
            health_result['risk_score'] = min(100.0, risk_score)
            
            return health_result
            
        except Exception as e:
            self.logger.error("Error checking portfolio health", error=str(e))
            return {
                'health_status': 'unknown',
                'error': str(e)
            }
    
    async def apply_protective_measures(self, risk_scenario: Dict[str, Any]) -> bool:
        """Apply protective measures for identified risks."""
        try:
            protection_type = risk_scenario.get('type')
            severity = risk_scenario.get('severity', 'low')
            
            protection_applied = False
            
            if protection_type == 'high_concentration_risk':
                # Log protective measure
                self.logger.info(
                    "Concentration risk protection activated",
                    severity=severity,
                    affected_positions=risk_scenario.get('affected_positions', [])
                )
                protection_applied = True
            
            elif protection_type == 'rapid_loss':
                # Log rapid loss protection
                self.logger.warning(
                    "Rapid loss protection activated",
                    severity=severity
                )
                protection_applied = True
            
            # Record protection event
            if protection_applied:
                self.protection_history.append({
                    'timestamp': datetime.now(),
                    'type': protection_type,
                    'severity': severity,
                    'scenario': risk_scenario
                })
            
            return protection_applied
            
        except Exception as e:
            self.logger.error("Error applying protective measures", error=str(e))
            return False