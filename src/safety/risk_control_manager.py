"""
Risk Control Manager - Phase 4.5 Risk Controls and Position Limits

Comprehensive risk control system for trading portfolio management.
Implements position limits, concentration limits, leverage controls,
drawdown protection, and real-time risk monitoring.

Key Features:
- Per-asset position limits with configurable thresholds
- Portfolio concentration limits to prevent overexposure
- Sector and correlation-based risk controls
- Leverage controls with emergency deleveraging
- Drawdown protection with automatic position scaling
- Real-time risk monitoring and alert generation
- Risk metrics calculation and historical tracking
- Integration with existing safety systems

Following TDD methodology - implementation satisfies test requirements.
"""

import asyncio
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Deque
from uuid import uuid4

from src.portfolio.base import Portfolio, Position, PositionStatus
from src.utils.base import Chain


logger = structlog.get_logger()


class RiskViolationType(Enum):
    """Types of risk violations."""
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    CONCENTRATION_EXCEEDED = "concentration_exceeded"
    SECTOR_CONCENTRATION_EXCEEDED = "sector_concentration_exceeded"
    CORRELATION_EXCEEDED = "correlation_exceeded"
    LEVERAGE_EXCEEDED = "leverage_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    LIQUIDITY_RISK = "liquidity_risk"
    SYSTEM_ERROR = "system_error"


class RiskAlertType(Enum):
    """Types of risk alerts."""
    POSITION_SIZE = "position_size"
    PORTFOLIO_RISK = "portfolio_risk"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    LIQUIDATION = "liquidation"
    MARGIN_CALL = "margin_call"
    EMERGENCY_STOP = "emergency_stop"


class RiskAlertSeverity(Enum):
    """Severity levels for risk alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PositionLimit:
    """Position limit configuration for a specific asset."""
    asset: str
    max_size_pct: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_valid_size(self, size: Decimal, portfolio_value: Decimal) -> bool:
        """Check if position size is within limits."""
        if not self.enabled:
            return True
        
        if self.max_size_pct and (size / portfolio_value) > self.max_size_pct:
            return False
        
        if self.max_value and size > self.max_value:
            return False
        
        return True


@dataclass
class ConcentrationLimit:
    """Concentration limit configuration."""
    limit_type: str  # "portfolio", "sector", "correlation"
    max_concentration_pct: Decimal
    assets: Optional[List[str]] = None
    sector: Optional[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.max_concentration_pct <= 1:
            raise ValueError("Concentration percentage must be between 0 and 1")


@dataclass
class LeverageLimit:
    """Leverage limit configuration."""
    max_leverage: Decimal
    include_derivatives: bool = True
    emergency_deleveraging_threshold: Optional[Decimal] = None
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_leverage <= 0:
            raise ValueError("Max leverage must be positive")
        
        if self.emergency_deleveraging_threshold and self.emergency_deleveraging_threshold >= self.max_leverage:
            raise ValueError("Emergency threshold must be less than max leverage")
    
    def calculate_leverage(self, total_exposure: Decimal, total_equity: Decimal) -> Decimal:
        """Calculate current leverage ratio."""
        if total_equity <= 0:
            return Decimal("0")
        return total_exposure / total_equity
    
    def is_within_limits(self, current_leverage: Decimal) -> bool:
        """Check if current leverage is within limits."""
        return current_leverage <= self.max_leverage
    
    def requires_emergency_deleveraging(self, current_leverage: Decimal) -> bool:
        """Check if emergency deleveraging is required."""
        if not self.emergency_deleveraging_threshold:
            return False
        return current_leverage >= self.emergency_deleveraging_threshold


@dataclass
class DrawdownProtection:
    """Drawdown protection configuration."""
    max_drawdown_pct: Decimal
    daily_loss_limit_pct: Decimal
    enable_stop_loss: bool = True
    enable_position_scaling: bool = True
    recovery_threshold_pct: Optional[Decimal] = None
    
    def should_trigger(self, current_drawdown: Decimal) -> bool:
        """Check if drawdown protection should trigger."""
        return current_drawdown > self.max_drawdown_pct
    
    def check_daily_loss(self, daily_pnl: Decimal, portfolio_value: Decimal) -> bool:
        """Check if daily loss limit is exceeded."""
        if portfolio_value <= 0:
            return False
        daily_loss_pct = abs(daily_pnl) / portfolio_value
        return daily_loss_pct > self.daily_loss_limit_pct


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis."""
    timestamp: datetime
    portfolio_volatility: Decimal
    value_at_risk: Decimal
    leverage_ratio: Decimal
    sharpe_ratio: Optional[Decimal]
    position_sizes: Dict[str, Decimal]
    sector_concentrations: Dict[str, Decimal] = field(default_factory=dict)
    correlation_risks: Dict[Tuple[str, str], Decimal] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """Risk alert information."""
    alert_type: RiskAlertType
    severity: RiskAlertSeverity
    message: str
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    asset: Optional[str] = None
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    recommendation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskControlResult:
    """Result of risk control validation."""
    is_valid: bool
    overall_risk_score: Decimal
    violations: List[RiskViolationType] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[RiskAlert] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Individual validation results
    position_size_valid: bool = True
    concentration_valid: bool = True
    leverage_valid: bool = True
    drawdown_valid: bool = True
    daily_loss_valid: bool = True
    
    # Emergency conditions
    requires_emergency_action: bool = False
    emergency_actions: Optional[List[str]] = None


@dataclass
class ValidationResult:
    """Individual validation result."""
    is_valid: bool
    violation_type: Optional[RiskViolationType] = None
    message: str = ""
    recommended_action: Optional[str] = None
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None


@dataclass
class RiskControlConfig:
    """Configuration for risk control manager."""
    # Position limits
    max_position_size_pct: Decimal = Decimal("0.1")  # 10% max per position
    max_portfolio_concentration_pct: Decimal = Decimal("0.3")  # 30% max concentration
    max_sector_concentration_pct: Decimal = Decimal("0.2")  # 20% max per sector
    
    # Correlation and diversification
    max_correlation_threshold: Decimal = Decimal("0.8")  # 80% correlation limit
    min_diversification_score: Decimal = Decimal("0.6")  # 60% min diversification
    
    # Leverage controls
    max_leverage: Decimal = Decimal("2.0")  # 2x max leverage
    emergency_deleveraging_threshold: Decimal = Decimal("1.8")  # 1.8x emergency threshold
    
    # Loss protection
    max_daily_loss_pct: Decimal = Decimal("0.05")  # 5% max daily loss
    max_total_drawdown_pct: Decimal = Decimal("0.15")  # 15% max total drawdown
    trailing_stop_loss_pct: Decimal = Decimal("0.1")  # 10% trailing stop loss
    
    # Risk monitoring
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    alert_thresholds: Dict[str, Decimal] = field(default_factory=lambda: {
        "warning": Decimal("0.7"),   # 70% of limit
        "critical": Decimal("0.9")   # 90% of limit
    })
    
    # Risk metrics
    var_confidence_level: Decimal = Decimal("0.95")  # 95% VaR confidence
    lookback_period_days: int = 30
    risk_free_rate: Decimal = Decimal("0.05")  # 5% annual
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.max_position_size_pct <= 1:
            raise ValueError("Position size percentage must be between 0 and 1")
        
        if not 0 < self.max_portfolio_concentration_pct <= 1:
            raise ValueError("Portfolio concentration percentage must be between 0 and 1")
        
        if self.max_leverage <= 0:
            raise ValueError("Leverage must be positive")
        
        if not 0 < self.max_daily_loss_pct <= 1:
            raise ValueError("Daily loss percentage must be between 0 and 1")
        
        if not 0 < self.max_total_drawdown_pct <= 1:
            raise ValueError("Total drawdown percentage must be between 0 and 1")
    
    @classmethod
    def production_config(cls) -> "RiskControlConfig":
        """Create production-safe configuration with stricter limits."""
        return cls(
            max_position_size_pct=Decimal("0.05"),  # 5% max
            max_portfolio_concentration_pct=Decimal("0.25"),  # 25% max
            max_sector_concentration_pct=Decimal("0.15"),  # 15% max per sector
            max_leverage=Decimal("1.5"),  # 1.5x max leverage
            emergency_deleveraging_threshold=Decimal("1.3"),  # 1.3x emergency
            max_daily_loss_pct=Decimal("0.03"),  # 3% max daily loss
            max_total_drawdown_pct=Decimal("0.1"),  # 10% max drawdown
            enable_real_time_monitoring=True,
            monitoring_interval_seconds=15  # More frequent monitoring
        )


class RiskControlManager:
    """
    Comprehensive risk control manager for trading portfolios.
    
    Provides position limits, concentration controls, leverage management,
    drawdown protection, and real-time risk monitoring.
    """
    
    def __init__(self, config: RiskControlConfig, portfolio: Portfolio):
        """Initialize risk control manager."""
        self.config = config
        self.portfolio = portfolio
        self.is_active = True
        self.monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Risk control components
        self.position_limits: Dict[str, PositionLimit] = {}
        self.sector_allocations: Dict[str, Decimal] = {}
        self._correlation_matrix: Dict[Tuple[str, str], Decimal] = {}
        
        # Risk metrics and history
        self.risk_metrics: Optional[RiskMetrics] = None
        self._risk_metrics_history: Deque[RiskMetrics] = deque(maxlen=1000)
        
        # Metrics tracking
        self.validation_metrics = {
            "total_validations": 0,
            "violations": defaultdict(int),
            "alerts_generated": 0,
            "emergency_actions": 0
        }
        
        logger.info("RiskControlManager initialized", config=config.__dict__)
    
    async def validate_position_size(
        self, 
        asset: str, 
        size: Decimal, 
        current_price: Decimal
    ) -> ValidationResult:
        """Validate position size against limits."""
        try:
            # Size is already in USD value, not quantity of tokens
            position_value = size
            portfolio_value = self.portfolio.total_value
            
            if portfolio_value <= 0:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.SYSTEM_ERROR,
                    message="Portfolio value is zero or negative"
                )
            
            position_pct = position_value / portfolio_value
            
            # Check global position size limit
            if position_pct > self.config.max_position_size_pct:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.POSITION_SIZE_EXCEEDED,
                    message=f"Position size {position_pct:.2%} exceeds maximum position size limit {self.config.max_position_size_pct:.2%}",
                    recommended_action=f"Reduce position size to maximum {self.config.max_position_size_pct * portfolio_value}",
                    current_value=position_value,
                    limit_value=self.config.max_position_size_pct * portfolio_value
                )
            
            # Check asset-specific limits if they exist
            if asset in self.position_limits:
                asset_limit = self.position_limits[asset]
                if not asset_limit.is_valid_size(position_value, portfolio_value):
                    return ValidationResult(
                        is_valid=False,
                        violation_type=RiskViolationType.POSITION_SIZE_EXCEEDED,
                        message=f"Position violates asset-specific limit for {asset}",
                        recommended_action="Reduce position to comply with asset-specific limits"
                    )
            
            return ValidationResult(
                is_valid=True,
                message=f"Position size within position size limit ({position_pct:.2%} of portfolio)"
            )
        
        except Exception as e:
            logger.error("Position size validation failed", error=str(e), asset=asset)
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Position size validation error: {str(e)}"
            )
    
    async def validate_concentration_limit(self, new_position_value: Decimal) -> ValidationResult:
        """Validate portfolio concentration limits."""
        try:
            # Calculate current portfolio exposure
            current_exposure = sum(
                pos.value for pos in self.portfolio.positions.values()
                if hasattr(pos, 'value')
            )
            
            total_exposure = current_exposure + new_position_value
            portfolio_value = self.portfolio.total_value
            
            if portfolio_value <= 0:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.SYSTEM_ERROR,
                    message="Portfolio value is zero or negative"
                )
            
            concentration_pct = total_exposure / portfolio_value
            
            if concentration_pct > self.config.max_portfolio_concentration_pct:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.CONCENTRATION_EXCEEDED,
                    message=f"Portfolio concentration {concentration_pct:.2%} exceeds limit {self.config.max_portfolio_concentration_pct:.2%}",
                    recommended_action="Reduce overall portfolio exposure"
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Portfolio concentration within limits ({concentration_pct:.2%})"
            )
        
        except Exception as e:
            logger.error("Concentration validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Concentration validation error: {str(e)}"
            )
    
    async def validate_sector_limit(self, sector: str, additional_allocation: Decimal) -> ValidationResult:
        """Validate sector concentration limits."""
        try:
            current_allocation = self.sector_allocations.get(sector, Decimal("0"))
            total_allocation = current_allocation + additional_allocation
            
            if total_allocation > self.config.max_sector_concentration_pct:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.SECTOR_CONCENTRATION_EXCEEDED,
                    message=f"Sector allocation {total_allocation:.2%} exceeds limit {self.config.max_sector_concentration_pct:.2%} for {sector}",
                    recommended_action=f"Reduce allocation to {sector} sector"
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Sector allocation within limits for {sector} ({total_allocation:.2%})"
            )
        
        except Exception as e:
            logger.error("Sector validation failed", error=str(e), sector=sector)
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Sector validation error: {str(e)}"
            )
    
    async def validate_correlation_limit(
        self, 
        asset1: str, 
        asset2: str, 
        position1_size: Decimal, 
        position2_size: Decimal
    ) -> ValidationResult:
        """Validate correlation limits between assets."""
        try:
            correlation_key = tuple(sorted([asset1, asset2]))
            correlation = self._correlation_matrix.get(correlation_key, Decimal("0"))
            
            if correlation > self.config.max_correlation_threshold:
                # High correlation - check if positions are significant
                total_value = position1_size + position2_size
                portfolio_value = self.portfolio.total_value
                
                if portfolio_value > 0:
                    combined_exposure = total_value / portfolio_value
                    
                    # If combined exposure is significant and correlation is high, reject
                    if combined_exposure > Decimal("0.05"):  # 5% combined exposure threshold
                        return ValidationResult(
                            is_valid=False,
                            violation_type=RiskViolationType.CORRELATION_EXCEEDED,
                            message=f"High correlation {correlation:.2%} between {asset1} and {asset2} with significant exposure",
                            recommended_action="Reduce exposure to highly correlated assets"
                        )
            
            return ValidationResult(
                is_valid=True,
                message=f"Correlation risk acceptable between {asset1} and {asset2} ({correlation:.2%})"
            )
        
        except Exception as e:
            logger.error("Correlation validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Correlation validation error: {str(e)}"
            )
    
    async def validate_leverage_limit(self, additional_exposure: Decimal) -> ValidationResult:
        """Validate leverage limits."""
        try:
            # Calculate current exposure
            current_exposure = getattr(self.portfolio, 'total_exposure', Decimal("0"))
            total_exposure = current_exposure + additional_exposure
            portfolio_value = self.portfolio.total_value
            
            if portfolio_value <= 0:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.SYSTEM_ERROR,
                    message="Portfolio value is zero or negative"
                )
            
            leverage_ratio = total_exposure / portfolio_value
            
            if leverage_ratio > self.config.max_leverage:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.LEVERAGE_EXCEEDED,
                    message=f"Leverage {leverage_ratio:.2f}x exceeds maximum {self.config.max_leverage:.2f}x",
                    recommended_action="Reduce exposure or increase capital"
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Leverage within limits ({leverage_ratio:.2f}x)"
            )
        
        except Exception as e:
            logger.error("Leverage validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Leverage validation error: {str(e)}"
            )
    
    async def validate_daily_loss_limit(self) -> ValidationResult:
        """Validate daily loss limits."""
        try:
            daily_pnl = getattr(self.portfolio, 'daily_pnl', Decimal("0"))
            portfolio_value = self.portfolio.total_value
            
            if portfolio_value <= 0:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.SYSTEM_ERROR,
                    message="Portfolio value is zero or negative"
                )
            
            daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else Decimal("0")
            
            if daily_loss_pct > self.config.max_daily_loss_pct:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.DAILY_LOSS_EXCEEDED,
                    message=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.config.max_daily_loss_pct:.2%}",
                    recommended_action="Halt trading and review positions"
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Daily loss within limits ({daily_loss_pct:.2%})"
            )
        
        except Exception as e:
            logger.error("Daily loss validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Daily loss validation error: {str(e)}"
            )
    
    async def validate_drawdown_protection(self) -> ValidationResult:
        """Validate drawdown protection limits."""
        try:
            max_drawdown = getattr(self.portfolio, 'max_drawdown', Decimal("0"))
            
            if max_drawdown > self.config.max_total_drawdown_pct:
                return ValidationResult(
                    is_valid=False,
                    violation_type=RiskViolationType.DRAWDOWN_EXCEEDED,
                    message=f"Drawdown {max_drawdown:.2%} exceeds limit {self.config.max_total_drawdown_pct:.2%}",
                    recommended_action="Implement emergency position reduction"
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Drawdown within limits ({max_drawdown:.2%})"
            )
        
        except Exception as e:
            logger.error("Drawdown validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                violation_type=RiskViolationType.SYSTEM_ERROR,
                message=f"Drawdown validation error: {str(e)}"
            )
    
    async def validate_comprehensive_risk(
        self,
        asset: str,
        size: Decimal,
        sector: str,
        additional_exposure: Decimal,
        current_price: Decimal = Decimal("1")
    ) -> RiskControlResult:
        """Perform comprehensive risk validation."""
        try:
            self.validation_metrics["total_validations"] += 1
            
            # Run all validations
            position_result = await self.validate_position_size(asset, size, current_price)
            concentration_result = await self.validate_concentration_limit(size * current_price)
            sector_result = await self.validate_sector_limit(sector, size * current_price / self.portfolio.total_value)
            leverage_result = await self.validate_leverage_limit(additional_exposure)
            daily_loss_result = await self.validate_daily_loss_limit()
            drawdown_result = await self.validate_drawdown_protection()
            
            # Collect results
            results = [
                position_result, concentration_result, sector_result,
                leverage_result, daily_loss_result, drawdown_result
            ]
            
            # Determine overall validity
            is_valid = all(result.is_valid for result in results)
            violations = [result.violation_type for result in results if result.violation_type]
            recommendations = [result.recommended_action for result in results if result.recommended_action]
            
            # Calculate risk score (0-1, where 1 is highest risk)
            risk_score = self._calculate_overall_risk_score(results)
            
            # Track violations
            for violation in violations:
                if violation:
                    self.validation_metrics["violations"][violation.value] += 1
            
            return RiskControlResult(
                is_valid=is_valid,
                overall_risk_score=risk_score,
                violations=violations,
                recommendations=recommendations,
                position_size_valid=position_result.is_valid,
                concentration_valid=concentration_result.is_valid,
                leverage_valid=leverage_result.is_valid,
                drawdown_valid=drawdown_result.is_valid,
                daily_loss_valid=daily_loss_result.is_valid
            )
        
        except Exception as e:
            logger.error("Comprehensive risk validation failed", error=str(e))
            return RiskControlResult(
                is_valid=False,
                overall_risk_score=Decimal("1.0"),  # Max risk on error
                violations=[RiskViolationType.SYSTEM_ERROR],
                recommendations=["System error during risk validation - contact support"]
            )
    
    def _calculate_overall_risk_score(self, results: List[ValidationResult]) -> Decimal:
        """Calculate overall risk score from validation results."""
        try:
            # Simple scoring: count violations and weight by severity
            violation_count = sum(1 for result in results if not result.is_valid)
            total_checks = len(results)
            
            if total_checks == 0:
                return Decimal("0")
            
            base_score = Decimal(str(violation_count)) / Decimal(str(total_checks))
            
            # Apply additional weighting for specific violation types
            severity_weights = {
                RiskViolationType.LEVERAGE_EXCEEDED: Decimal("0.3"),
                RiskViolationType.DRAWDOWN_EXCEEDED: Decimal("0.25"),
                RiskViolationType.DAILY_LOSS_EXCEEDED: Decimal("0.2"),
                RiskViolationType.POSITION_SIZE_EXCEEDED: Decimal("0.15"),
                RiskViolationType.CONCENTRATION_EXCEEDED: Decimal("0.1")
            }
            
            weighted_score = Decimal("0")
            for result in results:
                if not result.is_valid and result.violation_type:
                    weight = severity_weights.get(result.violation_type, Decimal("0.1"))
                    weighted_score += weight
            
            # Combine base score with weighted score
            final_score = (base_score + weighted_score) / Decimal("2")
            
            # Ensure score is between 0 and 1
            return min(max(final_score, Decimal("0")), Decimal("1"))
        
        except Exception as e:
            logger.error("Risk score calculation failed", error=str(e))
            return Decimal("0.5")  # Medium risk on error
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            timestamp = datetime.utcnow()
            
            # Calculate portfolio volatility (simplified)
            positions = getattr(self.portfolio, 'positions', {})
            total_value = self.portfolio.total_value
            
            if not positions or total_value <= 0:
                return RiskMetrics(
                    timestamp=timestamp,
                    portfolio_volatility=Decimal("0"),
                    value_at_risk=Decimal("0"),
                    leverage_ratio=Decimal("0"),
                    sharpe_ratio=Decimal("0"),
                    position_sizes={}
                )
            
            # Calculate position sizes
            position_sizes = {}
            portfolio_volatility = Decimal("0")
            
            for asset, position in positions.items():
                if hasattr(position, 'value') and hasattr(position, 'volatility'):
                    size_pct = position.value / total_value
                    position_sizes[asset] = size_pct
                    
                    # Weighted volatility contribution
                    volatility = getattr(position, 'volatility', Decimal("0.2"))
                    portfolio_volatility += size_pct * volatility
            
            # Calculate leverage
            total_exposure = getattr(self.portfolio, 'total_exposure', total_value)
            leverage_ratio = total_exposure / total_value if total_value > 0 else Decimal("0")
            
            # Calculate Value at Risk (simplified)
            confidence_level = self.config.var_confidence_level
            z_score = Decimal("1.645") if confidence_level == Decimal("0.95") else Decimal("2.33")
            value_at_risk = total_value * portfolio_volatility * z_score
            
            # Calculate Sharpe ratio (simplified)
            total_return = getattr(self.portfolio, 'total_pnl', Decimal("0"))
            risk_free_return = self.config.risk_free_rate * total_value / Decimal("365")  # Daily
            excess_return = total_return - risk_free_return
            sharpe_ratio = excess_return / (portfolio_volatility * total_value) if portfolio_volatility > 0 else Decimal("0")
            
            metrics = RiskMetrics(
                timestamp=timestamp,
                portfolio_volatility=portfolio_volatility,
                value_at_risk=value_at_risk,
                leverage_ratio=leverage_ratio,
                sharpe_ratio=sharpe_ratio,
                position_sizes=position_sizes,
                sector_concentrations=dict(self.sector_allocations)
            )
            
            # Store in history
            self._risk_metrics_history.append(metrics)
            self.risk_metrics = metrics
            
            return metrics
        
        except Exception as e:
            logger.error("Risk metrics calculation failed", error=str(e))
            return RiskMetrics(
                timestamp=datetime.utcnow(),
                portfolio_volatility=Decimal("0"),
                value_at_risk=Decimal("0"),
                leverage_ratio=Decimal("0"),
                sharpe_ratio=None,
                position_sizes={}
            )
    
    async def generate_risk_alerts(self) -> List[RiskAlert]:
        """Generate risk alerts based on current conditions."""
        alerts = []
        
        try:
            # Check leverage alert
            total_exposure = getattr(self.portfolio, 'total_exposure', Decimal("0"))
            leverage_ratio = total_exposure / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            
            leverage_pct = leverage_ratio / self.config.max_leverage
            
            if leverage_pct >= self.config.alert_thresholds.get("critical", Decimal("0.9")):
                alerts.append(RiskAlert(
                    alert_type=RiskAlertType.LEVERAGE,
                    severity=RiskAlertSeverity.CRITICAL,
                    message=f"Leverage at {leverage_ratio:.2f}x approaching limit {self.config.max_leverage:.2f}x",
                    current_value=leverage_ratio,
                    limit_value=self.config.max_leverage,
                    recommendation="Reduce positions or increase capital"
                ))
            elif leverage_pct >= self.config.alert_thresholds.get("warning", Decimal("0.7")):
                alerts.append(RiskAlert(
                    alert_type=RiskAlertType.LEVERAGE,
                    severity=RiskAlertSeverity.WARNING,
                    message=f"Leverage at {leverage_ratio:.2f}x approaching warning threshold",
                    current_value=leverage_ratio,
                    limit_value=self.config.max_leverage
                ))
            
            # Check drawdown alert
            max_drawdown = getattr(self.portfolio, 'max_drawdown', Decimal("0"))
            drawdown_pct = max_drawdown / self.config.max_total_drawdown_pct
            
            if drawdown_pct >= self.config.alert_thresholds.get("critical", Decimal("0.9")):
                alerts.append(RiskAlert(
                    alert_type=RiskAlertType.DRAWDOWN,
                    severity=RiskAlertSeverity.CRITICAL,
                    message=f"Drawdown at {max_drawdown:.2%} approaching limit {self.config.max_total_drawdown_pct:.2%}",
                    current_value=max_drawdown,
                    limit_value=self.config.max_total_drawdown_pct,
                    recommendation="Implement stop-loss protection"
                ))
            
            # Check daily loss alert
            daily_pnl = getattr(self.portfolio, 'daily_pnl', Decimal("0"))
            if daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / self.portfolio.total_value
                loss_ratio = daily_loss_pct / self.config.max_daily_loss_pct
                
                if loss_ratio >= self.config.alert_thresholds.get("critical", Decimal("0.9")):
                    alerts.append(RiskAlert(
                        alert_type=RiskAlertType.DAILY_LOSS,
                        severity=RiskAlertSeverity.CRITICAL,
                        message=f"Daily loss at {daily_loss_pct:.2%} approaching limit {self.config.max_daily_loss_pct:.2%}",
                        current_value=daily_loss_pct,
                        limit_value=self.config.max_daily_loss_pct,
                        recommendation="Consider halting trading for the day"
                    ))
            
            self.validation_metrics["alerts_generated"] += len(alerts)
            return alerts
        
        except Exception as e:
            logger.error("Risk alert generation failed", error=str(e))
            return []
    
    async def check_emergency_conditions(self) -> RiskControlResult:
        """Check for emergency conditions requiring immediate action."""
        try:
            emergency_actions = []
            requires_emergency = False
            
            # Check extreme daily loss
            daily_pnl = getattr(self.portfolio, 'daily_pnl', Decimal("0"))
            if daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / self.portfolio.total_value
                if daily_loss_pct > self.config.max_daily_loss_pct * Decimal("2"):  # 2x daily limit
                    emergency_actions.append("Immediately halt all trading")
                    emergency_actions.append("Liquidate high-risk positions")
                    requires_emergency = True
            
            # Check extreme drawdown
            max_drawdown = getattr(self.portfolio, 'max_drawdown', Decimal("0"))
            if max_drawdown > self.config.max_total_drawdown_pct:
                emergency_actions.append("Activate emergency stop-loss")
                emergency_actions.append("Reduce portfolio exposure by 50%")
                requires_emergency = True
            
            # Check extreme leverage
            total_exposure = getattr(self.portfolio, 'total_exposure', Decimal("0"))
            leverage_ratio = total_exposure / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            if leverage_ratio > self.config.max_leverage:
                emergency_actions.append("Emergency deleveraging required")
                emergency_actions.append("Close margin positions")
                requires_emergency = True
            
            if requires_emergency:
                self.validation_metrics["emergency_actions"] += 1
            
            return RiskControlResult(
                is_valid=not requires_emergency,
                overall_risk_score=Decimal("1.0") if requires_emergency else Decimal("0.3"),
                requires_emergency_action=requires_emergency,
                emergency_actions=emergency_actions if emergency_actions else None
            )
        
        except Exception as e:
            logger.error("Emergency condition check failed", error=str(e))
            return RiskControlResult(
                is_valid=False,
                overall_risk_score=Decimal("1.0"),
                requires_emergency_action=True,
                emergency_actions=["System error - manual review required"]
            )
    
    def add_position_limit(self, asset: str, max_size_pct: Decimal, max_value: Decimal) -> None:
        """Add asset-specific position limit."""
        self.position_limits[asset] = PositionLimit(
            asset=asset,
            max_size_pct=max_size_pct,
            max_value=max_value,
            enabled=True
        )
        logger.info("Position limit added", asset=asset, max_size_pct=max_size_pct, max_value=max_value)
    
    def remove_position_limit(self, asset: str) -> None:
        """Remove asset-specific position limit."""
        if asset in self.position_limits:
            del self.position_limits[asset]
            logger.info("Position limit removed", asset=asset)
    
    def update_sector_allocation(self, sector: str, allocation: Decimal) -> None:
        """Update sector allocation tracking."""
        self.sector_allocations[sector] = allocation
        logger.debug("Sector allocation updated", sector=sector, allocation=allocation)
    
    def get_sector_allocation(self, sector: str) -> Decimal:
        """Get current sector allocation."""
        return self.sector_allocations.get(sector, Decimal("0"))
    
    async def start_real_time_monitoring(self, interval_seconds: Optional[int] = None) -> None:
        """Start real-time risk monitoring."""
        if not self.config.enable_real_time_monitoring:
            logger.warning("Real-time monitoring is disabled in configuration")
            return
        
        interval = interval_seconds or self.config.monitoring_interval_seconds
        self.monitoring_active = True
        
        logger.info("Starting real-time risk monitoring", interval_seconds=interval)
        
        try:
            while self.monitoring_active:
                # Calculate current risk metrics
                await self.calculate_risk_metrics()
                
                # Generate alerts
                alerts = await self.generate_risk_alerts()
                
                # Check emergency conditions
                emergency_result = await self.check_emergency_conditions()
                
                if alerts:
                    logger.warning("Risk alerts generated", alert_count=len(alerts))
                
                if emergency_result.requires_emergency_action:
                    logger.critical("Emergency conditions detected", actions=emergency_result.emergency_actions)
                
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info("Real-time monitoring cancelled")
        except Exception as e:
            logger.error("Real-time monitoring failed", error=str(e))
        finally:
            self.monitoring_active = False
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        self.monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
        logger.info("Real-time risk monitoring stopped")
    
    def add_risk_metrics_to_history(self, metrics: RiskMetrics) -> None:
        """Add risk metrics to historical tracking."""
        self._risk_metrics_history.append(metrics)
    
    def get_risk_metrics_history(self, hours: int = 24) -> List[RiskMetrics]:
        """Get risk metrics history for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metrics for metrics in self._risk_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def export_risk_metrics(self) -> Dict[str, Any]:
        """Export risk control metrics for monitoring."""
        return {
            "total_validations": self.validation_metrics["total_validations"],
            "violation_counts": dict(self.validation_metrics["violations"]),
            "alerts_generated": self.validation_metrics["alerts_generated"],
            "emergency_actions": self.validation_metrics["emergency_actions"],
            "risk_score_history": [
                {
                    "timestamp": metrics.timestamp.isoformat(),
                    "portfolio_volatility": float(metrics.portfolio_volatility),
                    "value_at_risk": float(metrics.value_at_risk),
                    "leverage_ratio": float(metrics.leverage_ratio),
                    "sharpe_ratio": float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None
                }
                for metrics in list(self._risk_metrics_history)[-10:]  # Last 10 metrics
            ],
            "current_limits": {
                "max_position_size_pct": float(self.config.max_position_size_pct),
                "max_leverage": float(self.config.max_leverage),
                "max_daily_loss_pct": float(self.config.max_daily_loss_pct),
                "max_total_drawdown_pct": float(self.config.max_total_drawdown_pct)
            },
            "sector_allocations": {
                sector: float(allocation)
                for sector, allocation in self.sector_allocations.items()
            },
            "monitoring_active": self.monitoring_active,
            "is_active": self.is_active
        }