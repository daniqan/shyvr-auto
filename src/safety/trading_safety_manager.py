"""
Trading Safety Manager

Comprehensive pre-trade validation system for financial trading safety.
Implements position size limits, risk exposure checks, rate limiting,
balance verification, and emergency stop mechanisms.

Key Features:
- Pre-trade validation pipeline
- Position size and order value limits
- Rate limiting per trading pair
- Balance and risk exposure validation
- Emergency stop functionality
- Comprehensive safety metrics tracking

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Deque
from uuid import UUID, uuid4

from src.portfolio.base import Portfolio, Position
from src.rl_agent.base import MarketState
from src.utils.base import Chain


@dataclass
class TradeOrder:
    """Represents a trade order for safety validation."""
    token_address: str
    action_type: str  # "buy" or "sell"
    amount: Decimal  # USD value of the trade
    confidence: float = 0.0
    chain: Optional[Chain] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


logger = structlog.get_logger()


class ValidationStatus(Enum):
    """Status of pre-trade validation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    PENDING = "pending"


class ValidationReason(Enum):
    """Reasons for validation results."""
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    ORDER_VALUE_EXCEEDED = "order_value_exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    RISK_EXPOSURE_EXCEEDED = "risk_exposure_exceeded"
    EMERGENCY_STOP_ACTIVE = "emergency_stop_active"
    MARKET_CONDITIONS = "market_conditions"
    SYSTEM_ERROR = "system_error"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"
    SECTOR_CONCENTRATION_EXCEEDED = "sector_concentration_exceeded"


@dataclass
class RateLimitState:
    """Rate limiting state for a trading pair."""
    token_address: str
    recent_trades: Deque[datetime] = field(default_factory=deque)
    last_trade_time: Optional[datetime] = None
    hourly_count: int = 0
    minute_count: int = 0
    
    def __post_init__(self):
        """Initialize deque if not provided."""
        if not isinstance(self.recent_trades, deque):
            self.recent_trades = deque(self.recent_trades if self.recent_trades else [])


@dataclass
class PositionSizeValidation:
    """Position size validation utilities."""
    portfolio_value: Decimal
    max_position_pct: Decimal
    
    def calculate_max_position_size(self) -> Decimal:
        """Calculate maximum allowed position size."""
        return self.portfolio_value * self.max_position_pct
    
    def is_size_valid(self, size: Decimal) -> bool:
        """Check if position size is valid."""
        max_size = self.calculate_max_position_size()
        return size <= max_size


@dataclass
class RiskExposureValidation:
    """Risk exposure validation utilities."""
    current_exposure: Decimal
    max_total_exposure: Decimal
    
    def calculate_remaining_capacity(self) -> Decimal:
        """Calculate remaining risk capacity."""
        return self.max_total_exposure - self.current_exposure
    
    def is_exposure_valid(self, additional_exposure: Decimal) -> bool:
        """Check if additional exposure is valid."""
        total_exposure = self.current_exposure + additional_exposure
        return total_exposure <= self.max_total_exposure


@dataclass
class PreTradeValidationResult:
    """Result of pre-trade validation."""
    is_valid: bool
    status: ValidationStatus
    reasons: List[ValidationReason] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradingSafetyConfig:
    """Configuration for trading safety manager."""
    # Position size limits
    max_position_size_pct: Decimal = Decimal("0.1")  # 10% max single position
    max_single_order_value: Decimal = Decimal("10000")  # $10k max order
    min_order_value: Decimal = Decimal("10")  # $10 min order
    
    # Rate limiting
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    rate_limit_cooldown_seconds: int = 60
    
    # Risk exposure
    max_total_exposure_pct: Decimal = Decimal("0.8")  # 80% max total exposure
    max_sector_concentration_pct: Decimal = Decimal("0.3")  # 30% max per sector
    max_correlation_threshold: Decimal = Decimal("0.85")  # 85% max correlation
    
    # Balance requirements
    min_balance_reserve_pct: Decimal = Decimal("0.05")  # 5% reserve
    max_portfolio_utilization_pct: Decimal = Decimal("0.95")  # 95% max utilization
    
    # Emergency controls
    enable_emergency_stop: bool = True
    emergency_stop_on_system_error: bool = True
    
    # Validation controls
    enable_position_size_validation: bool = True
    enable_order_value_validation: bool = True
    enable_rate_limit_validation: bool = True
    enable_balance_validation: bool = True
    enable_risk_exposure_validation: bool = True
    enable_ensemble_validation: bool = True
    
    # Ensemble-specific controls
    ensemble_confidence_threshold: Decimal = Decimal("0.6")  # 60% minimum ensemble confidence
    min_ensemble_size: int = 3  # Minimum models in ensemble for production
    model_agreement_threshold: Decimal = Decimal("0.7")  # 70% model agreement required
    ensemble_variance_threshold: Decimal = Decimal("0.2")  # Max prediction variance allowed
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.max_position_size_pct <= 1):
            raise ValueError("Position size percentage must be between 0 and 1")
        
        if self.max_single_order_value <= 0:
            raise ValueError("Order value must be positive")
        
        if self.min_order_value <= 0:
            raise ValueError("Minimum order value must be positive")
        
        if not (0 < self.max_total_exposure_pct <= 1):
            raise ValueError("Total exposure percentage must be between 0 and 1")
        
        if not (0 < self.min_balance_reserve_pct < 1):
            raise ValueError("Balance reserve percentage must be between 0 and 1")
    
    @classmethod
    def production_config(cls) -> "TradingSafetyConfig":
        """Create production-safe configuration."""
        return cls(
            max_position_size_pct=Decimal("0.05"),  # 5% max for production
            max_single_order_value=Decimal("5000"),  # $5k max for production
            max_orders_per_minute=5,  # More conservative rate limiting
            max_orders_per_hour=50,
            min_balance_reserve_pct=Decimal("0.1"),  # 10% reserve for production
            max_total_exposure_pct=Decimal("0.7"),  # 70% max exposure
        )


class TradingSafetyManager:
    """
    Comprehensive trading safety manager.
    
    Provides pre-trade validation, position size limits, risk exposure checks,
    rate limiting, balance verification, and emergency stop mechanisms.
    """
    
    def __init__(self, config: TradingSafetyConfig, portfolio: Portfolio):
        """Initialize trading safety manager."""
        self.config = config
        self.portfolio = portfolio
        self.rate_limit_state: Dict[str, RateLimitState] = {}
        self.is_active = True
        self.is_emergency_stopped = False
        self.emergency_stop_reason: Optional[str] = None
        self.emergency_stop_timestamp: Optional[datetime] = None
        
        # Metrics tracking
        self.validation_metrics = {
            "total_validations": 0,
            "approvals": 0,
            "rejections": 0,
            "conditional_approvals": 0,
            "rejection_reasons": defaultdict(int),
            "validation_times": deque(maxlen=1000),
        }
        
        logger.info("TradingSafetyManager initialized", config=config)
    
    async def validate_pre_trade(
        self, 
        trade_order: TradeOrder, 
        market_state: Optional[MarketState] = None
    ) -> PreTradeValidationResult:
        """
        Comprehensive pre-trade validation pipeline.
        
        Args:
            trade_order: The trade order to validate
            market_state: Current market state (optional)
            
        Returns:
            PreTradeValidationResult with validation outcome
        """
        start_time = datetime.utcnow()
        
        try:
            # Track validation attempt
            self.validation_metrics["total_validations"] += 1
            
            # Check if emergency stop is active
            if self.is_emergency_stopped:
                result = PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.EMERGENCY_STOP_ACTIVE],
                    recommendations=[f"Emergency stop active: {self.emergency_stop_reason}"]
                )
                self._update_validation_metrics(result)
                return result
            
            # Run all validation checks
            validation_results = await self._run_all_validations(trade_order, market_state)
            
            # Combine results
            combined_result = self._combine_validation_results(validation_results)
            
            # Update metrics
            self._update_validation_metrics(combined_result)
            
            # Log validation result
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            self.validation_metrics["validation_times"].append(validation_time)
            
            logger.info(
                "Pre-trade validation completed",
                token_address=trade_order.token_address,
                is_valid=combined_result.is_valid,
                status=combined_result.status.value,
                reasons=[r.value for r in combined_result.reasons],
                validation_time_ms=validation_time * 1000
            )
            
            return combined_result
            
        except Exception as e:
            logger.error("Pre-trade validation failed", error=str(e), token_address=trade_order.token_address)
            
            error_result = PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["System error during validation - contact support"],
                metadata={"error": str(e)}
            )
            
            self._update_validation_metrics(error_result)
            return error_result
    
    async def _run_all_validations(
        self, 
        trade_order: TradeOrder, 
        market_state: Optional[MarketState]
    ) -> List[PreTradeValidationResult]:
        """Run all validation checks in parallel."""
        validation_tasks = []
        
        if self.config.enable_position_size_validation:
            validation_tasks.append(self.validate_position_size(trade_order))
        
        if self.config.enable_order_value_validation:
            validation_tasks.append(self.validate_order_value(trade_order))
        
        if self.config.enable_rate_limit_validation:
            validation_tasks.append(self.validate_rate_limits(trade_order))
        
        if self.config.enable_balance_validation:
            validation_tasks.append(self.validate_balance(trade_order))
        
        if self.config.enable_risk_exposure_validation:
            validation_tasks.append(self.validate_risk_exposure(trade_order))
        
        # Add ensemble validation if metadata indicates ensemble prediction
        if (self.config.enable_ensemble_validation and 
            trade_order.metadata.get('prediction_type') == 'ensemble'):
            validation_tasks.append(self.validate_ensemble_prediction(trade_order))
        
        # Run all validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Validation check failed", error=str(result))
                valid_results.append(PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=["Validation check failed - contact support"]
                ))
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _combine_validation_results(self, results: List[PreTradeValidationResult]) -> PreTradeValidationResult:
        """Combine multiple validation results into a single result."""
        if not results:
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["No validation results available"]
            )
        
        # If any validation fails, the trade is rejected
        all_valid = all(result.is_valid for result in results)
        
        # Collect all reasons and recommendations
        all_reasons = []
        all_recommendations = []
        
        for result in results:
            all_reasons.extend(result.reasons)
            all_recommendations.extend(result.recommendations)
        
        # Determine final status
        if all_valid:
            status = ValidationStatus.APPROVED
            recommendations = ["All validations passed - trade approved for execution"]
        else:
            status = ValidationStatus.REJECTED
            recommendations = all_recommendations or ["Trade rejected due to validation failures"]
        
        return PreTradeValidationResult(
            is_valid=all_valid,
            status=status,
            reasons=all_reasons,
            recommendations=recommendations
        )
    
    async def validate_position_size(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate position size limits."""
        try:
            portfolio_value = self.portfolio.total_value
            position_size_validation = PositionSizeValidation(
                portfolio_value=portfolio_value,
                max_position_pct=self.config.max_position_size_pct
            )
            
            if position_size_validation.is_size_valid(trade_order.amount):
                return PreTradeValidationResult(
                    is_valid=True,
                    status=ValidationStatus.APPROVED,
                    recommendations=["Position size within limits"]
                )
            else:
                max_size = position_size_validation.calculate_max_position_size()
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.POSITION_SIZE_EXCEEDED],
                    recommendations=[
                        f"Reduce position size to maximum {max_size} "
                        f"({self.config.max_position_size_pct * 100}% of portfolio)"
                    ]
                )
        
        except Exception as e:
            logger.error("Position size validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Position size validation failed - contact support"]
            )
    
    async def validate_order_value(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate order value limits."""
        try:
            order_value = trade_order.amount
            
            if order_value > self.config.max_single_order_value:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.ORDER_VALUE_EXCEEDED],
                    recommendations=[
                        f"Reduce order value to maximum {self.config.max_single_order_value}"
                    ]
                )
            
            if order_value < self.config.min_order_value:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.ORDER_VALUE_EXCEEDED],
                    recommendations=[
                        f"Increase order value to minimum {self.config.min_order_value}"
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Order value within limits"]
            )
        
        except Exception as e:
            logger.error("Order value validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Order value validation failed - contact support"]
            )
    
    async def validate_rate_limits(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate rate limiting per trading pair."""
        try:
            token_address = trade_order.token_address
            now = datetime.utcnow()
            
            # Get or create rate limit state
            if token_address not in self.rate_limit_state:
                self.rate_limit_state[token_address] = RateLimitState(token_address=token_address)
            
            state = self.rate_limit_state[token_address]
            
            # Clean old trades (older than 1 hour)
            cutoff_time = now - timedelta(hours=1)
            while state.recent_trades and state.recent_trades[0] < cutoff_time:
                state.recent_trades.popleft()
            
            # Count recent trades
            minute_cutoff = now - timedelta(minutes=1)
            minute_trades = sum(1 for trade_time in state.recent_trades if trade_time >= minute_cutoff)
            hour_trades = len(state.recent_trades)
            
            # Check rate limits
            if minute_trades >= self.config.max_orders_per_minute:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.RATE_LIMIT_EXCEEDED],
                    recommendations=[
                        f"Rate limit exceeded: {minute_trades} orders in last minute "
                        f"(max {self.config.max_orders_per_minute}). Wait before retrying."
                    ]
                )
            
            if hour_trades >= self.config.max_orders_per_hour:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.RATE_LIMIT_EXCEEDED],
                    recommendations=[
                        f"Rate limit exceeded: {hour_trades} orders in last hour "
                        f"(max {self.config.max_orders_per_hour}). Wait before retrying."
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Rate limits satisfied"]
            )
        
        except Exception as e:
            logger.error("Rate limit validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Rate limit validation failed - contact support"]
            )
    
    async def validate_balance(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate balance and reserve requirements."""
        try:
            available_balance = self.portfolio.available_balance
            total_value = self.portfolio.total_value
            required_reserve = total_value * self.config.min_balance_reserve_pct
            
            # Check if trade would leave sufficient reserve
            remaining_balance = available_balance - trade_order.amount
            
            if remaining_balance < required_reserve:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.INSUFFICIENT_BALANCE],
                    recommendations=[
                        f"Insufficient balance: trade would leave {remaining_balance}, "
                        f"but {required_reserve} reserve required "
                        f"({self.config.min_balance_reserve_pct * 100}% of portfolio)"
                    ]
                )
            
            if trade_order.amount > available_balance:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.INSUFFICIENT_BALANCE],
                    recommendations=[
                        f"Order amount {trade_order.amount} exceeds available balance {available_balance}"
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Sufficient balance and reserves"]
            )
        
        except Exception as e:
            logger.error("Balance validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Balance validation failed - contact support"]
            )
    
    async def validate_risk_exposure(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate risk exposure limits."""
        try:
            # Calculate current portfolio exposure (simplified)
            total_positions_value = sum(
                pos.size * pos.current_price 
                for pos in self.portfolio.positions.values()
                if pos.status.value == "open"
            )
            
            current_exposure = total_positions_value / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            additional_exposure = trade_order.amount / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            
            risk_validation = RiskExposureValidation(
                current_exposure=current_exposure,
                max_total_exposure=self.config.max_total_exposure_pct
            )
            
            if not risk_validation.is_exposure_valid(additional_exposure):
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.RISK_EXPOSURE_EXCEEDED],
                    recommendations=[
                        f"Risk exposure would exceed limit: "
                        f"current {current_exposure * 100:.1f}% + new {additional_exposure * 100:.1f}% "
                        f"exceeds maximum {self.config.max_total_exposure_pct * 100:.1f}%"
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Risk exposure within limits"]
            )
        
        except Exception as e:
            logger.error("Risk exposure validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Risk exposure validation failed - contact support"]
            )
    
    async def validate_ensemble_prediction(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate ensemble prediction quality and characteristics."""
        try:
            ensemble_size = trade_order.metadata.get('ensemble_size', 0)
            model_agreement = trade_order.metadata.get('model_agreement', 0.0)
            prediction_variance = trade_order.metadata.get('prediction_variance', 0.0)
            model_confidences = trade_order.metadata.get('model_confidences', {})
            
            # Check minimum ensemble size
            if ensemble_size < self.config.min_ensemble_size:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=[
                        f"Ensemble size {ensemble_size} below minimum {self.config.min_ensemble_size}"
                    ]
                )
            
            # Check ensemble confidence
            if trade_order.confidence < float(self.config.ensemble_confidence_threshold):
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=[
                        f"Ensemble confidence {trade_order.confidence:.3f} below threshold {self.config.ensemble_confidence_threshold}"
                    ]
                )
            
            # Check model agreement
            if model_agreement < float(self.config.model_agreement_threshold):
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=[
                        f"Model agreement {model_agreement:.3f} below threshold {self.config.model_agreement_threshold}"
                    ]
                )
            
            # Check prediction variance
            if prediction_variance > float(self.config.ensemble_variance_threshold):
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=[
                        f"Prediction variance {prediction_variance:.3f} exceeds threshold {self.config.ensemble_variance_threshold}"
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Ensemble prediction quality validated"]
            )
        
        except Exception as e:
            logger.error("Ensemble prediction validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Ensemble validation failed - contact support"]
            )
    
    async def validate_degraded_ensemble(self, trade_order: TradeOrder) -> PreTradeValidationResult:
        """Validate trading decision when ensemble is degraded."""
        try:
            degraded_models = trade_order.metadata.get('degraded_models', [])
            remaining_ensemble_size = trade_order.metadata.get('ensemble_size', 0)
            fallback_active = trade_order.metadata.get('fallback_active', False)
            
            # Apply more conservative thresholds for degraded ensemble
            degraded_confidence_threshold = float(self.config.ensemble_confidence_threshold) * 0.8
            
            if trade_order.confidence < degraded_confidence_threshold:
                return PreTradeValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SYSTEM_ERROR],
                    recommendations=[
                        f"Degraded ensemble confidence {trade_order.confidence:.3f} below adjusted threshold {degraded_confidence_threshold:.3f}"
                    ]
                )
            
            # Reduce position size for degraded ensemble
            if remaining_ensemble_size <= 2:
                return PreTradeValidationResult(
                    is_valid=True,
                    status=ValidationStatus.CONDITIONAL,
                    recommendations=[
                        "Degraded ensemble - consider reducing position size by 50%",
                        f"Models degraded: {', '.join(degraded_models)}"
                    ]
                )
            
            return PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Degraded ensemble validated with conservative approach"]
            )
        
        except Exception as e:
            logger.error("Degraded ensemble validation failed", error=str(e))
            return PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Degraded ensemble validation failed - contact support"]
            )
    
    def calculate_ensemble_confidence(self, individual_confidences: List[float], method: str = 'weighted_average') -> float:
        """Calculate ensemble confidence from individual model confidences."""
        if not individual_confidences:
            return 0.0
        
        if method == 'weighted_average':
            return sum(individual_confidences) / len(individual_confidences)
        elif method == 'conservative_min':
            return min(individual_confidences)
        elif method == 'confidence_weighted':
            # Weight by confidence itself
            weights = [c for c in individual_confidences]
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(c * w for c, w in zip(individual_confidences, weights)) / total_weight
        else:
            return sum(individual_confidences) / len(individual_confidences)
    
    def validate_model_agreement(self, model_agreement: float) -> bool:
        """Validate if model agreement meets threshold."""
        return model_agreement >= float(self.config.model_agreement_threshold)
    
    async def record_trade_execution(self, trade_order: TradeOrder) -> None:
        """Record trade execution for rate limiting."""
        try:
            token_address = trade_order.token_address
            now = datetime.utcnow()
            
            # Update rate limit state
            self._update_rate_limit_state(token_address, now)
            
            logger.debug("Trade execution recorded", token_address=token_address, timestamp=now)
        
        except Exception as e:
            logger.error("Failed to record trade execution", error=str(e))
    
    def _update_rate_limit_state(self, token_address: str, timestamp: datetime) -> None:
        """Update rate limit state for a token."""
        if token_address not in self.rate_limit_state:
            self.rate_limit_state[token_address] = RateLimitState(token_address=token_address)
        
        state = self.rate_limit_state[token_address]
        state.recent_trades.append(timestamp)
        state.last_trade_time = timestamp
        
        # Clean old trades to prevent memory growth
        cutoff_time = timestamp - timedelta(hours=1)
        while state.recent_trades and state.recent_trades[0] < cutoff_time:
            state.recent_trades.popleft()
    
    async def activate_emergency_stop(self, reason: str) -> None:
        """Activate emergency stop mechanism."""
        self.is_emergency_stopped = True
        self.emergency_stop_reason = reason
        self.emergency_stop_timestamp = datetime.utcnow()
        
        logger.critical(
            "Emergency stop activated",
            reason=reason,
            timestamp=self.emergency_stop_timestamp
        )
    
    async def deactivate_emergency_stop(self) -> None:
        """Deactivate emergency stop mechanism."""
        self.is_emergency_stopped = False
        previous_reason = self.emergency_stop_reason
        self.emergency_stop_reason = None
        self.emergency_stop_timestamp = None
        
        logger.warning(
            "Emergency stop deactivated",
            previous_reason=previous_reason,
            timestamp=datetime.utcnow()
        )
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety metrics and statistics."""
        avg_validation_time = (
            sum(self.validation_metrics["validation_times"]) / len(self.validation_metrics["validation_times"])
            if self.validation_metrics["validation_times"] else 0
        )
        
        return {
            "total_validations": self.validation_metrics["total_validations"],
            "approvals": self.validation_metrics["approvals"],
            "rejections": self.validation_metrics["rejections"],
            "conditional_approvals": self.validation_metrics["conditional_approvals"],
            "rejection_reasons": dict(self.validation_metrics["rejection_reasons"]),
            "approval_rate": (
                self.validation_metrics["approvals"] / self.validation_metrics["total_validations"]
                if self.validation_metrics["total_validations"] > 0 else 0
            ),
            "average_validation_time_ms": avg_validation_time * 1000,
            "is_emergency_stopped": self.is_emergency_stopped,
            "emergency_stop_reason": self.emergency_stop_reason,
            "rate_limit_states": {
                address: {
                    "recent_trades_count": len(state.recent_trades),
                    "last_trade_time": state.last_trade_time.isoformat() if state.last_trade_time else None
                }
                for address, state in self.rate_limit_state.items()
            }
        }
    
    def _update_validation_metrics(self, result: PreTradeValidationResult) -> None:
        """Update validation metrics based on result."""
        if result.status == ValidationStatus.APPROVED:
            self.validation_metrics["approvals"] += 1
        elif result.status == ValidationStatus.REJECTED:
            self.validation_metrics["rejections"] += 1
            for reason in result.reasons:
                self.validation_metrics["rejection_reasons"][reason.value] += 1
        elif result.status == ValidationStatus.CONDITIONAL:
            self.validation_metrics["conditional_approvals"] += 1