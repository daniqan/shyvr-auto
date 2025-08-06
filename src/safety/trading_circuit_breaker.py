"""
Trading Circuit Breaker System

Comprehensive circuit breaker implementation for extreme market conditions with:
- Price movement circuit breakers (drops and spikes)
- Volume spike detection and protection
- Volatility-based trading halts
- Market-wide circuit breakers for systemic risk
- Cooldown period management
- Graduated response levels (warning, critical, emergency)
- Integration with emergency stop system

Follows TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Deque, Set
from uuid import UUID, uuid4

from src.safety.emergency_stop_controller import EmergencyStopController, EmergencyStopReason


logger = structlog.get_logger()


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY = "volatility"
    MARKET_WIDE = "market_wide"
    FLASH_CRASH = "flash_crash"
    MANUAL = "manual"


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels."""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CircuitBreakerReason(Enum):
    """Specific reasons for circuit breaker activation."""
    # Price movement reasons
    PRICE_DROP_WARNING = "price_drop_warning"
    PRICE_DROP_CRITICAL = "price_drop_critical"
    PRICE_DROP_EMERGENCY = "price_drop_emergency"
    PRICE_SPIKE_WARNING = "price_spike_warning"
    PRICE_SPIKE_CRITICAL = "price_spike_critical"
    PRICE_SPIKE_EMERGENCY = "price_spike_emergency"
    
    # Volume spike reasons
    VOLUME_SPIKE_WARNING = "volume_spike_warning"
    VOLUME_SPIKE_CRITICAL = "volume_spike_critical"
    VOLUME_SPIKE_EMERGENCY = "volume_spike_emergency"
    
    # Volatility reasons
    VOLATILITY_WARNING = "volatility_warning"
    VOLATILITY_CRITICAL = "volatility_critical"
    VOLATILITY_EMERGENCY = "volatility_emergency"
    
    # Market-wide reasons
    MARKET_CRASH = "market_crash"
    FLASH_CRASH = "flash_crash"
    MARKET_INSTABILITY = "market_instability"
    
    # Manual reasons
    MANUAL_INTERVENTION = "manual_intervention"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class CircuitBreakerConfig:
    """Configuration for trading circuit breaker system."""
    # Price movement thresholds (as percentages)
    price_drop_warning_pct: Decimal = Decimal("0.05")  # 5%
    price_drop_critical_pct: Decimal = Decimal("0.10")  # 10%
    price_drop_emergency_pct: Decimal = Decimal("0.15")  # 15%
    price_spike_warning_pct: Decimal = Decimal("0.10")  # 10%
    price_spike_critical_pct: Decimal = Decimal("0.20")  # 20%
    price_spike_emergency_pct: Decimal = Decimal("0.30")  # 30%
    
    # Volume spike thresholds (multipliers of normal volume)
    volume_spike_warning_multiplier: Decimal = Decimal("3.0")  # 3x
    volume_spike_critical_multiplier: Decimal = Decimal("5.0")  # 5x
    volume_spike_emergency_multiplier: Decimal = Decimal("10.0")  # 10x
    
    # Volatility thresholds (as percentages)
    volatility_warning_pct: Decimal = Decimal("0.15")  # 15%
    volatility_critical_pct: Decimal = Decimal("0.25")  # 25%
    volatility_emergency_pct: Decimal = Decimal("0.40")  # 40%
    
    # Market-wide thresholds
    market_crash_threshold_pct: Decimal = Decimal("0.20")  # 20%
    flash_crash_threshold_pct: Decimal = Decimal("0.10")  # 10%
    flash_crash_time_window_minutes: int = 5
    market_wide_asset_threshold: int = 3  # Minimum assets affected for market-wide
    
    # Cooldown periods (in minutes)
    warning_cooldown_minutes: int = 5
    critical_cooldown_minutes: int = 15
    emergency_cooldown_minutes: int = 30
    
    # Time windows for analysis (in minutes)
    price_movement_window_minutes: int = 15
    volume_analysis_window_minutes: int = 60
    volatility_calculation_window_minutes: int = 30
    
    # Trade size restrictions during circuit breakers
    warning_max_trade_size_pct: Decimal = Decimal("0.5")  # 50% of normal
    critical_max_trade_size_pct: Decimal = Decimal("0.2")  # 20% of normal
    emergency_max_trade_size_pct: Decimal = Decimal("0.0")  # No trading
    
    # System settings
    max_concurrent_breakers: int = 5
    enable_graduated_response: bool = True
    require_manual_reset_for_emergency: bool = True
    auto_recovery_enabled: bool = False
    monitoring_interval_seconds: int = 10
    
    # Ensemble monitoring settings
    enable_ensemble_monitoring: bool = True
    ensemble_performance_threshold: Decimal = Decimal("0.70")  # 70% minimum ensemble performance
    ensemble_degradation_threshold: Decimal = Decimal("0.50")  # 50% degradation threshold
    model_failure_threshold: int = 2  # Max failed models before circuit breaker
    ensemble_confidence_threshold: Decimal = Decimal("0.60")  # 60% minimum ensemble confidence
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate thresholds are positive
        if self.price_drop_warning_pct <= 0:
            raise ValueError("Price drop warning threshold must be positive")
        
        if self.volume_spike_warning_multiplier <= 1:
            raise ValueError("Volume spike multiplier must be greater than 1")
        
        if self.warning_cooldown_minutes < 0:
            raise ValueError("Cooldown period must be positive")
        
        # Validate threshold ordering
        if self.price_drop_critical_pct <= self.price_drop_warning_pct:
            raise ValueError("Critical threshold must be greater than warning")
        
        if self.price_drop_emergency_pct <= self.price_drop_critical_pct:
            raise ValueError("Emergency threshold must be greater than critical")
    
    @classmethod
    def production_config(cls) -> "CircuitBreakerConfig":
        """Create production-safe configuration with stricter thresholds."""
        return cls(
            # Stricter price movement thresholds
            price_drop_warning_pct=Decimal("0.03"),  # 3%
            price_drop_critical_pct=Decimal("0.05"),  # 5%
            price_drop_emergency_pct=Decimal("0.10"),  # 10%
            
            # Stricter volume thresholds
            volume_spike_warning_multiplier=Decimal("2.0"),  # 2x
            volume_spike_critical_multiplier=Decimal("3.0"),  # 3x
            volume_spike_emergency_multiplier=Decimal("5.0"),  # 5x
            
            # Stricter volatility thresholds
            volatility_warning_pct=Decimal("0.10"),  # 10%
            volatility_critical_pct=Decimal("0.20"),  # 20%
            volatility_emergency_pct=Decimal("0.30"),  # 30%
            
            # Longer cooldowns for production safety
            warning_cooldown_minutes=10,
            critical_cooldown_minutes=30,
            emergency_cooldown_minutes=60,
            
            # Require manual intervention for emergency situations
            require_manual_reset_for_emergency=True,
            auto_recovery_enabled=False
        )


@dataclass
class MarketData:
    """Market data for circuit breaker analysis."""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid_ask_spread: Decimal
    market_cap: Decimal
    volatility_24h: Decimal
    ensemble_performance: Optional[Dict[str, Any]] = None  # Ensemble performance metrics
    ensemble_confidence: Optional[Decimal] = None  # Overall ensemble confidence
    
    def calculate_price_change_pct(self, previous_price: Decimal) -> Decimal:
        """Calculate price change percentage."""
        if previous_price <= 0:
            return Decimal("0")
        return (self.price - previous_price) / previous_price


@dataclass
class PriceMovementData:
    """Price movement analysis data."""
    symbol: str
    current_price: Decimal
    previous_price: Decimal
    change_pct: Decimal
    change_direction: str  # "up", "down", "stable"
    timestamp: datetime
    analysis_window_minutes: int


@dataclass
class VolumeData:
    """Volume analysis data."""
    symbol: str
    current_volume: Decimal
    average_volume: Decimal
    volume_multiplier: Decimal
    timestamp: datetime
    analysis_window_minutes: int


@dataclass
class VolatilityData:
    """Volatility analysis data."""
    symbol: str
    current_volatility: Decimal
    threshold_exceeded: bool
    timestamp: datetime
    calculation_method: str


@dataclass
class CircuitBreakerResult:
    """Result of circuit breaker trigger check."""
    should_trigger: bool
    breaker_type: Optional[CircuitBreakerType] = None
    level: Optional[CircuitBreakerLevel] = None
    reason: Optional[CircuitBreakerReason] = None
    message: Optional[str] = None
    symbol: Optional[str] = None
    market_data: Optional[MarketData] = None
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


@dataclass
class TradingDecisionResult:
    """Result of trading decision evaluation."""
    is_allowed: bool
    restrictions: List[str] = field(default_factory=list)
    max_trade_size: Optional[Decimal] = None
    reason: Optional[str] = None
    active_breakers: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerStatus:
    """Current circuit breaker system status."""
    total_active_breakers: int
    active_breakers: Dict[str, Any] = field(default_factory=dict)
    system_status: str = "operational"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CircuitBreakerActivation:
    """Record of circuit breaker activation."""
    activation_id: UUID = field(default_factory=uuid4)
    breaker_type: CircuitBreakerType = CircuitBreakerType.MANUAL
    level: CircuitBreakerLevel = CircuitBreakerLevel.WARNING
    reason: CircuitBreakerReason = CircuitBreakerReason.MANUAL_INTERVENTION
    symbol: str = ""
    message: str = ""
    activated_at: datetime = field(default_factory=datetime.now)
    deactivated_at: Optional[datetime] = None
    triggered_by: str = "system"
    is_active: bool = True
    cooldown_until: Optional[datetime] = None


@dataclass
class MarketCondition:
    """Market condition assessment."""
    overall_sentiment: str  # "bullish", "bearish", "neutral", "volatile"
    stress_level: str  # "low", "medium", "high", "extreme"
    affected_assets: List[str] = field(default_factory=list)
    market_wide_impact: bool = False
    assessment_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingResult:
    """Result of processing market data through circuit breakers."""
    triggered_breakers: List[CircuitBreakerResult] = field(default_factory=list)
    market_condition: Optional[MarketCondition] = None
    trading_allowed: bool = True
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of circuit breaker recovery attempt."""
    recovery_attempted: bool
    was_successful: bool = False
    recovery_method: str = "auto"  # "auto", "manual"
    recovery_timestamp: Optional[datetime] = None
    conditions_met: List[str] = field(default_factory=list)
    remaining_restrictions: List[str] = field(default_factory=list)


@dataclass
class ManualOperationResult:
    """Result of manual circuit breaker operation."""
    was_activated: bool = False
    was_deactivated: bool = False
    level: Optional[CircuitBreakerLevel] = None
    operation_timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    reason: Optional[str] = None


class TradingCircuitBreaker:
    """
    Comprehensive trading circuit breaker system for extreme market conditions.
    
    Provides automated detection and response to:
    - Price movement extremes (drops and spikes)
    - Volume spike anomalies
    - Excessive volatility
    - Market-wide crashes and instability
    - Flash crash events
    
    Features graduated response levels and integration with emergency stop systems.
    """
    
    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        emergency_stop_controller: Optional[EmergencyStopController] = None
    ):
        """Initialize trading circuit breaker system."""
        if emergency_stop_controller is None:
            raise ValueError("EmergencyStopController is required")
        
        self.config = config or CircuitBreakerConfig()
        self.emergency_stop_controller = emergency_stop_controller
        
        # Active circuit breakers by symbol
        self.active_breakers: Dict[str, CircuitBreakerActivation] = {}
        
        # Circuit breaker history
        self.breaker_history: List[CircuitBreakerActivation] = []
        
        # Market data history for analysis
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = defaultdict(list)
        self.volume_history: Dict[str, List[Tuple[datetime, Decimal]]] = defaultdict(list)
        self.volatility_history: Dict[str, List[Tuple[datetime, Decimal]]] = defaultdict(list)
        
        # Cooldown tracking
        self.cooldown_periods: Dict[Tuple[str, CircuitBreakerType, CircuitBreakerLevel], datetime] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics tracking
        self.activation_count = 0
        self.deactivation_count = 0
        self.manual_intervention_count = 0
        
        logger.info("TradingCircuitBreaker initialized", config=self.config)
    
    async def can_execute_trade(self, symbol: str, trade_amount: Decimal) -> TradingDecisionResult:
        """Check if trade can be executed given current circuit breaker state."""
        # Check emergency stop controller first
        if self.emergency_stop_controller.is_global_emergency_stopped:
            return TradingDecisionResult(
                is_allowed=False,
                restrictions=["global_emergency_stop"],
                reason="Global emergency stop is active"
            )
        
        # Check for active circuit breakers affecting this symbol
        active_restrictions = []
        max_trade_size = trade_amount
        
        if symbol in self.active_breakers:
            breaker = self.active_breakers[symbol]
            
            if breaker.level == CircuitBreakerLevel.WARNING:
                # Warning level: allow trading with size restrictions
                max_trade_size = trade_amount * self.config.warning_max_trade_size_pct
                active_restrictions.append(f"{breaker.breaker_type.value}_warning")
                
            elif breaker.level == CircuitBreakerLevel.CRITICAL:
                # Critical level: severely restrict trading
                max_trade_size = trade_amount * self.config.critical_max_trade_size_pct
                active_restrictions.append(f"{breaker.breaker_type.value}_critical")
                
            elif breaker.level == CircuitBreakerLevel.EMERGENCY:
                # Emergency level: halt trading completely
                return TradingDecisionResult(
                    is_allowed=False,
                    restrictions=["emergency_circuit_breaker"],
                    reason=f"Emergency circuit breaker active: {breaker.message}",
                    active_breakers=[breaker.breaker_type.value]
                )
        
        # Check for market-wide circuit breakers
        market_breakers = [b for b in self.active_breakers.values() 
                          if b.breaker_type == CircuitBreakerType.MARKET_WIDE and b.is_active]
        
        if market_breakers:
            emergency_breakers = [b for b in market_breakers if b.level == CircuitBreakerLevel.EMERGENCY]
            if emergency_breakers:
                return TradingDecisionResult(
                    is_allowed=False,
                    restrictions=["market_wide_emergency"],
                    reason="Market-wide emergency circuit breaker active"
                )
        
        return TradingDecisionResult(
            is_allowed=True,
            restrictions=active_restrictions,
            max_trade_size=max_trade_size,
            active_breakers=[b.breaker_type.value for b in self.active_breakers.values() if b.is_active]
        )
    
    async def check_price_movement_triggers(self, market_data: MarketData) -> CircuitBreakerResult:
        """Check for price movement circuit breaker triggers."""
        symbol = market_data.symbol
        current_price = market_data.price
        
        # Need price history for comparison
        if symbol not in self.price_history or not self.price_history[symbol]:
            # Store current price and return no trigger
            self.price_history[symbol].append((market_data.timestamp, current_price))
            return CircuitBreakerResult(should_trigger=False)
        
        # Get most recent price for comparison
        recent_prices = [
            (timestamp, price) for timestamp, price in self.price_history[symbol]
            if market_data.timestamp - timestamp <= timedelta(minutes=self.config.price_movement_window_minutes)
        ]
        
        if not recent_prices:
            self.price_history[symbol].append((market_data.timestamp, current_price))
            return CircuitBreakerResult(should_trigger=False)
        
        # Calculate price change from earliest price in window
        earliest_price = recent_prices[0][1]
        price_change_pct = abs(current_price - earliest_price) / earliest_price if earliest_price > 0 else Decimal("0")
        
        # Determine if it's a drop or spike
        is_drop = current_price < earliest_price
        
        # Check price drop thresholds
        if is_drop:
            if price_change_pct >= self.config.price_drop_emergency_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.EMERGENCY,
                    reason=CircuitBreakerReason.PRICE_DROP_EMERGENCY,
                    message=f"Emergency price drop: {price_change_pct * 100:.2f}% decline",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
            elif price_change_pct >= self.config.price_drop_critical_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.CRITICAL,
                    reason=CircuitBreakerReason.PRICE_DROP_CRITICAL,
                    message=f"Critical price drop: {price_change_pct * 100:.2f}% decline",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
            elif price_change_pct >= self.config.price_drop_warning_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.WARNING,
                    reason=CircuitBreakerReason.PRICE_DROP_WARNING,
                    message=f"Price drop warning: {price_change_pct * 100:.2f}% decline",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
        
        # Check price spike thresholds
        else:
            if price_change_pct >= self.config.price_spike_emergency_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.EMERGENCY,
                    reason=CircuitBreakerReason.PRICE_SPIKE_EMERGENCY,
                    message=f"Emergency price spike: {price_change_pct * 100:.2f}% increase",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
            elif price_change_pct >= self.config.price_spike_critical_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.CRITICAL,
                    reason=CircuitBreakerReason.PRICE_SPIKE_CRITICAL,
                    message=f"Critical price spike: {price_change_pct * 100:.2f}% increase",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
            elif price_change_pct >= self.config.price_spike_warning_pct:
                return CircuitBreakerResult(
                    should_trigger=True,
                    breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
                    level=CircuitBreakerLevel.WARNING,
                    reason=CircuitBreakerReason.PRICE_SPIKE_WARNING,
                    message=f"Price spike warning: {price_change_pct * 100:.2f}% increase",
                    symbol=symbol,
                    market_data=market_data,
                    triggered_at=market_data.timestamp
                )
        
        # Store current price for future comparisons
        self.price_history[symbol].append((market_data.timestamp, current_price))
        
        # Clean old price history
        cutoff_time = market_data.timestamp - timedelta(minutes=self.config.price_movement_window_minutes * 2)
        self.price_history[symbol] = [
            (timestamp, price) for timestamp, price in self.price_history[symbol]
            if timestamp >= cutoff_time
        ]
        
        return CircuitBreakerResult(should_trigger=False)
    
    async def check_volume_spike_triggers(self, market_data: MarketData) -> CircuitBreakerResult:
        """Check for volume spike circuit breaker triggers."""
        symbol = market_data.symbol
        current_volume = market_data.volume
        
        # Store current volume
        self.volume_history[symbol].append((market_data.timestamp, current_volume))
        
        # Need volume history for baseline calculation
        recent_volumes = [
            volume for timestamp, volume in self.volume_history[symbol]
            if market_data.timestamp - timestamp <= timedelta(minutes=self.config.volume_analysis_window_minutes)
        ]
        
        if len(recent_volumes) < 3:
            return CircuitBreakerResult(should_trigger=False)
        
        # Calculate average volume (excluding current volume)
        avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1]) if len(recent_volumes) > 1 else current_volume
        
        if avg_volume <= 0:
            return CircuitBreakerResult(should_trigger=False)
        
        # Calculate volume multiplier
        volume_multiplier = current_volume / avg_volume
        
        # Check volume spike thresholds
        if volume_multiplier >= self.config.volume_spike_emergency_multiplier:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLUME_SPIKE,
                level=CircuitBreakerLevel.EMERGENCY,
                reason=CircuitBreakerReason.VOLUME_SPIKE_EMERGENCY,
                message=f"Emergency volume spike: {volume_multiplier:.1f}x normal volume",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        elif volume_multiplier >= self.config.volume_spike_critical_multiplier:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLUME_SPIKE,
                level=CircuitBreakerLevel.CRITICAL,
                reason=CircuitBreakerReason.VOLUME_SPIKE_CRITICAL,
                message=f"Critical volume spike: {volume_multiplier:.1f}x normal volume",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        elif volume_multiplier >= self.config.volume_spike_warning_multiplier:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLUME_SPIKE,
                level=CircuitBreakerLevel.WARNING,
                reason=CircuitBreakerReason.VOLUME_SPIKE_WARNING,
                message=f"Volume spike warning: {volume_multiplier:.1f}x normal volume",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        
        # Clean old volume history
        cutoff_time = market_data.timestamp - timedelta(minutes=self.config.volume_analysis_window_minutes * 2)
        self.volume_history[symbol] = [
            (timestamp, volume) for timestamp, volume in self.volume_history[symbol]
            if timestamp >= cutoff_time
        ]
        
        return CircuitBreakerResult(should_trigger=False)
    
    async def check_volatility_triggers(self, market_data: MarketData) -> CircuitBreakerResult:
        """Check for volatility-based circuit breaker triggers."""
        symbol = market_data.symbol
        current_volatility = market_data.volatility_24h
        
        # Check volatility thresholds
        if current_volatility >= self.config.volatility_emergency_pct:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.EMERGENCY,
                reason=CircuitBreakerReason.VOLATILITY_EMERGENCY,
                message=f"Emergency volatility: {current_volatility * 100:.2f}% volatility",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        elif current_volatility >= self.config.volatility_critical_pct:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.CRITICAL,
                reason=CircuitBreakerReason.VOLATILITY_CRITICAL,
                message=f"Critical volatility: {current_volatility * 100:.2f}% volatility",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        elif current_volatility >= self.config.volatility_warning_pct:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.WARNING,
                reason=CircuitBreakerReason.VOLATILITY_WARNING,
                message=f"Volatility warning: {current_volatility * 100:.2f}% volatility",
                symbol=symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    async def check_market_wide_triggers(self, market_data_list: List[MarketData]) -> CircuitBreakerResult:
        """Check for market-wide circuit breaker triggers."""
        if len(market_data_list) < self.config.market_wide_asset_threshold:
            return CircuitBreakerResult(should_trigger=False)
        
        # Check how many assets are experiencing significant drops
        assets_in_decline = 0
        total_market_cap_change = Decimal("0")
        
        for market_data in market_data_list:
            symbol = market_data.symbol
            
            if symbol in self.price_history and self.price_history[symbol]:
                # Get price from analysis window ago
                cutoff_time = market_data.timestamp - timedelta(minutes=self.config.price_movement_window_minutes)
                historical_prices = [
                    price for timestamp, price in self.price_history[symbol]
                    if timestamp >= cutoff_time
                ]
                
                if historical_prices:
                    earliest_price = historical_prices[0]
                    price_change_pct = (market_data.price - earliest_price) / earliest_price if earliest_price > 0 else Decimal("0")
                    
                    # Check if this asset is in significant decline
                    if price_change_pct <= -self.config.market_crash_threshold_pct:
                        assets_in_decline += 1
                        
                        # Weight by market cap for total impact
                        market_cap_impact = price_change_pct * (market_data.market_cap / Decimal("1000000000"))  # Normalize
                        total_market_cap_change += market_cap_impact
        
        # Trigger market-wide circuit breaker if enough assets are affected
        if assets_in_decline >= self.config.market_wide_asset_threshold:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.EMERGENCY,
                reason=CircuitBreakerReason.MARKET_CRASH,
                message=f"Market crash detected: {assets_in_decline} assets in decline",
                symbol="MARKET",
                triggered_at=datetime.now()
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    async def check_flash_crash_triggers(self, market_data: MarketData) -> CircuitBreakerResult:
        """Check for flash crash detection within time window."""
        symbol = market_data.symbol
        current_price = market_data.price
        current_time = market_data.timestamp
        
        # Check for rapid price decline within flash crash time window
        flash_window_start = current_time - timedelta(minutes=self.config.flash_crash_time_window_minutes)
        
        recent_prices = [
            (timestamp, price) for timestamp, price in self.price_history[symbol]
            if timestamp >= flash_window_start
        ]
        
        if not recent_prices:
            return CircuitBreakerResult(should_trigger=False)
        
        # Find highest price in the time window
        highest_price = max(price for _, price in recent_prices)
        
        if highest_price <= 0:
            return CircuitBreakerResult(should_trigger=False)
        
        # Calculate drop from highest price in window
        price_drop_pct = (highest_price - current_price) / highest_price
        
        if price_drop_pct >= self.config.flash_crash_threshold_pct:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.FLASH_CRASH,
                level=CircuitBreakerLevel.EMERGENCY,
                reason=CircuitBreakerReason.FLASH_CRASH,
                message=f"Flash crash detected: {price_drop_pct * 100:.2f}% drop in {self.config.flash_crash_time_window_minutes} minutes",
                symbol=symbol,
                market_data=market_data,
                triggered_at=current_time
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    async def check_ensemble_performance_triggers(self, market_data: MarketData) -> CircuitBreakerResult:
        """Check for ensemble performance-based circuit breaker triggers."""
        if not self.config.enable_ensemble_monitoring:
            return CircuitBreakerResult(should_trigger=False)
        
        if not market_data.ensemble_performance:
            return CircuitBreakerResult(should_trigger=False)
        
        ensemble_perf = market_data.ensemble_performance
        overall_accuracy = Decimal(str(ensemble_perf.get('overall_accuracy', 1.0)))
        failed_models = ensemble_perf.get('failed_models', [])
        ensemble_confidence = market_data.ensemble_confidence or Decimal("1.0")
        
        # Check for too many model failures
        if len(failed_models) > self.config.model_failure_threshold:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.CRITICAL,
                reason=CircuitBreakerReason.SUSPICIOUS_ACTIVITY,
                message=f"Too many model failures: {len(failed_models)} models failed",
                symbol=market_data.symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        
        # Check ensemble performance degradation
        if overall_accuracy < self.config.ensemble_degradation_threshold:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.CRITICAL,
                reason=CircuitBreakerReason.VOLATILITY_CRITICAL,
                message=f"Ensemble performance degraded: {overall_accuracy:.2f} accuracy",
                symbol=market_data.symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        
        # Check ensemble confidence
        if ensemble_confidence < self.config.ensemble_confidence_threshold:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.WARNING,
                reason=CircuitBreakerReason.VOLATILITY_WARNING,
                message=f"Low ensemble confidence: {ensemble_confidence:.2f}",
                symbol=market_data.symbol,
                market_data=market_data,
                triggered_at=market_data.timestamp
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    def assess_ensemble_health(self, ensemble_health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall health of ensemble system."""
        overall_accuracy = ensemble_health_data.get('overall_accuracy', 0.0)
        model_failures = ensemble_health_data.get('model_failures', 0)
        ensemble_confidence = ensemble_health_data.get('ensemble_confidence', 0.0)
        prediction_variance = ensemble_health_data.get('prediction_variance', 0.0)
        
        is_healthy = True
        failing_components = []
        
        # Check accuracy
        if overall_accuracy < float(self.config.ensemble_performance_threshold):
            is_healthy = False
            failing_components.append('model_accuracy')
        
        # Check model failures
        if model_failures > self.config.model_failure_threshold:
            is_healthy = False
            failing_components.append('model_availability')
        
        # Check confidence
        if ensemble_confidence < float(self.config.ensemble_confidence_threshold):
            is_healthy = False
            failing_components.append('ensemble_confidence')
        
        # Check prediction stability
        if prediction_variance > 0.3:  # High variance threshold
            is_healthy = False
            failing_components.append('prediction_stability')
        
        # Determine degradation severity
        if len(failing_components) >= 3:
            severity = 'critical'
        elif len(failing_components) >= 2:
            severity = 'moderate'
        elif len(failing_components) >= 1:
            severity = 'mild'
        else:
            severity = 'none'
        
        return {
            'is_healthy': is_healthy,
            'degradation_severity': severity,
            'failing_components': failing_components,
            'overall_accuracy': overall_accuracy,
            'model_failures': model_failures,
            'ensemble_confidence': ensemble_confidence,
            'prediction_variance': prediction_variance
        }
    
    def check_model_failure_threshold(self, model_health_data: Dict[str, Any]) -> CircuitBreakerResult:
        """Check if model failure threshold is exceeded."""
        failed_models = model_health_data.get('failed_models', 0)
        total_models = model_health_data.get('total_models', 5)
        
        if failed_models > self.config.model_failure_threshold:
            severity = CircuitBreakerLevel.CRITICAL if failed_models >= total_models // 2 else CircuitBreakerLevel.WARNING
            
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.MARKET_WIDE,
                level=severity,
                reason=CircuitBreakerReason.SUSPICIOUS_ACTIVITY,
                message=f"Model failure threshold exceeded: {failed_models}/{total_models} models failed"
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    def validate_ensemble_recovery(self, recovery_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ensemble recovery conditions."""
        ensemble_accuracy = recovery_metrics.get('ensemble_accuracy', 0.0)
        model_health_scores = recovery_metrics.get('model_health_scores', {})
        ensemble_confidence = recovery_metrics.get('ensemble_confidence', 0.0)
        stability_window_minutes = recovery_metrics.get('stability_window_minutes', 0)
        
        is_recovered = True
        recovery_conditions = []
        
        # Check ensemble accuracy recovery
        if ensemble_accuracy >= float(self.config.ensemble_performance_threshold):
            recovery_conditions.append('accuracy_recovered')
        else:
            is_recovered = False
        
        # Check individual model health
        healthy_models = sum(1 for score in model_health_scores.values() if score >= 0.7)
        if healthy_models >= 3:  # At least 3 healthy models
            recovery_conditions.append('model_health_recovered')
        else:
            is_recovered = False
        
        # Check ensemble confidence
        if ensemble_confidence >= float(self.config.ensemble_confidence_threshold):
            recovery_conditions.append('confidence_recovered')
        else:
            is_recovered = False
        
        # Check stability duration
        if stability_window_minutes >= 10:  # Stable for at least 10 minutes
            recovery_conditions.append('stability_validated')
        else:
            is_recovered = False
        
        confidence_level = len(recovery_conditions) / 4.0  # 4 total conditions
        
        return {
            'is_recovered': is_recovered,
            'confidence_level': confidence_level,
            'stability_validated': 'stability_validated' in recovery_conditions,
            'recovery_conditions_met': recovery_conditions,
            'recovery_duration_minutes': stability_window_minutes
        }
    
    def handle_fallback_integration(self, fallback_scenario: Dict[str, Any]) -> CircuitBreakerResult:
        """Handle circuit breaker integration with fallback strategies."""
        primary_models_failed = fallback_scenario.get('primary_models_failed', [])
        fallback_models_active = fallback_scenario.get('fallback_models_active', [])
        fallback_confidence = fallback_scenario.get('fallback_confidence', 0.0)
        
        # If fallback is working well, don't trigger circuit breaker
        if (len(fallback_models_active) >= 2 and 
            fallback_confidence >= float(self.config.ensemble_confidence_threshold) * 0.8):
            return CircuitBreakerResult(
                should_trigger=False,
                message=f"Fallback models active: {', '.join(fallback_models_active)}"
            )
        
        # If fallback is struggling, consider circuit breaker
        if fallback_confidence < float(self.config.ensemble_confidence_threshold) * 0.6:
            return CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.MANUAL,
                level=CircuitBreakerLevel.WARNING,
                reason=CircuitBreakerReason.SUSPICIOUS_ACTIVITY,
                message="Fallback models underperforming, monitoring closely"
            )
        
        return CircuitBreakerResult(should_trigger=False)
    
    def _get_ensemble_confidence(self, symbol: str) -> float:
        """Get current ensemble confidence for a symbol."""
        # This would integrate with the actual ensemble system
        # For now, return a default value
        return 0.75
    
    async def can_trigger_circuit_breaker(
        self,
        breaker_type: CircuitBreakerType,
        level: CircuitBreakerLevel,
        symbol: str
    ) -> bool:
        """Check if circuit breaker can be triggered (not in cooldown)."""
        cooldown_key = (symbol, breaker_type, level)
        
        if cooldown_key in self.cooldown_periods:
            cooldown_end = self.cooldown_periods[cooldown_key]
            if datetime.now() < cooldown_end:
                return False
            else:
                # Cooldown expired, remove it
                del self.cooldown_periods[cooldown_key]
        
        return True
    
    async def activate_circuit_breaker(
        self,
        breaker_type: CircuitBreakerType,
        level: CircuitBreakerLevel,
        reason: CircuitBreakerReason,
        symbol: str,
        message: str,
        triggered_by: str = "system"
    ) -> CircuitBreakerActivation:
        """Activate a circuit breaker."""
        # Check if can trigger (cooldown)
        can_trigger = await self.can_trigger_circuit_breaker(breaker_type, level, symbol)
        if not can_trigger:
            logger.warning("Circuit breaker in cooldown", 
                          breaker_type=breaker_type.value, 
                          level=level.value, 
                          symbol=symbol)
            # Return existing activation if in cooldown
            if symbol in self.active_breakers:
                return self.active_breakers[symbol]
        
        # Calculate cooldown period
        cooldown_minutes = {
            CircuitBreakerLevel.WARNING: self.config.warning_cooldown_minutes,
            CircuitBreakerLevel.CRITICAL: self.config.critical_cooldown_minutes,
            CircuitBreakerLevel.EMERGENCY: self.config.emergency_cooldown_minutes
        }.get(level, self.config.warning_cooldown_minutes)
        
        cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        # Create activation record
        activation = CircuitBreakerActivation(
            breaker_type=breaker_type,
            level=level,
            reason=reason,
            symbol=symbol,
            message=message,
            triggered_by=triggered_by,
            cooldown_until=cooldown_until
        )
        
        # Store activation
        self.active_breakers[symbol] = activation
        self.breaker_history.append(activation)
        
        # Set cooldown period
        cooldown_key = (symbol, breaker_type, level)
        self.cooldown_periods[cooldown_key] = cooldown_until
        
        self.activation_count += 1
        
        # Trigger emergency stop for emergency level breakers
        if level == CircuitBreakerLevel.EMERGENCY:
            await self._trigger_emergency_stop_integration(activation)
        
        logger.critical(
            "Circuit breaker activated",
            breaker_type=breaker_type.value,
            level=level.value,
            reason=reason.value,
            symbol=symbol,
            message=message,
            triggered_by=triggered_by
        )
        
        return activation
    
    async def deactivate_circuit_breaker(
        self,
        symbol: str,
        breaker_type: CircuitBreakerType,
        user_id: str = "system",
        reason: str = "Auto-recovery"
    ) -> ManualOperationResult:
        """Deactivate a circuit breaker."""
        if symbol not in self.active_breakers:
            return ManualOperationResult(was_deactivated=False, reason="No active breaker found")
        
        activation = self.active_breakers[symbol]
        
        # Check if this is the correct breaker type
        if activation.breaker_type != breaker_type:
            return ManualOperationResult(was_deactivated=False, reason="Breaker type mismatch")
        
        # Deactivate the breaker
        activation.is_active = False
        activation.deactivated_at = datetime.now()
        
        # Remove from active breakers
        del self.active_breakers[symbol]
        
        self.deactivation_count += 1
        
        logger.info(
            "Circuit breaker deactivated",
            symbol=symbol,
            breaker_type=breaker_type.value,
            user_id=user_id,
            reason=reason
        )
        
        return ManualOperationResult(
            was_deactivated=True,
            level=activation.level,
            user_id=user_id,
            reason=reason
        )
    
    async def activate_manual_circuit_breaker(
        self,
        symbol: str,
        level: CircuitBreakerLevel,
        reason: str,
        user_id: str
    ) -> ManualOperationResult:
        """Manually activate a circuit breaker."""
        self.manual_intervention_count += 1
        
        activation = await self.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.MANUAL,
            level=level,
            reason=CircuitBreakerReason.MANUAL_INTERVENTION,
            symbol=symbol,
            message=f"Manual intervention: {reason}",
            triggered_by=user_id
        )
        
        return ManualOperationResult(
            was_activated=True,
            level=level,
            user_id=user_id,
            reason=reason
        )
    
    async def reset_emergency_circuit_breaker(
        self,
        user_id: str,
        auth_token: str
    ) -> ManualOperationResult:
        """Reset emergency circuit breakers (requires authentication)."""
        if not user_id or not auth_token:
            raise ValueError("Authentication required for emergency reset")
        
        # In production, validate auth_token properly
        if not await self._validate_emergency_auth(user_id, auth_token):
            raise ValueError("Invalid authentication credentials")
        
        emergency_breakers = [
            (symbol, activation) for symbol, activation in self.active_breakers.items()
            if activation.level == CircuitBreakerLevel.EMERGENCY and activation.is_active
        ]
        
        for symbol, activation in emergency_breakers:
            await self.deactivate_circuit_breaker(
                symbol=symbol,
                breaker_type=activation.breaker_type,
                user_id=user_id,
                reason="Emergency manual reset"
            )
        
        return ManualOperationResult(
            was_deactivated=len(emergency_breakers) > 0,
            user_id=user_id,
            reason="Emergency manual reset"
        )
    
    async def check_recovery_conditions(self, market_data: MarketData) -> bool:
        """Check if conditions are suitable for circuit breaker recovery."""
        symbol = market_data.symbol
        
        if symbol not in self.active_breakers:
            return True
        
        activation = self.active_breakers[symbol]
        
        # Emergency breakers require manual reset
        if activation.level == CircuitBreakerLevel.EMERGENCY and self.config.require_manual_reset_for_emergency:
            return False
        
        # Check if conditions have improved based on breaker type
        if activation.breaker_type == CircuitBreakerType.PRICE_MOVEMENT:
            # Check if price has stabilized
            return await self._check_price_stability(market_data)
        elif activation.breaker_type == CircuitBreakerType.VOLUME_SPIKE:
            # Check if volume has normalized
            return await self._check_volume_normalization(market_data)
        elif activation.breaker_type == CircuitBreakerType.VOLATILITY:
            # Check if volatility has decreased
            return market_data.volatility_24h < self.config.volatility_warning_pct
        
        return False
    
    async def attempt_auto_recovery(self, symbol: str) -> RecoveryResult:
        """Attempt automatic recovery of circuit breakers."""
        if not self.config.auto_recovery_enabled:
            return RecoveryResult(
                recovery_attempted=False,
                recovery_method="auto",
                remaining_restrictions=["auto_recovery_disabled"]
            )
        
        if symbol not in self.active_breakers:
            return RecoveryResult(
                recovery_attempted=False,
                remaining_restrictions=["no_active_breaker"]
            )
        
        activation = self.active_breakers[symbol]
        
        # Cannot auto-recover emergency breakers
        if activation.level == CircuitBreakerLevel.EMERGENCY:
            return RecoveryResult(
                recovery_attempted=False,
                recovery_method="auto",
                remaining_restrictions=["manual_reset_required"]
            )
        
        # Check if cooldown has expired
        if activation.cooldown_until and datetime.now() < activation.cooldown_until:
            return RecoveryResult(
                recovery_attempted=False,
                recovery_method="auto",
                remaining_restrictions=["cooldown_active"]
            )
        
        # Attempt recovery
        result = await self.deactivate_circuit_breaker(
            symbol=symbol,
            breaker_type=activation.breaker_type,
            user_id="auto_recovery",
            reason="Automatic recovery"
        )
        
        return RecoveryResult(
            recovery_attempted=True,
            was_successful=result.was_deactivated,
            recovery_method="auto",
            recovery_timestamp=datetime.now(),
            conditions_met=["cooldown_expired", "conditions_improved"]
        )
    
    async def process_market_data(self, market_data: MarketData) -> ProcessingResult:
        """Process market data through all circuit breaker checks."""
        triggered_breakers = []
        
        # Check each type of circuit breaker
        checks = [
            self.check_price_movement_triggers(market_data),
            self.check_volume_spike_triggers(market_data),
            self.check_volatility_triggers(market_data),
            self.check_flash_crash_triggers(market_data)
        ]
        
        for check in checks:
            result = await check
            if result.should_trigger:
                # Check if can trigger (not in cooldown)
                can_trigger = await self.can_trigger_circuit_breaker(
                    result.breaker_type,
                    result.level,
                    result.symbol
                )
                
                if can_trigger:
                    # Activate the circuit breaker
                    await self.activate_circuit_breaker(
                        breaker_type=result.breaker_type,
                        level=result.level,
                        reason=result.reason,
                        symbol=result.symbol,
                        message=result.message
                    )
                    triggered_breakers.append(result)
        
        # Assess overall market condition
        market_condition = self._assess_market_condition(market_data, triggered_breakers)
        
        # Determine if trading should be allowed
        trade_decision = await self.can_execute_trade(market_data.symbol, Decimal("1000"))  # Test amount
        
        return ProcessingResult(
            triggered_breakers=triggered_breakers,
            market_condition=market_condition,
            trading_allowed=trade_decision.is_allowed,
            recommendations=self._generate_recommendations(triggered_breakers)
        )
    
    def get_circuit_breaker_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker system status."""
        active_breaker_details = {}
        
        for symbol, activation in self.active_breakers.items():
            active_breaker_details[symbol] = {
                "type": activation.breaker_type.value,
                "level": activation.level,
                "reason": activation.reason.value,
                "message": activation.message,
                "activated_at": activation.activated_at,
                "cooldown_until": activation.cooldown_until
            }
        
        return CircuitBreakerStatus(
            total_active_breakers=len(self.active_breakers),
            active_breakers=active_breaker_details,
            system_status="emergency" if any(
                a.level == CircuitBreakerLevel.EMERGENCY for a in self.active_breakers.values()
            ) else "operational"
        )
    
    def get_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        # Calculate durations
        total_duration = 0
        duration_count = 0
        
        for activation in self.breaker_history:
            if activation.deactivated_at:
                duration = (activation.deactivated_at - activation.activated_at).total_seconds()
                total_duration += duration
                duration_count += 1
        
        avg_duration = total_duration / duration_count if duration_count > 0 else 0
        
        # Count by type and level
        activations_by_type = defaultdict(int)
        activations_by_level = defaultdict(int)
        
        for activation in self.breaker_history:
            activations_by_type[activation.breaker_type.value] += 1
            activations_by_level[activation.level.value] += 1
        
        return {
            "total_activations": self.activation_count,
            "total_deactivations": self.deactivation_count,
            "manual_interventions": self.manual_intervention_count,
            "active_breakers_count": len(self.active_breakers),
            "activations_by_type": dict(activations_by_type),
            "activations_by_level": dict(activations_by_level),
            "average_duration_seconds": avg_duration
        }
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail of circuit breaker activities."""
        audit_events = []
        
        for activation in self.breaker_history:
            # Activation event
            audit_events.append({
                "action": "circuit_breaker_activated",
                "timestamp": activation.activated_at.isoformat(),
                "symbol": activation.symbol,
                "type": activation.breaker_type.value,
                "level": activation.level.value,
                "reason": activation.reason.value,
                "message": activation.message,
                "triggered_by": activation.triggered_by,
                "activation_id": str(activation.activation_id)
            })
            
            # Deactivation event (if applicable)
            if activation.deactivated_at:
                audit_events.append({
                    "action": "circuit_breaker_deactivated",
                    "timestamp": activation.deactivated_at.isoformat(),
                    "symbol": activation.symbol,
                    "type": activation.breaker_type.value,
                    "level": activation.level.value,
                    "activation_id": str(activation.activation_id)
                })
        
        return sorted(audit_events, key=lambda x: x["timestamp"])
    
    async def _trigger_emergency_stop_integration(self, activation: CircuitBreakerActivation) -> None:
        """Trigger emergency stop controller integration for emergency breakers."""
        try:
            # Map circuit breaker reasons to emergency stop reasons
            emergency_reason_map = {
                CircuitBreakerReason.PRICE_DROP_EMERGENCY: EmergencyStopReason.FLASH_CRASH_DETECTED,
                CircuitBreakerReason.PRICE_SPIKE_EMERGENCY: EmergencyStopReason.EXTREME_VOLATILITY,
                CircuitBreakerReason.VOLUME_SPIKE_EMERGENCY: EmergencyStopReason.EXTREME_VOLATILITY,
                CircuitBreakerReason.VOLATILITY_EMERGENCY: EmergencyStopReason.EXTREME_VOLATILITY,
                CircuitBreakerReason.MARKET_CRASH: EmergencyStopReason.FLASH_CRASH_DETECTED,
                CircuitBreakerReason.FLASH_CRASH: EmergencyStopReason.FLASH_CRASH_DETECTED
            }
            
            emergency_reason = emergency_reason_map.get(
                activation.reason, 
                EmergencyStopReason.EXTREME_VOLATILITY
            )
            
            await self.emergency_stop_controller.activate_global_emergency_stop(
                reason=emergency_reason,
                message=f"Circuit breaker triggered: {activation.message}",
                triggered_by="circuit_breaker_system"
            )
            
        except Exception as e:
            logger.error("Failed to trigger emergency stop integration", error=str(e))
    
    async def _validate_emergency_auth(self, user_id: str, auth_token: str) -> bool:
        """Validate emergency authentication (simplified implementation)."""
        # In production, this would integrate with proper authentication system
        return len(auth_token) > 0 and user_id in ["admin_user", "emergency_operator"]
    
    async def _check_price_stability(self, market_data: MarketData) -> bool:
        """Check if price has stabilized for recovery."""
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            return False
        
        # Check recent price movements
        recent_prices = [
            price for timestamp, price in self.price_history[symbol]
            if market_data.timestamp - timestamp <= timedelta(minutes=5)
        ]
        
        if len(recent_prices) < 3:
            return False
        
        # Calculate volatility of recent prices
        avg_price = sum(recent_prices) / len(recent_prices)
        price_variance = sum((price - avg_price) ** 2 for price in recent_prices) / len(recent_prices)
        price_volatility = (price_variance ** Decimal("0.5")) / avg_price if avg_price > 0 else Decimal("1")
        
        # Consider stable if volatility is below warning threshold
        return price_volatility < self.config.volatility_warning_pct
    
    async def _check_volume_normalization(self, market_data: MarketData) -> bool:
        """Check if volume has normalized for recovery."""
        symbol = market_data.symbol
        current_volume = market_data.volume
        
        if symbol not in self.volume_history:
            return False
        
        # Get baseline volume
        baseline_volumes = [
            volume for timestamp, volume in self.volume_history[symbol]
            if market_data.timestamp - timestamp <= timedelta(hours=2)
        ]
        
        if len(baseline_volumes) < 3:
            return False
        
        avg_volume = sum(baseline_volumes) / len(baseline_volumes)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else Decimal("1")
        
        # Consider normalized if volume is below warning threshold
        return volume_ratio < self.config.volume_spike_warning_multiplier
    
    def _assess_market_condition(self, market_data: MarketData, triggered_breakers: List[CircuitBreakerResult]) -> MarketCondition:
        """Assess overall market condition based on circuit breaker activity."""
        if not triggered_breakers:
            return MarketCondition(
                overall_sentiment="neutral",
                stress_level="low",
                affected_assets=[market_data.symbol]
            )
        
        emergency_breakers = [b for b in triggered_breakers if b.level == CircuitBreakerLevel.EMERGENCY]
        critical_breakers = [b for b in triggered_breakers if b.level == CircuitBreakerLevel.CRITICAL]
        
        if emergency_breakers:
            sentiment = "bearish" if any("drop" in b.message.lower() for b in emergency_breakers) else "volatile"
            stress_level = "extreme"
        elif critical_breakers:
            sentiment = "bearish" if any("drop" in b.message.lower() for b in critical_breakers) else "volatile"
            stress_level = "high"
        else:
            sentiment = "neutral"
            stress_level = "medium"
        
        market_wide_impact = len(triggered_breakers) > 1 or any(
            b.breaker_type == CircuitBreakerType.MARKET_WIDE for b in triggered_breakers
        )
        
        return MarketCondition(
            overall_sentiment=sentiment,
            stress_level=stress_level,
            affected_assets=[market_data.symbol],
            market_wide_impact=market_wide_impact
        )
    
    def _generate_recommendations(self, triggered_breakers: List[CircuitBreakerResult]) -> List[str]:
        """Generate recommendations based on triggered circuit breakers."""
        recommendations = []
        
        if not triggered_breakers:
            recommendations.append("Market conditions are stable")
            return recommendations
        
        emergency_breakers = [b for b in triggered_breakers if b.level == CircuitBreakerLevel.EMERGENCY]
        critical_breakers = [b for b in triggered_breakers if b.level == CircuitBreakerLevel.CRITICAL]
        
        if emergency_breakers:
            recommendations.append("Consider halting all trading due to emergency conditions")
            recommendations.append("Review risk management parameters")
            recommendations.append("Monitor for market recovery signs")
        elif critical_breakers:
            recommendations.append("Reduce position sizes and limit new trades")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider defensive trading strategies")
        else:
            recommendations.append("Exercise caution with new positions")
            recommendations.append("Monitor market conditions closely")
        
        return recommendations