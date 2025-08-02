"""
Financial Data Validation System

Production-ready financial data validation system for trading applications.
Implements comprehensive validation including:
- Multi-source price verification
- Outlier detection and price manipulation checks
- Stale data detection and real-time monitoring
- Volume and liquidity validation
- Cross-reference validation between data sources

Following TDD methodology - implementation satisfies comprehensive test requirements.
No mock objects in production code - all real implementations.
"""

import asyncio
import statistics
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Deque, Set
from uuid import uuid4

logger = structlog.get_logger()


class ValidationStatus(Enum):
    """Status of financial data validation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"
    PENDING = "pending"


class ValidationReason(Enum):
    """Reasons for validation results."""
    PRICE_OUTLIER_DETECTED = "price_outlier_detected"
    STALE_DATA_DETECTED = "stale_data_detected"
    LOW_VOLUME_DETECTED = "low_volume_detected"
    PRICE_MANIPULATION_DETECTED = "price_manipulation_detected"
    SOURCE_INCONSISTENCY = "source_inconsistency"
    INVALID_DATA = "invalid_data"
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    VOLATILITY_EXCESSIVE = "volatility_excessive"
    SPREAD_TOO_WIDE = "spread_too_wide"
    FLASH_CRASH_DETECTED = "flash_crash_detected"
    PUMP_DUMP_DETECTED = "pump_dump_detected"
    WASH_TRADING_DETECTED = "wash_trading_detected"


class AlertType(Enum):
    """Types of data quality alerts."""
    STALE_DATA = "stale_data"
    LOW_VOLUME = "low_volume"
    PRICE_ANOMALY = "price_anomaly"
    SOURCE_FAILURE = "source_failure"
    MANIPULATION_DETECTED = "manipulation_detected"
    CRITICAL_DEVIATION = "critical_deviation"


@dataclass
class PriceData:
    """Price data from a specific source."""
    symbol: str
    price: Decimal
    timestamp: datetime
    source: str
    volume_24h: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None
    circulating_supply: Optional[Decimal] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate price data after initialization."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.volume_24h is not None and self.volume_24h < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class VolumeData:
    """Volume and liquidity data for validation."""
    symbol: str
    volume_24h: Decimal
    liquidity_score: float  # 0-1 scale
    bid_ask_spread: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_book_depth: Optional[Decimal] = None
    trade_count_24h: Optional[int] = None
    
    def __post_init__(self):
        """Validate volume data after initialization."""
        if self.volume_24h < 0:
            raise ValueError("Volume cannot be negative")
        
        if not (0 <= self.liquidity_score <= 1):
            raise ValueError("Liquidity score must be between 0 and 1")
        
        if self.bid_ask_spread < 0:
            raise ValueError("Bid-ask spread cannot be negative")


@dataclass
class ValidationResult:
    """Result of financial data validation."""
    is_valid: bool
    status: ValidationStatus
    reasons: List[ValidationReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)
    price_data: Optional[PriceData] = None
    confidence_score: float = 1.0
    
    # Multi-source validation specific fields
    consensus_price: Optional[Decimal] = None
    source_agreement_score: Optional[float] = None
    outlier_sources: List[str] = field(default_factory=list)
    
    # Manipulation detection specific fields
    is_manipulation_detected: Optional[bool] = None
    manipulation_type: Optional[str] = None
    suspicious_patterns: List[str] = field(default_factory=list)
    
    # Cross-validation specific fields
    is_consistent: Optional[bool] = None
    consensus_data: Optional[Dict[str, Any]] = None
    reliability_score: Optional[float] = None
    
    # Fallback validation fields
    used_fallback: bool = False


@dataclass
class DataQualityAlert:
    """Alert for data quality issues."""
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    alert_type: AlertType = AlertType.PRICE_ANOMALY
    symbol: str = ""
    message: str = ""
    severity: str = "medium"  # low, medium, high, critical
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalPatternAnalysis:
    """Analysis of historical price patterns."""
    trend_direction: str  # "bullish", "bearish", "sideways"
    volatility_score: float  # 0-1 scale
    pattern_reliability: float  # 0-1 scale
    support_levels: List[Decimal] = field(default_factory=list)
    resistance_levels: List[Decimal] = field(default_factory=list)
    volume_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringResult:
    """Result of real-time monitoring setup."""
    is_active: bool
    monitoring_interval: int  # seconds
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceValidationConfig:
    """Configuration for price validation."""
    max_price_deviation_percent: Decimal = Decimal("5.0")  # 5% max deviation
    max_stale_minutes: int = 30  # 30 minutes max staleness
    min_volume_threshold: Decimal = Decimal("1000000")  # $1M minimum volume
    min_liquidity_score: float = 0.3  # 30% minimum liquidity
    max_bid_ask_spread_percent: Decimal = Decimal("2.0")  # 2% max spread
    required_sources: List[str] = field(default_factory=lambda: ["coingecko", "coinmarketcap"])
    outlier_detection_enabled: bool = True
    manipulation_detection_enabled: bool = True
    min_source_agreement_threshold: float = 0.8  # 80% agreement required
    flash_crash_threshold_percent: Decimal = Decimal("20.0")  # 20% price drop
    pump_threshold_percent: Decimal = Decimal("50.0")  # 50% price increase
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_price_deviation_percent < 0:
            raise ValueError("Price deviation percentage must be positive")
        
        if self.max_stale_minutes < 0:
            raise ValueError("Stale minutes must be positive")
        
        if self.min_volume_threshold < 0:
            raise ValueError("Volume threshold must be positive")
        
        if not (0 <= self.min_liquidity_score <= 1):
            raise ValueError("Liquidity score must be between 0 and 1")
        
        if self.max_bid_ask_spread_percent < 0:
            raise ValueError("Bid-ask spread percentage must be positive")
        
        if not (0 <= self.min_source_agreement_threshold <= 1):
            raise ValueError("Source agreement threshold must be between 0 and 1")


class ValidationError(Exception):
    """Exception raised during validation errors."""
    pass


class FinancialDataValidator:
    """
    Comprehensive financial data validation system.
    
    Provides multi-source price verification, outlier detection,
    stale data detection, price manipulation checks, volume validation,
    and real-time monitoring capabilities.
    """
    
    def __init__(self, config: Optional[PriceValidationConfig] = None):
        """Initialize financial data validator."""
        self.config = config or PriceValidationConfig()
        self.is_active = True
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
        
        # Data quality metrics tracking
        self.quality_metrics = {
            "source_reliability": defaultdict(float),
            "data_freshness": defaultdict(float),
            "price_stability": defaultdict(float),
            "volume_consistency": defaultdict(float),
            "validation_count": 0,
            "rejection_count": 0,
            "alert_count": defaultdict(int)
        }
        
        # Historical data cache for pattern analysis
        self.price_history: Dict[str, Deque[PriceData]] = defaultdict(lambda: deque(maxlen=1000))
        
        # Source reliability tracking
        self.source_reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        
        logger.info("FinancialDataValidator initialized", config=self.config)
    
    async def validate_price_data(self, price_data: Union[PriceData, Dict, None]) -> ValidationResult:
        """
        Validate single source price data.
        
        Args:
            price_data: Price data to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        if price_data is None:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=["Price data is None"]
            )
        
        try:
            # Convert dict to PriceData if necessary
            if isinstance(price_data, dict):
                price_data = PriceData(**price_data)
            
            self.quality_metrics["validation_count"] += 1
            
            # Store in history for pattern analysis
            self.price_history[price_data.symbol].append(price_data)
            
            # Run validation checks
            validation_results = []
            
            # Check data staleness
            staleness_result = await self._check_data_staleness(price_data)
            validation_results.append(staleness_result)
            
            # Check volume if available
            if price_data.volume_24h is not None:
                volume_result = await self._check_volume_threshold(price_data)
                validation_results.append(volume_result)
            
            # Check bid-ask spread if available
            if price_data.bid_price and price_data.ask_price:
                spread_result = await self._check_bid_ask_spread(price_data)
                validation_results.append(spread_result)
            
            # Combine results
            combined_result = self._combine_validation_results(validation_results)
            combined_result.price_data = price_data
            
            # Update metrics
            self._update_quality_metrics(price_data.source, combined_result)
            
            logger.debug(
                "Price data validation completed",
                symbol=price_data.symbol,
                source=price_data.source,
                is_valid=combined_result.is_valid
            )
            
            return combined_result
            
        except Exception as e:
            logger.error("Price data validation failed", error=str(e))
            self.quality_metrics["rejection_count"] += 1
            
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Validation error: {str(e)}"]
            )
    
    async def validate_multi_source_prices(self, price_data_list: List[PriceData]) -> ValidationResult:
        """
        Validate prices from multiple sources and detect outliers.
        
        Args:
            price_data_list: List of price data from different sources
            
        Returns:
            ValidationResult with consensus price and outlier detection
        """
        if not price_data_list:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=["No price data provided"]
            )
        
        try:
            # Extract prices and sources
            prices = [float(pd.price) for pd in price_data_list]
            sources = [pd.source for pd in price_data_list]
            
            # Calculate consensus price (median)
            consensus_price = Decimal(str(statistics.median(prices)))
            
            # Detect outliers
            outlier_sources = await self._detect_price_outliers(price_data_list, consensus_price)
            
            # Calculate source agreement score
            price_range = max(prices) - min(prices)
            price_std = statistics.stdev(prices) if len(prices) > 1 else 0
            agreement_score = max(0, 1 - (price_std / float(consensus_price)))
            
            # Determine validation result
            is_valid = (
                len(outlier_sources) == 0 and
                agreement_score >= self.config.min_source_agreement_threshold
            )
            
            status = ValidationStatus.APPROVED if is_valid else ValidationStatus.REJECTED
            reasons = []
            
            if len(outlier_sources) > 0:
                reasons.append(ValidationReason.PRICE_OUTLIER_DETECTED)
            
            if agreement_score < self.config.min_source_agreement_threshold:
                reasons.append(ValidationReason.SOURCE_INCONSISTENCY)
            
            return ValidationResult(
                is_valid=is_valid,
                status=status,
                reasons=reasons,
                consensus_price=consensus_price,
                source_agreement_score=agreement_score,
                outlier_sources=outlier_sources,
                metadata={
                    "price_range": price_range,
                    "price_std": price_std,
                    "source_count": len(sources)
                }
            )
            
        except Exception as e:
            logger.error("Multi-source price validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Multi-source validation error: {str(e)}"]
            )
    
    async def detect_price_manipulation(self, price_history: List[PriceData]) -> ValidationResult:
        """
        Detect potential price manipulation patterns.
        
        Args:
            price_history: Historical price data to analyze
            
        Returns:
            ValidationResult with manipulation detection results
        """
        if len(price_history) < 3:
            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                is_manipulation_detected=False,
                manipulation_type=None,
                confidence_score=0.0
            )
        
        try:
            # Sort by timestamp
            sorted_history = sorted(price_history, key=lambda x: x.timestamp)
            
            # Detect flash crash
            flash_crash_detected, confidence = await self._detect_flash_crash(sorted_history)
            
            # Detect pump and dump
            pump_dump_detected, pump_confidence = await self._detect_pump_and_dump(sorted_history)
            
            # Detect wash trading
            wash_trading_detected, wash_confidence = await self._detect_wash_trading(sorted_history)
            
            # Determine overall manipulation
            manipulations = [
                ("flash_crash", flash_crash_detected, confidence),
                ("pump_and_dump", pump_dump_detected, pump_confidence),
                ("wash_trading", wash_trading_detected, wash_confidence)
            ]
            
            detected_patterns = [(name, conf) for name, detected, conf in manipulations if detected]
            
            if detected_patterns:
                # Use highest confidence pattern
                manipulation_type, max_confidence = max(detected_patterns, key=lambda x: x[1])
                
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.PRICE_MANIPULATION_DETECTED],
                    is_manipulation_detected=True,
                    manipulation_type=manipulation_type,
                    confidence_score=max_confidence,
                    suspicious_patterns=[name for name, _ in detected_patterns]
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    status=ValidationStatus.APPROVED,
                    is_manipulation_detected=False,
                    manipulation_type=None,
                    confidence_score=0.0
                )
                
        except Exception as e:
            logger.error("Price manipulation detection failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Manipulation detection error: {str(e)}"]
            )
    
    async def validate_volume_data(self, volume_data: VolumeData) -> ValidationResult:
        """
        Validate volume and liquidity data.
        
        Args:
            volume_data: Volume data to validate
            
        Returns:
            ValidationResult with volume validation outcome
        """
        try:
            reasons = []
            warnings = []
            
            # Check volume threshold
            if volume_data.volume_24h < self.config.min_volume_threshold:
                reasons.append(ValidationReason.LOW_VOLUME_DETECTED)
                warnings.append(
                    f"Volume {volume_data.volume_24h} below threshold {self.config.min_volume_threshold}"
                )
            
            # Check liquidity score
            if volume_data.liquidity_score < self.config.min_liquidity_score:
                reasons.append(ValidationReason.LIQUIDITY_INSUFFICIENT)
                warnings.append(
                    f"Liquidity score {volume_data.liquidity_score} below threshold {self.config.min_liquidity_score}"
                )
            
            # Check bid-ask spread
            if volume_data.bid_ask_spread > self.config.max_bid_ask_spread_percent / 100:
                reasons.append(ValidationReason.SPREAD_TOO_WIDE)
                warnings.append(
                    f"Bid-ask spread {volume_data.bid_ask_spread * 100}% exceeds {self.config.max_bid_ask_spread_percent}%"
                )
            
            is_valid = len(reasons) == 0
            status = ValidationStatus.APPROVED if is_valid else ValidationStatus.REJECTED
            
            return ValidationResult(
                is_valid=is_valid,
                status=status,
                reasons=reasons,
                warnings=warnings,
                metadata={
                    "volume_24h": float(volume_data.volume_24h),
                    "liquidity_score": volume_data.liquidity_score,
                    "bid_ask_spread": float(volume_data.bid_ask_spread)
                }
            )
            
        except Exception as e:
            logger.error("Volume data validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Volume validation error: {str(e)}"]
            )
    
    async def cross_validate_sources(self, source_data: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """
        Cross-validate data between multiple sources.
        
        Args:
            source_data: Dictionary mapping source names to their data
            
        Returns:
            ValidationResult with cross-validation results
        """
        try:
            if len(source_data) < 2:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SOURCE_INCONSISTENCY],
                    warnings=["Need at least 2 sources for cross-validation"]
                )
            
            # Extract symbols present in all sources
            all_symbols = set()
            for source_symbols in source_data.values():
                all_symbols.update(source_symbols.keys())
            
            common_symbols = all_symbols.copy()
            for source_symbols in source_data.values():
                common_symbols &= set(source_symbols.keys())
            
            if not common_symbols:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SOURCE_INCONSISTENCY],
                    warnings=["No common symbols across sources"]
                )
            
            # Calculate consensus data
            consensus_data = {}
            total_reliability = 0
            
            for symbol in common_symbols:
                symbol_prices = []
                symbol_volumes = []
                
                for source, data in source_data.items():
                    if symbol in data:
                        symbol_data = data[symbol]
                        if "price" in symbol_data:
                            symbol_prices.append(float(symbol_data["price"]))
                        if "volume" in symbol_data:
                            symbol_volumes.append(float(symbol_data["volume"]))
                
                # Calculate consensus price and volume
                consensus_price = Decimal(str(statistics.median(symbol_prices))) if symbol_prices else None
                consensus_volume = Decimal(str(statistics.median(symbol_volumes))) if symbol_volumes else None
                
                # Calculate reliability for this symbol
                if symbol_prices:
                    price_std = statistics.stdev(symbol_prices) if len(symbol_prices) > 1 else 0
                    symbol_reliability = max(0, 1 - (price_std / max(symbol_prices)))
                    total_reliability += symbol_reliability
                    
                    consensus_data[symbol] = {
                        "price": consensus_price,
                        "volume": consensus_volume,
                        "reliability": symbol_reliability
                    }
            
            # Calculate overall reliability score
            reliability_score = total_reliability / len(consensus_data) if consensus_data else 0
            
            is_consistent = reliability_score >= self.config.min_source_agreement_threshold
            status = ValidationStatus.APPROVED if is_consistent else ValidationStatus.REJECTED
            
            return ValidationResult(
                is_valid=is_consistent,
                status=status,
                is_consistent=is_consistent,
                consensus_data=consensus_data,
                reliability_score=reliability_score,
                metadata={
                    "source_count": len(source_data),
                    "common_symbols_count": len(common_symbols),
                    "coverage": len(common_symbols) / len(all_symbols) if all_symbols else 0
                }
            )
            
        except Exception as e:
            logger.error("Cross-validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Cross-validation error: {str(e)}"]
            )
    
    async def start_monitoring(self) -> MonitoringResult:
        """
        Start real-time data quality monitoring.
        
        Returns:
            MonitoringResult with monitoring setup information
        """
        try:
            self.monitoring_active = True
            
            # Initialize monitoring metrics
            quality_metrics = {
                "monitoring_start_time": datetime.utcnow().isoformat(),
                "validation_rate": 0,
                "rejection_rate": 0,
                "source_health": dict(self.source_reliability),
                "alert_frequency": dict(self.quality_metrics["alert_count"])
            }
            
            logger.info("Data quality monitoring started", interval=self.monitoring_interval)
            
            return MonitoringResult(
                is_active=True,
                monitoring_interval=self.monitoring_interval,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error("Failed to start monitoring", error=str(e))
            return MonitoringResult(
                is_active=False,
                monitoring_interval=0,
                quality_metrics={"error": str(e)}
            )
    
    async def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Get current data quality metrics.
        
        Returns:
            Dictionary containing quality metrics
        """
        try:
            total_validations = self.quality_metrics["validation_count"]
            rejection_rate = (
                self.quality_metrics["rejection_count"] / total_validations
                if total_validations > 0 else 0
            )
            
            return {
                "source_reliability": dict(self.quality_metrics["source_reliability"]),
                "data_freshness": dict(self.quality_metrics["data_freshness"]),
                "price_stability": dict(self.quality_metrics["price_stability"]),
                "volume_consistency": dict(self.quality_metrics["volume_consistency"]),
                "total_validations": total_validations,
                "rejection_rate": rejection_rate,
                "alert_counts": dict(self.quality_metrics["alert_count"]),
                "monitoring_active": self.monitoring_active,
                "source_health": dict(self.source_reliability)
            }
            
        except Exception as e:
            logger.error("Failed to get quality metrics", error=str(e))
            return {"error": str(e)}
    
    async def process_data_and_generate_alerts(self, data_batch: List[Dict[str, Any]]) -> List[DataQualityAlert]:
        """
        Process data batch and generate alerts for issues.
        
        Args:
            data_batch: Batch of data to process
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        try:
            for data_item in data_batch:
                # Validate each data item
                try:
                    price_data = PriceData(**data_item)
                    result = await self.validate_price_data(price_data)
                    
                    # Generate alerts based on validation results
                    if not result.is_valid:
                        for reason in result.reasons:
                            alert = await self._create_alert_from_reason(reason, price_data, result)
                            if alert:
                                alerts.append(alert)
                                
                except Exception as e:
                    # Generate alert for invalid data structure
                    alert = DataQualityAlert(
                        alert_type=AlertType.PRICE_ANOMALY,
                        symbol=data_item.get("symbol", "UNKNOWN"),
                        message=f"Invalid data structure: {str(e)}",
                        severity="high",
                        metadata={"raw_data": data_item}
                    )
                    alerts.append(alert)
            
            # Update alert counts
            for alert in alerts:
                self.quality_metrics["alert_count"][alert.alert_type.value] += 1
            
            logger.info("Data processing completed", 
                       processed_items=len(data_batch), 
                       alerts_generated=len(alerts))
            
            return alerts
            
        except Exception as e:
            logger.error("Failed to process data batch", error=str(e))
            # Return a critical alert about processing failure
            return [DataQualityAlert(
                alert_type=AlertType.SOURCE_FAILURE,
                message=f"Data processing failed: {str(e)}",
                severity="critical"
            )]
    
    async def analyze_historical_patterns(self, historical_data: List[Dict[str, Any]]) -> HistoricalPatternAnalysis:
        """
        Analyze historical price patterns for validation context.
        
        Args:
            historical_data: Historical price data
            
        Returns:
            HistoricalPatternAnalysis with pattern insights
        """
        try:
            if len(historical_data) < 5:
                return HistoricalPatternAnalysis(
                    trend_direction="sideways",
                    volatility_score=0.0,
                    pattern_reliability=0.0
                )
            
            # Convert to PriceData objects and sort by timestamp
            price_objects = []
            for item in historical_data:
                try:
                    price_data = PriceData(**item)
                    price_objects.append(price_data)
                except Exception:
                    continue
            
            price_objects.sort(key=lambda x: x.timestamp)
            prices = [float(p.price) for p in price_objects]
            
            # Calculate trend direction
            price_change = (prices[-1] - prices[0]) / prices[0]
            if price_change > 0.05:  # 5% increase
                trend_direction = "bullish"
            elif price_change < -0.05:  # 5% decrease
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"
            
            # Calculate volatility score
            price_returns = []
            for i in range(1, len(prices)):
                return_rate = (prices[i] - prices[i-1]) / prices[i-1]
                price_returns.append(return_rate)
            
            volatility_score = statistics.stdev(price_returns) if len(price_returns) > 1 else 0.0
            volatility_score = min(volatility_score * 10, 1.0)  # Scale to 0-1
            
            # Calculate pattern reliability based on trend consistency
            trend_changes = 0
            for i in range(1, len(prices) - 1):
                if ((prices[i] > prices[i-1]) != (prices[i+1] > prices[i])):
                    trend_changes += 1
            
            pattern_reliability = max(0, 1 - (trend_changes / len(prices)))
            
            # Identify support and resistance levels
            support_levels = await self._identify_support_levels(prices)
            resistance_levels = await self._identify_resistance_levels(prices)
            
            # Calculate volume profile
            volumes = [float(p.volume_24h) for p in price_objects if p.volume_24h]
            volume_profile = {
                "avg_volume": statistics.mean(volumes) if volumes else 0,
                "volume_trend": "increasing" if len(volumes) > 1 and volumes[-1] > volumes[0] else "decreasing"
            }
            
            return HistoricalPatternAnalysis(
                trend_direction=trend_direction,
                volatility_score=volatility_score,
                pattern_reliability=pattern_reliability,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volume_profile=volume_profile
            )
            
        except Exception as e:
            logger.error("Historical pattern analysis failed", error=str(e))
            return HistoricalPatternAnalysis(
                trend_direction="sideways",
                volatility_score=0.0,
                pattern_reliability=0.0
            )
    
    async def validate_with_fallback(self, data: Any) -> ValidationResult:
        """
        Validate data with fallback mechanisms.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with fallback indication
        """
        try:
            # Try normal validation first
            if isinstance(data, (PriceData, dict)):
                return await self.validate_price_data(data)
            else:
                # Fallback for invalid data types
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.INVALID_DATA],
                    warnings=["Invalid data type for validation"],
                    used_fallback=True
                )
                
        except Exception as e:
            # Ultimate fallback
            logger.warning("Primary validation failed, using fallback", error=str(e))
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Validation failed: {str(e)}"],
                used_fallback=True
            )
    
    async def validate_batch_prices(self, price_batch: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate a batch of prices efficiently.
        
        Args:
            price_batch: List of price data dictionaries
            
        Returns:
            List of validation results
        """
        try:
            # Process batch concurrently for performance
            tasks = []
            for price_data in price_batch:
                task = self.validate_price_data(price_data)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the results
            validated_results = []
            for result in results:
                if isinstance(result, Exception):
                    validated_results.append(ValidationResult(
                        is_valid=False,
                        status=ValidationStatus.REJECTED,
                        reasons=[ValidationReason.INVALID_DATA],
                        warnings=[f"Batch validation error: {str(result)}"]
                    ))
                else:
                    validated_results.append(result)
            
            return validated_results
            
        except Exception as e:
            logger.error("Batch validation failed", error=str(e))
            # Return error results for all items
            return [ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.INVALID_DATA],
                warnings=[f"Batch validation failed: {str(e)}"]
            ) for _ in price_batch]
    
    # Integration methods for safety systems
    
    def integrate_with_safety_manager(self, safety_manager) -> bool:
        """
        Integrate with TradingSafetyManager.
        
        Args:
            safety_manager: TradingSafetyManager instance
            
        Returns:
            True if integration successful
        """
        try:
            # Store reference to safety manager for alert escalation
            self.safety_manager = safety_manager
            logger.info("Successfully integrated with TradingSafetyManager")
            return True
        except Exception as e:
            logger.error("Failed to integrate with safety manager", error=str(e))
            return False
    
    async def check_emergency_conditions(self, validation_result: ValidationResult) -> bool:
        """
        Check if validation results warrant emergency stop.
        
        Args:
            validation_result: Result to check
            
        Returns:
            True if emergency stop should be triggered
        """
        emergency_reasons = [
            ValidationReason.PRICE_MANIPULATION_DETECTED,
            ValidationReason.FLASH_CRASH_DETECTED,
            ValidationReason.PUMP_DUMP_DETECTED
        ]
        
        # Check for critical validation failures
        has_emergency_reason = any(reason in validation_result.reasons for reason in emergency_reasons)
        
        # Check for low confidence in manipulation detection
        manipulation_confidence = validation_result.confidence_score if validation_result.is_manipulation_detected else 0
        
        return has_emergency_reason and manipulation_confidence > 0.8
    
    async def send_safety_alerts(self, alert: DataQualityAlert) -> bool:
        """
        Send alerts to safety systems.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert sent successfully
        """
        try:
            # Integration with safety manager if available
            if hasattr(self, 'safety_manager') and alert.severity in ["high", "critical"]:
                await self.safety_manager.activate_emergency_stop(
                    f"Data quality alert: {alert.message}"
                )
            
            logger.warning("Safety alert generated", 
                          alert_type=alert.alert_type.value,
                          severity=alert.severity,
                          message=alert.message)
            return True
        except Exception as e:
            logger.error("Failed to send safety alert", error=str(e))
            return False
    
    # Private helper methods
    
    async def _check_data_staleness(self, price_data: PriceData) -> ValidationResult:
        """Check if price data is stale."""
        age_minutes = (datetime.utcnow() - price_data.timestamp).total_seconds() / 60
        
        if age_minutes > self.config.max_stale_minutes:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.STALE_DATA_DETECTED],
                warnings=[f"Data age {age_minutes:.1f} minutes exceeds {self.config.max_stale_minutes} minutes"]
            )
        
        return ValidationResult(is_valid=True, status=ValidationStatus.APPROVED)
    
    async def _check_volume_threshold(self, price_data: PriceData) -> ValidationResult:
        """Check volume threshold."""
        if price_data.volume_24h < self.config.min_volume_threshold:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.LOW_VOLUME_DETECTED],
                warnings=[f"Volume {price_data.volume_24h} below threshold {self.config.min_volume_threshold}"]
            )
        
        return ValidationResult(is_valid=True, status=ValidationStatus.APPROVED)
    
    async def _check_bid_ask_spread(self, price_data: PriceData) -> ValidationResult:
        """Check bid-ask spread."""
        if price_data.bid_price and price_data.ask_price:
            spread = (price_data.ask_price - price_data.bid_price) / price_data.price
            max_spread = self.config.max_bid_ask_spread_percent / 100
            
            if spread > max_spread:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.REJECTED,
                    reasons=[ValidationReason.SPREAD_TOO_WIDE],
                    warnings=[f"Spread {spread * 100:.2f}% exceeds {self.config.max_bid_ask_spread_percent}%"]
                )
        
        return ValidationResult(is_valid=True, status=ValidationStatus.APPROVED)
    
    async def _detect_price_outliers(self, price_data_list: List[PriceData], consensus_price: Decimal) -> List[str]:
        """Detect price outliers using statistical methods."""
        outliers = []
        
        for pd in price_data_list:
            deviation = abs(pd.price - consensus_price) / consensus_price
            max_deviation = self.config.max_price_deviation_percent / 100
            
            if deviation > max_deviation:
                outliers.append(pd.source)
                # Reduce reliability of outlier source
                self.source_reliability[pd.source] *= 0.9
        
        return outliers
    
    async def _detect_flash_crash(self, price_history: List[PriceData]) -> Tuple[bool, float]:
        """Detect flash crash patterns."""
        if len(price_history) < 3:
            return False, 0.0
        
        prices = [float(p.price) for p in price_history]
        
        # Look for sudden price drops followed by recovery
        for i in range(1, len(prices) - 1):
            drop = (prices[i-1] - prices[i]) / prices[i-1]
            recovery = (prices[i+1] - prices[i]) / prices[i]
            
            if (drop > float(self.config.flash_crash_threshold_percent) / 100 and
                recovery > 0.1):  # 10% recovery
                confidence = min(drop * 2, 1.0)  # Scale confidence
                return True, confidence
        
        return False, 0.0
    
    async def _detect_pump_and_dump(self, price_history: List[PriceData]) -> Tuple[bool, float]:
        """Detect pump and dump patterns."""
        if len(price_history) < 4:
            return False, 0.0
        
        prices = [float(p.price) for p in price_history]
        volumes = [float(p.volume_24h) for p in price_history if p.volume_24h]
        
        # Look for rapid price increase followed by rapid decrease
        for i in range(2, len(prices) - 1):
            pump = (prices[i] - prices[i-2]) / prices[i-2]
            dump = (prices[i] - prices[i+1]) / prices[i]
            
            if (pump > float(self.config.pump_threshold_percent) / 100 and
                dump > 0.2):  # 20% dump
                
                # Check for volume spike during pump
                volume_spike = False
                if len(volumes) > i and i > 0:
                    if volumes[i] > volumes[i-1] * 2:  # 2x volume increase
                        volume_spike = True
                
                confidence = min(pump * dump, 1.0)
                if volume_spike:
                    confidence *= 1.2  # Increase confidence
                
                return True, min(confidence, 1.0)
        
        return False, 0.0
    
    async def _detect_wash_trading(self, price_history: List[PriceData]) -> Tuple[bool, float]:
        """Detect wash trading patterns."""
        if len(price_history) < 5:
            return False, 0.0
        
        volumes = [float(p.volume_24h) for p in price_history if p.volume_24h]
        prices = [float(p.price) for p in price_history]
        
        if len(volumes) < 3:
            return False, 0.0
        
        # Look for high volume with minimal price movement
        avg_volume = statistics.mean(volumes)
        price_volatility = statistics.stdev(prices) / statistics.mean(prices)
        
        # High volume periods with low volatility may indicate wash trading
        high_volume_periods = sum(1 for v in volumes if v > avg_volume * 2)
        
        if high_volume_periods >= 2 and price_volatility < 0.02:  # 2% volatility
            confidence = min((high_volume_periods / len(volumes)) * 2, 1.0)
            return True, confidence
        
        return False, 0.0
    
    async def _identify_support_levels(self, prices: List[float]) -> List[Decimal]:
        """Identify price support levels."""
        if len(prices) < 5:
            return []
        
        support_levels = []
        
        # Simple approach: find local minima
        for i in range(2, len(prices) - 2):
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                support_levels.append(Decimal(str(prices[i])))
        
        return sorted(set(support_levels))  # Remove duplicates and sort
    
    async def _identify_resistance_levels(self, prices: List[float]) -> List[Decimal]:
        """Identify price resistance levels."""
        if len(prices) < 5:
            return []
        
        resistance_levels = []
        
        # Simple approach: find local maxima
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                resistance_levels.append(Decimal(str(prices[i])))
        
        return sorted(set(resistance_levels), reverse=True)  # Remove duplicates and sort desc
    
    async def _create_alert_from_reason(self, reason: ValidationReason, price_data: PriceData, result: ValidationResult) -> Optional[DataQualityAlert]:
        """Create alert from validation reason."""
        alert_mapping = {
            ValidationReason.STALE_DATA_DETECTED: AlertType.STALE_DATA,
            ValidationReason.LOW_VOLUME_DETECTED: AlertType.LOW_VOLUME,
            ValidationReason.PRICE_OUTLIER_DETECTED: AlertType.PRICE_ANOMALY,
            ValidationReason.PRICE_MANIPULATION_DETECTED: AlertType.MANIPULATION_DETECTED,
            ValidationReason.SOURCE_INCONSISTENCY: AlertType.SOURCE_FAILURE
        }
        
        if reason not in alert_mapping:
            return None
        
        alert_type = alert_mapping[reason]
        
        # Determine severity
        severity = "medium"
        if reason in [ValidationReason.PRICE_MANIPULATION_DETECTED, ValidationReason.FLASH_CRASH_DETECTED]:
            severity = "critical"
        elif reason in [ValidationReason.PRICE_OUTLIER_DETECTED, ValidationReason.SOURCE_INCONSISTENCY]:
            severity = "high"
        
        return DataQualityAlert(
            alert_type=alert_type,
            symbol=price_data.symbol,
            message=f"{reason.value} for {price_data.symbol} from {price_data.source}",
            severity=severity,
            metadata={
                "price": float(price_data.price),
                "source": price_data.source,
                "timestamp": price_data.timestamp.isoformat(),
                "validation_details": result.warnings
            }
        )
    
    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results."""
        if not results:
            return ValidationResult(is_valid=False, status=ValidationStatus.REJECTED)
        
        # All must be valid for combined result to be valid
        is_valid = all(r.is_valid for r in results)
        
        # Collect all reasons and warnings
        all_reasons = []
        all_warnings = []
        
        for result in results:
            all_reasons.extend(result.reasons)
            all_warnings.extend(result.warnings)
        
        status = ValidationStatus.APPROVED if is_valid else ValidationStatus.REJECTED
        
        return ValidationResult(
            is_valid=is_valid,
            status=status,
            reasons=all_reasons,
            warnings=all_warnings
        )
    
    def _update_quality_metrics(self, source: str, result: ValidationResult) -> None:
        """Update quality metrics based on validation result."""
        if result.is_valid:
            self.quality_metrics["source_reliability"][source] = min(
                self.quality_metrics["source_reliability"][source] + 0.01, 1.0
            )
        else:
            self.quality_metrics["source_reliability"][source] = max(
                self.quality_metrics["source_reliability"][source] - 0.05, 0.0
            )
            self.quality_metrics["rejection_count"] += 1
        
        # Update freshness score based on data age
        if result.price_data:
            age_minutes = (datetime.utcnow() - result.price_data.timestamp).total_seconds() / 60
            freshness_score = max(0, 1 - (age_minutes / self.config.max_stale_minutes))
            self.quality_metrics["data_freshness"][source] = freshness_score