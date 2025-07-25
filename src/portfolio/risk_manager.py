"""
Risk Manager for Portfolio Management

This module provides comprehensive risk management functionality
for multi-chain, multi-DEX trading portfolios. It handles position
sizing, risk limits enforcement, real-time monitoring, and advanced
risk analytics.

Key Features:
- Position sizing management with multiple methodologies (Kelly, fixed fractional, volatility-adjusted)
- Risk limits enforcement (max drawdown, daily loss, leverage, concentration)
- Real-time risk monitoring with alerts and notifications
- Dynamic position sizing based on volatility and portfolio risk
- Portfolio risk metrics calculation (VaR, volatility, Sharpe ratio, drawdown)
- Risk-adjusted position recommendations
- Emergency stop-loss and liquidation protection
- Multi-chain and multi-DEX risk aggregation
- Correlation analysis and concentration risk management
- Risk budgeting and allocation optimization
"""

import asyncio
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID
import structlog

from .base import (
    Portfolio,
    Position,
    Transaction,
    PortfolioConfig,
    PositionType,
    PositionStatus,
    TransactionType,
    RiskMetrics,
    DrawdownMetrics,
    PerformanceMetrics,
    PortfolioError,
    RiskLimitExceededError,
)
from src.utils.base import Chain


logger = structlog.get_logger()


class RiskAlertType(Enum):
    """Type of risk alert."""
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
    """Severity of risk alert."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_portfolio_risk_pct: Decimal = Decimal("0.02")           # 2% max portfolio risk
    max_single_position_pct: Decimal = Decimal("0.1")           # 10% max single position
    max_correlation_threshold: Decimal = Decimal("0.8")         # 80% correlation limit
    var_confidence_level: Decimal = Decimal("0.95")             # 95% VaR confidence
    stress_test_scenarios: int = 5                              # Number of stress scenarios
    position_sizing_method: str = "kelly"                       # kelly, fixed_fractional, volatility_adjusted
    enable_dynamic_sizing: bool = True                          # Enable dynamic position sizing
    enable_correlation_limits: bool = True                      # Enable correlation limits
    enable_drawdown_protection: bool = True                     # Enable drawdown protection
    max_leverage: Decimal = Decimal("3.0")                      # Maximum leverage allowed
    emergency_stop_loss_pct: Decimal = Decimal("0.1")           # 10% emergency stop loss
    daily_loss_reset_hour: int = 0                              # Hour to reset daily loss (UTC)
    alert_thresholds: Dict[str, Decimal] = field(default_factory=lambda: {
        "warning": Decimal("0.75"),   # 75% of limit
        "critical": Decimal("0.9")    # 90% of limit
    })
    risk_free_rate: Decimal = Decimal("0.05")                   # 5% risk-free rate
    lookback_period_days: int = 30                              # Lookback period for calculations
    min_observations: int = 20                                  # Minimum observations required
    correlation_update_frequency_hours: int = 4                 # Correlation update frequency
    enable_real_time_monitoring: bool = True                    # Enable real-time monitoring
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (0 < self.max_portfolio_risk_pct <= 1):
            raise ValueError("max_portfolio_risk_pct must be between 0 and 1")
        if not (0 < self.max_single_position_pct <= 1):
            raise ValueError("max_single_position_pct must be between 0 and 1")
        if not (0 <= self.max_correlation_threshold <= 1):
            raise ValueError("max_correlation_threshold must be between 0 and 1")
        if not (0.5 <= self.var_confidence_level <= 0.99):
            raise ValueError("var_confidence_level must be between 0.5 and 0.99")
        if self.position_sizing_method not in ["kelly", "fixed_fractional", "volatility_adjusted"]:
            raise ValueError("position_sizing_method must be 'kelly', 'fixed_fractional', or 'volatility_adjusted'")


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    alert_type: RiskAlertType
    severity: RiskAlertSeverity
    message: str
    timestamp: datetime
    position_id: Optional[UUID] = None
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    recommended_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation result."""
    success: bool
    symbol: str
    recommended_size: Decimal = Decimal("0")
    recommended_size_usd: Decimal = Decimal("0")
    base_size: Decimal = Decimal("0")
    adjusted_size: Decimal = Decimal("0")
    risk_amount: Decimal = Decimal("0")
    position_risk_pct: Decimal = Decimal("0")
    sizing_method: str = ""
    
    # Sizing factors
    kelly_fraction: Optional[Decimal] = None
    volatility_adjustment: Optional[Decimal] = None
    correlation_adjustment: Optional[Decimal] = None
    risk_adjustment: Optional[Decimal] = None
    current_volatility: Optional[Decimal] = None
    
    # Metadata
    size_capped: bool = False
    warnings: List[str] = field(default_factory=list)
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class RiskLimitCheck:
    """Individual risk limit check result."""
    limit_type: str
    current_value: Decimal
    limit_value: Decimal
    limit_exceeded: bool
    severity: RiskAlertSeverity
    position_id: Optional[UUID] = None
    chain: Optional[Chain] = None
    dex_name: Optional[str] = None
    message: str = ""


@dataclass
class RiskAssessmentResult:
    """Risk assessment result."""
    success: bool
    total_portfolio_risk: Decimal = Decimal("0")
    position_count: int = 0
    risk_score: Decimal = Decimal("0")  # 0-100 risk score
    risk_level: str = "low"  # low, medium, high, critical
    assessment_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class RiskLimitResult:
    """Risk limit checking result."""
    success: bool
    limits_checked: int = 0
    limit_checks: List[RiskLimitCheck] = field(default_factory=list)
    limits_exceeded: int = 0
    emergency_stop_triggered: bool = False
    positions_reduced: bool = False
    actions_taken: List[str] = field(default_factory=list)
    check_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class PositionRisk:
    """Individual position risk information."""
    position_id: UUID
    symbol: str
    current_risk: Decimal
    risk_percentage: Decimal
    unrealized_pnl: Decimal
    risk_score: Decimal  # 0-100
    liquidation_risk_pct: Optional[Decimal] = None
    correlation_risk: Optional[Decimal] = None


@dataclass
class RiskMonitoringResult:
    """Risk monitoring result."""
    success: bool
    total_portfolio_risk: Decimal = Decimal("0")
    position_count: int = 0
    position_risks: List[PositionRisk] = field(default_factory=list)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    alerts_generated: int = 0
    monitoring_timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class PortfolioRiskResult:
    """Portfolio risk calculation result."""
    success: bool
    total_portfolio_risk: Decimal = Decimal("0")
    position_count: int = 0
    individual_risks: List[PositionRisk] = field(default_factory=list)
    risk_breakdown: Dict[str, Any] = field(default_factory=dict)
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class VaRCalculationResult:
    """Value at Risk calculation result."""
    success: bool
    var_amount: Decimal = Decimal("0")
    var_percentage: Decimal = Decimal("0")
    confidence_level: Decimal = Decimal("0.95")
    time_horizon_days: int = 1
    expected_shortfall: Optional[Decimal] = None
    calculation_method: str = "historical"  # historical, parametric, monte_carlo
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class StressTestScenario:
    """Stress test scenario definition."""
    scenario_name: str
    market_shock_pct: Decimal
    correlation_change: Optional[Decimal] = None
    volatility_multiplier: Optional[Decimal] = None
    duration_days: int = 1


@dataclass
class StressTestScenarioResult:
    """Stress test scenario result."""
    scenario_name: str
    portfolio_pnl: Decimal
    portfolio_pnl_pct: Decimal
    positions_affected: int
    worst_position_loss: Decimal
    liquidity_impact: Optional[Decimal] = None


@dataclass
class StressTestResult:
    """Stress test result."""
    success: bool
    scenarios: List[StressTestScenario] = field(default_factory=list)
    scenario_results: List[StressTestScenarioResult] = field(default_factory=list)
    worst_case_loss: Decimal = Decimal("0")
    best_case_gain: Decimal = Decimal("0")
    expected_loss: Decimal = Decimal("0")
    test_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class CorrelationPair:
    """Correlation between two symbols."""
    symbol1: str
    symbol2: str
    correlation: Decimal
    observations: int
    last_updated: datetime


@dataclass
class CorrelationAnalysisResult:
    """Correlation analysis result."""
    success: bool
    correlation_pairs: List[CorrelationPair] = field(default_factory=list)
    max_correlation: Decimal = Decimal("0")
    portfolio_concentration_risk: Decimal = Decimal("0")
    high_correlation_groups: List[List[str]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class PositionWeight:
    """Position weight in portfolio."""
    position_id: UUID
    symbol: str
    weight: Decimal
    market_value: Decimal


@dataclass
class ConcentrationResult:
    """Position concentration analysis result."""
    success: bool
    concentration_index: Decimal = Decimal("0")  # 0 (diversified) to 1 (concentrated)
    position_weights: List[PositionWeight] = field(default_factory=list)
    largest_position_pct: Decimal = Decimal("0")
    top_5_concentration: Decimal = Decimal("0")
    diversification_ratio: Decimal = Decimal("0")
    error_message: Optional[str] = None


@dataclass
class RiskAllocation:
    """Risk budget allocation for an asset."""
    symbol: str
    risk_allocation: Decimal
    expected_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    position_size: Decimal


@dataclass
class RiskBudgetResult:
    """Risk budgeting result."""
    success: bool
    allocations: List[RiskAllocation] = field(default_factory=list)
    total_risk_budget: Decimal = Decimal("0")
    expected_portfolio_return: Decimal = Decimal("0")
    portfolio_volatility: Decimal = Decimal("0")
    portfolio_sharpe_ratio: Decimal = Decimal("0")
    optimization_method: str = "equal_risk_contribution"
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class OptimalAllocation:
    """Optimal portfolio allocation."""
    symbol: str
    current_weight: Decimal
    optimal_weight: Decimal
    weight_change: Decimal
    trade_amount: Decimal


@dataclass
class AllocationOptimizationResult:
    """Portfolio allocation optimization result."""
    success: bool
    current_allocations: List[OptimalAllocation] = field(default_factory=list)
    optimal_allocations: List[OptimalAllocation] = field(default_factory=list)
    rebalancing_needed: bool = False
    expected_improvement: Optional[Decimal] = None
    rebalancing_trades: List[Dict[str, Any]] = field(default_factory=list)
    optimization_objective: str = "max_sharpe"  # max_sharpe, min_variance, equal_risk
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class EmergencyConditionResult:
    """Emergency condition check result."""
    emergency_triggered: bool
    trigger_reason: Optional[str] = None
    trigger_value: Optional[Decimal] = None
    threshold_value: Optional[Decimal] = None
    recommended_actions: List[str] = field(default_factory=list)
    check_time: datetime = field(default_factory=datetime.now)


@dataclass
class LiquidationRiskPosition:
    """Position at risk of liquidation."""
    position_id: UUID
    symbol: str
    current_price: Decimal
    liquidation_price: Decimal
    distance_to_liquidation: Decimal  # Percentage distance
    margin_ratio: Optional[Decimal] = None
    recommended_action: str = "monitor"  # monitor, reduce_size, close_position, add_margin


@dataclass
class LiquidationRiskResult:
    """Liquidation risk assessment result."""
    success: bool
    at_risk_positions: List[LiquidationRiskPosition] = field(default_factory=list)
    total_at_risk_value: Decimal = Decimal("0")
    highest_risk_position: Optional[UUID] = None
    check_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class StopLossTrigger:
    """Stop loss trigger information."""
    position_id: UUID
    symbol: str
    current_price: Decimal
    stop_loss_price: Decimal
    trigger_time: datetime


@dataclass
class StopLossResult:
    """Stop loss check result."""
    success: bool
    triggered_positions: List[StopLossTrigger] = field(default_factory=list)
    positions_closed: int = 0
    total_realized_loss: Decimal = Decimal("0")
    check_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class MarginCall:
    """Margin call information."""
    position_id: UUID
    symbol: str
    current_margin: Decimal
    required_margin: Decimal
    margin_call_amount: Decimal
    liquidation_risk: Decimal
    deadline: datetime


@dataclass
class MarginCallResult:
    """Margin call check result."""
    success: bool
    margin_calls: List[MarginCall] = field(default_factory=list)
    total_margin_required: Decimal = Decimal("0")
    critical_positions: int = 0
    check_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class ChainRisk:
    """Risk aggregated by blockchain chain."""
    chain: Chain
    position_count: int
    total_exposure: Decimal
    risk_percentage: Decimal
    unrealized_pnl: Decimal
    dex_breakdown: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class ChainRiskResult:
    """Chain risk aggregation result."""
    success: bool
    chain_risks: List[ChainRisk] = field(default_factory=list)
    total_exposure: Decimal = Decimal("0")
    highest_risk_chain: Optional[Chain] = None
    chain_concentration_risk: Decimal = Decimal("0")
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class DexRisk:
    """Risk aggregated by DEX."""
    dex_name: str
    position_count: int
    total_exposure: Decimal
    risk_percentage: Decimal
    unrealized_pnl: Decimal
    chain_breakdown: Dict[Chain, Decimal] = field(default_factory=dict)


@dataclass
class DexRiskResult:
    """DEX risk aggregation result."""
    success: bool
    dex_risks: List[DexRisk] = field(default_factory=list)
    total_exposure: Decimal = Decimal("0")
    highest_risk_dex: Optional[str] = None
    dex_concentration_risk: Decimal = Decimal("0")
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class CrossChainCorrelation:
    """Cross-chain correlation information."""
    symbol1: str
    chain1: Chain
    symbol2: str
    chain2: Chain
    correlation: Decimal
    observations: int


@dataclass
class CrossChainCorrelationResult:
    """Cross-chain correlation analysis result."""
    success: bool
    cross_chain_correlations: List[CrossChainCorrelation] = field(default_factory=list)
    max_cross_chain_correlation: Decimal = Decimal("0")
    chain_correlation_matrix: Dict[Tuple[Chain, Chain], Decimal] = field(default_factory=dict)
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class AssetConcentration:
    """Asset concentration information."""
    asset_symbol: str
    weight: Decimal
    exposure_chains: List[Chain]
    total_positions: int


@dataclass
class ConcentrationRiskResult:
    """Concentration risk calculation result."""
    success: bool
    concentration_index: Decimal = Decimal("0")
    asset_concentrations: List[AssetConcentration] = field(default_factory=list)
    largest_asset_weight: Decimal = Decimal("0")
    effective_number_of_positions: Decimal = Decimal("0")
    calculation_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


class RiskManagerError(PortfolioError):
    """Base risk manager error."""
    pass


class InsufficientDataError(RiskManagerError):
    """Error when insufficient data for calculation."""
    pass


class RiskLimitViolationError(RiskManagerError):
    """Error when risk limits are violated."""
    pass


class EmergencyStopError(RiskManagerError):
    """Error when emergency stop is triggered."""
    pass


class RiskManager:
    """
    Comprehensive risk manager for portfolio management.
    
    This class provides advanced risk management functionality including
    position sizing, risk limits enforcement, real-time monitoring,
    correlation analysis, and emergency protection mechanisms.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        config: RiskConfig,
        position_tracker: Optional[Any] = None,
        pnl_calculator: Optional[Any] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            portfolio: Portfolio to manage
            config: Risk management configuration
            position_tracker: Optional position tracker integration
            pnl_calculator: Optional P&L calculator integration
        """
        self.portfolio = portfolio
        self.config = config
        self.position_tracker = position_tracker
        self.pnl_calculator = pnl_calculator
        
        # Risk monitoring state
        self.alerts: List[RiskAlert] = []
        self.last_correlation_update: Optional[datetime] = None
        self._price_history: Dict[str, List[Dict[str, Any]]] = {}
        self._correlation_matrix: Dict[Tuple[str, str], Decimal] = {}
        self._daily_pnl_history: List[Dict[str, Any]] = []
        self._emergency_stop_active: bool = False
        
        self.logger = logger.bind(
            component="risk_manager",
            portfolio_id=str(portfolio.portfolio_id)
        )
    
    # Position Sizing Methods
    
    async def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        confidence_level: Optional[Decimal] = None,
        risk_fraction: Optional[Decimal] = None,
        target_volatility: Optional[Decimal] = None,
        side: str = "LONG"
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size using configured methodology.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for position
            stop_loss_price: Stop loss price (optional)
            confidence_level: Confidence level for Kelly criterion (optional)
            risk_fraction: Risk fraction for fixed fractional sizing (optional)
            target_volatility: Target volatility for volatility-adjusted sizing (optional)
            side: Position side (LONG/SHORT)
            
        Returns:
            PositionSizeRecommendation with sizing details
        """
        try:
            # Validate parameters
            if entry_price <= 0:
                return PositionSizeRecommendation(
                    success=False,
                    symbol=symbol,
                    error_message="Entry price must be positive",
                    calculation_time=datetime.now()
                )
            
            # Validate stop loss
            if stop_loss_price is not None:
                if side.upper() == "LONG" and stop_loss_price >= entry_price:
                    return PositionSizeRecommendation(
                        success=False,
                        symbol=symbol,
                        error_message="Invalid stop loss: stop loss must be below entry price for long positions",
                        calculation_time=datetime.now()
                    )
                elif side.upper() == "SHORT" and stop_loss_price <= entry_price:
                    return PositionSizeRecommendation(
                        success=False,
                        symbol=symbol,
                        error_message="Invalid stop loss: stop loss must be above entry price for short positions",
                        calculation_time=datetime.now()
                    )
            
            # Check available capital
            available_capital = self.portfolio.cash_balance
            min_capital_required = Decimal("1000")  # Minimum $1000 required
            if available_capital <= 0:
                return PositionSizeRecommendation(
                    success=False,
                    symbol=symbol,
                    error_message="Insufficient capital available",
                    calculation_time=datetime.now()
                )
            elif available_capital < min_capital_required:
                return PositionSizeRecommendation(
                    success=False,
                    symbol=symbol,
                    error_message=f"Insufficient capital: minimum ${min_capital_required} required, have ${available_capital}",
                    calculation_time=datetime.now()
                )
            
            # Calculate position size based on method
            if self.config.position_sizing_method == "kelly":
                result = await self._calculate_kelly_position_size(
                    symbol, entry_price, stop_loss_price, confidence_level, side
                )
            elif self.config.position_sizing_method == "fixed_fractional":
                result = await self._calculate_fixed_fractional_size(
                    symbol, entry_price, stop_loss_price, risk_fraction, side
                )
            elif self.config.position_sizing_method == "volatility_adjusted":
                result = await self._calculate_volatility_adjusted_size(
                    symbol, entry_price, stop_loss_price, target_volatility, side
                )
            else:
                return PositionSizeRecommendation(
                    success=False,
                    symbol=symbol,
                    error_message=f"Unknown sizing method: {self.config.position_sizing_method}",
                    calculation_time=datetime.now()
                )
            
            # Apply position size limits
            result = await self._apply_position_size_limits(result)
            
            # Apply dynamic adjustments if enabled
            if self.config.enable_dynamic_sizing:
                result = await self._apply_dynamic_adjustments(result)
            
            result.calculation_time = datetime.now()
            
            self.logger.info(
                "Position size calculated",
                symbol=symbol,
                method=self.config.position_sizing_method,
                recommended_size=str(result.recommended_size),
                recommended_size_usd=str(result.recommended_size_usd),
                risk_amount=str(result.risk_amount)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate position size",
                symbol=symbol,
                error=str(e)
            )
            return PositionSizeRecommendation(
                success=False,
                symbol=symbol,
                error_message=f"Position size calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_dynamic_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        base_size: Decimal,
        volatility_target: Optional[Decimal] = None
    ) -> PositionSizeRecommendation:
        """
        Calculate dynamic position size based on current market conditions.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for position
            base_size: Base position size
            volatility_target: Target volatility level
            
        Returns:
            PositionSizeRecommendation with dynamic adjustments
        """
        try:
            result = PositionSizeRecommendation(
                success=True,
                symbol=symbol,
                base_size=base_size,
                adjusted_size=base_size,
                recommended_size=base_size,
                sizing_method="dynamic"
            )
            
            # Get current volatility
            current_vol = await self._get_asset_volatility(symbol)
            if current_vol is not None:
                result.current_volatility = current_vol
                
                # Apply volatility adjustment
                if volatility_target:
                    vol_adjustment = volatility_target / current_vol
                    result.volatility_adjustment = vol_adjustment
                    result.adjusted_size = base_size * vol_adjustment
            
            # Apply portfolio risk adjustment
            portfolio_risk = await self._get_current_portfolio_risk()
            if portfolio_risk > self.config.max_portfolio_risk_pct * Decimal("0.8"):  # 80% of limit
                risk_reduction = Decimal("0.8")  # Reduce by 20%
                result.risk_adjustment = risk_reduction
                result.adjusted_size *= risk_reduction
            
            # Apply correlation adjustment
            correlation_adjustment = await self._get_correlation_adjustment(symbol)
            if correlation_adjustment is not None:
                result.correlation_adjustment = correlation_adjustment
                result.adjusted_size *= correlation_adjustment
            
            # Final recommended size
            result.recommended_size = result.adjusted_size
            result.recommended_size_usd = result.recommended_size * entry_price
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate dynamic position size",
                symbol=symbol,
                error=str(e)
            )
            return PositionSizeRecommendation(
                success=False,
                symbol=symbol,
                error_message=f"Dynamic sizing calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Risk Limits Enforcement
    
    async def check_risk_limits(self) -> RiskLimitResult:
        """
        Check all configured risk limits.
        
        Returns:
            RiskLimitResult with limit check details
        """
        try:
            limit_checks = []
            limits_exceeded = 0
            
            # Check position size limits
            position_checks = await self._check_position_size_limits()
            limit_checks.extend(position_checks)
            
            # Check portfolio risk limits
            portfolio_checks = await self._check_portfolio_risk_limits()
            limit_checks.extend(portfolio_checks)
            
            # Check daily loss limits
            daily_loss_checks = await self._check_daily_loss_limits()
            limit_checks.extend(daily_loss_checks)
            
            # Check drawdown limits
            drawdown_checks = await self._check_drawdown_limits()
            limit_checks.extend(drawdown_checks)
            
            # Check correlation limits if enabled
            if self.config.enable_correlation_limits:
                correlation_checks = await self._check_correlation_limits()
                limit_checks.extend(correlation_checks)
            
            # Check leverage limits
            leverage_checks = await self._check_leverage_limits()
            limit_checks.extend(leverage_checks)
            
            # Count exceeded limits
            limits_exceeded = sum(1 for check in limit_checks if check.limit_exceeded)
            
            result = RiskLimitResult(
                success=True,
                limits_checked=len(limit_checks),
                limit_checks=limit_checks,
                limits_exceeded=limits_exceeded,
                check_time=datetime.now()
            )
            
            self.logger.info(
                "Risk limits checked",
                limits_checked=len(limit_checks),
                limits_exceeded=limits_exceeded
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to check risk limits",
                error=str(e)
            )
            return RiskLimitResult(
                success=False,
                error_message=f"Risk limit check failed: {e}",
                check_time=datetime.now()
            )
    
    async def enforce_risk_limits(self) -> RiskLimitResult:
        """
        Enforce risk limits by taking corrective actions.
        
        Returns:
            RiskLimitResult with enforcement actions taken
        """
        try:
            # First check all limits
            check_result = await self.check_risk_limits()
            if not check_result.success:
                return check_result
            
            actions_taken = []
            positions_reduced = False
            emergency_stop_triggered = False
            
            # Process each exceeded limit
            for check in check_result.limit_checks:
                if not check.limit_exceeded:
                    continue
                
                # Handle critical violations
                if check.severity == RiskAlertSeverity.CRITICAL:
                    if check.limit_type == "drawdown" and check.current_value >= self.config.emergency_stop_loss_pct:
                        await self._trigger_emergency_stop()
                        emergency_stop_triggered = True
                        actions_taken.append("emergency_stop_triggered")
                    elif check.limit_type == "position_size" and check.position_id:
                        await self._reduce_position_size(check.position_id, Decimal("0.5"))
                        positions_reduced = True
                        actions_taken.append(f"reduced_position_{check.position_id}")
                    elif check.limit_type == "leverage" and check.position_id:
                        await self._reduce_leverage(check.position_id)
                        actions_taken.append(f"reduced_leverage_{check.position_id}")
            
            result = RiskLimitResult(
                success=True,
                limits_checked=check_result.limits_checked,
                limit_checks=check_result.limit_checks,
                limits_exceeded=check_result.limits_exceeded,
                emergency_stop_triggered=emergency_stop_triggered,
                positions_reduced=positions_reduced,
                actions_taken=actions_taken,
                check_time=datetime.now()
            )
            
            self.logger.info(
                "Risk limits enforced",
                actions_taken=len(actions_taken),
                emergency_stop=emergency_stop_triggered,
                positions_reduced=positions_reduced
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to enforce risk limits",
                error=str(e)
            )
            return RiskLimitResult(
                success=False,
                error_message=f"Risk limit enforcement failed: {e}",
                check_time=datetime.now()
            )
    
    # Risk Monitoring
    
    async def monitor_portfolio_risk(self) -> RiskMonitoringResult:
        """
        Monitor real-time portfolio risk.
        
        Returns:
            RiskMonitoringResult with current risk status
        """
        try:
            # Calculate portfolio-level risk
            total_portfolio_risk = await self._calculate_total_portfolio_risk()
            
            # Analyze individual position risks
            position_risks = []
            for position in self.portfolio.open_positions.values():
                position_risk = await self._calculate_position_risk(position)
                position_risks.append(position_risk)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics()
            
            # Generate alerts if needed
            alerts_generated = 0
            if self.config.enable_real_time_monitoring:
                alerts = await self.generate_risk_alerts()
                alerts_generated = len(alerts)
            
            result = RiskMonitoringResult(
                success=True,
                total_portfolio_risk=total_portfolio_risk,
                position_count=len(position_risks),
                position_risks=position_risks,
                risk_metrics=risk_metrics,
                alerts_generated=alerts_generated,
                monitoring_timestamp=datetime.now()
            )
            
            self.logger.debug(
                "Portfolio risk monitored",
                total_risk=str(total_portfolio_risk),
                position_count=len(position_risks),
                alerts_generated=alerts_generated
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to monitor portfolio risk",
                error=str(e)
            )
            return RiskMonitoringResult(
                success=False,
                error_message=f"Risk monitoring failed: {e}",
                monitoring_timestamp=datetime.now()
            )
    
    async def generate_risk_alerts(self) -> List[RiskAlert]:
        """
        Generate risk alerts based on current conditions.
        
        Returns:
            List of RiskAlert objects
        """
        try:
            alerts = []
            
            # Check position size alerts
            for position in self.portfolio.open_positions.values():
                position_value = position.market_value
                position_pct = position_value / self.portfolio.total_value
                
                if position_pct > self.config.max_single_position_pct * self.config.alert_thresholds["critical"]:
                    alerts.append(RiskAlert(
                        alert_id=f"pos_size_{position.position_id}_{datetime.now().timestamp()}",
                        alert_type=RiskAlertType.POSITION_SIZE,
                        severity=RiskAlertSeverity.CRITICAL,
                        message=f"Position {position.symbol} exceeds size limit: {position_pct:.2%}",
                        timestamp=datetime.now(),
                        position_id=position.position_id,
                        current_value=position_pct,
                        limit_value=self.config.max_single_position_pct,
                        recommended_action="reduce_position_size"
                    ))
                elif position_pct > self.config.max_single_position_pct * self.config.alert_thresholds["warning"]:
                    alerts.append(RiskAlert(
                        alert_id=f"pos_size_{position.position_id}_{datetime.now().timestamp()}",
                        alert_type=RiskAlertType.POSITION_SIZE,
                        severity=RiskAlertSeverity.WARNING,
                        message=f"Position {position.symbol} approaching size limit: {position_pct:.2%}",
                        timestamp=datetime.now(),
                        position_id=position.position_id,
                        current_value=position_pct,
                        limit_value=self.config.max_single_position_pct,
                        recommended_action="monitor_position"
                    ))
            
            # Check portfolio risk alerts
            portfolio_risk = await self._calculate_total_portfolio_risk()
            if portfolio_risk > self.config.max_portfolio_risk_pct * self.config.alert_thresholds["critical"]:
                alerts.append(RiskAlert(
                    alert_id=f"portfolio_risk_{datetime.now().timestamp()}",
                    alert_type=RiskAlertType.PORTFOLIO_RISK,
                    severity=RiskAlertSeverity.CRITICAL,
                    message=f"Portfolio risk exceeds limit: {portfolio_risk:.2%}",
                    timestamp=datetime.now(),
                    current_value=portfolio_risk,
                    limit_value=self.config.max_portfolio_risk_pct,
                    recommended_action="reduce_overall_exposure"
                ))
            
            # Check drawdown alerts
            if self.portfolio.drawdown_metrics:
                drawdown = self.portfolio.drawdown_metrics.current_drawdown
                if drawdown > self.config.emergency_stop_loss_pct * self.config.alert_thresholds["critical"]:
                    alerts.append(RiskAlert(
                        alert_id=f"drawdown_{datetime.now().timestamp()}",
                        alert_type=RiskAlertType.DRAWDOWN,
                        severity=RiskAlertSeverity.EMERGENCY,
                        message=f"Drawdown approaching emergency stop: {drawdown:.2%}",
                        timestamp=datetime.now(),
                        current_value=drawdown,
                        limit_value=self.config.emergency_stop_loss_pct,
                        recommended_action="prepare_for_emergency_stop"
                    ))
            
            # Store alerts
            self.alerts.extend(alerts)
            
            self.logger.info(
                "Risk alerts generated",
                alert_count=len(alerts),
                critical_alerts=len([a for a in alerts if a.severity == RiskAlertSeverity.CRITICAL]),
                emergency_alerts=len([a for a in alerts if a.severity == RiskAlertSeverity.EMERGENCY])
            )
            
            return alerts
            
        except Exception as e:
            self.logger.error(
                "Failed to generate risk alerts",
                error=str(e)
            )
            return []
    
    async def update_risk_metrics(self) -> VaRCalculationResult:
        """
        Update portfolio risk metrics.
        
        Returns:
            VaRCalculationResult with updated metrics
        """
        try:
            # Calculate VaR using configured confidence level
            var_result = await self.calculate_portfolio_var(
                confidence_level=self.config.var_confidence_level,
                time_horizon_days=1
            )
            
            if not var_result.success:
                return var_result
            
            # Calculate portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility()
            
            # Calculate expected shortfall (CVaR)
            expected_shortfall = await self._calculate_expected_shortfall(self.config.var_confidence_level)
            
            # Update portfolio risk metrics
            risk_metrics = RiskMetrics(
                var_95=var_result.var_amount if self.config.var_confidence_level == Decimal("0.95") else await self._calculate_var_95(),
                var_99=var_result.var_amount if self.config.var_confidence_level == Decimal("0.99") else await self._calculate_var_99(),
                expected_shortfall=expected_shortfall,
                volatility=portfolio_volatility,
                max_position_risk=await self._calculate_max_position_risk(),
                concentration_risk=await self._calculate_concentration_risk(),
                leverage_ratio=await self._calculate_portfolio_leverage()
            )
            
            self.portfolio.risk_metrics = risk_metrics
            
            result = VaRCalculationResult(
                success=True,
                var_amount=var_result.var_amount,
                var_percentage=var_result.var_percentage,
                confidence_level=self.config.var_confidence_level,
                expected_shortfall=expected_shortfall,
                portfolio_volatility=portfolio_volatility,
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Risk metrics updated",
                var_95=str(risk_metrics.var_95),
                var_99=str(risk_metrics.var_99),
                volatility=str(portfolio_volatility)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to update risk metrics",
                error=str(e)
            )
            return VaRCalculationResult(
                success=False,
                error_message=f"Risk metrics update failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Value at Risk Calculation
    
    async def calculate_portfolio_var(
        self,
        confidence_level: Decimal,
        time_horizon_days: int = 1
    ) -> VaRCalculationResult:
        """
        Calculate portfolio Value at Risk.
        
        Args:
            confidence_level: Confidence level (0.95 for 95% VaR)
            time_horizon_days: Time horizon in days
            
        Returns:
            VaRCalculationResult with VaR details
        """
        try:
            # Get portfolio returns
            returns = await self._get_portfolio_returns()
            
            if len(returns) < self.config.min_observations:
                return VaRCalculationResult(
                    success=False,
                    error_message="Insufficient data for VaR calculation",
                    calculation_time=datetime.now()
                )
            
            # Calculate VaR using historical method
            var_amount, var_percentage = await self._calculate_historical_var(returns, confidence_level)
            
            # Adjust for time horizon (assuming i.i.d. returns)
            if time_horizon_days != 1:
                scaling_factor = Decimal(str(math.sqrt(time_horizon_days)))
                var_amount *= scaling_factor
                var_percentage *= scaling_factor
            
            # Calculate expected shortfall
            expected_shortfall = await self._calculate_expected_shortfall_from_returns(returns, confidence_level)
            
            result = VaRCalculationResult(
                success=True,
                var_amount=var_amount,
                var_percentage=var_percentage,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                expected_shortfall=expected_shortfall,
                calculation_method="historical",
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "Portfolio VaR calculated",
                var_amount=str(var_amount),
                var_percentage=str(var_percentage),
                confidence_level=str(confidence_level)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate portfolio VaR",
                error=str(e)
            )
            return VaRCalculationResult(
                success=False,
                error_message=f"VaR calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Stress Testing
    
    async def perform_stress_test(self) -> StressTestResult:
        """
        Perform portfolio stress testing.
        
        Returns:
            StressTestResult with stress test scenarios and results
        """
        try:
            # Define stress test scenarios
            scenarios = [
                StressTestScenario("Market Crash", Decimal("-0.20")),
                StressTestScenario("Flash Crash", Decimal("-0.10")),
                StressTestScenario("Crypto Winter", Decimal("-0.50")),
                StressTestScenario("Correlation Breakdown", Decimal("-0.15"), correlation_change=Decimal("-0.5")),
                StressTestScenario("Volatility Spike", Decimal("-0.10"), volatility_multiplier=Decimal("2.0"))
            ]
            
            scenario_results = []
            
            # Run each scenario
            for scenario in scenarios:
                scenario_result = await self._run_stress_scenario(scenario)
                scenario_results.append(scenario_result)
            
            # Calculate summary statistics
            worst_case_loss = min(r.portfolio_pnl for r in scenario_results)
            best_case_gain = max(r.portfolio_pnl for r in scenario_results)
            expected_loss = sum(r.portfolio_pnl for r in scenario_results) / len(scenario_results)
            
            result = StressTestResult(
                success=True,
                scenarios=scenarios,
                scenario_results=scenario_results,
                worst_case_loss=worst_case_loss,
                best_case_gain=best_case_gain,
                expected_loss=expected_loss,
                test_time=datetime.now()
            )
            
            self.logger.info(
                "Stress test completed",
                scenarios_tested=len(scenarios),
                worst_case_loss=str(worst_case_loss),
                expected_loss=str(expected_loss)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to perform stress test",
                error=str(e)
            )
            return StressTestResult(
                success=False,
                error_message=f"Stress test failed: {e}",
                test_time=datetime.now()
            )
    
    # Correlation Analysis
    
    async def calculate_correlation_matrix(self) -> CorrelationAnalysisResult:
        """
        Calculate correlation matrix for all positions.
        
        Returns:
            CorrelationAnalysisResult with correlation data
        """
        try:
            correlation_pairs = []
            symbols = list(set(pos.symbol for pos in self.portfolio.positions.values()))
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = await self._calculate_pairwise_correlation(symbol1, symbol2)
                    if correlation is not None:
                        pair = CorrelationPair(
                            symbol1=symbol1,
                            symbol2=symbol2,
                            correlation=correlation,
                            observations=self.config.lookback_period_days,
                            last_updated=datetime.now()
                        )
                        correlation_pairs.append(pair)
                        
                        # Store in correlation matrix
                        self._correlation_matrix[(symbol1, symbol2)] = correlation
                        self._correlation_matrix[(symbol2, symbol1)] = correlation
            
            # Find maximum correlation
            max_correlation = max((pair.correlation for pair in correlation_pairs), default=Decimal("0"))
            
            # Calculate portfolio concentration risk
            concentration_risk = await self._calculate_concentration_risk()
            
            # Update timestamp
            self.last_correlation_update = datetime.now()
            
            result = CorrelationAnalysisResult(
                success=True,
                correlation_pairs=correlation_pairs,
                max_correlation=max_correlation,
                portfolio_concentration_risk=concentration_risk,
                last_updated=self.last_correlation_update
            )
            
            self.logger.info(
                "Correlation matrix calculated",
                correlation_pairs=len(correlation_pairs),
                max_correlation=str(max_correlation)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate correlation matrix",
                error=str(e)
            )
            return CorrelationAnalysisResult(
                success=False,
                error_message=f"Correlation calculation failed: {e}"
            )
    
    async def analyze_position_correlations(self) -> CorrelationAnalysisResult:
        """
        Analyze position correlations and identify high correlation groups.
        
        Returns:
            CorrelationAnalysisResult with correlation analysis
        """
        try:
            # First ensure correlation matrix is up to date
            if (self.last_correlation_update is None or 
                datetime.now() - self.last_correlation_update > timedelta(hours=self.config.correlation_update_frequency_hours)):
                correlation_result = await self.calculate_correlation_matrix()
                if not correlation_result.success:
                    return correlation_result
            
            # Find high correlation groups
            high_correlation_groups = []
            processed_symbols = set()
            
            for (symbol1, symbol2), correlation in self._correlation_matrix.items():
                if symbol1 in processed_symbols or symbol2 in processed_symbols:
                    continue
                
                if abs(correlation) > self.config.max_correlation_threshold:
                    # Find all symbols highly correlated with this pair
                    group = [symbol1, symbol2]
                    for symbol in set(pos.symbol for pos in self.portfolio.positions.values()):
                        if symbol in group:
                            continue
                        
                        # Check if symbol is highly correlated with any in the group
                        highly_correlated = any(
                            abs(self._correlation_matrix.get((symbol, group_symbol), Decimal("0"))) > self.config.max_correlation_threshold
                            for group_symbol in group
                        )
                        
                        if highly_correlated:
                            group.append(symbol)
                    
                    if len(group) > 2:
                        high_correlation_groups.append(group)
                        processed_symbols.update(group)
            
            # Calculate maximum correlation
            max_correlation = max(abs(corr) for corr in self._correlation_matrix.values()) if self._correlation_matrix else Decimal("0")
            
            # Calculate concentration risk
            concentration_risk = await self._calculate_concentration_risk()
            
            result = CorrelationAnalysisResult(
                success=True,
                correlation_pairs=[],  # Use existing pairs from last calculation
                max_correlation=max_correlation,
                portfolio_concentration_risk=concentration_risk,
                high_correlation_groups=high_correlation_groups,
                last_updated=self.last_correlation_update
            )
            
            self.logger.info(
                "Position correlations analyzed",
                high_correlation_groups=len(high_correlation_groups),
                max_correlation=str(max_correlation)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to analyze position correlations",
                error=str(e)
            )
            return CorrelationAnalysisResult(
                success=False,
                error_message=f"Correlation analysis failed: {e}"
            )
    
    async def calculate_position_concentration(self) -> ConcentrationResult:
        """
        Calculate position concentration metrics.
        
        Returns:
            ConcentrationResult with concentration analysis
        """
        try:
            # Calculate position weights
            total_value = self.portfolio.total_value
            if total_value <= 0:
                return ConcentrationResult(
                    success=False,
                    error_message="Portfolio has no value for concentration calculation"
                )
            
            position_weights = []
            for position in self.portfolio.positions.values():
                weight = position.market_value / total_value
                position_weights.append(PositionWeight(
                    position_id=position.position_id,
                    symbol=position.symbol,
                    weight=weight,
                    market_value=position.market_value
                ))
            
            # Sort by weight (largest first)
            position_weights.sort(key=lambda x: x.weight, reverse=True)
            
            # Calculate concentration index (Herfindahl-Hirschman Index)
            concentration_index = sum(weight.weight ** 2 for weight in position_weights)
            
            # Calculate largest position percentage
            largest_position_pct = position_weights[0].weight if position_weights else Decimal("0")
            
            # Calculate top 5 concentration
            top_5_concentration = sum(weight.weight for weight in position_weights[:5])
            
            # Calculate diversification ratio (1/HHI)
            diversification_ratio = Decimal("1") / concentration_index if concentration_index > 0 else Decimal("0")
            
            result = ConcentrationResult(
                success=True,
                concentration_index=concentration_index,
                position_weights=position_weights,
                largest_position_pct=largest_position_pct,
                top_5_concentration=top_5_concentration,
                diversification_ratio=diversification_ratio
            )
            
            self.logger.debug(
                "Position concentration calculated",
                concentration_index=str(concentration_index),
                largest_position=str(largest_position_pct),
                diversification_ratio=str(diversification_ratio)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate position concentration",
                error=str(e)
            )
            return ConcentrationResult(
                success=False,
                error_message=f"Concentration calculation failed: {e}"
            )
    
    # Risk Budgeting
    
    async def calculate_risk_budget(
        self,
        symbols: List[str],
        total_risk_budget: Decimal
    ) -> RiskBudgetResult:
        """
        Calculate optimal risk budget allocation.
        
        Args:
            symbols: List of symbols to allocate risk budget to
            total_risk_budget: Total risk budget as percentage of portfolio
            
        Returns:
            RiskBudgetResult with risk allocations
        """
        try:
            allocations = []
            
            # Calculate risk metrics for each symbol
            symbol_metrics = {}
            for symbol in symbols:
                volatility = await self._get_asset_volatility(symbol)
                expected_return = await self._get_expected_return(symbol)
                
                if volatility is None or expected_return is None:
                    continue
                
                # Calculate Sharpe ratio
                excess_return = expected_return - self.config.risk_free_rate
                sharpe_ratio = excess_return / volatility if volatility > 0 else Decimal("0")
                
                symbol_metrics[symbol] = {
                    "volatility": volatility,
                    "expected_return": expected_return,
                    "sharpe_ratio": sharpe_ratio
                }
            
            # Allocate risk budget using equal risk contribution method
            if symbol_metrics:
                # Simple equal risk allocation
                risk_per_asset = total_risk_budget / len(symbol_metrics)
                
                for symbol, metrics in symbol_metrics.items():
                    # Calculate position size based on risk allocation
                    # Position Size = (Risk Budget / Volatility) * Portfolio Value
                    position_size = (risk_per_asset / metrics["volatility"]) * self.portfolio.total_value if metrics["volatility"] > 0 else Decimal("0")
                    
                    allocation = RiskAllocation(
                        symbol=symbol,
                        risk_allocation=risk_per_asset,
                        expected_return=metrics["expected_return"],
                        volatility=metrics["volatility"],
                        sharpe_ratio=metrics["sharpe_ratio"],
                        position_size=position_size
                    )
                    allocations.append(allocation)
            
            # Calculate portfolio-level metrics
            if allocations:
                total_expected_return = sum(alloc.expected_return * alloc.risk_allocation for alloc in allocations) / total_risk_budget
                portfolio_volatility = Decimal(str(math.sqrt(float(sum(alloc.volatility ** 2 * alloc.risk_allocation for alloc in allocations)))))
                portfolio_sharpe = (total_expected_return - self.config.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else Decimal("0")
            else:
                total_expected_return = Decimal("0")
                portfolio_volatility = Decimal("0")
                portfolio_sharpe = Decimal("0")
            
            result = RiskBudgetResult(
                success=True,
                allocations=allocations,
                total_risk_budget=total_risk_budget,
                expected_portfolio_return=total_expected_return,
                portfolio_volatility=portfolio_volatility,
                portfolio_sharpe_ratio=portfolio_sharpe,
                optimization_method="equal_risk_contribution",
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Risk budget calculated",
                symbols=len(symbols),
                allocations=len(allocations),
                total_risk_budget=str(total_risk_budget)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate risk budget",
                error=str(e)
            )
            return RiskBudgetResult(
                success=False,
                error_message=f"Risk budget calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def optimize_portfolio_allocation(self) -> AllocationOptimizationResult:
        """
        Optimize portfolio allocation for risk-adjusted returns.
        
        Returns:
            AllocationOptimizationResult with optimization recommendations
        """
        try:
            current_allocations = []
            symbols = []
            
            # Get current allocations
            total_value = self.portfolio.total_value
            for position in self.portfolio.positions.values():
                if position.status == PositionStatus.OPEN:
                    weight = position.market_value / total_value if total_value > 0 else Decimal("0")
                    current_allocations.append(OptimalAllocation(
                        symbol=position.symbol,
                        current_weight=weight,
                        optimal_weight=weight,  # Will be updated
                        weight_change=Decimal("0"),
                        trade_amount=Decimal("0")
                    ))
                    symbols.append(position.symbol)
            
            # Calculate optimal allocations using risk budgeting
            risk_budget_result = await self.calculate_risk_budget(
                symbols=symbols,
                total_risk_budget=self.config.max_portfolio_risk_pct
            )
            
            if not risk_budget_result.success:
                return AllocationOptimizationResult(
                    success=False,
                    error_message=f"Risk budget calculation failed: {risk_budget_result.error_message}",
                    calculation_time=datetime.now()
                )
            
            # Update optimal allocations
            optimal_allocations = []
            rebalancing_trades = []
            rebalancing_needed = False
            
            for current_alloc in current_allocations:
                # Find corresponding risk allocation
                risk_alloc = next(
                    (ra for ra in risk_budget_result.allocations if ra.symbol == current_alloc.symbol),
                    None
                )
                
                if risk_alloc:
                    # Calculate optimal weight based on risk allocation
                    optimal_weight = risk_alloc.position_size / total_value if total_value > 0 else Decimal("0")
                    weight_change = optimal_weight - current_alloc.current_weight
                    trade_amount = weight_change * total_value
                    
                    optimal_alloc = OptimalAllocation(
                        symbol=current_alloc.symbol,
                        current_weight=current_alloc.current_weight,
                        optimal_weight=optimal_weight,
                        weight_change=weight_change,
                        trade_amount=trade_amount
                    )
                    optimal_allocations.append(optimal_alloc)
                    
                    # Check if rebalancing is needed (>5% weight change)
                    if abs(weight_change) > Decimal("0.05"):
                        rebalancing_needed = True
                        rebalancing_trades.append({
                            "symbol": current_alloc.symbol,
                            "action": "buy" if weight_change > 0 else "sell",
                            "amount": abs(trade_amount),
                            "weight_change": weight_change
                        })
            
            result = AllocationOptimizationResult(
                success=True,
                current_allocations=current_allocations,
                optimal_allocations=optimal_allocations,
                rebalancing_needed=rebalancing_needed,
                rebalancing_trades=rebalancing_trades,
                optimization_objective="max_sharpe",
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Portfolio allocation optimized",
                positions=len(optimal_allocations),
                rebalancing_needed=rebalancing_needed,
                trades_needed=len(rebalancing_trades)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to optimize portfolio allocation",
                error=str(e)
            )
            return AllocationOptimizationResult(
                success=False,
                error_message=f"Allocation optimization failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Emergency Protection
    
    async def check_emergency_conditions(self) -> EmergencyConditionResult:
        """
        Check for emergency stop conditions.
        
        Returns:
            EmergencyConditionResult with emergency status
        """
        try:
            emergency_triggered = False
            trigger_reason = None
            trigger_value = None
            threshold_value = None
            recommended_actions = []
            
            # Check drawdown limit
            if self.portfolio.drawdown_metrics:
                current_drawdown = self.portfolio.drawdown_metrics.current_drawdown
                if current_drawdown >= self.config.emergency_stop_loss_pct:
                    emergency_triggered = True
                    trigger_reason = "drawdown_limit_exceeded"
                    trigger_value = current_drawdown
                    threshold_value = self.config.emergency_stop_loss_pct
                    recommended_actions.extend(["close_all_positions", "stop_new_trades"])
            
            # Check daily loss limit
            daily_loss = await self._get_daily_loss()
            daily_loss_limit = self.portfolio.config.max_daily_loss_pct * self.portfolio.total_value
            if daily_loss > daily_loss_limit:
                if not emergency_triggered or daily_loss > trigger_value:
                    emergency_triggered = True
                    trigger_reason = "daily_loss_limit_exceeded"
                    trigger_value = daily_loss / self.portfolio.total_value
                    threshold_value = self.portfolio.config.max_daily_loss_pct
                    recommended_actions.extend(["halt_trading", "review_positions"])
            
            # Take emergency action if triggered
            if emergency_triggered:
                await self._trigger_emergency_stop()
            
            result = EmergencyConditionResult(
                emergency_triggered=emergency_triggered,
                trigger_reason=trigger_reason,
                trigger_value=trigger_value,
                threshold_value=threshold_value,
                recommended_actions=recommended_actions,
                check_time=datetime.now()
            )
            
            if emergency_triggered:
                self.logger.critical(
                    "Emergency condition triggered",
                    reason=trigger_reason,
                    trigger_value=str(trigger_value),
                    threshold=str(threshold_value)
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to check emergency conditions",
                error=str(e)
            )
            return EmergencyConditionResult(
                emergency_triggered=False,
                check_time=datetime.now()
            )
    
    async def check_liquidation_risks(self) -> LiquidationRiskResult:
        """
        Check positions for liquidation risk.
        
        Returns:
            LiquidationRiskResult with at-risk positions
        """
        try:
            at_risk_positions = []
            total_at_risk_value = Decimal("0")
            highest_risk_position = None
            highest_risk_score = Decimal("0")
            
            # Check each leveraged position
            for position in self.portfolio.open_positions.values():
                if position.liquidation_price is None:
                    continue
                
                # Calculate distance to liquidation
                if position.side == "LONG":
                    distance_pct = (position.current_price - position.liquidation_price) / position.current_price
                elif position.side == "SHORT":
                    distance_pct = (position.liquidation_price - position.current_price) / position.current_price
                else:
                    continue
                
                # Position is at risk if within 10% of liquidation
                if distance_pct <= Decimal("0.1"):
                    # Determine recommended action based on risk level
                    if distance_pct <= Decimal("0.02"):  # 2% or less
                        recommended_action = "close_position"
                    elif distance_pct <= Decimal("0.05"):  # 5% or less
                        recommended_action = "reduce_size"
                    else:  # Up to 10%
                        recommended_action = "add_margin"
                    
                    risk_position = LiquidationRiskPosition(
                        position_id=position.position_id,
                        symbol=position.symbol,
                        current_price=position.current_price,
                        liquidation_price=position.liquidation_price,
                        distance_to_liquidation=distance_pct,
                        margin_ratio=position.margin_ratio,
                        recommended_action=recommended_action
                    )
                    
                    at_risk_positions.append(risk_position)
                    total_at_risk_value += position.market_value
                    
                    # Track highest risk position
                    risk_score = Decimal("1") - distance_pct  # Higher score = higher risk
                    if risk_score > highest_risk_score:
                        highest_risk_score = risk_score
                        highest_risk_position = position.position_id
            
            result = LiquidationRiskResult(
                success=True,
                at_risk_positions=at_risk_positions,
                total_at_risk_value=total_at_risk_value,
                highest_risk_position=highest_risk_position,
                check_time=datetime.now()
            )
            
            if at_risk_positions:
                self.logger.warning(
                    "Liquidation risk detected",
                    at_risk_positions=len(at_risk_positions),
                    total_at_risk_value=str(total_at_risk_value)
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to check liquidation risks",
                error=str(e)
            )
            return LiquidationRiskResult(
                success=False,
                error_message=f"Liquidation risk check failed: {e}",
                check_time=datetime.now()
            )
    
    async def check_stop_loss_triggers(self) -> StopLossResult:
        """
        Check for stop loss triggers and execute if needed.
        
        Returns:
            StopLossResult with triggered stop losses
        """
        try:
            triggered_positions = []
            positions_closed = 0
            total_realized_loss = Decimal("0")
            
            # Check each position for stop loss trigger
            for position in self.portfolio.open_positions.values():
                if position.stop_loss_price is None:
                    continue
                
                # Check if stop loss is triggered
                stop_loss_triggered = False
                if position.side == "LONG" and position.current_price <= position.stop_loss_price:
                    stop_loss_triggered = True
                elif position.side == "SHORT" and position.current_price >= position.stop_loss_price:
                    stop_loss_triggered = True
                elif position.position_type == PositionType.SPOT and position.current_price <= position.stop_loss_price:
                    stop_loss_triggered = True
                
                if stop_loss_triggered:
                    trigger = StopLossTrigger(
                        position_id=position.position_id,
                        symbol=position.symbol,
                        current_price=position.current_price,
                        stop_loss_price=position.stop_loss_price,
                        trigger_time=datetime.now()
                    )
                    triggered_positions.append(trigger)
                    
                    # Execute stop loss
                    try:
                        await self._execute_stop_loss(position.position_id)
                        positions_closed += 1
                        
                        # Calculate realized loss
                        loss = (position.entry_price - position.stop_loss_price) * position.size
                        if position.side == "SHORT":
                            loss = -loss
                        total_realized_loss += loss
                        
                    except Exception as e:
                        self.logger.error(
                            "Failed to execute stop loss",
                            position_id=str(position.position_id),
                            error=str(e)
                        )
            
            result = StopLossResult(
                success=True,
                triggered_positions=triggered_positions,
                positions_closed=positions_closed,
                total_realized_loss=total_realized_loss,
                check_time=datetime.now()
            )
            
            if triggered_positions:
                self.logger.info(
                    "Stop losses triggered",
                    triggered_count=len(triggered_positions),
                    positions_closed=positions_closed,
                    total_loss=str(total_realized_loss)
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to check stop loss triggers",
                error=str(e)
            )
            return StopLossResult(
                success=False,
                error_message=f"Stop loss check failed: {e}",
                check_time=datetime.now()
            )
    
    async def check_margin_requirements(self) -> MarginCallResult:
        """
        Check margin requirements for leveraged positions.
        
        Returns:
            MarginCallResult with margin call information
        """
        try:
            margin_calls = []
            total_margin_required = Decimal("0")
            critical_positions = 0
            
            # Check each leveraged position
            for position in self.portfolio.open_positions.values():
                if position.leverage <= 1 or position.margin_used is None:
                    continue
                
                # Calculate current margin ratio
                current_margin = position.margin_used
                position_value = position.market_value
                required_margin_ratio = Decimal("1") / position.leverage  # e.g., 20% for 5x leverage
                required_margin = position_value * required_margin_ratio
                
                # Check if margin call is needed (current margin < 150% of required)
                margin_buffer = required_margin * Decimal("1.5")
                if current_margin < margin_buffer:
                    margin_call_amount = margin_buffer - current_margin
                    
                    # Calculate liquidation risk based on margin shortfall
                    liquidation_risk = Decimal("1") - (current_margin / required_margin)
                    liquidation_risk = max(Decimal("0"), min(Decimal("1"), liquidation_risk))
                    
                    margin_call = MarginCall(
                        position_id=position.position_id,
                        symbol=position.symbol,
                        current_margin=current_margin,
                        required_margin=margin_buffer,
                        margin_call_amount=margin_call_amount,
                        liquidation_risk=liquidation_risk,
                        deadline=datetime.now() + timedelta(hours=24)  # 24-hour deadline
                    )
                    
                    margin_calls.append(margin_call)
                    total_margin_required += margin_call_amount
                    
                    # Critical if liquidation risk > 50%
                    if liquidation_risk > Decimal("0.5"):
                        critical_positions += 1
            
            result = MarginCallResult(
                success=True,
                margin_calls=margin_calls,
                total_margin_required=total_margin_required,
                critical_positions=critical_positions,
                check_time=datetime.now()
            )
            
            if margin_calls:
                self.logger.warning(
                    "Margin calls required",
                    margin_calls=len(margin_calls),
                    total_margin_required=str(total_margin_required),
                    critical_positions=critical_positions
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to check margin requirements",
                error=str(e)
            )
            return MarginCallResult(
                success=False,
                error_message=f"Margin requirement check failed: {e}",
                check_time=datetime.now()
            )
    
    # Multi-Chain Risk Aggregation
    
    async def aggregate_risk_by_chain(self) -> ChainRiskResult:
        """
        Aggregate risk by blockchain chain.
        
        Returns:
            ChainRiskResult with chain-level risk breakdown
        """
        try:
            chain_risks = []
            total_exposure = Decimal("0")
            highest_risk_chain = None
            highest_risk_amount = Decimal("0")
            
            # Group positions by chain
            chain_positions = {}
            for position in self.portfolio.positions.values():
                if position.chain not in chain_positions:
                    chain_positions[position.chain] = []
                chain_positions[position.chain].append(position)
            
            # Calculate risk for each chain
            for chain, positions in chain_positions.items():
                chain_exposure = sum(pos.market_value for pos in positions if pos.status == PositionStatus.OPEN)
                chain_pnl = sum(pos.unrealized_pnl for pos in positions if pos.status == PositionStatus.OPEN)
                risk_percentage = chain_exposure / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
                
                # DEX breakdown within chain
                dex_breakdown = {}
                for position in positions:
                    if position.status == PositionStatus.OPEN:
                        if position.dex_name not in dex_breakdown:
                            dex_breakdown[position.dex_name] = Decimal("0")
                        dex_breakdown[position.dex_name] += position.market_value
                
                chain_risk = ChainRisk(
                    chain=chain,
                    position_count=len([p for p in positions if p.status == PositionStatus.OPEN]),
                    total_exposure=chain_exposure,
                    risk_percentage=risk_percentage,
                    unrealized_pnl=chain_pnl,
                    dex_breakdown=dex_breakdown
                )
                
                chain_risks.append(chain_risk)
                total_exposure += chain_exposure
                
                # Track highest risk chain
                if chain_exposure > highest_risk_amount:
                    highest_risk_amount = chain_exposure
                    highest_risk_chain = chain
            
            # Calculate chain concentration risk (similar to HHI)
            chain_concentration_risk = sum(cr.risk_percentage ** 2 for cr in chain_risks)
            
            result = ChainRiskResult(
                success=True,
                chain_risks=chain_risks,
                total_exposure=total_exposure,
                highest_risk_chain=highest_risk_chain,
                chain_concentration_risk=chain_concentration_risk,
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Chain risk aggregated",
                chains=len(chain_risks),
                total_exposure=str(total_exposure),
                highest_risk_chain=highest_risk_chain.value if highest_risk_chain else None
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to aggregate risk by chain",
                error=str(e)
            )
            return ChainRiskResult(
                success=False,
                error_message=f"Chain risk aggregation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def aggregate_risk_by_dex(self) -> DexRiskResult:
        """
        Aggregate risk by DEX.
        
        Returns:
            DexRiskResult with DEX-level risk breakdown
        """
        try:
            dex_risks = []
            total_exposure = Decimal("0")
            highest_risk_dex = None
            highest_risk_amount = Decimal("0")
            
            # Group positions by DEX
            dex_positions = {}
            for position in self.portfolio.positions.values():
                if position.dex_name not in dex_positions:
                    dex_positions[position.dex_name] = []
                dex_positions[position.dex_name].append(position)
            
            # Calculate risk for each DEX
            for dex_name, positions in dex_positions.items():
                dex_exposure = sum(pos.market_value for pos in positions if pos.status == PositionStatus.OPEN)
                dex_pnl = sum(pos.unrealized_pnl for pos in positions if pos.status == PositionStatus.OPEN)
                risk_percentage = dex_exposure / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
                
                # Chain breakdown within DEX
                chain_breakdown = {}
                for position in positions:
                    if position.status == PositionStatus.OPEN:
                        if position.chain not in chain_breakdown:
                            chain_breakdown[position.chain] = Decimal("0")
                        chain_breakdown[position.chain] += position.market_value
                
                dex_risk = DexRisk(
                    dex_name=dex_name,
                    position_count=len([p for p in positions if p.status == PositionStatus.OPEN]),
                    total_exposure=dex_exposure,
                    risk_percentage=risk_percentage,
                    unrealized_pnl=dex_pnl,
                    chain_breakdown=chain_breakdown
                )
                
                dex_risks.append(dex_risk)
                total_exposure += dex_exposure
                
                # Track highest risk DEX
                if dex_exposure > highest_risk_amount:
                    highest_risk_amount = dex_exposure
                    highest_risk_dex = dex_name
            
            # Calculate DEX concentration risk
            dex_concentration_risk = sum(dr.risk_percentage ** 2 for dr in dex_risks)
            
            result = DexRiskResult(
                success=True,
                dex_risks=dex_risks,
                total_exposure=total_exposure,
                highest_risk_dex=highest_risk_dex,
                dex_concentration_risk=dex_concentration_risk,
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "DEX risk aggregated",
                dexs=len(dex_risks),
                total_exposure=str(total_exposure),
                highest_risk_dex=highest_risk_dex
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to aggregate risk by DEX",
                error=str(e)
            )
            return DexRiskResult(
                success=False,
                error_message=f"DEX risk aggregation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def analyze_cross_chain_correlations(self) -> CrossChainCorrelationResult:
        """
        Analyze correlations across different chains.
        
        Returns:
            CrossChainCorrelationResult with cross-chain correlation data
        """
        try:
            cross_chain_correlations = []
            chain_correlation_matrix = {}
            max_cross_chain_correlation = Decimal("0")
            
            # Get symbols by chain
            chain_symbols = {}
            for position in self.portfolio.positions.values():
                if position.chain not in chain_symbols:
                    chain_symbols[position.chain] = []
                if position.symbol not in chain_symbols[position.chain]:
                    chain_symbols[position.chain].append(position.symbol)
            
            # Calculate cross-chain correlations
            chains = list(chain_symbols.keys())
            for i, chain1 in enumerate(chains):
                for chain2 in chains[i+1:]:
                    # Find symbols that might be correlated across chains
                    for symbol1 in chain_symbols[chain1]:
                        for symbol2 in chain_symbols[chain2]:
                            # Check if symbols are related (e.g., BTC/USDC on different chains)
                            if self._symbols_are_related(symbol1, symbol2):
                                correlation = await self._calculate_cross_chain_correlation(
                                    symbol1, chain1, symbol2, chain2
                                )
                                
                                if correlation is not None:
                                    cross_chain_corr = CrossChainCorrelation(
                                        symbol1=symbol1,
                                        chain1=chain1,
                                        symbol2=symbol2,
                                        chain2=chain2,
                                        correlation=correlation,
                                        observations=self.config.lookback_period_days
                                    )
                                    cross_chain_correlations.append(cross_chain_corr)
                                    
                                    if abs(correlation) > abs(max_cross_chain_correlation):
                                        max_cross_chain_correlation = correlation
                    
                    # Calculate overall chain correlation
                    chain_corr = await self._calculate_chain_correlation(chain1, chain2)
                    if chain_corr is not None:
                        chain_correlation_matrix[(chain1, chain2)] = chain_corr
                        chain_correlation_matrix[(chain2, chain1)] = chain_corr
            
            result = CrossChainCorrelationResult(
                success=True,
                cross_chain_correlations=cross_chain_correlations,
                max_cross_chain_correlation=max_cross_chain_correlation,
                chain_correlation_matrix=chain_correlation_matrix,
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Cross-chain correlations analyzed",
                correlations=len(cross_chain_correlations),
                max_correlation=str(max_cross_chain_correlation)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to analyze cross-chain correlations",
                error=str(e)
            )
            return CrossChainCorrelationResult(
                success=False,
                error_message=f"Cross-chain correlation analysis failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_concentration_risk(self) -> ConcentrationRiskResult:
        """
        Calculate concentration risk across chains and assets.
        
        Returns:
            ConcentrationRiskResult with comprehensive concentration analysis
        """
        try:
            # Group positions by underlying asset (e.g., BTC across different chains)
            asset_exposures = {}
            total_portfolio_value = self.portfolio.total_value
            
            for position in self.portfolio.open_positions.values():
                # Extract base asset from symbol (e.g., BTC from BTC/USDC)
                base_asset = self._extract_base_asset(position.symbol)
                
                if base_asset not in asset_exposures:
                    asset_exposures[base_asset] = {
                        "total_exposure": Decimal("0"),
                        "chains": set(),
                        "positions": []
                    }
                
                asset_exposures[base_asset]["total_exposure"] += position.market_value
                asset_exposures[base_asset]["chains"].add(position.chain)
                asset_exposures[base_asset]["positions"].append(position)
            
            # Create asset concentration list
            asset_concentrations = []
            for asset, data in asset_exposures.items():
                weight = data["total_exposure"] / total_portfolio_value if total_portfolio_value > 0 else Decimal("0")
                concentration = AssetConcentration(
                    asset_symbol=asset,
                    weight=weight,
                    exposure_chains=list(data["chains"]),
                    total_positions=len(data["positions"])
                )
                asset_concentrations.append(concentration)
            
            # Sort by weight (largest first)
            asset_concentrations.sort(key=lambda x: x.weight, reverse=True)
            
            # Calculate concentration index (Herfindahl-Hirschman Index)
            concentration_index = sum(ac.weight ** 2 for ac in asset_concentrations)
            
            # Find largest asset weight
            largest_asset_weight = asset_concentrations[0].weight if asset_concentrations else Decimal("0")
            
            # Calculate effective number of positions (1/HHI)
            effective_number_of_positions = Decimal("1") / concentration_index if concentration_index > 0 else Decimal("0")
            
            result = ConcentrationRiskResult(
                success=True,
                concentration_index=concentration_index,
                asset_concentrations=asset_concentrations,
                largest_asset_weight=largest_asset_weight,
                effective_number_of_positions=effective_number_of_positions,
                calculation_time=datetime.now()
            )
            
            self.logger.info(
                "Concentration risk calculated",
                unique_assets=len(asset_concentrations),
                concentration_index=str(concentration_index),
                largest_asset_weight=str(largest_asset_weight)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate concentration risk",
                error=str(e)
            )
            return ConcentrationRiskResult(
                success=False,
                error_message=f"Concentration risk calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Private Helper Methods
    
    async def _calculate_kelly_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_level: Optional[Decimal],
        side: str
    ) -> PositionSizeRecommendation:
        """Calculate position size using Kelly criterion."""
        # Get win probability and win/loss ratio
        win_probability = await self._calculate_win_probability(symbol)
        win_loss_ratio = await self._calculate_average_win_loss_ratio(symbol)
        
        if win_probability is None or win_loss_ratio is None:
            # Use default values if no historical data
            win_probability = Decimal("0.55")  # Slight edge
            win_loss_ratio = Decimal("1.5")    # 1.5:1 win/loss ratio
        
        # Kelly fraction = (bp - q) / b
        # where b = win/loss ratio, p = win probability, q = loss probability
        loss_probability = Decimal("1") - win_probability
        kelly_fraction = (win_loss_ratio * win_probability - loss_probability) / win_loss_ratio
        
        # Cap Kelly fraction at reasonable level (25%)
        kelly_fraction = min(kelly_fraction, Decimal("0.25"))
        kelly_fraction = max(kelly_fraction, Decimal("0.01"))  # Minimum 1%
        
        # Calculate position size
        risk_amount = self.portfolio.total_value * kelly_fraction
        
        if stop_loss_price:
            # Position size = Risk Amount / Stop Loss Distance
            stop_distance = abs(entry_price - stop_loss_price)
            position_size = risk_amount / stop_distance
        else:
            # Use 2% default risk if no stop loss
            default_risk_pct = Decimal("0.02")
            position_size = self.portfolio.total_value * default_risk_pct / entry_price
        
        position_value = position_size * entry_price
        position_risk_pct = risk_amount / self.portfolio.total_value
        
        return PositionSizeRecommendation(
            success=True,
            symbol=symbol,
            recommended_size=position_size,
            recommended_size_usd=position_value,
            risk_amount=risk_amount,
            position_risk_pct=position_risk_pct,
            sizing_method="kelly",
            kelly_fraction=kelly_fraction
        )
    
    async def _calculate_fixed_fractional_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        risk_fraction: Optional[Decimal],
        side: str
    ) -> PositionSizeRecommendation:
        """Calculate position size using fixed fractional method."""
        if risk_fraction is None:
            risk_fraction = self.config.max_portfolio_risk_pct
        
        risk_amount = self.portfolio.total_value * risk_fraction
        
        if stop_loss_price:
            stop_distance = abs(entry_price - stop_loss_price)
            position_size = risk_amount / stop_distance
        else:
            # Use default 5% stop loss distance
            default_stop_distance = entry_price * Decimal("0.05")
            position_size = risk_amount / default_stop_distance
        
        position_value = position_size * entry_price
        position_risk_pct = risk_fraction
        
        return PositionSizeRecommendation(
            success=True,
            symbol=symbol,
            recommended_size=position_size,
            recommended_size_usd=position_value,
            risk_amount=risk_amount,
            position_risk_pct=position_risk_pct,
            sizing_method="fixed_fractional"
        )
    
    async def _calculate_volatility_adjusted_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        target_volatility: Optional[Decimal],
        side: str
    ) -> PositionSizeRecommendation:
        """Calculate position size using volatility adjustment."""
        asset_volatility = await self._get_asset_volatility(symbol)
        if asset_volatility is None:
            asset_volatility = Decimal("0.3")  # Default 30% volatility
        
        if target_volatility is None:
            target_volatility = Decimal("0.15")  # Default 15% target
        
        # Volatility adjustment factor
        vol_adjustment = target_volatility / asset_volatility
        
        # Base position size (2% of portfolio)
        base_risk = self.portfolio.total_value * Decimal("0.02")
        adjusted_risk = base_risk * vol_adjustment
        
        if stop_loss_price:
            stop_distance = abs(entry_price - stop_loss_price)
            position_size = adjusted_risk / stop_distance
        else:
            # Use volatility-based stop (2 * daily volatility)
            vol_stop_distance = entry_price * asset_volatility * Decimal("2")
            position_size = adjusted_risk / vol_stop_distance
        
        position_value = position_size * entry_price
        position_risk_pct = adjusted_risk / self.portfolio.total_value
        
        return PositionSizeRecommendation(
            success=True,
            symbol=symbol,
            recommended_size=position_size,
            recommended_size_usd=position_value,
            risk_amount=adjusted_risk,
            position_risk_pct=position_risk_pct,
            sizing_method="volatility_adjusted",
            volatility_adjustment=vol_adjustment,
            current_volatility=asset_volatility
        )
    
    async def _apply_position_size_limits(self, result: PositionSizeRecommendation) -> PositionSizeRecommendation:
        """Apply position size limits to recommendation."""
        max_position_value = self.portfolio.total_value * self.config.max_single_position_pct
        
        if result.recommended_size_usd > max_position_value:
            # Cap position size
            scaling_factor = max_position_value / result.recommended_size_usd
            result.recommended_size *= scaling_factor
            result.recommended_size_usd = max_position_value
            result.size_capped = True
            result.warnings.append(f"Position size capped at {self.config.max_single_position_pct:.1%} of portfolio")
            
            # Update position risk percentage to match the capped size
            result.position_risk_pct = self.config.max_single_position_pct
        
        return result
    
    async def _apply_dynamic_adjustments(self, result: PositionSizeRecommendation) -> PositionSizeRecommendation:
        """Apply dynamic adjustments to position size."""
        # Portfolio risk adjustment
        current_portfolio_risk = await self._get_current_portfolio_risk()
        if current_portfolio_risk > self.config.max_portfolio_risk_pct * Decimal("0.8"):
            risk_reduction = Decimal("0.8")  # Reduce by 20%
            result.recommended_size *= risk_reduction
            result.recommended_size_usd *= risk_reduction
            result.risk_adjustment = risk_reduction
            result.warnings.append("Position size reduced due to high portfolio risk")
        
        # Correlation adjustment
        correlation_adjustment = await self._get_correlation_adjustment(result.symbol)
        if correlation_adjustment is not None and correlation_adjustment < Decimal("1"):
            result.recommended_size *= correlation_adjustment
            result.recommended_size_usd *= correlation_adjustment
            result.correlation_adjustment = correlation_adjustment
            result.warnings.append("Position size reduced due to high correlation with existing positions")
        
        return result
    
    async def _check_position_size_limits(self) -> List[RiskLimitCheck]:
        """Check position size limits."""
        checks = []
        
        for position in self.portfolio.open_positions.values():
            position_value = position.market_value
            position_pct = position_value / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            
            limit_exceeded = position_pct > self.config.max_single_position_pct
            severity = RiskAlertSeverity.CRITICAL if limit_exceeded else RiskAlertSeverity.INFO
            
            if position_pct > self.config.max_single_position_pct * self.config.alert_thresholds["warning"]:
                severity = RiskAlertSeverity.WARNING
            if position_pct > self.config.max_single_position_pct * self.config.alert_thresholds["critical"]:
                severity = RiskAlertSeverity.CRITICAL
            
            check = RiskLimitCheck(
                limit_type="position_size",
                current_value=position_pct,
                limit_value=self.config.max_single_position_pct,
                limit_exceeded=limit_exceeded,
                severity=severity,
                position_id=position.position_id,
                chain=position.chain,
                dex_name=position.dex_name,
                message=f"Position {position.symbol} size: {position_pct:.2%}"
            )
            checks.append(check)
        
        return checks
    
    async def _check_portfolio_risk_limits(self) -> List[RiskLimitCheck]:
        """Check portfolio risk limits."""
        portfolio_risk = await self._calculate_total_portfolio_risk()
        limit_exceeded = portfolio_risk > self.config.max_portfolio_risk_pct
        
        severity = RiskAlertSeverity.INFO
        if portfolio_risk > self.config.max_portfolio_risk_pct * self.config.alert_thresholds["warning"]:
            severity = RiskAlertSeverity.WARNING
        if portfolio_risk > self.config.max_portfolio_risk_pct * self.config.alert_thresholds["critical"]:
            severity = RiskAlertSeverity.CRITICAL
        
        check = RiskLimitCheck(
            limit_type="portfolio_risk",
            current_value=portfolio_risk,
            limit_value=self.config.max_portfolio_risk_pct,
            limit_exceeded=limit_exceeded,
            severity=severity,
            message=f"Portfolio risk: {portfolio_risk:.2%}"
        )
        
        return [check]
    
    async def _check_daily_loss_limits(self) -> List[RiskLimitCheck]:
        """Check daily loss limits."""
        daily_loss = await self._get_daily_loss()
        daily_loss_pct = daily_loss / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
        limit_exceeded = daily_loss_pct > self.portfolio.config.max_daily_loss_pct
        
        severity = RiskAlertSeverity.INFO
        if daily_loss_pct > self.portfolio.config.max_daily_loss_pct * self.config.alert_thresholds["warning"]:
            severity = RiskAlertSeverity.WARNING
        if daily_loss_pct > self.portfolio.config.max_daily_loss_pct * self.config.alert_thresholds["critical"]:
            severity = RiskAlertSeverity.CRITICAL
        
        check = RiskLimitCheck(
            limit_type="daily_loss",
            current_value=daily_loss_pct,
            limit_value=self.portfolio.config.max_daily_loss_pct,
            limit_exceeded=limit_exceeded,
            severity=severity,
            message=f"Daily loss: {daily_loss_pct:.2%}"
        )
        
        return [check]
    
    async def _check_drawdown_limits(self) -> List[RiskLimitCheck]:
        """Check drawdown limits."""
        checks = []
        
        if self.portfolio.drawdown_metrics:
            current_drawdown = self.portfolio.drawdown_metrics.current_drawdown
            limit_exceeded = current_drawdown > self.portfolio.config.max_drawdown_pct
            
            severity = RiskAlertSeverity.INFO
            if current_drawdown > self.portfolio.config.max_drawdown_pct * self.config.alert_thresholds["warning"]:
                severity = RiskAlertSeverity.WARNING
            if current_drawdown > self.portfolio.config.max_drawdown_pct * self.config.alert_thresholds["critical"]:
                severity = RiskAlertSeverity.CRITICAL
            
            check = RiskLimitCheck(
                limit_type="drawdown",
                current_value=current_drawdown,
                limit_value=self.portfolio.config.max_drawdown_pct,
                limit_exceeded=limit_exceeded,
                severity=severity,
                message=f"Drawdown: {current_drawdown:.2%}"
            )
            checks.append(check)
        
        return checks
    
    async def _check_correlation_limits(self) -> List[RiskLimitCheck]:
        """Check correlation limits."""
        checks = []
        
        for (symbol1, symbol2), correlation in self._correlation_matrix.items():
            if abs(correlation) > self.config.max_correlation_threshold:
                check = RiskLimitCheck(
                    limit_type="correlation",
                    current_value=abs(correlation),
                    limit_value=self.config.max_correlation_threshold,
                    limit_exceeded=True,
                    severity=RiskAlertSeverity.WARNING,
                    message=f"High correlation between {symbol1} and {symbol2}: {correlation:.2%}"
                )
                checks.append(check)
        
        return checks
    
    async def _check_leverage_limits(self) -> List[RiskLimitCheck]:
        """Check leverage limits."""
        checks = []
        
        for position in self.portfolio.open_positions.values():
            if position.leverage > self.config.max_leverage:
                check = RiskLimitCheck(
                    limit_type="leverage",
                    current_value=position.leverage,
                    limit_value=self.config.max_leverage,
                    limit_exceeded=True,
                    severity=RiskAlertSeverity.CRITICAL,
                    position_id=position.position_id,
                    message=f"Position {position.symbol} leverage exceeds limit: {position.leverage}x"
                )
                checks.append(check)
        
        return checks
    
    async def _calculate_total_portfolio_risk(self) -> Decimal:
        """Calculate total portfolio risk."""
        # Simple implementation: sum of position risks
        total_risk = Decimal("0")
        
        for position in self.portfolio.open_positions.values():
            # Position risk = position value * volatility
            position_volatility = await self._get_asset_volatility(position.symbol)
            if position_volatility is None:
                position_volatility = Decimal("0.3")  # Default
            
            position_risk = (position.market_value / self.portfolio.total_value) * position_volatility
            total_risk += position_risk
        
        return total_risk
    
    async def _calculate_position_risk(self, position: Position) -> PositionRisk:
        """Calculate individual position risk."""
        # Position weight in portfolio
        weight = position.market_value / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
        
        # Asset volatility
        volatility = await self._get_asset_volatility(position.symbol)
        if volatility is None:
            volatility = Decimal("0.3")
        
        # Risk = weight * volatility
        current_risk = weight * volatility
        risk_percentage = current_risk * 100  # As percentage
        
        # Risk score (0-100)
        risk_score = min(Decimal("100"), risk_percentage * 10)  # Scale risk percentage
        
        # Liquidation risk for leveraged positions
        liquidation_risk_pct = None
        if position.liquidation_price is not None:
            if position.side == "LONG":
                liquidation_risk_pct = (position.current_price - position.liquidation_price) / position.current_price
            elif position.side == "SHORT":
                liquidation_risk_pct = (position.liquidation_price - position.current_price) / position.current_price
            liquidation_risk_pct = max(Decimal("0"), liquidation_risk_pct)
        
        return PositionRisk(
            position_id=position.position_id,
            symbol=position.symbol,
            current_risk=current_risk,
            risk_percentage=risk_percentage,
            unrealized_pnl=position.unrealized_pnl,
            risk_score=risk_score,
            liquidation_risk_pct=liquidation_risk_pct
        )
    
    async def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        return {
            "portfolio_value": self.portfolio.total_value,
            "cash_balance": self.portfolio.cash_balance,
            "total_positions": len(self.portfolio.open_positions),
            "total_unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "portfolio_leverage": await self._calculate_portfolio_leverage(),
        }
    
    async def _get_portfolio_returns(self) -> List[Decimal]:
        """Get historical portfolio returns."""
        # Simplified implementation - would need historical portfolio values
        # For now, return mock data
        returns = []
        for i in range(30):  # 30 days of returns
            # Generate random returns around 0
            daily_return = Decimal(str((i % 10 - 5) * 0.002))  # -1% to +1%
            returns.append(daily_return)
        return returns
    
    async def _calculate_historical_var(self, returns: List[Decimal], confidence_level: Decimal) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using historical method."""
        if len(returns) < 2:
            return Decimal("0"), Decimal("0")
        
        # Sort returns (worst first)
        sorted_returns = sorted(returns)
        
        # Find percentile
        percentile = 1 - confidence_level
        index = int(len(sorted_returns) * percentile)
        index = max(0, min(index, len(sorted_returns) - 1))
        
        var_return = sorted_returns[index]
        var_percentage = abs(var_return)
        var_amount = var_percentage * self.portfolio.total_value
        
        return var_amount, var_percentage
    
    async def _calculate_expected_shortfall(self, confidence_level: Decimal) -> Decimal:
        """Calculate expected shortfall (CVaR)."""
        returns = await self._get_portfolio_returns()
        return await self._calculate_expected_shortfall_from_returns(returns, confidence_level)
    
    async def _calculate_expected_shortfall_from_returns(self, returns: List[Decimal], confidence_level: Decimal) -> Decimal:
        """Calculate expected shortfall from returns."""
        if len(returns) < 2:
            return Decimal("0")
        
        # Sort returns (worst first)
        sorted_returns = sorted(returns)
        
        # Find tail returns (beyond VaR)
        percentile = 1 - confidence_level
        tail_index = int(len(sorted_returns) * percentile)
        tail_returns = sorted_returns[:tail_index] if tail_index > 0 else [sorted_returns[0]]
        
        # Average of tail returns
        if tail_returns:
            avg_tail_return = sum(tail_returns) / len(tail_returns)
            return abs(avg_tail_return) * self.portfolio.total_value
        
        return Decimal("0")
    
    async def _calculate_portfolio_volatility(self) -> Decimal:
        """Calculate portfolio volatility."""
        returns = await self._get_portfolio_returns()
        if len(returns) < 2:
            return Decimal("0.2")  # Default 20%
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        volatility = Decimal(str(math.sqrt(float(variance))))
        
        return volatility
    
    async def _calculate_var_95(self) -> Decimal:
        """Calculate 95% VaR."""
        var_result = await self.calculate_portfolio_var(Decimal("0.95"))
        return var_result.var_amount if var_result.success else Decimal("0")
    
    async def _calculate_var_99(self) -> Decimal:
        """Calculate 99% VaR."""
        var_result = await self.calculate_portfolio_var(Decimal("0.99"))
        return var_result.var_amount if var_result.success else Decimal("0")
    
    async def _calculate_max_position_risk(self) -> Decimal:
        """Calculate maximum single position risk."""
        max_risk = Decimal("0")
        
        for position in self.portfolio.open_positions.values():
            position_weight = position.market_value / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
            volatility = await self._get_asset_volatility(position.symbol) or Decimal("0.3")
            position_risk = position_weight * volatility
            
            if position_risk > max_risk:
                max_risk = position_risk
        
        return max_risk
    
    async def _calculate_concentration_risk(self) -> Decimal:
        """Calculate portfolio concentration risk (HHI)."""
        if self.portfolio.total_value <= 0:
            return Decimal("0")
        
        concentration = Decimal("0")
        for position in self.portfolio.open_positions.values():
            weight = position.market_value / self.portfolio.total_value
            concentration += weight ** 2
        
        return concentration
    
    async def _calculate_portfolio_leverage(self) -> Decimal:
        """Calculate overall portfolio leverage."""
        total_notional = Decimal("0")
        total_equity = self.portfolio.total_value
        
        for position in self.portfolio.open_positions.values():
            notional_value = position.market_value * position.leverage
            total_notional += notional_value
        
        return total_notional / total_equity if total_equity > 0 else Decimal("1")
    
    async def _run_stress_scenario(self, scenario: StressTestScenario) -> StressTestScenarioResult:
        """Run individual stress test scenario."""
        portfolio_pnl = Decimal("0")
        positions_affected = 0
        worst_position_loss = Decimal("0")
        
        for position in self.portfolio.open_positions.values():
            # Apply market shock
            shocked_price = position.current_price * (Decimal("1") + scenario.market_shock_pct)
            
            # Calculate P&L impact
            if position.side == "LONG" or position.position_type == PositionType.SPOT:
                position_pnl = (shocked_price - position.entry_price) * position.size
            else:  # SHORT
                position_pnl = (position.entry_price - shocked_price) * position.size
            
            # Apply leverage
            if position.leverage > 1:
                position_pnl *= position.leverage
            
            portfolio_pnl += position_pnl
            positions_affected += 1
            
            if position_pnl < worst_position_loss:
                worst_position_loss = position_pnl
        
        portfolio_pnl_pct = portfolio_pnl / self.portfolio.total_value if self.portfolio.total_value > 0 else Decimal("0")
        
        return StressTestScenarioResult(
            scenario_name=scenario.scenario_name,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            positions_affected=positions_affected,
            worst_position_loss=worst_position_loss
        )
    
    async def _calculate_pairwise_correlation(self, symbol1: str, symbol2: str) -> Optional[Decimal]:
        """Calculate correlation between two symbols."""
        # Get price history for both symbols
        history1 = self._price_history.get(symbol1, [])
        history2 = self._price_history.get(symbol2, [])
        
        if len(history1) < self.config.min_observations or len(history2) < self.config.min_observations:
            return None
        
        # Calculate returns
        returns1 = []
        returns2 = []
        
        min_length = min(len(history1), len(history2))
        for i in range(1, min_length):
            ret1 = (history1[i]["price"] - history1[i-1]["price"]) / history1[i-1]["price"]
            ret2 = (history2[i]["price"] - history2[i-1]["price"]) / history2[i-1]["price"]
            returns1.append(ret1)
            returns2.append(ret2)
        
        if len(returns1) < 2:
            return None
        
        # Calculate correlation coefficient
        try:
            correlation = Decimal(str(statistics.correlation([float(r) for r in returns1], [float(r) for r in returns2])))
            return correlation
        except:
            return None
    
    async def _get_asset_volatility(self, symbol: str) -> Optional[Decimal]:
        """Get asset volatility from price history."""
        history = self._price_history.get(symbol, [])
        if len(history) < self.config.min_observations:
            return None
        
        # Calculate returns
        returns = []
        for i in range(1, len(history)):
            ret = (history[i]["price"] - history[i-1]["price"]) / history[i-1]["price"]
            returns.append(ret)
        
        if len(returns) < 2:
            return None
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        volatility = Decimal(str(math.sqrt(float(variance))))
        
        return volatility
    
    async def _get_expected_return(self, symbol: str) -> Optional[Decimal]:
        """Get expected return for asset."""
        history = self._price_history.get(symbol, [])
        if len(history) < self.config.min_observations:
            return Decimal("0.1")  # Default 10% expected return
        
        # Calculate average return
        returns = []
        for i in range(1, len(history)):
            ret = (history[i]["price"] - history[i-1]["price"]) / history[i-1]["price"]
            returns.append(ret)
        
        if returns:
            # Annualize daily returns
            avg_daily_return = sum(returns) / len(returns)
            annual_return = avg_daily_return * 365
            return annual_return
        
        return Decimal("0.1")
    
    async def _get_current_portfolio_risk(self) -> Decimal:
        """Get current portfolio risk level."""
        return await self._calculate_total_portfolio_risk()
    
    async def _get_correlation_adjustment(self, symbol: str) -> Optional[Decimal]:
        """Get correlation adjustment factor for position sizing."""
        # Check correlation with existing positions
        max_correlation = Decimal("0")
        
        for position in self.portfolio.open_positions.values():
            if position.symbol == symbol:
                continue
            
            correlation = self._correlation_matrix.get((symbol, position.symbol))
            if correlation is None:
                correlation = self._correlation_matrix.get((position.symbol, symbol))
            
            if correlation is not None and abs(correlation) > abs(max_correlation):
                max_correlation = correlation
        
        # If high correlation, reduce position size
        if abs(max_correlation) > self.config.max_correlation_threshold:
            # Reduce size by correlation strength
            reduction_factor = Decimal("1") - (abs(max_correlation) - self.config.max_correlation_threshold)
            return max(Decimal("0.5"), reduction_factor)  # Minimum 50% size
        
        return None
    
    async def _calculate_win_probability(self, symbol: str) -> Optional[Decimal]:
        """Calculate historical win probability for symbol."""
        # Mock implementation - would analyze historical trades
        return Decimal("0.6")  # 60% win rate
    
    async def _calculate_average_win_loss_ratio(self, symbol: str) -> Optional[Decimal]:
        """Calculate average win/loss ratio for symbol."""
        # Mock implementation - would analyze historical trades
        return Decimal("1.8")  # 1.8:1 win/loss ratio
    
    async def _get_daily_loss(self) -> Decimal:
        """Get current daily loss."""
        # Check if we have daily P&L history
        today = datetime.now().date()
        today_pnl = next(
            (entry["pnl"] for entry in self._daily_pnl_history if entry["date"] == today),
            Decimal("0")
        )
        
        # Return absolute loss value
        return abs(min(today_pnl, Decimal("0")))
    
    async def _trigger_emergency_stop(self) -> None:
        """Trigger emergency stop procedures."""
        self._emergency_stop_active = True
        
        # Log emergency stop
        self.logger.critical("Emergency stop triggered - stopping all trading")
        
        # Close all positions (mock implementation)
        await self._close_all_positions()
    
    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        # Mock implementation - would execute actual trades
        for position in list(self.portfolio.open_positions.values()):
            position.status = PositionStatus.CLOSED
            self.logger.info(f"Emergency closed position {position.symbol}")
    
    async def _reduce_position_size(self, position_id: UUID, reduction_factor: Decimal) -> None:
        """Reduce position size by factor."""
        position = self.portfolio.positions.get(position_id)
        if position:
            position.size *= reduction_factor
            self.logger.info(f"Reduced position {position.symbol} by {reduction_factor}")
    
    async def _reduce_leverage(self, position_id: UUID) -> None:
        """Reduce position leverage."""
        position = self.portfolio.positions.get(position_id)
        if position and position.leverage > 1:
            position.leverage = min(position.leverage * Decimal("0.8"), self.config.max_leverage)
            self.logger.info(f"Reduced leverage for position {position.symbol} to {position.leverage}")
    
    async def _execute_stop_loss(self, position_id: UUID) -> None:
        """Execute stop loss for position."""
        position = self.portfolio.positions.get(position_id)
        if position:
            position.status = PositionStatus.CLOSED
            self.logger.info(f"Executed stop loss for position {position.symbol}")
    
    async def _execute_risk_action(self, action: str, position_id: Optional[UUID] = None) -> None:
        """Execute risk management action."""
        if action == "reduce_position_size" and position_id:
            await self._reduce_position_size(position_id, Decimal("0.8"))
        elif action == "close_position" and position_id:
            await self._execute_stop_loss(position_id)
        elif action == "emergency_stop":
            await self._trigger_emergency_stop()
    
    def _symbols_are_related(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are related (e.g., same base asset)."""
        # Extract base asset from symbols
        base1 = self._extract_base_asset(symbol1)
        base2 = self._extract_base_asset(symbol2)
        return base1 == base2
    
    def _extract_base_asset(self, symbol: str) -> str:
        """Extract base asset from trading symbol."""
        # Handle common formats: BTC/USDC, BTC-USD, BTC-PERP
        if "/" in symbol:
            return symbol.split("/")[0]
        elif "-" in symbol:
            parts = symbol.split("-")
            return parts[0]
        else:
            return symbol
    
    async def _calculate_cross_chain_correlation(
        self,
        symbol1: str,
        chain1: Chain,
        symbol2: str,
        chain2: Chain
    ) -> Optional[Decimal]:
        """Calculate correlation between symbols on different chains."""
        # Mock implementation - would fetch cross-chain price data
        if self._symbols_are_related(symbol1, symbol2):
            return Decimal("0.9")  # High correlation for same asset on different chains
        return Decimal("0.3")  # Lower correlation for different assets
    
    async def _calculate_chain_correlation(self, chain1: Chain, chain2: Chain) -> Optional[Decimal]:
        """Calculate overall correlation between two chains."""
        # Mock implementation - would analyze chain-level performance
        return Decimal("0.7")  # Default chain correlation
    
    async def _calculate_sharpe_ratio(self, returns: List[Decimal], risk_free_rate: Decimal) -> Decimal:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return Decimal("0")
        
        mean_return = sum(returns) / len(returns)
        std_dev = Decimal(str(statistics.stdev([float(r) for r in returns])))
        
        if std_dev == 0:
            return Decimal("0")
        
        excess_return = mean_return - risk_free_rate / 365  # Daily risk-free rate
        sharpe_ratio = excess_return / std_dev
        
        return sharpe_ratio * Decimal(str(math.sqrt(365)))  # Annualized
    
    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for symbol."""
        # Mock implementation - would fetch from external APIs
        return {
            "price": Decimal("50000"),
            "volume": Decimal("1000000"),
            "volatility": Decimal("0.3")
        }