"""
P&L Calculator for Portfolio Management

This module provides comprehensive P&L calculation functionality
for multi-chain, multi-DEX trading across different position types.
It handles real-time and historical P&L calculations with caching
for optimal performance.

Key Features:
- Real-time P&L calculations for individual positions and portfolios
- Historical P&L calculations for closed positions
- Support for spot and perpetual futures positions
- Multi-chain and multi-DEX calculations (Jupiter, Hyperliquid, Uniswap V3)
- Aggregated P&L views by time period, chain, and DEX
- Funding payments for perpetual positions
- Transaction fees and costs analysis
- P&L attribution analysis (price vs funding vs fees)
- Performance optimization with intelligent caching
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import structlog

from .base import (
    Portfolio,
    Position,
    Transaction,
    PositionType,
    PositionStatus,
    TransactionType,
    PortfolioError,
)
from src.utils.base import Chain


logger = structlog.get_logger()


class PnLAggregationPeriod(Enum):
    """Time period for P&L aggregation."""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class PnLAttribution:
    """P&L attribution analysis breaking down sources of P&L."""
    price_pnl: Decimal                          # P&L from price movements
    funding_pnl: Decimal                        # P&L from funding payments (perpetuals)
    fees_paid: Decimal                          # Total fees paid
    slippage_cost: Decimal = Decimal("0")       # Cost due to slippage
    borrowing_cost: Decimal = Decimal("0")      # Borrowing costs (margin positions)
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after all costs."""
        return self.price_pnl + self.funding_pnl - self.fees_paid - self.slippage_cost - self.borrowing_cost
    
    @property
    def attribution_breakdown(self) -> Dict[str, Decimal]:
        """Get attribution breakdown as dictionary."""
        return {
            "price_pnl": self.price_pnl,
            "funding_pnl": self.funding_pnl,
            "fees_paid": self.fees_paid,
            "slippage_cost": self.slippage_cost,
            "borrowing_cost": self.borrowing_cost,
            "net_pnl": self.net_pnl
        }


@dataclass
class PnLTimeSeriesPoint:
    """Single point in P&L time series."""
    timestamp: datetime                         # Time point
    total_pnl: Decimal                         # Total P&L at this time
    unrealized_pnl: Decimal                    # Unrealized P&L
    realized_pnl: Decimal                      # Realized P&L
    position_count: int                        # Number of positions
    total_fees: Decimal = Decimal("0")         # Total fees at this time
    funding_pnl: Decimal = Decimal("0")        # Funding P&L
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after fees."""
        return self.total_pnl - self.total_fees


@dataclass
class PnLCalculationResult:
    """Result of P&L calculation for individual position."""
    success: bool
    position_id: UUID
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    
    # Position details
    position_type: Optional[PositionType] = None
    position_status: Optional[PositionStatus] = None
    chain: Optional[Chain] = None
    dex_name: Optional[str] = None
    symbol: Optional[str] = None
    
    # Additional P&L components
    funding_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    leverage: Optional[Decimal] = None
    
    # Calculation metadata
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None
    cached: bool = False
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after fees."""
        return self.total_pnl - self.fees_paid


@dataclass
class PnLSummary:
    """Summary P&L information for multiple positions."""
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal
    total_pnl: Decimal
    total_positions: int
    total_fees: Decimal = Decimal("0")
    total_funding: Decimal = Decimal("0")
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after fees."""
        return self.total_pnl - self.total_fees


@dataclass
class PortfolioPnLResult:
    """Result of portfolio-wide P&L calculation."""
    success: bool
    portfolio_id: UUID
    total_unrealized_pnl: Decimal = Decimal("0")
    total_realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    total_positions: int = 0
    position_pnls: List[PnLCalculationResult] = field(default_factory=list)
    summary: Optional[PnLSummary] = None
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ChainPnLSummary:
    """P&L summary for a specific blockchain chain."""
    chain: Chain
    total_pnl: Decimal
    position_count: int
    total_fees: Decimal = Decimal("0")
    dex_breakdown: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class ChainPnLResult:
    """Result of chain-aggregated P&L calculation."""
    success: bool
    total_pnl: Decimal = Decimal("0")
    chain_pnls: Dict[Chain, ChainPnLSummary] = field(default_factory=dict)
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class DexPnLSummary:
    """P&L summary for a specific DEX."""
    dex_name: str
    total_pnl: Decimal
    position_count: int
    total_fees: Decimal = Decimal("0")
    chain_breakdown: Dict[Chain, Decimal] = field(default_factory=dict)


@dataclass
class DexPnLResult:
    """Result of DEX-aggregated P&L calculation."""
    success: bool
    total_pnl: Decimal = Decimal("0")
    dex_pnls: Dict[str, DexPnLSummary] = field(default_factory=dict)
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class TimeSeriesPnLResult:
    """Result of time series P&L calculation."""
    success: bool
    start_date: datetime
    end_date: datetime
    period: PnLAggregationPeriod
    time_series: List[PnLTimeSeriesPoint] = field(default_factory=list)
    summary: Optional[PnLSummary] = None
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class HistoricalPnLResult:
    """Result of historical P&L calculation."""
    success: bool
    position_id: UUID
    start_date: datetime
    end_date: datetime
    realized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    transaction_count: int = 0
    net_pnl: Decimal = Decimal("0")
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class FundingPnLResult:
    """Result of funding P&L calculation."""
    success: bool
    position_id: UUID
    total_funding: Decimal = Decimal("0")
    funding_count: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class FeesResult:
    """Result of fees and costs calculation."""
    success: bool
    position_id: Optional[UUID] = None
    total_fees: Decimal = Decimal("0")
    trading_fees: Decimal = Decimal("0")
    gas_fees: Decimal = Decimal("0")
    transaction_count: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class DexFeesResult:
    """Result of DEX-aggregated fees calculation."""
    success: bool
    dex_fees: Dict[str, FeesResult] = field(default_factory=dict)
    total_fees: Decimal = Decimal("0")
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class PnLAttributionResult:
    """Result of P&L attribution analysis."""
    success: bool
    position_id: UUID
    attribution: PnLAttribution
    total_pnl: Decimal = Decimal("0")
    calculation_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry for P&L calculations."""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)


class PnLCalculationError(PortfolioError):
    """Error in P&L calculation."""
    
    def __init__(self, message: str, position_id: Optional[UUID] = None):
        super().__init__(message)
        self.position_id = position_id


class PnLCalculator:
    """
    P&L calculator for comprehensive profit and loss analysis.
    
    This class provides methods for calculating P&L across different
    dimensions: individual positions, portfolios, chains, DEXs, and
    time periods. It includes intelligent caching for performance
    optimization and supports both real-time and historical calculations.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        cache_ttl_seconds: int = 300,  # 5 minutes default
        enable_attribution: bool = True,
        enable_time_series: bool = True
    ):
        """
        Initialize P&L calculator.
        
        Args:
            portfolio: Portfolio to calculate P&L for
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes)
            enable_attribution: Enable P&L attribution analysis
            enable_time_series: Enable time series calculations
        """
        self.portfolio = portfolio
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_attribution = enable_attribution
        self.enable_time_series = enable_time_series
        
        # Internal cache for performance optimization
        self._cache: Dict[str, CacheEntry] = {}
        
        self.logger = logger.bind(
            component="pnl_calculator",
            portfolio_id=str(portfolio.portfolio_id)
        )
    
    def clear_cache(self) -> None:
        """Clear all cached P&L calculations."""
        self._cache.clear()
        self.logger.debug("P&L calculation cache cleared")
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        if cache_key not in self._cache:
            return None
        
        entry = self._cache[cache_key]
        if entry.is_expired:
            del self._cache[cache_key]
            return None
        
        return entry.data
    
    def _cache_result(self, cache_key: str, data: Any) -> None:
        """Cache calculation result."""
        self._cache[cache_key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=self.cache_ttl_seconds
        )
    
    async def calculate_position_pnl(self, position_id: UUID) -> PnLCalculationResult:
        """
        Calculate P&L for an individual position.
        
        Args:
            position_id: Position identifier
            
        Returns:
            PnLCalculationResult with detailed P&L information
        """
        try:
            # Check cache first
            cache_key = f"position_pnl_{position_id}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                cached_result.cached = True
                return cached_result
            
            # Get position
            position = self.portfolio.positions.get(position_id)
            if position is None:
                return PnLCalculationResult(
                    success=False,
                    position_id=position_id,
                    error_message="Position not found",
                    calculation_time=datetime.now()
                )
            
            # Calculate unrealized P&L
            unrealized_pnl = position.unrealized_pnl
            
            # Calculate realized P&L from transactions
            realized_pnl = await self._calculate_realized_pnl(position_id)
            
            # Calculate total fees
            fees_paid = await self._calculate_position_fees(position_id)
            
            # Extract funding P&L for perpetuals
            funding_pnl = position.unrealized_funding or Decimal("0")
            
            total_pnl = unrealized_pnl + realized_pnl
            
            result = PnLCalculationResult(
                success=True,
                position_id=position_id,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                position_type=position.position_type,
                position_status=position.status,
                chain=position.chain,
                dex_name=position.dex_name,
                symbol=position.symbol,
                funding_pnl=funding_pnl,
                fees_paid=fees_paid,
                leverage=position.leverage,
                calculation_time=datetime.now(),
                cached=False
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            self.logger.debug(
                "Position P&L calculated",
                position_id=str(position_id),
                symbol=position.symbol,
                total_pnl=str(total_pnl),
                unrealized=str(unrealized_pnl),
                realized=str(realized_pnl)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate position P&L",
                position_id=str(position_id),
                error=str(e)
            )
            return PnLCalculationResult(
                success=False,
                position_id=position_id,
                error_message=f"Calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_portfolio_pnl(self) -> PortfolioPnLResult:
        """
        Calculate P&L for entire portfolio.
        
        Returns:
            PortfolioPnLResult with portfolio-wide P&L information
        """
        try:
            # Check cache
            cache_key = "portfolio_pnl"
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
            
            position_pnls = []
            total_unrealized = Decimal("0")
            total_realized = Decimal("0")
            
            # Calculate P&L for each position
            for position_id in self.portfolio.positions.keys():
                position_result = await self.calculate_position_pnl(position_id)
                if position_result.success:
                    position_pnls.append(position_result)
                    total_unrealized += position_result.unrealized_pnl
                    total_realized += position_result.realized_pnl
            
            total_pnl = total_unrealized + total_realized
            
            summary = PnLSummary(
                total_unrealized_pnl=total_unrealized,
                total_realized_pnl=total_realized,
                total_pnl=total_pnl,
                total_positions=len(position_pnls),
                total_fees=sum(p.fees_paid for p in position_pnls),
                total_funding=sum(p.funding_pnl for p in position_pnls)
            )
            
            result = PortfolioPnLResult(
                success=True,
                portfolio_id=self.portfolio.portfolio_id,
                total_unrealized_pnl=total_unrealized,
                total_realized_pnl=total_realized,
                total_pnl=total_pnl,
                total_positions=len(position_pnls),
                position_pnls=position_pnls,
                summary=summary,
                calculation_time=datetime.now()
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            self.logger.info(
                "Portfolio P&L calculated",
                total_positions=len(position_pnls),
                total_pnl=str(total_pnl),
                unrealized=str(total_unrealized),
                realized=str(total_realized)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate portfolio P&L",
                error=str(e)
            )
            return PortfolioPnLResult(
                success=False,
                portfolio_id=self.portfolio.portfolio_id,
                error_message=f"Portfolio P&L calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_historical_pnl(
        self,
        position_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> HistoricalPnLResult:
        """
        Calculate historical P&L for closed positions.
        
        Args:
            position_id: Position identifier
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            HistoricalPnLResult with historical P&L data
        """
        try:
            # Validate date range
            if end_date <= start_date:
                return HistoricalPnLResult(
                    success=False,
                    position_id=position_id,
                    start_date=start_date,
                    end_date=end_date,
                    error_message="Invalid date range: end_date must be after start_date",
                    calculation_time=datetime.now()
                )
            
            # Get position
            position = self.portfolio.positions.get(position_id)
            if position is None:
                return HistoricalPnLResult(
                    success=False,
                    position_id=position_id,
                    start_date=start_date,
                    end_date=end_date,
                    error_message="Position not found",
                    calculation_time=datetime.now()
                )
            
            # Get transactions in date range
            position_transactions = [
                tx for tx in self.portfolio.transactions
                if (tx.position_id == position_id and
                    start_date <= tx.timestamp <= end_date)
            ]
            
            # Calculate realized P&L from transactions
            realized_pnl = Decimal("0")
            total_fees = Decimal("0")
            
            for tx in position_transactions:
                if tx.transaction_type == TransactionType.SELL:
                    # Calculate realized P&L (simplified - assumes FIFO)
                    price_diff = tx.price - position.entry_price
                    if position.side == "SHORT":
                        price_diff = -price_diff
                    
                    pnl = price_diff * tx.size
                    if position.leverage > 1:
                        pnl *= position.leverage
                    
                    realized_pnl += pnl
                
                # Add fees
                total_fees += tx.total_fees
            
            net_pnl = realized_pnl - total_fees
            
            result = HistoricalPnLResult(
                success=True,
                position_id=position_id,
                start_date=start_date,
                end_date=end_date,
                realized_pnl=realized_pnl,
                total_fees=total_fees,
                transaction_count=len(position_transactions),
                net_pnl=net_pnl,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "Historical P&L calculated",
                position_id=str(position_id),
                realized_pnl=str(realized_pnl),
                total_fees=str(total_fees),
                transaction_count=len(position_transactions)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate historical P&L",
                position_id=str(position_id),
                error=str(e)
            )
            return HistoricalPnLResult(
                success=False,
                position_id=position_id,
                start_date=start_date,
                end_date=end_date,
                error_message=f"Historical P&L calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_pnl_by_chain(self) -> ChainPnLResult:
        """
        Calculate P&L aggregated by blockchain chain.
        
        Returns:
            ChainPnLResult with chain-aggregated P&L data
        """
        try:
            chain_pnls = {}
            total_pnl = Decimal("0")
            
            # Group positions by chain
            chain_positions = {}
            for position in self.portfolio.positions.values():
                if position.chain not in chain_positions:
                    chain_positions[position.chain] = []
                chain_positions[position.chain].append(position)
            
            # Calculate P&L for each chain
            for chain, positions in chain_positions.items():
                chain_total_pnl = Decimal("0")
                chain_total_fees = Decimal("0")
                dex_breakdown = {}
                
                for position in positions:
                    position_result = await self.calculate_position_pnl(position.position_id)
                    if position_result.success:
                        chain_total_pnl += position_result.total_pnl
                        chain_total_fees += position_result.fees_paid
                        
                        # DEX breakdown
                        if position.dex_name not in dex_breakdown:
                            dex_breakdown[position.dex_name] = Decimal("0")
                        dex_breakdown[position.dex_name] += position_result.total_pnl
                
                chain_summary = ChainPnLSummary(
                    chain=chain,
                    total_pnl=chain_total_pnl,
                    position_count=len(positions),
                    total_fees=chain_total_fees,
                    dex_breakdown=dex_breakdown
                )
                
                chain_pnls[chain] = chain_summary
                total_pnl += chain_total_pnl
            
            result = ChainPnLResult(
                success=True,
                total_pnl=total_pnl,
                chain_pnls=chain_pnls,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "Chain P&L calculated",
                total_chains=len(chain_pnls),
                total_pnl=str(total_pnl)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate chain P&L",
                error=str(e)
            )
            return ChainPnLResult(
                success=False,
                error_message=f"Chain P&L calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_pnl_by_dex(self) -> DexPnLResult:
        """
        Calculate P&L aggregated by DEX.
        
        Returns:
            DexPnLResult with DEX-aggregated P&L data
        """
        try:
            dex_pnls = {}
            total_pnl = Decimal("0")
            
            # Group positions by DEX
            dex_positions = {}
            for position in self.portfolio.positions.values():
                if position.dex_name not in dex_positions:
                    dex_positions[position.dex_name] = []
                dex_positions[position.dex_name].append(position)
            
            # Calculate P&L for each DEX
            for dex_name, positions in dex_positions.items():
                dex_total_pnl = Decimal("0")
                dex_total_fees = Decimal("0")
                chain_breakdown = {}
                
                for position in positions:
                    position_result = await self.calculate_position_pnl(position.position_id)
                    if position_result.success:
                        dex_total_pnl += position_result.total_pnl
                        dex_total_fees += position_result.fees_paid
                        
                        # Chain breakdown
                        if position.chain not in chain_breakdown:
                            chain_breakdown[position.chain] = Decimal("0")
                        chain_breakdown[position.chain] += position_result.total_pnl
                
                dex_summary = DexPnLSummary(
                    dex_name=dex_name,
                    total_pnl=dex_total_pnl,
                    position_count=len(positions),
                    total_fees=dex_total_fees,
                    chain_breakdown=chain_breakdown
                )
                
                dex_pnls[dex_name] = dex_summary
                total_pnl += dex_total_pnl
            
            result = DexPnLResult(
                success=True,
                total_pnl=total_pnl,
                dex_pnls=dex_pnls,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "DEX P&L calculated",
                total_dexs=len(dex_pnls),
                total_pnl=str(total_pnl)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate DEX P&L",
                error=str(e)
            )
            return DexPnLResult(
                success=False,
                error_message=f"DEX P&L calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_pnl_time_series(
        self,
        start_date: datetime,
        end_date: datetime,
        period: PnLAggregationPeriod
    ) -> TimeSeriesPnLResult:
        """
        Calculate P&L time series for specified period.
        
        Args:
            start_date: Start date for time series
            end_date: End date for time series
            period: Aggregation period
            
        Returns:
            TimeSeriesPnLResult with time series data
        """
        try:
            if not self.enable_time_series:
                return TimeSeriesPnLResult(
                    success=False,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    error_message="Time series calculations are disabled",
                    calculation_time=datetime.now()
                )
            
            # Calculate current portfolio P&L
            portfolio_result = await self.calculate_portfolio_pnl()
            if not portfolio_result.success:
                return TimeSeriesPnLResult(
                    success=False,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    error_message="Failed to calculate portfolio P&L",
                    calculation_time=datetime.now()
                )
            
            # Generate time series points (simplified implementation)
            time_series = []
            current_time = start_date
            
            # Calculate time delta based on period
            if period == PnLAggregationPeriod.MINUTE:
                delta = timedelta(minutes=1)
            elif period == PnLAggregationPeriod.HOURLY:
                delta = timedelta(hours=1)
            elif period == PnLAggregationPeriod.DAILY:
                delta = timedelta(days=1)
            elif period == PnLAggregationPeriod.WEEKLY:
                delta = timedelta(weeks=1)
            elif period == PnLAggregationPeriod.MONTHLY:
                delta = timedelta(days=30)
            else:
                delta = timedelta(hours=1)  # Default to hourly
            
            # Generate time series points
            while current_time <= end_date:
                # For simplicity, use current P&L for all points
                # In production, this would query historical data
                point = PnLTimeSeriesPoint(
                    timestamp=current_time,
                    total_pnl=portfolio_result.total_pnl,
                    unrealized_pnl=portfolio_result.total_unrealized_pnl,
                    realized_pnl=portfolio_result.total_realized_pnl,
                    position_count=portfolio_result.total_positions,
                    total_fees=portfolio_result.summary.total_fees if portfolio_result.summary else Decimal("0"),
                    funding_pnl=portfolio_result.summary.total_funding if portfolio_result.summary else Decimal("0")
                )
                time_series.append(point)
                current_time += delta
            
            # Create summary
            summary = PnLSummary(
                total_unrealized_pnl=portfolio_result.total_unrealized_pnl,
                total_realized_pnl=portfolio_result.total_realized_pnl,
                total_pnl=portfolio_result.total_pnl,
                total_positions=portfolio_result.total_positions,
                total_fees=portfolio_result.summary.total_fees if portfolio_result.summary else Decimal("0"),
                total_funding=portfolio_result.summary.total_funding if portfolio_result.summary else Decimal("0")
            )
            
            result = TimeSeriesPnLResult(
                success=True,
                start_date=start_date,
                end_date=end_date,
                period=period,
                time_series=time_series,
                summary=summary,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "P&L time series calculated",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                period=period.value,
                points=len(time_series)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate P&L time series",
                error=str(e)
            )
            return TimeSeriesPnLResult(
                success=False,
                start_date=start_date,
                end_date=end_date,
                period=period,
                error_message=f"Time series calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_funding_pnl(
        self,
        position_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> FundingPnLResult:
        """
        Calculate funding payments for perpetual positions.
        
        Args:
            position_id: Position identifier
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            FundingPnLResult with funding payment data
        """
        try:
            # Get position
            position = self.portfolio.positions.get(position_id)
            if position is None:
                return FundingPnLResult(
                    success=False,
                    position_id=position_id,
                    error_message="Position not found",
                    calculation_time=datetime.now()
                )
            
            # Only perpetual positions have funding
            if position.position_type != PositionType.PERPETUAL:
                return FundingPnLResult(
                    success=True,
                    position_id=position_id,
                    total_funding=Decimal("0"),
                    funding_count=0,
                    start_date=start_date,
                    end_date=end_date,
                    calculation_time=datetime.now()
                )
            
            # Get funding transactions in date range
            funding_transactions = [
                tx for tx in self.portfolio.transactions
                if (tx.position_id == position_id and
                    tx.transaction_type == TransactionType.FUNDING and
                    start_date <= tx.timestamp <= end_date)
            ]
            
            # Calculate total funding from transactions (funding payments, not fees)
            total_funding = Decimal("0")
            
            # For funding transactions, the actual funding amount should be in metadata
            # or calculated based on funding rate. For simplicity, we'll use the position's
            # unrealized funding which represents the cumulative funding received/paid
            if position.unrealized_funding:
                total_funding = position.unrealized_funding
            
            result = FundingPnLResult(
                success=True,
                position_id=position_id,
                total_funding=total_funding,
                funding_count=len(funding_transactions),
                start_date=start_date,
                end_date=end_date,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "Funding P&L calculated",
                position_id=str(position_id),
                total_funding=str(total_funding),
                funding_count=len(funding_transactions)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate funding P&L",
                position_id=str(position_id),
                error=str(e)
            )
            return FundingPnLResult(
                success=False,
                position_id=position_id,
                error_message=f"Funding P&L calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_fees_and_costs(
        self,
        position_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> FeesResult:
        """
        Calculate transaction fees and costs.
        
        Args:
            position_id: Position identifier
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            FeesResult with fees and costs data
        """
        try:
            # Get position transactions in date range
            position_transactions = [
                tx for tx in self.portfolio.transactions
                if (tx.position_id == position_id and
                    start_date <= tx.timestamp <= end_date)
            ]
            
            total_fees = Decimal("0")
            trading_fees = Decimal("0")
            gas_fees = Decimal("0")
            
            for tx in position_transactions:
                total_fees += tx.total_fees
                trading_fees += tx.fees
                if tx.gas_fees:
                    gas_fees += tx.gas_fees
            
            result = FeesResult(
                success=True,
                position_id=position_id,
                total_fees=total_fees,
                trading_fees=trading_fees,
                gas_fees=gas_fees,
                transaction_count=len(position_transactions),
                start_date=start_date,
                end_date=end_date,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "Fees and costs calculated",
                position_id=str(position_id),
                total_fees=str(total_fees),
                transaction_count=len(position_transactions)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate fees and costs",
                position_id=str(position_id),
                error=str(e)
            )
            return FeesResult(
                success=False,
                position_id=position_id,
                error_message=f"Fees calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_fees_by_dex(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DexFeesResult:
        """
        Calculate fees aggregated by DEX.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            DexFeesResult with DEX-aggregated fees data
        """
        try:
            dex_fees = {}
            total_fees = Decimal("0")
            
            # Group transactions by DEX
            dex_transactions = {}
            for tx in self.portfolio.transactions:
                if start_date <= tx.timestamp <= end_date:
                    if tx.dex_name not in dex_transactions:
                        dex_transactions[tx.dex_name] = []
                    dex_transactions[tx.dex_name].append(tx)
            
            # Calculate fees for each DEX
            for dex_name, transactions in dex_transactions.items():
                dex_total_fees = Decimal("0")
                dex_trading_fees = Decimal("0")
                dex_gas_fees = Decimal("0")
                
                for tx in transactions:
                    dex_total_fees += tx.total_fees
                    dex_trading_fees += tx.fees
                    if tx.gas_fees:
                        dex_gas_fees += tx.gas_fees
                
                dex_result = FeesResult(
                    success=True,
                    total_fees=dex_total_fees,
                    trading_fees=dex_trading_fees,
                    gas_fees=dex_gas_fees,
                    transaction_count=len(transactions),
                    start_date=start_date,
                    end_date=end_date,
                    calculation_time=datetime.now()
                )
                
                dex_fees[dex_name] = dex_result
                total_fees += dex_total_fees
            
            result = DexFeesResult(
                success=True,
                dex_fees=dex_fees,
                total_fees=total_fees,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "DEX fees calculated",
                total_dexs=len(dex_fees),
                total_fees=str(total_fees)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate DEX fees",
                error=str(e)
            )
            return DexFeesResult(
                success=False,
                error_message=f"DEX fees calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    async def calculate_pnl_attribution(self, position_id: UUID) -> PnLAttributionResult:
        """
        Calculate P&L attribution analysis (price vs funding vs fees).
        
        Args:
            position_id: Position identifier
            
        Returns:
            PnLAttributionResult with attribution breakdown
        """
        try:
            if not self.enable_attribution:
                return PnLAttributionResult(
                    success=False,
                    position_id=position_id,
                    attribution=PnLAttribution(
                        price_pnl=Decimal("0"),
                        funding_pnl=Decimal("0"),
                        fees_paid=Decimal("0")
                    ),
                    error_message="Attribution analysis is disabled",
                    calculation_time=datetime.now()
                )
            
            # Get position
            position = self.portfolio.positions.get(position_id)
            if position is None:
                return PnLAttributionResult(
                    success=False,
                    position_id=position_id,
                    attribution=PnLAttribution(
                        price_pnl=Decimal("0"),
                        funding_pnl=Decimal("0"),
                        fees_paid=Decimal("0")
                    ),
                    error_message="Position not found",
                    calculation_time=datetime.now()
                )
            
            # Calculate price P&L (excluding funding)
            price_diff = position.current_price - position.entry_price
            
            if position.position_type == PositionType.SPOT or position.side == "LONG":
                price_pnl = price_diff * position.size
            elif position.side == "SHORT":
                price_pnl = -price_diff * position.size
            else:
                price_pnl = price_diff * position.size
            
            # Apply leverage
            if position.leverage > 1:
                price_pnl *= position.leverage
            
            # Get funding P&L
            funding_pnl = position.unrealized_funding or Decimal("0")
            
            # Calculate fees paid
            fees_paid = await self._calculate_position_fees(position_id)
            
            # Create attribution
            attribution = PnLAttribution(
                price_pnl=price_pnl,
                funding_pnl=funding_pnl,
                fees_paid=fees_paid
            )
            
            # Total P&L before fees (matches position.unrealized_pnl)
            total_pnl = price_pnl + funding_pnl
            
            result = PnLAttributionResult(
                success=True,
                position_id=position_id,
                attribution=attribution,
                total_pnl=total_pnl,
                calculation_time=datetime.now()
            )
            
            self.logger.debug(
                "P&L attribution calculated",
                position_id=str(position_id),
                price_pnl=str(price_pnl),
                funding_pnl=str(funding_pnl),
                fees_paid=str(fees_paid)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate P&L attribution",
                position_id=str(position_id),
                error=str(e)
            )
            return PnLAttributionResult(
                success=False,
                position_id=position_id,
                attribution=PnLAttribution(
                    price_pnl=Decimal("0"),
                    funding_pnl=Decimal("0"),
                    fees_paid=Decimal("0")
                ),
                error_message=f"Attribution calculation failed: {e}",
                calculation_time=datetime.now()
            )
    
    # Private helper methods
    
    async def _calculate_realized_pnl(self, position_id: UUID) -> Decimal:
        """Calculate realized P&L from transactions."""
        position = self.portfolio.positions.get(position_id)
        if position is None:
            return Decimal("0")
        
        realized_pnl = Decimal("0")
        
        # Get sell transactions for this position
        sell_transactions = [
            tx for tx in self.portfolio.transactions
            if (tx.position_id == position_id and
                tx.transaction_type == TransactionType.SELL)
        ]
        
        for tx in sell_transactions:
            # Calculate P&L for this sale (simplified FIFO)
            price_diff = tx.price - position.entry_price
            
            if position.side == "SHORT":
                price_diff = -price_diff
            
            pnl = price_diff * tx.size
            
            if position.leverage > 1:
                pnl *= position.leverage
            
            realized_pnl += pnl
        
        return realized_pnl
    
    async def _calculate_position_fees(self, position_id: UUID) -> Decimal:
        """Calculate total fees for a position."""
        total_fees = Decimal("0")
        
        position_transactions = [
            tx for tx in self.portfolio.transactions
            if tx.position_id == position_id
        ]
        
        for tx in position_transactions:
            total_fees += tx.total_fees
        
        return total_fees