"""
Portfolio base data structures and enums.

This module defines the core data structures for portfolio management,
including Portfolio, Position, Transaction, and performance metrics.
Designed for multi-chain, multi-DEX trading across Jupiter (Solana),
Hyperliquid (perpetuals), and Uniswap V3 (Ethereum/Base).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import structlog

from src.utils.base import Chain


logger = structlog.get_logger()


class PositionType(Enum):
    """Type of trading position."""
    SPOT = "spot"                    # Spot trading (Jupiter, Uniswap V3)
    PERPETUAL = "perpetual"          # Perpetual futures (Hyperliquid)
    FUTURES = "futures"              # Dated futures contracts
    OPTION = "option"                # Options contracts


class PositionStatus(Enum):
    """Status of a trading position."""
    OPEN = "open"                    # Position is currently open
    CLOSED = "closed"                # Position fully closed
    PARTIAL = "partial"              # Position partially closed
    LIQUIDATED = "liquidated"        # Position was liquidated


class TransactionType(Enum):
    """Type of portfolio transaction."""
    BUY = "buy"                      # Buy/long transaction
    SELL = "sell"                    # Sell/short transaction
    DEPOSIT = "deposit"              # Cash deposit
    WITHDRAWAL = "withdrawal"        # Cash withdrawal
    FEE = "fee"                      # Trading fee
    FUNDING = "funding"              # Funding payment (perpetuals)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    initial_balance: Decimal                         # Starting balance
    base_currency: str                               # Base currency (USDC, USDT, etc.)
    max_position_size_pct: Decimal = Decimal("0.1") # 10% max position size
    max_daily_loss_pct: Decimal = Decimal("0.05")   # 5% max daily loss
    max_drawdown_pct: Decimal = Decimal("0.15")     # 15% max drawdown
    stop_loss_pct: Decimal = Decimal("0.08")        # 8% stop loss
    take_profit_pct: Decimal = Decimal("0.4")       # 40% take profit
    max_open_positions: int = 10                     # Max concurrent positions
    min_trade_amount_usd: Decimal = Decimal("10")   # Min trade size
    enable_risk_management: bool = True              # Enable risk controls
    enable_auto_rebalancing: bool = False           # Enable auto rebalancing
    rebalance_threshold_pct: Decimal = Decimal("0.05")  # 5% rebalance threshold
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (0 <= self.max_position_size_pct <= 1):
            raise ValueError("max_position_size_pct must be between 0 and 1")
        if not (0 <= self.max_daily_loss_pct <= 1):
            raise ValueError("max_daily_loss_pct must be between 0 and 1")
        if not (0 <= self.max_drawdown_pct <= 1):
            raise ValueError("max_drawdown_pct must be between 0 and 1")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.min_trade_amount_usd <= 0:
            raise ValueError("min_trade_amount_usd must be positive")


@dataclass
class Position:
    """
    Individual trading position across different DEXs and chains.
    
    Supports both spot positions (Jupiter, Uniswap V3) and perpetual
    futures positions (Hyperliquid).
    """
    position_id: UUID                                # Unique position identifier
    symbol: str                                      # Trading pair (BTC/USDC, ETH-USD)
    position_type: PositionType                      # Spot, perpetual, etc.
    chain: Chain                                     # Blockchain network
    dex_name: str                                    # DEX name (jupiter, hyperliquid, etc.)
    size: Decimal                                    # Position size in base asset
    entry_price: Decimal                             # Average entry price
    current_price: Decimal                           # Current market price
    status: PositionStatus                           # Position status
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Optional fields for different position types
    leverage: Decimal = Decimal("1")                 # Leverage (1 = no leverage)
    side: Optional[str] = None                       # LONG/SHORT (for perpetuals)
    margin_used: Optional[Decimal] = None            # Margin used (for leveraged positions)
    liquidation_price: Optional[Decimal] = None     # Liquidation price (for perpetuals)
    funding_rate: Optional[Decimal] = None          # Current funding rate (for perpetuals)
    unrealized_funding: Optional[Decimal] = None    # Unrealized funding payments
    stop_loss_price: Optional[Decimal] = None       # Stop loss price
    take_profit_price: Optional[Decimal] = None     # Take profit price
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L for the position."""
        if self.status == PositionStatus.CLOSED:
            return Decimal("0")
        
        price_diff = self.current_price - self.entry_price
        
        # For spot positions or long perpetuals
        if self.position_type == PositionType.SPOT or self.side == "LONG":
            pnl = price_diff * self.size
        # For short perpetuals
        elif self.side == "SHORT":
            pnl = -price_diff * self.size
        else:
            pnl = price_diff * self.size  # Default to long calculation
        
        # Apply leverage for leveraged positions
        if self.leverage > 1:
            pnl *= self.leverage
        
        # Add unrealized funding for perpetuals
        if self.unrealized_funding is not None:
            pnl += self.unrealized_funding
        
        return pnl
    
    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """Calculate unrealized P&L as percentage of entry value."""
        if self.entry_price == 0:
            return Decimal("0")
        
        price_diff = self.current_price - self.entry_price
        pnl_pct = price_diff / self.entry_price
        
        # Adjust for short positions
        if self.side == "SHORT":
            pnl_pct = -pnl_pct
        
        # Apply leverage
        if self.leverage > 1:
            pnl_pct *= self.leverage
        
        return pnl_pct
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position."""
        return self.current_price * self.size
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis (entry value) of the position."""
        return self.entry_price * self.size
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0
    
    @property
    def margin_ratio(self) -> Optional[Decimal]:
        """Calculate margin ratio for leveraged positions."""
        if self.margin_used is None or self.margin_used == 0:
            return None
        return self.market_value / self.margin_used
    
    def update_price(self, new_price: Decimal) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.updated_at = datetime.now()
    
    def update_funding(self, funding_payment: Decimal) -> None:
        """Update unrealized funding for perpetual positions."""
        if self.unrealized_funding is None:
            self.unrealized_funding = funding_payment
        else:
            self.unrealized_funding += funding_payment
        self.updated_at = datetime.now()


@dataclass
class Transaction:
    """
    Individual transaction record across all DEXs.
    
    Tracks all portfolio transactions including trades, deposits,
    withdrawals, and fees across different chains and DEXs.
    """
    transaction_id: UUID                             # Unique transaction identifier
    position_id: Optional[UUID]                     # Associated position (if applicable)
    transaction_type: TransactionType               # Type of transaction
    symbol: str                                     # Trading pair or asset
    size: Decimal                                   # Transaction size
    price: Decimal                                  # Execution price
    timestamp: datetime                             # Transaction timestamp
    chain: Chain                                    # Blockchain network
    dex_name: str                                   # DEX name
    transaction_hash: str                           # Blockchain transaction hash
    fees: Decimal = Decimal("0")                    # Transaction fees
    gas_fees: Optional[Decimal] = None              # Gas fees (Ethereum/Base)
    slippage_bps: Optional[int] = None             # Actual slippage in basis points
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def value(self) -> Decimal:
        """Calculate gross transaction value."""
        return self.size * self.price
    
    @property
    def net_value(self) -> Decimal:
        """Calculate net transaction value after fees."""
        total_fees = self.fees
        if self.gas_fees is not None:
            total_fees += self.gas_fees
        return self.value - total_fees
    
    @property
    def total_fees(self) -> Decimal:
        """Calculate total fees including gas."""
        total = self.fees
        if self.gas_fees is not None:
            total += self.gas_fees
        return total


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics and statistics."""
    total_pnl: Decimal                              # Total P&L (realized + unrealized)
    realized_pnl: Decimal                           # Realized P&L from closed positions
    unrealized_pnl: Decimal                         # Unrealized P&L from open positions
    total_fees: Decimal                             # Total fees paid
    win_rate: Decimal                               # Percentage of winning trades
    avg_win: Decimal                                # Average winning trade amount
    avg_loss: Decimal                               # Average losing trade amount
    profit_factor: Decimal                          # Gross profit / gross loss
    sharpe_ratio: Decimal                           # Risk-adjusted return metric
    max_drawdown: Decimal                           # Maximum drawdown percentage
    total_trades: int                               # Total number of trades
    
    # Optional fields
    initial_balance: Optional[Decimal] = None       # Starting balance for ROI calculation
    current_balance: Optional[Decimal] = None       # Current portfolio balance
    winning_trades: Optional[int] = None            # Number of winning trades
    losing_trades: Optional[int] = None            # Number of losing trades
    
    @property
    def roi(self) -> Optional[Decimal]:
        """Calculate return on investment."""
        if self.initial_balance is None or self.initial_balance == 0:
            return None
        return self.total_pnl / self.initial_balance
    
    @property
    def roi_pct(self) -> Optional[Decimal]:
        """Calculate ROI as percentage."""
        roi = self.roi
        return roi * 100 if roi is not None else None
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after fees."""
        return self.total_pnl - self.total_fees


@dataclass
class RiskMetrics:
    """Portfolio risk assessment metrics."""
    var_95: Decimal                                 # 95% Value at Risk
    var_99: Decimal                                 # 99% Value at Risk
    expected_shortfall: Decimal                     # Expected shortfall (CVaR)
    volatility: Decimal                            # Portfolio volatility
    beta: Optional[Decimal] = None                 # Market beta
    correlation_btc: Optional[Decimal] = None      # Correlation with BTC
    correlation_eth: Optional[Decimal] = None      # Correlation with ETH
    max_position_risk: Decimal = Decimal("0")      # Largest position risk
    concentration_risk: Decimal = Decimal("0")     # Portfolio concentration
    leverage_ratio: Decimal = Decimal("1")         # Overall portfolio leverage


@dataclass 
class DrawdownMetrics:
    """Portfolio drawdown analysis."""
    current_drawdown: Decimal                       # Current drawdown from peak
    max_drawdown: Decimal                          # Maximum historical drawdown
    max_drawdown_duration_days: int                # Longest drawdown period
    recovery_time_days: Optional[int] = None       # Time to recover from max drawdown
    drawdown_start: Optional[datetime] = None      # Start of current drawdown
    peak_value: Optional[Decimal] = None           # All-time high portfolio value
    trough_value: Optional[Decimal] = None         # Lowest value during max drawdown


@dataclass
class Portfolio:
    """
    Core portfolio data structure for multi-chain, multi-DEX trading.
    
    Aggregates positions, transactions, and performance metrics across
    all supported DEXs and blockchain networks.
    """
    portfolio_id: UUID                              # Unique portfolio identifier
    name: str                                       # Portfolio name
    config: PortfolioConfig                         # Portfolio configuration
    cash_balance: Decimal                           # Available cash balance
    total_value: Decimal                           # Total portfolio value
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Portfolio contents
    positions: Dict[UUID, Position] = field(default_factory=dict)
    transactions: List[Transaction] = field(default_factory=list)
    
    # Performance tracking
    performance_metrics: Optional[PerformanceMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None
    drawdown_metrics: Optional[DrawdownMetrics] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def open_positions(self) -> Dict[UUID, Position]:
        """Get all open positions."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.status == PositionStatus.OPEN
        }
    
    @property
    def closed_positions(self) -> Dict[UUID, Position]:
        """Get all closed positions."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.status == PositionStatus.CLOSED
        }
    
    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L from all open positions."""
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())
    
    @property
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(pos.market_value for pos in self.open_positions.values())
    
    @property
    def total_cost_basis(self) -> Decimal:
        """Calculate total cost basis of all positions.""" 
        return sum(pos.cost_basis for pos in self.positions.values())
    
    @property
    def equity(self) -> Decimal:
        """Calculate total equity (cash + position values)."""
        return self.cash_balance + self.total_market_value
    
    @property
    def buying_power(self) -> Decimal:
        """Calculate available buying power."""
        # For now, just return cash balance
        # Could be enhanced with margin calculations
        return self.cash_balance
    
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        self.positions[position.position_id] = position
        self.updated_at = datetime.now()
        logger.info(
            "Position added to portfolio",
            portfolio_id=str(self.portfolio_id),
            position_id=str(position.position_id),
            symbol=position.symbol,
            size=str(position.size)
        )
    
    def remove_position(self, position_id: UUID) -> None:
        """Remove a position from the portfolio."""
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            self.updated_at = datetime.now()
            logger.info(
                "Position removed from portfolio",
                portfolio_id=str(self.portfolio_id),
                position_id=str(position_id),
                symbol=position.symbol
            )
    
    def add_transaction(self, transaction: Transaction) -> None:
        """Add a new transaction to the portfolio."""
        self.transactions.append(transaction)
        self.updated_at = datetime.now()
        logger.info(
            "Transaction added to portfolio",
            portfolio_id=str(self.portfolio_id),
            transaction_id=str(transaction.transaction_id),
            type=transaction.transaction_type.value,
            symbol=transaction.symbol,
            size=str(transaction.size)
        )
    
    def update_cash_balance(self, amount: Decimal, reason: str = "") -> None:
        """Update cash balance with logging."""
        old_balance = self.cash_balance
        self.cash_balance += amount
        self.updated_at = datetime.now()
        
        logger.info(
            "Cash balance updated",
            portfolio_id=str(self.portfolio_id),
            old_balance=str(old_balance),
            new_balance=str(self.cash_balance),
            change=str(amount),
            reason=reason
        )
    
    def get_positions_by_chain(self, chain: Chain) -> Dict[UUID, Position]:
        """Get all positions for a specific chain."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.chain == chain
        }
    
    def get_positions_by_dex(self, dex_name: str) -> Dict[UUID, Position]:
        """Get all positions for a specific DEX."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.dex_name == dex_name
        }
    
    def get_positions_by_symbol(self, symbol: str) -> Dict[UUID, Position]:
        """Get all positions for a specific trading pair."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.symbol == symbol
        }


# Portfolio Exception Classes
class PortfolioError(Exception):
    """Base portfolio error."""
    pass


class InsufficientFundsError(PortfolioError):
    """Error when insufficient funds for operation."""
    pass


class RiskLimitExceededError(PortfolioError):
    """Error when risk limits are exceeded."""
    pass


class PositionNotFoundError(PortfolioError):
    """Error when position is not found."""
    pass


class InvalidPositionError(PortfolioError):
    """Error when position data is invalid."""
    pass


class PortfolioConfigError(PortfolioError):
    """Error in portfolio configuration."""
    pass