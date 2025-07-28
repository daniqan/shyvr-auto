"""
Virtual Portfolio for Simulation Trading

This module provides a virtual portfolio implementation specifically for simulation trading,
including virtual cash management, position tracking, and performance calculations.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
import structlog

from src.portfolio.base import (
    Position, Transaction, PositionType, PositionStatus,
    TransactionType, PerformanceMetrics, RiskMetrics
)
from src.utils.base import Chain


logger = structlog.get_logger()


class VirtualPortfolio:
    """Virtual portfolio for simulation trading."""
    
    def __init__(self, initial_balance: Decimal, base_currency: str = "USDC", enable_fees: bool = True):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.base_currency = base_currency
        self.enable_fees = enable_fees
        
        # Portfolio state
        self.positions: Dict[UUID, Position] = {}
        self.transaction_history: List[Transaction] = []
        self.cash_transactions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.peak_value = initial_balance
        self.current_value = initial_balance
        self.daily_pnl = Decimal("0")
        
        self.logger = logger.bind(component="virtual_portfolio")
    
    @property
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def equity(self) -> Decimal:
        """Calculate total equity."""
        return self.cash_balance + self.total_market_value
    
    def add_position(self, position: Position) -> None:
        """Add position to virtual portfolio."""
        self.positions[position.position_id] = position
        self.current_value = self.equity
        
        self.logger.info(
            "Position added to virtual portfolio",
            position_id=str(position.position_id),
            symbol=position.symbol,
            size=str(position.size),
            entry_price=str(position.entry_price)
        )
    
    def record_transaction(self, transaction: Transaction) -> None:
        """Record transaction in virtual portfolio."""
        self.transaction_history.append(transaction)
        
        # Update cash balance based on transaction
        if transaction.transaction_type == TransactionType.BUY:
            self.cash_balance -= transaction.value + transaction.fees
        elif transaction.transaction_type == TransactionType.SELL:
            self.cash_balance += transaction.value - transaction.fees
        elif transaction.transaction_type == TransactionType.FEE:
            self.cash_balance -= transaction.fees
        
        self.current_value = self.equity
        
        self.logger.info(
            "Transaction recorded",
            transaction_id=str(transaction.transaction_id),
            type=transaction.transaction_type.value,
            symbol=transaction.symbol,
            size=str(transaction.size),
            price=str(transaction.price)
        )
    
    def update_position_price(self, position_id: UUID, new_price: Decimal) -> None:
        """Update position with new market price."""
        if position_id in self.positions:
            position = self.positions[position_id]
            position.update_price(new_price)
            self.current_value = self.equity
    
    def spend_cash(self, amount: Decimal, reason: str = "") -> None:
        """Deduct cash from balance."""
        self.cash_balance -= amount
        self.cash_transactions.append({
            "amount": -amount,
            "reason": reason,
            "timestamp": datetime.now(),
            "balance_after": self.cash_balance
        })
    
    def add_cash(self, amount: Decimal, reason: str = "") -> None:
        """Add cash to balance."""
        self.cash_balance += amount
        self.cash_transactions.append({
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.now(),
            "balance_after": self.cash_balance
        })
    
    def calculate_performance(self) -> PerformanceMetrics:
        """Calculate virtual portfolio performance metrics."""
        # Basic calculations for simulation
        total_pnl = self.current_value - self.initial_balance
        realized_pnl = sum(
            tx.net_value for tx in self.transaction_history
            if tx.transaction_type == TransactionType.SELL
        )
        unrealized_pnl = self.total_unrealized_pnl
        
        total_fees = sum(tx.fees for tx in self.transaction_history)
        
        # Calculate win/loss metrics
        closed_trades = [
            tx for tx in self.transaction_history
            if tx.transaction_type == TransactionType.SELL
        ]
        
        winning_trades = len([tx for tx in closed_trades if tx.net_value > 0])
        total_trades = len(closed_trades)
        win_rate = Decimal(winning_trades / total_trades) if total_trades > 0 else Decimal("0")
        
        # Simplified metrics for simulation
        avg_win = Decimal("0")
        avg_loss = Decimal("0")
        profit_factor = Decimal("1")
        sharpe_ratio = Decimal("0")
        max_drawdown = (self.peak_value - self.current_value) / self.peak_value if self.peak_value > 0 else Decimal("0")
        
        return PerformanceMetrics(
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=total_fees,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            initial_balance=self.initial_balance,
            current_balance=self.current_value
        )
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate virtual portfolio risk metrics."""
        # Simplified risk metrics for simulation
        return RiskMetrics(
            var_95=Decimal("0"),
            var_99=Decimal("0"),
            expected_shortfall=Decimal("0"),
            volatility=Decimal("0"),
            max_position_risk=Decimal("0"),
            concentration_risk=Decimal("0"),
            leverage_ratio=Decimal("1")
        )