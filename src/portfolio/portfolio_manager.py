"""
Portfolio Manager - Central orchestrator for portfolio operations.

This module provides the main interface for managing portfolios across
multiple DEXs and blockchain networks. It integrates with PositionTracker
for position management and provides comprehensive portfolio operations.

Key Features:
- Create and manage multiple portfolios
- Portfolio-level operations (add/close positions, rebalancing)
- Multi-chain and multi-DEX support (Solana, Ethereum, Hyperliquid)
- Batch operations for multiple positions
- Portfolio-level P&L and risk calculations
- Risk management and validation
- Performance metrics and reporting
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
import structlog

from .base import (
    Portfolio,
    Position,
    PositionType,
    PositionStatus,
    Transaction,
    TransactionType,
    PortfolioConfig,
    PerformanceMetrics,
    RiskMetrics,
    DrawdownMetrics,
    PortfolioError,
    InsufficientFundsError,
    RiskLimitExceededError,
    PositionNotFoundError,
    InvalidPositionError,
)
from .position_tracker import (
    PositionTracker,
    PositionUpdateResult,
    PositionCloseResult,
)
from src.utils.base import Chain


logger = structlog.get_logger()


@dataclass
class PortfolioCreateResult:
    """Result of portfolio creation operation."""
    success: bool
    portfolio_id: Optional[UUID] = None
    message: str = ""
    initial_balance: Optional[Decimal] = None


@dataclass
class PortfolioOperationResult:
    """Generic result for portfolio operations."""
    success: bool
    portfolio_id: UUID
    message: str = ""
    position_id: Optional[UUID] = None
    operation_type: str = ""
    
    # Additional fields for specific operations
    close_size: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    remaining_size: Optional[Decimal] = None
    fees: Optional[Decimal] = None


@dataclass
class DrawdownRiskResult:
    """Result of drawdown risk assessment."""
    at_risk: bool
    current_drawdown: Decimal
    max_allowed_drawdown: Decimal
    portfolio_id: UUID
    current_value: Decimal
    peak_value: Decimal
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class RebalancingCheck:
    """Result of rebalancing need assessment."""
    auto_rebalancing_enabled: bool
    rebalancing_needed: bool
    portfolio_id: UUID
    current_imbalance: Decimal = Decimal("0")
    threshold: Decimal = Decimal("0")
    reason: str = ""


@dataclass
class RebalancingAdjustment:
    """Individual rebalancing adjustment suggestion."""
    position_id: UUID
    symbol: str
    current_weight: Decimal
    target_weight: Decimal
    adjustment_amount: Decimal
    action: str  # "REDUCE", "INCREASE", "CLOSE"


@dataclass
class RebalancingSuggestions:
    """Portfolio rebalancing suggestions."""
    portfolio_id: UUID
    total_imbalance: Decimal
    suggested_adjustments: List[RebalancingAdjustment]
    estimated_fees: Decimal = Decimal("0")
    expected_improvement: Optional[Decimal] = None


class PortfolioManager:
    """
    Central orchestrator for portfolio operations across multiple DEXs.
    
    Manages multiple portfolios, positions, and provides comprehensive
    portfolio-level operations including risk management, performance
    tracking, and rebalancing capabilities.
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio configuration
            
        Raises:
            ValueError: If config is invalid
        """
        if config is None:
            raise ValueError("Portfolio configuration cannot be None")
        
        self.config = config
        self.portfolios: Dict[UUID, Portfolio] = {}
        self.position_tracker = PositionTracker(config)
        self.default_portfolio_id: Optional[UUID] = None
        self.logger = logger.bind(component="portfolio_manager")
        
        # Performance tracking
        self._portfolio_peaks: Dict[UUID, Decimal] = {}
        self._performance_cache: Dict[UUID, Tuple[datetime, PerformanceMetrics]] = {}
        self._risk_cache: Dict[UUID, Tuple[datetime, RiskMetrics]] = {}
        
        self.logger.info(
            "Portfolio manager initialized",
            initial_balance=str(config.initial_balance),
            base_currency=config.base_currency,
            max_positions=config.max_open_positions,
            risk_management_enabled=config.enable_risk_management
        )
    
    @property
    def total_portfolios(self) -> int:
        """Get total number of portfolios."""
        return len(self.portfolios)
    
    @property
    def total_value(self) -> Decimal:
        """Get total value across all portfolios."""
        return sum(portfolio.equity for portfolio in self.portfolios.values())
    
    @property
    def total_cash(self) -> Decimal:
        """Get total cash across all portfolios."""
        return sum(portfolio.cash_balance for portfolio in self.portfolios.values())
    
    @property
    def total_positions(self) -> int:
        """Get total number of positions across all portfolios."""
        return sum(len(portfolio.positions) for portfolio in self.portfolios.values())
    
    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L across all portfolios."""
        return sum(portfolio.total_unrealized_pnl for portfolio in self.portfolios.values())
    
    async def create_portfolio(
        self,
        name: str,
        initial_balance: Decimal,
        custom_config: Optional[PortfolioConfig] = None
    ) -> PortfolioCreateResult:
        """
        Create a new portfolio.
        
        Args:
            name: Portfolio name
            initial_balance: Initial cash balance
            custom_config: Optional custom configuration
            
        Returns:
            PortfolioCreateResult with creation status
        """
        try:
            # Validate inputs
            if initial_balance <= 0:
                return PortfolioCreateResult(
                    success=False,
                    message="Initial balance must be positive"
                )
            
            # Check for duplicate names
            if self.get_portfolio_by_name(name) is not None:
                return PortfolioCreateResult(
                    success=False,
                    message=f"Portfolio with name '{name}' already exists"
                )
            
            # Use custom config or default
            config = custom_config or self.config
            
            # Create portfolio
            portfolio_id = uuid4()
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                name=name,
                config=config,
                cash_balance=initial_balance,
                total_value=initial_balance
            )
            
            # Add to portfolios
            self.portfolios[portfolio_id] = portfolio
            
            # Initialize peak tracking
            self._portfolio_peaks[portfolio_id] = initial_balance
            
            # Set as default if first portfolio
            if self.default_portfolio_id is None:
                self.default_portfolio_id = portfolio_id
            
            self.logger.info(
                "Portfolio created successfully",
                portfolio_id=str(portfolio_id),
                name=name,
                initial_balance=str(initial_balance),
                is_default=self.default_portfolio_id == portfolio_id
            )
            
            return PortfolioCreateResult(
                success=True,
                portfolio_id=portfolio_id,
                message="Portfolio created successfully",
                initial_balance=initial_balance
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to create portfolio",
                name=name,
                initial_balance=str(initial_balance),
                error=str(e)
            )
            return PortfolioCreateResult(
                success=False,
                message=f"Failed to create portfolio: {e}"
            )
    
    async def set_default_portfolio(self, portfolio_id: UUID) -> PortfolioOperationResult:
        """
        Set default portfolio.
        
        Args:
            portfolio_id: Portfolio to set as default
            
        Returns:
            PortfolioOperationResult with operation status
        """
        try:
            if portfolio_id not in self.portfolios:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    message="Portfolio not found"
                )
            
            self.default_portfolio_id = portfolio_id
            
            self.logger.info(
                "Default portfolio set",
                portfolio_id=str(portfolio_id),
                portfolio_name=self.portfolios[portfolio_id].name
            )
            
            return PortfolioOperationResult(
                success=True,
                portfolio_id=portfolio_id,
                message="Default portfolio set successfully"
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to set default portfolio",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return PortfolioOperationResult(
                success=False,
                portfolio_id=portfolio_id,
                message=f"Failed to set default portfolio: {e}"
            )
    
    def get_portfolio(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """
        Get portfolio by ID.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Portfolio if found, None otherwise
        """
        return self.portfolios.get(portfolio_id)
    
    def get_portfolio_by_name(self, name: str) -> Optional[Portfolio]:
        """
        Get portfolio by name.
        
        Args:
            name: Portfolio name
            
        Returns:
            Portfolio if found, None otherwise
        """
        for portfolio in self.portfolios.values():
            if portfolio.name == name:
                return portfolio
        return None
    
    async def add_position(
        self,
        portfolio_id: UUID,
        position: Position
    ) -> PortfolioOperationResult:
        """
        Add position to portfolio.
        
        Args:
            portfolio_id: Target portfolio
            position: Position to add
            
        Returns:
            PortfolioOperationResult with operation status
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    message="Portfolio not found",
                    operation_type="add_position"
                )
            
            # Validate position data
            validation_result = await self._validate_position(portfolio, position)
            if not validation_result.success:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    position_id=position.position_id,
                    message=validation_result.message,
                    operation_type="add_position"
                )
            
            # Check risk limits
            risk_check = await self._check_risk_limits(portfolio, position)
            if not risk_check.success:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    position_id=position.position_id,
                    message=risk_check.message,
                    operation_type="add_position"
                )
            
            # Add to position tracker
            tracker_result = self.position_tracker.add_position(position)
            if not tracker_result.success:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    position_id=position.position_id,
                    message=tracker_result.message,
                    operation_type="add_position"
                )
            
            # Add to portfolio
            portfolio.add_position(position)
            
            # Update portfolio cash balance (reduce by position cost)
            position_cost = position.cost_basis
            portfolio.update_cash_balance(-position_cost, "position_opened")
            
            # Update portfolio total value
            await self._update_portfolio_value(portfolio)
            
            # Clear performance cache
            self._clear_portfolio_cache(portfolio_id)
            
            self.logger.info(
                "Position added to portfolio",
                portfolio_id=str(portfolio_id),
                position_id=str(position.position_id),
                symbol=position.symbol,
                size=str(position.size),
                cost=str(position_cost)
            )
            
            return PortfolioOperationResult(
                success=True,
                portfolio_id=portfolio_id,
                position_id=position.position_id,
                message="Position added successfully",
                operation_type="add_position"
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to add position to portfolio",
                portfolio_id=str(portfolio_id),
                position_id=str(position.position_id),
                error=str(e)
            )
            return PortfolioOperationResult(
                success=False,
                portfolio_id=portfolio_id,
                position_id=position.position_id,
                message=f"Failed to add position: {e}",
                operation_type="add_position"
            )
    
    async def close_position(
        self,
        portfolio_id: UUID,
        position_id: UUID,
        close_size: Decimal,
        close_price: Decimal,
        reason: str = "manual_close",
        fees: Decimal = Decimal("0")
    ) -> PortfolioOperationResult:
        """
        Close position (fully or partially).
        
        Args:
            portfolio_id: Portfolio containing the position
            position_id: Position to close
            close_size: Size to close
            close_price: Price at which to close
            reason: Reason for closing
            fees: Transaction fees
            
        Returns:
            PortfolioOperationResult with close details
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    message="Portfolio not found",
                    operation_type="close_position"
                )
            
            # Close through position tracker
            close_result = self.position_tracker.close_position(
                position_id=position_id,
                close_size=close_size,
                close_price=close_price,
                reason=reason,
                fees=fees
            )
            
            if not close_result.success:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio_id,
                    position_id=position_id,
                    message=close_result.message,
                    operation_type="close_position"
                )
            
            # Update portfolio cash balance (add proceeds)
            proceeds = close_size * close_price - fees
            portfolio.update_cash_balance(proceeds, f"position_closed_{reason}")
            
            # Create transaction record
            transaction = Transaction(
                transaction_id=uuid4(),
                position_id=position_id,
                transaction_type=TransactionType.SELL,
                symbol=portfolio.positions[position_id].symbol,
                size=close_size,
                price=close_price,
                timestamp=datetime.now(),
                chain=portfolio.positions[position_id].chain,
                dex_name=portfolio.positions[position_id].dex_name,
                transaction_hash=f"close_{position_id}_{datetime.now().isoformat()}",
                fees=fees
            )
            portfolio.add_transaction(transaction)
            
            # Update portfolio total value
            await self._update_portfolio_value(portfolio)
            
            # Clear performance cache
            self._clear_portfolio_cache(portfolio_id)
            
            self.logger.info(
                "Position closed in portfolio",
                portfolio_id=str(portfolio_id),
                position_id=str(position_id),
                close_size=str(close_size),
                close_price=str(close_price),
                realized_pnl=str(close_result.realized_pnl),
                reason=reason
            )
            
            return PortfolioOperationResult(
                success=True,
                portfolio_id=portfolio_id,
                position_id=position_id,
                message="Position closed successfully",
                operation_type="close_position",
                close_size=close_size,
                close_price=close_price,
                realized_pnl=close_result.realized_pnl,
                remaining_size=close_result.remaining_size,
                fees=fees
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to close position",
                portfolio_id=str(portfolio_id),
                position_id=str(position_id),
                error=str(e)
            )
            return PortfolioOperationResult(
                success=False,
                portfolio_id=portfolio_id,
                position_id=position_id,
                message=f"Failed to close position: {e}",
                operation_type="close_position"
            )
    
    async def update_position_price(
        self,
        position_id: UUID,
        new_price: Decimal
    ) -> PositionUpdateResult:
        """
        Update position price.
        
        Args:
            position_id: Position to update
            new_price: New market price
            
        Returns:
            PositionUpdateResult with update details
        """
        try:
            # Update through position tracker
            result = self.position_tracker.update_position_price(position_id, new_price)
            
            if result.success:
                # Update all portfolios containing this position
                for portfolio in self.portfolios.values():
                    if position_id in portfolio.positions:
                        await self._update_portfolio_value(portfolio)
                        self._clear_portfolio_cache(portfolio.portfolio_id)
                        break
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to update position price",
                position_id=str(position_id),
                new_price=str(new_price),
                error=str(e)
            )
            return PositionUpdateResult(
                success=False,
                position_id=position_id,
                message=f"Failed to update price: {e}"
            )
    
    async def batch_add_positions(
        self,
        portfolio_id: UUID,
        positions: List[Position]
    ) -> List[PortfolioOperationResult]:
        """
        Add multiple positions in batch.
        
        Args:
            portfolio_id: Target portfolio
            positions: Positions to add
            
        Returns:
            List of PortfolioOperationResult for each position
        """
        results = []
        
        for position in positions:
            result = await self.add_position(portfolio_id, position)
            results.append(result)
        
        self.logger.info(
            "Batch position addition completed",
            portfolio_id=str(portfolio_id),
            total_positions=len(positions),
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success)
        )
        
        return results
    
    async def batch_update_prices(
        self,
        price_updates: Dict[UUID, Decimal]
    ) -> List[PositionUpdateResult]:
        """
        Update multiple position prices in batch.
        
        Args:
            price_updates: Dictionary of position_id -> new_price
            
        Returns:
            List of PositionUpdateResult for each update
        """
        results = []
        
        for position_id, new_price in price_updates.items():
            result = await self.update_position_price(position_id, new_price)
            results.append(result)
        
        self.logger.info(
            "Batch price update completed",
            total_updates=len(price_updates),
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success)
        )
        
        return results
    
    async def batch_close_positions(
        self,
        portfolio_id: UUID,
        close_requests: List[Dict[str, Any]]
    ) -> List[PortfolioOperationResult]:
        """
        Close multiple positions in batch.
        
        Args:
            portfolio_id: Portfolio containing positions
            close_requests: List of close request dictionaries
            
        Returns:
            List of PortfolioOperationResult for each close operation
        """
        results = []
        
        for request in close_requests:
            result = await self.close_position(
                portfolio_id=portfolio_id,
                position_id=request['position_id'],
                close_size=request['close_size'],
                close_price=request['close_price'],
                reason=request.get('reason', 'batch_close'),
                fees=request.get('fees', Decimal("0"))
            )
            results.append(result)
        
        self.logger.info(
            "Batch position close completed",
            portfolio_id=str(portfolio_id),
            total_closes=len(close_requests),
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success)
        )
        
        return results
    
    async def get_positions_by_chain(
        self,
        portfolio_id: UUID
    ) -> Dict[Chain, List[Position]]:
        """
        Get positions grouped by blockchain chain.
        
        Args:
            portfolio_id: Portfolio to analyze
            
        Returns:
            Dictionary of Chain -> List[Position]
        """
        portfolio = self.get_portfolio(portfolio_id)
        if portfolio is None:
            return {}
        
        chain_positions: Dict[Chain, List[Position]] = {}
        
        for position in portfolio.positions.values():
            if position.chain not in chain_positions:
                chain_positions[position.chain] = []
            chain_positions[position.chain].append(position)
        
        return chain_positions
    
    async def get_positions_by_dex(
        self,
        portfolio_id: UUID
    ) -> Dict[str, List[Position]]:
        """
        Get positions grouped by DEX.
        
        Args:
            portfolio_id: Portfolio to analyze
            
        Returns:
            Dictionary of dex_name -> List[Position]
        """
        portfolio = self.get_portfolio(portfolio_id)
        if portfolio is None:
            return {}
        
        dex_positions: Dict[str, List[Position]] = {}
        
        for position in portfolio.positions.values():
            if position.dex_name not in dex_positions:
                dex_positions[position.dex_name] = []
            dex_positions[position.dex_name].append(position)
        
        return dex_positions
    
    async def calculate_performance_metrics(
        self,
        portfolio_id: UUID,
        use_cache: bool = True
    ) -> Optional[PerformanceMetrics]:
        """
        Calculate comprehensive performance metrics for portfolio.
        
        Args:
            portfolio_id: Portfolio to analyze
            use_cache: Whether to use cached metrics if available
            
        Returns:
            PerformanceMetrics if successful, None otherwise
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return None
            
            # Check cache first
            if use_cache and portfolio_id in self._performance_cache:
                cache_time, metrics = self._performance_cache[portfolio_id]
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return metrics
            
            # Calculate metrics
            total_realized_pnl = Decimal("0")
            total_unrealized_pnl = portfolio.total_unrealized_pnl
            total_fees = Decimal("0")
            
            winning_trades = 0
            losing_trades = 0
            total_win_amount = Decimal("0")
            total_loss_amount = Decimal("0")
            
            # Analyze closed positions for realized P&L
            for position in portfolio.closed_positions.values():
                # Calculate realized P&L (simplified)
                realized_pnl = (position.current_price - position.entry_price) * position.size
                if position.leverage > 1:
                    realized_pnl *= position.leverage
                
                total_realized_pnl += realized_pnl
                
                if realized_pnl > 0:
                    winning_trades += 1
                    total_win_amount += realized_pnl
                else:
                    losing_trades += 1
                    total_loss_amount += abs(realized_pnl)
            
            # Analyze transactions for fees
            for transaction in portfolio.transactions:
                total_fees += transaction.total_fees
            
            # Calculate metrics
            total_trades = winning_trades + losing_trades
            win_rate = Decimal(winning_trades) / Decimal(total_trades) if total_trades > 0 else Decimal("0")
            avg_win = total_win_amount / Decimal(winning_trades) if winning_trades > 0 else Decimal("0")
            avg_loss = total_loss_amount / Decimal(losing_trades) if losing_trades > 0 else Decimal("0")
            profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else Decimal("0")
            
            # Get the actual initial balance from portfolio peaks (tracks original portfolio value)
            actual_initial_balance = self._portfolio_peaks.get(portfolio_id, portfolio.config.initial_balance)
            
            # Calculate Sharpe ratio (simplified)
            returns = (total_realized_pnl + total_unrealized_pnl) / actual_initial_balance
            sharpe_ratio = returns * Decimal("3.46")  # Simplified: assuming annual returns and 10% volatility
            
            # Calculate max drawdown
            current_value = portfolio.equity
            peak_value = self._portfolio_peaks.get(portfolio_id, actual_initial_balance)
            max_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else Decimal("0")
            
            metrics = PerformanceMetrics(
                total_pnl=total_realized_pnl + total_unrealized_pnl,
                realized_pnl=total_realized_pnl,
                unrealized_pnl=total_unrealized_pnl,
                total_fees=total_fees,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_trades=total_trades,
                initial_balance=actual_initial_balance,
                current_balance=current_value,
                winning_trades=winning_trades,
                losing_trades=losing_trades
            )
            
            # Cache metrics
            self._performance_cache[portfolio_id] = (datetime.now(), metrics)
            
            # Update portfolio metrics
            portfolio.performance_metrics = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate performance metrics",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return None
    
    async def calculate_risk_metrics(
        self,
        portfolio_id: UUID,
        use_cache: bool = True
    ) -> Optional[RiskMetrics]:
        """
        Calculate risk metrics for portfolio.
        
        Args:
            portfolio_id: Portfolio to analyze
            use_cache: Whether to use cached metrics if available
            
        Returns:
            RiskMetrics if successful, None otherwise
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return None
            
            # Check cache first
            if use_cache and portfolio_id in self._risk_cache:
                cache_time, metrics = self._risk_cache[portfolio_id]
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return metrics
            
            # Calculate risk metrics (simplified implementation)
            total_value = portfolio.equity
            total_positions = len(portfolio.open_positions)
            
            # Calculate portfolio volatility (simplified)
            if total_positions > 0:
                position_volatilities = []
                for position in portfolio.open_positions.values():
                    # Simplified volatility calculation
                    price_change_pct = abs(position.unrealized_pnl_pct)
                    position_volatilities.append(price_change_pct)
                
                avg_volatility = sum(position_volatilities) / len(position_volatilities)
            else:
                avg_volatility = Decimal("0")
            
            # Calculate VaR (simplified - 95% and 99%)
            portfolio_return = portfolio.total_unrealized_pnl / total_value if total_value > 0 else Decimal("0")
            var_95 = abs(portfolio_return * Decimal("1.65"))  # 95% confidence
            var_99 = abs(portfolio_return * Decimal("2.33"))  # 99% confidence
            expected_shortfall = var_95 * Decimal("1.3")  # Simplified CVaR
            
            # Calculate concentration risk
            if total_positions > 0:
                largest_position_value = max(pos.market_value for pos in portfolio.open_positions.values())
                concentration_risk = largest_position_value / total_value if total_value > 0 else Decimal("1")
                max_position_risk = concentration_risk
            else:
                concentration_risk = Decimal("0")
                max_position_risk = Decimal("0")
            
            # Calculate leverage ratio
            total_margin = sum(
                pos.margin_used or Decimal("0") 
                for pos in portfolio.open_positions.values()
            )
            leverage_ratio = total_value / total_margin if total_margin > 0 else Decimal("1")
            
            metrics = RiskMetrics(
                var_95=var_95 * total_value,
                var_99=var_99 * total_value,
                expected_shortfall=expected_shortfall * total_value,
                volatility=avg_volatility,
                max_position_risk=max_position_risk,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio
            )
            
            # Cache metrics
            self._risk_cache[portfolio_id] = (datetime.now(), metrics)
            
            # Update portfolio metrics
            portfolio.risk_metrics = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate risk metrics",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return None
    
    async def get_portfolio_summary(
        self,
        portfolio_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive portfolio summary.
        
        Args:
            portfolio_id: Portfolio to summarize
            
        Returns:
            Summary dictionary if successful, None otherwise
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return None
            
            # Calculate metrics
            performance_metrics = await self.calculate_performance_metrics(portfolio_id)
            risk_metrics = await self.calculate_risk_metrics(portfolio_id)
            
            # Get position breakdowns
            positions_by_chain = await self.get_positions_by_chain(portfolio_id)
            positions_by_dex = await self.get_positions_by_dex(portfolio_id)
            
            # Convert position lists to counts for summary
            chain_counts = {chain.value: len(positions) for chain, positions in positions_by_chain.items()}
            dex_counts = {dex: len(positions) for dex, positions in positions_by_dex.items()}
            
            summary = {
                'portfolio_id': str(portfolio_id),
                'name': portfolio.name,
                'total_value': portfolio.equity,
                'cash_balance': portfolio.cash_balance,
                'total_positions': len(portfolio.positions),
                'open_positions': len(portfolio.open_positions),
                'closed_positions': len(portfolio.closed_positions),
                'unrealized_pnl': portfolio.total_unrealized_pnl,
                'total_market_value': portfolio.total_market_value,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'positions_by_chain': chain_counts,
                'positions_by_dex': dex_counts,
                'created_at': portfolio.created_at.isoformat(),
                'updated_at': portfolio.updated_at.isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(
                "Failed to get portfolio summary",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return None
    
    async def check_drawdown_risk(
        self,
        portfolio_id: UUID
    ) -> DrawdownRiskResult:
        """
        Check portfolio drawdown risk.
        
        Args:
            portfolio_id: Portfolio to check
            
        Returns:
            DrawdownRiskResult with risk assessment
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return DrawdownRiskResult(
                    at_risk=False,
                    current_drawdown=Decimal("0"),
                    max_allowed_drawdown=self.config.max_drawdown_pct,
                    portfolio_id=portfolio_id,
                    current_value=Decimal("0"),
                    peak_value=Decimal("0"),
                    risk_level="UNKNOWN"
                )
            
            current_value = portfolio.equity
            peak_value = self._portfolio_peaks.get(portfolio_id, portfolio.config.initial_balance)
            
            # Update peak if current value is higher
            if current_value > peak_value:
                self._portfolio_peaks[portfolio_id] = current_value
                peak_value = current_value
            
            # Calculate current drawdown
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else Decimal("0")
            max_allowed = self.config.max_drawdown_pct
            
            # Determine risk level
            risk_level = "LOW"
            if current_drawdown > max_allowed:
                risk_level = "CRITICAL"
            elif current_drawdown > max_allowed * Decimal("0.8"):
                risk_level = "HIGH"
            elif current_drawdown > max_allowed * Decimal("0.5"):
                risk_level = "MEDIUM"
            
            at_risk = current_drawdown > max_allowed
            
            return DrawdownRiskResult(
                at_risk=at_risk,
                current_drawdown=current_drawdown,
                max_allowed_drawdown=max_allowed,
                portfolio_id=portfolio_id,
                current_value=current_value,
                peak_value=peak_value,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to check drawdown risk",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return DrawdownRiskResult(
                at_risk=False,
                current_drawdown=Decimal("0"),
                max_allowed_drawdown=self.config.max_drawdown_pct,
                portfolio_id=portfolio_id,
                current_value=Decimal("0"),
                peak_value=Decimal("0"),
                risk_level="ERROR"
            )
    
    async def check_rebalancing_needed(
        self,
        portfolio_id: UUID
    ) -> RebalancingCheck:
        """
        Check if portfolio rebalancing is needed.
        
        Args:
            portfolio_id: Portfolio to check
            
        Returns:
            RebalancingCheck with rebalancing assessment
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return RebalancingCheck(
                    auto_rebalancing_enabled=False,
                    rebalancing_needed=False,
                    portfolio_id=portfolio_id
                )
            
            auto_enabled = portfolio.config.enable_auto_rebalancing
            threshold = portfolio.config.rebalance_threshold_pct
            
            if not auto_enabled:
                return RebalancingCheck(
                    auto_rebalancing_enabled=False,
                    rebalancing_needed=False,
                    portfolio_id=portfolio_id,
                    threshold=threshold,
                    reason="Auto-rebalancing disabled"
                )
            
            # Calculate position weights and imbalances (simplified)
            total_value = portfolio.total_market_value
            if total_value == 0:
                return RebalancingCheck(
                    auto_rebalancing_enabled=auto_enabled,
                    rebalancing_needed=False,
                    portfolio_id=portfolio_id,
                    threshold=threshold,
                    reason="No positions to rebalance"
                )
            
            # Calculate maximum position imbalance
            max_imbalance = Decimal("0")
            for position in portfolio.open_positions.values():
                position_weight = position.market_value / total_value
                target_weight = Decimal("1") / len(portfolio.open_positions)  # Equal weight target
                imbalance = abs(position_weight - target_weight)
                max_imbalance = max(max_imbalance, imbalance)
            
            rebalancing_needed = max_imbalance > threshold
            
            return RebalancingCheck(
                auto_rebalancing_enabled=auto_enabled,
                rebalancing_needed=rebalancing_needed,
                portfolio_id=portfolio_id,
                current_imbalance=max_imbalance,
                threshold=threshold,
                reason=f"Max imbalance: {max_imbalance:.2%}, threshold: {threshold:.2%}"
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to check rebalancing need",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return RebalancingCheck(
                auto_rebalancing_enabled=False,
                rebalancing_needed=False,
                portfolio_id=portfolio_id,
                reason=f"Error: {e}"
            )
    
    async def get_rebalancing_suggestions(
        self,
        portfolio_id: UUID
    ) -> Optional[RebalancingSuggestions]:
        """
        Get rebalancing suggestions for portfolio.
        
        Args:
            portfolio_id: Portfolio to analyze
            
        Returns:
            RebalancingSuggestions if successful, None otherwise
        """
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if portfolio is None:
                return None
            
            total_value = portfolio.total_market_value
            if total_value == 0 or len(portfolio.open_positions) == 0:
                return RebalancingSuggestions(
                    portfolio_id=portfolio_id,
                    total_imbalance=Decimal("0"),
                    suggested_adjustments=[]
                )
            
            # Calculate target weights (equal weight for simplicity)
            num_positions = len(portfolio.open_positions)
            target_weight = Decimal("1") / num_positions
            
            adjustments = []
            total_imbalance = Decimal("0")
            
            for position in portfolio.open_positions.values():
                current_weight = position.market_value / total_value
                weight_diff = current_weight - target_weight
                imbalance = abs(weight_diff)
                total_imbalance += imbalance
                
                if imbalance > portfolio.config.rebalance_threshold_pct:
                    adjustment_amount = weight_diff * total_value
                    action = "REDUCE" if weight_diff > 0 else "INCREASE"
                    
                    adjustment = RebalancingAdjustment(
                        position_id=position.position_id,
                        symbol=position.symbol,
                        current_weight=current_weight,
                        target_weight=target_weight,
                        adjustment_amount=abs(adjustment_amount),
                        action=action
                    )
                    adjustments.append(adjustment)
            
            # Estimate fees (simplified)
            estimated_fees = sum(adj.adjustment_amount * Decimal("0.003") for adj in adjustments)  # 0.3% fee estimate
            
            return RebalancingSuggestions(
                portfolio_id=portfolio_id,
                total_imbalance=total_imbalance,
                suggested_adjustments=adjustments,
                estimated_fees=estimated_fees,
                expected_improvement=total_imbalance * Decimal("0.8")  # Expected 80% improvement
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to get rebalancing suggestions",
                portfolio_id=str(portfolio_id),
                error=str(e)
            )
            return None
    
    # Private helper methods
    
    async def _validate_position(
        self,
        portfolio: Portfolio,
        position: Position
    ) -> PortfolioOperationResult:
        """Validate position data and constraints."""
        try:
            # Basic validation
            if not position.symbol or position.symbol.strip() == "":
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message="Invalid position data: empty symbol"
                )
            
            if position.size <= 0:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message="Invalid position data: negative or zero size"
                )
            
            if position.entry_price <= 0:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message="Invalid position data: negative or zero entry price"
                )
            
            return PortfolioOperationResult(
                success=True,
                portfolio_id=portfolio.portfolio_id,
                message="Position validation passed"
            )
            
        except Exception as e:
            return PortfolioOperationResult(
                success=False,
                portfolio_id=portfolio.portfolio_id,
                message=f"Position validation error: {e}"
            )
    
    async def _check_risk_limits(
        self,
        portfolio: Portfolio,
        position: Position
    ) -> PortfolioOperationResult:
        """Check position against risk limits."""
        try:
            if not portfolio.config.enable_risk_management:
                return PortfolioOperationResult(
                    success=True,
                    portfolio_id=portfolio.portfolio_id,
                    message="Risk management disabled"
                )
            
            # Check maximum positions limit
            if len(portfolio.open_positions) >= portfolio.config.max_open_positions:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message=f"Maximum open positions limit ({portfolio.config.max_open_positions}) reached"
                )
            
            # Check position size limit
            position_value = position.cost_basis
            max_position_value = portfolio.equity * portfolio.config.max_position_size_pct
            
            if position_value > max_position_value:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message=f"Position value {position_value} exceeds maximum position size {max_position_value}"
                )
            
            # Check minimum trade amount
            if position_value < portfolio.config.min_trade_amount_usd:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message=f"Position value {position_value} below minimum trade amount {portfolio.config.min_trade_amount_usd}"
                )
            
            # Check sufficient funds
            if position_value > portfolio.cash_balance:
                return PortfolioOperationResult(
                    success=False,
                    portfolio_id=portfolio.portfolio_id,
                    message=f"Insufficient funds: need {position_value}, have {portfolio.cash_balance}"
                )
            
            return PortfolioOperationResult(
                success=True,
                portfolio_id=portfolio.portfolio_id,
                message="Risk limits check passed"
            )
            
        except Exception as e:
            return PortfolioOperationResult(
                success=False,
                portfolio_id=portfolio.portfolio_id,
                message=f"Risk limits check error: {e}"
            )
    
    async def _update_portfolio_value(self, portfolio: Portfolio) -> None:
        """Update portfolio total value."""
        try:
            portfolio.total_value = portfolio.equity
            portfolio.updated_at = datetime.now()
            
        except Exception as e:
            self.logger.error(
                "Failed to update portfolio value",
                portfolio_id=str(portfolio.portfolio_id),
                error=str(e)
            )
    
    def _clear_portfolio_cache(self, portfolio_id: UUID) -> None:
        """Clear cached metrics for portfolio."""
        if portfolio_id in self._performance_cache:
            del self._performance_cache[portfolio_id]
        if portfolio_id in self._risk_cache:
            del self._risk_cache[portfolio_id]