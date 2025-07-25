"""
Simulation mode implementation for paper trading.

This module implements a comprehensive simulation trading mode that provides
realistic paper trading capabilities with real-time market data integration,
virtual portfolio management, and risk controls.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import structlog

from src.modes.base import ModeBase, ModeConfig, ModeStatus
from src.portfolio.base import (
    Portfolio, Position, Transaction, PositionType, PositionStatus,
    TransactionType, PerformanceMetrics, RiskMetrics
)
from src.rl_agent.base import MarketState, TradeAction
from src.utils.base import Chain
from src.dex.base import SwapQuote, SwapResult, SwapStatus, DEXBase


logger = structlog.get_logger()


@dataclass
class SimulationConfig:
    """Configuration for simulation mode."""
    initial_balance: Decimal = Decimal("10000")
    base_currency: str = "USDC"
    enable_fees: bool = True
    slippage_bps: int = 10
    max_drawdown_pct: Decimal = Decimal("0.15")
    enable_risk_management: bool = True
    market_data_frequency_ms: int = 1000
    enable_real_time_pnl: bool = True
    enable_batch_execution: bool = True
    simulation_speed_multiplier: Decimal = Decimal("1.0")


@dataclass
class SimulationMetrics:
    """Simulation-specific performance metrics."""
    total_trades: int = 0
    virtual_balance: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    execution_latency_ms: List[float] = field(default_factory=list)
    average_slippage_bps: Decimal = Decimal("0")
    total_fees_paid: Decimal = Decimal("0")


class SimulationError(Exception):
    """Base simulation error."""
    pass


class MarketDataFeed:
    """Real-time market data feed for simulation."""
    
    def __init__(self, dex_clients: Dict[str, DEXBase], frequency_ms: int = 1000):
        self.dex_clients = dex_clients
        self.frequency_ms = frequency_ms
        self.is_active = False
        self._subscribers = []
        self._task = None
        self.logger = logger.bind(component="market_data_feed")
    
    async def start(self):
        """Start market data feed."""
        self.is_active = True
        self._task = asyncio.create_task(self._run_feed())
        self.logger.info("Market data feed started")
    
    async def stop(self):
        """Stop market data feed."""
        self.is_active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Market data feed stopped")
    
    def subscribe(self, callback):
        """Subscribe to market data updates."""
        self._subscribers.append(callback)
    
    async def _run_feed(self):
        """Run market data feed loop."""
        while self.is_active:
            try:
                # Get market data from DEX clients
                market_data = await self._fetch_market_data()
                
                # Notify subscribers
                for callback in self._subscribers:
                    try:
                        await callback(market_data)
                    except Exception as e:
                        self.logger.error("Error in market data callback", error=str(e))
                
                await asyncio.sleep(self.frequency_ms / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in market data feed", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data from DEX clients."""
        market_data = {}
        
        for dex_name, dex_client in self.dex_clients.items():
            try:
                # This would fetch real market data
                # For now, return empty dict
                market_data[dex_name] = {}
            except Exception as e:
                self.logger.error(f"Error fetching data from {dex_name}", error=str(e))
        
        return market_data


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


@dataclass
class RiskValidationResult:
    """Result of risk validation check."""
    is_valid: bool
    reason: str = ""
    risk_level: str = "low"


class SimulationRiskManager:
    """Risk management for simulation trading."""
    
    def __init__(self, virtual_portfolio: VirtualPortfolio, config: Dict[str, Any]):
        self.virtual_portfolio = virtual_portfolio
        self.config = config
        self.logger = logger.bind(component="simulation_risk_manager")
    
    async def validate_position_size(self, token_address: str, amount_usd: Decimal) -> RiskValidationResult:
        """Validate position size against risk limits."""
        max_position_pct = Decimal(str(self.config.get("max_position_size_pct", 0.1)))
        portfolio_value = self.virtual_portfolio.equity
        max_position_value = portfolio_value * max_position_pct
        
        if amount_usd > max_position_value:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Position size exceeds maximum ({max_position_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_daily_loss_limit(self) -> RiskValidationResult:
        """Check daily loss limit."""
        max_daily_loss_pct = Decimal(str(self.config.get("max_daily_loss_pct", 0.05)))
        portfolio_value = self.virtual_portfolio.equity
        max_daily_loss = portfolio_value * max_daily_loss_pct
        
        if abs(self.virtual_portfolio.daily_pnl) > max_daily_loss:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Daily loss limit exceeded ({max_daily_loss_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_max_drawdown(self) -> RiskValidationResult:
        """Check maximum drawdown limit."""
        max_drawdown_pct = Decimal(str(self.config.get("max_drawdown_pct", 0.15)))
        current_drawdown = (self.virtual_portfolio.peak_value - self.virtual_portfolio.current_value) / self.virtual_portfolio.peak_value
        
        if current_drawdown > max_drawdown_pct:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Maximum drawdown exceeded ({max_drawdown_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_stop_losses(self) -> List[Position]:
        """Check for positions that should trigger stop losses."""
        stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", 0.08)))
        stop_loss_positions = []
        
        for position in self.virtual_portfolio.positions.values():
            if position.status == PositionStatus.OPEN:
                loss_pct = (position.entry_price - position.current_price) / position.entry_price
                if loss_pct > stop_loss_pct:
                    stop_loss_positions.append(position)
        
        return stop_loss_positions
    
    async def check_take_profits(self) -> List[Position]:
        """Check for positions that should trigger take profits."""
        take_profit_pct = Decimal(str(self.config.get("take_profit_pct", 0.4)))
        take_profit_positions = []
        
        for position in self.virtual_portfolio.positions.values():
            if position.status == PositionStatus.OPEN:
                profit_pct = (position.current_price - position.entry_price) / position.entry_price
                if profit_pct > take_profit_pct:
                    take_profit_positions.append(position)
        
        return take_profit_positions
    
    async def validate_new_position(self, token_address: str) -> RiskValidationResult:
        """Validate that a new position can be opened."""
        max_positions = self.config.get("max_open_positions", 10)
        current_positions = len([p for p in self.virtual_portfolio.positions.values() if p.status == PositionStatus.OPEN])
        
        if current_positions >= max_positions:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Maximum number of positions exceeded ({max_positions})",
                risk_level="medium"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment."""
        risk_factors = []
        
        # Check various risk factors
        daily_loss_check = await self.check_daily_loss_limit()
        if not daily_loss_check.is_valid:
            risk_factors.append("daily_loss_limit")
        
        drawdown_check = await self.check_max_drawdown()
        if not drawdown_check.is_valid:
            risk_factors.append("max_drawdown")
        
        stop_losses = await self.check_stop_losses()
        if stop_losses:
            risk_factors.append("stop_loss_triggers")
        
        # Determine overall risk level
        if len(risk_factors) >= 3:
            overall_risk_level = "high"
        elif len(risk_factors) >= 1:
            overall_risk_level = "medium"
        else:
            overall_risk_level = "low"
        
        return {
            "overall_risk_level": overall_risk_level,
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if "daily_loss_limit" in risk_factors:
            recommendations.append("Consider stopping trading for the day")
        if "max_drawdown" in risk_factors:
            recommendations.append("Reduce position sizes and review strategy")
        if "stop_loss_triggers" in risk_factors:
            recommendations.append("Close positions that hit stop losses")
        
        return recommendations


class SimulationExecutor:
    """Executes paper trading operations with realistic simulation."""
    
    def __init__(self, virtual_portfolio: VirtualPortfolio, dex_clients: Dict[str, DEXBase], 
                 enable_fees: bool = True, slippage_bps: int = 10):
        self.virtual_portfolio = virtual_portfolio
        self.dex_clients = dex_clients
        self.enable_fees = enable_fees
        self.slippage_bps = slippage_bps
        self.is_active = False
        self.logger = logger.bind(component="simulation_executor")
    
    async def execute_buy_order(self, token_address: str, amount_usd: Decimal, max_slippage_bps: int = 10) -> SwapResult:
        """Execute simulated buy order."""
        try:
            start_time = datetime.now()
            
            # Get quote from DEX client
            dex_client = list(self.dex_clients.values())[0]  # Use first available DEX
            quote = await dex_client.get_quote(
                input_token="USDC",
                output_token=token_address,
                amount=amount_usd,
                slippage_bps=max_slippage_bps
            )
            
            # Apply simulation effects
            adjusted_result = await self._apply_simulation_effects(quote)
            
            # Create virtual position and update cash balance
            if adjusted_result.status == SwapStatus.CONFIRMED:
                # Deduct cost from virtual portfolio
                total_cost = amount_usd + Decimal(str(adjusted_result.transaction_data["fees"]))
                self.virtual_portfolio.spend_cash(total_cost, f"Buy {token_address}")
                
                # Create position
                await self._create_virtual_position(
                    token_address=token_address,
                    amount=adjusted_result.actual_output_amount,
                    entry_price=Decimal(str(adjusted_result.transaction_data["actual_price"]))
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                "Buy order executed",
                token=token_address,
                amount_usd=str(amount_usd),
                execution_time_ms=execution_time,
                status=adjusted_result.status.value
            )
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error("Error executing buy order", error=str(e))
            return SwapResult(
                transaction_hash="SIMULATED_ERROR",
                status=SwapStatus.FAILED,
                input_token="USDC",
                output_token=token_address,
                input_amount=amount_usd,
                error_message=str(e)
            )
    
    async def execute_sell_order(self, token_address: str, amount: Decimal, max_slippage_bps: int = 10) -> SwapResult:
        """Execute simulated sell order."""
        try:
            start_time = datetime.now()
            
            # Get quote from DEX client
            dex_client = list(self.dex_clients.values())[0]
            quote = await dex_client.get_quote(
                input_token=token_address,
                output_token="USDC",
                amount=amount,
                slippage_bps=max_slippage_bps
            )
            
            # Apply simulation effects
            adjusted_result = await self._apply_simulation_effects(quote)
            
            # Update virtual portfolio
            if adjusted_result.status == SwapStatus.CONFIRMED:
                self.virtual_portfolio.add_cash(
                    adjusted_result.actual_output_amount - adjusted_result.fees,
                    "Token sale"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                "Sell order executed",
                token=token_address,
                amount=str(amount),
                execution_time_ms=execution_time,
                status=adjusted_result.status.value
            )
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error("Error executing sell order", error=str(e))
            return SwapResult(
                transaction_hash="SIMULATED_ERROR",
                status=SwapStatus.FAILED,
                input_token=token_address,
                output_token="USDC",
                input_amount=amount,
                error_message=str(e)
            )
    
    async def execute_batch_orders(self, orders: List[Dict[str, Any]]) -> List[SwapResult]:
        """Execute multiple orders in batch."""
        results = []
        
        for order in orders:
            if order["action"] == "buy":
                result = await self.execute_buy_order(
                    order["token"],
                    order["amount_usd"]
                )
            elif order["action"] == "sell":
                result = await self.execute_sell_order(
                    order["token"],
                    order["amount"]
                )
            else:
                result = SwapResult(
                    transaction_hash="INVALID_ACTION",
                    status=SwapStatus.FAILED,
                    input_token="",
                    output_token="",
                    input_amount=Decimal("0"),
                    error_message=f"Invalid action: {order['action']}"
                )
            
            results.append(result)
        
        return results
    
    async def update_market_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """Update market prices for positions."""
        for token_address, new_price in price_updates.items():
            # Find positions with this token
            for position in self.virtual_portfolio.positions.values():
                if position.symbol.startswith(token_address.split('_')[0]):  # Simple matching
                    self.virtual_portfolio.update_position_price(position.position_id, new_price)
    
    async def _apply_simulation_effects(self, quote: SwapQuote) -> SwapResult:
        """Apply realistic simulation effects like slippage and fees."""
        # Calculate adjusted output with slippage
        slippage_factor = Decimal(1) - (Decimal(self.slippage_bps) / Decimal(10000))
        adjusted_output = quote.output_amount * slippage_factor
        
        # Calculate fees
        fees = Decimal("0")
        if self.enable_fees:
            fees = quote.input_amount * Decimal("0.003")  # 0.3% fee
        
        # Calculate actual price impact
        actual_price_impact_bps = quote.price_impact_bps + (self.slippage_bps // 2)
        
        actual_price = adjusted_output / quote.input_amount if quote.input_amount > 0 else Decimal("0")
        
        result = SwapResult(
            transaction_hash=f"SIMULATED_{uuid4().hex[:8]}",
            status=SwapStatus.CONFIRMED,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            actual_output_amount=adjusted_output,
            timestamp=datetime.now(),
            dex_name=quote.dex_name,
            quote_used=quote,
            actual_price_impact_bps=actual_price_impact_bps,
            transaction_data={
                "is_simulated": True, 
                "fees": fees,
                "actual_price": actual_price
            }
        )
        
        return result
    
    async def _create_virtual_position(self, token_address: str, amount: Decimal, entry_price: Decimal) -> None:
        """Create virtual position in portfolio."""
        position = Position(
            position_id=uuid4(),
            symbol=f"{token_address}/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,  # Default to Solana
            dex_name="jupiter",   # Default to Jupiter
            size=amount,
            entry_price=entry_price,
            current_price=entry_price,
            status=PositionStatus.OPEN
        )
        
        self.virtual_portfolio.add_position(position)


class SimulationMode(ModeBase):
    """
    Comprehensive simulation trading mode implementation.
    
    Provides realistic paper trading with real-time market data,
    virtual portfolio management, and risk controls.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize simulation mode with enhanced simulation capabilities."""
        super().__init__(mode_id, config, portfolio)
        
        # Extract simulation parameters
        params = config.parameters
        self.virtual_balance = Decimal(str(params.get("initial_balance", 10000)))
        self.enable_fees = params.get("enable_fees", True)
        self.slippage_bps = params.get("slippage_bps", 10)
        self.max_drawdown_pct = Decimal(str(params.get("max_drawdown_pct", 0.15)))
        self.enable_risk_management = params.get("enable_risk_management", True)
        self.market_data_frequency_ms = params.get("market_data_frequency_ms", 1000)
        self.enable_real_time_pnl = params.get("enable_real_time_pnl", True)
        
        # Initialize simulation components
        self.virtual_portfolio = VirtualPortfolio(
            initial_balance=self.virtual_balance,
            enable_fees=self.enable_fees
        )
        
        self.dex_clients: Dict[str, DEXBase] = {}
        
        self.simulation_executor = SimulationExecutor(
            virtual_portfolio=self.virtual_portfolio,
            dex_clients=self.dex_clients,
            enable_fees=self.enable_fees,
            slippage_bps=self.slippage_bps
        )
        
        risk_config = {
            "max_position_size_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.4,
            "max_open_positions": 10
        }
        
        self.risk_manager = SimulationRiskManager(
            virtual_portfolio=self.virtual_portfolio,
            config=risk_config
        )
        
        self.market_data_feed = MarketDataFeed(
            dex_clients=self.dex_clients,
            frequency_ms=self.market_data_frequency_ms
        )
        
        # Simulation metrics
        self.simulation_metrics = SimulationMetrics(
            virtual_balance=self.virtual_balance
        )
    
    async def initialize(self) -> None:
        """Initialize simulation mode resources."""
        self.logger.info("Initializing simulation mode")
        
        # Initialize DEX clients (would be injected in real implementation)
        await self._initialize_dex_clients()
        
        # Subscribe to market data updates
        self.market_data_feed.subscribe(self.process_market_update)
        
        self._set_status(ModeStatus.INACTIVE)
        self.logger.info("Simulation mode initialized")
    
    async def start(self) -> None:
        """Start simulation mode."""
        self.logger.info("Starting simulation mode")
        self.start_time = datetime.now()
        
        # Start market data feed
        await self.market_data_feed.start()
        
        # Activate simulation executor
        self.simulation_executor.is_active = True
        
        self._set_status(ModeStatus.ACTIVE)
        self.logger.info("Simulation mode started")
    
    async def stop(self) -> None:
        """Stop simulation mode."""
        self.logger.info("Stopping simulation mode")
        self._set_status(ModeStatus.STOPPING)
        
        # Deactivate executor
        self.simulation_executor.is_active = False
        
        # Stop market data feed
        await self.market_data_feed.stop()
        
        self.logger.info("Simulation mode stopped")
    
    async def pause(self) -> None:
        """Pause simulation mode."""
        self.logger.info("Pausing simulation mode")
        self._set_status(ModeStatus.PAUSED)
        
        # Pause executor but keep market data active
        self.simulation_executor.is_active = False
    
    async def resume(self) -> None:
        """Resume simulation mode."""
        self.logger.info("Resuming simulation mode")
        self._set_status(ModeStatus.ACTIVE)
        
        # Resume executor
        self.simulation_executor.is_active = True
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and make simulated trading decisions."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        try:
            # Record market metrics
            self._record_metric("last_price", market_state.price_usd)
            self._record_metric("last_rsi", market_state.rsi)
            self._record_metric("last_volume", market_state.volume_24h)
            
            # Make trading decision based on market conditions
            action = self._make_trading_decision(market_state)
            
            # Execute simulated trade if action is not HOLD
            if action != TradeAction.HOLD:
                await self._execute_simulated_trade(action, market_state)
            
            # Update simulation metrics
            self._update_simulation_metrics()
            
            return action
            
        except Exception as e:
            self.logger.error("Error processing market tick", error=str(e))
            return None
    
    async def cleanup(self) -> None:
        """Clean up simulation mode resources."""
        self.logger.info("Cleaning up simulation mode")
        
        # Stop market data feed
        if self.market_data_feed.is_active:
            await self.market_data_feed.stop()
        
        # Generate final performance report
        final_performance = self.virtual_portfolio.calculate_performance()
        self.logger.info(
            "Simulation completed",
            total_trades=self.simulation_metrics.total_trades,
            final_balance=str(self.virtual_portfolio.equity),
            total_pnl=str(final_performance.total_pnl),
            win_rate=str(final_performance.win_rate)
        )
    
    async def process_market_update(self, market_data: Dict[str, Any]) -> None:
        """Process real-time market data updates."""
        if not self.enable_real_time_pnl:
            return
        
        try:
            # Extract price updates from market data
            price_updates = {}
            for dex_name, dex_data in market_data.items():
                if isinstance(dex_data, dict):
                    for token, data in dex_data.items():
                        if isinstance(data, dict) and "price" in data:
                            price_updates[token] = Decimal(str(data["price"]))
            
            # Update position prices
            if price_updates:
                await self.simulation_executor.update_market_prices(price_updates)
            
        except Exception as e:
            self.logger.error("Error processing market update", error=str(e))
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        performance = self.virtual_portfolio.calculate_performance()
        risk_metrics = self.virtual_portfolio.calculate_risk_metrics()
        
        return {
            "total_trades": self.simulation_metrics.total_trades,
            "win_rate": float(performance.win_rate),
            "profit_factor": float(performance.profit_factor),
            "total_pnl": float(performance.total_pnl),
            "max_drawdown": float(performance.max_drawdown),
            "sharpe_ratio": float(performance.sharpe_ratio),
            "virtual_balance": float(self.virtual_portfolio.equity),
            "total_fees": float(performance.total_fees),
            "simulation_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
        }
    
    def _make_trading_decision(self, market_state: MarketState) -> TradeAction:
        """Make trading decision based on market state."""
        # Simple RSI-based strategy for simulation
        if market_state.rsi is not None:
            if market_state.rsi < 25:  # Very oversold
                return TradeAction.STRONG_BUY
            elif market_state.rsi < 35:  # Oversold
                return TradeAction.BUY
            elif market_state.rsi > 75:  # Very overbought
                return TradeAction.STRONG_SELL
            elif market_state.rsi > 65:  # Overbought
                return TradeAction.SELL
        
        return TradeAction.HOLD
    
    async def _execute_simulated_trade(self, action: TradeAction, market_state: MarketState) -> None:
        """Execute simulated trade based on action."""
        try:
            if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
                # Simulate buy order
                position_size = Decimal("1000")  # Fixed position size for simulation
                if action == TradeAction.STRONG_BUY:
                    position_size *= Decimal("1.5")
                
                # Check risk limits
                risk_check = await self.risk_manager.validate_position_size("SIMULATION_TOKEN", position_size)
                if risk_check.is_valid:
                    # Execute simulated buy (would use real token address in production)
                    pass  # Placeholder for actual execution
            
            elif action in [TradeAction.SELL, TradeAction.STRONG_SELL]:
                # Simulate sell order for existing positions
                open_positions = [p for p in self.virtual_portfolio.positions.values() if p.status == PositionStatus.OPEN]
                if open_positions:
                    position = open_positions[0]  # Sell first position
                    # Execute simulated sell
                    pass  # Placeholder for actual execution
            
            # Update trade counter
            self.simulation_metrics.total_trades += 1
            self._record_metric("simulated_trades", self.simulation_metrics.total_trades)
            
        except Exception as e:
            self.logger.error("Error executing simulated trade", error=str(e))
    
    def _update_simulation_metrics(self) -> None:
        """Update simulation performance metrics."""
        self.simulation_metrics.virtual_balance = self.virtual_portfolio.equity
        self.simulation_metrics.unrealized_pnl = self.virtual_portfolio.total_unrealized_pnl
        
        # Update metrics in mode base
        self._record_metric("virtual_balance", self.simulation_metrics.virtual_balance)
        self._record_metric("unrealized_pnl", self.simulation_metrics.unrealized_pnl)
        self._record_metric("simulated_action", "processed")
    
    async def _initialize_dex_clients(self) -> None:
        """Initialize DEX clients for market data."""
        # This would initialize real DEX clients in production
        # For now, create mock clients
        self.logger.info("DEX clients would be initialized here")
        
    def get_result(self):
        """Get simulation mode result with enhanced metrics."""
        result = super().get_result()
        
        # Add simulation-specific metadata
        result.metadata.update({
            "virtual_balance": float(self.virtual_portfolio.equity),
            "total_trades": self.simulation_metrics.total_trades,
            "simulation_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "enable_fees": self.enable_fees,
            "slippage_bps": self.slippage_bps
        })
        
        return result