"""
Trading Environment Simulation

Provides a realistic trading environment for RL agent training with
portfolio management, transaction costs, slippage, and market dynamics.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import structlog

from .base import (
    TradeAction, MarketState, TradingResult,
    RLTrainingError
)
from src.discovery.base import DiscoveredToken

logger = structlog.get_logger()


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment"""
    
    # Portfolio settings
    initial_cash: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio per position
    
    # Trading costs
    transaction_fee: float = 0.001  # 0.1% transaction fee
    slippage_factor: float = 0.002  # 0.2% slippage
    
    # Episode settings
    max_episode_steps: int = 1000
    lookback_window: int = 20  # Number of historical data points for state
    
    # Market simulation
    price_volatility: float = 0.02  # 2% price volatility
    trend_strength: float = 0.001   # Slight upward bias
    
    # Risk management
    max_drawdown_limit: float = 0.2  # 20% max drawdown
    margin_call_threshold: float = 0.05  # 5% remaining cash triggers warning


class Portfolio:
    """Portfolio management for trading simulation"""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}  # token_address -> position info
        self.trade_history: List[TradingResult] = []
        
        # Performance tracking
        self.total_value = initial_cash
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_value = initial_cash
        self.max_drawdown = 0.0
        
        # Current token prices for valuation
        self.current_prices: Dict[str, float] = {}
        
        self.logger = structlog.get_logger().bind(component="Portfolio")
    
    def execute_order(self, action: TradeAction, token: DiscoveredToken, 
                     quantity: float, price: float) -> TradingResult:
        """Execute a trading order"""
        
        timestamp = datetime.now()
        
        if action == TradeAction.BUY or action == TradeAction.STRONG_BUY:
            return self._execute_buy_order(token, quantity, price, timestamp)
        
        elif action == TradeAction.SELL or action == TradeAction.STRONG_SELL:
            return self._execute_sell_order(token, quantity, price, timestamp)
        
        else:  # HOLD
            return TradingResult(
                action=action,
                token=token,
                executed_at=timestamp,
                price=price,
                quantity=0,
                value_usd=0,
                success=True,
                portfolio_value_before=self.total_value,
                portfolio_value_after=self.total_value
            )
    
    def _execute_buy_order(self, token: DiscoveredToken, quantity: float, 
                          price: float, timestamp: datetime) -> TradingResult:
        """Execute a buy order"""
        
        total_cost = quantity * price
        
        # Check if we have enough cash
        if total_cost > self.cash:
            return TradingResult(
                action=TradeAction.BUY,
                token=token,
                executed_at=timestamp,
                price=price,
                quantity=quantity,
                value_usd=total_cost,
                success=False,
                error_message=f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}",
                portfolio_value_before=self.total_value,
                portfolio_value_after=self.total_value
            )
        
        portfolio_before = self.total_value
        
        # Execute the buy
        self.cash -= total_cost
        
        # Update position
        if token.address in self.positions:
            # Add to existing position
            old_quantity = self.positions[token.address]['quantity']
            old_avg_price = self.positions[token.address]['avg_price']
            
            new_quantity = old_quantity + quantity
            new_avg_price = (old_quantity * old_avg_price + quantity * price) / new_quantity
            
            self.positions[token.address].update({
                'quantity': new_quantity,
                'avg_price': new_avg_price,
                'last_update': timestamp
            })
        else:
            # Create new position
            self.positions[token.address] = {
                'token': token,
                'quantity': quantity,
                'avg_price': price,
                'created_at': timestamp,
                'last_update': timestamp
            }
        
        # Update current price
        self.current_prices[token.address] = price
        
        result = TradingResult(
            action=TradeAction.BUY,
            token=token,
            executed_at=timestamp,
            price=price,
            quantity=quantity,
            value_usd=total_cost,
            success=True,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=self.calculate_total_value(),
            cash_change=-total_cost,
            position_change=quantity
        )
        
        self.trade_history.append(result)
        self._update_performance_metrics()
        
        return result
    
    def _execute_sell_order(self, token: DiscoveredToken, quantity: float, 
                           price: float, timestamp: datetime) -> TradingResult:
        """Execute a sell order"""
        
        # Check if we have the position
        if token.address not in self.positions:
            return TradingResult(
                action=TradeAction.SELL,
                token=token,
                executed_at=timestamp,
                price=price,
                quantity=quantity,
                value_usd=quantity * price,
                success=False,
                error_message=f"No position in {token.symbol}",
                portfolio_value_before=self.total_value,
                portfolio_value_after=self.total_value
            )
        
        position = self.positions[token.address]
        available_quantity = position['quantity']
        
        # Check if we have enough tokens
        if quantity > available_quantity:
            return TradingResult(
                action=TradeAction.SELL,
                token=token,
                executed_at=timestamp,
                price=price,
                quantity=quantity,
                value_usd=quantity * price,
                success=False,
                error_message=f"Insufficient tokens: need {quantity}, have {available_quantity}",
                portfolio_value_before=self.total_value,
                portfolio_value_after=self.total_value
            )
        
        portfolio_before = self.total_value
        
        # Execute the sell
        sale_value = quantity * price
        self.cash += sale_value
        
        # Calculate realized PnL
        avg_cost = position['avg_price']
        realized_pnl = (price - avg_cost) * quantity
        self.realized_pnl += realized_pnl
        
        # Update position
        new_quantity = available_quantity - quantity
        if new_quantity <= 0:
            # Close position completely
            del self.positions[token.address]
        else:
            # Reduce position
            self.positions[token.address].update({
                'quantity': new_quantity,
                'last_update': timestamp
            })
        
        # Update current price
        self.current_prices[token.address] = price
        
        result = TradingResult(
            action=TradeAction.SELL,
            token=token,
            executed_at=timestamp,
            price=price,
            quantity=quantity,
            value_usd=sale_value,
            success=True,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=self.calculate_total_value(),
            cash_change=sale_value,
            position_change=-quantity,
            realized_pnl=realized_pnl
        )
        
        self.trade_history.append(result)
        self._update_performance_metrics()
        
        return result
    
    def update_token_price(self, token_address: str, new_price: float):
        """Update the current price for a token"""
        self.current_prices[token_address] = new_price
        self._update_performance_metrics()
    
    def get_position(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get position information for a token"""
        return self.positions.get(token_address)
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL for all positions"""
        unrealized_pnl = 0.0
        
        for token_address, position in self.positions.items():
            current_price = self.current_prices.get(token_address, position['avg_price'])
            quantity = position['quantity']
            avg_price = position['avg_price']
            
            position_unrealized = (current_price - avg_price) * quantity
            unrealized_pnl += position_unrealized
        
        self.unrealized_pnl = unrealized_pnl
        return unrealized_pnl
    
    def calculate_total_value(self) -> float:
        """Calculate total portfolio value"""
        # Cash + current value of all positions
        positions_value = 0.0
        
        for token_address, position in self.positions.items():
            current_price = self.current_prices.get(token_address, position['avg_price'])
            positions_value += position['quantity'] * current_price
        
        self.total_value = self.cash + positions_value
        return self.total_value
    
    def _update_performance_metrics(self):
        """Update portfolio performance metrics"""
        current_value = self.calculate_total_value()
        
        # Update max value and drawdown
        if current_value > self.max_value:
            self.max_value = current_value
        
        drawdown = (self.max_value - current_value) / self.max_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_position_size(self, token_address: str) -> float:
        """Get position size as fraction of total portfolio value"""
        if token_address not in self.positions:
            return 0.0
        
        position = self.positions[token_address]
        current_price = self.current_prices.get(token_address, position['avg_price'])
        position_value = position['quantity'] * current_price
        
        return position_value / self.total_value if self.total_value > 0 else 0.0


class TradingEnvironment:
    """Simulated trading environment for RL training"""
    
    def __init__(self, config: EnvironmentConfig, tokens: List[DiscoveredToken], 
                 historical_data: Dict[str, Dict[str, Any]]):
        self.config = config
        self.tokens = tokens
        self.historical_data = historical_data
        
        # Environment state
        self.current_step = 0
        self.done = False
        self.portfolio = Portfolio(config.initial_cash)
        self.current_token_index = 0
        
        # Market state
        self.current_prices = {}
        self.price_history = {}
        
        # Performance tracking
        self.episode_rewards = []
        self.last_trading_result: Optional[TradingResult] = None
        
        self.logger = structlog.get_logger().bind(component="TradingEnvironment")
        
        # Initialize price data
        self._initialize_price_data()
    
    def _initialize_price_data(self):
        """Initialize price data for all tokens"""
        for token in self.tokens:
            if token.address in self.historical_data:
                self.price_history[token.address] = self.historical_data[token.address]['prices']
                # Set initial price
                if self.price_history[token.address]:
                    self.current_prices[token.address] = self.price_history[token.address][0]
                    self.portfolio.update_token_price(token.address, self.current_prices[token.address])
            else:
                # Use token's default price if no historical data
                self.current_prices[token.address] = token.price_usd or 1.0
                self.portfolio.update_token_price(token.address, self.current_prices[token.address])
    
    def reset(self) -> MarketState:
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.done = False
        self.portfolio = Portfolio(self.config.initial_cash)
        self.episode_rewards = []
        self.last_trading_result = None
        self.current_token_index = 0
        
        # Reset prices to start of historical data
        self._initialize_price_data()
        
        # Return initial market state
        return self._get_market_state()
    
    def step(self, action: TradeAction, token: DiscoveredToken, 
             position_size: float) -> Tuple[MarketState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Trading action to take
            token: Token to trade
            position_size: Size of position as fraction of portfolio (0-1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        
        if self.done:
            raise TradingEnvironmentError("Environment is done, call reset() first")
        
        # Calculate position details
        portfolio_value = self.portfolio.calculate_total_value()
        max_position_value = portfolio_value * self.config.max_position_size
        
        # Current price with slippage
        current_price = self.get_current_price(token.address)
        execution_price = self._apply_slippage(current_price, action, position_size)
        
        # Calculate quantity based on position size
        if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
            # For buy orders, position_size is fraction of portfolio to allocate
            target_value = min(portfolio_value * position_size, max_position_value)
            quantity = target_value / execution_price if execution_price > 0 else 0
        
        elif action in [TradeAction.SELL, TradeAction.STRONG_SELL]:
            # For sell orders, position_size is fraction of position to sell
            current_position = self.portfolio.get_position(token.address)
            if current_position:
                quantity = current_position['quantity'] * position_size
            else:
                quantity = 0
        
        else:  # HOLD
            quantity = 0
        
        # Apply transaction fees
        if quantity > 0:
            fee_rate = self.config.transaction_fee
            execution_price *= (1 + fee_rate) if action in [TradeAction.BUY, TradeAction.STRONG_BUY] else (1 - fee_rate)
        
        # Execute the trade
        portfolio_value_before = self.portfolio.total_value
        trading_result = self.portfolio.execute_order(action, token, quantity, execution_price)
        self.last_trading_result = trading_result
        
        # Update environment state
        self.current_step += 1
        self._update_prices()
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value_before, trading_result)
        self.episode_rewards.append(reward)
        
        # Check termination conditions
        self.done = self._check_done()
        
        # Create next state
        next_state = self._get_market_state()
        
        # Info dictionary
        info = {
            'trading_result': trading_result.to_dict() if trading_result else None,
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': len(self.portfolio.positions),
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'max_drawdown': self.portfolio.max_drawdown,
            'episode_step': self.current_step
        }
        
        return next_state, reward, self.done, info
    
    def get_current_price(self, token_address: str) -> float:
        """Get current price for a token"""
        return self.current_prices.get(token_address, 1.0)
    
    def _apply_slippage(self, price: float, action: TradeAction, position_size: float) -> float:
        """Apply price slippage based on order size"""
        if action == TradeAction.HOLD:
            return price
        
        # Slippage increases with position size
        slippage_factor = self.config.slippage_factor * position_size
        
        if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
            # Buy at higher price (adverse slippage)
            return price * (1 + slippage_factor)
        else:
            # Sell at lower price (adverse slippage)
            return price * (1 - slippage_factor)
    
    def _update_prices(self):
        """Update token prices based on historical data or simulation"""
        for token in self.tokens:
            token_address = token.address
            
            if token_address in self.price_history:
                # Use historical data if available
                if self.current_step < len(self.price_history[token_address]):
                    new_price = self.price_history[token_address][self.current_step]
                    self.current_prices[token_address] = new_price
                    self.portfolio.update_token_price(token_address, new_price)
            else:
                # Simulate price movement
                current_price = self.current_prices[token_address]
                volatility = self.config.price_volatility
                trend = self.config.trend_strength
                
                # Random walk with slight upward bias
                change = np.random.normal(trend, volatility)
                new_price = current_price * (1 + change)
                new_price = max(new_price, 0.01)  # Minimum price floor
                
                self.current_prices[token_address] = new_price
                self.portfolio.update_token_price(token_address, new_price)
    
    def _calculate_reward(self, portfolio_value_before: float, 
                         trading_result: TradingResult) -> float:
        """Calculate reward for the step"""
        
        # Portfolio value change
        current_value = self.portfolio.calculate_total_value()
        value_change = current_value - portfolio_value_before
        
        # Base reward: percentage change in portfolio value
        if portfolio_value_before > 0:
            return_pct = value_change / portfolio_value_before
        else:
            return_pct = 0.0
        
        # Scale to reasonable range
        reward = return_pct * 100  # Convert to percentage points
        
        # Penalty for failed trades
        if trading_result and not trading_result.success:
            reward -= 1.0  # Penalty for failed execution
        
        # Risk-adjusted reward (penalize high drawdown)
        if self.portfolio.max_drawdown > 0.1:  # 10% drawdown threshold
            drawdown_penalty = self.portfolio.max_drawdown * 10
            reward -= drawdown_penalty
        
        # Bonus for successful trades with good returns
        if trading_result and trading_result.success and trading_result.realized_pnl > 0:
            reward += min(trading_result.realized_pnl / 1000.0, 0.5)  # Cap bonus
        
        return float(reward)
    
    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        
        # Max steps reached
        if self.current_step >= self.config.max_episode_steps:
            return True
        
        # Portfolio value fell too low
        current_value = self.portfolio.calculate_total_value()
        if current_value < self.config.initial_cash * 0.1:  # Lost 90% of value
            return True
        
        # Max drawdown exceeded
        if self.portfolio.max_drawdown > self.config.max_drawdown_limit:
            return True
        
        # No more historical data
        for token in self.tokens:
            if token.address in self.price_history:
                if self.current_step >= len(self.price_history[token.address]) - 1:
                    return True
        
        return False
    
    def _get_market_state(self) -> MarketState:
        """Generate current market state"""
        
        # Get current token (rotate through available tokens)
        if self.tokens:
            token = self.tokens[self.current_token_index % len(self.tokens)]
            self.current_token_index += 1
        else:
            raise TradingEnvironmentError("No tokens available")
        
        current_price = self.get_current_price(token.address)
        
        # Calculate technical indicators (simplified)
        rsi = self._calculate_rsi(token.address)
        macd = self._calculate_macd(token.address)
        
        # Portfolio metrics
        portfolio_value = self.portfolio.calculate_total_value()
        cash_balance = self.portfolio.cash
        current_position = self.portfolio.get_position_size(token.address)
        
        # Market context
        portfolio_drawdown = self.portfolio.max_drawdown
        daily_pnl = self.portfolio.realized_pnl + self.portfolio.unrealized_pnl
        
        return MarketState(
            token=token,
            price_usd=current_price,
            price_change_24h=self._calculate_price_change(token.address),
            volume_24h=token.volume_24h or 0,
            market_cap=token.market_cap,
            rsi=rsi,
            macd=macd,
            current_position=current_position,
            portfolio_value=portfolio_value,
            cash_balance=cash_balance,
            portfolio_drawdown=portfolio_drawdown,
            daily_pnl=daily_pnl,
            timestamp=datetime.now()
        )
    
    def _calculate_rsi(self, token_address: str) -> Optional[float]:
        """Calculate simplified RSI"""
        if token_address not in self.price_history:
            return 50.0  # Neutral RSI
        
        prices = self.price_history[token_address]
        if self.current_step < self.config.lookback_window:
            return 50.0
        
        # Get recent prices
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = min(self.current_step + 1, len(prices))
        recent_prices = prices[start_idx:end_idx]
        
        if len(recent_prices) < 2:
            return 50.0
        
        # Calculate price changes
        changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        gains = [change for change in changes if change > 0]
        losses = [-change for change in changes if change < 0]
        
        if len(gains) == 0:
            return 0.0
        if len(losses) == 0:
            return 100.0
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return max(0, min(100, rsi))
    
    def _calculate_macd(self, token_address: str) -> Optional[float]:
        """Calculate simplified MACD"""
        if token_address not in self.price_history:
            return 0.0
        
        prices = self.price_history[token_address]
        if self.current_step < 12:  # Need at least 12 periods
            return 0.0
        
        # Get recent prices
        start_idx = max(0, self.current_step - 26)  # 26 period window
        end_idx = min(self.current_step + 1, len(prices))
        recent_prices = prices[start_idx:end_idx]
        
        if len(recent_prices) < 12:
            return 0.0
        
        # Simple moving averages as proxy for EMA
        if len(recent_prices) >= 12:
            sma_12 = sum(recent_prices[-12:]) / 12
        else:
            sma_12 = recent_prices[-1]
        
        if len(recent_prices) >= 26:
            sma_26 = sum(recent_prices[-26:]) / 26
            macd = sma_12 - sma_26
        else:
            macd = 0.0
        
        return macd
    
    def _calculate_price_change(self, token_address: str) -> float:
        """Calculate 24h price change percentage"""
        if token_address not in self.price_history:
            return 0.0
        
        prices = self.price_history[token_address]
        if self.current_step < 24:  # Need at least 24 periods
            return 0.0
        
        current_price = prices[self.current_step]
        price_24h_ago = prices[max(0, self.current_step - 24)]
        
        if price_24h_ago == 0:
            return 0.0
        
        change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100
        return change_pct


class TradingEnvironmentError(Exception):
    """Raised when trading environment operations fail"""
    pass