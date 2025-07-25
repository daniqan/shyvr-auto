"""
Base classes and data structures for trading modes.

This module defines the core data structures for mode management,
including ModeBase abstract class, ModeType enum, and concrete mode
implementations for analysis, simulation, and live trading.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import structlog

from src.portfolio.base import Portfolio
from src.rl_agent.base import TradeAction, MarketState
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


class ModeType(Enum):
    """Type of trading mode."""
    ANALYSIS = "analysis"           # Analysis-only mode (no trading)
    SIMULATION = "simulation"       # Paper trading simulation
    LIVE_TRADING = "live_trading"   # Live trading with real funds
    BACKTESTING = "backtesting"     # Historical backtesting
    PAPER_TRADING = "paper_trading" # Paper trading with real market data


class ModeStatus(Enum):
    """Status of a trading mode."""
    INACTIVE = "inactive"           # Mode is not running
    ACTIVE = "active"               # Mode is actively running
    PAUSED = "paused"               # Mode is paused
    ERROR = "error"                 # Mode encountered an error
    STOPPING = "stopping"           # Mode is in the process of stopping


@dataclass
class ModeConfig:
    """Configuration for trading modes."""
    mode_type: ModeType
    enabled: bool
    auto_start: bool = False
    max_runtime_minutes: Optional[int] = None
    stop_on_error: bool = True
    log_level: str = "INFO"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_runtime_minutes is not None and self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive")
        
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")


@dataclass
class ModeResult:
    """Result data structure for mode execution."""
    mode_id: UUID
    mode_type: ModeType
    status: ModeStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class ModeBase(ABC):
    """
    Abstract base class for all trading modes.
    
    Defines the common interface and lifecycle management for different
    trading modes (analysis, simulation, live trading, etc.).
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize mode with configuration and portfolio."""
        self.mode_id = mode_id
        self.config = config
        self.portfolio = portfolio
        self.status = ModeStatus.INACTIVE
        self.start_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Configure logger
        self.logger = logger.bind(
            mode_id=str(mode_id),
            mode_type=config.mode_type.value
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize mode-specific resources."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the trading mode."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the trading mode."""
        pass
    
    @abstractmethod
    async def pause(self) -> None:
        """Pause the trading mode."""
        pass
    
    @abstractmethod
    async def resume(self) -> None:
        """Resume the trading mode from paused state."""
        pass
    
    @abstractmethod
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process a market tick and return trading action if applicable."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up mode resources."""
        pass
    
    def get_result(self) -> ModeResult:
        """Get current mode execution result."""
        return ModeResult(
            mode_id=self.mode_id,
            mode_type=self.config.mode_type,
            status=self.status,
            start_time=self.start_time or datetime.now(),
            end_time=datetime.now() if self.status == ModeStatus.INACTIVE else None,
            error_message=self.error_message,
            metrics=self.metrics.copy(),
            metadata=self.metadata.copy()
        )
    
    def _set_status(self, status: ModeStatus, error_message: Optional[str] = None) -> None:
        """Internal method to update mode status."""
        self.status = status
        self.error_message = error_message
        
        self.logger.info(
            "Mode status changed",
            old_status=self.status.value if hasattr(self, '_previous_status') else None,
            new_status=status.value,
            error_message=error_message
        )
    
    def _record_metric(self, key: str, value: Any) -> None:
        """Record a performance metric."""
        self.metrics[key] = value
        self.logger.debug("Metric recorded", metric=key, value=value)


class TradingMode(ModeBase):
    """
    Concrete implementation for live trading mode.
    
    Executes real trades using the RL agent and portfolio management.
    """
    
    async def initialize(self) -> None:
        """Initialize trading mode resources."""
        self.logger.info("Initializing trading mode")
        # Initialize any trading-specific resources
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start live trading mode."""
        self.logger.info("Starting trading mode")
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop live trading mode."""
        self.logger.info("Stopping trading mode")
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause live trading mode."""
        self.logger.info("Pausing trading mode")
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume live trading mode."""
        self.logger.info("Resuming trading mode")
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and execute trades if conditions are met."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # In a real implementation, this would:
        # 1. Analyze market state using ML/RL models
        # 2. Make trading decisions based on portfolio state
        # 3. Execute trades through DEX clients
        # 4. Update portfolio positions
        
        # For now, return a simple action based on RSI
        if market_state.rsi is not None:
            if market_state.rsi < 30:  # Oversold
                return TradeAction.BUY
            elif market_state.rsi > 70:  # Overbought
                return TradeAction.SELL
        
        return TradeAction.HOLD
    
    async def cleanup(self) -> None:
        """Clean up trading mode resources."""
        self.logger.info("Cleaning up trading mode")
        # Close any open connections, save state, etc.


class AnalysisMode(ModeBase):
    """
    Concrete implementation for analysis-only mode.
    
    Performs market analysis without executing trades.
    """
    
    async def initialize(self) -> None:
        """Initialize analysis mode resources."""
        self.logger.info("Initializing analysis mode")
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start analysis mode."""
        self.logger.info("Starting analysis mode")
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop analysis mode."""
        self.logger.info("Stopping analysis mode")
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause analysis mode."""
        self.logger.info("Pausing analysis mode")
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume analysis mode."""
        self.logger.info("Resuming analysis mode")
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick for analysis only (no trading actions)."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # Analysis mode only analyzes, doesn't trade
        # Record analysis metrics
        self._record_metric("last_price", market_state.price_usd)
        self._record_metric("last_rsi", market_state.rsi)
        self._record_metric("last_volume", market_state.volume_24h)
        
        # Return HOLD or None since this is analysis only
        return TradeAction.HOLD
    
    async def cleanup(self) -> None:
        """Clean up analysis mode resources."""
        self.logger.info("Cleaning up analysis mode")


class SimulationMode(ModeBase):
    """
    Concrete implementation for simulation/paper trading mode.
    
    Simulates trading with virtual funds to test strategies.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize simulation mode with virtual portfolio."""
        super().__init__(mode_id, config, portfolio)
        
        # Initialize simulation-specific parameters
        self.virtual_balance = config.parameters.get("initial_balance", 10000)
        self.enable_fees = config.parameters.get("enable_fees", True)
        self.slippage_bps = config.parameters.get("slippage_bps", 10)
    
    async def initialize(self) -> None:
        """Initialize simulation mode resources."""
        self.logger.info("Initializing simulation mode")
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start simulation mode."""
        self.logger.info("Starting simulation mode")
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop simulation mode."""
        self.logger.info("Stopping simulation mode")
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause simulation mode."""
        self.logger.info("Pausing simulation mode")
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume simulation mode."""
        self.logger.info("Resuming simulation mode")
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and simulate trading decisions."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # Simulation mode can make trading decisions
        # This would integrate with RL agent for decision making
        
        # Simple simulation logic based on technical indicators
        action = TradeAction.HOLD
        
        if market_state.rsi is not None:
            if market_state.rsi < 25:  # Very oversold
                action = TradeAction.STRONG_BUY
            elif market_state.rsi < 35:  # Oversold
                action = TradeAction.BUY
            elif market_state.rsi > 75:  # Very overbought
                action = TradeAction.STRONG_SELL
            elif market_state.rsi > 65:  # Overbought
                action = TradeAction.SELL
        
        # Record simulation metrics
        self._record_metric("simulated_action", action.value)
        self._record_metric("virtual_balance", self.virtual_balance)
        
        return action
    
    async def cleanup(self) -> None:
        """Clean up simulation mode resources."""
        self.logger.info("Cleaning up simulation mode")


# Mode Exception Classes
class ModeError(Exception):
    """Base mode error."""
    pass


class ModeNotFoundError(ModeError):
    """Error when mode is not found."""
    pass


class ModeConfigError(ModeError):
    """Error in mode configuration."""
    pass


class InvalidModeTransitionError(ModeError):
    """Error when invalid mode transition is attempted."""
    pass