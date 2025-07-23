"""
Base classes and data structures for RL trading agent
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch

from src.discovery.base import DiscoveredToken
from src.ml_analysis.base import PredictionResult


class TradeAction(Enum):
    """Trading actions the RL agent can take"""
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ModelType(Enum):
    """Types of RL models available"""
    DQN = "dqn"
    DDQN = "double_dqn"  # Double DQN
    DUELING_DQN = "dueling_dqn"
    RAINBOW = "rainbow"  # Rainbow DQN with all improvements


@dataclass
class MarketState:
    """Represents the current market state for RL decision making"""
    token: DiscoveredToken
    price_usd: float
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float] = None
    
    # Technical indicators from ML analysis
    rsi: Optional[float] = None
    macd: Optional[float] = None
    sma_20: Optional[float] = None
    ema_12: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    
    # ML predictions
    ml_prediction: Optional[PredictionResult] = None
    prediction_confidence: Optional[float] = None
    
    # Portfolio context
    current_position: float = 0.0  # Current position size (-1 to 1)
    portfolio_value: float = 10000.0  # Total portfolio value in USD
    cash_balance: float = 10000.0  # Available cash
    
    # Risk metrics
    portfolio_drawdown: float = 0.0
    daily_pnl: float = 0.0
    sharpe_ratio: Optional[float] = None
    
    # Market context
    market_volatility: float = 0.0
    fear_greed_index: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """Convert market state to feature vector for neural network"""
        features = [
            # Price features (normalized)
            np.log1p(self.price_usd) if self.price_usd > 0 else 0.0,
            self.price_change_24h / 100.0,  # Convert percentage to decimal
            np.log1p(self.volume_24h) if self.volume_24h > 0 else 0.0,
            np.log1p(self.market_cap) if self.market_cap else 0.0,
            
            # Technical indicators (already normalized 0-100 mostly)
            (self.rsi / 100.0) if self.rsi is not None else 0.5,
            np.tanh(self.macd / 100.0) if self.macd is not None else 0.0,
            np.log1p(self.sma_20) if self.sma_20 else 0.0,
            np.log1p(self.ema_12) if self.ema_12 else 0.0,
            
            # Bollinger bands (relative position)
            self._bollinger_position(),
            
            # Position and portfolio features
            self.current_position,  # Already normalized -1 to 1
            np.log1p(self.portfolio_value) / 10.0,  # Scale down
            np.log1p(self.cash_balance) / 10.0,
            
            # Risk metrics
            np.tanh(self.portfolio_drawdown),  # Limit extreme values
            np.tanh(self.daily_pnl / 1000.0),  # Scale to reasonable range
            self.sharpe_ratio / 3.0 if self.sharpe_ratio else 0.0,  # Cap at reasonable range
            
            # Market context
            self.market_volatility,
            (self.fear_greed_index / 100.0) if self.fear_greed_index else 0.5,
            
            # ML prediction features
            self.prediction_confidence if self.prediction_confidence else 0.0,
            self._ml_direction_encoding()
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _bollinger_position(self) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        if not all([self.bollinger_upper, self.bollinger_lower, self.price_usd]):
            return 0.5
        
        if self.bollinger_upper <= self.bollinger_lower:
            return 0.5
            
        position = (self.price_usd - self.bollinger_lower) / (self.bollinger_upper - self.bollinger_lower)
        return np.clip(position, 0.0, 1.0)
    
    def _ml_direction_encoding(self) -> float:
        """Encode ML prediction direction as float"""
        if not self.ml_prediction:
            return 0.0
        
        direction_map = {
            'strong_sell': -1.0,
            'sell': -0.5,
            'hold': 0.0,
            'buy': 0.5,
            'strong_buy': 1.0
        }
        
        return direction_map.get(self.ml_prediction.direction.value, 0.0)
    
    @classmethod
    def get_feature_size(cls) -> int:
        """Get the size of the feature vector"""
        return 19  # Number of features in to_vector()


@dataclass
class TradingResult:
    """Result of a trading action"""
    action: TradeAction
    token: DiscoveredToken
    executed_at: datetime
    
    # Trade details
    price: float
    quantity: float
    value_usd: float
    
    # Execution details
    success: bool = True
    slippage: float = 0.0
    fees: float = 0.0
    error_message: Optional[str] = None
    
    # Portfolio impact
    portfolio_value_before: float = 0.0
    portfolio_value_after: float = 0.0
    cash_change: float = 0.0
    position_change: float = 0.0
    
    # Performance metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action': self.action.value,
            'token_address': self.token.address,
            'token_symbol': self.token.symbol,
            'executed_at': self.executed_at.isoformat(),
            'price': self.price,
            'quantity': self.quantity,
            'value_usd': self.value_usd,
            'success': self.success,
            'slippage': self.slippage,
            'fees': self.fees,
            'error_message': self.error_message,
            'portfolio_value_before': self.portfolio_value_before,
            'portfolio_value_after': self.portfolio_value_after,
            'cash_change': self.cash_change,
            'position_change': self.position_change,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }


@dataclass
class RewardMetrics:
    """Metrics used for reward calculation in RL training"""
    
    # Returns
    absolute_return: float = 0.0
    relative_return: float = 0.0
    risk_adjusted_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trade quality
    win_rate: float = 0.0
    profit_factor: float = 1.0  # Gross profit / Gross loss
    average_trade_return: float = 0.0
    
    # Risk penalties
    drawdown_penalty: float = 0.0
    volatility_penalty: float = 0.0
    concentration_penalty: float = 0.0
    
    # Final reward
    total_reward: float = 0.0
    
    def calculate_total_reward(self, config: Optional[Dict] = None) -> float:
        """Calculate total reward from individual components"""
        if not config:
            config = {}
        
        # Weights for different reward components
        return_weight = config.get('return_weight', 0.4)
        sharpe_weight = config.get('sharpe_weight', 0.3)
        drawdown_weight = config.get('drawdown_weight', -0.2)
        win_rate_weight = config.get('win_rate_weight', 0.1)
        
        self.total_reward = (
            self.risk_adjusted_return * return_weight +
            self.sharpe_ratio * sharpe_weight +
            self.max_drawdown * drawdown_weight +  # Negative because penalty
            self.win_rate * win_rate_weight
        )
        
        return self.total_reward


@dataclass 
class AgentConfig:
    """Configuration for RL trading agent"""
    
    # Model architecture
    model_type: ModelType = ModelType.DQN
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_frequency: int = 100
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    
    # Reward configuration
    reward_config: Dict[str, float] = field(default_factory=lambda: {
        'return_weight': 0.4,
        'sharpe_weight': 0.3,
        'drawdown_weight': -0.2,
        'win_rate_weight': 0.1
    })
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    
    # Training settings
    episodes: int = 1000
    steps_per_episode: int = 100
    validation_frequency: int = 50
    
    # Performance targets
    target_sharpe_ratio: float = 1.5
    target_win_rate: float = 0.6
    target_max_drawdown: float = 0.15


class RLAgentBase(ABC):
    """Abstract base class for reinforcement learning trading agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.is_trained = False
        self.training_episodes = 0
        self.performance_metrics = RewardMetrics()
        
    @abstractmethod
    async def predict_action(self, state: MarketState) -> Tuple[TradeAction, float]:
        """
        Predict the best trading action for given market state
        
        Args:
            state: Current market state
            
        Returns:
            Tuple of (action, confidence_score)
        """
        pass
    
    @abstractmethod
    async def train_step(self, batch_experiences: List[Dict]) -> Dict[str, float]:
        """
        Train the agent on a batch of experiences
        
        Args:
            batch_experiences: List of experience dictionaries
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        pass
    
    def update_performance_metrics(self, trading_results: List[TradingResult]) -> RewardMetrics:
        """Update performance metrics from trading results"""
        if not trading_results:
            return self.performance_metrics
            
        # Calculate returns
        total_return = sum(r.realized_pnl for r in trading_results)
        portfolio_values = [r.portfolio_value_after for r in trading_results if r.success]
        
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            self.performance_metrics.absolute_return = total_return
            self.performance_metrics.relative_return = returns[-1] if len(returns) > 0 else 0.0
            self.performance_metrics.volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            if self.performance_metrics.volatility > 0:
                self.performance_metrics.sharpe_ratio = np.mean(returns) / self.performance_metrics.volatility
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (peak - portfolio_values) / peak
            self.performance_metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Calculate win rate
        successful_trades = [r for r in trading_results if r.success and r.realized_pnl != 0]
        if successful_trades:
            winning_trades = [r for r in successful_trades if r.realized_pnl > 0]
            self.performance_metrics.win_rate = len(winning_trades) / len(successful_trades)
        
        return self.performance_metrics
    
    def meets_performance_targets(self) -> bool:
        """Check if agent meets performance targets"""
        return (
            self.performance_metrics.sharpe_ratio >= self.config.target_sharpe_ratio and
            self.performance_metrics.win_rate >= self.config.target_win_rate and
            self.performance_metrics.max_drawdown <= self.config.target_max_drawdown
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health and readiness"""
        return {
            'is_trained': self.is_trained,
            'training_episodes': self.training_episodes,
            'meets_targets': self.meets_performance_targets(),
            'performance_metrics': {
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                'win_rate': self.performance_metrics.win_rate,
                'max_drawdown': self.performance_metrics.max_drawdown,
                'total_reward': self.performance_metrics.total_reward
            },
            'config': {
                'model_type': self.config.model_type.value,
                'max_position_size': self.config.max_position_size,
                'max_daily_loss': self.config.max_daily_loss
            }
        }


class RLTrainingError(Exception):
    """Raised when RL training fails"""
    pass


class RLPredictionError(Exception):
    """Raised when RL prediction fails"""
    pass


class RLModelError(Exception):
    """Raised when RL model operations fail"""
    pass