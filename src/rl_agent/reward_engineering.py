"""
Advanced Reward Engineering for RL Trading

Implements sophisticated reward calculation mechanisms that incorporate
risk metrics, portfolio theory, and trading psychology for optimal
RL agent learning.
"""

import numpy as np
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from .base import TradeAction, TradingResult
from src.discovery.base import DiscoveredToken

logger = structlog.get_logger()


@dataclass
class RiskMetrics:
    """Risk metrics for reward calculation"""
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'value_at_risk': self.value_at_risk,
            'expected_shortfall': self.expected_shortfall,
            'calmar_ratio': self.calmar_ratio
        }


@dataclass
class RewardConfig:
    """Configuration for reward engineering"""
    
    # Risk-adjusted return parameters
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    target_return: float = 0.15   # 15% annual target return
    
    # Reward component weights
    return_weight: float = 0.4
    risk_weight: float = 0.3
    consistency_weight: float = 0.2
    efficiency_weight: float = 0.1
    
    # Risk thresholds
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    volatility_threshold: float = 0.25    # 25% annualized volatility
    var_confidence: float = 0.05          # 5% VaR confidence level
    
    # Performance tracking
    lookback_window: int = 252            # 1 year of trading days
    min_observations: int = 30            # Minimum observations for metrics
    
    # Reward scaling
    reward_scale: float = 100.0           # Scale rewards to reasonable range
    penalty_multiplier: float = 2.0      # Penalty severity multiplier


class AdvancedRewardCalculator:
    """Advanced reward calculator with risk-adjusted metrics"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = structlog.get_logger().bind(component="AdvancedRewardCalculator")
        
        # Performance tracking
        self.returns_history = deque(maxlen=config.lookback_window)
        self.portfolio_values = deque(maxlen=config.lookback_window)
        self.drawdown_history = deque(maxlen=config.lookback_window)
        self.trade_results = deque(maxlen=config.lookback_window)
        
        # Risk metrics cache
        self.last_metrics: Optional[RiskMetrics] = None
        self.metrics_timestamp: Optional[datetime] = None
        
        self.logger.info("Advanced reward calculator initialized",
                        return_weight=config.return_weight,
                        risk_weight=config.risk_weight,
                        lookback_window=config.lookback_window)
    
    def calculate_reward(self, 
                        portfolio_value_before: float,
                        portfolio_value_after: float,
                        trading_result: Optional[TradingResult],
                        market_conditions: Dict[str, Any] = None) -> Tuple[float, RiskMetrics]:
        """
        Calculate sophisticated risk-adjusted reward
        
        Args:
            portfolio_value_before: Portfolio value before action
            portfolio_value_after: Portfolio value after action
            trading_result: Result of the trading action
            market_conditions: Current market conditions
            
        Returns:
            Tuple of (reward, risk_metrics)
        """
        
        # Update performance history
        self._update_performance_history(
            portfolio_value_before, 
            portfolio_value_after, 
            trading_result
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Component rewards
        return_reward = self._calculate_return_reward(
            portfolio_value_before, portfolio_value_after
        )
        risk_reward = self._calculate_risk_reward(risk_metrics)
        consistency_reward = self._calculate_consistency_reward()
        efficiency_reward = self._calculate_efficiency_reward(trading_result)
        
        # Market condition adjustments
        market_adjustment = self._calculate_market_adjustment(market_conditions)
        
        # Weighted total reward
        total_reward = (
            self.config.return_weight * return_reward +
            self.config.risk_weight * risk_reward +
            self.config.consistency_weight * consistency_reward +
            self.config.efficiency_weight * efficiency_reward +
            market_adjustment
        )
        
        # Apply scaling
        scaled_reward = total_reward * self.config.reward_scale
        
        self.logger.debug("Reward components calculated",
                         return_reward=return_reward,
                         risk_reward=risk_reward,
                         consistency_reward=consistency_reward,
                         efficiency_reward=efficiency_reward,
                         market_adjustment=market_adjustment,
                         total_reward=scaled_reward)
        
        return float(scaled_reward), risk_metrics
    
    def _update_performance_history(self, 
                                  value_before: float, 
                                  value_after: float,
                                  trading_result: Optional[TradingResult]):
        """Update performance tracking history"""
        
        # Calculate return
        if value_before > 0:
            period_return = (value_after - value_before) / value_before
        else:
            period_return = 0.0
        
        self.returns_history.append(period_return)
        self.portfolio_values.append(value_after)
        
        # Calculate drawdown
        if len(self.portfolio_values) > 0:
            peak_value = max(self.portfolio_values)
            current_drawdown = (peak_value - value_after) / peak_value if peak_value > 0 else 0.0
            self.drawdown_history.append(current_drawdown)
        
        # Store trading result
        if trading_result:
            self.trade_results.append(trading_result)
    
    def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        if len(self.returns_history) < self.config.min_observations:
            return RiskMetrics()
        
        returns = np.array(self.returns_history)
        
        # Basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_return = mean_return * 252 - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
        
        # Calmar ratio
        annual_return = mean_return * 252
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Value at Risk (VaR) and Expected Shortfall (ES)
        var_95 = np.percentile(returns, self.config.var_confidence * 100)
        tail_returns = returns[returns <= var_95]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            value_at_risk=var_95,
            expected_shortfall=expected_shortfall,
            calmar_ratio=calmar_ratio
        )
        
        self.last_metrics = metrics
        self.metrics_timestamp = datetime.now()
        
        return metrics
    
    def _calculate_return_reward(self, value_before: float, value_after: float) -> float:
        """Calculate return-based reward component"""
        
        if value_before <= 0:
            return 0.0
        
        # Period return
        period_return = (value_after - value_before) / value_before
        
        # Annualized return estimate
        if len(self.returns_history) > 0:
            annualized_return = np.mean(self.returns_history) * 252
            
            # Reward based on excess return over target
            excess_return = annualized_return - self.config.target_return
            return_reward = np.tanh(excess_return * 10)  # Smooth reward function
        else:
            return_reward = period_return * 100  # Simple period return
        
        return float(return_reward)
    
    def _calculate_risk_reward(self, metrics: RiskMetrics) -> float:
        """Calculate risk-based reward component"""
        
        if len(self.returns_history) < self.config.min_observations:
            return 0.0
        
        risk_reward = 0.0
        
        # Sharpe ratio reward
        if metrics.sharpe_ratio > 0:
            sharpe_reward = min(metrics.sharpe_ratio / 2.0, 1.0)  # Cap at 1.0
            risk_reward += sharpe_reward * 0.4
        else:
            risk_reward -= abs(metrics.sharpe_ratio) * 0.2
        
        # Drawdown penalty
        if metrics.max_drawdown > self.config.max_drawdown_threshold:
            drawdown_penalty = (metrics.max_drawdown - self.config.max_drawdown_threshold) * 5
            risk_reward -= drawdown_penalty * self.config.penalty_multiplier
        else:
            # Reward for staying within drawdown limits
            risk_reward += (self.config.max_drawdown_threshold - metrics.max_drawdown) * 2
        
        # Volatility penalty/reward
        if metrics.volatility > self.config.volatility_threshold:
            vol_penalty = (metrics.volatility - self.config.volatility_threshold) * 2
            risk_reward -= vol_penalty
        else:
            # Reward for controlled volatility
            risk_reward += (self.config.volatility_threshold - metrics.volatility) * 0.5
        
        # VaR reward (less negative VaR is better)
        var_reward = -metrics.value_at_risk * 10 if metrics.value_at_risk < 0 else 0
        risk_reward += var_reward * 0.2
        
        return float(risk_reward)
    
    def _calculate_consistency_reward(self) -> float:
        """Calculate consistency-based reward component"""
        
        if len(self.returns_history) < self.config.min_observations:
            return 0.0
        
        returns = np.array(self.returns_history)
        
        # Rolling Sharpe ratio stability
        window_size = min(30, len(returns) // 2)
        if len(returns) >= window_size * 2:
            rolling_sharpes = []
            for i in range(window_size, len(returns)):
                window_returns = returns[i-window_size:i]
                window_mean = np.mean(window_returns)
                window_std = np.std(window_returns)
                if window_std > 0:
                    rolling_sharpes.append(window_mean / window_std)
            
            if len(rolling_sharpes) > 1:
                sharpe_stability = 1.0 / (1.0 + np.std(rolling_sharpes))
                consistency_reward = sharpe_stability * 2 - 1  # Scale to [-1, 1]
            else:
                consistency_reward = 0.0
        else:
            consistency_reward = 0.0
        
        # Win rate consistency
        if len(self.trade_results) >= 10:
            successful_trades = sum(1 for result in self.trade_results if result.success)
            win_rate = successful_trades / len(self.trade_results)
            
            # Reward stable win rates around 60-70%
            optimal_win_rate = 0.65
            win_rate_reward = 1.0 - abs(win_rate - optimal_win_rate) * 2
            consistency_reward += win_rate_reward * 0.3
        
        return float(consistency_reward)
    
    def _calculate_efficiency_reward(self, trading_result: Optional[TradingResult]) -> float:
        """Calculate trading efficiency reward"""
        
        efficiency_reward = 0.0
        
        if trading_result:
            # Successful execution reward
            if trading_result.success:
                efficiency_reward += 0.1
                
                # Profit efficiency
                if hasattr(trading_result, 'realized_pnl') and trading_result.realized_pnl:
                    if trading_result.realized_pnl > 0:
                        efficiency_reward += min(trading_result.realized_pnl / 1000.0, 0.5)
                    else:
                        efficiency_reward += max(trading_result.realized_pnl / 1000.0, -0.5)
            else:
                # Penalty for failed trades
                efficiency_reward -= 0.2
        
        # Transaction cost consideration
        if len(self.trade_results) >= 5:
            recent_trades = list(self.trade_results)[-5:]
            avg_trade_value = np.mean([abs(t.value_usd or 0) for t in recent_trades])
            
            # Reward larger, less frequent trades (lower transaction costs)
            if avg_trade_value > 1000:
                efficiency_reward += 0.1
            elif avg_trade_value < 100:
                efficiency_reward -= 0.1
        
        return float(efficiency_reward)
    
    def _calculate_market_adjustment(self, market_conditions: Optional[Dict[str, Any]]) -> float:
        """Calculate market condition adjustments"""
        
        if not market_conditions:
            return 0.0
        
        adjustment = 0.0
        
        # Volatility regime adjustment
        if 'volatility_regime' in market_conditions:
            vol_regime = market_conditions['volatility_regime']
            if vol_regime == 'high':
                adjustment -= 0.1  # Penalize in high volatility
            elif vol_regime == 'low':
                adjustment += 0.1  # Reward in low volatility
        
        # Market trend adjustment
        if 'market_trend' in market_conditions:
            trend = market_conditions['market_trend']
            if trend == 'bull':
                adjustment += 0.05  # Slight boost in bull market
            elif trend == 'bear':
                adjustment -= 0.05  # Slight penalty in bear market
        
        # Fear/Greed index adjustment
        if 'fear_greed_index' in market_conditions:
            fg_index = market_conditions['fear_greed_index']
            if fg_index < 20:  # Extreme fear
                adjustment += 0.15  # Reward contrarian actions
            elif fg_index > 80:  # Extreme greed
                adjustment -= 0.15  # Penalize following the crowd
        
        return float(adjustment)
    
    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get the most recent risk metrics"""
        return self.last_metrics
    
    def reset(self):
        """Reset all tracking data"""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.drawdown_history.clear()
        self.trade_results.clear()
        self.last_metrics = None
        self.metrics_timestamp = None
        
        self.logger.info("Reward calculator reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        if len(self.returns_history) < self.config.min_observations:
            return {
                'status': 'insufficient_data',
                'observations': len(self.returns_history),
                'min_required': self.config.min_observations
            }
        
        returns = np.array(self.returns_history)
        metrics = self.last_metrics or self._calculate_risk_metrics()
        
        return {
            'status': 'active',
            'observations': len(self.returns_history),
            'total_return': float(np.sum(returns)),
            'annualized_return': float(np.mean(returns) * 252),
            'annualized_volatility': float(np.std(returns) * np.sqrt(252)),
            'risk_metrics': metrics.to_dict(),
            'trade_statistics': {
                'total_trades': len(self.trade_results),
                'successful_trades': sum(1 for t in self.trade_results if t.success),
                'win_rate': sum(1 for t in self.trade_results if t.success) / len(self.trade_results) if self.trade_results else 0.0,
                'avg_trade_value': float(np.mean([abs(t.value_usd or 0) for t in self.trade_results])) if self.trade_results else 0.0
            }
        }


def create_reward_calculator(config: Optional[RewardConfig] = None) -> AdvancedRewardCalculator:
    """Factory function to create reward calculator"""
    if config is None:
        config = RewardConfig()
    
    return AdvancedRewardCalculator(config)