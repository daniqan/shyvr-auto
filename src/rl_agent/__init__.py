"""
Reinforcement Learning Trading Agent Module

This module implements Deep Q-Network (DQN) based reinforcement learning
for cryptocurrency trading decisions with risk-adjusted reward optimization.
"""

from .base import (
    RLAgentBase, TradeAction, MarketState, TradingResult,
    RewardMetrics, AgentConfig, ModelType as RLModelType
)
from .dqn_agent import DQNTradingAgent, DQNTrainingError

__all__ = [
    # Base classes and enums
    'RLAgentBase',
    'TradeAction', 
    'MarketState',
    'TradingResult',
    'RewardMetrics',
    'AgentConfig',
    'RLModelType',
    
    # Implementation classes
    'DQNTradingAgent',
    'DQNTrainingError'
]