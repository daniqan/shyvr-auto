"""
Reinforcement Learning Trading Agent Module

This module implements Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)
based reinforcement learning for cryptocurrency trading decisions.
"""

from .base import (
    RLAgentBase, TradeAction, MarketState, TradingResult,
    RewardMetrics, AgentConfig, ModelType as RLModelType
)
from .dqn_agent import DQNTradingAgent, DQNTrainingError
from .ppo_agent import PPOTradingAgent
from .factory import AgentFactory
from .normalization import RunningMeanStd
from .reward_engineering import (
    RiskMetrics, RewardConfig, AdvancedRewardCalculator,
    create_reward_calculator
)
from .training_pipeline import (
    TrainingConfig, TrainingMetrics, DQNTrainingPipeline,
    TrainingPipelineError, HyperparameterSearch
)
from .trading_environment import (
    TradingEnvironment, Portfolio, EnvironmentConfig, TradingEnvironmentError
)
from .experience_replay import (
    ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer, 
    Experience, ReplayBufferConfig, ExperienceReplayError
)

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
    'DQNTrainingError',
    'PPOTradingAgent',
    'AgentFactory',
    'RunningMeanStd',
    
    # Reward engineering
    'RiskMetrics',
    'RewardConfig',
    'AdvancedRewardCalculator',
    'create_reward_calculator',
    
    # Training pipeline
    'TrainingConfig',
    'TrainingMetrics',
    'DQNTrainingPipeline',
    'TrainingPipelineError',
    'HyperparameterSearch',
    
    # Trading environment
    'TradingEnvironment',
    'Portfolio',
    'EnvironmentConfig',
    'TradingEnvironmentError',
    
    # Experience replay
    'ExperienceReplayBuffer',
    'PrioritizedExperienceReplayBuffer',
    'Experience',
    'ReplayBufferConfig',
    'ExperienceReplayError'
]