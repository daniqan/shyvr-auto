"""
RL Agent Factory
Creates appropriate RL agent instances based on configuration
"""

import structlog
from typing import Union

from .base import AgentConfig, ModelType, RLAgentBase
from .dqn_agent import DQNTradingAgent
from .ppo_agent import PPOTradingAgent

logger = structlog.get_logger()

class AgentFactory:
    """Factory for creating RL agents"""
    
    @staticmethod
    def create_agent(config: AgentConfig, use_enhanced_features: bool = True) -> RLAgentBase:
        """
        Create an RL agent based on the configuration
        
        Args:
            config: Agent configuration
            use_enhanced_features: Whether to use enhanced features
            
        Returns:
            Instantiated agent (DQN or PPO)
        """
        logger.info("Creating RL agent", model_type=config.model_type.value)
        
        if config.model_type in [ModelType.DQN, ModelType.DDQN, ModelType.DUELING_DQN, ModelType.RAINBOW]:
            return DQNTradingAgent(config, use_enhanced_features)
            
        elif config.model_type in [ModelType.PPO, ModelType.TRANSFORMER_PPO]:
            return PPOTradingAgent(config, use_enhanced_features)
            
        else:
            logger.warning("Unknown model type, defaulting to DQN", model_type=config.model_type)
            return DQNTradingAgent(config, use_enhanced_features)
