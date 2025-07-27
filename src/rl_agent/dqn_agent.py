"""
Deep Q-Network (DQN) Trading Agent Implementation

Implements a DQN-based reinforcement learning agent for cryptocurrency trading
with experience replay, target networks, and epsilon-greedy exploration.
"""

import asyncio
import math
import random
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import structlog

from .base import (
    RLAgentBase, TradeAction, MarketState, AgentConfig,
    RLTrainingError, RLPredictionError, RLModelError
)
from src.logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)

logger = structlog.get_logger()


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading action value estimation"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, output_size: int):
        super(DQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class DQNTradingAgent(RLAgentBase):
    """DQN-based trading agent with experience replay and target networks"""
    
    def __init__(self, config: AgentConfig, use_enhanced_features: bool = False):
        super().__init__(config)
        
        self.logger = structlog.get_logger().bind(agent=self.__class__.__name__)
        
        # Network architecture
        self.use_enhanced_features = use_enhanced_features
        if use_enhanced_features:
            self.input_size = MarketState.get_enhanced_feature_size()
        else:
            self.input_size = MarketState.get_feature_size()
        self.output_size = len(TradeAction)
        
        # Create main and target networks
        self.q_network = DQNNetwork(
            input_size=self.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_size=self.output_size
        )
        
        self.target_network = DQNNetwork(
            input_size=self.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_size=self.output_size
        )
        
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config.learning_rate
        )
        
        # Training state
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        # Action mapping
        self.action_list = list(TradeAction)
        self.action_to_idx = {action: i for i, action in enumerate(self.action_list)}
        
        self.logger.info("DQN Trading Agent initialized", 
                        input_size=self.input_size,
                        output_size=self.output_size,
                        hidden_size=config.hidden_size,
                        num_layers=config.num_layers)
    
    async def predict_action(self, state: MarketState) -> Tuple[TradeAction, float]:
        """
        Predict the best trading action for given market state
        
        Args:
            state: Current market state
            
        Returns:
            Tuple of (action, confidence_score)
        """
        async with performance_tracker(
            source="dqn_agent",
            operation="predict_action",
            category=ActivityCategory.ML_RL
        ) as tracker:
            try:
                # Convert state to tensor
                state_tensor = self._state_to_tensor(state)
                
                # Epsilon-greedy action selection
                is_exploration = random.random() <= self.epsilon
                
                if not is_exploration:
                    # Exploitation: use network to select action
                    with torch.no_grad():
                        q_values = self.q_network(state_tensor)
                        action_idx = q_values.argmax().item()
                        confidence = torch.softmax(q_values, dim=1).max().item()
                else:
                    # Exploration: random action
                    action_idx = random.randrange(self.output_size)
                    confidence = 1.0 / self.output_size  # Uniform confidence for random action
                
                action = self._index_to_action(action_idx)
                
                # Log action prediction
                await activity_logger.log_activity(
                    category=ActivityCategory.ML_RL,
                    action=ActivityAction.EXECUTE,
                    source="dqn_agent",
                    event_type="action_predicted",
                    title=f"DQN agent predicted action: {action.value}",
                    severity=ActivitySeverity.DEBUG,
                    metadata={
                        "predicted_action": action.value,
                        "confidence": confidence,
                        "epsilon": self.epsilon,
                        "is_exploration": is_exploration,
                        "q_values_max": float(q_values.max().item()) if not is_exploration else None,
                        "episode": getattr(self, 'current_episode', 0),
                        "steps_done": self.steps_done
                    }
                )
                
                # Update epsilon for next prediction
                self._update_epsilon()
                
                self.logger.debug("Action predicted",
                                action=action.value,
                                confidence=confidence,
                                epsilon=self.epsilon,
                                exploration=is_exploration)
                
                return action, confidence
                
            except Exception as e:
                # Log prediction failure
                await activity_logger.log_error(
                    category=ActivityCategory.ML_RL,
                    source="dqn_agent",
                    event_type="action_prediction_failed",
                    title="DQN agent failed to predict action",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    metadata={
                        "epsilon": self.epsilon,
                        "steps_done": self.steps_done,
                        "episode": getattr(self, 'current_episode', 0)
                    }
                )
                
                self.logger.error("Action prediction failed", error=str(e))
                raise RLPredictionError(f"Failed to predict action: {str(e)}")
    
    async def train_step(self, batch_experiences: List[Dict]) -> Dict[str, float]:
        """
        Train the agent on a batch of experiences
        
        Args:
            batch_experiences: List of experience dictionaries
            
        Returns:
            Training metrics dictionary
        """
        try:
            if len(batch_experiences) < self.config.batch_size:
                raise DQNTrainingError(
                    f"Insufficient experiences: got {len(batch_experiences)}, "
                    f"need {self.config.batch_size}"
                )
            
            # Convert experiences to tensors
            states = torch.stack([
                torch.FloatTensor(exp['state']) for exp in batch_experiences
            ])
            actions = torch.LongTensor([exp['action'] for exp in batch_experiences])
            rewards = torch.FloatTensor([exp['reward'] for exp in batch_experiences])
            next_states = torch.stack([
                torch.FloatTensor(exp['next_state']) for exp in batch_experiences
            ])
            dones = torch.BoolTensor([exp['done'] for exp in batch_experiences])
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                if self.config.model_type.value in ['double_dqn', 'dueling_dqn', 'rainbow']:
                    # Double DQN: use main network to select actions, target network to evaluate
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
                else:
                    # Standard DQN: use target network for both selection and evaluation
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                
                # Target Q values
                target_q_values = rewards.unsqueeze(1) + (
                    0.99 * next_q_values * (~dones).float().unsqueeze(1)
                )
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Update training state
            self.training_episodes += 1
            
            # Calculate metrics
            metrics = {
                'loss': loss.item(),
                'q_value_mean': current_q_values.mean().item(),
                'target_q_value_mean': target_q_values.mean().item(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done
            }
            
            self.logger.debug("Training step completed", **metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Training step failed", error=str(e))
            raise DQNTrainingError(f"Training step failed: {str(e)}")
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("Target network updated")
    
    def get_q_values(self, state: MarketState) -> torch.Tensor:
        """Get Q-values for all actions given a state"""
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.squeeze(0)
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        try:
            checkpoint = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__,
                'training_episodes': self.training_episodes,
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'performance_metrics': self.performance_metrics.__dict__
            }
            
            torch.save(checkpoint, filepath)
            
            self.logger.info("Model saved successfully", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save model", filepath=filepath, error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Load network states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.training_episodes = checkpoint.get('training_episodes', 0)
            self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
            self.steps_done = checkpoint.get('steps_done', 0)
            
            # Load performance metrics if available
            if 'performance_metrics' in checkpoint:
                metrics_dict = checkpoint['performance_metrics']
                for key, value in metrics_dict.items():
                    setattr(self.performance_metrics, key, value)
            
            self.is_trained = True
            
            self.logger.info("Model loaded successfully", 
                           filepath=filepath,
                           training_episodes=self.training_episodes,
                           epsilon=self.epsilon)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load model", filepath=filepath, error=str(e))
            return False
    
    def _state_to_tensor(self, state: MarketState) -> torch.Tensor:
        """Convert market state to tensor"""
        if self.use_enhanced_features and hasattr(state, 'to_feature_vector'):
            # Use enhanced feature vector if available
            state_vector = state.to_feature_vector()
        else:
            # Use standard feature vector
            state_vector = state.to_vector()
        return torch.FloatTensor(state_vector).unsqueeze(0)
    
    def _action_to_index(self, action: TradeAction) -> int:
        """Convert action to network output index"""
        return self.action_to_idx[action]
    
    def _index_to_action(self, index: int) -> TradeAction:
        """Convert network output index to action"""
        return self.action_list[index]
    
    def _update_epsilon(self):
        """Update epsilon for epsilon-greedy exploration"""
        self.steps_done += 1
        
        # Exponential decay
        self.epsilon = self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.config.epsilon_decay)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health and readiness"""
        base_health = await super().health_check()
        
        # Add DQN-specific health info
        dqn_health = {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'optimizer_state': len(self.optimizer.state_dict()),
            'model_architecture': {
                'input_size': self.input_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'output_size': self.output_size
            }
        }
        
        # Merge with base health check
        base_health.update(dqn_health)
        
        return base_health


class DQNTrainingError(RLTrainingError):
    """Raised when DQN training fails"""
    pass


class DQNPredictionError(RLPredictionError):
    """Raised when DQN prediction fails"""
    pass


class DQNModelError(RLModelError):
    """Raised when DQN model operations fail"""
    pass