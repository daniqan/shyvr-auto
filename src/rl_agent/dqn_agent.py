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
import pickle

from .base import (
    RLAgentBase, TradeAction, MarketState, AgentConfig, ModelType,
    RLTrainingError, RLPredictionError, RLModelError
)
from src.activity_logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)

# Import preservation components
try:
    from src.model_preservation.manager import PreservationManager, PreservationConfig
    from src.model_preservation.base import PreservationError, PreservationPriority
    PRESERVATION_AVAILABLE = True
except ImportError:
    PRESERVATION_AVAILABLE = False
    PreservationManager = None
    PreservationConfig = None
    PreservationError = Exception
    PreservationPriority = None

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


class DuelingDQNNetwork(nn.Module):
    """Dueling Deep Q-Network with separate value and advantage streams"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, output_size: int):
        super(DuelingDQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        
        # Shared feature extraction layers
        feature_layers = []
        
        # Input layer
        feature_layers.append(nn.Linear(input_size, hidden_size))
        feature_layers.append(nn.ReLU())
        if dropout > 0:
            feature_layers.append(nn.Dropout(dropout))
        
        # Hidden layers (leave one layer for the dueling heads)
        for _ in range(num_layers - 2):
            feature_layers.append(nn.Linear(hidden_size, hidden_size))
            feature_layers.append(nn.ReLU())
            if dropout > 0:
                feature_layers.append(nn.Dropout(dropout))
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream - outputs single scalar value V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream - outputs advantage for each action A(s,a)
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network"""
        # Extract shared features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_head(features)  # (batch_size, 1)
        advantage = self.advantage_head(features)  # (batch_size, num_actions)
        
        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # This ensures that the advantage has zero mean
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class NoisyLinear(nn.Module):
    """Noisy linear layer for Rainbow DQN exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters for weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        # Learnable parameters for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize the learnable parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new noise for exploration"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate noise with factorized gaussian noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class RainbowDQNNetwork(nn.Module):
    """Rainbow DQN network with dueling architecture, noisy networks, and distributional RL"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, output_size: int, num_atoms: int = 51, 
                 noisy: bool = True, v_min: float = -10.0, v_max: float = 10.0):
        super(RainbowDQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        self.num_atoms = num_atoms
        self.noisy = noisy
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Shared feature extraction layers
        feature_layers = []
        
        # Input layer
        if noisy:
            feature_layers.append(NoisyLinear(input_size, hidden_size))
        else:
            feature_layers.append(nn.Linear(input_size, hidden_size))
        feature_layers.append(nn.ReLU())
        if dropout > 0:
            feature_layers.append(nn.Dropout(dropout))
        
        # Hidden layers (leave one layer for the dueling heads)
        for _ in range(num_layers - 2):
            if noisy:
                feature_layers.append(NoisyLinear(hidden_size, hidden_size))
            else:
                feature_layers.append(nn.Linear(hidden_size, hidden_size))
            feature_layers.append(nn.ReLU())
            if dropout > 0:
                feature_layers.append(nn.Dropout(dropout))
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream - outputs distribution over atoms for V(s)
        value_layers = []
        if noisy:
            value_layers.append(NoisyLinear(hidden_size, hidden_size // 2))
        else:
            value_layers.append(nn.Linear(hidden_size, hidden_size // 2))
        value_layers.append(nn.ReLU())
        if dropout > 0:
            value_layers.append(nn.Dropout(dropout))
        if noisy:
            value_layers.append(NoisyLinear(hidden_size // 2, num_atoms))
        else:
            value_layers.append(nn.Linear(hidden_size // 2, num_atoms))
        
        self.value_head = nn.Sequential(*value_layers)
        
        # Advantage stream - outputs distribution over atoms for each action A(s,a)
        advantage_layers = []
        if noisy:
            advantage_layers.append(NoisyLinear(hidden_size, hidden_size // 2))
        else:
            advantage_layers.append(nn.Linear(hidden_size, hidden_size // 2))
        advantage_layers.append(nn.ReLU())
        if dropout > 0:
            advantage_layers.append(nn.Dropout(dropout))
        if noisy:
            advantage_layers.append(NoisyLinear(hidden_size // 2, output_size * num_atoms))
        else:
            advantage_layers.append(nn.Linear(hidden_size // 2, output_size * num_atoms))
        
        self.advantage_head = nn.Sequential(*advantage_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the rainbow network"""
        batch_size = x.size(0)
        
        # Extract shared features
        features = self.feature_layer(x)
        
        # Compute value and advantage distributions
        value = self.value_head(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_head(features).view(batch_size, self.output_size, self.num_atoms)
        
        # Dueling aggregation for distributional case
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_atoms
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values by taking expectation over the distribution"""
        q_atoms = self(x)  # (batch_size, num_actions, num_atoms)
        q_probs = torch.softmax(q_atoms, dim=2)
        q_values = (q_probs * self.support.view(1, 1, -1)).sum(dim=2)
        return q_values


class DQNTradingAgent(RLAgentBase):
    """DQN-based trading agent with experience replay and target networks"""
    
    def __init__(self, config: AgentConfig, use_enhanced_features: bool = False):
        super().__init__(config)
        
        self.logger = structlog.get_logger().bind(agent=self.__class__.__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Network architecture
        self.use_enhanced_features = use_enhanced_features
        if use_enhanced_features:
            self.input_size = MarketState.get_enhanced_feature_size()
        else:
            self.input_size = MarketState.get_feature_size()
        self.output_size = len(TradeAction)
        
        # Create main and target networks based on model type
        self.q_network = self._create_network()
        self.target_network = self._create_network()
        
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
        
        # Current operational mode
        self._current_mode = 'training'
        
        # Initialize preservation if available and configured
        self._preservation_manager = None
        self._preservation_enabled = False
        if PRESERVATION_AVAILABLE and hasattr(config, 'preservation') and config.preservation.get('enabled', False):
            try:
                pres_config = PreservationConfig(
                    gcs_bucket=config.preservation.get('gcs_bucket', 'shyvr-models-prod'),
                    backup_interval_hours=config.preservation.get('backup_interval_hours', 3),
                    max_versions_per_model=config.preservation.get('max_versions_per_model', 20),
                    enable_compression=config.preservation.get('enable_compression', True),
                    mode_isolation=config.preservation.get('mode_isolation', True)
                )
                self._preservation_manager = PreservationManager(pres_config)
                self._preservation_enabled = True
                self.logger.info("Agent preservation enabled", bucket=pres_config.gcs_bucket)
            except Exception as e:
                self.logger.warning("Failed to initialize preservation manager", error=str(e))
        
        self.logger.info("DQN Trading Agent initialized", 
                        input_size=self.input_size,
                        output_size=self.output_size,
                        hidden_size=config.hidden_size,
                        num_layers=config.num_layers,
                        model_type=config.model_type.value,
                        preservation_enabled=self._preservation_enabled)
    
    def _validate_config(self):
        """Validate configuration for different model types"""
        if self.config.model_type == ModelType.RAINBOW:
            # Rainbow DQN specific validation
            if not hasattr(self.config, 'num_atoms') or self.config.num_atoms < 1:
                raise ValueError("Rainbow DQN requires num_atoms >= 1")
            if not hasattr(self.config, 'v_min') or not hasattr(self.config, 'v_max'):
                raise ValueError("Rainbow DQN requires v_min and v_max to be set")
            if self.config.v_min >= self.config.v_max:
                raise ValueError("Rainbow DQN requires v_min < v_max")
        
        # General validation
        if self.config.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.config.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
    
    def _create_network(self) -> nn.Module:
        """Create network based on model type configuration"""
        if self.config.model_type == ModelType.DUELING_DQN:
            return DuelingDQNNetwork(
                input_size=self.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                output_size=self.output_size
            )
        elif self.config.model_type == ModelType.RAINBOW:
            return RainbowDQNNetwork(
                input_size=self.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                output_size=self.output_size,
                num_atoms=getattr(self.config, 'num_atoms', 51),
                noisy=getattr(self.config, 'noisy_networks', True),
                v_min=getattr(self.config, 'v_min', -10.0),
                v_max=getattr(self.config, 'v_max', 10.0)
            )
        else:
            # Default to standard DQN for DQN and DDQN (Double DQN uses same architecture)
            return DQNNetwork(
                input_size=self.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                output_size=self.output_size
            )
    
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
                        if self.config.model_type == ModelType.RAINBOW:
                            # Rainbow DQN: get Q-values from distributional output
                            q_values = self.q_network.get_q_values(state_tensor)
                            # Reset noise for next prediction
                            if hasattr(self.q_network, 'reset_noise'):
                                self.q_network.reset_noise()
                        else:
                            # Standard DQN or Dueling DQN
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
            Training metrics dictionary including TD errors for prioritized replay
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
            
            # Extract importance sampling weights and buffer indices (if using prioritized replay)
            weights = None
            buffer_indices = None
            if 'weight' in batch_experiences[0]:
                weights = torch.FloatTensor([exp['weight'] for exp in batch_experiences])
            if 'index' in batch_experiences[0]:
                buffer_indices = [exp['index'] for exp in batch_experiences]
            
            # Training logic depends on model type
            if self.config.model_type == ModelType.RAINBOW:
                # Distributional RL training for Rainbow DQN
                loss, td_errors = self._train_distributional(
                    states, actions, rewards, next_states, dones, weights
                )
            else:
                # Standard Q-learning for DQN, Double DQN, and Dueling DQN
                loss, td_errors = self._train_standard(
                    states, actions, rewards, next_states, dones, weights
                )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Update training state
            self.training_episodes += 1
            
            # Calculate metrics - handle different output types
            metrics = {
                'loss': loss.item(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done
            }
            
            # Add Q-value metrics if available (not for distributional case)
            if self.config.model_type != ModelType.RAINBOW:
                # For standard and dueling DQN, we have current_q_values and target_q_values from _train_standard
                # Need to recalculate since they're local to _train_standard
                with torch.no_grad():
                    temp_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                    metrics['q_value_mean'] = temp_q_values.mean().item()
            
            # Add TD errors and indices for prioritized replay buffer updates
            if buffer_indices is not None:
                metrics['td_errors'] = td_errors.detach().cpu().numpy().tolist()
                metrics['indices'] = buffer_indices
            
            self.logger.debug("Training step completed", **metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Training step failed", error=str(e))
            raise DQNTrainingError(f"Training step failed: {str(e)}")
    
    def _train_standard(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: torch.Tensor, 
                       dones: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard Q-learning training for DQN, Double DQN, and Dueling DQN"""
        
        # Current Q values
        if self.config.model_type == ModelType.DUELING_DQN:
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        else:
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            if self.config.model_type.value in ['double_dqn', 'dueling_dqn', 'rainbow']:
                # Double DQN: use main network to select actions, target network to evaluate
                if self.config.model_type == ModelType.DUELING_DQN:
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
                else:
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # Target Q values
            target_q_values = rewards.unsqueeze(1) + (
                0.99 * next_q_values * (~dones).float().unsqueeze(1)
            )
        
        # Calculate TD errors (before applying importance sampling weights)
        td_errors = torch.abs(current_q_values - target_q_values).squeeze(1)
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            # Weighted loss for prioritized experience replay
            loss = (weights * (current_q_values.squeeze(1) - target_q_values.squeeze(1)) ** 2).mean()
        else:
            # Standard MSE loss
            loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss, td_errors
    
    def _train_distributional(self, states: torch.Tensor, actions: torch.Tensor,
                             rewards: torch.Tensor, next_states: torch.Tensor,
                             dones: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distributional RL training for Rainbow DQN"""
        
        batch_size = states.size(0)
        
        # Current distribution over atoms: (batch_size, num_actions, num_atoms)
        current_dist = self.q_network(states)
        # Select distribution for taken actions: (batch_size, num_atoms)
        current_dist = current_dist[range(batch_size), actions]
        
        with torch.no_grad():
            # Double DQN action selection using current network
            next_q_values = self.q_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            
            # Get target distribution for selected actions
            next_dist = self.target_network(next_states)
            next_dist = next_dist[range(batch_size), next_actions]
            next_dist = torch.softmax(next_dist, dim=1)
            
            # Compute target distribution support
            target_support = rewards.unsqueeze(1) + (
                0.99 * (~dones).float().unsqueeze(1) * self.target_network.support.unsqueeze(0)
            )
            
            # Clamp target support to valid range
            target_support = target_support.clamp(
                self.target_network.v_min, self.target_network.v_max
            )
            
            # Distribute probability mass to nearest atoms
            target_dist = self._project_distribution(
                next_dist, target_support, self.target_network.support, 
                self.target_network.delta_z
            )
        
        # Cross-entropy loss between current and target distributions
        log_current_dist = torch.log(torch.softmax(current_dist, dim=1) + 1e-8)
        loss = -(target_dist * log_current_dist).sum(dim=1)
        
        # Calculate TD errors for prioritized replay (use Q-value differences)
        with torch.no_grad():
            current_q = (torch.softmax(current_dist, dim=1) * self.q_network.support.unsqueeze(0)).sum(dim=1)
            target_q = (target_dist * self.target_network.support.unsqueeze(0)).sum(dim=1)
            td_errors = torch.abs(current_q - target_q)
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            loss = (weights * loss).mean()
        else:
            loss = loss.mean()
        
        return loss, td_errors
    
    def _project_distribution(self, next_dist: torch.Tensor, target_support: torch.Tensor,
                             support: torch.Tensor, delta_z: float) -> torch.Tensor:
        """Project target distribution onto current support"""
        batch_size = next_dist.size(0)
        num_atoms = len(support)
        
        # Calculate indices for distributing probability mass
        b = (target_support - support[0]) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle edge cases
        l[(u > 0) * (l == u)] -= 1
        u[(l < (num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability mass
        target_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
        
        target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
        return target_dist
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("Target network updated")
    
    def get_q_values(self, state: MarketState) -> torch.Tensor:
        """Get Q-values for all actions given a state"""
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            if self.config.model_type == ModelType.RAINBOW:
                # Rainbow DQN: get Q-values from distributional output
                q_values = self.q_network.get_q_values(state_tensor)
            else:
                # Standard DQN or Dueling DQN
                q_values = self.q_network(state_tensor)
        return q_values.squeeze(0)
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file with preservation support"""
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
            
            # Trigger preservation if enabled
            if self._preservation_enabled and self._preservation_manager:
                try:
                    # Try to get running event loop
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._preserve_checkpoint(checkpoint, filepath))
                except RuntimeError:
                    # No running loop, run synchronously
                    asyncio.run(self._preserve_checkpoint(checkpoint, filepath))
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to save model", filepath=filepath, error=str(e))
            return False
    
    async def _preserve_checkpoint(self, checkpoint: Dict[str, Any], filepath: str):
        """Preserve checkpoint to cloud storage"""
        try:
            # Serialize checkpoint
            checkpoint_data = pickle.dumps(checkpoint)
            
            # Determine model type for preservation
            model_type_str = self.config.model_type.value.lower().replace('_', '')
            
            # Prepare metadata
            metadata = {
                'training_episodes': self.training_episodes,
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'performance': {
                    'total_reward': self.performance_metrics.total_reward,
                    'win_rate': self.performance_metrics.win_rate,
                    'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                    'max_drawdown': self.performance_metrics.max_drawdown,
                    'profit_factor': self.performance_metrics.profit_factor
                },
                'architecture': {
                    'input_size': self.input_size,
                    'hidden_size': self.config.hidden_size,
                    'num_layers': self.config.num_layers,
                    'output_size': self.output_size
                },
                'saved_from': filepath
            }
            
            # Add Rainbow-specific metadata
            if self.config.model_type == ModelType.RAINBOW:
                metadata['num_atoms'] = getattr(self.config, 'num_atoms', 51)
                metadata['v_min'] = getattr(self.config, 'v_min', -10.0)
                metadata['v_max'] = getattr(self.config, 'v_max', 10.0)
            
            # Determine if this is a checkpoint or final save
            tags = ['dqn_checkpoint' if 'checkpoint' in filepath.lower() else 'dqn_model']
            tags.append(f'episodes_{self.training_episodes}')
            
            # Save to preservation
            await self._preservation_manager.save_model(
                model_data=checkpoint_data,
                model_type=model_type_str,
                mode=self._current_mode,
                tags=tags,
                metadata=metadata,
                priority=PreservationPriority.HIGH if self._current_mode == 'live_trading' else PreservationPriority.NORMAL
            )
            
            self.logger.info("Model checkpoint preserved", model_type=model_type_str)
            
        except Exception as e:
            # Log error but don't fail the save operation
            self.logger.error("Checkpoint preservation failed", error=str(e))
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file with preservation fallback"""
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
            
        except FileNotFoundError:
            # Try preservation fallback if enabled
            if self._preservation_enabled and self._preservation_manager:
                self.logger.info("Local model not found, attempting preservation fallback", filepath=filepath)
                # Create a new event loop if none exists
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async function
                try:
                    if asyncio.iscoroutinefunction(self._load_from_preservation):
                        return loop.run_until_complete(self._load_from_preservation())
                    else:
                        return self._load_from_preservation()
                finally:
                    # Clean up the loop if we created it
                    try:
                        if loop and not asyncio.get_running_loop():
                            loop.close()
                    except RuntimeError:
                        pass
            else:
                self.logger.error("Model file not found", filepath=filepath)
                return False
        except Exception as e:
            self.logger.error("Failed to load model", filepath=filepath, error=str(e))
            return False
    
    async def _load_from_preservation(self) -> bool:
        """Load model from preserved state"""
        try:
            # Determine model type for preservation
            model_type_str = self.config.model_type.value.lower().replace('_', '')
            
            # Load from preservation
            model_data, metadata = await self._preservation_manager.load_model(
                model_type=model_type_str,
                version=None,  # Get latest version
                mode=self._current_mode,
                fallback=True
            )
            
            # Deserialize checkpoint
            checkpoint = pickle.loads(model_data)
            
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
            
            self.logger.info("Model restored from preservation", 
                           model_type=model_type_str,
                           version=metadata.get('version'),
                           training_episodes=self.training_episodes)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load from preservation", error=str(e), exception_type=type(e).__name__)
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


    def set_mode(self, mode: str):
        """Set the current operational mode"""
        self._current_mode = mode
        self.logger.info("Operational mode changed", mode=mode)


class DQNTrainingError(RLTrainingError):
    """Raised when DQN training fails"""
    pass


class DQNPredictionError(RLPredictionError):
    """Raised when DQN prediction fails"""
    pass


class DQNModelError(RLModelError):
    """Raised when DQN model operations fail"""
    pass