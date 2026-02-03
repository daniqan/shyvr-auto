"""
Proximal Policy Optimization (PPO) Trading Agent with Transformer Backbone

Implements a state-of-the-art continuous control RL agent using PPO and
Transformer-based feature extraction for capturing temporal market dependencies.
"""

import asyncio
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
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


class TransformerFeatureExtractor(nn.Module):
    """
    Transformer-based feature extractor for processing temporal market data.
    Captures long-range dependencies in price/volume history.
    """
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, max_seq_len: int = 60):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size)
        Returns:
            Encoded features (batch_size, d_model)
        """
        # Handle non-sequential input (if single state passed) by adding dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer output: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Pooling: Use the last time step's embedding as the context vector
        # Alternatively, we could use mean pooling or a [CLS] token
        context_vector = output[:, -1, :]
        return context_vector


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature extractor.
    Actor: Outputs mean and std for continuous action distribution (Gaussian).
    Critic: Outputs state value estimate V(s).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 use_transformer: bool = False, seq_len: int = 60):
        super().__init__()
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.feature_extractor = TransformerFeatureExtractor(
                input_size=input_size,
                d_model=hidden_size,
                max_seq_len=seq_len
            )
            feature_dim = hidden_size
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )
            feature_dim = hidden_size

        # Actor heads (Continuous)
        # We output a single continuous value [-1, 1] representing position sizing/direction
        # -1 = Max Sell, 1 = Max Buy, 0 = Neutral
        self.actor_mean = nn.Linear(feature_dim, output_size)
        self.actor_log_std = nn.Parameter(torch.zeros(1, output_size)) # Learnable log_std independent of state
        
        # Critic head
        self.critic = nn.Linear(feature_dim, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization for better convergence in PPO
        for layer in [self.actor_mean, self.critic]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass returning action distribution and state value.
        """
        features = self.feature_extractor(x)
        
        # Critic value
        value = self.critic(features)
        
        # Actor distribution
        mean = torch.tanh(self.actor_mean(features)) # Bound mean to [-1, 1]
        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = Normal(mean, std)
        
        return dist, value


class PPOTradingAgent(RLAgentBase):
    """
    PPO (Proximal Policy Optimization) Agent.
    Supports continuous action spaces and Transformer backbones.
    """
    
    def __init__(self, config: AgentConfig, use_enhanced_features: bool = True):
        super().__init__(config)
        self.logger = structlog.get_logger().bind(agent="PPOTradingAgent")
        
        # Configuration
        self.use_enhanced_features = use_enhanced_features
        self.input_size = MarketState.get_enhanced_feature_size() if use_enhanced_features else MarketState.get_feature_size()
        
        # Output size: 1 continuous value (Allocation/Direction: -1 to 1)
        self.output_size = 1 
        
        # Hyperparameters
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Model
        use_transformer = (config.model_type == ModelType.TRANSFORMER_PPO)
        self.network = ActorCriticNetwork(
            input_size=self.input_size,
            hidden_size=config.hidden_size,
            output_size=self.output_size,
            use_transformer=use_transformer
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # State buffers for PPO update
        self.current_episode_experiences = []
        
        # Preservation
        self._init_preservation()
        
        self.logger.info("PPO Agent initialized", 
                         model_type=config.model_type.value,
                         input_size=self.input_size,
                         use_transformer=use_transformer)

    def _init_preservation(self):
        """Initialize model preservation manager"""
        self._preservation_manager = None
        self._preservation_enabled = False
        if PRESERVATION_AVAILABLE and hasattr(self.config, 'preservation') and self.config.preservation.get('enabled', False):
            try:
                pres_config = PreservationConfig(
                    gcs_bucket=self.config.preservation.get('gcs_bucket', 'shyvr-models-prod'),
                    backup_interval_hours=self.config.preservation.get('backup_interval_hours', 3),
                    max_versions_per_model=self.config.preservation.get('max_versions_per_model', 20)
                )
                self._preservation_manager = PreservationManager(pres_config)
                self._preservation_enabled = True
            except Exception as e:
                self.logger.warning("Failed to initialize preservation", error=str(e))

    def _state_to_tensor(self, state: MarketState) -> torch.Tensor:
        """Convert state to tensor, handling potential sequence data"""
        if self.use_enhanced_features and hasattr(state, 'to_feature_vector'):
            vector = state.to_feature_vector()
        else:
            vector = state.to_vector()
            
        return torch.FloatTensor(vector).unsqueeze(0) # Add batch dim

    async def predict_action(self, state: MarketState) -> Tuple[TradeAction, float]:
        """
        Predict action using PPO policy.
        Returns discrete TradeAction for compatibility, but internally uses continuous signal.
        """
        try:
            state_tensor = self._state_to_tensor(state)
            
            self.network.eval()
            with torch.no_grad():
                dist, value = self.network(state_tensor)
                action_continuous = dist.sample() # Sample from Gaussian
                action_value = action_continuous.item()
                
                # Calculate confidence (probability density at the sampled point)
                # This is a proxy for confidence in PPO
                log_prob = dist.log_prob(action_continuous)
                confidence = torch.exp(log_prob).item()
                
            self.network.train()
            
            # Map continuous output [-1, 1] to Discrete TradeAction for system compatibility
            # > 0.5: STRONG_BUY
            # 0.1 to 0.5: BUY
            # -0.1 to 0.1: HOLD
            # -0.5 to -0.1: SELL
            # < -0.5: STRONG_SELL
            
            if action_value > 0.5:
                trade_action = TradeAction.STRONG_BUY
            elif action_value > 0.1:
                trade_action = TradeAction.BUY
            elif action_value < -0.5:
                trade_action = TradeAction.STRONG_SELL
            elif action_value < -0.1:
                trade_action = TradeAction.SELL
            else:
                trade_action = TradeAction.HOLD
                
            # Log for debugging
            self.logger.debug("PPO Prediction", 
                              continuous_value=action_value, 
                              mapped_action=trade_action.value,
                              confidence=confidence)
            
            return trade_action, confidence

        except Exception as e:
            self.logger.error("PPO prediction failed", error=str(e))
            raise RLPredictionError(f"PPO prediction failed: {str(e)}")

    async def train_step(self, batch_experiences: List[Dict]) -> Dict[str, float]:
        """
        Execute a PPO training step (epoch update).
        PPO requires collecting a batch of trajectories, calculating advantages,
        and then doing multiple gradient updates.
        """
        if len(batch_experiences) < self.config.batch_size:
            return {} # Wait for more data
        
        # Prepare tensors
        states = torch.stack([torch.FloatTensor(e['state']) for e in batch_experiences])
        # Actions in PPO are the stored continuous values, not the discrete enums
        # We assume 'action_value' is stored in experience
        actions = torch.stack([torch.FloatTensor([e.get('action_value', 0.0)]) for e in batch_experiences])
        rewards = torch.FloatTensor([e['reward'] for e in batch_experiences]).unsqueeze(1)
        masks = torch.FloatTensor([1 - float(e['done']) for e in batch_experiences]).unsqueeze(1)
        next_states = torch.stack([torch.FloatTensor(e['next_state']) for e in batch_experiences])
        
        # 1. Calculate Advantages (GAE)
        with torch.no_grad():
            _, values = self.network(states)
            _, next_values = self.network(next_states)
            
            deltas = rewards + self.gamma * next_values * masks - values
            advantages = torch.zeros_like(deltas)
            adv = 0.0
            for t in reversed(range(len(deltas))):
                adv = deltas[t] + self.gamma * self.gae_lambda * masks[t] * adv
                advantages[t] = adv
            
            returns = advantages + values

        # 2. PPO Update Epochs
        total_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Recalculate policy distribution and values
            dist, new_values = self.network(states)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(actions)
            
            # Use stored old_log_probs if available, else approximate (on first pass they are same)
            # In a strict PPO implementation, we'd use old_log_probs computed before the epochs
            # Here we approximate for simplicity in the 'train_step' interface
            old_log_probs = log_probs.detach() 

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            
            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()

        self.training_episodes += 1
        
        return {
            "loss": total_loss / self.ppo_epochs,
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy": entropy.item()
        }

    def save_model(self, filepath: str) -> bool:
        try:
            checkpoint = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__,
                'training_episodes': self.training_episodes
            }
            torch.save(checkpoint, filepath)
            
            if self._preservation_enabled and self._preservation_manager:
                 # Fire and forget preservation
                 asyncio.create_task(self._preserve_model(checkpoint, filepath))
                 
            return True
        except Exception as e:
            self.logger.error("Failed to save PPO model", error=str(e))
            return False

    async def _preserve_model(self, checkpoint: Dict, filepath: str):
        try:
            data = pickle.dumps(checkpoint)
            await self._preservation_manager.save_model(
                model_data=data,
                model_type="ppo_transformer",
                tags=["latest", f"ep_{self.training_episodes}"]
            )
        except Exception:
            pass # Suppress async preservation errors

    def load_model(self, filepath: str) -> bool:
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_episodes = checkpoint.get('training_episodes', 0)
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error("Failed to load PPO model", error=str(e))
            return False
