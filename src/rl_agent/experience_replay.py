"""
Experience Replay Buffer System

Implements both standard and prioritized experience replay buffers
for DQN training with efficient memory management and sampling.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class Experience:
    """Single experience for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary for training"""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create experience from dictionary"""
        return cls(
            state=np.array(data['state'], dtype=np.float32),
            action=data['action'],
            reward=float(data['reward']),
            next_state=np.array(data['next_state'], dtype=np.float32),
            done=bool(data['done']),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class ReplayBufferConfig:
    """Configuration for experience replay buffers"""
    
    # Buffer size
    max_size: int = 10000
    batch_size: int = 32
    min_size: int = 100  # Minimum experiences before sampling
    
    # Prioritized replay settings
    prioritized: bool = False
    alpha: float = 0.6      # Priority exponent
    beta_start: float = 0.4  # Importance sampling start
    beta_end: float = 1.0    # Importance sampling end
    epsilon: float = 1e-6    # Small constant to avoid zero priorities


class ExperienceReplayBuffer:
    """Standard experience replay buffer with uniform sampling"""
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.buffer = deque(maxlen=config.max_size)
        self.logger = structlog.get_logger().bind(component="ExperienceReplayBuffer")
        
        self.logger.info("Experience replay buffer initialized",
                        max_size=config.max_size,
                        batch_size=config.batch_size,
                        min_size=config.min_size)
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
        if len(self.buffer) % 1000 == 0:
            self.logger.debug("Buffer size updated", size=len(self.buffer))
    
    def sample(self) -> List[Dict[str, Any]]:
        """Sample batch of experiences uniformly"""
        if not self.can_sample():
            raise ExperienceReplayError(
                f"Cannot sample: buffer has {len(self.buffer)} experiences, "
                f"need at least {self.config.min_size}"
            )
        
        # Sample random experiences
        batch_experiences = random.sample(list(self.buffer), self.config.batch_size)
        
        # Convert to dictionaries
        batch = []
        for experience in batch_experiences:
            exp_dict = experience.to_dict()
            # Remove timestamp for training (not needed)
            exp_dict.pop('timestamp', None)
            batch.append(exp_dict)
        
        return batch
    
    def can_sample(self) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return len(self.buffer) >= self.config.min_size
    
    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer.clear()
        self.logger.info("Buffer cleared")
    
    def __len__(self) -> int:
        """Get number of experiences in buffer"""
        return len(self.buffer)
    
    @property
    def max_size(self) -> int:
        """Get maximum buffer size"""
        return self.config.max_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'max_size': self.config.max_size,
                'can_sample': False,
                'utilization': 0.0,
                'average_reward': 0.0,
                'action_distribution': {}
            }
        
        # Calculate statistics
        rewards = [exp.reward for exp in self.buffer]
        actions = [exp.action for exp in self.buffer]
        
        # Action distribution
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'size': len(self.buffer),
            'max_size': self.config.max_size,
            'can_sample': self.can_sample(),
            'utilization': len(self.buffer) / self.config.max_size,
            'average_reward': np.mean(rewards),
            'action_distribution': action_counts
        }


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    """Prioritized experience replay buffer"""
    
    def __init__(self, config: ReplayBufferConfig):
        super().__init__(config)
        
        self.priorities = deque(maxlen=config.max_size)
        self.max_priority = 1.0
        self.alpha = config.alpha
        self.beta = config.beta_start
        
        self.logger = structlog.get_logger().bind(component="PrioritizedExperienceReplayBuffer")
        
        self.logger.info("Prioritized experience replay buffer initialized",
                        alpha=self.alpha,
                        beta_start=config.beta_start,
                        beta_end=config.beta_end)
    
    def add(self, experience: Experience):
        """Add experience with maximum priority"""
        super().add(experience)
        
        # Assign maximum priority to new experiences
        self.priorities.append(self.max_priority)
        
        # Ensure priorities and buffer have same length
        assert len(self.priorities) == len(self.buffer)
    
    def sample(self) -> List[Dict[str, Any]]:
        """Sample batch using prioritized sampling"""
        if not self.can_sample():
            raise ExperienceReplayError(
                f"Cannot sample: buffer has {len(self.buffer)} experiences, "
                f"need at least {self.config.min_size}"
            )
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer),
            size=self.config.batch_size,
            p=probabilities,
            replace=False
        )
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Create batch with weights and indices
        batch = []
        for i, (idx, weight) in enumerate(zip(indices, weights)):
            experience = list(self.buffer)[idx]
            exp_dict = experience.to_dict()
            exp_dict.pop('timestamp', None)  # Remove timestamp
            exp_dict['weight'] = float(weight)
            exp_dict['index'] = int(idx)
            batch.append(exp_dict)
        
        return batch
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                # Priority is TD error + small epsilon to avoid zero priority
                priority = (abs(td_error) + self.config.epsilon) ** self.alpha
                self.priorities[idx] = priority
                
                # Update max priority
                if priority > self.max_priority:
                    self.max_priority = priority
    
    def anneal_beta(self, step: int, total_steps: int):
        """Anneal beta parameter for importance sampling"""
        progress = min(step / total_steps, 1.0)
        self.beta = self.config.beta_start + progress * (self.config.beta_end - self.config.beta_start)
    
    def clear(self):
        """Clear buffer and priorities"""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0
        self.beta = self.config.beta_start
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics including priority information"""
        stats = super().get_statistics()
        
        if len(self.priorities) > 0:
            priorities = np.array(self.priorities)
            stats.update({
                'priority_stats': {
                    'mean': float(np.mean(priorities)),
                    'std': float(np.std(priorities)),
                    'min': float(np.min(priorities)),
                    'max': float(np.max(priorities))
                },
                'beta': self.beta,
                'alpha': self.alpha
            })
        
        return stats


def create_replay_buffer(config: ReplayBufferConfig) -> ExperienceReplayBuffer:
    """Factory function to create appropriate replay buffer"""
    if config.prioritized:
        return PrioritizedExperienceReplayBuffer(config)
    else:
        return ExperienceReplayBuffer(config)


class ExperienceReplayError(Exception):
    """Raised when experience replay operations fail"""
    pass