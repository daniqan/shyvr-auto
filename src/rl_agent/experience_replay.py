"""
Experience Replay Buffer System

Implements both standard and prioritized experience replay buffers
for DQN training with efficient memory management and sampling.
Includes database-backed storage for persistent experience replay.
"""

import asyncio
import random
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
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


class DatabaseExperienceReplayBuffer:
    """Database-backed experience replay buffer with memory caching"""
    
    def __init__(self, config: ReplayBufferConfig):
        """Initialize database experience replay buffer"""
        self.config = config
        self.logger = structlog.get_logger().bind(component="DatabaseExperienceReplayBuffer")
        
        # Import database buffer (defer import to avoid circular dependencies)
        try:
            from .experience_database import DatabaseExperienceBuffer, DatabaseExperienceConfig
            
            # Convert ReplayBufferConfig to DatabaseExperienceConfig
            db_config = DatabaseExperienceConfig(
                max_size=config.max_size,
                batch_size=config.batch_size,
                min_size=config.min_size,
                prioritized=config.prioritized,
                alpha=config.alpha,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                epsilon=config.epsilon,
                cache_size=min(1000, config.max_size // 10),  # Cache 10% of max size
                connection_pool_size=10,
                query_timeout=30.0,
                cleanup_threshold=0.9,
                archive_old_experiences=True
            )
            
            self._db_buffer = DatabaseExperienceBuffer(db_config)
            self._initialized = False
            
            # Performance tracking
            self._sync_operation_count = 0
            self._async_operation_count = 0
            
            self.logger.info("Database experience replay buffer created",
                           max_size=config.max_size,
                           batch_size=config.batch_size,
                           prioritized=config.prioritized,
                           cache_enabled=True)
            
        except ImportError as e:
            self.logger.error("Failed to import database buffer", error=str(e))
            raise ExperienceReplayError(f"Database buffer import failed: {e}")
    
    async def initialize(self) -> None:
        """Initialize database connection and buffer"""
        try:
            await self._db_buffer.initialize()
            self._initialized = True
            self.logger.info("Database experience replay buffer initialized")
        except Exception as e:
            self.logger.error("Failed to initialize database buffer", error=str(e))
            raise ExperienceReplayError(f"Database buffer initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup database resources"""
        try:
            if self._initialized:
                await self._db_buffer.cleanup()
                self._initialized = False
                self.logger.info("Database experience replay buffer cleaned up")
        except Exception as e:
            self.logger.error("Failed to cleanup database buffer", error=str(e))
    
    # Synchronous API methods for backward compatibility
    def add(self, experience: Experience) -> None:
        """Add experience synchronously (backward compatibility)"""
        try:
            # Run async method in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule as task
                task = asyncio.create_task(self._add_async(experience))
                # We can't wait here, so just schedule it
                self._sync_operation_count += 1
            else:
                # If no loop running, create one
                loop.run_until_complete(self._add_async(experience))
                self._sync_operation_count += 1
        except Exception as e:
            self.logger.error("Failed to add experience synchronously", error=str(e))
            raise ExperienceReplayError(f"Add experience failed: {e}")
    
    def sample(self) -> List[Dict[str, Any]]:
        """Sample experiences synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # This is problematic in async context - we should use async version
                self.logger.warning("Synchronous sample called in async context")
                raise ExperienceReplayError("Use async sample() method in async context")
            else:
                result = loop.run_until_complete(self._sample_async())
                self._sync_operation_count += 1
                return result
        except Exception as e:
            self.logger.error("Failed to sample experiences synchronously", error=str(e))
            raise ExperienceReplayError(f"Sample experiences failed: {e}")
    
    def can_sample(self) -> bool:
        """Check if buffer can sample synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.logger.warning("Synchronous can_sample called in async context")
                # Return a reasonable default
                return self._sync_operation_count > self.config.min_size
            else:
                return loop.run_until_complete(self._can_sample_async())
        except Exception as e:
            self.logger.error("Failed to check sampling capability", error=str(e))
            return False
    
    def clear(self) -> None:
        """Clear buffer synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(self._clear_async())
                self._sync_operation_count = 0
            else:
                loop.run_until_complete(self._clear_async())
                self._sync_operation_count = 0
        except Exception as e:
            self.logger.error("Failed to clear buffer synchronously", error=str(e))
            raise ExperienceReplayError(f"Clear buffer failed: {e}")
    
    def __len__(self) -> int:
        """Get buffer size synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Return cached count for async contexts
                return self._sync_operation_count
            else:
                return loop.run_until_complete(self._size_async())
        except Exception as e:
            self.logger.error("Failed to get buffer size", error=str(e))
            return 0
    
    @property
    def max_size(self) -> int:
        """Get maximum buffer size"""
        return self.config.max_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Return basic stats for async contexts
                return {
                    'size': self._sync_operation_count,
                    'max_size': self.config.max_size,
                    'can_sample': self._sync_operation_count >= self.config.min_size,
                    'utilization': self._sync_operation_count / self.config.max_size,
                    'average_reward': 0.0,  # Would need to calculate from database
                    'action_distribution': {},
                    'database_backed': True,
                    'sync_operations': self._sync_operation_count,
                    'async_operations': self._async_operation_count
                }
            else:
                return loop.run_until_complete(self._get_statistics_async())
        except Exception as e:
            self.logger.error("Failed to get buffer statistics", error=str(e))
            return {
                'size': 0,
                'max_size': self.config.max_size,
                'can_sample': False,
                'utilization': 0.0,
                'average_reward': 0.0,
                'action_distribution': {},
                'error': str(e)
            }
    
    # Async API methods (preferred)
    async def add_async(self, experience: Experience, priority: Optional[float] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add experience asynchronously"""
        await self._add_async(experience, priority, metadata)
    
    async def add_batch_async(self, experiences: List[Experience], 
                             priorities: Optional[List[float]] = None,
                             metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add batch of experiences asynchronously"""
        if not self._initialized:
            await self.initialize()
        
        await self._db_buffer.add_batch(experiences, priorities, metadatas)
        self._async_operation_count += len(experiences)
        
        self.logger.debug("Added experience batch async", count=len(experiences))
    
    async def sample_async(self) -> List[Dict[str, Any]]:
        """Sample experiences asynchronously"""
        return await self._sample_async()
    
    async def can_sample_async(self) -> bool:
        """Check if buffer can sample asynchronously"""
        return await self._can_sample_async()
    
    async def size_async(self) -> int:
        """Get buffer size asynchronously"""
        return await self._size_async()
    
    async def update_priorities_async(self, priority_updates: List[Dict[str, Any]]) -> int:
        """Update experience priorities asynchronously"""
        if not self._initialized:
            await self.initialize()
        
        return await self._db_buffer.update_priorities(priority_updates)
    
    async def get_performance_metrics_async(self) -> Dict[str, Any]:
        """Get performance metrics asynchronously"""
        if not self._initialized:
            await self.initialize()
        
        db_metrics = await self._db_buffer.get_performance_metrics()
        
        # Add buffer-specific metrics
        db_metrics.update({
            'sync_operations': self._sync_operation_count,
            'async_operations': self._async_operation_count,
            'total_operations': self._sync_operation_count + self._async_operation_count
        })
        
        return db_metrics
    
    async def get_recent_async(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent experiences asynchronously"""
        if not self._initialized:
            await self.initialize()
        
        return await self._db_buffer.get_recent(limit)
    
    async def clear_async(self) -> None:
        """Clear buffer asynchronously"""
        await self._clear_async()
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """Get comprehensive statistics asynchronously"""
        return await self._get_statistics_async()
    
    # Private implementation methods
    async def _add_async(self, experience: Experience, priority: Optional[float] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Internal async add implementation"""
        if not self._initialized:
            await self.initialize()
        
        await self._db_buffer.add(experience, priority, metadata)
        self._async_operation_count += 1
        
        self.logger.debug("Added experience async", action=experience.action, reward=experience.reward)
    
    async def _sample_async(self) -> List[Dict[str, Any]]:
        """Internal async sample implementation"""
        if not self._initialized:
            await self.initialize()
        
        if not await self._can_sample_async():
            raise ExperienceReplayError(
                f"Cannot sample: buffer has {await self._size_async()} experiences, "
                f"need at least {self.config.min_size}"
            )
        
        batch = await self._db_buffer.sample()
        
        # Convert database format to replay buffer format for backward compatibility
        formatted_batch = []
        for exp in batch:
            # Remove database-specific fields for standard replay buffer compatibility
            formatted_exp = {
                'state': exp['state'],
                'action': exp['action'],
                'reward': exp['reward'],
                'next_state': exp['next_state'],
                'done': exp['done']
            }
            
            # Add prioritized replay specific fields if needed
            if self.config.prioritized:
                formatted_exp['weight'] = exp.get('weight', 1.0)
                formatted_exp['index'] = exp.get('database_id', exp.get('id', 0))
            
            formatted_batch.append(formatted_exp)
        
        self.logger.debug("Sampled experience batch async", count=len(formatted_batch))
        return formatted_batch
    
    async def _can_sample_async(self) -> bool:
        """Internal async can_sample implementation"""
        if not self._initialized:
            await self.initialize()
        
        return await self._db_buffer.can_sample()
    
    async def _size_async(self) -> int:
        """Internal async size implementation"""
        if not self._initialized:
            await self.initialize()
        
        return await self._db_buffer.size()
    
    async def _clear_async(self) -> None:
        """Internal async clear implementation"""
        if not self._initialized:
            await self.initialize()
        
        # Clear database buffer (this would be a complex operation)
        # For now, we'll create a new session
        await self._db_buffer.cleanup()
        await self._db_buffer.initialize()
        
        self._sync_operation_count = 0
        self._async_operation_count = 0
        
        self.logger.info("Database buffer cleared async")
    
    async def _get_statistics_async(self) -> Dict[str, Any]:
        """Internal async get_statistics implementation"""
        if not self._initialized:
            await self.initialize()
        
        db_stats = await self._db_buffer.get_statistics()
        
        # Enhance with buffer-specific statistics
        db_stats.update({
            'database_backed': True,
            'sync_operations': self._sync_operation_count,
            'async_operations': self._async_operation_count,
            'total_operations': self._sync_operation_count + self._async_operation_count,
            'can_sample': await self._can_sample_async(),
            'utilization': (await self._size_async()) / self.config.max_size if self.config.max_size > 0 else 0.0
        })
        
        return db_stats


def create_replay_buffer(config: ReplayBufferConfig) -> ExperienceReplayBuffer:
    """Factory function to create appropriate replay buffer"""
    if config.prioritized:
        return PrioritizedExperienceReplayBuffer(config)
    else:
        return ExperienceReplayBuffer(config)


def create_database_replay_buffer(config: ReplayBufferConfig) -> DatabaseExperienceReplayBuffer:
    """Factory function to create database-backed replay buffer"""
    return DatabaseExperienceReplayBuffer(config)


class DatabasePrioritizedExperienceReplayBuffer(DatabaseExperienceReplayBuffer):
    """Database-backed prioritized experience replay buffer"""
    
    def __init__(self, config: ReplayBufferConfig):
        """Initialize database prioritized buffer"""
        # Ensure prioritized is enabled
        config.prioritized = True
        super().__init__(config)
        
        # Additional prioritized replay state
        self.beta = config.beta_start
        self._training_step = 0
        
        self.logger = structlog.get_logger().bind(component="DatabasePrioritizedExperienceReplayBuffer")
        self.logger.info("Database prioritized experience replay buffer created",
                        alpha=config.alpha,
                        beta_start=config.beta_start,
                        beta_end=config.beta_end)
    
    def update_priorities(self, indices: List[int], td_errors: Union[np.ndarray, List[float]]) -> None:
        """Update priorities synchronously (backward compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Convert indices and td_errors to database format
                priority_updates = [
                    {
                        'database_id': int(idx),  # Using index as database_id for compatibility
                        'priority': float(abs(td_error) + self.config.epsilon) ** self.config.alpha,
                        'td_error': float(td_error)
                    }
                    for idx, td_error in zip(indices, td_errors)
                ]
                task = asyncio.create_task(self.update_priorities_async(priority_updates))
                self._training_step += 1
                self.anneal_beta()
            else:
                # Convert to database format
                priority_updates = [
                    {
                        'database_id': int(idx),
                        'priority': float(abs(td_error) + self.config.epsilon) ** self.config.alpha,
                        'td_error': float(td_error)
                    }
                    for idx, td_error in zip(indices, td_errors)
                ]
                loop.run_until_complete(self.update_priorities_async(priority_updates))
                self._training_step += 1
                self.anneal_beta()
        except Exception as e:
            self.logger.error("Failed to update priorities synchronously", error=str(e))
            raise ExperienceReplayError(f"Update priorities failed: {e}")
    
    def anneal_beta(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> None:
        """Anneal beta parameter for importance sampling"""
        if step is not None:
            self._training_step = step
        
        if total_steps is None:
            # Default to 100,000 steps for annealing
            total_steps = 100000
        
        progress = min(self._training_step / total_steps, 1.0)
        self.beta = self.config.beta_start + progress * (self.config.beta_end - self.config.beta_start)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics including priority information"""
        stats = super().get_statistics()
        
        # Add prioritized replay specific statistics
        if isinstance(stats, dict):
            stats.update({
                'beta': self.beta,
                'alpha': self.config.alpha,
                'training_step': self._training_step,
                'priority_enabled': True
            })
        
        return stats
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """Get comprehensive statistics asynchronously with priority info"""
        stats = await super()._get_statistics_async()
        
        # Add prioritized replay specific statistics
        stats.update({
            'beta': self.beta,
            'alpha': self.config.alpha,
            'training_step': self._training_step,
            'priority_enabled': True
        })
        
        return stats


def create_prioritized_database_replay_buffer(config: ReplayBufferConfig) -> DatabasePrioritizedExperienceReplayBuffer:
    """Factory function to create database-backed prioritized replay buffer"""
    return DatabasePrioritizedExperienceReplayBuffer(config)


class ExperienceReplayError(Exception):
    """Raised when experience replay operations fail"""
    pass