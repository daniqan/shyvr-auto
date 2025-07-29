"""
Database Experience Buffer System

Implements database-backed experience storage for RL training with efficient
batch operations, prioritized sampling, and performance optimization.
"""

import asyncio
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import structlog

from ..utils.database import (
    get_database_connection,
    insert_experience_batch,
    query_experiences_by_session,
    sample_prioritized_experiences,
    update_experience_priorities,
    get_session_statistics,
    atomic_experience_batch_operation,
    DatabaseQueryError
)
from .experience_replay import Experience

logger = structlog.get_logger()


@dataclass
class DatabaseExperienceConfig:
    """Configuration for database experience buffer"""
    
    # Buffer size limits
    max_size: int = 10000
    batch_size: int = 32
    min_size: int = 100
    
    # Prioritized replay settings
    prioritized: bool = True
    alpha: float = 0.6      # Priority exponent
    beta_start: float = 0.4  # Importance sampling start
    beta_end: float = 1.0    # Importance sampling end
    epsilon: float = 1e-6    # Small constant to avoid zero priorities
    
    # Performance optimization
    cache_size: int = 1000   # Number of experiences to cache in memory
    connection_pool_size: int = 10
    query_timeout: float = 30.0  # Query timeout in seconds
    
    # Lifecycle management
    cleanup_threshold: float = 0.9  # Cleanup when buffer reaches this ratio
    archive_old_experiences: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if self.min_size >= self.max_size:
            raise ValueError("min_size must be less than max_size")
        if self.batch_size > self.max_size:
            raise ValueError("batch_size must not exceed max_size")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 <= self.beta_start <= 1:
            raise ValueError("beta_start must be between 0 and 1")
        if not 0 <= self.beta_end <= 1:
            raise ValueError("beta_end must be between 0 and 1")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.connection_pool_size <= 0:
            raise ValueError("connection_pool_size must be positive")
        if self.query_timeout <= 0:
            raise ValueError("query_timeout must be positive")


class DatabaseExperienceError(Exception):
    """Raised when database experience operations fail"""
    pass


class DatabaseExperienceBuffer:
    """Database-backed experience replay buffer with performance optimization"""
    
    def __init__(self, config: DatabaseExperienceConfig):
        self.config = config
        self.session_id = uuid.uuid4()
        self.logger = structlog.get_logger().bind(
            component="DatabaseExperienceBuffer",
            session_id=str(self.session_id)
        )
        
        # State tracking
        self._initialized = False
        self._cache_enabled = config.cache_size > 0
        self._experience_cache: OrderedDict = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._operation_times = []
        self._insertion_count = 0
        self._query_count = 0
        
        # Priority sampling state
        self.beta = config.beta_start
        self._training_step = 0
        
        self.logger.info("Database experience buffer created",
                        max_size=config.max_size,
                        batch_size=config.batch_size,
                        prioritized=config.prioritized,
                        cache_enabled=self._cache_enabled)
    
    async def initialize(self) -> None:
        """Initialize database connection and session"""
        try:
            # Verify database connection
            async with get_database_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            # Create training session record
            session_data = {
                'session_id': str(self.session_id),
                'user_id': None,  # System-generated
                'session_name': f'DatabaseExperienceBuffer_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'trading_mode': 'simulation',  # Default mode
                'agent_config': {
                    'buffer_type': 'database',
                    'prioritized': self.config.prioritized,
                    'alpha': self.config.alpha,
                    'beta_range': [self.config.beta_start, self.config.beta_end]
                },
                'environment_config': {
                    'max_size': self.config.max_size,
                    'batch_size': self.config.batch_size,
                    'cache_size': self.config.cache_size
                }
            }
            
            # Insert session record (implementation would depend on actual database schema)
            async with get_database_connection() as conn:
                await conn.execute("""
                    INSERT INTO rl_training_sessions (
                        session_id, user_id, session_name, trading_mode,
                        agent_config, environment_config, session_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (session_id) DO NOTHING
                """, 
                uuid.UUID(session_data['session_id']),
                session_data['user_id'],
                session_data['session_name'],
                session_data['trading_mode'],
                json.dumps(session_data['agent_config']),
                json.dumps(session_data['environment_config']),
                'running'
                )
            
            self._initialized = True
            self.logger.info("Database experience buffer initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize database experience buffer", error=str(e))
            raise DatabaseExperienceError(f"Initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup resources and close session"""
        try:
            if self._initialized:
                # Update session status to completed
                async with get_database_connection() as conn:
                    await conn.execute("""
                        UPDATE rl_training_sessions 
                        SET session_status = 'completed', ended_at = NOW()
                        WHERE session_id = $1
                    """, self.session_id)
                
                self._initialized = False
                self.logger.info("Database experience buffer cleaned up")
                
        except Exception as e:
            self.logger.error("Failed to cleanup database experience buffer", error=str(e))
    
    async def add(self, experience: Experience, priority: Optional[float] = None, 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a single experience to the buffer"""
        await self.add_batch([experience], priorities=[priority] if priority else None, 
                           metadatas=[metadata] if metadata else None)
    
    async def add_batch(self, experiences: List[Experience], 
                       priorities: Optional[List[float]] = None,
                       metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add batch of experiences efficiently"""
        if not self._initialized:
            raise DatabaseExperienceError("Buffer not initialized")
        
        if not experiences:
            return
        
        start_time = time.time()
        
        try:
            # Validate experiences
            for i, exp in enumerate(experiences):
                if exp.state.size == 0:
                    raise DatabaseExperienceError(f"Experience {i} has invalid state: empty array")
                if exp.next_state is not None and exp.next_state.size == 0:
                    raise DatabaseExperienceError(f"Experience {i} has invalid next_state: empty array")
            
            # Prepare database records
            db_experiences = []
            for i, exp in enumerate(experiences):
                priority_val = priorities[i] if priorities and i < len(priorities) else 1.0
                metadata_val = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                db_exp = {
                    'session_id': str(self.session_id),
                    'step_number': self._insertion_count + i,
                    'state': exp.state.tolist(),
                    'action': exp.action,
                    'reward': exp.reward,
                    'next_state': exp.next_state.tolist() if exp.next_state is not None else None,
                    'done': exp.done,
                    'priority': priority_val,
                    'created_at': exp.timestamp,
                    'metadata': metadata_val
                }
                db_experiences.append(db_exp)
            
            # Insert batch into database
            inserted_count = await insert_experience_batch(db_experiences)
            
            # Update counters
            self._insertion_count += inserted_count
            
            # Invalidate cache
            if self._cache_enabled:
                self._experience_cache.clear()
            
            # Check for cleanup if approaching max size
            if await self._should_cleanup():
                await self._cleanup_old_experiences()
            
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            self.logger.debug("Added experience batch",
                            count=inserted_count,
                            operation_time_ms=operation_time*1000)
            
        except Exception as e:
            self.logger.error("Failed to add experience batch", error=str(e))
            raise DatabaseExperienceError(f"Batch addition failed: {e}")
    
    async def sample(self) -> List[Dict[str, Any]]:
        """Sample batch of experiences"""
        if not self._initialized:
            raise DatabaseExperienceError("Buffer not initialized")
        
        if not await self.can_sample():
            raise DatabaseExperienceError("Insufficient experiences for sampling")
        
        start_time = time.time()
        
        try:
            if self.config.prioritized:
                experiences = await sample_prioritized_experiences(
                    batch_size=self.config.batch_size,
                    alpha=self.config.alpha,
                    session_ids=[str(self.session_id)]
                )
                
                # Add importance sampling weights
                for exp in experiences:
                    # Calculate importance sampling weight (beta annealing)
                    priority = exp.get('priority', 1.0)
                    weight = (1.0 / (len(experiences) * priority)) ** self.beta
                    exp['weight'] = weight
                    exp['database_id'] = exp['id']  # For priority updates
            else:
                # Uniform sampling
                experiences = await query_experiences_by_session(
                    session_id=str(self.session_id),
                    limit=self.config.batch_size
                )
                
                # Convert to sampling format
                for exp in experiences:
                    exp['database_id'] = exp['id']
                    # Remove timestamp for training
                    exp.pop('created_at', None)
            
            self._query_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            # Normalize importance weights if prioritized
            if self.config.prioritized and experiences:
                weights = [exp['weight'] for exp in experiences]
                max_weight = max(weights)
                for exp in experiences:
                    exp['weight'] /= max_weight
            
            self.logger.debug("Sampled experiences",
                            count=len(experiences),
                            operation_time_ms=operation_time*1000)
            
            return experiences
            
        except Exception as e:
            self.logger.error("Failed to sample experiences", error=str(e))
            raise DatabaseExperienceError(f"Sampling failed: {e}")
    
    async def can_sample(self) -> bool:
        """Check if buffer has enough experiences for sampling"""
        try:
            current_size = await self.size()
            return current_size >= self.config.min_size
        except Exception:
            return False
    
    async def size(self) -> int:
        """Get current number of experiences in buffer"""
        try:
            async with get_database_connection() as conn:
                count = await conn.fetchval(
                    "SELECT count(*) FROM rl_experiences WHERE session_id = $1",
                    self.session_id
                )
                return count or 0
        except Exception as e:
            self.logger.error("Failed to get buffer size", error=str(e))
            return 0
    
    async def update_priorities(self, priority_updates: List[Dict[str, Any]]) -> int:
        """Update experience priorities after training"""
        if not priority_updates:
            return 0
        
        try:
            # Convert to database format
            db_updates = []
            for update in priority_updates:
                db_updates.append({
                    'id': update['database_id'],
                    'priority': update['priority'],
                    'td_error': update.get('td_error', 0.0)
                })
            
            updated_count = await update_experience_priorities(db_updates)
            
            # Anneal beta for importance sampling
            self._training_step += 1
            self.anneal_beta()
            
            self.logger.debug("Updated experience priorities", count=updated_count)
            return updated_count
            
        except Exception as e:
            self.logger.error("Failed to update priorities", error=str(e))
            raise DatabaseExperienceError(f"Priority update failed: {e}")
    
    def anneal_beta(self, total_steps: Optional[int] = None) -> None:
        """Anneal beta parameter for importance sampling"""
        if total_steps is None:
            # Simple linear annealing over 100k steps
            total_steps = 100000
        
        progress = min(self._training_step / total_steps, 1.0)
        self.beta = self.config.beta_start + progress * (self.config.beta_end - self.config.beta_start)
    
    async def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent experiences"""
        cache_key = f"recent_{limit}"
        
        # Check cache first
        if self._cache_enabled and cache_key in self._experience_cache:
            self._cache_hits += 1
            return self._experience_cache[cache_key]
        
        try:
            experiences = await query_experiences_by_session(
                session_id=str(self.session_id),
                limit=limit
            )
            
            # Cache results
            if self._cache_enabled:
                self._experience_cache[cache_key] = experiences
                if len(self._experience_cache) > self.config.cache_size:
                    # Remove oldest entry
                    self._experience_cache.popitem(last=False)
                self._cache_misses += 1
            
            return experiences
            
        except Exception as e:
            self.logger.error("Failed to get recent experiences", error=str(e))
            return []
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        try:
            stats = await get_session_statistics(str(self.session_id))
            return stats
        except Exception as e:
            self.logger.error("Failed to get session statistics", error=str(e))
            return {'session_id': str(self.session_id), 'total_experiences': 0}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        try:
            session_stats = await self.get_session_stats()
            
            # Add buffer-specific statistics
            cache_hit_rate = 0.0
            if self._cache_hits + self._cache_misses > 0:
                cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
            
            avg_operation_time = 0.0
            if self._operation_times:
                avg_operation_time = sum(self._operation_times) / len(self._operation_times)
            
            insertion_rate = 0.0
            if avg_operation_time > 0 and self._insertion_count > 0:
                insertion_rate = self._insertion_count / (sum(self._operation_times) or 1)
            
            stats = {
                **session_stats,
                'cache_hit_rate': cache_hit_rate,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'insertion_rate': insertion_rate,
                'query_time_avg': avg_operation_time * 1000,  # Convert to ms
                'total_operations': len(self._operation_times),
                'beta_current': self.beta,
                'training_step': self._training_step
            }
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get buffer statistics", error=str(e))
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            operation_times_ms = [t * 1000 for t in self._operation_times]
            
            metrics = {
                'insertion_latency_ms': sum(operation_times_ms) / max(len(operation_times_ms), 1),
                'query_latency_ms': sum(operation_times_ms[-10:]) / max(min(len(operation_times_ms), 10), 1),
                'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
                'total_operations': len(self._operation_times),
                'insertion_count': self._insertion_count,
                'query_count': self._query_count
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to get performance metrics", error=str(e))
            return {}
    
    async def _should_cleanup(self) -> bool:
        """Check if cleanup is needed"""
        if not self.config.archive_old_experiences:
            return False
        
        current_size = await self.size()
        return current_size >= (self.config.max_size * self.config.cleanup_threshold)
    
    async def _cleanup_old_experiences(self) -> None:
        """Clean up old experiences to maintain buffer size"""
        try:
            # Calculate how many experiences to remove
            current_size = await self.size()
            target_size = int(self.config.max_size * 0.8)  # Remove 20% to avoid frequent cleanups
            experiences_to_remove = current_size - target_size
            
            if experiences_to_remove <= 0:
                return
            
            # Remove oldest experiences (lowest priority + oldest timestamps)
            async with get_database_connection() as conn:
                await conn.execute("""
                    DELETE FROM rl_experiences 
                    WHERE session_id = $1 
                    AND id IN (
                        SELECT id 
                        FROM rl_experiences 
                        WHERE session_id = $1
                        ORDER BY priority ASC, created_at ASC 
                        LIMIT $2
                    )
                """, self.session_id, experiences_to_remove)
            
            # Invalidate cache
            if self._cache_enabled:
                self._experience_cache.clear()
            
            self.logger.info("Cleaned up old experiences", 
                           removed=experiences_to_remove,
                           new_size=await self.size())
            
        except Exception as e:
            self.logger.error("Failed to cleanup old experiences", error=str(e))


def create_database_experience_buffer(config: DatabaseExperienceConfig) -> DatabaseExperienceBuffer:
    """Factory function to create database experience buffer"""
    return DatabaseExperienceBuffer(config)