"""
Unit tests for database experience buffer

Tests the DatabaseExperienceBuffer class using TDD methodology.
Tests are written before implementation to define expected behavior.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.rl_agent.experience_database import (
    DatabaseExperienceBuffer,
    DatabaseExperienceConfig,
    DatabaseExperienceError
)
from src.rl_agent.experience_replay import Experience


class TestDatabaseExperienceBuffer:
    """Test DatabaseExperienceBuffer class with TDD approach"""
    
    @pytest.fixture
    def config(self):
        """Test configuration for database experience buffer"""
        return DatabaseExperienceConfig(
            max_size=1000,
            batch_size=32,
            min_size=10,
            prioritized=True,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            cache_size=100
        )
    
    @pytest.fixture
    def sample_experience(self):
        """Create a sample experience for testing"""
        return Experience(
            state=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            action=1,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1], dtype=np.float32),
            done=False
        )
    
    @pytest.fixture
    def sample_experiences(self):
        """Create multiple sample experiences for batch testing"""
        experiences = []
        for i in range(50):
            exp = Experience(
                state=np.array([float(i), float(i+1), float(i+2)], dtype=np.float32),
                action=i % 3,
                reward=0.1 * i,
                next_state=np.array([float(i+0.1), float(i+1.1), float(i+2.1)], dtype=np.float32),
                done=(i % 10 == 9)
            )
            experiences.append(exp)
        return experiences
    
    @pytest.fixture
    async def buffer(self, config):
        """Create database experience buffer instance"""
        # This test will fail initially since the class doesn't exist
        buffer = DatabaseExperienceBuffer(config)
        await buffer.initialize()
        yield buffer
        await buffer.cleanup()

    # =============================================================================
    # INITIALIZATION AND CONFIGURATION TESTS
    # =============================================================================
    
    def test_database_experience_config_creation(self):
        """Test DatabaseExperienceConfig creation with default values"""
        config = DatabaseExperienceConfig()
        
        assert config.max_size == 10000
        assert config.batch_size == 32
        assert config.min_size == 100
        assert config.prioritized == True
        assert config.alpha == 0.6
        assert config.beta_start == 0.4
        assert config.beta_end == 1.0
        assert config.cache_size == 1000
        assert config.connection_pool_size == 10
        assert config.query_timeout == 30.0
    
    def test_database_experience_config_custom_values(self):
        """Test DatabaseExperienceConfig with custom values"""
        config = DatabaseExperienceConfig(
            max_size=5000,
            batch_size=64,
            min_size=50,
            prioritized=False,
            alpha=0.8,
            cache_size=500
        )
        
        assert config.max_size == 5000
        assert config.batch_size == 64
        assert config.min_size == 50
        assert config.prioritized == False
        assert config.alpha == 0.8
        assert config.cache_size == 500
    
    @pytest.mark.asyncio
    async def test_database_experience_buffer_initialization(self, config):
        """Test DatabaseExperienceBuffer initialization"""
        # This will fail initially - class doesn't exist yet
        buffer = DatabaseExperienceBuffer(config)
        
        assert buffer.config == config
        assert buffer.session_id is not None
        assert isinstance(buffer.session_id, uuid.UUID)
        assert buffer._cache_enabled == (config.cache_size > 0)
        assert buffer._initialized == False
        
        await buffer.initialize()
        assert buffer._initialized == True

    # =============================================================================
    # EXPERIENCE ADDITION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_add_single_experience(self, buffer, sample_experience):
        """Test adding a single experience to database"""
        # Test will fail - method doesn't exist
        initial_size = await buffer.size()
        assert initial_size == 0
        
        await buffer.add(sample_experience)
        
        new_size = await buffer.size()
        assert new_size == 1
        
        # Verify experience was stored correctly
        stored_experiences = await buffer.get_recent(limit=1)
        assert len(stored_experiences) == 1
        
        stored_exp = stored_experiences[0]
        np.testing.assert_array_equal(stored_exp['state'], sample_experience.state)
        assert stored_exp['action'] == sample_experience.action
        assert stored_exp['reward'] == sample_experience.reward
        np.testing.assert_array_equal(stored_exp['next_state'], sample_experience.next_state)
        assert stored_exp['done'] == sample_experience.done
    
    @pytest.mark.asyncio
    async def test_add_batch_experiences(self, buffer, sample_experiences):
        """Test adding batch of experiences efficiently"""
        initial_size = await buffer.size()
        
        await buffer.add_batch(sample_experiences)
        
        new_size = await buffer.size()
        assert new_size == initial_size + len(sample_experiences)
        
        # Verify all experiences were stored
        stored_experiences = await buffer.get_recent(limit=len(sample_experiences))
        assert len(stored_experiences) == len(sample_experiences)
    
    @pytest.mark.asyncio
    async def test_add_experience_with_priority(self, buffer):
        """Test adding experience with custom priority"""
        experience = Experience(
            state=np.array([1.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.array([2.0], dtype=np.float32),
            done=True
        )
        
        priority = 0.8
        await buffer.add(experience, priority=priority)
        
        # Verify priority was stored
        stored_experiences = await buffer.get_recent(limit=1)
        assert len(stored_experiences) == 1
        assert abs(stored_experiences[0]['priority'] - priority) < 1e-6
    
    @pytest.mark.asyncio
    async def test_add_experience_database_failure(self, buffer, sample_experience):
        """Test handling database failure during experience addition"""
        with patch('src.utils.database.insert_experience_batch', side_effect=Exception("Database error")):
            with pytest.raises(DatabaseExperienceError):
                await buffer.add(sample_experience)

    # =============================================================================
    # EXPERIENCE SAMPLING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_uniform_sampling(self, buffer, sample_experiences):
        """Test uniform sampling when prioritized=False"""
        # Create buffer with uniform sampling
        config_uniform = DatabaseExperienceConfig(prioritized=False, min_size=5)
        buffer_uniform = DatabaseExperienceBuffer(config_uniform)
        await buffer_uniform.initialize()
        
        # Add experiences
        await buffer_uniform.add_batch(sample_experiences[:10])
        
        # Sample experiences
        sampled = await buffer_uniform.sample()
        assert len(sampled) == config_uniform.batch_size
        
        # Verify sampled experiences have required fields
        for exp in sampled:
            assert 'state' in exp
            assert 'action' in exp
            assert 'reward' in exp
            assert 'next_state' in exp
            assert 'done' in exp
            assert 'weight' not in exp  # Uniform sampling doesn't include weights
        
        await buffer_uniform.cleanup()
    
    @pytest.mark.asyncio
    async def test_prioritized_sampling(self, buffer, sample_experiences):
        """Test prioritized sampling with importance weights"""
        # Add experiences with different priorities
        experiences_with_priorities = []
        for i, exp in enumerate(sample_experiences[:20]):
            priority = 0.1 + (i * 0.04)  # Increasing priorities
            experiences_with_priorities.append((exp, priority))
        
        for exp, priority in experiences_with_priorities:
            await buffer.add(exp, priority=priority)
        
        # Sample experiences
        sampled = await buffer.sample()
        assert len(sampled) <= buffer.config.batch_size
        
        # Verify sampled experiences have weights and indices
        for exp in sampled:
            assert 'state' in exp
            assert 'action' in exp
            assert 'reward' in exp
            assert 'weight' in exp  # Prioritized sampling includes weights
            assert 'database_id' in exp  # For priority updates
            assert 0.0 <= exp['weight'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_sampling_insufficient_experiences(self, buffer):
        """Test sampling when buffer has insufficient experiences"""
        # Add fewer experiences than min_size
        experiences = [Experience(
            state=np.array([1.0], dtype=np.float32),
            action=0,
            reward=0.5,
            next_state=np.array([2.0], dtype=np.float32),
            done=False
        )] * 5  # Less than min_size=10
        
        await buffer.add_batch(experiences)
        
        # Should raise error or return empty list
        with pytest.raises(DatabaseExperienceError, match="insufficient experiences"):
            await buffer.sample()
    
    @pytest.mark.asyncio
    async def test_can_sample_check(self, buffer, sample_experiences):
        """Test can_sample() method"""
        # Initially should be False
        assert await buffer.can_sample() == False
        
        # Add minimum required experiences
        await buffer.add_batch(sample_experiences[:buffer.config.min_size])
        
        # Now should be True
        assert await buffer.can_sample() == True

    # =============================================================================
    # PRIORITY UPDATE TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_update_priorities(self, buffer, sample_experiences):
        """Test updating experience priorities after training"""
        # Add experiences
        await buffer.add_batch(sample_experiences[:10])
        
        # Sample experiences to get database IDs
        sampled = await buffer.sample()
        
        # Prepare priority updates
        priority_updates = []
        for exp in sampled[:5]:
            priority_updates.append({
                'database_id': exp['database_id'],
                'priority': 0.95,
                'td_error': 0.8
            })
        
        # Update priorities
        updated_count = await buffer.update_priorities(priority_updates)
        assert updated_count == len(priority_updates)
        
        # Verify priorities were updated by sampling again
        # Higher priority experiences should be more likely to be sampled
        new_sampled = await buffer.sample()
        updated_ids = {upd['database_id'] for upd in priority_updates}
        sampled_ids = {exp['database_id'] for exp in new_sampled}
        
        # At least some updated experiences should appear in new sample
        overlap = len(updated_ids.intersection(sampled_ids))
        assert overlap > 0

    # =============================================================================
    # METADATA AND LIFECYCLE TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_experience_metadata_tracking(self, buffer, sample_experience):
        """Test experience metadata and lifecycle tracking"""
        metadata = {
            'episode': 1,
            'step': 10,
            'strategy': 'dqn',
            'market_conditions': {'volatility': 0.15}
        }
        
        await buffer.add(sample_experience, metadata=metadata)
        
        # Retrieve and verify metadata
        stored_experiences = await buffer.get_recent(limit=1)
        assert len(stored_experiences) == 1
        
        stored_metadata = stored_experiences[0]['metadata']
        assert stored_metadata['episode'] == metadata['episode']
        assert stored_metadata['step'] == metadata['step']
        assert stored_metadata['strategy'] == metadata['strategy']
        assert stored_metadata['market_conditions'] == metadata['market_conditions']
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_tracking(self, buffer, sample_experiences):
        """Test session lifecycle and statistics tracking"""
        # Add experiences
        await buffer.add_batch(sample_experiences)
        
        # Get session statistics
        stats = await buffer.get_session_stats()
        
        assert stats['session_id'] == str(buffer.session_id)
        assert stats['total_experiences'] == len(sample_experiences)
        assert stats['avg_reward'] > 0  # Based on sample_experiences
        assert 'total_steps' in stats
        assert 'completion_rate' in stats
    
    @pytest.mark.asyncio
    async def test_experience_lifecycle_cleanup(self, buffer, sample_experiences):
        """Test automatic cleanup of old experiences"""
        # Fill buffer beyond max_size
        large_batch = sample_experiences * 25  # 50 * 25 = 1250 > max_size=1000
        
        await buffer.add_batch(large_batch)
        
        # Verify size is maintained at max_size
        final_size = await buffer.size()
        assert final_size <= buffer.config.max_size

    # =============================================================================
    # PERFORMANCE AND CACHING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, buffer, sample_experiences):
        """Test experience caching for performance"""
        # Add experiences
        await buffer.add_batch(sample_experiences[:20])
        
        # First query should populate cache
        experiences1 = await buffer.get_recent(limit=10)
        
        # Second identical query should use cache (faster)
        import time
        start_time = time.time()
        experiences2 = await buffer.get_recent(limit=10)
        cache_time = time.time() - start_time
        
        # Verify results are identical
        assert len(experiences1) == len(experiences2)
        assert experiences1[0]['database_id'] == experiences2[0]['database_id']
        
        # Cache should be faster (this is approximate)
        assert cache_time < 0.1  # Should be very fast with cache
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, buffer, sample_experiences):
        """Test cache invalidation when new experiences are added"""
        # Add initial experiences
        await buffer.add_batch(sample_experiences[:10])
        
        # Query to populate cache
        experiences1 = await buffer.get_recent(limit=5)
        
        # Add more experiences (should invalidate cache)
        await buffer.add_batch(sample_experiences[10:15])
        
        # Query again - should reflect new experiences
        experiences2 = await buffer.get_recent(limit=5)
        
        # Results should be different
        assert experiences1 != experiences2
        
        # Newer experiences should appear first
        assert experiences2[0]['created_at'] >= experiences1[0]['created_at']
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, buffer, sample_experiences):
        """Test thread-safe concurrent access to buffer"""
        async def add_experiences_concurrently(start_idx, count):
            batch = sample_experiences[start_idx:start_idx + count]
            await buffer.add_batch(batch)
        
        # Run concurrent additions
        tasks = [
            add_experiences_concurrently(0, 10),
            add_experiences_concurrently(10, 10),
            add_experiences_concurrently(20, 10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all experiences were added safely
        final_size = await buffer.size()
        assert final_size == 30
    
    @pytest.mark.asyncio
    async def test_high_volume_operations(self, config):
        """Test high-volume experience operations (10,000+ experiences)"""
        # Create larger buffer for high-volume test
        high_volume_config = DatabaseExperienceConfig(
            max_size=15000,
            batch_size=64,
            min_size=1000,
            cache_size=2000
        )
        
        buffer = DatabaseExperienceBuffer(high_volume_config)
        await buffer.initialize()
        
        try:
            # Generate 12,000 experiences
            large_batch = []
            for i in range(12000):
                exp = Experience(
                    state=np.random.rand(10).astype(np.float32),
                    action=i % 5,
                    reward=np.random.rand(),
                    next_state=np.random.rand(10).astype(np.float32),
                    done=(i % 100 == 99)
                )
                large_batch.append(exp)
            
            # Test batch insertion performance
            import time
            start_time = time.time()
            await buffer.add_batch(large_batch)
            insertion_time = time.time() - start_time
            
            # Verify all experiences were added
            final_size = await buffer.size()
            assert final_size == len(large_batch)
            
            # Test sampling performance
            start_time = time.time()
            sampled = await buffer.sample()
            sampling_time = time.time() - start_time
            
            assert len(sampled) == high_volume_config.batch_size
            
            # Performance requirements
            insertion_rate = len(large_batch) / insertion_time
            assert insertion_rate > 1000  # Should handle >1000 experiences per second
            assert sampling_time < 0.1  # Sampling should be under 100ms
            
        finally:
            await buffer.cleanup()

    # =============================================================================
    # ERROR HANDLING AND EDGE CASES
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, config):
        """Test handling database connection failures"""
        buffer = DatabaseExperienceBuffer(config)
        
        with patch('src.utils.database.get_database_connection', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseExperienceError):
                await buffer.initialize()
    
    @pytest.mark.asyncio
    async def test_invalid_experience_data(self, buffer):
        """Test handling invalid experience data"""
        # Test with invalid state shape
        invalid_experience = Experience(
            state=np.array([], dtype=np.float32),  # Empty state
            action=1,
            reward=0.5,
            next_state=np.array([1.0], dtype=np.float32),
            done=False
        )
        
        with pytest.raises(DatabaseExperienceError, match="invalid state"):
            await buffer.add(invalid_experience)
    
    @pytest.mark.asyncio
    async def test_buffer_cleanup_on_error(self, buffer, sample_experience):
        """Test proper cleanup when operations fail"""
        # Add an experience successfully first
        await buffer.add(sample_experience)
        initial_size = await buffer.size()
        
        # Simulate database error during batch add
        invalid_batch = [sample_experience] * 10
        with patch('src.utils.database.insert_experience_batch', side_effect=Exception("Database error")):
            with pytest.raises(DatabaseExperienceError):
                await buffer.add_batch(invalid_batch)
        
        # Verify buffer state is consistent (no partial additions)
        final_size = await buffer.size()
        assert final_size == initial_size  # Should be unchanged
    
    @pytest.mark.asyncio
    async def test_memory_management_large_experiences(self, buffer):
        """Test memory management with large experience states"""
        # Create experiences with large state vectors
        large_experiences = []
        for i in range(100):
            exp = Experience(
                state=np.random.rand(1000).astype(np.float32),  # Large state
                action=i % 3,
                reward=0.1,
                next_state=np.random.rand(1000).astype(np.float32),
                done=False
            )
            large_experiences.append(exp)
        
        # Should handle large states without memory issues
        await buffer.add_batch(large_experiences)
        
        # Verify experiences were stored
        final_size = await buffer.size()
        assert final_size == len(large_experiences)
        
        # Sample should work efficiently
        sampled = await buffer.sample()
        assert len(sampled) == buffer.config.batch_size
        
        # Verify state shapes are preserved
        for exp in sampled:
            assert len(exp['state']) == 1000
            assert len(exp['next_state']) == 1000

    # =============================================================================
    # STATISTICS AND MONITORING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_buffer_statistics(self, buffer, sample_experiences):
        """Test comprehensive buffer statistics"""
        # Add experiences with different rewards
        await buffer.add_batch(sample_experiences)
        
        stats = await buffer.get_statistics()
        
        # Verify all expected statistics are present
        required_fields = [
            'total_experiences', 'avg_reward', 'max_reward', 'min_reward',
            'reward_stddev', 'avg_priority', 'completion_rate', 'session_id',
            'cache_hit_rate', 'insertion_rate', 'query_time_avg'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Verify statistical accuracy
        assert stats['total_experiences'] == len(sample_experiences)
        assert stats['avg_reward'] > 0
        assert stats['max_reward'] >= stats['avg_reward'] >= stats['min_reward']
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, buffer, sample_experiences):
        """Test performance monitoring and metrics"""
        # Add experiences and perform operations
        await buffer.add_batch(sample_experiences[:20])
        await buffer.sample()
        await buffer.sample()
        
        # Get performance metrics
        metrics = await buffer.get_performance_metrics()
        
        # Verify metrics structure
        assert 'insertion_latency_ms' in metrics
        assert 'query_latency_ms' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'total_operations' in metrics
        
        # Verify reasonable values
        assert metrics['insertion_latency_ms'] > 0
        assert metrics['query_latency_ms'] > 0
        assert 0 <= metrics['cache_hit_rate'] <= 1
        assert metrics['total_operations'] > 0


# =============================================================================
# CONFIGURATION AND ERROR CLASSES (EXPECTED TO EXIST)
# =============================================================================

class TestDatabaseExperienceConfig:
    """Test DatabaseExperienceConfig class"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config should work
        config = DatabaseExperienceConfig(
            max_size=1000,
            batch_size=32,
            min_size=10
        )
        assert config.max_size == 1000
        
        # Invalid config should raise error
        with pytest.raises(ValueError):
            DatabaseExperienceConfig(max_size=-1)
        
        with pytest.raises(ValueError):
            DatabaseExperienceConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            DatabaseExperienceConfig(min_size=-1)
    
    def test_config_relationships(self):
        """Test configuration parameter relationships"""
        # min_size should be less than max_size
        with pytest.raises(ValueError):
            DatabaseExperienceConfig(max_size=100, min_size=200)
        
        # batch_size should be reasonable
        with pytest.raises(ValueError):
            DatabaseExperienceConfig(max_size=100, batch_size=150)


class TestDatabaseExperienceError:
    """Test DatabaseExperienceError exception class"""
    
    def test_error_creation(self):
        """Test error creation and message handling"""
        error = DatabaseExperienceError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_error_chaining(self):
        """Test error chaining with underlying exceptions"""
        original_error = ValueError("Original error")
        try:
            raise DatabaseExperienceError("Database error") from original_error
        except DatabaseExperienceError as chained_error:
            assert str(chained_error) == "Database error"
            assert chained_error.__cause__ == original_error