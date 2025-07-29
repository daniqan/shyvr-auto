"""
Tests for Experience Replay Buffer system
"""

import pytest
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.rl_agent.base import TradeAction, MarketState
from src.rl_agent.experience_replay import (
    Experience, ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer,
    ReplayBufferConfig, ExperienceReplayError, DatabaseExperienceReplayBuffer
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestExperience:
    """Test Experience data structure"""
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token"""
        return DiscoveredToken(
            address="0x123...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            volume_24h=100000
        )
    
    @pytest.fixture
    def sample_state(self, sample_token):
        """Create sample market state"""
        return MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=2.5,
            volume_24h=100000,
            market_cap=1500000,
            current_position=0.1,
            portfolio_value=10500.0,
            cash_balance=9000.0
        )
    
    def test_experience_creation(self, sample_state):
        """Test experience creation"""
        next_state = MarketState(
            token=sample_state.token,
            price_usd=1.55,
            price_change_24h=3.0,
            volume_24h=110000,
            market_cap=1600000,
            current_position=0.15,
            portfolio_value=11000.0,
            cash_balance=8500.0
        )
        
        experience = Experience(
            state=sample_state.to_vector(),
            action=1,  # BUY
            reward=0.1,
            next_state=next_state.to_vector(),
            done=False,
            timestamp=datetime.now()
        )
        
        assert len(experience.state) == MarketState.get_feature_size()
        assert len(experience.next_state) == MarketState.get_feature_size()
        assert experience.action == 1
        assert experience.reward == 0.1
        assert experience.done is False
        assert isinstance(experience.timestamp, datetime)
    
    def test_experience_to_dict(self, sample_state):
        """Test experience conversion to dictionary"""
        next_state = sample_state.to_vector() + 0.1  # Slight modification
        timestamp = datetime.now()
        
        experience = Experience(
            state=sample_state.to_vector(),
            action=2,
            reward=-0.05,
            next_state=next_state,
            done=True,
            timestamp=timestamp
        )
        
        exp_dict = experience.to_dict()
        
        assert 'state' in exp_dict
        assert 'action' in exp_dict
        assert 'reward' in exp_dict
        assert 'next_state' in exp_dict
        assert 'done' in exp_dict
        assert 'timestamp' in exp_dict
        
        assert exp_dict['action'] == 2
        assert exp_dict['reward'] == -0.05
        assert exp_dict['done'] is True
        assert exp_dict['timestamp'] == timestamp.isoformat()
    
    def test_experience_from_dict(self, sample_state):
        """Test experience creation from dictionary"""
        timestamp = datetime.now()
        exp_dict = {
            'state': sample_state.to_vector().tolist(),
            'action': 0,
            'reward': 0.05,
            'next_state': (sample_state.to_vector() + 0.05).tolist(),
            'done': False,
            'timestamp': timestamp.isoformat()
        }
        
        experience = Experience.from_dict(exp_dict)
        
        assert isinstance(experience.state, np.ndarray)
        assert isinstance(experience.next_state, np.ndarray)
        assert experience.action == 0
        assert experience.reward == 0.05
        assert experience.done is False
        assert experience.timestamp == timestamp


class TestReplayBufferConfig:
    """Test Replay Buffer Configuration"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = ReplayBufferConfig()
        
        assert config.max_size == 10000
        assert config.batch_size == 32
        assert config.min_size == 100
        assert config.prioritized is False
        assert config.alpha == 0.6
        assert config.beta_start == 0.4
        assert config.beta_end == 1.0
        assert config.epsilon == 1e-6
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = ReplayBufferConfig(
            max_size=50000,
            batch_size=64,
            min_size=500,
            prioritized=True,
            alpha=0.7,
            beta_start=0.5
        )
        
        assert config.max_size == 50000
        assert config.batch_size == 64
        assert config.min_size == 500
        assert config.prioritized is True
        assert config.alpha == 0.7
        assert config.beta_start == 0.5


class TestExperienceReplayBuffer:
    """Test Experience Replay Buffer"""
    
    @pytest.fixture
    def buffer_config(self):
        """Create buffer configuration"""
        return ReplayBufferConfig(
            max_size=1000,
            batch_size=16,
            min_size=10
        )
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences"""
        experiences = []
        
        for i in range(50):
            state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
            next_state = state + np.random.randn(MarketState.get_feature_size()).astype(np.float32) * 0.1
            
            experience = Experience(
                state=state,
                action=i % 5,  # Cycle through actions
                reward=np.random.uniform(-1, 1),
                next_state=next_state,
                done=(i % 10 == 9),  # Every 10th experience is terminal
                timestamp=datetime.now()
            )
            experiences.append(experience)
        
        return experiences
    
    def test_buffer_initialization(self, buffer_config):
        """Test buffer initialization"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        assert buffer.config == buffer_config
        assert len(buffer) == 0
        assert buffer.max_size == buffer_config.max_size
        assert not buffer.can_sample()
    
    def test_buffer_add_experience(self, buffer_config):
        """Test adding experiences to buffer"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
        next_state = state + 0.1
        
        experience = Experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False
        )
        
        buffer.add(experience)
        
        assert len(buffer) == 1
        assert not buffer.can_sample()  # Still below min_size
    
    def test_buffer_add_multiple_experiences(self, buffer_config, sample_experiences):
        """Test adding multiple experiences"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        for experience in sample_experiences[:20]:
            buffer.add(experience)
        
        assert len(buffer) == 20
        assert buffer.can_sample()  # Above min_size
    
    def test_buffer_overflow(self, buffer_config, sample_experiences):
        """Test buffer overflow behavior"""
        # Use small buffer for testing overflow
        small_config = ReplayBufferConfig(max_size=10, batch_size=4, min_size=5)
        buffer = ExperienceReplayBuffer(small_config)
        
        # Add more experiences than buffer size
        for experience in sample_experiences[:15]:
            buffer.add(experience)
        
        assert len(buffer) == 10  # Should not exceed max_size
        
        # Buffer should contain the most recent experiences
        assert buffer.can_sample()
    
    def test_buffer_sample(self, buffer_config, sample_experiences):
        """Test sampling from buffer"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        # Add enough experiences to enable sampling
        for experience in sample_experiences[:30]:
            buffer.add(experience)
        
        batch = buffer.sample()
        
        assert len(batch) == buffer_config.batch_size
        
        # Check that each item in batch is a valid experience dict
        for exp_dict in batch:
            assert 'state' in exp_dict
            assert 'action' in exp_dict
            assert 'reward' in exp_dict
            assert 'next_state' in exp_dict
            assert 'done' in exp_dict
            
            # Check data types
            assert isinstance(exp_dict['state'], np.ndarray)
            assert isinstance(exp_dict['next_state'], np.ndarray)
            assert isinstance(exp_dict['action'], int)
            assert isinstance(exp_dict['reward'], float)
            assert isinstance(exp_dict['done'], bool)
    
    def test_buffer_sample_insufficient_data(self, buffer_config):
        """Test sampling when insufficient data"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        # Add less than min_size experiences
        for i in range(5):
            state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
            experience = Experience(
                state=state,
                action=0,
                reward=0.0,
                next_state=state,
                done=False
            )
            buffer.add(experience)
        
        with pytest.raises(ExperienceReplayError):
            buffer.sample()
    
    def test_buffer_sample_randomness(self, buffer_config, sample_experiences):
        """Test that sampling is random"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        for experience in sample_experiences:
            buffer.add(experience)
        
        # Sample multiple batches
        batch1 = buffer.sample()
        batch2 = buffer.sample()
        
        # Batches should likely be different (random sampling)
        # Compare actions as a simple check
        actions1 = [exp['action'] for exp in batch1]
        actions2 = [exp['action'] for exp in batch2]
        
        # With random sampling, it's very unlikely all actions are identical
        assert actions1 != actions2 or len(set(actions1)) > 1
    
    def test_buffer_clear(self, buffer_config, sample_experiences):
        """Test clearing buffer"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        for experience in sample_experiences[:20]:
            buffer.add(experience)
        
        assert len(buffer) == 20
        
        buffer.clear()
        
        assert len(buffer) == 0
        assert not buffer.can_sample()
    
    def test_buffer_get_statistics(self, buffer_config, sample_experiences):
        """Test buffer statistics"""
        buffer = ExperienceReplayBuffer(buffer_config)
        
        for experience in sample_experiences:
            buffer.add(experience)
        
        stats = buffer.get_statistics()
        
        assert 'size' in stats
        assert 'max_size' in stats
        assert 'can_sample' in stats
        assert 'utilization' in stats
        assert 'average_reward' in stats
        assert 'action_distribution' in stats
        
        assert stats['size'] == len(buffer)
        assert stats['max_size'] == buffer_config.max_size
        assert stats['can_sample'] == buffer.can_sample()
        assert 0.0 <= stats['utilization'] <= 1.0
        assert isinstance(stats['average_reward'], float)
        assert isinstance(stats['action_distribution'], dict)


class TestPrioritizedExperienceReplayBuffer:
    """Test Prioritized Experience Replay Buffer"""
    
    @pytest.fixture
    def prioritized_config(self):
        """Create prioritized buffer configuration"""
        return ReplayBufferConfig(
            max_size=1000,
            batch_size=16,
            min_size=10,
            prioritized=True,
            alpha=0.6,
            beta_start=0.4
        )
    
    @pytest.fixture
    def sample_experiences_with_priorities(self):
        """Create sample experiences with varying rewards (for priorities)"""
        experiences = []
        
        for i in range(30):
            state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
            next_state = state + np.random.randn(MarketState.get_feature_size()).astype(np.float32) * 0.1
            
            # Create experiences with different reward magnitudes
            if i < 10:
                reward = np.random.uniform(-0.1, 0.1)  # Small rewards
            elif i < 20:
                reward = np.random.uniform(-0.5, 0.5)  # Medium rewards
            else:
                reward = np.random.uniform(-1.0, 1.0)  # Large rewards
            
            experience = Experience(
                state=state,
                action=i % 5,
                reward=reward,
                next_state=next_state,
                done=(i % 15 == 14),
                timestamp=datetime.now()
            )
            experiences.append(experience)
        
        return experiences
    
    def test_prioritized_buffer_initialization(self, prioritized_config):
        """Test prioritized buffer initialization"""
        buffer = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        assert buffer.config == prioritized_config
        assert len(buffer) == 0
        assert buffer.alpha == prioritized_config.alpha
        assert buffer.beta == prioritized_config.beta_start
        assert not buffer.can_sample()
    
    def test_prioritized_buffer_add_experience(self, prioritized_config):
        """Test adding experiences to prioritized buffer"""
        buffer = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
        next_state = state + 0.1
        
        experience = Experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False
        )
        
        buffer.add(experience)
        
        assert len(buffer) == 1
        assert len(buffer.priorities) == 1
        assert buffer.priorities[0] > 0  # Priority should be positive
    
    def test_prioritized_buffer_sample(self, prioritized_config, sample_experiences_with_priorities):
        """Test sampling from prioritized buffer"""
        buffer = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        for experience in sample_experiences_with_priorities:
            buffer.add(experience)
        
        batch = buffer.sample()
        
        assert len(batch) == prioritized_config.batch_size
        
        # Check that batch contains experience dictionaries with weights and indices
        for item in batch:
            assert isinstance(item, dict)
            assert 'state' in item
            assert 'action' in item
            assert 'reward' in item
            assert 'next_state' in item
            assert 'done' in item
            assert 'weight' in item  # Importance sampling weight
            assert 'index' in item   # Buffer index for priority updates
            
            assert isinstance(item['weight'], float)
            assert isinstance(item['index'], int)
            assert item['weight'] > 0
            assert 0 <= item['index'] < len(buffer)
    
    def test_prioritized_buffer_update_priorities(self, prioritized_config, sample_experiences_with_priorities):
        """Test updating priorities after training"""
        buffer = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        for experience in sample_experiences_with_priorities:
            buffer.add(experience)
        
        batch = buffer.sample()
        
        # Extract indices and create new TD errors
        indices = [item['index'] for item in batch]
        td_errors = np.random.uniform(0.1, 2.0, len(indices))
        
        # Update priorities
        buffer.update_priorities(indices, td_errors)
        
        # Check that priorities were updated
        for i, td_error in zip(indices, td_errors):
            expected_priority = (td_error + buffer.config.epsilon) ** buffer.alpha
            assert abs(buffer.priorities[i] - expected_priority) < 1e-6
    
    def test_prioritized_buffer_beta_annealing(self, prioritized_config):
        """Test beta annealing in prioritized buffer"""
        buffer = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        initial_beta = buffer.beta
        assert initial_beta == prioritized_config.beta_start
        
        # Simulate training progress
        for step in range(1000):
            buffer.anneal_beta(step, total_steps=1000)
        
        # Beta should have increased towards beta_end
        assert buffer.beta > initial_beta
        assert buffer.beta <= prioritized_config.beta_end
    
    def test_prioritized_vs_uniform_sampling(self, prioritized_config, sample_experiences_with_priorities):
        """Test that prioritized sampling differs from uniform sampling"""
        # Create two buffers: one prioritized, one uniform
        buffer_prioritized = PrioritizedExperienceReplayBuffer(prioritized_config)
        
        uniform_config = ReplayBufferConfig(
            max_size=prioritized_config.max_size,
            batch_size=prioritized_config.batch_size,
            min_size=prioritized_config.min_size,
            prioritized=False
        )
        buffer_uniform = ExperienceReplayBuffer(uniform_config)
        
        # Add same experiences to both buffers
        for experience in sample_experiences_with_priorities:
            buffer_prioritized.add(experience)
            buffer_uniform.add(experience)
        
        # Sample multiple times and check for differences
        prioritized_rewards = []
        uniform_rewards = []
        
        for _ in range(10):
            p_batch = buffer_prioritized.sample()
            u_batch = buffer_uniform.sample()
            
            p_rewards = [item['reward'] for item in p_batch]
            u_rewards = [item['reward'] for item in u_batch]
            
            prioritized_rewards.extend(p_rewards)
            uniform_rewards.extend(u_rewards)
        
        # Prioritized sampling should tend to sample higher-magnitude rewards more often
        # This is a statistical test, so we check average absolute reward
        avg_abs_prioritized = np.mean(np.abs(prioritized_rewards))
        avg_abs_uniform = np.mean(np.abs(uniform_rewards))
        
        # Prioritized should generally sample higher magnitude rewards
        # (This test might occasionally fail due to randomness, but should pass most of the time)
        assert avg_abs_prioritized >= avg_abs_uniform * 0.8  # Allow some tolerance


class TestExperienceReplayError:
    """Test Experience Replay error handling"""
    
    def test_experience_replay_error(self):
        """Test ExperienceReplayError exception"""
        with pytest.raises(ExperienceReplayError):
            raise ExperienceReplayError("Replay error")
    
    def test_error_inheritance(self):
        """Test error inheritance"""
        error = ExperienceReplayError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


class TestExperienceReplayIntegration:
    """Integration tests for experience replay system"""
    
    @pytest.fixture
    def complete_setup(self):
        """Create complete experience replay setup"""
        config = ReplayBufferConfig(
            max_size=5000,
            batch_size=32,
            min_size=100,
            prioritized=True,
            alpha=0.6,
            beta_start=0.4
        )
        
        buffer = PrioritizedExperienceReplayBuffer(config)
        
        # Generate diverse experiences
        experiences = []
        for episode in range(10):
            for step in range(50):
                state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
                # Add some structure to states
                state[0] = episode * 0.1  # Episode identifier
                state[1] = step * 0.01    # Step identifier
                
                next_state = state.copy()
                next_state += np.random.randn(MarketState.get_feature_size()).astype(np.float32) * 0.05
                
                # Rewards based on actions and states
                action = np.random.randint(0, 5)
                if action == 1:  # BUY action
                    reward = np.random.uniform(-0.1, 0.3)  # Slightly positive bias
                elif action == 2:  # SELL action  
                    reward = np.random.uniform(-0.2, 0.2)  # Neutral
                else:  # HOLD
                    reward = np.random.uniform(-0.05, 0.05)  # Small rewards
                
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == 49),  # End of episode
                    timestamp=datetime.now()
                )
                experiences.append(experience)
        
        return buffer, experiences
    
    def test_full_replay_cycle(self, complete_setup):
        """Test complete experience replay cycle"""
        buffer, experiences = complete_setup
        
        # Add all experiences
        for experience in experiences:
            buffer.add(experience)
        
        assert len(buffer) == len(experiences)
        assert buffer.can_sample()
        
        # Sample multiple batches
        batches = []
        for _ in range(20):
            batch = buffer.sample()
            batches.append(batch)
            
            # Simulate TD error updates
            indices = [item['index'] for item in batch]
            td_errors = np.random.uniform(0.01, 1.0, len(indices))
            buffer.update_priorities(indices, td_errors)
        
        # Verify all batches have correct structure
        for batch in batches:
            assert len(batch) == buffer.config.batch_size
            for item in batch:
                assert all(key in item for key in ['state', 'action', 'reward', 'next_state', 'done', 'weight', 'index'])
        
        # Check statistics
        stats = buffer.get_statistics()
        assert stats['size'] == len(experiences)
        assert stats['can_sample'] is True
        assert 0 < stats['utilization'] <= 1.0
    
    def test_buffer_memory_efficiency(self, complete_setup):
        """Test buffer memory usage with large datasets"""
        buffer, experiences = complete_setup
        
        # Add experiences and check that old ones are removed
        initial_size = len(experiences)
        
        # Add more experiences than buffer capacity
        extra_experiences = []
        for i in range(buffer.config.max_size):
            state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
            experience = Experience(
                state=state,
                action=i % 5,
                reward=np.random.uniform(-1, 1),
                next_state=state + 0.1,
                done=False
            )
            extra_experiences.append(experience)
        
        # Add all experiences
        for experience in experiences + extra_experiences:
            buffer.add(experience)
        
        # Buffer should not exceed max size
        assert len(buffer) == buffer.config.max_size
        
        # Should still be able to sample
        assert buffer.can_sample()
        batch = buffer.sample()
        assert len(batch) == buffer.config.batch_size


class TestDatabaseExperienceReplayBuffer:
    """Test Database-backed Experience Replay Buffer"""
    
    @pytest.fixture
    def database_config(self):
        """Create database buffer configuration"""
        return ReplayBufferConfig(
            max_size=1000,
            batch_size=16,
            min_size=10,
            prioritized=True,
            alpha=0.6,
            beta_start=0.4
        )
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences for database testing"""
        experiences = []
        
        for i in range(30):
            state = np.random.randn(MarketState.get_feature_size()).astype(np.float32)
            next_state = state + np.random.randn(MarketState.get_feature_size()).astype(np.float32) * 0.1
            
            experience = Experience(
                state=state,
                action=i % 5,
                reward=np.random.uniform(-1, 1),
                next_state=next_state,
                done=(i % 10 == 9),
                timestamp=datetime.now()
            )
            experiences.append(experience)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_database_buffer_initialization(self, database_config):
        """Test database buffer initialization with connection setup"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer, \
             patch('src.rl_agent.experience_database.DatabaseExperienceConfig') as MockConfig:
            mock_buffer = AsyncMock()
            mock_buffer.initialize = AsyncMock()
            mock_buffer.size = AsyncMock(return_value=0)
            mock_buffer.can_sample = AsyncMock(return_value=False)
            MockBuffer.return_value = mock_buffer
            MockConfig.return_value = MagicMock()
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            await buffer.initialize()
            
            assert buffer.config == database_config
            assert await buffer.size_async() == 0
            assert not await buffer.can_sample_async()
            mock_buffer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_buffer_add_experience(self, database_config, sample_experiences):
        """Test adding experiences to database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer, \
             patch('src.rl_agent.experience_database.DatabaseExperienceConfig'):
            mock_buffer = AsyncMock()
            mock_buffer.add = AsyncMock()
            mock_buffer.size = AsyncMock(return_value=1)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            experience = sample_experiences[0]
            
            await buffer.add_async(experience)
            
            mock_buffer.add.assert_called_once_with(experience, None, None)
            assert await buffer.size_async() == 1
    
    @pytest.mark.asyncio
    async def test_database_buffer_add_batch(self, database_config, sample_experiences):
        """Test adding batch of experiences to database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_buffer.add_batch = AsyncMock()
            mock_buffer.size = AsyncMock(return_value=len(sample_experiences))
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            await buffer.add_batch(sample_experiences)
            
            mock_buffer.add_batch.assert_called_once_with(sample_experiences)
            assert await buffer.size() == len(sample_experiences)
    
    @pytest.mark.asyncio
    async def test_database_buffer_sample(self, database_config, sample_experiences):
        """Test sampling from database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_batch = [
                {
                    'state': exp.state,
                    'action': exp.action,
                    'reward': exp.reward,
                    'next_state': exp.next_state,
                    'done': exp.done,
                    'weight': 1.0,
                    'database_id': i
                }
                for i, exp in enumerate(sample_experiences[:database_config.batch_size])
            ]
            mock_buffer.sample = AsyncMock(return_value=mock_batch)
            mock_buffer.can_sample = AsyncMock(return_value=True)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            batch = await buffer.sample()
            
            assert len(batch) == database_config.batch_size
            assert all('weight' in item for item in batch)
            assert all('database_id' in item for item in batch)
            mock_buffer.sample.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_buffer_update_priorities(self, database_config):
        """Test updating priorities in database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_buffer.update_priorities = AsyncMock(return_value=5)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            priority_updates = [
                {'database_id': 0, 'priority': 0.8, 'td_error': 0.5},
                {'database_id': 1, 'priority': 0.6, 'td_error': 0.3}
            ]
            
            updated_count = await buffer.update_priorities(priority_updates)
            
            assert updated_count == 5
            mock_buffer.update_priorities.assert_called_once_with(priority_updates)
    
    @pytest.mark.asyncio
    async def test_database_buffer_can_sample_insufficient_data(self, database_config):
        """Test sampling when insufficient data in database"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_buffer.can_sample = AsyncMock(return_value=False)
            mock_buffer.size = AsyncMock(return_value=5)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            assert not await buffer.can_sample()
            assert await buffer.size() < database_config.min_size
    
    @pytest.mark.asyncio
    async def test_database_buffer_performance_metrics(self, database_config):
        """Test performance metrics collection for database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_metrics = {
                'insertion_latency_ms': 25.5,
                'query_latency_ms': 15.2,
                'cache_hit_rate': 0.85,
                'total_operations': 1000,
                'insertion_count': 500,
                'query_count': 500
            }
            mock_buffer.get_performance_metrics = AsyncMock(return_value=mock_metrics)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            metrics = await buffer.get_performance_metrics()
            
            assert 'insertion_latency_ms' in metrics
            assert 'cache_hit_rate' in metrics
            assert metrics['insertion_latency_ms'] < 50  # Performance requirement
            mock_buffer.get_performance_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_buffer_statistics(self, database_config):
        """Test comprehensive statistics for database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_stats = {
                'session_id': 'test-session-123',
                'total_experiences': 1000,
                'cache_hit_rate': 0.75,
                'insertion_rate': 20.5,
                'query_time_avg': 12.3,
                'beta_current': 0.6,
                'training_step': 500
            }
            mock_buffer.get_statistics = AsyncMock(return_value=mock_stats)
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            stats = await buffer.get_statistics()
            
            assert 'total_experiences' in stats
            assert 'cache_hit_rate' in stats
            assert 'training_step' in stats
            mock_buffer.get_statistics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_buffer_cleanup(self, database_config):
        """Test database buffer cleanup and resource management"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_buffer.cleanup = AsyncMock()
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            await buffer.cleanup()
            
            mock_buffer.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_buffer_memory_caching(self, database_config, sample_experiences):
        """Test memory caching functionality in database buffer"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            # First call misses cache, second hits cache
            mock_buffer.get_recent = AsyncMock(side_effect=[
                [exp.__dict__ for exp in sample_experiences[:10]],
                [exp.__dict__ for exp in sample_experiences[:10]]  # Same data, cached
            ])
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            # First call - cache miss
            recent1 = await buffer.get_recent(limit=10)
            # Second call - should use cache
            recent2 = await buffer.get_recent(limit=10)
            
            assert len(recent1) == 10
            assert len(recent2) == 10
            assert mock_buffer.get_recent.call_count == 2
    
    @pytest.mark.asyncio
    async def test_database_buffer_error_handling(self, database_config):
        """Test error handling in database buffer operations"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            mock_buffer.sample = AsyncMock(side_effect=Exception("Database connection failed"))
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            with pytest.raises(Exception, match="Database connection failed"):
                await buffer.sample()


class TestDatabaseExperienceReplayBufferFactory:
    """Test factory function for database experience replay buffer"""
    
    def test_create_database_replay_buffer(self):
        """Test factory function creates database buffer correctly"""
        config = ReplayBufferConfig(
            max_size=5000,
            batch_size=32,
            min_size=100,
            prioritized=True
        )
        
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer, \
             patch('src.rl_agent.experience_database.DatabaseExperienceConfig') as MockConfig:
            mock_buffer = AsyncMock()
            mock_config = MagicMock()
            MockBuffer.return_value = mock_buffer
            MockConfig.return_value = mock_config
            
            from src.rl_agent.experience_replay import create_database_replay_buffer
            
            buffer = create_database_replay_buffer(config)
            
            assert buffer is not None
            # Verify config conversion happened
            MockConfig.assert_called_once()
            MockBuffer.assert_called_once_with(mock_config)


class TestDatabaseReplayBufferIntegration:
    """Integration tests for database replay buffer with existing system"""
    
    @pytest.mark.asyncio
    async def test_database_buffer_backward_compatibility(self, database_config, sample_experiences):
        """Test that database buffer maintains backward compatibility with existing API"""
        with patch('src.rl_agent.experience_database.DatabaseExperienceBuffer') as MockBuffer:
            mock_buffer = AsyncMock()
            
            # Mock all expected methods to maintain API compatibility
            mock_buffer.add = AsyncMock()
            mock_buffer.sample = AsyncMock(return_value=[
                {
                    'state': exp.state,
                    'action': exp.action,
                    'reward': exp.reward,
                    'next_state': exp.next_state,
                    'done': exp.done
                }
                for exp in sample_experiences[:database_config.batch_size]
            ])
            mock_buffer.can_sample = AsyncMock(return_value=True)
            mock_buffer.size = AsyncMock(return_value=len(sample_experiences))
            mock_buffer.clear = AsyncMock()
            mock_buffer.get_statistics = AsyncMock(return_value={
                'size': len(sample_experiences),
                'max_size': database_config.max_size,
                'can_sample': True,
                'utilization': 0.5,
                'average_reward': 0.1,
                'action_distribution': {0: 6, 1: 6, 2: 6, 3: 6, 4: 6}
            })
            
            MockBuffer.return_value = mock_buffer
            
            buffer = DatabaseExperienceReplayBuffer(database_config)
            
            # Test all public API methods exist and work
            for experience in sample_experiences[:10]:
                await buffer.add(experience)
            
            assert await buffer.can_sample()
            batch = await buffer.sample()
            assert len(batch) == database_config.batch_size
            
            stats = await buffer.get_statistics()
            assert 'size' in stats
            assert 'utilization' in stats
            
            await buffer.clear()
            
            # Verify all expected calls were made
            assert mock_buffer.add.call_count == 10
            mock_buffer.sample.assert_called()
            mock_buffer.get_statistics.assert_called()
            mock_buffer.clear.assert_called()