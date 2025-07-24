"""
Real RL Model Integration Tests

Tests actual RL training pipeline, memory management, and model persistence
with real PyTorch DQN models instead of mocks.
"""

import asyncio
import gc
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List
import numpy as np
import pytest
import torch
import psutil

from src.rl_agent.base import AgentConfig, TradeAction, MarketState
from src.rl_agent.dqn_agent import DQNTradingAgent, DQNNetwork
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
from src.rl_agent.experience_replay import create_replay_buffer, ReplayBufferConfig
from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain


class TestRealRLIntegration:
    """Test real RL model integration and performance"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        return [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test",
                status=TokenStatus.DISCOVERED,
                price_usd=1.50,
                volume_24h=100000,
                market_cap=1000000
            ),
            DiscoveredToken(
                address="0x456",
                chain=Chain.ETHEREUM,
                symbol="TEST2",
                name="Test Token 2",
                discovered_at=datetime.now(),
                discovery_source="test",
                status=TokenStatus.DISCOVERED,
                price_usd=2.25,
                volume_24h=200000,
                market_cap=2000000
            )
        ]
    
    @pytest.fixture
    def historical_data(self):
        """Create realistic historical price data"""
        # Generate realistic price movements for 200 steps
        base_prices = [1.50, 2.25]
        historical = {}
        
        for i, address in enumerate(["0x123", "0x456"]):
            prices = []
            current_price = base_prices[i]
            
            for step in range(200):
                # Random walk with small volatility
                change = np.random.normal(0, 0.02)  # 2% volatility
                current_price *= (1 + change)
                current_price = max(current_price, 0.01)  # Price floor
                prices.append(current_price)
            
            historical[address] = {"prices": prices}
        
        return historical
    
    @pytest.fixture
    def real_dqn_agent(self):
        """Create real DQN agent with actual PyTorch network"""
        config = AgentConfig(
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99,
            batch_size=16  # Small batch for faster testing
        )
        return DQNTradingAgent(config)
    
    @pytest.fixture
    def real_trading_environment(self, sample_tokens, historical_data):
        """Create real trading environment"""
        config = EnvironmentConfig(
            initial_cash=10000.0,
            max_episode_steps=50,  # Short episodes for testing
            max_position_size=0.2
        )
        return TradingEnvironment(config, sample_tokens, historical_data)
    
    def test_real_dqn_network_creation(self):
        """Test that real DQN networks are created properly"""
        # Test network creation with correct parameters
        network = DQNNetwork(
            input_size=19,
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
            output_size=5
        )
        
        assert isinstance(network, torch.nn.Module)
        assert hasattr(network, 'network')  # Sequential network
        
        # Test forward pass with real tensor
        state = torch.randn(1, 19)
        output = network(state)
        
        assert output.shape == (1, 5)
        assert torch.is_tensor(output)
        assert not torch.isnan(output).any()
    
    def test_real_dqn_agent_training_step(self, real_dqn_agent, sample_tokens):
        """Test real DQN agent training with actual gradients"""
        # Create real market state
        token = sample_tokens[0]
        state = MarketState(
            token=token,
            price_usd=1.50,
            price_change_24h=2.5,
            volume_24h=100000,
            portfolio_value=10000.0,
            cash_balance=8000.0,
            current_position=0.1
        )
        
        # Record initial network parameters
        initial_params = {}
        for name, param in real_dqn_agent.q_network.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Perform training step
        action, confidence = real_dqn_agent.predict_action(state)
        
        # Verify action is valid
        assert isinstance(action, TradeAction)
        assert 0.0 <= confidence <= 1.0
        
        # Add experience and train (need multiple experiences for batch)
        next_state = MarketState(
            token=token,
            price_usd=1.52,
            price_change_24h=3.0,
            volume_24h=105000,
            portfolio_value=10200.0,
            cash_balance=7800.0,
            current_position=0.15
        )
        
        reward = 15.0
        done = False
        
        # Add multiple experiences to enable training
        for _ in range(20):  # Add enough for a batch
            real_dqn_agent.store_experience(state, action, reward, next_state, done)
        
        # Train the agent
        loss = real_dqn_agent.train_step()
        
        # Verify training occurred
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0
        
        # Verify parameters actually changed
        params_changed = False
        for name, param in real_dqn_agent.q_network.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        
        assert params_changed, "Network parameters should change after training"
    
    def test_real_trading_environment_episode(self, real_trading_environment, sample_tokens):
        """Test complete episode with real trading environment"""
        env = real_trading_environment
        
        # Reset environment
        initial_state = env.reset()
        assert isinstance(initial_state, MarketState)
        assert initial_state.portfolio_value == 10000.0
        
        # Run episode steps
        total_reward = 0.0
        steps_taken = 0
        max_steps = 20
        
        while not env.done and steps_taken < max_steps:
            # Random action for testing
            action = np.random.choice(list(TradeAction))
            token = sample_tokens[steps_taken % len(sample_tokens)]
            position_size = 0.1
            
            # Execute step
            next_state, reward, done, info = env.step(action, token, position_size)
            
            # Verify outputs
            assert isinstance(next_state, MarketState)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            # Verify info contains expected keys
            expected_keys = ['portfolio_value', 'cash', 'positions', 'episode_step']
            for key in expected_keys:
                assert key in info
            
            total_reward += reward
            steps_taken += 1
            
            if done:
                break
        
        # Verify episode ran
        assert steps_taken > 0
        assert isinstance(total_reward, float)
        
        # Verify portfolio state is valid
        final_value = env.portfolio.calculate_total_value()
        assert final_value > 0
        assert env.portfolio.cash >= 0
    
    def test_real_experience_replay_memory_management(self):
        """Test experience replay buffer memory usage"""
        config = ReplayBufferConfig(
            capacity=1000,
            batch_size=32,
            prioritized=False
        )
        buffer = create_replay_buffer(config)
        
        # Monitor memory before adding experiences
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add many experiences
        for i in range(1500):  # More than capacity
            state = np.random.randn(19).astype(np.float32)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.randn(19).astype(np.float32)
            done = np.random.choice([True, False])
            
            buffer.store(state, action, reward, next_state, done)
        
        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100, f"Memory increase too large: {memory_increase}MB"
        
        # Verify buffer maintained capacity limit
        assert len(buffer.buffer) <= config.capacity
        
        # Test sampling doesn't cause memory leaks
        for _ in range(10):
            if len(buffer.buffer) >= config.batch_size:
                batch = buffer.sample()
                assert len(batch) == 5  # state, action, reward, next_state, done
                del batch  # Explicit cleanup
        
        # Force garbage collection
        gc.collect()
    
    def test_real_model_persistence(self, real_dqn_agent):
        """Test saving and loading real PyTorch models"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_dqn.pth")
            
            # Save model
            real_dqn_agent.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Record original parameters
            original_params = {}
            for name, param in real_dqn_agent.q_network.named_parameters():
                original_params[name] = param.clone().detach()
            
            # Modify the model
            with torch.no_grad():
                for param in real_dqn_agent.q_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # Verify model was modified
            params_changed = False
            for name, param in real_dqn_agent.q_network.named_parameters():
                if not torch.allclose(original_params[name], param, atol=1e-6):
                    params_changed = True
                    break
            assert params_changed, "Model parameters should be different after modification"
            
            # Load model
            real_dqn_agent.load_model(model_path)
            
            # Verify parameters were restored
            for name, param in real_dqn_agent.q_network.named_parameters():
                assert torch.allclose(original_params[name], param, atol=1e-6), \
                    f"Parameter {name} was not restored correctly"
    
    def test_real_training_pipeline_integration(self, sample_tokens, historical_data):
        """Test complete training pipeline with real components"""
        # Create minimal training config for fast testing
        config = TrainingConfig(
            num_episodes=5,  # Very short for testing
            max_steps_per_episode=10,
            learning_rate=0.01,  # Higher LR for faster learning
            epsilon_start=0.5,  # Less exploration for stability
            epsilon_end=0.1,
            target_update_frequency=3,  # More frequent updates
            batch_size=8,  # Smaller batch
            save_frequency=10  # No saves during test
        )
        
        # Create pipeline with real components
        pipeline = DQNTrainingPipeline(config, sample_tokens, historical_data)
        
        # Verify components are real (not mocks)
        assert isinstance(pipeline.agent, DQNTradingAgent)
        assert isinstance(pipeline.environment, TradingEnvironment)
        assert hasattr(pipeline.agent.q_network, 'parameters')  # Real PyTorch network
        
        # Run training
        start_time = time.time()
        results = pipeline.train()
        training_time = time.time() - start_time
        
        # Verify training completed
        assert results['episodes_completed'] == config.num_episodes
        assert results['training_time'] > 0
        assert 'final_metrics' in results
        
        # Verify metrics are realistic
        metrics = results['final_metrics']
        assert metrics['episodes_completed'] == config.num_episodes
        assert 'mean_reward' in metrics
        assert 'mean_portfolio_value' in metrics
        
        # Training should complete reasonably quickly (< 30 seconds)
        assert training_time < 30, f"Training took too long: {training_time}s"
    
    def test_gpu_availability_and_usage(self, real_dqn_agent):
        """Test GPU usage if available"""
        if torch.cuda.is_available():
            # Move model to GPU
            device = torch.device('cuda')
            real_dqn_agent.q_network.to(device)
            real_dqn_agent.target_network.to(device)
            
            # Test forward pass on GPU
            state = torch.randn(1, 19).to(device)
            output = real_dqn_agent.q_network(state)
            
            assert output.device.type == 'cuda'
            assert not torch.isnan(output).any()
            
            # Move back to CPU for cleanup
            real_dqn_agent.q_network.to('cpu')
            real_dqn_agent.target_network.to('cpu')
        else:
            # Test that CPU-only training works
            state = torch.randn(1, 19)
            output = real_dqn_agent.q_network(state)
            assert output.device.type == 'cpu'
    
    def test_real_rl_memory_leak_detection(self, real_dqn_agent, sample_tokens):
        """Test for memory leaks in RL training loop"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create market state
        token = sample_tokens[0]
        state = MarketState(
            token=token,
            price_usd=1.50,
            volume_24h=100000,
            portfolio_value=10000.0,
            cash_balance=8000.0,
            current_position=0.1
        )
        
        # Run many prediction cycles
        for i in range(100):
            action, confidence = real_dqn_agent.predict_action(state)
            
            # Add some experiences
            next_state = MarketState(
                token=token,
                price_usd=1.50 + np.random.normal(0, 0.05),
                volume_24h=100000,
                portfolio_value=10000.0 + np.random.normal(0, 100),
                cash_balance=8000.0,
                current_position=0.1
            )
            
            real_dqn_agent.store_experience(state, action, 10.0, next_state, False)
            
            # Train periodically
            if i % 20 == 0 and len(real_dqn_agent.replay_buffer.buffer) >= real_dqn_agent.config.batch_size:
                real_dqn_agent.train_step()
            
            # Update state for next iteration
            state = next_state
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50, f"Potential memory leak: {memory_increase}MB increase"
    
    def test_real_rl_performance_benchmarks(self, sample_tokens, historical_data):
        """Benchmark real RL training performance"""
        config = TrainingConfig(
            num_episodes=10,
            max_steps_per_episode=20,
            batch_size=16
        )
        
        pipeline = DQNTrainingPipeline(config, sample_tokens, historical_data)
        
        # Benchmark training time
        start_time = time.time()
        results = pipeline.train()
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        episodes_per_second = config.num_episodes / total_time
        steps_per_second = (config.num_episodes * config.max_steps_per_episode) / total_time
        
        # Verify reasonable performance
        assert episodes_per_second > 0.1, f"Training too slow: {episodes_per_second} episodes/sec"
        assert steps_per_second > 1.0, f"Step execution too slow: {steps_per_second} steps/sec"
        
        # Log performance metrics
        print(f"\nRL Performance Benchmarks:")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Episodes per second: {episodes_per_second:.2f}")
        print(f"Steps per second: {steps_per_second:.2f}")
        print(f"Mean episode reward: {results['final_metrics']['mean_reward']:.2f}")


class TestRealRLComponentIntegration:
    """Test integration between real RL components"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        return [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test",
                status=TokenStatus.DISCOVERED,
                price_usd=1.50,
                volume_24h=100000,
                market_cap=1000000
            ),
            DiscoveredToken(
                address="0x456",
                chain=Chain.ETHEREUM,
                symbol="TEST2",
                name="Test Token 2",
                discovered_at=datetime.now(),
                discovery_source="test",
                status=TokenStatus.DISCOVERED,
                price_usd=2.25,
                volume_24h=200000,
                market_cap=2000000
            )
        ]
    
    @pytest.fixture
    def historical_data(self):
        """Create realistic historical price data"""
        # Generate realistic price movements for 200 steps
        base_prices = [1.50, 2.25]
        historical = {}
        
        for i, address in enumerate(["0x123", "0x456"]):
            prices = []
            current_price = base_prices[i]
            
            for step in range(200):
                # Random walk with small volatility
                change = np.random.normal(0, 0.02)  # 2% volatility
                current_price *= (1 + change)
                current_price = max(current_price, 0.01)  # Price floor
                prices.append(current_price)
            
            historical[address] = {"prices": prices}
        
        return historical
    
    def test_dqn_agent_environment_integration(self, sample_tokens, historical_data):
        """Test DQN agent integrated with trading environment"""
        # Create real components
        agent_config = AgentConfig(batch_size=8, learning_rate=0.01)
        agent = DQNTradingAgent(agent_config)
        
        env_config = EnvironmentConfig(max_episode_steps=15)
        env = TradingEnvironment(env_config, sample_tokens, historical_data)
        
        # Run integrated episode
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        while not env.done and steps < 10:
            # Agent makes decision
            action, confidence = agent.predict_action(state)
            
            # Execute in environment
            token = sample_tokens[steps % len(sample_tokens)]
            next_state, reward, done, info = env.step(action, token, 0.1)
            
            # Store experience in agent
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent if enough experiences
            if len(agent.replay_buffer.buffer) >= agent.config.batch_size:
                loss = agent.train_step()
                assert loss is not None
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Verify episode completed successfully
        assert steps > 0
        assert isinstance(total_reward, float)
    
    def test_experience_replay_integration(self):
        """Test experience replay with real DQN training"""
        # Create agent with prioritized replay
        config = AgentConfig(batch_size=16, learning_rate=0.001)
        agent = DQNTradingAgent(config)
        
        # Use prioritized replay
        replay_config = ReplayBufferConfig(prioritized=True, capacity=1000)
        agent.replay_buffer = create_replay_buffer(replay_config)
        
        # Add experiences with varying rewards
        for i in range(50):
            state = np.random.randn(19).astype(np.float32)
            action = TradeAction.BUY
            reward = np.random.uniform(-10, 10)  # Variable rewards
            next_state = np.random.randn(19).astype(np.float32)
            done = False
            
            agent.store_experience(state, action, reward, next_state, done)
        
        # Train with prioritized sampling
        initial_priorities = agent.replay_buffer.priorities.copy()
        loss = agent.train_step()
        
        assert loss is not None
        assert loss >= 0.0
        
        # Verify priorities were updated
        if hasattr(agent.replay_buffer, 'priorities'):
            assert not np.array_equal(initial_priorities, agent.replay_buffer.priorities)