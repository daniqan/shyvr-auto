"""
Tests for DQN Trading Agent implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any

from src.rl_agent.base import (
    TradeAction, MarketState, AgentConfig, ModelType, 
    RewardMetrics, TradingResult
)
from src.rl_agent.dqn_agent import (
    DQNNetwork, DQNTradingAgent, DQNTrainingError
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestDQNNetwork:
    """Test DQN neural network implementation"""
    
    @pytest.fixture
    def network_config(self):
        """Create network configuration"""
        return {
            'input_size': 19,
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'output_size': 5  # Number of trading actions
        }
    
    def test_dqn_network_creation(self, network_config):
        """Test DQN network creation"""
        network = DQNNetwork(**network_config)
        
        assert isinstance(network, nn.Module)
        assert network.input_size == 19
        assert network.hidden_size == 256
        assert network.num_layers == 3
        assert network.dropout_rate == 0.1
        assert network.output_size == 5
    
    def test_dqn_network_forward_pass(self, network_config):
        """Test forward pass through network"""
        network = DQNNetwork(**network_config)
        batch_size = 32
        
        # Create sample input tensor
        x = torch.randn(batch_size, network_config['input_size'])
        
        # Forward pass
        output = network(x)
        
        assert output.shape == (batch_size, network_config['output_size'])
        assert torch.all(torch.isfinite(output))  # No NaN or infinite values
    
    def test_dqn_network_single_input(self, network_config):
        """Test network with single state input"""
        network = DQNNetwork(**network_config)
        
        # Single state input
        x = torch.randn(1, network_config['input_size'])
        output = network(x)
        
        assert output.shape == (1, network_config['output_size'])
        assert torch.all(torch.isfinite(output))
    
    def test_dqn_network_gradients(self, network_config):
        """Test that network produces gradients"""
        network = DQNNetwork(**network_config)
        
        x = torch.randn(1, network_config['input_size'], requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in network.parameters():
            assert param.grad is not None
    
    def test_dqn_network_different_architectures(self):
        """Test different network architectures"""
        # Smaller network
        small_config = {
            'input_size': 10,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'output_size': 3
        }
        small_network = DQNNetwork(**small_config)
        
        x = torch.randn(1, 10)
        output = small_network(x)
        assert output.shape == (1, 3)
        
        # Larger network
        large_config = {
            'input_size': 50,
            'hidden_size': 512,
            'num_layers': 4,
            'dropout': 0.05,
            'output_size': 10
        }
        large_network = DQNNetwork(**large_config)
        
        x = torch.randn(1, 50)
        output = large_network(x)
        assert output.shape == (1, 10)


class TestDQNTradingAgent:
    """Test DQN Trading Agent implementation"""
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration"""
        return AgentConfig(
            model_type=ModelType.DQN,
            hidden_size=128,
            num_layers=2,
            learning_rate=1e-3,
            batch_size=16,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=500
        )
    
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
    def sample_market_state(self, sample_token):
        """Create sample market state"""
        return MarketState(
            token=sample_token,
            price_usd=1.50,
            price_change_24h=2.5,
            volume_24h=100000,
            market_cap=1500000,
            rsi=55.0,
            macd=0.02,
            current_position=0.0,
            portfolio_value=10000.0,
            cash_balance=10000.0
        )
    
    def test_dqn_agent_initialization(self, agent_config):
        """Test DQN agent initialization"""
        agent = DQNTradingAgent(agent_config)
        
        assert agent.config == agent_config
        assert isinstance(agent.q_network, DQNNetwork)
        assert isinstance(agent.target_network, DQNNetwork)
        assert isinstance(agent.optimizer, torch.optim.Adam)
        assert agent.epsilon == agent_config.epsilon_start
        assert agent.steps_done == 0
        assert not agent.is_trained
    
    def test_dqn_agent_network_architecture(self, agent_config):
        """Test that networks have correct architecture"""
        agent = DQNTradingAgent(agent_config)
        
        # Check main network
        assert agent.q_network.input_size == MarketState.get_feature_size()
        assert agent.q_network.hidden_size == agent_config.hidden_size
        assert agent.q_network.num_layers == agent_config.num_layers
        assert agent.q_network.output_size == len(TradeAction)
        
        # Check target network has same architecture
        assert agent.target_network.input_size == agent.q_network.input_size
        assert agent.target_network.hidden_size == agent.q_network.hidden_size
        assert agent.target_network.output_size == agent.q_network.output_size
    
    @pytest.mark.asyncio
    async def test_predict_action_exploration(self, agent_config, sample_market_state):
        """Test action prediction during exploration phase"""
        agent = DQNTradingAgent(agent_config)
        
        # During high epsilon, should mostly explore (random actions)
        agent.epsilon = 0.9
        
        actions = []
        confidences = []
        
        # Run multiple predictions to check for randomness
        for _ in range(20):
            action, confidence = await agent.predict_action(sample_market_state)
            actions.append(action)
            confidences.append(confidence)
        
        # Should have variety in actions due to exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should see different actions
        
        # All actions should be valid
        for action in actions:
            assert isinstance(action, TradeAction)
        
        # Confidences should be reasonable
        for confidence in confidences:
            assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_action_exploitation(self, agent_config, sample_market_state):
        """Test action prediction during exploitation phase"""
        agent = DQNTradingAgent(agent_config)
        
        # During low epsilon, should mostly exploit (use network)
        agent.epsilon = 0.05
        
        # Make multiple predictions
        actions = []
        for _ in range(10):
            action, confidence = await agent.predict_action(sample_market_state)
            actions.append(action)
        
        # Should be more consistent due to low exploration
        # (though network is untrained, so may still vary)
        for action in actions:
            assert isinstance(action, TradeAction)
    
    def test_epsilon_decay(self, agent_config):
        """Test epsilon decay mechanism"""
        agent = DQNTradingAgent(agent_config)
        
        initial_epsilon = agent.epsilon
        assert initial_epsilon == agent_config.epsilon_start
        
        # Simulate many steps
        for _ in range(1000):
            agent._update_epsilon()
        
        # Epsilon should have decayed
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent_config.epsilon_end
    
    def test_state_to_tensor(self, agent_config, sample_market_state):
        """Test state conversion to tensor"""
        agent = DQNTradingAgent(agent_config)
        
        tensor = agent._state_to_tensor(sample_market_state)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, MarketState.get_feature_size())
        assert tensor.dtype == torch.float32
        assert torch.all(torch.isfinite(tensor))
    
    def test_action_to_index_and_back(self, agent_config):
        """Test action to index conversion and back"""
        agent = DQNTradingAgent(agent_config)
        
        # Test all actions
        for action in TradeAction:
            index = agent._action_to_index(action)
            assert 0 <= index < len(TradeAction)
            
            recovered_action = agent._index_to_action(index)
            assert recovered_action == action
    
    @pytest.mark.asyncio
    async def test_train_step_with_experiences(self, agent_config, sample_market_state, sample_token):
        """Test training step with sample experiences"""
        agent = DQNTradingAgent(agent_config)
        
        # Create sample experiences
        next_state = MarketState(
            token=sample_token,
            price_usd=1.55,
            price_change_24h=3.0,
            volume_24h=110000,
            market_cap=1600000,
            current_position=0.1,
            portfolio_value=10050.0
        )
        
        experiences = [
            {
                'state': sample_market_state.to_vector(),
                'action': 1,  # BUY action index
                'reward': 0.1,
                'next_state': next_state.to_vector(),
                'done': False
            }
        ] * agent_config.batch_size  # Repeat to make a full batch
        
        # Train step
        metrics = await agent.train_step(experiences)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'q_value_mean' in metrics
        assert 'epsilon' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_train_step_returns_td_errors(self, agent_config, sample_market_state, sample_token):
        """Test that training step returns TD errors for prioritized replay"""
        agent = DQNTradingAgent(agent_config)
        
        # Create sample experiences with indices (for prioritized replay)
        next_state = MarketState(
            token=sample_token,
            price_usd=1.55,
            price_change_24h=3.0,
            volume_24h=110000,
            market_cap=1600000,
            current_position=0.1,
            portfolio_value=10050.0
        )
        
        experiences = []
        for i in range(agent_config.batch_size):
            exp = {
                'state': sample_market_state.to_vector(),
                'action': 1,  # BUY action index
                'reward': 0.1 + i * 0.01,  # Varying rewards
                'next_state': next_state.to_vector(),
                'done': False,
                'index': i,  # Buffer index for prioritized replay
                'weight': 1.0  # Importance sampling weight
            }
            experiences.append(exp)
        
        # Train step
        metrics = await agent.train_step(experiences)
        
        # Should return TD errors for prioritized replay buffer updates
        assert isinstance(metrics, dict)
        assert 'td_errors' in metrics
        assert 'indices' in metrics
        assert isinstance(metrics['td_errors'], list)
        assert isinstance(metrics['indices'], list)
        assert len(metrics['td_errors']) == agent_config.batch_size
        assert len(metrics['indices']) == agent_config.batch_size
        
        # TD errors should be numeric values
        for td_error in metrics['td_errors']:
            assert isinstance(td_error, (int, float))
            assert td_error >= 0.0  # TD errors should be absolute values
        
        # Indices should match the input experience indices
        for i, idx in enumerate(metrics['indices']):
            assert idx == i
    
    @pytest.mark.asyncio
    async def test_train_step_insufficient_batch(self, agent_config, sample_market_state):
        """Test training step with insufficient batch size"""
        agent = DQNTradingAgent(agent_config)
        
        # Too few experiences
        experiences = [
            {
                'state': sample_market_state.to_vector(),
                'action': 1,
                'reward': 0.1,
                'next_state': sample_market_state.to_vector(),
                'done': False
            }
        ]  # Only 1 experience, need batch_size
        
        with pytest.raises(DQNTrainingError):
            await agent.train_step(experiences)
    
    def test_update_target_network(self, agent_config):
        """Test target network update"""
        agent = DQNTradingAgent(agent_config)
        
        # Modify main network weights slightly
        with torch.no_grad():
            for param in agent.q_network.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Get target network state before update
        target_params_before = [param.clone() for param in agent.target_network.parameters()]
        
        # Update target network
        agent.update_target_network()
        
        # Check that target network was updated
        target_params_after = list(agent.target_network.parameters())
        main_params = list(agent.q_network.parameters())
        
        for target_before, target_after, main_param in zip(
            target_params_before, target_params_after, main_params
        ):
            # Target network should now match main network
            assert torch.allclose(target_after, main_param, atol=1e-6)
            # And should be different from before (unless networks were identical)
    
    def test_save_and_load_model(self, agent_config, tmp_path):
        """Test model saving and loading"""
        agent = DQNTradingAgent(agent_config)
        
        # Train a bit to change weights
        dummy_input = torch.randn(1, MarketState.get_feature_size())
        output = agent.q_network(dummy_input)
        loss = output.sum()
        loss.backward()
        agent.optimizer.step()
        
        # Save model
        model_path = tmp_path / "test_dqn_model.pth"
        success = agent.save_model(str(model_path))
        assert success
        assert model_path.exists()
        
        # Create new agent and load model
        new_agent = DQNTradingAgent(agent_config)
        load_success = new_agent.load_model(str(model_path))
        assert load_success
        assert new_agent.is_trained
        
        # Check that weights match
        for orig_param, loaded_param in zip(
            agent.q_network.parameters(), new_agent.q_network.parameters()
        ):
            assert torch.allclose(orig_param, loaded_param, atol=1e-6)
    
    def test_save_model_invalid_path(self, agent_config):
        """Test model saving with invalid path"""
        agent = DQNTradingAgent(agent_config)
        
        # Try to save to invalid path
        success = agent.save_model("/invalid/path/model.pth")
        assert not success
    
    def test_load_model_nonexistent_file(self, agent_config):
        """Test loading nonexistent model file"""
        agent = DQNTradingAgent(agent_config)
        
        success = agent.load_model("/nonexistent/model.pth")
        assert not success
        assert not agent.is_trained
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent_config):
        """Test agent health check"""
        agent = DQNTradingAgent(agent_config)
        
        health = await agent.health_check()
        
        assert isinstance(health, dict)
        assert 'is_trained' in health
        assert 'training_episodes' in health
        assert 'meets_targets' in health
        assert 'performance_metrics' in health
        assert 'config' in health
        assert 'epsilon' in health
        assert 'steps_done' in health
        
        assert health['is_trained'] is False
        assert health['epsilon'] == agent_config.epsilon_start
        assert health['steps_done'] == 0
    
    def test_get_q_values(self, agent_config, sample_market_state):
        """Test getting Q-values for a state"""
        agent = DQNTradingAgent(agent_config)
        
        q_values = agent.get_q_values(sample_market_state)
        
        assert isinstance(q_values, torch.Tensor)
        assert q_values.shape == (len(TradeAction),)
        assert torch.all(torch.isfinite(q_values))
    
    def test_different_model_types(self):
        """Test creating agents with different model types"""
        # DQN
        dqn_config = AgentConfig(model_type=ModelType.DQN)
        dqn_agent = DQNTradingAgent(dqn_config)
        assert dqn_agent.config.model_type == ModelType.DQN
        
        # Double DQN should also work with DQNTradingAgent
        ddqn_config = AgentConfig(model_type=ModelType.DDQN)
        ddqn_agent = DQNTradingAgent(ddqn_config)
        assert ddqn_agent.config.model_type == ModelType.DDQN


class TestDQNTrainingError:
    """Test DQN training error exception"""
    
    def test_dqn_training_error(self):
        """Test DQNTrainingError exception"""
        with pytest.raises(DQNTrainingError):
            raise DQNTrainingError("Training failed")
    
    def test_dqn_training_error_inheritance(self):
        """Test that DQNTrainingError inherits from correct base"""
        from src.rl_agent.base import RLTrainingError
        
        error = DQNTrainingError("Test error")
        assert isinstance(error, RLTrainingError)
        assert str(error) == "Test error"


class TestDQNIntegration:
    """Integration tests for DQN agent components"""
    
    @pytest.fixture
    def agent_config(self):
        """Create integration test configuration"""
        return AgentConfig(
            model_type=ModelType.DQN,
            hidden_size=64,  # Smaller for faster tests
            num_layers=2,
            learning_rate=1e-3,
            batch_size=8,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=100,
            target_update_frequency=10
        )
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token"""
        return DiscoveredToken(
            address="0x456...",
            symbol="INTEG",
            name="Integration Token",
            chain=Chain.SOLANA,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=2.00,
            volume_24h=200000
        )
    
    @pytest.mark.asyncio
    async def test_full_training_episode(self, agent_config, sample_token):
        """Test a complete training episode"""
        agent = DQNTradingAgent(agent_config)
        
        # Create a sequence of market states
        states = []
        for i in range(5):
            state = MarketState(
                token=sample_token,
                price_usd=2.00 + i * 0.01,
                price_change_24h=1.0 + i * 0.5,
                volume_24h=200000 + i * 10000,
                market_cap=2000000 + i * 50000,
                current_position=i * 0.02,
                portfolio_value=10000 + i * 100,
                rsi=50 + i * 2,
                macd=0.01 * i
            )
            states.append(state)
        
        # Simulate episode
        experiences = []
        for i in range(len(states) - 1):
            action, confidence = await agent.predict_action(states[i])
            
            # Create experience
            experience = {
                'state': states[i].to_vector(),
                'action': agent._action_to_index(action),
                'reward': np.random.uniform(-0.1, 0.1),
                'next_state': states[i + 1].to_vector(),
                'done': i == len(states) - 2
            }
            experiences.append(experience)
        
        # Train if we have enough experiences
        if len(experiences) >= agent_config.batch_size:
            # Pad experiences to batch size
            while len(experiences) < agent_config.batch_size:
                experiences.append(experiences[-1])
            
            metrics = await agent.train_step(experiences)
            assert 'loss' in metrics
            assert metrics['loss'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_epsilon_annealing_during_training(self, agent_config):
        """Test that epsilon decreases during training"""
        agent = DQNTradingAgent(agent_config)
        
        initial_epsilon = agent.epsilon
        
        # Simulate training steps
        for step in range(200):
            agent._update_epsilon()
            
            # Epsilon should be decreasing
            assert agent.epsilon <= initial_epsilon
            assert agent.epsilon >= agent_config.epsilon_end
        
        # After many steps, should be close to epsilon_end
        assert agent.epsilon < initial_epsilon * 0.8