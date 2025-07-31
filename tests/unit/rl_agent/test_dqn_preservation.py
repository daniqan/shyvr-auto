"""
Test preservation integration for DQN Agent

Tests for Phase 1.5: Integration with Existing System
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import torch
from datetime import datetime

from src.rl_agent.dqn_agent import DQNTradingAgent, AgentConfig, ModelType
from src.model_preservation.manager import PreservationManager
from src.model_preservation.base import PreservationError, PreservationPriority


@pytest.fixture
def mock_preservation_manager():
    """Create a mock preservation manager"""
    manager = Mock(spec=PreservationManager)
    manager.save_model = AsyncMock(return_value="agent_123")
    manager.load_model = AsyncMock(return_value=(b"model_data", {"model_id": "agent_123"}))
    return manager


@pytest.fixture
def agent_config():
    """Create test agent configuration"""
    config = AgentConfig(
        model_type=ModelType.DQN,
        hidden_size=128,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        replay_buffer_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
        target_update_frequency=100,
        dropout=0.1
    )
    # Add preservation config
    config.preservation = {
        'enabled': True,
        'gcs_bucket': 'test-bucket',
        'backup_interval_hours': 3,
        'max_versions_per_model': 20
    }
    return config


@pytest.fixture
def dqn_agent_with_preservation(agent_config, mock_preservation_manager):
    """Create a DQN agent with preservation integration"""
    agent = DQNTradingAgent(agent_config)
    # Inject mock preservation manager
    agent._preservation_manager = mock_preservation_manager
    agent._preservation_enabled = True
    return agent


class TestDQNPreservationIntegration:
    """Test DQN Agent preservation integration"""
    
    def test_save_model_with_preservation(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that save_model triggers preservation"""
        # Mock torch.save
        with patch('torch.save'):
            # Execute
            filepath = "test_model.pt"
            result = dqn_agent_with_preservation.save_model(filepath)
            
            # Verify local save succeeded
            assert result is True
            
            # Verify preservation was triggered
            dqn_agent_with_preservation._preservation_manager.save_model.assert_called_once()
            
            # Check preservation call arguments
            call_args = dqn_agent_with_preservation._preservation_manager.save_model.call_args
            assert call_args.kwargs['model_type'] == 'dqn'
            assert 'model_data' in call_args.kwargs
            assert isinstance(call_args.kwargs['model_data'], bytes)
            assert call_args.kwargs.get('priority') == PreservationPriority.NORMAL
    
    def test_save_checkpoint_with_preservation(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that checkpoint saves trigger preservation"""
        # Execute
        filepath = "checkpoint.pt"
        result = dqn_agent_with_preservation.save_model(filepath)
        
        # Verify preservation was called with checkpoint tag
        call_args = mock_preservation_manager.save_model.call_args
        tags = call_args.kwargs.get('tags', [])
        assert 'checkpoint' in tags or 'dqn_checkpoint' in tags
    
    def test_save_model_preservation_fallback(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that local save succeeds even if preservation fails"""
        # Make preservation fail
        mock_preservation_manager.save_model.side_effect = PreservationError("Storage failed")
        
        # Execute - should not raise
        filepath = "test_model.pt"
        result = dqn_agent_with_preservation.save_model(filepath)
        
        # Verify local save still succeeded
        assert result is True
    
    @pytest.mark.asyncio
    async def test_load_model_with_preservation_fallback(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that _load_from_preservation works correctly"""
        # Mock preserved checkpoint data
        checkpoint = {
            'q_network_state_dict': dqn_agent_with_preservation.q_network.state_dict(),
            'target_network_state_dict': dqn_agent_with_preservation.target_network.state_dict(),
            'optimizer_state_dict': dqn_agent_with_preservation.optimizer.state_dict(),
            'training_episodes': 100,
            'epsilon': 0.5,
            'steps_done': 1000
        }
        import pickle
        preserved_data = pickle.dumps(checkpoint)
        mock_preservation_manager.load_model.return_value = (preserved_data, {"model_id": "agent_123"})
        
        # Execute async method directly
        result = await dqn_agent_with_preservation._load_from_preservation()
        
        # Verify preservation fallback was used
        dqn_agent_with_preservation._preservation_manager.load_model.assert_called_once_with(
            model_type='dqn',
            version=None,
            mode='training',
            fallback=True
        )
        
        # Verify model was restored
        assert result is True
        assert dqn_agent_with_preservation.training_episodes == 100
        assert dqn_agent_with_preservation.epsilon == 0.5
    
    def test_load_model_no_preservation_when_local_succeeds(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that preservation is not used when local load succeeds"""
        # Mock successful local load
        checkpoint = {
            'q_network_state_dict': dqn_agent_with_preservation.q_network.state_dict(),
            'target_network_state_dict': dqn_agent_with_preservation.target_network.state_dict(),
            'optimizer_state_dict': dqn_agent_with_preservation.optimizer.state_dict(),
            'training_episodes': 50,
            'epsilon': 0.3,
            'steps_done': 500
        }
        
        with patch('torch.load', return_value=checkpoint):
            result = dqn_agent_with_preservation.load_model("existing_model.pt")
        
        # Verify preservation was not called
        mock_preservation_manager.load_model.assert_not_called()
        
        # Verify result
        assert result is True
        assert dqn_agent_with_preservation.training_episodes == 50
    
    def test_preservation_disabled(self):
        """Test that preservation is not used when disabled"""
        # Create config without preservation
        config = AgentConfig(
            model_type=ModelType.DQN,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=1000,
            target_update_frequency=100,
            dropout=0.1
        )
        # Explicitly disable preservation
        config.preservation = {'enabled': False}
        
        # Create agent without preservation
        agent = DQNTradingAgent(config)
        
        # Verify preservation is not enabled
        assert agent._preservation_manager is None
        assert not agent._preservation_enabled
    
    def test_save_with_training_metadata(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that preservation includes training metadata"""
        # Set some training state
        dqn_agent_with_preservation.training_episodes = 500
        dqn_agent_with_preservation.epsilon = 0.1
        dqn_agent_with_preservation.performance_metrics.total_reward = 1000.0
        dqn_agent_with_preservation.performance_metrics.win_rate = 0.65
        
        # Execute
        dqn_agent_with_preservation.save_model("test_model.pt")
        
        # Verify metadata was included
        call_args = mock_preservation_manager.save_model.call_args
        metadata = call_args.kwargs.get('metadata', {})
        assert 'training_episodes' in metadata
        assert metadata['training_episodes'] == 500
        assert 'performance' in metadata
        assert metadata['performance']['win_rate'] == 0.65
    
    @pytest.mark.asyncio
    async def test_recovery_from_preserved_state(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test complete recovery from preserved state"""
        # Create a different agent state
        original_epsilon = dqn_agent_with_preservation.epsilon
        
        # Mock preserved state with different values
        preserved_checkpoint = {
            'q_network_state_dict': dqn_agent_with_preservation.q_network.state_dict(),
            'target_network_state_dict': dqn_agent_with_preservation.target_network.state_dict(),
            'optimizer_state_dict': dqn_agent_with_preservation.optimizer.state_dict(),
            'config': dqn_agent_with_preservation.config.__dict__,
            'training_episodes': 1000,
            'epsilon': 0.05,
            'steps_done': 5000,
            'performance_metrics': {
                'total_reward': 2000.0,
                'win_rate': 0.75,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.1,
                'profit_factor': 2.0,
                'absolute_return': 0.0,
                'relative_return': 0.0,
                'risk_adjusted_return': 0.0,
                'volatility': 0.0,
                'sortino_ratio': 0.0,
                'average_trade_return': 0.0,
                'drawdown_penalty': 0.0,
                'volatility_penalty': 0.0,
                'concentration_penalty': 0.0
            }
        }
        
        import pickle
        preserved_data = pickle.dumps(preserved_checkpoint)
        mock_preservation_manager.load_model.return_value = (preserved_data, {
            "model_id": "agent_123",
            "version": "v1.2.0",
            "preserved_at": datetime.now().isoformat()
        })
        
        # Execute recovery directly via async method
        result = await dqn_agent_with_preservation._load_from_preservation()
        
        # Verify complete state recovery
        assert result is True
        assert dqn_agent_with_preservation.training_episodes == 1000
        assert dqn_agent_with_preservation.epsilon == 0.05
        assert dqn_agent_with_preservation.steps_done == 5000
        assert dqn_agent_with_preservation.performance_metrics.win_rate == 0.75
        assert dqn_agent_with_preservation.is_trained is True
    
    def test_preservation_with_mode_context(self, dqn_agent_with_preservation, mock_preservation_manager):
        """Test that preservation includes current mode context"""
        # Set agent mode
        dqn_agent_with_preservation._current_mode = 'live_trading'
        
        # Execute
        dqn_agent_with_preservation.save_model("test_model.pt")
        
        # Verify mode was passed to preservation
        call_args = mock_preservation_manager.save_model.call_args
        assert call_args.kwargs.get('mode') == 'live_trading'
    
    def test_dueling_dqn_preservation(self, mock_preservation_manager):
        """Test preservation for Dueling DQN variant"""
        # Create Dueling DQN agent
        config = AgentConfig(
            model_type=ModelType.DUELING_DQN,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=1000,
            target_update_frequency=100,
            dropout=0.1
        )
        # Add preservation config
        config.preservation = {
            'enabled': True,
            'gcs_bucket': 'test-bucket',
            'backup_interval_hours': 3,
            'max_versions_per_model': 20
        }
        
        agent = DQNTradingAgent(config)
        agent._preservation_manager = mock_preservation_manager
        agent._preservation_enabled = True
        
        # Execute
        with patch('torch.save'):
            agent.save_model("dueling_model.pt")
        
        # Verify correct model type (duelingdqn without underscore)
        call_args = mock_preservation_manager.save_model.call_args
        assert call_args.kwargs['model_type'] == 'duelingdqn'
    
    def test_rainbow_dqn_preservation(self, mock_preservation_manager):
        """Test preservation for Rainbow DQN variant"""
        # Create Rainbow DQN agent
        config = AgentConfig(
            model_type=ModelType.RAINBOW,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=1000,
            target_update_frequency=100,
            dropout=0.1,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            noisy_networks=True
        )
        # Add preservation config
        config.preservation = {
            'enabled': True,
            'gcs_bucket': 'test-bucket',
            'backup_interval_hours': 3,
            'max_versions_per_model': 20
        }
        
        agent = DQNTradingAgent(config)
        agent._preservation_manager = mock_preservation_manager
        agent._preservation_enabled = True
        
        # Execute
        with patch('torch.save'):
            agent.save_model("rainbow_model.pt")
        
        # Verify correct model type and additional metadata
        call_args = mock_preservation_manager.save_model.call_args
        assert call_args.kwargs['model_type'] == 'rainbow'
        metadata = call_args.kwargs.get('metadata', {})
        assert 'num_atoms' in metadata
        assert metadata['num_atoms'] == 51