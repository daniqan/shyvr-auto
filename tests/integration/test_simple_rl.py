"""
Simple RL Integration Tests
Test basic RL functionality without complex dependencies
"""

import asyncio
import time
from datetime import datetime
import numpy as np
import pytest
import torch

from src.rl_agent.base import AgentConfig, TradeAction, MarketState
from src.rl_agent.dqn_agent import DQNTradingAgent, DQNNetwork
from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain


class TestSimpleRLIntegration:
    """Simple RL integration tests"""
    
    def test_dqn_network_basic(self):
        """Test basic DQN network creation and forward pass"""
        network = DQNNetwork(
            input_size=19,
            hidden_size=64,  # Smaller for faster testing
            num_layers=2,
            dropout=0.1,
            output_size=5
        )
        
        # Test forward pass
        state = torch.randn(1, 19)
        output = network(state)
        
        assert output.shape == (1, 5)
        assert torch.is_tensor(output)
        assert not torch.isnan(output).any()
    
    @pytest.mark.asyncio
    async def test_dqn_agent_predict(self):
        """Test DQN agent prediction"""
        config = AgentConfig(
            learning_rate=0.01,
            batch_size=8,
            hidden_size=64,
            num_layers=2
        )
        agent = DQNTradingAgent(config)
        
        # Create test token and market state
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            status=TokenStatus.DISCOVERED,
            price_usd=1.50
        )
        
        state = MarketState(
            token=token,
            price_usd=1.50,
            price_change_24h=2.5,
            volume_24h=100000,
            portfolio_value=10000.0,
            cash_balance=8000.0,
            current_position=0.1
        )
        
        # Test prediction
        action, confidence = await agent.predict_action(state)
        
        assert isinstance(action, TradeAction)
        assert 0.0 <= confidence <= 1.0
    
    def test_model_save_load(self):
        """Test model persistence"""
        import tempfile
        import os
        
        config = AgentConfig(hidden_size=32, num_layers=2)
        agent = DQNTradingAgent(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            
            # Save model
            success = agent.save_model(model_path)
            assert success
            assert os.path.exists(model_path)
            
            # Modify model to test loading
            with torch.no_grad():
                for param in agent.q_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # Load model
            success = agent.load_model(model_path)
            assert success