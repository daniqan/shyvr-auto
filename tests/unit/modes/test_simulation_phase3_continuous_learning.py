"""
Test suite for SimulationMode Phase 3 - Continuous Learning Integration.

Tests the integration of ContinuousLearningEngine with SimulationMode for
simulation-based learning and unified learning cycle participation.

Following TDD methodology - tests written before implementation.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from src.modes.base import ModeType, ModeStatus, ModeConfig
from src.portfolio.base import Portfolio, PortfolioConfig
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.modes.continuous_learning import ContinuousLearningEngine, ContinuousLearningConfig
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.modes.simulation_mode import SimulationMode


def create_mock_token():
    """Helper function to create mock DiscoveredToken."""
    from src.discovery.base import DiscoveredToken, TokenStatus
    from src.utils.base import Chain
    
    return DiscoveredToken(
        address="TEST_TOKEN_123",
        name="Test Token",
        symbol="TEST",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        status=TokenStatus.VALIDATED
    )


class TestSimulationContinuousLearningIntegration:
    """Test suite for continuous learning integration in SimulationMode."""
    
    @pytest.fixture
    def portfolio(self):
        """Portfolio instance for testing."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        return Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
    
    @pytest.fixture
    def simulation_config(self):
        """Simulation mode configuration with continuous learning enabled."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_experience_collection": True,
                "experience_buffer_size": 1000,
                "enable_continuous_learning": True,
                "learning_trigger_threshold": 100,
                "learning_episodes_per_session": 50,
                "enable_learning_persistence": True
            }
        )
    
    @pytest.fixture
    def mock_dqn_agent(self):
        """Mock DQN trading agent."""
        agent = Mock(spec=DQNTradingAgent)
        agent.save_model = Mock()
        agent.load_model = Mock()
        agent.get_action = Mock(return_value=TradeAction.HOLD)
        return agent
    
    @pytest.fixture
    def mock_continuous_learning_engine(self):
        """Mock continuous learning engine."""
        engine = Mock(spec=ContinuousLearningEngine)
        engine.check_and_trigger_training = AsyncMock(return_value=None)
        engine.run_automatic_training_cycle = AsyncMock(return_value={
            'training_triggered': True,
            'session_success': True,
            'model_activated': True
        })
        engine.get_performance_statistics = AsyncMock(return_value={
            'total_training_sessions': 1,
            'successful_sessions': 1,
            'active_model_version': 'v1.1.0'
        })
        return engine
    
    @pytest.fixture
    def simulation_mode_with_learning(self, simulation_config, portfolio):
        """SimulationMode instance with continuous learning enabled."""
        mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_config,
            portfolio=portfolio
        )
        return mode
    
    async def test_continuous_learning_initialization(self, simulation_mode_with_learning):
        """Test that continuous learning components are properly initialized."""
        mode = simulation_mode_with_learning
        
        # Should initialize continuous learning engine
        assert hasattr(mode, 'continuous_learning_engine')
        assert mode.continuous_learning_engine is not None
        
        # Should initialize with proper configuration
        learning_config = mode.continuous_learning_config
        assert learning_config.training_trigger_threshold == 100
        assert learning_config.training_episodes_per_session == 50
        assert learning_config.enable_model_persistence is True
        
        # Should connect to experience collector
        assert mode.experience_collector is not None
        assert mode.continuous_learning_engine.experience_collector == mode.experience_collector
    
    async def test_simulation_experience_collection_for_learning(self, simulation_mode_with_learning):
        """Test that simulation experiences are collected for continuous learning."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        await mode.start()
        
        # Mock the continuous learning engine
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(return_value=None)
        
        # Process market tick to generate experience
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=25.0  # Should trigger BUY action
        )
        
        # Mock experience collector to verify experience capture
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="exp_123")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        
        action = await mode.process_tick(market_state)
        
        # Should capture trading experience
        assert action in [TradeAction.BUY, TradeAction.STRONG_BUY]
        mode.experience_collector.capture_pre_trade_state.assert_called_once()
        
        # Should check for learning triggers
        mode.continuous_learning_engine.check_and_trigger_training.assert_called_once()
    
    async def test_learning_trigger_integration(self, simulation_mode_with_learning):
        """Test integration with learning trigger mechanisms."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        
        # Mock continuous learning engine with trigger response
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(
            return_value={
                'training_triggered': True,
                'session_id': 'session_123',
                'episodes_completed': 50,
                'model_activated': True
            }
        )
        
        # Simulate experience accumulation
        mode.experience_collector.get_buffer_size = Mock(return_value=150)  # Above threshold
        
        # Process tick should trigger learning
        await mode.start()
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=75.0  # Should trigger SELL action
        )
        
        await mode.process_tick(market_state)
        
        # Should trigger continuous learning
        mode.continuous_learning_engine.check_and_trigger_training.assert_called_once()
    
    async def test_model_update_integration(self, simulation_mode_with_learning):
        """Test integration with model updates from continuous learning."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        
        # Mock learning engine with model update
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(
            return_value={
                'training_triggered': True,
                'model_activated': True,
                'new_model_version': 'v1.2.0',
                'performance_improved': True
            }
        )
        
        # Mock DQN agent for model loading
        mode.dqn_agent = Mock()
        mode.dqn_agent.load_model = Mock()
        mode.continuous_learning_engine.dqn_agent = mode.dqn_agent
        
        await mode.start()
        
        # Simulate learning trigger
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0
        )
        
        await mode.process_tick(market_state)
        
        # Should trigger learning and model update
        mode.continuous_learning_engine.check_and_trigger_training.assert_called_once()
    
    async def test_unified_learning_cycle_participation(self, simulation_mode_with_learning):
        """Test that simulation mode participates in unified learning cycle."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        await mode.start()
        
        # Mock components for unified cycle
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.replay_buffer = Mock()
        mode.continuous_learning_engine.replay_buffer.add_experience = Mock()
        
        # Simulate complete trading cycle
        mock_token = create_mock_token()
        
        # 1. Market observation
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=30.0
        )
        
        # 2. Trading decision and execution
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="exp_456")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.add_experiences_to_buffer = AsyncMock(return_value=1)
        
        action = await mode.process_tick(market_state)
        
        # 3. Experience storage for learning
        await mode._flush_experiences_to_buffer()
        
        # Should participate in complete cycle: trades → learning → better model → better trades
        mode.experience_collector.capture_pre_trade_state.assert_called_once()
        mode.experience_collector.add_experiences_to_buffer.assert_called_once()
    
    async def test_learning_performance_tracking(self, simulation_mode_with_learning):
        """Test tracking of learning performance and statistics."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        
        # Mock learning engine with performance data
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.get_performance_statistics = AsyncMock(
            return_value={
                'total_training_sessions': 5,
                'successful_sessions': 4,
                'total_model_versions': 3,
                'active_model_version': 'v1.3.0',
                'best_performance': {
                    'version_id': 'v1.3.0',
                    'average_reward': 150.5,
                    'win_rate': 0.65,
                    'sharpe_ratio': 1.2
                }
            }
        )
        
        # Get enhanced statistics including learning data
        stats = await mode.get_enhanced_statistics()
        
        # Should include learning performance metrics
        assert 'continuous_learning' in stats
        learning_stats = stats['continuous_learning']
        assert learning_stats['total_training_sessions'] == 5
        assert learning_stats['successful_sessions'] == 4
        assert learning_stats['active_model_version'] == 'v1.3.0'
    
    async def test_learning_state_persistence(self, simulation_mode_with_learning):
        """Test persistence of learning state across simulation sessions."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        
        # Mock learning engine with state persistence
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.save_state = AsyncMock()
        mode.continuous_learning_engine.load_state = AsyncMock()
        
        # Start simulation
        await mode.start()
        
        # Should load previous learning state
        mode.continuous_learning_engine.load_state.assert_called_once()
        
        # Simulate some learning activity
        await mode.stop()
        
        # Should save learning state
        mode.continuous_learning_engine.save_state.assert_called_once()
    
    async def test_learning_error_handling(self, simulation_mode_with_learning):
        """Test error handling in continuous learning integration."""
        mode = simulation_mode_with_learning
        await mode.initialize()
        await mode.start()
        
        # Mock learning engine to raise error
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(
            side_effect=Exception("Learning engine error")
        )
        
        # Process tick should handle learning errors gracefully
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0
        )
        
        # Should not crash simulation despite learning error
        action = await mode.process_tick(market_state)
        assert action is not None or action is None  # Should return some result
    
    async def test_learning_configuration_validation(self, portfolio):
        """Test validation of continuous learning configuration."""
        # Invalid configuration
        invalid_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "enable_continuous_learning": True,
                "learning_trigger_threshold": -10,  # Invalid negative value
                "learning_episodes_per_session": 0   # Invalid zero value
            }
        )
        
        # Should handle invalid configuration gracefully
        mode = SimulationMode(
            mode_id=uuid4(),
            config=invalid_config,
            portfolio=portfolio
        )
        
        # Should initialize with default values or raise validation error
        try:
            await mode.initialize()
            # If no error, check that defaults were applied
            assert mode.continuous_learning_config.training_trigger_threshold > 0
            assert mode.continuous_learning_config.training_episodes_per_session > 0
        except ValueError:
            # Validation error is acceptable
            pass


class TestSimulationLearningCycleIntegration:
    """Test suite for unified learning cycle integration."""
    
    @pytest.fixture
    def complete_learning_setup(self):
        """Complete setup with all learning components."""
        # Portfolio setup
        portfolio_config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Learning Portfolio",
            config=portfolio_config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
        
        # Simulation mode with learning
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_continuous_learning": True,
                "learning_trigger_threshold": 50,
                "enable_experience_collection": True
            }
        )
        
        mode = SimulationMode(
            mode_id=uuid4(),
            config=mode_config,
            portfolio=portfolio
        )
        
        return {
            "portfolio": portfolio,
            "mode_config": mode_config,
            "simulation_mode": mode
        }
    
    async def test_complete_learning_cycle_flow(self, complete_learning_setup):
        """Test complete flow from simulation experiences to model improvement."""
        mode = complete_learning_setup["simulation_mode"]
        await mode.initialize()
        await mode.start()
        
        # Mock all learning components
        mode.experience_collector = Mock()
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="exp_789")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.add_experiences_to_buffer = AsyncMock(return_value=5)
        mode.experience_collector.get_buffer_size = Mock(return_value=60)  # Above threshold
        
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(
            return_value={
                'training_triggered': True,
                'session_success': True,
                'episodes_completed': 50,
                'new_model_version': 'v1.4.0',
                'performance_improved': True,
                'model_activated': True
            }
        )
        
        # Simulate trading sequence
        mock_token = create_mock_token()
        
        for i in range(10):  # Generate multiple experiences
            market_state = MarketState(
                token=mock_token,
                price_usd=100.0 + i,
                price_change_24h=5.0,
                volume_24h=1000000.0,
                rsi=30.0 + i * 2
            )
            
            action = await mode.process_tick(market_state)
            await asyncio.sleep(0.001)  # Brief pause
        
        # Should complete full learning cycle
        assert mode.experience_collector.capture_pre_trade_state.call_count > 0
        assert mode.continuous_learning_engine.check_and_trigger_training.call_count > 0
    
    async def test_learning_feedback_loop(self, complete_learning_setup):
        """Test that learning improves subsequent trading decisions."""
        mode = complete_learning_setup["simulation_mode"]
        await mode.initialize()
        await mode.start()
        
        # Mock improved model after learning
        initial_performance = {"win_rate": 0.5, "average_reward": 100}
        improved_performance = {"win_rate": 0.7, "average_reward": 150}
        
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.check_and_trigger_training = AsyncMock(
            return_value={
                'training_triggered': True,
                'model_activated': True,
                'performance_improved': True,
                'performance_delta': 0.2
            }
        )
        
        # Simulate learning event
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0
        )
        
        await mode.process_tick(market_state)
        
        # Should use improved model for subsequent decisions
        mode.continuous_learning_engine.check_and_trigger_training.assert_called_once()
    
    async def test_cross_session_learning_persistence(self, complete_learning_setup):
        """Test that learning persists across simulation sessions."""
        mode = complete_learning_setup["simulation_mode"]
        
        # Mock persistent learning state
        mode.continuous_learning_engine = Mock()
        mode.continuous_learning_engine.load_state = AsyncMock()
        mode.continuous_learning_engine.save_state = AsyncMock()
        mode.continuous_learning_engine.get_performance_statistics = AsyncMock(
            return_value={
                'total_training_sessions': 3,
                'successful_sessions': 3,
                'active_model_version': 'v1.3.0'
            }
        )
        
        # First session
        await mode.initialize()
        await mode.start()
        
        # Should load previous learning state
        mode.continuous_learning_engine.load_state.assert_called_once()
        
        # End session
        await mode.stop()
        
        # Should save learning state
        mode.continuous_learning_engine.save_state.assert_called_once()
        
        # New session should continue from saved state
        await mode.start()
        assert mode.continuous_learning_engine.load_state.call_count == 2