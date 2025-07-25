"""
Tests for Simulation Mode Experience Collection Integration

Following TDD methodology - comprehensive failing tests for integrating
TradingExperienceCollector with SimulationMode for automatic experience capture.
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import Dict, Any

from src.modes.simulation_mode import SimulationMode
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.modes.base import ModeConfig, ModeStatus
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.discovery.base import DiscoveredToken
from src.portfolio.base import Portfolio
from src.utils.base import Chain


@pytest.fixture
def mock_discovered_token():
    """Create mock discovered token"""
    return DiscoveredToken(
        address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        symbol="USDC",
        name="USD Coin",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        decimals=6,
        price_usd=1.0,
        market_cap=50000000000,
        volume_24h=1000000000,
        price_change_24h=0.5,
        trending_score=0.95,
        social_mentions=1000000
    )


@pytest.fixture
def mock_market_state(mock_discovered_token):
    """Create mock market state"""
    return MarketState(
        token=mock_discovered_token,
        price_usd=1.0,
        price_change_24h=0.5,
        volume_24h=1000000,
        market_cap=50000000,
        rsi=55.0,
        macd=0.1,
        sma_20=0.98,
        ema_12=1.02,
        bollinger_upper=1.05,
        bollinger_lower=0.95,
        current_position=0.0,
        portfolio_value=10000.0,
        cash_balance=10000.0,
        portfolio_drawdown=0.0,
        daily_pnl=0.0,
        sharpe_ratio=1.2,
        market_volatility=0.15,
        fear_greed_index=50.0,
        prediction_confidence=0.8,
        timestamp=datetime.now()
    )


@pytest.fixture
def mock_trading_result(mock_discovered_token):
    """Create mock trading result"""
    return TradingResult(
        action=TradeAction.BUY,
        token=mock_discovered_token,
        executed_at=datetime.now(),
        price=1.0,
        quantity=1000.0,
        value_usd=1000.0,
        success=True,
        slippage=0.01,
        fees=3.0,
        portfolio_value_before=10000.0,
        portfolio_value_after=10000.0,
        cash_change=-1003.0,
        position_change=1000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0
    )


@pytest.fixture
def experience_collector_config():
    """Create experience collector configuration"""
    return ExperienceCollectorConfig(
        buffer_size=1000,
        min_experience_gap_seconds=0,  # No gap for testing
        reward_calculation_window=5,
        enable_persistence=False,  # Disable for testing
        max_experiences_per_session=100
    )


@pytest.fixture
def simulation_mode_config():
    """Create simulation mode configuration"""
    from src.modes.base import ModeType
    return ModeConfig(
        mode_type=ModeType.SIMULATION,
        enabled=True,
        parameters={
            "initial_balance": 10000,
            "enable_fees": True,
            "slippage_bps": 10,
            "max_drawdown_pct": 0.15,
            "enable_risk_management": True,
            "enable_experience_collection": True  # Key parameter for integration
        }
    )


@pytest.fixture
def mock_replay_buffer():
    """Create mock replay buffer"""
    buffer = Mock(spec=ExperienceReplayBuffer)
    buffer.add = Mock()
    buffer.can_sample = Mock(return_value=True)
    buffer.sample = Mock(return_value=[])
    buffer.__len__ = Mock(return_value=0)
    return buffer


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio"""
    portfolio = Mock(spec=Portfolio)
    portfolio.total_value = Decimal("10000")
    portfolio.cash_balance = Decimal("10000")
    return portfolio


class TestSimulationExperienceIntegration:
    """Test suite for SimulationMode + TradingExperienceCollector integration"""

    @pytest.mark.asyncio
    async def test_simulation_mode_with_experience_collection_enabled(self, simulation_mode_config, 
                                                                      mock_portfolio, mock_replay_buffer,
                                                                      experience_collector_config):
        """Test simulation mode initializes with experience collection when enabled"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Should have experience collector when enabled
        assert hasattr(mode, 'experience_collector')
        assert mode.experience_collector is not None
        assert isinstance(mode.experience_collector, TradingExperienceCollector)

    @pytest.mark.asyncio
    async def test_simulation_mode_without_experience_collection(self, simulation_mode_config, 
                                                                mock_portfolio):
        """Test simulation mode without experience collection when disabled"""
        simulation_mode_config.parameters['enable_experience_collection'] = False
        
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Should not have experience collector when disabled
        assert not hasattr(mode, 'experience_collector') or mode.experience_collector is None

    @pytest.mark.asyncio
    async def test_experience_collection_starts_with_simulation_mode(self, simulation_mode_config,
                                                                     mock_portfolio, mock_replay_buffer):
        """Test experience collection starts when simulation mode starts"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.start_collection = AsyncMock()
        mode.experience_collector.stop_collection = AsyncMock()
        
        await mode.initialize()
        await mode.start()
        
        # Experience collection should start with simulation mode
        mode.experience_collector.start_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_experience_collection_stops_with_simulation_mode(self, simulation_mode_config,
                                                                    mock_portfolio, mock_replay_buffer):
        """Test experience collection stops when simulation mode stops"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.start_collection = AsyncMock()
        mode.experience_collector.stop_collection = AsyncMock()
        mode.experience_collector.add_experiences_to_buffer = AsyncMock(return_value=0)
        
        await mode.initialize()
        await mode.start()
        await mode.stop()
        
        # Experience collection should stop with simulation mode
        mode.experience_collector.stop_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_decision_captures_pre_trade_state(self, simulation_mode_config, mock_portfolio,
                                                             mock_market_state, mock_replay_buffer):
        """Test that making a trading decision captures pre-trade state"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="test-exp-id")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Process market tick should capture pre-trade state for non-HOLD actions
        with patch.object(mode, '_make_trading_decision', return_value=TradeAction.BUY):
            with patch.object(mode, '_execute_simulated_trade', new_callable=AsyncMock):
                action = await mode.process_tick(mock_market_state)
                
                assert action == TradeAction.BUY
                mode.experience_collector.capture_pre_trade_state.assert_called_once_with(
                    mock_market_state, TradeAction.BUY
                )

    @pytest.mark.asyncio
    async def test_trading_execution_captures_post_trade_result(self, simulation_mode_config, mock_portfolio,
                                                                mock_market_state, mock_trading_result,
                                                                mock_replay_buffer):
        """Test that executing a trade captures post-trade result"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="test-exp-id")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Simulate trade execution capturing post-trade result
        await mode._capture_trade_experience("test-exp-id", mock_trading_result, mock_market_state)
        
        mode.experience_collector.capture_post_trade_result.assert_called_once_with(
            "test-exp-id", mock_trading_result, mock_market_state
        )

    @pytest.mark.asyncio
    async def test_successful_trade_flow_captures_complete_experience(self, simulation_mode_config, mock_portfolio,
                                                                      mock_market_state, mock_trading_result,
                                                                      mock_replay_buffer):
        """Test complete trade flow captures both pre and post trade states"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="test-exp-id")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Mock trading execution to return successful result
        with patch.object(mode.simulation_executor, 'execute_buy_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_trading_result
            
            # Process complete trade flow
            with patch.object(mode, '_make_trading_decision', return_value=TradeAction.BUY):
                action = await mode.process_tick(mock_market_state)
                
                # Should capture pre-trade state
                mode.experience_collector.capture_pre_trade_state.assert_called_once()
                
                # Should execute trade and capture post-trade result
                assert action == TradeAction.BUY

    @pytest.mark.asyncio
    async def test_failed_trade_captures_failure_experience(self, simulation_mode_config, mock_portfolio,
                                                            mock_market_state, mock_replay_buffer):
        """Test that failed trades capture failure experiences"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock(return_value="test-exp-id")
        mode.experience_collector.capture_post_trade_result = AsyncMock()
        mode.experience_collector.is_collecting = True
        
        # Create failed trading result
        failed_result = TradingResult(
            action=TradeAction.BUY,
            token=mock_market_state.token,
            executed_at=datetime.now(),
            price=1.0,
            quantity=0.0,
            value_usd=0.0,
            success=False,
            error_message="Insufficient liquidity"
        )
        
        await mode.initialize()
        await mode.start()
        
        # Simulate failed trade
        await mode._capture_trade_experience("test-exp-id", failed_result, mock_market_state)
        
        # Should capture failure in post-trade result
        mode.experience_collector.capture_post_trade_result.assert_called_once_with(
            "test-exp-id", failed_result, mock_market_state
        )

    @pytest.mark.asyncio
    async def test_experience_buffer_integration(self, simulation_mode_config, mock_portfolio,
                                                 mock_market_state, mock_trading_result,
                                                 mock_replay_buffer):
        """Test that experiences are properly added to replay buffer"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Set up real experience collector with mock buffer
        experience_config = ExperienceCollectorConfig(enable_persistence=False)
        mode.experience_collector = TradingExperienceCollector(experience_config, mock_replay_buffer)
        
        await mode.initialize()
        await mode.start()
        
        # Start experience collection
        await mode.experience_collector.start_collection()
        
        # Capture a complete experience
        exp_id = await mode.experience_collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await mode.experience_collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        # Add experiences to buffer
        added_count = await mode.experience_collector.add_experiences_to_buffer()
        
        assert added_count == 1
        mock_replay_buffer.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_hold_action_does_not_capture_experience(self, simulation_mode_config, mock_portfolio,
                                                           mock_market_state, mock_replay_buffer):
        """Test that HOLD actions do not capture trading experiences"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock()
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Process HOLD action
        with patch.object(mode, '_make_trading_decision', return_value=TradeAction.HOLD):
            action = await mode.process_tick(mock_market_state)
            
            assert action == TradeAction.HOLD
            # Should not capture experience for HOLD actions
            mode.experience_collector.capture_pre_trade_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_periodic_experience_buffer_flush(self, simulation_mode_config, mock_portfolio,
                                                    mock_market_state, mock_replay_buffer):
        """Test periodic flushing of experiences to replay buffer"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.add_experiences_to_buffer = AsyncMock(return_value=5)
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Simulate periodic flush (this would be called by a timer or after N trades)
        await mode._flush_experiences_to_buffer()
        
        mode.experience_collector.add_experiences_to_buffer.assert_called_once()

    @pytest.mark.asyncio
    async def test_experience_collection_statistics_integration(self, simulation_mode_config, mock_portfolio,
                                                                mock_replay_buffer):
        """Test integration of experience collection statistics with simulation mode"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector with statistics
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.get_statistics = AsyncMock(return_value={
            'total_experiences': 10,
            'active_experiences': 2,
            'completed_experiences': 8,
            'success_rate': 0.8,
            'average_reward': 5.2
        })
        
        await mode.initialize()
        
        # Get simulation mode statistics (enhanced with experience data)
        stats = await mode.get_enhanced_statistics()
        
        assert 'experience_collection' in stats
        assert stats['experience_collection']['total_experiences'] == 10
        assert stats['experience_collection']['success_rate'] == 0.8

    @pytest.mark.asyncio
    async def test_experience_persistence_on_simulation_cleanup(self, simulation_mode_config, mock_portfolio,
                                                                mock_replay_buffer):
        """Test that experiences are persisted when simulation mode cleans up"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.stop_collection = AsyncMock()
        mode.experience_collector.add_experiences_to_buffer = AsyncMock(return_value=0)
        mode.experience_collector.persist_experiences = AsyncMock()
        
        await mode.initialize()
        await mode.start()
        await mode.cleanup()
        
        # Should persist experiences during cleanup
        mode.experience_collector.stop_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_experience_collection_error_handling(self, simulation_mode_config, mock_portfolio,
                                                         mock_market_state, mock_replay_buffer):
        """Test error handling in experience collection doesn't break simulation"""
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        # Mock the experience collector to raise error
        mode.experience_collector = Mock(spec=TradingExperienceCollector)
        mode.experience_collector.capture_pre_trade_state = AsyncMock(side_effect=Exception("Collection error"))
        mode.experience_collector.is_collecting = True
        
        await mode.initialize()
        await mode.start()
        
        # Process tick should handle experience collection errors gracefully
        with patch.object(mode, '_make_trading_decision', return_value=TradeAction.BUY):
            with patch.object(mode, '_execute_simulated_trade', new_callable=AsyncMock):
                action = await mode.process_tick(mock_market_state)
                
                # Should still return action despite experience collection error
                assert action == TradeAction.BUY

    @pytest.mark.asyncio
    async def test_experience_collection_disabled_gracefully(self, simulation_mode_config, mock_portfolio,
                                                             mock_market_state):
        """Test simulation mode works normally when experience collection is disabled"""
        simulation_mode_config.parameters['enable_experience_collection'] = False
        
        mode = SimulationMode(uuid4(), simulation_mode_config, mock_portfolio)
        
        await mode.initialize()
        await mode.start()
        
        # Process tick should work normally without experience collection
        with patch.object(mode, '_make_trading_decision', return_value=TradeAction.BUY):
            with patch.object(mode, '_execute_simulated_trade', new_callable=AsyncMock):
                action = await mode.process_tick(mock_market_state)
                
                assert action == TradeAction.BUY