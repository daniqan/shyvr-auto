"""
Tests for TradingExperienceCollector

Following TDD methodology - comprehensive failing tests first, then implementation.
Tests for the Real Experience Collection Pipeline that captures trading experiences
from all modes and converts them to RL training data.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import List, Dict, Any
import numpy as np

from src.modes.experience_collector import (
    TradingExperienceCollector,
    ExperienceCollectorConfig,
    TradingExperienceData,
    ExperienceCollectionError
)
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import Experience, ExperienceReplayBuffer, ReplayBufferConfig
from src.discovery.base import DiscoveredToken
from src.ml_analysis.base import PredictionResult
from src.portfolio.base import Position, PositionType, PositionStatus
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
def collector_config():
    """Create experience collector configuration"""
    return ExperienceCollectorConfig(
        buffer_size=1000,
        min_experience_gap_seconds=5,
        reward_calculation_window=10,
        enable_persistence=True,
        persistence_path="test_experiences.json",
        max_experiences_per_session=100,
        experience_timeout_hours=24
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


class TestTradingExperienceCollector:
    """Test suite for TradingExperienceCollector"""

    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector_config, mock_replay_buffer):
        """Test collector initialization"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        
        assert collector.config == collector_config
        assert collector.replay_buffer == mock_replay_buffer
        assert collector.active_experiences == {}
        assert collector.completed_experiences == []
        assert collector.is_collecting is False
        assert collector.session_start_time is None
        
    @pytest.mark.asyncio
    async def test_start_collection(self, collector_config, mock_replay_buffer):
        """Test starting experience collection"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        
        await collector.start_collection()
        
        assert collector.is_collecting is True
        assert collector.session_start_time is not None
        assert isinstance(collector.session_start_time, datetime)

    @pytest.mark.asyncio
    async def test_stop_collection(self, collector_config, mock_replay_buffer):
        """Test stopping experience collection"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        
        await collector.start_collection()
        await collector.stop_collection()
        
        assert collector.is_collecting is False

    @pytest.mark.asyncio
    async def test_capture_trade_experience_buy_order(self, collector_config, mock_replay_buffer, 
                                                      mock_market_state, mock_trading_result):
        """Test capturing experience from buy order"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # Capture before state
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        
        assert experience_id is not None
        assert experience_id in collector.active_experiences
        
        # Capture after state and result
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        # Experience should be completed and moved to completed list
        assert experience_id not in collector.active_experiences
        assert len(collector.completed_experiences) == 1
        
        completed_exp = collector.completed_experiences[0]
        assert completed_exp.experience_id == experience_id
        assert completed_exp.action == TradeAction.BUY
        assert completed_exp.success == True

    @pytest.mark.asyncio
    async def test_capture_trade_experience_sell_order(self, collector_config, mock_replay_buffer, 
                                                       mock_market_state, mock_trading_result):
        """Test capturing experience from sell order"""
        mock_trading_result.action = TradeAction.SELL
        mock_trading_result.cash_change = 997.0  # Positive for sell
        mock_trading_result.position_change = -1000.0  # Negative for sell
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.SELL)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        assert len(collector.completed_experiences) == 1
        completed_exp = collector.completed_experiences[0]
        assert completed_exp.action == TradeAction.SELL

    @pytest.mark.asyncio
    async def test_reward_calculation_profitable_trade(self, collector_config, mock_replay_buffer, 
                                                       mock_market_state, mock_trading_result):
        """Test reward calculation for profitable trade"""
        # Make trade profitable
        mock_trading_result.realized_pnl = 50.0
        mock_trading_result.portfolio_value_after = 10050.0
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        
        # Reward should be positive for profitable trade
        assert completed_exp.reward > 0
        assert completed_exp.realized_pnl == 50.0

    @pytest.mark.asyncio
    async def test_reward_calculation_losing_trade(self, collector_config, mock_replay_buffer, 
                                                   mock_market_state, mock_trading_result):
        """Test reward calculation for losing trade"""
        # Make trade unprofitable
        mock_trading_result.realized_pnl = -25.0
        mock_trading_result.portfolio_value_after = 9975.0
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        
        # Reward should be negative for losing trade
        assert completed_exp.reward < 0
        assert completed_exp.realized_pnl == -25.0

    @pytest.mark.asyncio
    async def test_reward_calculation_with_fees_and_slippage(self, collector_config, mock_replay_buffer, 
                                                             mock_market_state, mock_trading_result):
        """Test reward calculation includes fees and slippage penalties"""
        mock_trading_result.fees = 10.0
        mock_trading_result.slippage = 0.05  # 5% slippage
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        
        # High fees and slippage should reduce reward
        assert completed_exp.execution_quality_penalty > 0
        assert completed_exp.total_fees == 10.0
        assert completed_exp.slippage_penalty > 0

    @pytest.mark.asyncio
    async def test_convert_to_rl_experience(self, collector_config, mock_replay_buffer, 
                                            mock_market_state, mock_trading_result):
        """Test converting trading experience to RL experience format"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        
        # Modify next state to show market change
        next_market_state = mock_market_state
        next_market_state.price_usd = 1.05  # 5% price increase
        next_market_state.portfolio_value = 10050.0
        
        await collector.capture_post_trade_result(experience_id, mock_trading_result, next_market_state)
        
        completed_exp = collector.completed_experiences[0]
        rl_experience = collector._convert_to_rl_experience(completed_exp)
        
        assert isinstance(rl_experience, Experience)
        assert rl_experience.action == 1  # BUY encoded as 1
        assert isinstance(rl_experience.state, np.ndarray)
        assert isinstance(rl_experience.next_state, np.ndarray)
        assert isinstance(rl_experience.reward, float)
        assert isinstance(rl_experience.done, bool)

    @pytest.mark.asyncio
    async def test_batch_add_to_replay_buffer(self, collector_config, mock_replay_buffer, 
                                              mock_market_state, mock_trading_result):
        """Test batch adding experiences to replay buffer"""
        collector_config.min_experience_gap_seconds = 0  # No gap for testing
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # Create multiple experiences
        for i in range(3):
            experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        # Add to replay buffer
        added_count = await collector.add_experiences_to_buffer()
        
        assert added_count == 3
        assert mock_replay_buffer.add.call_count == 3
        assert len(collector.completed_experiences) == 0  # Should be cleared after adding

    @pytest.mark.asyncio
    async def test_experience_persistence(self, collector_config, mock_replay_buffer, 
                                          mock_market_state, mock_trading_result):
        """Test experience persistence to file"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        with patch('builtins.open', mock=Mock()) as mock_open:
            with patch('json.dump') as mock_json_dump:
                await collector.persist_experiences()
                
                mock_open.assert_called_once()
                mock_json_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_experience_loading(self, collector_config, mock_replay_buffer):
        """Test loading experiences from file"""
        mock_experience_data = [
            {
                'experience_id': str(uuid4()),
                'timestamp': datetime.now().isoformat(),
                'action': 'buy',
                'pre_trade_state': [1.0] * 19,
                'post_trade_state': [1.1] * 19,
                'reward': 10.0,
                'done': False,
                'success': True,
                'realized_pnl': 50.0,
                'fees': 3.0,
                'slippage': 0.01
            }
        ]
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        
        with patch('builtins.open', mock=Mock()) as mock_open:
            with patch('json.load', return_value=mock_experience_data) as mock_json_load:
                loaded_count = await collector.load_experiences()
                
                assert loaded_count == 1
                mock_open.assert_called_once()
                mock_json_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_trade_experience(self, collector_config, mock_replay_buffer, 
                                           mock_market_state, mock_trading_result):
        """Test capturing experience from failed trade"""
        mock_trading_result.success = False
        mock_trading_result.error_message = "Insufficient liquidity"
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(experience_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        
        # Failed trades should have negative reward and be marked as done
        assert completed_exp.success is False
        assert completed_exp.reward < 0
        assert completed_exp.done is True

    @pytest.mark.asyncio
    async def test_experience_timeout_cleanup(self, collector_config, mock_replay_buffer, 
                                              mock_market_state):
        """Test cleanup of timed-out incomplete experiences"""
        # Set very short timeout for testing
        collector_config.experience_timeout_hours = 0.001  # ~3.6 seconds
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        experience_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        
        # Wait for timeout
        await asyncio.sleep(4)
        
        # Clean up timed out experiences
        await collector.cleanup_timed_out_experiences()
        
        # Experience should be removed from active experiences
        assert experience_id not in collector.active_experiences

    @pytest.mark.asyncio
    async def test_min_experience_gap_enforcement(self, collector_config, mock_replay_buffer, 
                                                  mock_market_state):
        """Test minimum time gap between experiences is enforced"""
        collector_config.min_experience_gap_seconds = 10
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # First experience should work
        exp_id_1 = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        assert exp_id_1 is not None
        
        # Second experience immediately should be rejected
        exp_id_2 = await collector.capture_pre_trade_state(mock_market_state, TradeAction.SELL)
        assert exp_id_2 is None  # Should be rejected due to time gap

    @pytest.mark.asyncio
    async def test_max_experiences_per_session_limit(self, collector_config, mock_replay_buffer, 
                                                     mock_market_state, mock_trading_result):
        """Test maximum experiences per session limit"""
        collector_config.max_experiences_per_session = 2
        collector_config.min_experience_gap_seconds = 0  # No gap for testing
        
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # Add experiences up to limit
        for i in range(2):
            exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        # Third experience should be rejected
        exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        assert exp_id is None  # Should be rejected due to session limit

    @pytest.mark.asyncio
    async def test_experience_statistics(self, collector_config, mock_replay_buffer, 
                                         mock_market_state, mock_trading_result):
        """Test getting experience collection statistics"""
        collector_config.min_experience_gap_seconds = 0  # No gap for testing
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # Add some experiences
        for i in range(3):
            exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        stats = await collector.get_statistics()
        
        assert stats['total_experiences'] == 3
        assert stats['active_experiences'] == 0
        assert stats['completed_experiences'] == 3
        assert stats['session_duration'] > 0
        assert 'average_reward' in stats
        assert 'success_rate' in stats

    @pytest.mark.asyncio
    async def test_error_handling_invalid_experience_id(self, collector_config, mock_replay_buffer, 
                                                        mock_trading_result, mock_market_state):
        """Test error handling for invalid experience ID"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        invalid_id = str(uuid4())
        
        with pytest.raises(ExperienceCollectionError):
            await collector.capture_post_trade_result(invalid_id, mock_trading_result, mock_market_state)

    @pytest.mark.asyncio
    async def test_error_handling_collection_not_started(self, collector_config, mock_replay_buffer, 
                                                          mock_market_state):
        """Test error handling when collection not started"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        
        with pytest.raises(ExperienceCollectionError):
            await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)

    @pytest.mark.asyncio
    async def test_action_encoding_for_rl_training(self, collector_config, mock_replay_buffer, 
                                                   mock_market_state, mock_trading_result):
        """Test proper action encoding for RL training"""
        collector_config.min_experience_gap_seconds = 0  # No gap for testing
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        # Test all action types
        actions = [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD, 
                  TradeAction.STRONG_BUY, TradeAction.STRONG_SELL]
        
        for action in actions:
            mock_trading_result.action = action
            exp_id = await collector.capture_pre_trade_state(mock_market_state, action)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        assert len(collector.completed_experiences) == 5
        
        # Check action encoding
        expected_encodings = [1, 2, 0, 3, 4]  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
        for i, action in enumerate(actions):
            rl_exp = collector._convert_to_rl_experience(collector.completed_experiences[i])
            assert rl_exp.action == expected_encodings[i]

    @pytest.mark.asyncio 
    async def test_state_vector_consistency(self, collector_config, mock_replay_buffer, 
                                            mock_market_state, mock_trading_result):
        """Test state vector consistency and feature size"""
        collector = TradingExperienceCollector(collector_config, mock_replay_buffer)
        await collector.start_collection()
        
        exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        rl_exp = collector._convert_to_rl_experience(completed_exp)
        
        # State vectors should have consistent size
        assert len(rl_exp.state) == MarketState.get_feature_size()
        assert len(rl_exp.next_state) == MarketState.get_feature_size()
        
        # Values should be properly normalized
        assert np.all(np.isfinite(rl_exp.state))
        assert np.all(np.isfinite(rl_exp.next_state))


class TestExperienceCollectorConfig:
    """Test suite for ExperienceCollectorConfig"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = ExperienceCollectorConfig()
        
        assert config.buffer_size == 10000
        assert config.min_experience_gap_seconds == 1
        assert config.reward_calculation_window == 5
        assert config.enable_persistence is True
        assert config.persistence_path == "experiences.json"
        assert config.max_experiences_per_session == 1000
        assert config.experience_timeout_hours == 24

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = ExperienceCollectorConfig(
            buffer_size=5000,
            min_experience_gap_seconds=10,
            enable_persistence=False
        )
        
        assert config.buffer_size == 5000
        assert config.min_experience_gap_seconds == 10
        assert config.enable_persistence is False


class TestTradingExperienceData:
    """Test suite for TradingExperienceData"""
    
    def test_experience_data_creation(self, mock_market_state):
        """Test creating trading experience data"""
        exp_data = TradingExperienceData(
            experience_id=str(uuid4()),
            timestamp=datetime.now(),
            action=TradeAction.BUY,
            pre_trade_state=mock_market_state.to_vector(),
            post_trade_state=mock_market_state.to_vector(),
            reward=10.0,
            done=False,
            success=True,
            realized_pnl=50.0,
            fees=3.0,
            slippage=0.01
        )
        
        assert exp_data.action == TradeAction.BUY
        assert exp_data.reward == 10.0
        assert exp_data.success is True
        assert exp_data.done is False

    def test_experience_data_to_dict(self, mock_market_state):
        """Test converting experience data to dictionary"""
        exp_data = TradingExperienceData(
            experience_id=str(uuid4()),
            timestamp=datetime.now(),
            action=TradeAction.BUY,
            pre_trade_state=mock_market_state.to_vector(),
            post_trade_state=mock_market_state.to_vector(),
            reward=10.0,
            done=False,
            success=True,
            realized_pnl=50.0,
            fees=3.0,
            slippage=0.01
        )
        
        exp_dict = exp_data.to_dict()
        
        assert 'experience_id' in exp_dict
        assert 'timestamp' in exp_dict
        assert 'action' in exp_dict
        assert 'pre_trade_state' in exp_dict
        assert 'post_trade_state' in exp_dict
        assert 'reward' in exp_dict
        assert exp_dict['success'] is True

    def test_experience_data_from_dict(self, mock_market_state):
        """Test creating experience data from dictionary"""
        exp_dict = {
            'experience_id': str(uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'buy',
            'pre_trade_state': mock_market_state.to_vector().tolist(),
            'post_trade_state': mock_market_state.to_vector().tolist(),
            'reward': 10.0,
            'done': False,
            'success': True,
            'realized_pnl': 50.0,
            'fees': 3.0,
            'slippage': 0.01
        }
        
        exp_data = TradingExperienceData.from_dict(exp_dict)
        
        assert exp_data.action == TradeAction.BUY
        assert exp_data.reward == 10.0
        assert exp_data.success is True
        assert isinstance(exp_data.pre_trade_state, np.ndarray)
        assert isinstance(exp_data.post_trade_state, np.ndarray)