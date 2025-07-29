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


class TestTradingExperienceCollectorDatabaseIntegration:
    """Test suite for TradingExperienceCollector database integration"""

    @pytest.fixture
    def mock_database_connection(self):
        """Create mock database connection"""
        return AsyncMock()

    @pytest.fixture
    def collector_with_database_config(self):
        """Create experience collector configuration with database enabled"""
        return ExperienceCollectorConfig(
            buffer_size=1000,
            min_experience_gap_seconds=5,
            reward_calculation_window=10,
            enable_persistence=True,
            enable_database_storage=True,  # New database flag
            persistence_path=None,  # No file persistence when database enabled
            max_experiences_per_session=100,
            experience_timeout_hours=24,
            database_batch_size=50,  # New batch size for database operations
            database_connection_timeout=30  # New database timeout
        )

    @pytest.mark.asyncio
    async def test_database_connection_initialization(self, collector_with_database_config, mock_replay_buffer):
        """Test that collector initializes database connection on startup"""
        with patch('src.utils.database.get_database_connection') as mock_get_conn:
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            # Should initialize database connection
            mock_get_conn.assert_called()

    @pytest.mark.asyncio
    async def test_database_experience_persistence_batch_insert(self, collector_with_database_config, 
                                                               mock_replay_buffer, mock_market_state, 
                                                               mock_trading_result):
        """Test batch insertion of experiences to database"""
        with patch('src.utils.database.insert_experience_batch') as mock_batch_insert:
            mock_batch_insert.return_value = 3
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            # Create multiple experiences
            for i in range(3):
                exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
                await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            # Persist to database
            await collector.persist_experiences_to_database()
            
            # Should call batch insert with 3 experiences
            mock_batch_insert.assert_called_once()
            call_args = mock_batch_insert.call_args[0][0]
            assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_database_experience_loading_from_session(self, collector_with_database_config, mock_replay_buffer):
        """Test loading experiences from database by session ID"""
        mock_session_id = str(uuid4())
        mock_experiences = [
            {
                'id': 1,
                'session_id': mock_session_id,
                'step_number': 1,
                'state': [1.0] * 19,
                'action': 1,
                'reward': 10.0,
                'next_state': [1.1] * 19,
                'done': False,
                'priority': 0.8,
                'metadata': {'test': True},
                'created_at': datetime.now()
            }
        ]
        
        with patch('src.utils.database.query_experiences_by_session') as mock_query:
            mock_query.return_value = mock_experiences
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            # Load experiences from database
            loaded_experiences = await collector.load_experiences_from_database(mock_session_id)
            
            assert len(loaded_experiences) == 1
            mock_query.assert_called_once_with(mock_session_id, limit=1000, offset=0)

    @pytest.mark.asyncio
    async def test_database_experience_prioritized_sampling(self, collector_with_database_config, mock_replay_buffer):
        """Test prioritized sampling of experiences from database"""
        mock_experiences = [
            {'id': 1, 'priority': 0.9, 'reward': 15.0},
            {'id': 2, 'priority': 0.7, 'reward': 10.0},
            {'id': 3, 'priority': 0.5, 'reward': 5.0}
        ]
        
        with patch('src.utils.database.sample_prioritized_experiences') as mock_sample:
            mock_sample.return_value = mock_experiences
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            # Sample prioritized experiences
            sampled = await collector.sample_prioritized_experiences_from_database(batch_size=32, alpha=0.6)
            
            assert len(sampled) == 3
            mock_sample.assert_called_once_with(batch_size=32, alpha=0.6, session_ids=None)

    @pytest.mark.asyncio
    async def test_database_experience_checksum_validation(self, collector_with_database_config, 
                                                          mock_replay_buffer, mock_market_state, 
                                                          mock_trading_result):
        """Test experience validation with checksums for database storage"""
        # Mock database connection to avoid actual database calls
        with patch('src.utils.database.get_database_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            completed_exp = collector.completed_experiences[0]
            
            # Generate checksum for experience
            checksum = collector._generate_experience_checksum(completed_exp)
            
            # Checksum should be a valid hash string
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA-256 hash length
            
            # Validation should pass for correct checksum
            is_valid = collector._validate_experience_checksum(completed_exp, checksum)
            assert is_valid is True
            
            # Validation should fail for incorrect checksum
            is_valid = collector._validate_experience_checksum(completed_exp, "invalid_checksum")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_database_batch_processing_optimization(self, collector_with_database_config, 
                                                         mock_replay_buffer, mock_market_state, 
                                                         mock_trading_result):
        """Test batch processing optimization for database operations"""
        collector_with_database_config.database_batch_size = 2  # Small batch size for testing
        
        with patch('src.utils.database.insert_experience_batch') as mock_batch_insert:
            mock_batch_insert.return_value = 2
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            # Create 5 experiences (should trigger 3 batches: 2+2+1)
            for i in range(5):
                exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
                await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            # Process in batches
            await collector.process_experiences_in_batches()
            
            # Should call batch insert 3 times (2+2+1)
            assert mock_batch_insert.call_count == 3

    @pytest.mark.asyncio
    async def test_database_error_handling_connection_failure(self, collector_with_database_config, mock_replay_buffer):
        """Test database error handling for connection failures"""
        with patch('src.utils.database.get_database_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Database connection failed")
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            # Should handle database connection error gracefully
            with pytest.raises(ExperienceCollectionError, match="Database connection failed"):
                await collector.start_collection()

    @pytest.mark.asyncio
    async def test_database_error_handling_insertion_failure(self, collector_with_database_config, 
                                                            mock_replay_buffer, mock_market_state, 
                                                            mock_trading_result):
        """Test database error handling for insertion failures"""
        with patch('src.utils.database.insert_experience_batch') as mock_batch_insert:
            mock_batch_insert.side_effect = Exception("Database insertion failed")
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            # Should handle insertion failure and retry with fallback
            with patch.object(collector, '_fallback_to_file_persistence') as mock_fallback:
                await collector.persist_experiences_to_database()
                mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_error_recovery_with_retry_logic(self, collector_with_database_config, 
                                                          mock_replay_buffer, mock_market_state, 
                                                          mock_trading_result):
        """Test database error recovery with exponential backoff retry logic"""
        with patch('src.utils.database.insert_experience_batch') as mock_batch_insert:
            # Fail first two attempts, succeed on third
            mock_batch_insert.side_effect = [
                Exception("Temporary database error"),
                Exception("Temporary database error"),
                3  # Success
            ]
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            # Create experiences
            for i in range(3):
                exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
                await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            # Should retry and eventually succeed
            with patch('asyncio.sleep') as mock_sleep:  # Mock sleep to speed up test
                result = await collector.persist_experiences_to_database_with_retry()
                
                assert result == 3  # Successfully inserted 3 experiences
                assert mock_batch_insert.call_count == 3  # Two failures + one success
                assert mock_sleep.call_count == 2  # Two retries

    @pytest.mark.asyncio
    async def test_database_concurrent_access_handling(self, collector_with_database_config, mock_replay_buffer):
        """Test handling concurrent database access from multiple collectors"""
        with patch('src.utils.database.execute_concurrent_batch_operations') as mock_concurrent:
            mock_concurrent.return_value = [{'success': True, 'results': ['INSERT 0 1']}]
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            # Simulate concurrent operations
            operations = [
                [("INSERT INTO rl_experiences VALUES (...)", "param1")],
                [("INSERT INTO rl_experiences VALUES (...)", "param2")]
            ]
            
            results = await collector.execute_concurrent_database_operations(operations)
            
            assert len(results) == 1
            assert results[0]['success'] is True
            mock_concurrent.assert_called_once_with(operations)

    @pytest.mark.asyncio
    async def test_database_transaction_rollback_on_failure(self, collector_with_database_config, 
                                                          mock_replay_buffer, mock_market_state, 
                                                          mock_trading_result):
        """Test database transaction rollback on partial failure"""
        with patch('src.utils.database.atomic_experience_batch_operation') as mock_atomic:
            mock_atomic.side_effect = Exception("Transaction failed")
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            # Create experiences
            for i in range(3):
                exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
                await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            # Should handle transaction failure
            with pytest.raises(ExperienceCollectionError, match="Database transaction failed"):
                await collector.persist_experiences_atomically()

    @pytest.mark.asyncio
    async def test_database_experience_metadata_enrichment(self, collector_with_database_config, 
                                                          mock_replay_buffer, mock_market_state, 
                                                          mock_trading_result):
        """Test enrichment of experience metadata for database storage"""
        collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
        await collector.start_collection()
        
        exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
        await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
        
        completed_exp = collector.completed_experiences[0]
        
        # Enrich experience with database-specific metadata
        enriched_exp = collector._enrich_experience_for_database(completed_exp)
        
        # Should add database-specific fields
        assert hasattr(enriched_exp, 'database_session_id')
        assert hasattr(enriched_exp, 'checksum')
        assert hasattr(enriched_exp, 'version')
        assert hasattr(enriched_exp, 'storage_metadata')
        
        # Session ID should be valid UUID
        import uuid
        uuid.UUID(enriched_exp.database_session_id)  # Should not raise exception

    @pytest.mark.asyncio
    async def test_database_experience_lifecycle_tracking(self, collector_with_database_config, 
                                                         mock_replay_buffer, mock_market_state, 
                                                         mock_trading_result):
        """Test experience lifecycle tracking in database"""
        with patch('src.utils.database.update_experience_priorities') as mock_update_priorities:
            mock_update_priorities.return_value = 1
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            await collector.start_collection()
            
            exp_id = await collector.capture_pre_trade_state(mock_market_state, TradeAction.BUY)
            await collector.capture_post_trade_result(exp_id, mock_trading_result, mock_market_state)
            
            completed_exp = collector.completed_experiences[0]
            
            # Update experience priority after training
            priority_updates = [{
                'id': 1,
                'priority': 0.95,
                'td_error': 0.15
            }]
            
            updated_count = await collector.update_experience_priorities_in_database(priority_updates)
            
            assert updated_count == 1
            mock_update_priorities.assert_called_once_with(priority_updates)

    @pytest.mark.asyncio
    async def test_database_session_statistics_tracking(self, collector_with_database_config, 
                                                       mock_replay_buffer, mock_market_state, 
                                                       mock_trading_result):
        """Test session statistics tracking in database"""
        mock_session_id = str(uuid4())
        mock_stats = {
            'session_id': mock_session_id,
            'total_experiences': 5,
            'avg_reward': 8.5,
            'max_reward': 15.0,
            'min_reward': 2.0,
            'total_steps': 5,
            'completion_rate': 0.8,
            'avg_priority': 0.7,
            'reward_stddev': 4.2
        }
        
        with patch('src.utils.database.get_session_statistics') as mock_get_stats:
            mock_get_stats.return_value = mock_stats
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            stats = await collector.get_session_statistics_from_database(mock_session_id)
            
            assert stats['total_experiences'] == 5
            assert stats['avg_reward'] == 8.5
            assert stats['completion_rate'] == 0.8
            mock_get_stats.assert_called_once_with(mock_session_id)

    @pytest.mark.asyncio
    async def test_database_health_monitoring_integration(self, collector_with_database_config, mock_replay_buffer):
        """Test integration with database health monitoring"""
        mock_health_data = {
            'status': 'healthy',
            'total_experiences': 1500,
            'recent_experiences': 50,
            'active_sessions': 3,
            'avg_insertion_time_ms': 35.2,
            'avg_query_time_ms': 18.5,
            'indexes_healthy': True,
            'storage_utilization': 65.0
        }
        
        with patch('src.utils.database.check_rl_database_health') as mock_health_check:
            mock_health_check.return_value = mock_health_data
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            health_status = await collector.check_database_health()
            
            assert health_status['status'] == 'healthy'
            assert health_status['total_experiences'] == 1500
            assert health_status['avg_insertion_time_ms'] < 50  # Performance requirement
            mock_health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_memory_optimization_streaming(self, collector_with_database_config, mock_replay_buffer):
        """Test memory-optimized streaming of large experience datasets"""
        mock_session_id = str(uuid4())
        
        async def mock_stream_generator():
            """Mock streaming generator for large dataset"""
            for i in range(3):  # Simulate 3 batches
                yield [{'id': j + i*1000, 'step_number': j + i*1000} for j in range(1000)]
        
        with patch('src.utils.database.stream_experiences_with_memory_optimization') as mock_stream:
            mock_stream.return_value = mock_stream_generator()
            
            collector = TradingExperienceCollector(collector_with_database_config, mock_replay_buffer)
            
            total_experiences = 0
            async for batch in collector.stream_experiences_from_database(mock_session_id, batch_size=1000):
                total_experiences += len(batch)
                # Each batch should be 1000 experiences
                assert len(batch) == 1000
            
            # Should have streamed 3000 total experiences
            assert total_experiences == 3000
            mock_stream.assert_called_once_with(mock_session_id, batch_size=1000)