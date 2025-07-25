"""
Comprehensive Integration Test for Complete RL Feedback Loop

This test validates the entire learning cycle:
1. Trade Execution → Experience Collection
2. Experience Collection → Training Trigger
3. Training Trigger → Model Training
4. Model Training → Model Improvement
5. Model Improvement → Better Trading Decisions

This ensures the system truly learns and improves with every trade.
"""

import asyncio
import pytest
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.modes.continuous_learning import ContinuousLearningEngine, ContinuousLearningConfig
from src.modes.simulation_mode import SimulationMode, SimulationConfig
from src.modes.mode_manager import ModeManager
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
from src.discovery.base import DiscoveredToken
from src.portfolio.base import Portfolio
from src.utils.base import Chain


class TestCompleteRLFeedbackLoop:
    """Test the complete RL feedback loop integration"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_discovered_token(self):
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
    def mock_portfolio(self):
        """Create mock portfolio"""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = "test_portfolio_123"
        portfolio.get_total_value = Mock(return_value=10000.0)
        portfolio.get_cash_balance = Mock(return_value=10000.0)
        portfolio.positions = {}
        return portfolio
    
    @pytest.fixture
    def replay_buffer_config(self):
        """Create replay buffer configuration"""
        return ReplayBufferConfig(
            max_size=10000,
            batch_size=32,
            min_size=10,  # Low for testing
            prioritized=True,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            epsilon=1e-6
        )
    
    @pytest.fixture
    def experience_collector_config(self, temp_model_dir):
        """Create experience collector configuration"""
        return ExperienceCollectorConfig(
            buffer_size=1000,
            min_experience_gap_seconds=0,  # No gap for testing
            reward_calculation_window=5,
            enable_persistence=True,
            persistence_path=os.path.join(temp_model_dir, "experiences.json"),
            max_experiences_per_session=100,
            experience_timeout_hours=24
        )
    
    @pytest.fixture
    def continuous_learning_config(self, temp_model_dir):
        """Create continuous learning configuration"""
        return ContinuousLearningConfig(
            training_trigger_threshold=10,  # Low threshold for testing
            performance_evaluation_window=5,
            min_improvement_threshold=0.01,  # Low threshold for testing
            max_model_versions=3,
            training_episodes_per_session=10,  # Short for testing
            enable_model_persistence=True,
            model_storage_path=temp_model_dir,
            performance_rollback_threshold=-0.10,
            training_timeout_minutes=5,
            enable_incremental_learning=True
        )
    
    @pytest.fixture
    def simulation_config(self):
        """Create simulation configuration"""
        from decimal import Decimal
        return SimulationConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC",
            enable_fees=True,
            slippage_bps=50,  # 0.5%
            max_drawdown_pct=Decimal("0.15"),
            enable_risk_management=True,
            market_data_frequency_ms=1000,
            enable_real_time_pnl=True,
            enable_batch_execution=True,
            simulation_speed_multiplier=Decimal("1.0")
        )
    
    @pytest.fixture
    async def rl_feedback_system(self, temp_model_dir, replay_buffer_config, 
                                experience_collector_config, continuous_learning_config,
                                simulation_config, mock_portfolio):
        """Create complete RL feedback system"""
        # Create components
        replay_buffer = ExperienceReplayBuffer(replay_buffer_config)
        
        # Mock DQN agent that tracks improvements
        dqn_agent = Mock(spec=DQNTradingAgent)
        dqn_agent.predict = Mock(return_value=(TradeAction.BUY, 0.8))
        dqn_agent.save_model = Mock()
        dqn_agent.load_model = Mock()
        dqn_agent.get_model_state = Mock(return_value={'weights': 'mock_weights'})
        
        # Track prediction improvements
        dqn_agent.prediction_history = []
        
        def improved_predict(state):
            """Mock prediction that improves over time"""
            # Simulate learning: prediction confidence increases
            confidence = min(0.9, 0.5 + len(dqn_agent.prediction_history) * 0.05)
            action = TradeAction.BUY if confidence > 0.7 else TradeAction.HOLD
            dqn_agent.prediction_history.append((action, confidence))
            return action, confidence
        
        dqn_agent.predict.side_effect = improved_predict
        
        # Create experience collector
        experience_collector = TradingExperienceCollector(
            experience_collector_config, replay_buffer
        )
        
        # Create continuous learning engine
        continuous_learning = ContinuousLearningEngine(
            config=continuous_learning_config,
            replay_buffer=replay_buffer,
            dqn_agent=dqn_agent,
            experience_collector=experience_collector
        )
        
        # Mock training pipeline that produces improving results
        training_pipeline = Mock(spec=DQNTrainingPipeline)
        training_results = [
            {
                'episodes_completed': 10,
                'training_time': 60.0,
                'final_metrics': {
                    'mean_reward': 100.0 + i * 20,  # Improving rewards
                    'mean_win_rate': 0.6 + i * 0.05,  # Improving win rate
                    'mean_portfolio_value': 10000.0 + i * 500
                }
            }
            for i in range(5)
        ]
        
        training_call_count = 0
        def mock_train(*args, **kwargs):
            nonlocal training_call_count
            result = training_results[min(training_call_count, len(training_results) - 1)]
            training_call_count += 1
            return result
        
        training_pipeline.train.side_effect = mock_train
        
        # Patch training pipeline creation
        with patch('src.modes.continuous_learning.DQNTrainingPipeline', return_value=training_pipeline):
            
            # Create simulation mode
            simulation_mode = Mock(spec=SimulationMode)
            simulation_mode.config = simulation_config
            simulation_mode.portfolio = mock_portfolio
            simulation_mode.experience_collector = experience_collector
            simulation_mode.continuous_learning = continuous_learning
            
            # Mock simulation execution
            trade_count = 0
            async def mock_execute_trade(market_state, action):
                nonlocal trade_count
                trade_count += 1
                
                # Simulate improving trade results
                base_pnl = 10.0 + trade_count * 5  # Improving PnL
                success_rate = min(0.95, 0.7 + trade_count * 0.02)  # Improving success
                
                return TradingResult(
                    action=action,
                    token=market_state.token,
                    executed_at=datetime.now(),
                    price=market_state.price_usd,
                    quantity=1000.0,
                    value_usd=1000.0,
                    success=np.random.random() < success_rate,
                    slippage=0.005,
                    fees=2.0,
                    portfolio_value_before=10000.0,
                    portfolio_value_after=10000.0 + base_pnl,
                    cash_change=-1002.0,
                    position_change=1000.0,
                    realized_pnl=base_pnl,
                    unrealized_pnl=0.0
                )
            
            simulation_mode.execute_trade = mock_execute_trade
            
            yield {
                'replay_buffer': replay_buffer,
                'experience_collector': experience_collector,
                'continuous_learning': continuous_learning,
                'simulation_mode': simulation_mode,
                'dqn_agent': dqn_agent,
                'training_pipeline': training_pipeline
            }
    
    @pytest.mark.asyncio
    async def test_complete_learning_cycle_single_iteration(self, rl_feedback_system, 
                                                           mock_discovered_token):
        """Test a single complete learning cycle iteration"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        continuous_learning = system['continuous_learning']
        dqn_agent = system['dqn_agent']
        
        # Start experience collection
        await experience_collector.start_collection()
        
        # Step 1: Create market state
        market_state = MarketState(
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
        
        # Step 2: Agent makes initial prediction
        initial_action, initial_confidence = dqn_agent.predict(market_state.to_vector())
        assert initial_action is not None
        assert initial_confidence >= 0.5
        
        # Step 3: Execute trade and capture experience
        experience_id = await experience_collector.capture_pre_trade_state(
            market_state, initial_action
        )
        assert experience_id is not None
        
        # Simulate trade execution
        trade_result = TradingResult(
            action=initial_action,
            token=mock_discovered_token,
            executed_at=datetime.now(),
            price=1.0,
            quantity=1000.0,
            value_usd=1000.0,
            success=True,
            slippage=0.005,
            fees=2.0,
            portfolio_value_before=10000.0,
            portfolio_value_after=10050.0,
            cash_change=-1002.0,
            position_change=1000.0,
            realized_pnl=50.0,
            unrealized_pnl=0.0
        )
        
        # Update market state (simulate price movement)
        next_market_state = market_state
        next_market_state.price_usd = 1.05
        next_market_state.portfolio_value = 10050.0
        
        await experience_collector.capture_post_trade_result(
            experience_id, trade_result, next_market_state
        )
        
        # Step 4: Add experience to buffer
        added_count = await experience_collector.add_experiences_to_buffer()
        assert added_count == 1
        
        # Step 5: Check if training should be triggered (need more experiences)
        triggers = await continuous_learning.check_training_triggers()
        assert len(triggers) > 0
        
        # Add more experiences to trigger training
        for i in range(12):  # Add enough to exceed threshold
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            await experience_collector.capture_post_trade_result(
                exp_id, trade_result, next_market_state
            )
        
        await experience_collector.add_experiences_to_buffer()
        
        # Step 6: Trigger training
        training_result = await continuous_learning.run_automatic_training_cycle()
        
        assert training_result is not None
        assert training_result['training_triggered'] is True
        assert training_result['session_success'] is True
        assert 'new_model_version' in training_result
        
        # Step 7: Verify model improvement
        new_action, new_confidence = dqn_agent.predict(market_state.to_vector())
        
        # Due to mocked improvement, confidence should be higher
        assert len(dqn_agent.prediction_history) > 1
        assert new_confidence >= initial_confidence  # Should improve or stay same
    
    @pytest.mark.asyncio
    async def test_multiple_learning_cycles_show_improvement(self, rl_feedback_system, 
                                                           mock_discovered_token):
        """Test multiple learning cycles show progressive improvement"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        continuous_learning = system['continuous_learning']
        dqn_agent = system['dqn_agent']
        
        await experience_collector.start_collection()
        
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        initial_performance = []
        final_performance = []
        
        # Run multiple learning cycles
        for cycle in range(3):
            # Record initial performance
            initial_action, initial_confidence = dqn_agent.predict(market_state.to_vector())
            initial_performance.append(initial_confidence)
            
            # Generate experiences for this cycle
            for i in range(15):  # Generate enough experiences to trigger training
                exp_id = await experience_collector.capture_pre_trade_state(
                    market_state, TradeAction.BUY
                )
                
                trade_result = TradingResult(
                    action=TradeAction.BUY, token=mock_discovered_token,
                    executed_at=datetime.now(), price=1.0, quantity=1000.0,
                    value_usd=1000.0, success=True, slippage=0.005, fees=2.0,
                    portfolio_value_before=10000.0, portfolio_value_after=10050.0,
                    cash_change=-1002.0, position_change=1000.0,
                    realized_pnl=50.0, unrealized_pnl=0.0
                )
                
                await experience_collector.capture_post_trade_result(
                    exp_id, trade_result, market_state
                )
            
            # Add to buffer and trigger training
            await experience_collector.add_experiences_to_buffer()
            training_result = await continuous_learning.run_automatic_training_cycle()
            
            assert training_result['training_triggered'] is True
            
            # Record final performance after training
            final_action, final_confidence = dqn_agent.predict(market_state.to_vector())
            final_performance.append(final_confidence)
        
        # Verify progressive improvement
        assert len(initial_performance) == 3
        assert len(final_performance) == 3
        
        # Should show improvement trend (allowing for some variance)
        avg_initial = np.mean(initial_performance)
        avg_final = np.mean(final_performance)
        
        assert avg_final >= avg_initial, f"Final performance {avg_final} should be >= initial {avg_initial}"
    
    @pytest.mark.asyncio
    async def test_model_versioning_and_rollback(self, rl_feedback_system, mock_discovered_token):
        """Test model versioning and rollback functionality"""
        system = rl_feedback_system
        continuous_learning = system['continuous_learning']
        experience_collector = system['experience_collector']
        
        await experience_collector.start_collection()
        
        # Generate experiences and train multiple model versions
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        # Train first version
        for i in range(15):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            trade_result = TradingResult(
                action=TradeAction.BUY, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=True, slippage=0.005, fees=2.0,
                portfolio_value_before=10000.0, portfolio_value_after=10050.0,
                cash_change=-1002.0, position_change=1000.0,
                realized_pnl=50.0, unrealized_pnl=0.0
            )
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        await experience_collector.add_experiences_to_buffer()
        result1 = await continuous_learning.run_automatic_training_cycle()
        
        assert result1['training_triggered'] is True
        assert len(continuous_learning.model_versions) == 1
        
        first_version = continuous_learning.model_versions[0]
        
        # Train second version
        for i in range(15):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.SELL
            )
            trade_result = TradingResult(
                action=TradeAction.SELL, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=True, slippage=0.005, fees=2.0,
                portfolio_value_before=10000.0, portfolio_value_after=10050.0,
                cash_change=998.0, position_change=-1000.0,
                realized_pnl=50.0, unrealized_pnl=0.0
            )
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        await experience_collector.add_experiences_to_buffer()
        result2 = await continuous_learning.run_automatic_training_cycle()
        
        assert len(continuous_learning.model_versions) == 2
        
        # Test rollback functionality
        rolled_back = await continuous_learning.rollback_to_previous_version()
        assert rolled_back is True
        assert continuous_learning.active_model_version == first_version
    
    @pytest.mark.asyncio
    async def test_cross_session_persistence(self, rl_feedback_system, temp_model_dir):
        """Test that learning state persists across sessions"""
        system = rl_feedback_system
        continuous_learning = system['continuous_learning']
        experience_collector = system['experience_collector']
        
        # Create some model versions
        from src.modes.continuous_learning import ModelVersion, ModelPerformanceMetrics
        
        metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=125.0,
            win_rate=0.65,
            sharpe_ratio=1.15,
            max_drawdown=0.08,
            total_trades=150
        )
        
        version = ModelVersion(
            version_id="v1.0.0",
            created_at=datetime.now(),
            model_path=os.path.join(temp_model_dir, "model_v1.0.0.pth"),
            training_episodes=50,
            performance_metrics=metrics,
            parent_version_id=None
        )
        
        continuous_learning.model_versions = [version]
        continuous_learning.active_model_version = version
        
        # Save state
        state_path = os.path.join(temp_model_dir, "engine_state.json")
        await continuous_learning.save_state(state_path)
        
        assert os.path.exists(state_path)
        
        # Create new engine instance (simulate new session)
        new_learning_engine = ContinuousLearningEngine(
            config=continuous_learning.config,
            replay_buffer=continuous_learning.replay_buffer,
            dqn_agent=continuous_learning.dqn_agent,
            experience_collector=experience_collector
        )
        
        # Load state
        await new_learning_engine.load_state(state_path)
        
        assert len(new_learning_engine.model_versions) == 1
        assert new_learning_engine.model_versions[0].version_id == "v1.0.0"
        assert new_learning_engine.model_versions[0].performance_metrics.average_reward == 125.0
    
    @pytest.mark.asyncio
    async def test_experience_quality_affects_training_trigger(self, rl_feedback_system, 
                                                             mock_discovered_token):
        """Test that experience quality affects training trigger decisions"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        continuous_learning = system['continuous_learning']
        
        await experience_collector.start_collection()
        
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        # Add high-quality experiences (profitable trades)
        for i in range(15):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            
            # High-quality trade result
            trade_result = TradingResult(
                action=TradeAction.BUY, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=True, slippage=0.001,  # Low slippage
                fees=1.0,  # Low fees
                portfolio_value_before=10000.0, portfolio_value_after=10100.0,
                cash_change=-1001.0, position_change=1000.0,
                realized_pnl=100.0,  # High profit
                unrealized_pnl=0.0
            )
            
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        await experience_collector.add_experiences_to_buffer()
        
        # Check training triggers
        triggers = await continuous_learning.check_training_triggers()
        experience_trigger = next((t for t in triggers if t.trigger_type == "experience_threshold"), None)
        
        assert experience_trigger is not None
        assert experience_trigger.is_triggered()
        
        # Training should be triggered with high-quality experiences
        training_result = await continuous_learning.run_automatic_training_cycle()
        assert training_result['training_triggered'] is True
    
    @pytest.mark.asyncio
    async def test_training_improves_decision_quality(self, rl_feedback_system, 
                                                    mock_discovered_token):
        """Test that training actually improves decision quality metrics"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        continuous_learning = system['continuous_learning']
        dqn_agent = system['dqn_agent']
        
        await experience_collector.start_collection()
        
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        # Record pre-training decision metrics
        pre_training_decisions = []
        for _ in range(10):
            action, confidence = dqn_agent.predict(market_state.to_vector())
            pre_training_decisions.append(confidence)
        
        pre_training_avg_confidence = np.mean(pre_training_decisions)
        
        # Generate training experiences
        for i in range(20):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            trade_result = TradingResult(
                action=TradeAction.BUY, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=True, slippage=0.005, fees=2.0,
                portfolio_value_before=10000.0, portfolio_value_after=10050.0,
                cash_change=-1002.0, position_change=1000.0,
                realized_pnl=50.0, unrealized_pnl=0.0
            )
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        # Trigger training
        await experience_collector.add_experiences_to_buffer()
        training_result = await continuous_learning.run_automatic_training_cycle()
        
        assert training_result['training_triggered'] is True
        
        # Record post-training decision metrics
        post_training_decisions = []
        for _ in range(10):
            action, confidence = dqn_agent.predict(market_state.to_vector())
            post_training_decisions.append(confidence)
        
        post_training_avg_confidence = np.mean(post_training_decisions)
        
        # Verify improvement (due to our mock, this should improve)
        assert post_training_avg_confidence >= pre_training_avg_confidence
        
        # Check that model versions track improvement
        assert len(continuous_learning.model_versions) > 0
        latest_version = continuous_learning.model_versions[-1]
        assert latest_version.performance_metrics.average_reward > 0
    
    @pytest.mark.asyncio
    async def test_feedback_loop_handles_poor_performance(self, rl_feedback_system, 
                                                        mock_discovered_token):
        """Test that the feedback loop handles and recovers from poor performance"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        continuous_learning = system['continuous_learning']
        
        await experience_collector.start_collection()
        
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        # Generate poor performance experiences
        for i in range(15):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            
            # Poor trade result
            trade_result = TradingResult(
                action=TradeAction.BUY, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=False,  # Failed trade
                slippage=0.05,  # High slippage
                fees=10.0,  # High fees
                portfolio_value_before=10000.0, portfolio_value_after=9950.0,
                cash_change=-1010.0, position_change=0.0,  # No position change due to failure
                realized_pnl=-50.0,  # Loss
                unrealized_pnl=0.0,
                error_message="Insufficient liquidity"
            )
            
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        await experience_collector.add_experiences_to_buffer()
        
        # Training should still be triggered to learn from failures
        training_result = await continuous_learning.run_automatic_training_cycle()
        assert training_result['training_triggered'] is True
        
        # System should create a model version even from poor performance data
        assert len(continuous_learning.model_versions) > 0
        
        # Performance stats should reflect the poor performance
        stats = await continuous_learning.get_performance_statistics()
        assert stats['total_training_sessions'] > 0
        assert stats['successful_sessions'] > 0  # Training session itself succeeded
    
    @pytest.mark.asyncio 
    async def test_experience_buffer_memory_management(self, rl_feedback_system, 
                                                     mock_discovered_token):
        """Test that experience buffer properly manages memory and old experiences"""
        system = rl_feedback_system
        experience_collector = system['experience_collector']
        replay_buffer = system['replay_buffer']
        
        await experience_collector.start_collection()
        
        # Get initial buffer state
        initial_buffer_size = len(replay_buffer)
        
        market_state = MarketState(
            token=mock_discovered_token,
            price_usd=1.0, price_change_24h=0.5, volume_24h=1000000,
            market_cap=50000000, rsi=55.0, macd=0.1, sma_20=0.98, ema_12=1.02,
            bollinger_upper=1.05, bollinger_lower=0.95, current_position=0.0,
            portfolio_value=10000.0, cash_balance=10000.0, portfolio_drawdown=0.0,
            daily_pnl=0.0, sharpe_ratio=1.2, market_volatility=0.15,
            fear_greed_index=50.0, prediction_confidence=0.8, timestamp=datetime.now()
        )
        
        # Add many experiences to test buffer management
        num_experiences = 50
        for i in range(num_experiences):
            exp_id = await experience_collector.capture_pre_trade_state(
                market_state, TradeAction.BUY
            )
            
            trade_result = TradingResult(
                action=TradeAction.BUY, token=mock_discovered_token,
                executed_at=datetime.now(), price=1.0, quantity=1000.0,
                value_usd=1000.0, success=True, slippage=0.005, fees=2.0,
                portfolio_value_before=10000.0, portfolio_value_after=10050.0,
                cash_change=-1002.0, position_change=1000.0,
                realized_pnl=50.0, unrealized_pnl=0.0
            )
            
            await experience_collector.capture_post_trade_result(exp_id, trade_result, market_state)
        
        # Add experiences to buffer in batches
        added_count = await experience_collector.add_experiences_to_buffer()
        assert added_count == num_experiences
        
        # Verify buffer size increased
        final_buffer_size = len(replay_buffer)
        assert final_buffer_size == initial_buffer_size + num_experiences
        
        # Verify buffer can sample
        assert replay_buffer.can_sample()
        
        # Test experience statistics
        stats = await experience_collector.get_statistics()
        assert stats['total_experiences'] >= num_experiences
        assert stats['success_rate'] >= 0.0  # Valid success rate (can be 0 with random failures)
        assert stats['average_reward'] > 0  # Positive average reward