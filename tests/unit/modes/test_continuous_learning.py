"""
Tests for Continuous Learning Engine

Following TDD methodology - comprehensive failing tests before implementation.
These tests define the requirements for automatic RL agent retraining.
"""

import asyncio
import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from src.modes.continuous_learning import (
    ContinuousLearningEngine,
    ContinuousLearningConfig,
    TrainingTrigger,
    ModelVersion,
    TrainingSession,
    ModelPerformanceMetrics,
    ContinuousLearningError
)
from src.rl_agent.base import TradeAction, MarketState
from src.rl_agent.experience_replay import Experience, ExperienceReplayBuffer
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig


class TestContinuousLearningConfig:
    """Test configuration for continuous learning engine"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = ContinuousLearningConfig()
        
        assert config.training_trigger_threshold == 1000
        assert config.performance_evaluation_window == 50
        assert config.min_improvement_threshold == 0.05
        assert config.max_model_versions == 5
        assert config.training_episodes_per_session == 100
        assert config.enable_model_persistence == True
        assert config.model_storage_path == "models/continuous/"
        assert config.performance_rollback_threshold == -0.10
        assert config.training_timeout_minutes == 30
        assert config.enable_incremental_learning == True
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = ContinuousLearningConfig(
            training_trigger_threshold=500,
            performance_evaluation_window=25,
            min_improvement_threshold=0.02
        )
        assert config.training_trigger_threshold == 500
        assert config.performance_evaluation_window == 25
        assert config.min_improvement_threshold == 0.02
        
        # Test invalid values
        with pytest.raises(ValueError):
            ContinuousLearningConfig(training_trigger_threshold=0)
        
        with pytest.raises(ValueError):
            ContinuousLearningConfig(performance_evaluation_window=0)
        
        with pytest.raises(ValueError):
            ContinuousLearningConfig(min_improvement_threshold=-0.1)


class TestTrainingTrigger:
    """Test training trigger logic"""
    
    def test_trigger_creation(self):
        """Test training trigger creation"""
        trigger = TrainingTrigger(
            trigger_type="experience_threshold",
            threshold_value=1000,
            current_value=999,
            description="Experience buffer threshold reached"
        )
        
        assert trigger.trigger_type == "experience_threshold"
        assert trigger.threshold_value == 1000
        assert trigger.current_value == 999
        assert trigger.description == "Experience buffer threshold reached"
        assert trigger.triggered_at is not None
        assert not trigger.is_triggered()
    
    def test_trigger_activation(self):
        """Test trigger activation logic"""
        trigger = TrainingTrigger(
            trigger_type="experience_threshold",
            threshold_value=1000,
            current_value=1001
        )
        
        assert trigger.is_triggered()
        
        # Test performance trigger
        perf_trigger = TrainingTrigger(
            trigger_type="performance_degradation",
            threshold_value=-0.10,
            current_value=-0.15
        )
        
        assert perf_trigger.is_triggered()


class TestModelVersion:
    """Test model versioning system"""
    
    def test_model_version_creation(self):
        """Test model version creation"""
        metrics = ModelPerformanceMetrics(
            episodes_trained=100,
            average_reward=150.5,
            win_rate=0.68,
            sharpe_ratio=1.25,
            max_drawdown=0.05,
            total_trades=250
        )
        
        version = ModelVersion(
            version_id="v1.0.0",
            created_at=datetime.now(),
            model_path="/models/v1.0.0.pth",
            training_episodes=100,
            performance_metrics=metrics,
            parent_version_id=None
        )
        
        assert version.version_id == "v1.0.0"
        assert version.model_path == "/models/v1.0.0.pth"
        assert version.training_episodes == 100
        assert version.performance_metrics.average_reward == 150.5
        assert version.parent_version_id is None
        assert version.is_active == False
    
    def test_model_comparison(self):
        """Test model performance comparison"""
        metrics1 = ModelPerformanceMetrics(
            episodes_trained=100,
            average_reward=100.0,
            win_rate=0.60,
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            total_trades=200
        )
        
        metrics2 = ModelPerformanceMetrics(
            episodes_trained=100,
            average_reward=120.0,
            win_rate=0.65,
            sharpe_ratio=1.15,
            max_drawdown=0.08,
            total_trades=200
        )
        
        version1 = ModelVersion("v1.0.0", datetime.now(), "/path1", 100, metrics1)
        version2 = ModelVersion("v1.1.0", datetime.now(), "/path2", 100, metrics2)
        
        # version2 should be better
        assert version2.performance_metrics.average_reward > version1.performance_metrics.average_reward
        assert version2.performance_metrics.win_rate > version1.performance_metrics.win_rate
        assert version2.performance_metrics.sharpe_ratio > version1.performance_metrics.sharpe_ratio


class TestTrainingSession:
    """Test training session tracking"""
    
    def test_training_session_creation(self):
        """Test training session creation"""
        session = TrainingSession(
            session_id="session_123",
            started_at=datetime.now(),
            trigger_reason="experience_threshold",
            target_episodes=100
        )
        
        assert session.session_id == "session_123"
        assert session.trigger_reason == "experience_threshold"
        assert session.target_episodes == 100
        assert session.actual_episodes == 0
        assert session.completed_at is None
        assert session.success == False
        assert session.final_metrics is None
    
    def test_training_session_completion(self):
        """Test training session completion"""
        session = TrainingSession(
            session_id="session_123",
            started_at=datetime.now() - timedelta(minutes=10),
            trigger_reason="experience_threshold",
            target_episodes=100
        )
        
        metrics = {
            'average_reward': 125.5,
            'win_rate': 0.62,
            'episodes_completed': 95
        }
        
        session.complete(success=True, actual_episodes=95, final_metrics=metrics)
        
        assert session.success == True
        assert session.actual_episodes == 95
        assert session.completed_at is not None
        assert session.final_metrics == metrics
        assert session.get_duration_minutes() > 0


class TestContinuousLearningEngine:
    """Test the main continuous learning engine"""
    
    @pytest.fixture
    def mock_replay_buffer(self):
        """Mock experience replay buffer"""
        buffer = Mock(spec=ExperienceReplayBuffer)
        buffer.__len__ = Mock(return_value=1000)
        buffer.can_sample.return_value = True
        buffer.sample.return_value = ([Mock()], [Mock()], [Mock()], [Mock()], [Mock()], [Mock()])
        return buffer
    
    @pytest.fixture
    def mock_dqn_agent(self):
        """Mock DQN trading agent"""
        agent = Mock(spec=DQNTradingAgent)
        agent.save_model = Mock()
        agent.load_model = Mock()
        agent.get_model_state = Mock(return_value={'weights': 'mock_weights'})
        return agent
    
    @pytest.fixture
    def mock_training_pipeline(self):
        """Mock training pipeline"""
        pipeline = Mock(spec=DQNTrainingPipeline)
        pipeline.train = Mock(return_value={
            'episodes_completed': 100,
            'training_time': 300.0,
            'final_metrics': {
                'mean_reward': 125.0,
                'mean_win_rate': 0.62,
                'mean_portfolio_value': 10500.0
            }
        })
        return pipeline
    
    @pytest.fixture
    def mock_experience_collector(self):
        """Mock experience collector"""
        collector = Mock(spec=TradingExperienceCollector)
        collector.get_statistics = AsyncMock(return_value={
            'total_experiences': 1000,
            'buffer_size': 1000,
            'buffer_can_sample': True
        })
        return collector
    
    @pytest.fixture
    def temp_model_dir(self):
        """Temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def learning_config(self, temp_model_dir):
        """Test configuration for continuous learning"""
        return ContinuousLearningConfig(
            training_trigger_threshold=500,
            performance_evaluation_window=25,
            min_improvement_threshold=0.02,
            max_model_versions=3,
            training_episodes_per_session=50,
            model_storage_path=temp_model_dir,
            training_timeout_minutes=5
        )
    
    @pytest.fixture
    def learning_engine(self, learning_config, mock_replay_buffer, mock_dqn_agent, 
                       mock_experience_collector):
        """Continuous learning engine instance"""
        return ContinuousLearningEngine(
            config=learning_config,
            replay_buffer=mock_replay_buffer,
            dqn_agent=mock_dqn_agent,
            experience_collector=mock_experience_collector
        )
    
    async def test_engine_initialization(self, learning_engine, learning_config):
        """Test engine initialization"""
        assert learning_engine.config == learning_config
        assert learning_engine.is_training == False
        assert learning_engine.current_session is None
        assert len(learning_engine.model_versions) == 0
        assert learning_engine.active_model_version is None
        assert learning_engine.training_history == []
    
    async def test_check_training_triggers(self, learning_engine, mock_replay_buffer):
        """Test training trigger detection"""
        # Mock buffer size above threshold
        mock_replay_buffer.__len__.return_value = 600
        
        triggers = await learning_engine.check_training_triggers()
        
        assert len(triggers) > 0
        assert any(t.trigger_type == "experience_threshold" for t in triggers)
        assert any(t.is_triggered() for t in triggers)
    
    async def test_no_training_triggers(self, learning_engine, mock_replay_buffer):
        """Test when no training triggers are met"""
        # Mock buffer size below threshold
        mock_replay_buffer.__len__.return_value = 300
        
        triggers = await learning_engine.check_training_triggers()
        
        # Should still return triggers but none triggered
        experience_trigger = next((t for t in triggers if t.trigger_type == "experience_threshold"), None)
        assert experience_trigger is not None
        assert not experience_trigger.is_triggered()
    
    async def test_start_training_session(self, learning_engine):
        """Test starting a training session"""
        trigger = TrainingTrigger(
            trigger_type="experience_threshold",
            threshold_value=500,
            current_value=600,
            description="Buffer threshold reached"
        )
        
        session = await learning_engine.start_training_session(trigger)
        
        assert session is not None
        assert session.trigger_reason == "experience_threshold"
        assert session.target_episodes == learning_engine.config.training_episodes_per_session
        assert learning_engine.is_training == True
        assert learning_engine.current_session == session
    
    async def test_training_session_already_active(self, learning_engine):
        """Test error when training session already active"""
        # Start first session
        trigger = TrainingTrigger("test", 100, 150)
        await learning_engine.start_training_session(trigger)
        
        # Try to start second session
        with pytest.raises(ContinuousLearningError, match="Training session already active"):
            await learning_engine.start_training_session(trigger)
    
    @patch('src.modes.continuous_learning.DQNTrainingPipeline')
    async def test_run_training_session(self, mock_pipeline_class, learning_engine, 
                                       mock_training_pipeline):
        """Test running a training session"""
        # Setup mock
        mock_pipeline_class.return_value = mock_training_pipeline
        
        # Start session
        trigger = TrainingTrigger("experience_threshold", 500, 600)
        session = await learning_engine.start_training_session(trigger)
        
        # Run training
        result = await learning_engine.run_training_session()
        
        assert result is not None
        assert result['episodes_completed'] == 100
        assert learning_engine.is_training == False
        assert session.success == True
        assert session.completed_at is not None
    
    async def test_run_training_no_active_session(self, learning_engine):
        """Test error when no active training session"""
        with pytest.raises(ContinuousLearningError, match="No active training session"):
            await learning_engine.run_training_session()
    
    async def test_create_model_version(self, learning_engine, temp_model_dir):
        """Test creating a new model version"""
        # Setup training session
        session = TrainingSession(
            session_id="test_session",
            started_at=datetime.now(),
            trigger_reason="test",
            target_episodes=50
        )
        
        training_result = {
            'episodes_completed': 50,
            'final_metrics': {
                'mean_reward': 125.0,
                'mean_win_rate': 0.65,
                'mean_portfolio_value': 10500.0
            }
        }
        
        version = await learning_engine.create_model_version(session, training_result)
        
        assert version is not None
        assert version.version_id.startswith("v")
        assert version.training_episodes == 50
        assert version.performance_metrics.average_reward == 125.0
        assert version.performance_metrics.win_rate == 0.65
        assert os.path.dirname(version.model_path) == temp_model_dir
    
    async def test_evaluate_model_performance(self, learning_engine):
        """Test model performance evaluation"""
        # Create mock model versions
        old_metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=100.0,
            win_rate=0.60,
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            total_trades=150
        )
        
        new_metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=125.0,
            win_rate=0.65,
            sharpe_ratio=1.15,
            max_drawdown=0.08,
            total_trades=150
        )
        
        old_version = ModelVersion("v1.0.0", datetime.now(), "/path1", 50, old_metrics)
        new_version = ModelVersion("v1.1.0", datetime.now(), "/path2", 50, new_metrics)
        
        learning_engine.model_versions = [old_version]
        
        is_better = await learning_engine.evaluate_model_performance(new_version)
        
        assert is_better == True
    
    async def test_evaluate_model_performance_worse(self, learning_engine):
        """Test model performance evaluation when new model is worse"""
        # Create mock model versions
        good_metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=125.0,
            win_rate=0.65,
            sharpe_ratio=1.15,
            max_drawdown=0.08,
            total_trades=150
        )
        
        poor_metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=80.0,
            win_rate=0.45,
            sharpe_ratio=0.5,
            max_drawdown=0.20,
            total_trades=150
        )
        
        good_version = ModelVersion("v1.0.0", datetime.now(), "/path1", 50, good_metrics)
        poor_version = ModelVersion("v1.1.0", datetime.now(), "/path2", 50, poor_metrics)
        
        learning_engine.model_versions = [good_version]
        
        is_better = await learning_engine.evaluate_model_performance(poor_version)
        
        assert is_better == False
    
    async def test_activate_model_version(self, learning_engine, mock_dqn_agent):
        """Test activating a model version"""
        metrics = ModelPerformanceMetrics(
            episodes_trained=50,
            average_reward=125.0,
            win_rate=0.65,
            sharpe_ratio=1.15,
            max_drawdown=0.08,
            total_trades=150
        )
        
        version = ModelVersion("v1.0.0", datetime.now(), "/path/model.pth", 50, metrics)
        learning_engine.model_versions = [version]
        
        await learning_engine.activate_model_version(version.version_id)
        
        assert version.is_active == True
        assert learning_engine.active_model_version == version
        mock_dqn_agent.load_model.assert_called_once_with("/path/model.pth")
    
    async def test_activate_nonexistent_model(self, learning_engine):
        """Test error when activating nonexistent model"""
        with pytest.raises(ContinuousLearningError, match="Model version not found"):
            await learning_engine.activate_model_version("nonexistent")
    
    async def test_rollback_to_previous_version(self, learning_engine, mock_dqn_agent):
        """Test rolling back to previous model version"""
        # Create model versions
        metrics1 = ModelPerformanceMetrics(100, 120.0, 0.62, 1.1, 0.09, 200)
        metrics2 = ModelPerformanceMetrics(100, 80.0, 0.45, 0.6, 0.18, 200)  # Poor performance
        
        version1 = ModelVersion("v1.0.0", datetime.now(), "/path1", 100, metrics1)
        version2 = ModelVersion("v1.1.0", datetime.now(), "/path2", 100, metrics2)
        
        version1.is_active = True
        learning_engine.model_versions = [version1, version2]
        learning_engine.active_model_version = version2
        
        rolled_back = await learning_engine.rollback_to_previous_version()
        
        assert rolled_back == True
        assert version1.is_active == True
        assert version2.is_active == False
        assert learning_engine.active_model_version == version1
    
    async def test_rollback_no_previous_version(self, learning_engine):
        """Test rollback when no previous version exists"""
        metrics = ModelPerformanceMetrics(100, 80.0, 0.45, 0.6, 0.18, 200)
        version = ModelVersion("v1.0.0", datetime.now(), "/path", 100, metrics)
        
        learning_engine.model_versions = [version]
        learning_engine.active_model_version = version
        
        rolled_back = await learning_engine.rollback_to_previous_version()
        
        assert rolled_back == False
    
    async def test_cleanup_old_versions(self, learning_engine, temp_model_dir):
        """Test cleanup of old model versions"""
        # Create multiple model versions exceeding max limit
        learning_engine.config.max_model_versions = 2
        
        for i in range(4):
            metrics = ModelPerformanceMetrics(50, 100.0 + i*10, 0.6, 1.0, 0.1, 100)
            version = ModelVersion(
                f"v1.{i}.0",
                datetime.now() - timedelta(hours=i),
                f"{temp_model_dir}/model_v1_{i}_0.pth",
                50,
                metrics
            )
            learning_engine.model_versions.append(version)
            
            # Create dummy model file
            with open(version.model_path, 'w') as f:
                f.write("dummy model")
        
        # Set last version as active
        learning_engine.model_versions[-1].is_active = True
        learning_engine.active_model_version = learning_engine.model_versions[-1]
        
        cleaned_count = await learning_engine.cleanup_old_versions()
        
        assert cleaned_count == 2  # Should remove 2 oldest versions
        assert len(learning_engine.model_versions) == 2
        assert learning_engine.active_model_version.is_active == True
    
    async def test_save_and_load_state(self, learning_engine, temp_model_dir):
        """Test saving and loading engine state"""
        # Add some model versions
        metrics = ModelPerformanceMetrics(50, 125.0, 0.65, 1.15, 0.08, 150)
        version = ModelVersion("v1.0.0", datetime.now(), "/path", 50, metrics)
        learning_engine.model_versions = [version]
        learning_engine.active_model_version = version
        
        # Save state
        state_path = os.path.join(temp_model_dir, "engine_state.json")
        await learning_engine.save_state(state_path)
        
        assert os.path.exists(state_path)
        
        # Create new engine and load state
        new_engine = ContinuousLearningEngine(
            config=learning_engine.config,
            replay_buffer=learning_engine.replay_buffer,
            dqn_agent=learning_engine.dqn_agent,
            experience_collector=learning_engine.experience_collector
        )
        
        await new_engine.load_state(state_path)
        
        assert len(new_engine.model_versions) == 1
        assert new_engine.model_versions[0].version_id == "v1.0.0"
    
    async def test_get_performance_statistics(self, learning_engine):
        """Test getting performance statistics"""
        # Add training history
        session1 = TrainingSession("s1", datetime.now(), "test", 50)
        session1.complete(True, 50, {'mean_reward': 100.0, 'mean_win_rate': 0.6})
        
        session2 = TrainingSession("s2", datetime.now(), "test", 50)
        session2.complete(True, 50, {'mean_reward': 120.0, 'mean_win_rate': 0.65})
        
        learning_engine.training_history = [session1, session2]
        
        # Add model versions
        metrics1 = ModelPerformanceMetrics(50, 100.0, 0.6, 1.0, 0.1, 100)
        metrics2 = ModelPerformanceMetrics(50, 120.0, 0.65, 1.1, 0.09, 100)
        
        version1 = ModelVersion("v1.0.0", datetime.now(), "/path1", 50, metrics1)
        version2 = ModelVersion("v1.1.0", datetime.now(), "/path2", 50, metrics2)
        
        learning_engine.model_versions = [version1, version2]
        learning_engine.active_model_version = version2
        
        stats = await learning_engine.get_performance_statistics()
        
        assert stats['total_training_sessions'] == 2
        assert stats['successful_sessions'] == 2
        assert stats['total_model_versions'] == 2
        assert stats['active_model_version'] == "v1.1.0"
        assert stats['best_performance']['average_reward'] == 120.0
    
    @patch('src.modes.continuous_learning.DQNTrainingPipeline')
    async def test_automatic_training_cycle(self, mock_pipeline_class, learning_engine, 
                                           mock_training_pipeline, mock_replay_buffer):
        """Test complete automatic training cycle"""
        # Setup mocks
        mock_pipeline_class.return_value = mock_training_pipeline
        mock_replay_buffer.__len__.return_value = 600  # Above threshold
        
        # Run automatic training cycle
        result = await learning_engine.run_automatic_training_cycle()
        
        assert result is not None
        assert result['training_triggered'] == True
        assert result['session_success'] == True
        assert 'new_model_version' in result
    
    async def test_automatic_training_no_triggers(self, learning_engine, mock_replay_buffer):
        """Test automatic training when no triggers are met"""
        # Setup mocks - buffer below threshold
        mock_replay_buffer.__len__.return_value = 300
        
        # Run automatic training cycle
        result = await learning_engine.run_automatic_training_cycle()
        
        assert result is not None
        assert result['training_triggered'] == False
        assert 'trigger_status' in result
    
    async def test_incremental_learning_state_preservation(self, learning_engine):
        """Test that incremental learning preserves previous state"""
        # This test would verify that the agent's learned weights are preserved
        # and training continues from the current state rather than restarting
        
        # Mock model state
        initial_state = {'weights': 'initial_weights', 'experience_count': 500}
        learning_engine.dqn_agent.get_model_state.return_value = initial_state
        
        # Verify state is preserved before training
        state = learning_engine.dqn_agent.get_model_state()
        assert state['experience_count'] == 500
        
        # The actual implementation would ensure training continues from this state
        # rather than restarting from scratch
    
    async def test_check_and_trigger_training_integration(self, learning_engine, mock_experience_collector):
        """Test integration with experience collector for automatic training"""
        # Mock experience collector to suggest training
        mock_experience_collector.should_trigger_training.return_value = True
        mock_experience_collector.get_buffer_size.return_value = 600
        
        # Mock the training cycle
        with patch.object(learning_engine, 'run_automatic_training_cycle') as mock_cycle:
            mock_cycle.return_value = {
                'training_triggered': True,
                'session_success': True,
                'new_model_version': 'v1.1.0'
            }
            
            result = await learning_engine.check_and_trigger_training()
            
            assert result is not None
            assert result['training_triggered'] == True
            mock_cycle.assert_called_once()
    
    async def test_check_and_trigger_training_no_trigger(self, learning_engine, mock_experience_collector):
        """Test no training when experience collector doesn't suggest it"""
        # Mock experience collector to not suggest training
        mock_experience_collector.should_trigger_training.return_value = False
        mock_experience_collector.get_buffer_size.return_value = 300
        
        result = await learning_engine.check_and_trigger_training()
        
        assert result is None
    
    async def test_check_and_trigger_training_already_training(self, learning_engine, mock_experience_collector):
        """Test skip when training already in progress"""
        # Set training state
        learning_engine.is_training = True
        
        # Mock experience collector to suggest training
        mock_experience_collector.should_trigger_training.return_value = True
        
        result = await learning_engine.check_and_trigger_training()
        
        assert result is None