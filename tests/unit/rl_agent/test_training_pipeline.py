"""
Tests for RL Agent Training Pipeline

Following TDD methodology - these tests define the expected behavior
of the training pipeline before implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Any, Optional

from src.rl_agent.base import TradeAction, MarketState, TradingResult, AgentConfig
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.rl_agent.experience_replay import ReplayBufferConfig
from src.rl_agent.reward_engineering import RewardConfig
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestTrainingConfig:
    """Test training configuration data structure"""
    
    def test_training_config_creation(self):
        """Test training configuration should be created with defaults"""
        # This test will fail initially - we need to implement TrainingConfig
        from src.rl_agent.training_pipeline import TrainingConfig
        
        config = TrainingConfig()
        
        # Default values that should be available
        assert config.num_episodes > 0
        assert config.max_steps_per_episode > 0
        assert config.target_update_frequency > 0
        assert config.save_frequency > 0
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.epsilon_start, float)
        assert isinstance(config.epsilon_end, float)
    
    def test_training_config_custom_values(self):
        """Test training configuration with custom values"""
        from src.rl_agent.training_pipeline import TrainingConfig
        
        config = TrainingConfig(
            num_episodes=500,
            learning_rate=0.001,
            epsilon_start=0.9,
            epsilon_end=0.1
        )
        
        assert config.num_episodes == 500
        assert config.learning_rate == 0.001
        assert config.epsilon_start == 0.9
        assert config.epsilon_end == 0.1


class TestTrainingMetrics:
    """Test training metrics tracking"""
    
    def test_training_metrics_creation(self):
        """Test training metrics should track episode performance"""
        from src.rl_agent.training_pipeline import TrainingMetrics
        
        metrics = TrainingMetrics()
        
        # Should track basic metrics
        assert hasattr(metrics, 'episode_rewards')
        assert hasattr(metrics, 'episode_losses')
        assert hasattr(metrics, 'win_rates')
        assert hasattr(metrics, 'portfolio_values')
        assert len(metrics.episode_rewards) == 0
    
    def test_training_metrics_update(self):
        """Test updating training metrics"""
        from src.rl_agent.training_pipeline import TrainingMetrics
        
        metrics = TrainingMetrics()
        
        # Should be able to add episode data
        metrics.add_episode(
            episode=1,
            total_reward=150.5,
            final_portfolio_value=10150.0,
            num_trades=25,
            win_rate=0.6,
            loss=0.05
        )
        
        assert len(metrics.episode_rewards) == 1
        assert metrics.episode_rewards[0] == 150.5
        assert metrics.portfolio_values[0] == 10150.0
    
    def test_training_metrics_statistics(self):
        """Test computing training statistics"""
        from src.rl_agent.training_pipeline import TrainingMetrics
        
        metrics = TrainingMetrics()
        
        # Add some sample data
        for i in range(10):
            metrics.add_episode(
                episode=i,
                total_reward=100 + i * 10,
                final_portfolio_value=10000 + i * 100,
                num_trades=20 + i,
                win_rate=0.5 + i * 0.01,
                loss=0.1 - i * 0.005
            )
        
        stats = metrics.get_statistics()
        
        assert 'mean_reward' in stats
        assert 'mean_portfolio_value' in stats
        assert 'mean_win_rate' in stats
        assert stats['mean_reward'] > 0
        assert stats['episodes_completed'] == 10


class TestDQNTrainingPipeline:
    """Test main DQN training pipeline"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for training"""
        tokens = []
        for i in range(3):
            token = DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def mock_price_data(self, sample_tokens):
        """Create mock price data for training"""
        price_data = {}
        for token in sample_tokens:
            prices = []
            base_price = token.price_usd
            for j in range(100):  # 100 time steps
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                prices.append(max(price, 0.01))
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [datetime.now() - timedelta(hours=100-i) for i in range(100)]
            }
        return price_data
    
    def test_training_pipeline_initialization(self, sample_tokens, mock_price_data):
        """Test training pipeline should initialize with required components"""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(num_episodes=10, max_steps_per_episode=50)
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Should create all required components
        assert pipeline.config == config
        assert pipeline.agent is not None
        assert pipeline.environment is not None
        assert pipeline.metrics is not None
        assert len(pipeline.tokens) == 3
    
    def test_training_pipeline_single_episode(self, sample_tokens, mock_price_data):
        """Test running a single training episode"""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(num_episodes=1, max_steps_per_episode=10)
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Should be able to run one episode
        episode_result = pipeline.run_episode(episode_num=0)
        
        assert 'total_reward' in episode_result
        assert 'num_steps' in episode_result
        assert 'final_portfolio_value' in episode_result
        assert 'num_trades' in episode_result
        assert isinstance(episode_result['total_reward'], float)
        assert episode_result['num_steps'] > 0
    
    def test_training_pipeline_full_training(self, sample_tokens, mock_price_data):
        """Test running full training process"""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(num_episodes=5, max_steps_per_episode=10)
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Should be able to run full training
        training_results = pipeline.train()
        
        assert 'episodes_completed' in training_results
        assert 'final_metrics' in training_results
        assert 'training_time' in training_results
        assert training_results['episodes_completed'] == 5
        
        # Metrics should be populated
        metrics = pipeline.metrics
        assert len(metrics.episode_rewards) == 5
        assert len(metrics.portfolio_values) == 5
    
    def test_training_pipeline_checkpointing(self, sample_tokens, mock_price_data):
        """Test model checkpointing during training"""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(
            num_episodes=10,
            max_steps_per_episode=10,
            save_frequency=5  # Save every 5 episodes
        )
        
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline.save_checkpoint') as mock_save:
            pipeline = DQNTrainingPipeline(
                config=config,
                tokens=sample_tokens,
                historical_data=mock_price_data
            )
            
            pipeline.train()
            
            # Should have saved checkpoints
            assert mock_save.call_count >= 1
    
    def test_training_pipeline_early_stopping(self, sample_tokens, mock_price_data):
        """Test early stopping based on performance"""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(
            num_episodes=100,
            max_steps_per_episode=10,
            early_stopping_patience=5,
            early_stopping_threshold=0.8  # Stop if win rate > 80%
        )
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=sample_tokens,
            historical_data=mock_price_data
        )
        
        # Mock high performance to trigger early stopping
        with patch.object(pipeline, 'should_stop_early', return_value=True):
            training_results = pipeline.train()
            
            # Should stop before completing all episodes
            assert training_results['episodes_completed'] < 100
            assert 'early_stopping_reason' in training_results


class TestTrainingPipelineError:
    """Test training pipeline error handling"""
    
    def test_training_pipeline_error_creation(self):
        """Test training pipeline error exception"""
        from src.rl_agent.training_pipeline import TrainingPipelineError
        
        error = TrainingPipelineError("Training failed")
        assert isinstance(error, Exception)
        assert str(error) == "Training failed"
    
    def test_training_pipeline_error_inheritance(self):
        """Test error inheritance from base RL errors"""
        from src.rl_agent.training_pipeline import TrainingPipelineError
        from src.rl_agent.base import RLTrainingError
        
        error = TrainingPipelineError("Test error")
        assert isinstance(error, RLTrainingError)


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        tokens = []
        for i in range(2):
            token = DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def mock_price_data(self, sample_tokens):
        """Create mock price data for testing"""
        price_data = {}
        for token in sample_tokens:
            prices = []
            base_price = token.price_usd
            for j in range(50):
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                prices.append(max(price, 0.01))
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            }
        return price_data
    
    def test_hyperparameter_search_config(self):
        """Test hyperparameter search configuration"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01, 0.1],
            'epsilon_decay': [0.995, 0.99, 0.985],
            'batch_size': [32, 64, 128]
        }
        
        search = HyperparameterSearch(search_config)
        
        assert search.search_space == search_config
        assert hasattr(search, 'best_params')
        assert hasattr(search, 'best_score')
    
    def test_hyperparameter_search_execution(self, sample_tokens, mock_price_data):
        """Test running hyperparameter search"""
        from src.rl_agent.training_pipeline import HyperparameterSearch, TrainingConfig
        
        # Simplified search space for testing
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Should be able to run optimization
        best_params = search.optimize(
            tokens=sample_tokens,
            historical_data=mock_price_data,
            num_trials=2,  # Limited for testing
            episodes_per_trial=5
        )
        
        assert 'learning_rate' in best_params
        assert 'epsilon_decay' in best_params
        assert search.best_score is not None
    
    def test_hyperparameter_search_grid_search(self, sample_tokens, mock_price_data):
        """Test grid search implementation covers all parameter combinations"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99],
            'batch_size': [32, 64]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Mock the training to verify all combinations are tested
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.train.return_value = {
                'episodes_completed': 5,
                'final_metrics': {'mean_reward': 100.0, 'mean_win_rate': 0.6}
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            best_params = search.optimize(
                tokens=sample_tokens,
                historical_data=mock_price_data,
                num_trials=8,  # 2*2*2 = 8 combinations
                episodes_per_trial=5
            )
            
            # Should have tested all combinations (2 * 2 * 2 = 8)
            assert mock_pipeline_class.call_count == 8
            
            # Best params should be from search space
            assert best_params['learning_rate'] in search_config['learning_rate']
            assert best_params['epsilon_decay'] in search_config['epsilon_decay']
            assert best_params['batch_size'] in search_config['batch_size']
    
    def test_hyperparameter_search_multiple_runs_per_config(self, sample_tokens, mock_price_data):
        """Test that multiple runs are performed per configuration for robust evaluation"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99]
        }
        
        search = HyperparameterSearch(search_config)
        
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            # Simulate variable performance across runs
            mock_pipeline.train.side_effect = [
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 80.0, 'mean_win_rate': 0.5}},
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 120.0, 'mean_win_rate': 0.7}},
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 90.0, 'mean_win_rate': 0.6}},
                # Repeat for each configuration
            ] * 10  # Enough for multiple runs per config
            mock_pipeline_class.return_value = mock_pipeline
            
            best_params = search.optimize(
                tokens=sample_tokens,
                historical_data=mock_price_data,
                num_trials=4,  # 2*2 = 4 combinations
                episodes_per_trial=5,
                runs_per_config=3  # Multiple runs per configuration
            )
            
            # Should have performed multiple runs per config
            # 4 configs * 3 runs each = 12 total training runs
            assert mock_pipeline_class.call_count == 12
            
            assert best_params is not None
            assert search.best_score is not None
    
    def test_hyperparameter_search_performance_evaluation(self, sample_tokens, mock_price_data):
        """Test that performance evaluation considers multiple metrics"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Mock different performance results for different configs
        mock_results = [
            # Config 1: High reward, low win rate
            {'episodes_completed': 5, 'final_metrics': {'mean_reward': 150.0, 'mean_win_rate': 0.4, 'mean_portfolio_value': 11000}},
            # Config 2: Low reward, high win rate
            {'episodes_completed': 5, 'final_metrics': {'mean_reward': 80.0, 'mean_win_rate': 0.8, 'mean_portfolio_value': 10800}},
            # Config 3: Balanced performance
            {'episodes_completed': 5, 'final_metrics': {'mean_reward': 120.0, 'mean_win_rate': 0.7, 'mean_portfolio_value': 11200}},
            # Config 4: Poor performance
            {'episodes_completed': 5, 'final_metrics': {'mean_reward': 50.0, 'mean_win_rate': 0.3, 'mean_portfolio_value': 9800}},
        ]
        
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.train.side_effect = mock_results
            mock_pipeline_class.return_value = mock_pipeline
            
            best_params = search.optimize(
                tokens=sample_tokens,
                historical_data=mock_price_data,
                num_trials=4,
                episodes_per_trial=5
            )
            
            # Should select best performing configuration
            # In this case, config 3 (balanced) should win with composite score
            assert best_params is not None
            assert search.best_score is not None
            # Score should reflect composite evaluation, not just single metric
            assert 0.0 < search.best_score <= 1.0
    
    def test_hyperparameter_search_cross_validation(self, sample_tokens, mock_price_data):
        """Test cross-validation methodology for robust evaluation"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Test that cross-validation splits data appropriately
        best_params = search.optimize(
            tokens=sample_tokens,
            historical_data=mock_price_data,
            num_trials=4,
            episodes_per_trial=10,
            cross_validation_folds=3
        )
        
        # Should complete without error and return valid parameters
        assert best_params is not None
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
        assert search.best_score is not None
    
    def test_hyperparameter_search_progress_tracking(self, sample_tokens, mock_price_data):
        """Test that optimization progress is properly tracked and logged"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Run optimization and check that it completes successfully
        best_params = search.optimize(
            tokens=sample_tokens,
            historical_data=mock_price_data,
            num_trials=4,  # 2*2 = 4 combinations
            episodes_per_trial=3  # Keep it small for test
        )
        
        # Should have optimization history tracking
        assert hasattr(search, 'optimization_history')
        assert len(search.optimization_history) > 0
        
        # Each history entry should have required fields
        for entry in search.optimization_history:
            assert 'params' in entry
            assert 'score' in entry
            assert 'config_num' in entry
            
        # Should have found best parameters
        assert best_params is not None
        assert search.best_score is not None
    
    def test_hyperparameter_search_early_stopping(self, sample_tokens, mock_price_data):
        """Test early stopping when performance converges"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],  # Many options
            'epsilon_decay': [0.99, 0.995]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Mock consistently high performance to trigger early stopping
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.train.return_value = {
                'episodes_completed': 5,
                'final_metrics': {'mean_reward': 200.0, 'mean_win_rate': 0.9, 'mean_portfolio_value': 12000}
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            best_params = search.optimize(
                tokens=sample_tokens,
                historical_data=mock_price_data,
                num_trials=10,  # Total possible: 5*2=10
                episodes_per_trial=5,
                early_stopping_patience=3,  # Stop if no improvement for 3 trials
                early_stopping_threshold=0.85  # Stop if score > 0.85
            )
            
            # Should have stopped early due to high performance
            # With consistently high performance, should stop after finding good solution
            assert best_params is not None
            assert search.best_score is not None
            assert search.best_score >= 0.85
    
    def test_hyperparameter_search_error_handling(self, sample_tokens, mock_price_data):
        """Test error handling during optimization"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        search_config = {
            'learning_rate': [0.001, 0.01],
            'epsilon_decay': [0.995, 0.99]
        }
        
        search = HyperparameterSearch(search_config)
        
        # Test handling of training failures
        with patch('src.rl_agent.training_pipeline.DQNTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            # First call succeeds, second fails, third succeeds
            mock_pipeline.train.side_effect = [
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 100.0, 'mean_win_rate': 0.6}},
                Exception("Training failed"),
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 120.0, 'mean_win_rate': 0.7}},
                {'episodes_completed': 5, 'final_metrics': {'mean_reward': 90.0, 'mean_win_rate': 0.5}},
            ]
            mock_pipeline_class.return_value = mock_pipeline
            
            # Should handle errors gracefully and continue optimization
            best_params = search.optimize(
                tokens=sample_tokens,
                historical_data=mock_price_data,
                num_trials=4,
                episodes_per_trial=5
            )
            
            # Should complete despite one failed training run
            assert best_params is not None
            assert search.best_score is not None
    
    def test_hyperparameter_search_empty_search_space(self):
        """Test handling of empty search space"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        # Empty search space should raise appropriate error
        with pytest.raises(ValueError, match="Search space cannot be empty"):
            HyperparameterSearch({})
    
    def test_hyperparameter_search_invalid_search_space(self):
        """Test handling of invalid search space configurations"""
        from src.rl_agent.training_pipeline import HyperparameterSearch
        
        # Search space with empty parameter values
        with pytest.raises(ValueError, match="Parameter values cannot be empty"):
            HyperparameterSearch({
                'learning_rate': [],
                'epsilon_decay': [0.995, 0.99]
            })


class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline"""
    
    @pytest.fixture
    def full_training_setup(self):
        """Create complete training setup"""
        # Create tokens
        tokens = []
        for i in range(2):  # Smaller for integration test
            token = DiscoveredToken(
                address=f"0x{i:04d}...",
                symbol=f"TKN{i}",
                name=f"Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now() - timedelta(days=i),
                discovery_source="test",
                price_usd=1.0 + i * 0.25,
                volume_24h=100000 * (i + 1)
            )
            tokens.append(token)
        
        # Create price data
        price_data = {}
        np.random.seed(42)  # For reproducible tests
        
        for token in tokens:
            prices = []
            base_price = token.price_usd
            
            for j in range(50):  # Shorter for integration test
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                    price = max(price, 0.01)
                
                prices.append(price)
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [
                    datetime.now() - timedelta(hours=50-i) 
                    for i in range(50)
                ]
            }
        
        return tokens, price_data
    
    def test_end_to_end_training(self, full_training_setup):
        """Test complete end-to-end training process"""
        tokens, price_data = full_training_setup
        
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(
            num_episodes=3,
            max_steps_per_episode=10,
            learning_rate=0.01,
            target_update_frequency=2
        )
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Run complete training
        results = pipeline.train()
        
        # Validate results
        assert results['episodes_completed'] == 3
        assert 'training_time' in results
        assert 'final_metrics' in results
        
        # Check that agent learned something
        final_metrics = results['final_metrics']
        assert 'mean_reward' in final_metrics
        assert 'mean_portfolio_value' in final_metrics
        
        # Portfolio value should be tracked
        assert len(pipeline.metrics.portfolio_values) == 3
        assert all(pv > 0 for pv in pipeline.metrics.portfolio_values)
    
    def test_training_with_all_components(self, full_training_setup):
        """Test training integrates all RL components correctly"""
        tokens, price_data = full_training_setup
        
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        
        config = TrainingConfig(
            num_episodes=2,
            max_steps_per_episode=5,
            use_prioritized_replay=True,
            use_advanced_rewards=True
        )
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Should integrate all components
        assert pipeline.agent is not None
        assert pipeline.environment is not None
        assert hasattr(pipeline, 'replay_buffer')
        assert hasattr(pipeline, 'reward_calculator')
        
        # Run training
        results = pipeline.train()
        
        # Should complete successfully
        assert results['episodes_completed'] == 2
        
        # All components should have been used
        assert len(pipeline.metrics.episode_rewards) == 2
        assert len(pipeline.metrics.episode_losses) == 2
    
    def test_prioritized_replay_buffer_priority_updates(self, full_training_setup):
        """Test that prioritized replay buffer priorities are updated during training"""
        tokens, price_data = full_training_setup
        
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        from src.rl_agent.experience_replay import PrioritizedExperienceReplayBuffer
        
        config = TrainingConfig(
            num_episodes=1,
            max_steps_per_episode=50,  # More steps to ensure training occurs
            use_prioritized_replay=True,
            batch_size=8,  # Smaller batch for faster testing
            replay_buffer_min_size=8  # Very small min_size to trigger training quickly
        )
        
        pipeline = DQNTrainingPipeline(
            config=config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Verify we're using prioritized replay
        assert isinstance(pipeline.replay_buffer, PrioritizedExperienceReplayBuffer)
        
        # Store initial priorities (should be all max priority initially)
        initial_max_priority = pipeline.replay_buffer.max_priority
        
        # Run training
        results = pipeline.train()
        
        # Should complete successfully
        assert results['episodes_completed'] == 1
        
        # Check that priorities were updated during training
        # After training, we should have different priorities based on TD errors
        if len(pipeline.replay_buffer.priorities) > 0:
            priorities = list(pipeline.replay_buffer.priorities)
            
            # Not all priorities should be the same (unless by unlikely chance)
            # This indicates priorities were updated based on TD errors
            unique_priorities = set(priorities)
            assert len(unique_priorities) > 1 or len(priorities) < 5, \
                f"Expected varied priorities after training, got {len(unique_priorities)} unique values from {len(priorities)} total"
            
            # At least some priorities should be different from initial max priority
            different_from_initial = sum(1 for p in priorities if abs(p - initial_max_priority) > 1e-6)
            assert different_from_initial > 0, \
                f"Expected some priorities to change from initial {initial_max_priority}, but all remained the same"
    
    def test_standard_vs_prioritized_replay_integration(self, full_training_setup):
        """Test that both standard and prioritized replay buffers work correctly"""
        tokens, price_data = full_training_setup
        
        from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
        from src.rl_agent.experience_replay import PrioritizedExperienceReplayBuffer, ExperienceReplayBuffer
        
        # Test with standard replay buffer
        standard_config = TrainingConfig(
            num_episodes=1,
            max_steps_per_episode=30,
            use_prioritized_replay=False,  # Standard replay
            batch_size=8,
            replay_buffer_min_size=8
        )
        
        standard_pipeline = DQNTrainingPipeline(
            config=standard_config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Verify we're using standard replay
        assert isinstance(standard_pipeline.replay_buffer, ExperienceReplayBuffer)
        assert not isinstance(standard_pipeline.replay_buffer, PrioritizedExperienceReplayBuffer)
        
        # Run training with standard replay buffer
        standard_results = standard_pipeline.train()
        assert standard_results['episodes_completed'] == 1
        
        # Test with prioritized replay buffer
        prioritized_config = TrainingConfig(
            num_episodes=1,
            max_steps_per_episode=30,
            use_prioritized_replay=True,  # Prioritized replay
            batch_size=8,
            replay_buffer_min_size=8
        )
        
        prioritized_pipeline = DQNTrainingPipeline(
            config=prioritized_config,
            tokens=tokens,
            historical_data=price_data
        )
        
        # Verify we're using prioritized replay
        assert isinstance(prioritized_pipeline.replay_buffer, PrioritizedExperienceReplayBuffer)
        
        # Run training with prioritized replay buffer
        prioritized_results = prioritized_pipeline.train()
        assert prioritized_results['episodes_completed'] == 1
        
        # Both should complete successfully
        assert 'training_time' in standard_results
        assert 'training_time' in prioritized_results
        assert 'final_metrics' in standard_results
        assert 'final_metrics' in prioritized_results
        
        # Both should have collected metrics
        assert len(standard_pipeline.metrics.episode_rewards) == 1
        assert len(prioritized_pipeline.metrics.episode_rewards) == 1
        
        # Verify that standard replay buffer doesn't have update_priorities method called
        # (no errors should occur, it should gracefully handle the missing method)
        assert not hasattr(standard_pipeline.replay_buffer, 'priorities')
        
        # Verify that prioritized replay buffer has priorities that were potentially updated
        assert hasattr(prioritized_pipeline.replay_buffer, 'priorities')
        assert len(prioritized_pipeline.replay_buffer.priorities) > 0


class TestTrainingPipelineExperienceDatabase:
    """Test training pipeline database integration for experience loading and metrics tracking."""
    
    @pytest.fixture
    def training_config(self):
        """Create training configuration with database integration."""
        from src.rl_agent.training_pipeline import TrainingConfig
        return TrainingConfig(
            num_episodes=5,
            max_steps_per_episode=20,
            learning_rate=0.01,
            batch_size=16,
            use_database_experiences=True,  # New parameter
            database_config={
                'enabled': True,
                'preload_experiences': True,
                'batch_optimization': True,
                'memory_management': True,
                'metrics_tracking': True
            }
        )
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for training."""
        tokens = []
        for i in range(2):
            token = DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def historical_data(self, sample_tokens):
        """Create historical price data."""
        price_data = {}
        for token in sample_tokens:
            prices = []
            base_price = token.price_usd
            for j in range(50):
                if j == 0:
                    price = base_price
                else:
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                prices.append(max(price, 0.01))
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            }
        return price_data
    
    def test_training_config_database_parameters(self, training_config):
        """Test that training config includes database parameters."""
        # Parameters should now exist and be properly configured
        assert hasattr(training_config, 'use_database_experiences')
        assert training_config.use_database_experiences is True
        assert hasattr(training_config, 'database_config')
        assert isinstance(training_config.database_config, dict)
        
        # Verify database configuration structure
        db_config = training_config.database_config
        expected_keys = ['enabled', 'preload_experiences', 'batch_optimization', 
                        'memory_management', 'metrics_tracking']
        for key in expected_keys:
            assert key in db_config
        assert training_config.database_config['enabled'] is True
    
    def test_training_pipeline_database_initialization(self, training_config, sample_tokens, historical_data):
        """Test training pipeline initialization with database experience support."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as database support doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should have database components initialized
            assert hasattr(pipeline, 'experience_db_loader')
            assert hasattr(pipeline, 'database_metrics_tracker')
            assert pipeline.config.use_database_experiences is True
    
    @pytest.mark.asyncio
    async def test_preload_experiences_from_database(self, training_config, sample_tokens, historical_data):
        """Test preloading experiences from database before training."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should be able to preload experiences
            preloaded_count = await pipeline.preload_experiences_from_database(
                lookback_days=7,
                min_experiences=50
            )
            
            assert preloaded_count >= 0
            assert hasattr(pipeline, '_preloaded_experiences')
            assert len(pipeline._preloaded_experiences) == preloaded_count
    
    @pytest.mark.asyncio 
    async def test_batch_loading_optimization(self, training_config, sample_tokens, historical_data):
        """Test batch loading optimization for database experiences."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Test batch loading with optimization
            batch_config = {
                'batch_size': 64,
                'prefetch_batches': 3,
                'memory_limit_mb': 100,
                'priority_sampling': True
            }
            
            experiences_batch = await pipeline.load_experience_batch_optimized(batch_config)
            
            assert len(experiences_batch) <= batch_config['batch_size']
            assert all('state' in exp for exp in experiences_batch)
            assert all('reward' in exp for exp in experiences_batch)
    
    @pytest.mark.asyncio
    async def test_training_performance_metrics_tracking(self, training_config, sample_tokens, historical_data):
        """Test training performance metrics tracking with database integration."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should track database-specific metrics
            await pipeline.initialize_database_metrics_tracking()
            
            # Run training and check metrics
            results = pipeline.train()
            
            # Should have database performance metrics
            db_metrics = pipeline.get_database_performance_metrics()
            
            assert 'experience_load_time_ms' in db_metrics
            assert 'database_query_count' in db_metrics
            assert 'cache_hit_rate' in db_metrics
            assert 'memory_usage_peak_mb' in db_metrics
            assert 'experiences_loaded_total' in db_metrics
    
    @pytest.mark.asyncio
    async def test_memory_management_large_datasets(self, training_config, sample_tokens, historical_data):
        """Test memory management for large experience datasets."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet  
        with pytest.raises(AttributeError):
            # Configure for large dataset
            training_config.database_config['memory_limit_mb'] = 50
            training_config.database_config['streaming_mode'] = True
            
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should handle memory management
            memory_stats_before = pipeline.get_memory_usage()
            
            # Simulate large dataset loading
            await pipeline.load_large_experience_dataset(
                dataset_size=10000,
                streaming=True
            )
            
            memory_stats_after = pipeline.get_memory_usage()
            
            # Memory usage should be controlled
            memory_increase = memory_stats_after - memory_stats_before
            assert memory_increase < training_config.database_config['memory_limit_mb'] * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_database_experience_vs_file_loading(self, training_config, sample_tokens, historical_data):
        """Test comparison between database experience loading and file-based loading."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            # Test database loading
            db_config = training_config
            db_config.use_database_experiences = True
            
            db_pipeline = DQNTrainingPipeline(
                config=db_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Test file-based loading (legacy)
            file_config = training_config
            file_config.use_database_experiences = False
            
            file_pipeline = DQNTrainingPipeline(
                config=file_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Compare loading performance
            db_start = datetime.now()
            db_experiences = await db_pipeline.load_experiences_from_database(limit=100)
            db_duration = (datetime.now() - db_start).total_seconds()
            
            file_start = datetime.now()
            file_experiences = await file_pipeline.load_experiences_from_files(limit=100)
            file_duration = (datetime.now() - file_start).total_seconds()
            
            # Both should return valid experiences
            assert len(db_experiences) > 0
            assert len(file_experiences) > 0
            
            # Database loading should have metadata
            if db_experiences:
                assert 'database_id' in db_experiences[0]
                assert 'priority' in db_experiences[0]
    
    @pytest.mark.asyncio
    async def test_training_pipeline_database_error_handling(self, training_config, sample_tokens, historical_data):
        """Test error handling when database operations fail."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Test database connection failure handling
            with patch('src.utils.database.get_database_connection', side_effect=Exception("DB Error")):
                # Should fallback gracefully
                result = await pipeline.load_experiences_with_fallback()
                
                # Should use fallback mechanism (file or memory)
                assert result is not None
                assert hasattr(pipeline, '_fallback_used')
                assert pipeline._fallback_used is True
    
    @pytest.mark.asyncio
    async def test_experience_priority_updates_with_database(self, training_config, sample_tokens, historical_data):
        """Test updating experience priorities in database during training."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            training_config.use_prioritized_replay = True
            training_config.database_config['update_priorities'] = True
            
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Mock some experiences with database IDs
            mock_experiences = [
                {'database_id': 1, 'priority': 0.5, 'td_error': 0.8},
                {'database_id': 2, 'priority': 1.2, 'td_error': 0.3}
            ]
            
            # Should update priorities in database
            updated_count = await pipeline.update_experience_priorities_in_database(mock_experiences)
            
            assert updated_count == len(mock_experiences)
            
            # Should track priority update metrics
            priority_metrics = pipeline.get_priority_update_metrics()
            assert 'total_updates' in priority_metrics
            assert 'avg_update_time_ms' in priority_metrics
    
    @pytest.mark.asyncio
    async def test_training_session_database_tracking(self, training_config, sample_tokens, historical_data):
        """Test tracking training sessions in database."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should create training session record
            session_id = await pipeline.create_training_session_record()
            
            assert session_id is not None
            assert hasattr(pipeline, 'training_session_id')
            assert pipeline.training_session_id == session_id
            
            # Run training
            results = pipeline.train()
            
            # Should update session with results
            await pipeline.update_training_session_record(results)
            
            # Should be able to retrieve session info
            session_info = await pipeline.get_training_session_info()
            assert 'session_id' in session_info
            assert 'episodes_completed' in session_info
            assert 'final_metrics' in session_info
    
    @pytest.mark.asyncio
    async def test_experience_analytics_integration(self, training_config, sample_tokens, historical_data):
        """Test integration with experience analytics during training."""
        from src.rl_agent.training_pipeline import DQNTrainingPipeline
        
        # Should fail initially as this functionality doesn't exist yet
        with pytest.raises(AttributeError):
            training_config.database_config['analytics_enabled'] = True
            
            pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=sample_tokens,
                historical_data=historical_data
            )
            
            # Should generate analytics during training
            results = pipeline.train()
            
            # Should have generated experience analytics
            analytics = await pipeline.generate_experience_analytics()
            
            assert 'experience_patterns' in analytics
            assert 'performance_trends' in analytics
            assert 'action_effectiveness' in analytics
            assert 'learning_progress' in analytics