"""
RL Agent Training Pipeline

Implements automated training orchestration for DQN trading agents
with hyperparameter optimization and performance monitoring.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

from .base import RLTrainingError, AgentConfig, TradeAction
from .dqn_agent import DQNTradingAgent
from .trading_environment import TradingEnvironment, EnvironmentConfig
from .experience_replay import create_replay_buffer, ReplayBufferConfig
from .reward_engineering import create_reward_calculator, RewardConfig

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for RL training pipeline"""
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Network updates
    target_update_frequency: int = 100
    batch_size: int = 32
    
    # Checkpointing
    save_frequency: int = 50
    model_save_path: str = "models/dqn_checkpoint.pth"
    
    # Early stopping
    early_stopping_patience: int = 100
    early_stopping_threshold: float = 0.75
    
    # Component options
    use_prioritized_replay: bool = True
    use_advanced_rewards: bool = True
    
    # Replay buffer options
    replay_buffer_min_size: int = 100  # Minimum experiences before training
    
    # ML-RL integration options
    use_ml_features: bool = False
    ml_weight: float = 0.3  # Weight for ML predictions in decisions


class TrainingPipelineError(RLTrainingError):
    """Raised when training pipeline operations fail"""
    pass


class TrainingMetrics:
    """Tracks training metrics and performance"""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.win_rates: List[float] = []
        self.portfolio_values: List[float] = []
        self.num_trades: List[int] = []
        self.episodes_completed: int = 0
    
    def add_episode(self, episode: int, total_reward: float, 
                   final_portfolio_value: float, num_trades: int,
                   win_rate: float, loss: float = 0.0):
        """Add episode results to metrics"""
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(loss)
        self.win_rates.append(win_rate)
        self.portfolio_values.append(final_portfolio_value)
        self.num_trades.append(num_trades)
        self.episodes_completed = episode + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        if not self.episode_rewards:
            return {
                'episodes_completed': 0,
                'mean_reward': 0.0,
                'mean_portfolio_value': 0.0,
                'mean_win_rate': 0.0
            }
        
        return {
            'episodes_completed': self.episodes_completed,
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'mean_portfolio_value': float(np.mean(self.portfolio_values)),
            'mean_win_rate': float(np.mean(self.win_rates)),
            'mean_loss': float(np.mean(self.episode_losses)) if self.episode_losses else 0.0,
            'total_trades': sum(self.num_trades)
        }


class DQNTrainingPipeline:
    """Main DQN training pipeline orchestrator"""
    
    def __init__(self, config: TrainingConfig, tokens, historical_data):
        self.config = config
        self.tokens = tokens
        self.historical_data = historical_data
        self.metrics = TrainingMetrics()
        
        # Initialize all required components
        self._initialize_components()
        
        self.logger = structlog.get_logger().bind(component="DQNTrainingPipeline")
        
        self.logger.info("Training pipeline initialized",
                        num_episodes=config.num_episodes,
                        max_steps=config.max_steps_per_episode,
                        learning_rate=config.learning_rate)
    
    def _initialize_components(self):
        """Initialize all RL components"""
        # Agent configuration
        agent_config = AgentConfig(
            learning_rate=self.config.learning_rate,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=self.config.epsilon_decay,
            batch_size=self.config.batch_size
        )
        
        # Create DQN agent
        self.agent = DQNTradingAgent(agent_config)
        
        # Environment configuration
        env_config = EnvironmentConfig(
            max_episode_steps=self.config.max_steps_per_episode
        )
        
        # Create trading environment
        self.environment = TradingEnvironment(env_config, self.tokens, self.historical_data)
        
        # Replay buffer configuration
        replay_config = ReplayBufferConfig(
            batch_size=self.config.batch_size,
            prioritized=self.config.use_prioritized_replay,
            min_size=self.config.replay_buffer_min_size
        )
        
        # Create experience replay buffer
        self.replay_buffer = create_replay_buffer(replay_config)
        
        # Reward calculator configuration
        reward_config = RewardConfig()
        
        # Create advanced reward calculator
        self.reward_calculator = create_reward_calculator(reward_config) if self.config.use_advanced_rewards else None
    
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode with complete RL training loop"""
        episode_start_time = time.time()
        
        # Reset environment and get initial state
        state = self.environment.reset()
        current_state = state.to_vector()
        
        # Episode tracking
        total_reward = 0.0
        num_steps = 0
        num_trades = 0
        successful_trades = 0
        episode_loss = 0.0
        
        # Get initial portfolio value
        initial_portfolio_value = self.environment.portfolio.total_value
        
        self.logger.debug("Starting episode", 
                         episode=episode_num,
                         initial_portfolio_value=initial_portfolio_value)
        
        for step in range(self.config.max_steps_per_episode):
            # Agent predicts action with exploration
            try:
                # Use asyncio to run the async predict_action method
                action, confidence = asyncio.run(self.agent.predict_action(state))
                action_idx = list(TradeAction).index(action)
                
                # Select a token for trading (simple strategy: cycle through tokens)
                token_idx = step % len(self.tokens)
                selected_token = self.tokens[token_idx]
                
                # Calculate position size based on action strength
                if action in [TradeAction.STRONG_BUY, TradeAction.STRONG_SELL]:
                    position_size = 0.2  # 20% of portfolio for strong actions
                elif action in [TradeAction.BUY, TradeAction.SELL]:
                    position_size = 0.1  # 10% of portfolio for regular actions
                else:
                    position_size = 0.0  # No trade for HOLD
                
                # Execute action in environment
                next_state, reward, done, info = self.environment.step(
                    action=action,
                    token=selected_token,
                    position_size=position_size
                )
                
                next_state_vector = next_state.to_vector()
                
                # Track trading statistics
                if action != TradeAction.HOLD:
                    num_trades += 1
                    if info.get('trade_success', False):
                        successful_trades += 1
                
                # Store experience in replay buffer
                from .experience_replay import Experience
                experience = Experience(
                    state=current_state,
                    action=action_idx,
                    reward=reward,
                    next_state=next_state_vector,
                    done=done
                )
                self.replay_buffer.add(experience)
                
                # Train agent if we have enough experiences
                if (len(self.replay_buffer.buffer) >= self.replay_buffer.config.min_size and
                    step % 4 == 0):  # Train every 4 steps
                    
                    try:
                        batch_experiences = self.replay_buffer.sample()
                        loss_info = asyncio.run(self.agent.train_step(batch_experiences))
                        episode_loss += loss_info.get('loss', 0.0)
                        
                        # Update priorities in prioritized replay buffer based on TD errors
                        if ('td_errors' in loss_info and 'indices' in loss_info and 
                            hasattr(self.replay_buffer, 'update_priorities')):
                            
                            td_errors = loss_info['td_errors']
                            indices = loss_info['indices']
                            
                            # Convert to numpy array for update_priorities
                            import numpy as np
                            td_errors_array = np.array(td_errors, dtype=np.float64)
                            
                            # Update buffer priorities with TD errors
                            self.replay_buffer.update_priorities(indices, td_errors_array)
                            
                            self.logger.debug("Updated replay buffer priorities",
                                            num_updates=len(indices),
                                            avg_td_error=float(np.mean(td_errors_array)),
                                            max_td_error=float(np.max(td_errors_array)),
                                            min_td_error=float(np.min(td_errors_array)))
                            
                    except Exception as e:
                        self.logger.warning("Training step failed", error=str(e))
                
                # Update target network periodically
                if step % self.config.target_update_frequency == 0:
                    self.agent.update_target_network()
                
                # Update tracking
                total_reward += reward
                num_steps = step + 1
                current_state = next_state_vector
                state = next_state
                
                # Check if episode should end
                if done:
                    self.logger.debug("Episode terminated early", 
                                    step=step, 
                                    reason=info.get('termination_reason', 'unknown'))
                    break
                    
            except Exception as e:
                self.logger.error("Error during episode step", 
                                step=step, 
                                episode=episode_num, 
                                error=str(e))
                # For debugging, let's not continue but break the loop with minimal tracking
                num_steps = step + 1
                break
        
        # Calculate final metrics
        final_portfolio_value = self.environment.portfolio.total_value
        win_rate = successful_trades / max(num_trades, 1)  # Avoid division by zero
        
        episode_time = time.time() - episode_start_time
        
        self.logger.info("Episode completed",
                        episode=episode_num,
                        steps=num_steps,
                        total_reward=total_reward,
                        trades=num_trades,
                        win_rate=win_rate,
                        portfolio_value=final_portfolio_value,
                        duration_seconds=episode_time)
        
        return {
            'total_reward': float(total_reward),
            'num_steps': num_steps,
            'final_portfolio_value': float(final_portfolio_value),
            'num_trades': num_trades,
            'win_rate': float(win_rate),
            'loss': float(episode_loss / max(num_steps // 4, 1))  # Average loss per training step
        }
    
    def train(self) -> Dict[str, Any]:
        """Run complete training process"""
        start_time = time.time()
        
        self.logger.info("Starting training",
                        episodes=self.config.num_episodes)
        
        for episode in range(self.config.num_episodes):
            # Run episode
            episode_result = self.run_episode(episode)
            
            # Update metrics
            self.metrics.add_episode(
                episode=episode,
                total_reward=episode_result['total_reward'],
                final_portfolio_value=episode_result['final_portfolio_value'],
                num_trades=episode_result['num_trades'],
                win_rate=episode_result['win_rate'],
                loss=episode_result.get('loss', 0.0)
            )
            
            # Check early stopping
            if self.should_stop_early():
                self.logger.info("Early stopping triggered", episode=episode)
                return {
                    'episodes_completed': self.metrics.episodes_completed,
                    'training_time': time.time() - start_time,
                    'final_metrics': self.metrics.get_statistics(),
                    'early_stopping_reason': 'performance_threshold_reached'
                }
            
            # Save checkpoint
            if (episode + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(episode)
        
        training_time = time.time() - start_time
        
        return {
            'episodes_completed': self.metrics.episodes_completed,
            'training_time': training_time,
            'final_metrics': self.metrics.get_statistics()
        }
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met with performance-based stopping"""
        
        # Need at least patience episodes to check for early stopping
        if self.metrics.episodes_completed < self.config.early_stopping_patience:
            return False
        
        # Get recent performance metrics
        recent_win_rates = self.metrics.win_rates[-self.config.early_stopping_patience:]
        recent_rewards = self.metrics.episode_rewards[-self.config.early_stopping_patience:]
        recent_portfolio_values = self.metrics.portfolio_values[-self.config.early_stopping_patience:]
        
        # Check win rate threshold
        mean_recent_win_rate = np.mean(recent_win_rates)
        if mean_recent_win_rate >= self.config.early_stopping_threshold:
            self.logger.info("Early stopping: win rate threshold reached",
                           mean_win_rate=mean_recent_win_rate,
                           threshold=self.config.early_stopping_threshold)
            return True
        
        # Check for performance plateau (no improvement in recent episodes)
        if len(recent_rewards) >= self.config.early_stopping_patience:
            # Split recent rewards into two halves and compare
            half_point = self.config.early_stopping_patience // 2
            first_half_rewards = recent_rewards[:half_point]
            second_half_rewards = recent_rewards[half_point:]
            
            first_half_mean = np.mean(first_half_rewards)
            second_half_mean = np.mean(second_half_rewards)
            
            # Check if there's minimal improvement (less than 1% relative improvement)
            if first_half_mean > 0:
                improvement_ratio = (second_half_mean - first_half_mean) / abs(first_half_mean)
                if improvement_ratio < 0.01:  # Less than 1% improvement
                    self.logger.info("Early stopping: performance plateau detected",
                                   first_half_mean=first_half_mean,
                                   second_half_mean=second_half_mean,
                                   improvement_ratio=improvement_ratio)
                    return True
        
        # Check for consistent portfolio growth
        if len(recent_portfolio_values) >= self.config.early_stopping_patience:
            portfolio_growth = (recent_portfolio_values[-1] - recent_portfolio_values[0]) / recent_portfolio_values[0]
            
            # If portfolio has grown significantly (>50%) with good win rate, consider stopping
            if portfolio_growth > 0.5 and mean_recent_win_rate > 0.6:
                self.logger.info("Early stopping: significant portfolio growth achieved",
                               portfolio_growth=portfolio_growth,
                               win_rate=mean_recent_win_rate)
                return True
        
        # Check for excessive losses (safety mechanism)
        recent_portfolio_min = np.min(recent_portfolio_values)
        recent_portfolio_max = np.max(recent_portfolio_values)
        if recent_portfolio_max > 0:
            recent_drawdown = (recent_portfolio_max - recent_portfolio_min) / recent_portfolio_max
            if recent_drawdown > 0.3:  # More than 30% drawdown in recent episodes
                self.logger.warning("Early stopping: excessive recent drawdown detected",
                                  recent_drawdown=recent_drawdown)
                return True
        
        return False
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint with proper state management"""
        import os
        import torch
        import json
        from pathlib import Path
        
        try:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = Path(self.config.model_save_path).parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create versioned checkpoint filename
            checkpoint_name = f"dqn_checkpoint_episode_{episode}.pth"
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            # Prepare checkpoint data
            checkpoint_data = {
                'episode': episode,
                'timestamp': datetime.now().isoformat(),
                
                # Model state
                'model_state_dict': self.agent.q_network.state_dict(),
                'target_model_state_dict': self.agent.target_network.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                
                # Training configuration
                'config': {
                    'num_episodes': self.config.num_episodes,
                    'max_steps_per_episode': self.config.max_steps_per_episode,
                    'learning_rate': self.config.learning_rate,
                    'epsilon_start': self.config.epsilon_start,
                    'epsilon_end': self.config.epsilon_end,
                    'epsilon_decay': self.config.epsilon_decay,
                    'target_update_frequency': self.config.target_update_frequency,
                    'batch_size': self.config.batch_size,
                    'early_stopping_patience': self.config.early_stopping_patience,
                    'early_stopping_threshold': self.config.early_stopping_threshold
                },
                
                # Agent configuration
                'agent_config': {
                    'epsilon': self.agent.epsilon,
                    'learning_rate': self.agent.config.learning_rate,
                    'hidden_size': self.agent.config.hidden_size,
                    'num_layers': self.agent.config.num_layers,
                    'dropout': self.agent.config.dropout
                },
                
                # Training metrics
                'metrics': {
                    'episodes_completed': self.metrics.episodes_completed,
                    'episode_rewards': self.metrics.episode_rewards,
                    'episode_losses': self.metrics.episode_losses,
                    'win_rates': self.metrics.win_rates,
                    'portfolio_values': self.metrics.portfolio_values,
                    'num_trades': self.metrics.num_trades
                },
                
                # Training statistics
                'statistics': self.metrics.get_statistics(),
                
                # Replay buffer size (don't save the actual buffer due to memory)
                'replay_buffer_size': len(self.replay_buffer.buffer)
            }
            
            # Save the checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save human-readable metrics to JSON
            metrics_file = checkpoint_dir / f"metrics_episode_{episode}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'episode': episode,
                    'timestamp': datetime.now().isoformat(),
                    'statistics': checkpoint_data['statistics'],
                    'recent_performance': {
                        'last_10_rewards': self.metrics.episode_rewards[-10:] if len(self.metrics.episode_rewards) >= 10 else self.metrics.episode_rewards,
                        'last_10_win_rates': self.metrics.win_rates[-10:] if len(self.metrics.win_rates) >= 10 else self.metrics.win_rates,
                        'last_10_portfolio_values': self.metrics.portfolio_values[-10:] if len(self.metrics.portfolio_values) >= 10 else self.metrics.portfolio_values
                    }
                }, f, indent=2)
            
            # Update the main checkpoint file to point to latest
            main_checkpoint_path = Path(self.config.model_save_path)
            
            # Copy latest checkpoint to main path for easy loading
            if main_checkpoint_path != checkpoint_path:
                import shutil
                shutil.copy2(checkpoint_path, main_checkpoint_path)
            
            # Clean up old checkpoints (keep only last 5)
            checkpoint_files = sorted(
                checkpoint_dir.glob("dqn_checkpoint_episode_*.pth"),
                key=lambda x: int(x.stem.split('_')[-1])
            )
            
            # Keep only the 5 most recent checkpoints
            for old_checkpoint in checkpoint_files[:-5]:
                try:
                    old_checkpoint.unlink()
                    # Also remove corresponding metrics file
                    old_episode = old_checkpoint.stem.split('_')[-1]
                    old_metrics_file = checkpoint_dir / f"metrics_episode_{old_episode}.json"
                    if old_metrics_file.exists():
                        old_metrics_file.unlink()
                except Exception as cleanup_error:
                    self.logger.warning("Failed to clean up old checkpoint",
                                      file=str(old_checkpoint),
                                      error=str(cleanup_error))
            
            self.logger.info("Checkpoint saved successfully",
                           episode=episode,
                           checkpoint_path=str(checkpoint_path),
                           metrics_file=str(metrics_file),
                           episodes_completed=self.metrics.episodes_completed,
                           current_performance=self.metrics.get_statistics())
                           
        except Exception as e:
            self.logger.error("Failed to save checkpoint",
                            episode=episode,
                            error=str(e),
                            checkpoint_path=self.config.model_save_path)
            raise TrainingPipelineError(f"Failed to save checkpoint for episode {episode}: {e}")


class HyperparameterSearch:
    """Hyperparameter optimization for training"""
    
    def __init__(self, search_space: Dict[str, List]):
        if not search_space:
            raise ValueError("Search space cannot be empty")
        
        for param, values in search_space.items():
            if not values:
                raise ValueError(f"Parameter values cannot be empty for parameter: {param}")
        
        self.search_space = search_space
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        self.logger = structlog.get_logger().bind(component="HyperparameterSearch")
    
    def _generate_parameter_combinations(self):
        """Generate all possible parameter combinations using grid search"""
        import itertools
        
        # Get parameter names and their values
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _split_data_for_cross_validation(self, historical_data, folds: int):
        """Split historical data for cross-validation"""
        cv_splits = []
        
        for token_address, data in historical_data.items():
            prices = data['prices']
            timestamps = data['timestamps']
            
            # Calculate split size
            total_samples = len(prices)
            fold_size = total_samples // folds
            
            token_splits = []
            for fold in range(folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < folds - 1 else total_samples
                
                # Create training and validation sets
                train_prices = prices[:start_idx] + prices[end_idx:]
                train_timestamps = timestamps[:start_idx] + timestamps[end_idx:]
                
                val_prices = prices[start_idx:end_idx]
                val_timestamps = timestamps[start_idx:end_idx]
                
                train_data = {'prices': train_prices, 'timestamps': train_timestamps}
                val_data = {'prices': val_prices, 'timestamps': val_timestamps}
                
                token_splits.append((train_data, val_data))
            
            cv_splits.append(token_splits)
        
        # Combine splits across tokens
        fold_splits = []
        for fold in range(folds):
            train_data = {}
            val_data = {}
            
            for token_idx, token_address in enumerate(historical_data.keys()):
                train_data[token_address] = cv_splits[token_idx][fold][0]
                val_data[token_address] = cv_splits[token_idx][fold][1]
            
            fold_splits.append((train_data, val_data))
        
        return fold_splits
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite performance score from multiple metrics"""
        # Normalize and combine multiple metrics for robust evaluation
        
        # Extract key metrics with fallback values
        mean_reward = metrics.get('mean_reward', 0.0)
        mean_win_rate = metrics.get('mean_win_rate', 0.0)
        mean_portfolio_value = metrics.get('mean_portfolio_value', 10000.0)  # Starting value
        total_trades = metrics.get('total_trades', 0)
        
        # Normalize metrics to [0, 1] scale
        # Reward score: positive rewards are good, normalize by expected range
        reward_score = max(0.0, min(1.0, (mean_reward + 100) / 300))  # Assuming range [-100, 200]
        
        # Win rate is already in [0, 1]
        win_rate_score = max(0.0, min(1.0, mean_win_rate))
        
        # Portfolio growth score
        portfolio_growth = max(0.0, (mean_portfolio_value - 10000) / 10000)  # Growth from 10k base
        portfolio_score = max(0.0, min(1.0, portfolio_growth + 0.5))  # Center around 50% = score 1.0
        
        # Trading activity score (some trading is good, but not excessive)
        activity_score = 1.0 if total_trades == 0 else max(0.0, min(1.0, total_trades / 100))
        
        # Composite score with weighted combination
        weights = {
            'reward': 0.3,
            'win_rate': 0.3,
            'portfolio': 0.3,
            'activity': 0.1
        }
        
        composite_score = (
            weights['reward'] * reward_score +
            weights['win_rate'] * win_rate_score +
            weights['portfolio'] * portfolio_score +
            weights['activity'] * activity_score
        )
        
        return max(0.0, min(1.0, composite_score))
    
    def _evaluate_parameter_configuration(self, params: Dict[str, Any], tokens, historical_data,
                                        episodes_per_trial: int, runs_per_config: int = 1,
                                        cross_validation_folds: int = None) -> float:
        """Evaluate a single parameter configuration"""
        
        if cross_validation_folds and cross_validation_folds > 1:
            # Use cross-validation
            cv_splits = self._split_data_for_cross_validation(historical_data, cross_validation_folds)
            scores = []
            
            for fold_idx, (train_data, val_data) in enumerate(cv_splits):
                if len(train_data[list(train_data.keys())[0]]['prices']) < 10:
                    # Skip folds with insufficient training data
                    continue
                
                try:
                    # Create training configuration with current parameters
                    config = TrainingConfig(
                        num_episodes=episodes_per_trial,
                        max_steps_per_episode=50,  # Reasonable default for hyperparameter search
                        **params  # Unpack hyperparameters
                    )
                    
                    # Train on training fold
                    pipeline = DQNTrainingPipeline(config, tokens, train_data)
                    train_results = pipeline.train()
                    
                    # Evaluate on validation fold if it has sufficient data
                    if len(val_data[list(val_data.keys())[0]]['prices']) >= 5:
                        val_config = TrainingConfig(
                            num_episodes=5,  # Short evaluation
                            max_steps_per_episode=20,
                            **params
                        )
                        val_pipeline = DQNTrainingPipeline(val_config, tokens, val_data)
                        val_results = val_pipeline.train()
                        fold_score = self._calculate_composite_score(val_results['final_metrics'])
                    else:
                        # Use training performance if validation set too small
                        fold_score = self._calculate_composite_score(train_results['final_metrics'])
                    
                    scores.append(fold_score)
                    
                except Exception as e:
                    self.logger.warning("Cross-validation fold failed", 
                                      fold=fold_idx, error=str(e), params=params)
                    continue
            
            if scores:
                return float(np.mean(scores))
            else:
                return 0.0
        
        else:
            # Use multiple independent runs
            scores = []
            
            for run in range(runs_per_config):
                try:
                    # Create training configuration with current parameters
                    config = TrainingConfig(
                        num_episodes=episodes_per_trial,
                        max_steps_per_episode=50,
                        **params
                    )
                    
                    # Create and run training pipeline
                    pipeline = DQNTrainingPipeline(config, tokens, historical_data)
                    results = pipeline.train()
                    
                    # Calculate performance score
                    score = self._calculate_composite_score(results['final_metrics'])
                    scores.append(score)
                    
                    self.logger.debug("Configuration run completed",
                                    run=run,
                                    params=params,
                                    score=score,
                                    episodes_completed=results['episodes_completed'])
                    
                except Exception as e:
                    self.logger.warning("Training run failed",
                                      run=run, error=str(e), params=params)
                    continue
            
            if scores:
                # Return mean score across runs for robustness
                return float(np.mean(scores))
            else:
                return 0.0
    
    def optimize(self, tokens, historical_data, num_trials: int = 10,
                episodes_per_trial: int = 100, runs_per_config: int = 1,
                cross_validation_folds: int = None,
                early_stopping_patience: int = None,
                early_stopping_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Run systematic hyperparameter optimization using grid search
        
        Args:
            tokens: List of tokens for training
            historical_data: Historical price data
            num_trials: Maximum number of parameter combinations to test
            episodes_per_trial: Number of episodes to train per configuration
            runs_per_config: Number of independent runs per configuration
            cross_validation_folds: Number of CV folds (if None, uses multiple runs)
            early_stopping_patience: Stop after N configs without improvement
            early_stopping_threshold: Stop if score exceeds this threshold
        
        Returns:
            Best parameter configuration found
        """
        
        self.logger.info("Starting hyperparameter optimization",
                        search_space=self.search_space,
                        num_trials=num_trials,
                        episodes_per_trial=episodes_per_trial,
                        runs_per_config=runs_per_config,
                        cross_validation_folds=cross_validation_folds)
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        self.logger.info("Generated parameter combinations",
                        total_combinations=len(param_combinations),
                        will_test=min(num_trials, len(param_combinations)))
        
        # Limit to num_trials if specified
        if num_trials < len(param_combinations):
            # Randomly sample combinations for efficiency
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:num_trials]
        
        best_score = -float('inf')
        best_params = None
        no_improvement_count = 0
        
        # Track optimization history
        self.optimization_history = []
        
        for idx, params in enumerate(param_combinations):
            self.logger.info("Testing parameter configuration",
                           config_num=idx + 1,
                           total_configs=len(param_combinations),
                           params=params)
            
            try:
                # Evaluate parameter configuration
                score = self._evaluate_parameter_configuration(
                    params=params,
                    tokens=tokens,
                    historical_data=historical_data,
                    episodes_per_trial=episodes_per_trial,
                    runs_per_config=runs_per_config,
                    cross_validation_folds=cross_validation_folds
                )
                
                # Record in optimization history
                self.optimization_history.append({
                    'params': params.copy(),
                    'score': score,
                    'config_num': idx + 1
                })
                
                self.logger.info("Configuration evaluation completed",
                               config_num=idx + 1,
                               params=params,
                               score=score,
                               current_best=best_score)
                
                # Update best parameters if score improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    no_improvement_count = 0
                    
                    self.logger.info("New best configuration found",
                                   best_params=best_params,
                                   best_score=best_score)
                else:
                    no_improvement_count += 1
                
                # Early stopping based on performance threshold
                if early_stopping_threshold and score >= early_stopping_threshold:
                    self.logger.info("Early stopping: performance threshold reached",
                                   score=score,
                                   threshold=early_stopping_threshold,
                                   configs_tested=idx + 1)
                    break
                
                # Early stopping based on lack of improvement
                if (early_stopping_patience and 
                    no_improvement_count >= early_stopping_patience):
                    self.logger.info("Early stopping: no improvement patience exceeded",
                                   no_improvement_count=no_improvement_count,
                                   patience=early_stopping_patience,
                                   configs_tested=idx + 1)
                    break
                    
            except Exception as e:
                self.logger.error("Parameter configuration evaluation failed",
                                config_num=idx + 1,
                                params=params,
                                error=str(e))
                
                # Record failed configuration
                self.optimization_history.append({
                    'params': params.copy(),
                    'score': 0.0,
                    'config_num': idx + 1,
                    'error': str(e)
                })
                continue
        
        # Store final results
        self.best_params = best_params
        self.best_score = best_score
        
        if best_params is None:
            self.logger.error("No valid parameter configuration found")
            # Return a default configuration from search space
            default_params = {}
            for param, values in self.search_space.items():
                default_params[param] = values[0]  # Take first value
            self.best_params = default_params
            self.best_score = 0.0
        
        self.logger.info("Hyperparameter optimization completed",
                        best_params=self.best_params,
                        best_score=self.best_score,
                        configurations_tested=len(self.optimization_history),
                        total_possible=len(self._generate_parameter_combinations()))
        
        return self.best_params