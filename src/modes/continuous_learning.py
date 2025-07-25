"""
Continuous Learning Engine for RL Trading Agent

Implements automatic retraining triggers, incremental learning, model versioning,
and performance monitoring to complete the RL feedback loop. This allows the
trading agent to continuously improve from real trading experiences.

Following TDD methodology - implementation after comprehensive tests.
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

from src.rl_agent.base import TradeAction, MarketState
from src.rl_agent.experience_replay import Experience, ExperienceReplayBuffer
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig


logger = structlog.get_logger()


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning engine"""
    
    # Training triggers
    training_trigger_threshold: int = 1000  # New experiences to trigger training
    performance_evaluation_window: int = 50  # Episodes to evaluate performance
    min_improvement_threshold: float = 0.05  # Minimum improvement to keep new model
    
    # Model management
    max_model_versions: int = 5  # Maximum number of model versions to keep
    training_episodes_per_session: int = 100  # Episodes per training session
    
    # Persistence
    enable_model_persistence: bool = True
    model_storage_path: str = "models/continuous/"
    
    # Performance monitoring
    performance_rollback_threshold: float = -0.10  # Performance drop to trigger rollback
    training_timeout_minutes: int = 30  # Training session timeout
    
    # Incremental learning
    enable_incremental_learning: bool = True  # Continue from current weights vs restart
    
    def __post_init__(self):
        """Validate configuration"""
        if self.training_trigger_threshold <= 0:
            raise ValueError("training_trigger_threshold must be positive")
        if self.performance_evaluation_window <= 0:
            raise ValueError("performance_evaluation_window must be positive")
        if self.min_improvement_threshold < 0:
            raise ValueError("min_improvement_threshold must be non-negative")


@dataclass
class TrainingTrigger:
    """Represents a trigger for starting training"""
    
    trigger_type: str  # e.g., "experience_threshold", "performance_degradation"
    threshold_value: float
    current_value: float
    description: str = ""
    triggered_at: datetime = field(default_factory=datetime.now)
    
    def is_triggered(self) -> bool:
        """Check if trigger condition is met"""
        if self.trigger_type == "experience_threshold":
            return self.current_value >= self.threshold_value
        elif self.trigger_type == "performance_degradation":
            return self.current_value <= self.threshold_value  # Note: negative threshold
        else:
            return False


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model version"""
    
    episodes_trained: int
    average_reward: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    
    def calculate_composite_score(self) -> float:
        """Calculate composite performance score for comparison"""
        # Weighted combination of metrics
        reward_score = self.average_reward / 100.0  # Normalize to reasonable scale
        win_rate_score = self.win_rate * 100.0
        sharpe_score = self.sharpe_ratio * 10.0
        drawdown_penalty = self.max_drawdown * 100.0  # Penalty for drawdown
        
        composite = (reward_score * 0.3 + 
                    win_rate_score * 0.3 + 
                    sharpe_score * 0.3 - 
                    drawdown_penalty * 0.1)
        
        return float(composite)


@dataclass
class ModelVersion:
    """Represents a trained model version"""
    
    version_id: str
    created_at: datetime
    model_path: str
    training_episodes: int
    performance_metrics: ModelPerformanceMetrics
    parent_version_id: Optional[str] = None
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'version_id': self.version_id,
            'created_at': self.created_at.isoformat(),
            'model_path': self.model_path,
            'training_episodes': self.training_episodes,
            'performance_metrics': asdict(self.performance_metrics),
            'parent_version_id': self.parent_version_id,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        metrics_data = data['performance_metrics']
        metrics = ModelPerformanceMetrics(**metrics_data)
        
        return cls(
            version_id=data['version_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            model_path=data['model_path'],
            training_episodes=data['training_episodes'],
            performance_metrics=metrics,
            parent_version_id=data.get('parent_version_id'),
            is_active=data.get('is_active', False)
        )


@dataclass
class TrainingSession:
    """Represents a single training session"""
    
    session_id: str
    started_at: datetime
    trigger_reason: str
    target_episodes: int
    actual_episodes: int = 0
    completed_at: Optional[datetime] = None
    success: bool = False
    final_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def complete(self, success: bool, actual_episodes: int, 
                final_metrics: Optional[Dict[str, Any]] = None,
                error_message: Optional[str] = None) -> None:
        """Mark session as completed"""
        self.completed_at = datetime.now()
        self.success = success
        self.actual_episodes = actual_episodes
        self.final_metrics = final_metrics
        self.error_message = error_message
    
    def get_duration_minutes(self) -> float:
        """Get session duration in minutes"""
        if self.completed_at:
            delta = self.completed_at - self.started_at
        else:
            delta = datetime.now() - self.started_at
        return delta.total_seconds() / 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'started_at': self.started_at.isoformat(),
            'trigger_reason': self.trigger_reason,
            'target_episodes': self.target_episodes,
            'actual_episodes': self.actual_episodes,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'success': self.success,
            'final_metrics': self.final_metrics,
            'error_message': self.error_message
        }


class ContinuousLearningEngine:
    """
    Continuous Learning Engine for RL Trading Agent
    
    Provides automatic retraining, model versioning, performance monitoring,
    and incremental learning to complete the RL feedback loop.
    """
    
    def __init__(self, config: ContinuousLearningConfig, 
                 replay_buffer: ExperienceReplayBuffer,
                 dqn_agent: DQNTradingAgent,
                 experience_collector: TradingExperienceCollector):
        self.config = config
        self.replay_buffer = replay_buffer
        self.dqn_agent = dqn_agent
        self.experience_collector = experience_collector
        
        self.logger = structlog.get_logger().bind(component="ContinuousLearningEngine")
        
        # Training state
        self.is_training = False
        self.current_session: Optional[TrainingSession] = None
        
        # Model versioning
        self.model_versions: List[ModelVersion] = []
        self.active_model_version: Optional[ModelVersion] = None
        
        # Training history
        self.training_history: List[TrainingSession] = []
        
        # Performance tracking
        self.last_performance_check: Optional[datetime] = None
        self.performance_history: List[Dict[str, Any]] = []
        
        # Ensure model storage directory exists
        if self.config.enable_model_persistence:
            os.makedirs(self.config.model_storage_path, exist_ok=True)
        
        self.logger.info("Continuous Learning Engine initialized",
                        trigger_threshold=config.training_trigger_threshold,
                        model_storage_path=config.model_storage_path,
                        incremental_learning=config.enable_incremental_learning)
    
    async def check_training_triggers(self) -> List[TrainingTrigger]:
        """
        Check all training triggers and return activated ones
        
        Returns:
            List of training triggers with their status
        """
        triggers = []
        
        # Experience threshold trigger
        buffer_size = len(self.replay_buffer)
        experience_trigger = TrainingTrigger(
            trigger_type="experience_threshold",
            threshold_value=self.config.training_trigger_threshold,
            current_value=buffer_size,
            description=f"Experience buffer contains {buffer_size} experiences"
        )
        triggers.append(experience_trigger)
        
        # Performance degradation trigger (if we have performance history)
        if self.performance_history:
            recent_performance = self.performance_history[-self.config.performance_evaluation_window:]
            if len(recent_performance) >= self.config.performance_evaluation_window:
                avg_recent = np.mean([p['average_reward'] for p in recent_performance])
                baseline_performance = self.performance_history[0]['average_reward']
                
                performance_change = (avg_recent - baseline_performance) / baseline_performance
                
                perf_trigger = TrainingTrigger(
                    trigger_type="performance_degradation",
                    threshold_value=self.config.performance_rollback_threshold,
                    current_value=performance_change,
                    description=f"Performance change: {performance_change:.3f}"
                )
                triggers.append(perf_trigger)
        
        # Log trigger status
        activated_triggers = [t for t in triggers if t.is_triggered()]
        if activated_triggers:
            self.logger.info("Training triggers activated",
                           count=len(activated_triggers),
                           triggers=[t.trigger_type for t in activated_triggers])
        
        return triggers
    
    async def start_training_session(self, trigger: TrainingTrigger) -> TrainingSession:
        """
        Start a new training session
        
        Args:
            trigger: The trigger that initiated training
            
        Returns:
            TrainingSession object
            
        Raises:
            ContinuousLearningError: If training is already active
        """
        if self.is_training:
            raise ContinuousLearningError("Training session already active")
        
        # Create new session
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session = TrainingSession(
            session_id=session_id,
            started_at=datetime.now(),
            trigger_reason=trigger.trigger_type,
            target_episodes=self.config.training_episodes_per_session
        )
        
        self.current_session = session
        self.is_training = True
        
        self.logger.info("Training session started",
                        session_id=session_id,
                        trigger=trigger.trigger_type,
                        target_episodes=session.target_episodes)
        
        return session
    
    async def run_training_session(self) -> Dict[str, Any]:
        """
        Run the active training session
        
        Returns:
            Training results
            
        Raises:
            ContinuousLearningError: If no active session
        """
        if not self.current_session:
            raise ContinuousLearningError("No active training session")
        
        session = self.current_session
        
        try:
            self.logger.info("Running training session", session_id=session.session_id)
            
            # Create training configuration
            training_config = TrainingConfig(
                num_episodes=session.target_episodes,
                learning_rate=0.001,
                batch_size=32,
                save_frequency=25,
                use_prioritized_replay=True,
                use_advanced_rewards=True,
                use_ml_features=False  # Can be configured based on ML integration
            )
            
            # Create mock tokens and historical data for training
            # In real implementation, this would use actual market data
            mock_tokens = [{'symbol': 'BTC', 'address': '0x123'}]
            mock_historical_data = {'BTC': np.random.randn(1000, 5)}  # Mock price data
            
            # Initialize training pipeline
            training_pipeline = DQNTrainingPipeline(
                config=training_config,
                tokens=mock_tokens,
                historical_data=mock_historical_data
            )
            
            # Set the agent's replay buffer to use accumulated experiences
            training_pipeline.replay_buffer = self.replay_buffer
            
            # Run training
            training_result = training_pipeline.train()
            
            # Mark session as completed
            session.complete(
                success=True,
                actual_episodes=training_result['episodes_completed'],
                final_metrics=training_result['final_metrics']
            )
            
            self.logger.info("Training session completed successfully",
                           session_id=session.session_id,
                           episodes=training_result['episodes_completed'],
                           duration_minutes=session.get_duration_minutes())
            
            return training_result
            
        except Exception as e:
            # Mark session as failed
            session.complete(
                success=False,
                actual_episodes=0,
                error_message=str(e)
            )
            
            self.logger.error("Training session failed",
                            session_id=session.session_id,
                            error=str(e))
            raise
        
        finally:
            # Clean up training state
            self.is_training = False
            if session:
                self.training_history.append(session)
            self.current_session = None
    
    async def create_model_version(self, session: TrainingSession, 
                                  training_result: Dict[str, Any]) -> ModelVersion:
        """
        Create a new model version from training session
        
        Args:
            session: Completed training session
            training_result: Results from training
            
        Returns:
            New ModelVersion object
        """
        # Generate version ID
        version_number = len(self.model_versions) + 1
        version_id = f"v1.{version_number}.0"
        
        # Create performance metrics from training results
        final_metrics = training_result['final_metrics']
        performance_metrics = ModelPerformanceMetrics(
            episodes_trained=training_result['episodes_completed'],
            average_reward=final_metrics.get('mean_reward', 0.0),
            win_rate=final_metrics.get('mean_win_rate', 0.0),
            sharpe_ratio=final_metrics.get('mean_portfolio_value', 10000) / 10000.0,  # Mock Sharpe
            max_drawdown=0.05,  # Mock max drawdown
            total_trades=final_metrics.get('total_trades', 100)
        )
        
        # Generate model file path
        model_filename = f"model_{version_id.replace('.', '_')}.pth"
        model_path = os.path.join(self.config.model_storage_path, model_filename)
        
        # Save model if persistence enabled
        if self.config.enable_model_persistence:
            self.dqn_agent.save_model(model_path)
        
        # Create model version
        parent_version_id = self.active_model_version.version_id if self.active_model_version else None
        
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now(),
            model_path=model_path,
            training_episodes=training_result['episodes_completed'],
            performance_metrics=performance_metrics,
            parent_version_id=parent_version_id
        )
        
        self.model_versions.append(version)
        
        self.logger.info("Model version created",
                        version_id=version_id,
                        episodes=version.training_episodes,
                        avg_reward=performance_metrics.average_reward,
                        model_path=model_path)
        
        return version
    
    async def evaluate_model_performance(self, new_version: ModelVersion) -> bool:
        """
        Evaluate if new model version is better than current
        
        Args:
            new_version: New model version to evaluate
            
        Returns:
            True if new version is better
        """
        if not self.model_versions:
            # First model is always accepted
            return True
        
        # Get current best version
        active_versions = [v for v in self.model_versions if v != new_version]
        if not active_versions:
            return True
        
        # Compare with best existing version
        best_existing = max(active_versions, 
                          key=lambda v: v.performance_metrics.calculate_composite_score())
        
        new_score = new_version.performance_metrics.calculate_composite_score()
        existing_score = best_existing.performance_metrics.calculate_composite_score()
        
        improvement = (new_score - existing_score) / existing_score
        
        is_better = improvement >= self.config.min_improvement_threshold
        
        self.logger.info("Model performance evaluation",
                        new_version=new_version.version_id,
                        new_score=new_score,
                        existing_score=existing_score,
                        improvement=improvement,
                        is_better=is_better)
        
        return is_better
    
    async def activate_model_version(self, version_id: str) -> None:
        """
        Activate a specific model version
        
        Args:
            version_id: ID of version to activate
            
        Raises:
            ContinuousLearningError: If version not found
        """
        # Find version
        version = next((v for v in self.model_versions if v.version_id == version_id), None)
        if not version:
            raise ContinuousLearningError(f"Model version not found: {version_id}")
        
        # Deactivate current version
        if self.active_model_version:
            self.active_model_version.is_active = False
        
        # Activate new version
        version.is_active = True
        self.active_model_version = version
        
        # Load model into agent
        self.dqn_agent.load_model(version.model_path)
        
        self.logger.info("Model version activated",
                        version_id=version_id,
                        model_path=version.model_path)
    
    async def rollback_to_previous_version(self) -> bool:
        """
        Rollback to the previous best model version
        
        Returns:
            True if rollback successful, False if no previous version
        """
        if len(self.model_versions) < 2:
            self.logger.warning("No previous version available for rollback")
            return False
        
        # Find previous best version (exclude current active)
        available_versions = [v for v in self.model_versions 
                            if v != self.active_model_version]
        
        if not available_versions:
            return False
        
        # Select best available version
        best_previous = max(available_versions,
                          key=lambda v: v.performance_metrics.calculate_composite_score())
        
        await self.activate_model_version(best_previous.version_id)
        
        self.logger.info("Rolled back to previous version",
                        version_id=best_previous.version_id)
        
        return True
    
    async def cleanup_old_versions(self) -> int:
        """
        Clean up old model versions exceeding max limit
        
        Returns:
            Number of versions cleaned up
        """
        if len(self.model_versions) <= self.config.max_model_versions:
            return 0
        
        # Sort by creation date (oldest first)
        sorted_versions = sorted(self.model_versions, key=lambda v: v.created_at)
        
        # Keep max_model_versions most recent, prioritizing active version
        if self.active_model_version:
            # Always keep active version
            versions_to_keep = [self.active_model_version]
            # Fill remaining slots with most recent versions (excluding active)
            non_active_versions = [v for v in sorted_versions if v != self.active_model_version]
            remaining_slots = self.config.max_model_versions - 1
            if remaining_slots > 0:
                versions_to_keep.extend(non_active_versions[-remaining_slots:])
        else:
            # No active version, just keep most recent
            versions_to_keep = sorted_versions[-self.config.max_model_versions:]
        
        versions_to_remove = [v for v in self.model_versions if v not in versions_to_keep]
        
        # Remove old versions
        cleaned_count = 0
        for version in versions_to_remove:
            # Remove model file
            if self.config.enable_model_persistence:
                try:
                    if os.path.exists(version.model_path):
                        os.remove(version.model_path)
                except Exception as e:
                    self.logger.warning("Failed to remove model file",
                                      path=version.model_path, error=str(e))
            
            # Remove from list
            self.model_versions.remove(version)
            cleaned_count += 1
            
            self.logger.debug("Removed old model version", version_id=version.version_id)
        
        if cleaned_count > 0:
            self.logger.info("Cleaned up old model versions", count=cleaned_count)
        
        return cleaned_count
    
    async def save_state(self, file_path: str) -> None:
        """
        Save engine state to file
        
        Args:
            file_path: Path to save state file
        """
        state = {
            'model_versions': [v.to_dict() for v in self.model_versions],
            'active_model_version_id': self.active_model_version.version_id if self.active_model_version else None,
            'training_history': [s.to_dict() for s in self.training_history],
            'performance_history': self.performance_history,
            'last_performance_check': self.last_performance_check.isoformat() if self.last_performance_check else None
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info("Engine state saved", path=file_path)
    
    async def load_state(self, file_path: str) -> None:
        """
        Load engine state from file
        
        Args:
            file_path: Path to state file
        """
        if not os.path.exists(file_path):
            self.logger.info("No previous state file found", path=file_path)
            return
        
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        # Load model versions
        self.model_versions = [ModelVersion.from_dict(v) for v in state['model_versions']]
        
        # Set active version
        active_version_id = state.get('active_model_version_id')
        if active_version_id:
            self.active_model_version = next(
                (v for v in self.model_versions if v.version_id == active_version_id),
                None
            )
        
        # Load history
        self.performance_history = state.get('performance_history', [])
        
        last_check_str = state.get('last_performance_check')
        if last_check_str:
            self.last_performance_check = datetime.fromisoformat(last_check_str)
        
        self.logger.info("Engine state loaded",
                        path=file_path,
                        model_versions=len(self.model_versions),
                        active_version=active_version_id)
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'total_training_sessions': len(self.training_history),
            'successful_sessions': sum(1 for s in self.training_history if s.success),
            'total_model_versions': len(self.model_versions),
            'active_model_version': self.active_model_version.version_id if self.active_model_version else None,
            'is_training': self.is_training,
            'current_session_id': self.current_session.session_id if self.current_session else None
        }
        
        # Training session statistics
        if self.training_history:
            successful_sessions = [s for s in self.training_history if s.success]
            if successful_sessions:
                avg_duration = np.mean([s.get_duration_minutes() for s in successful_sessions])
                avg_episodes = np.mean([s.actual_episodes for s in successful_sessions])
                
                stats.update({
                    'average_session_duration_minutes': float(avg_duration),
                    'average_episodes_per_session': float(avg_episodes)
                })
        
        # Model performance statistics
        if self.model_versions:
            best_version = max(self.model_versions,
                             key=lambda v: v.performance_metrics.calculate_composite_score())
            
            stats['best_performance'] = {
                'version_id': best_version.version_id,
                'average_reward': best_version.performance_metrics.average_reward,
                'win_rate': best_version.performance_metrics.win_rate,
                'sharpe_ratio': best_version.performance_metrics.sharpe_ratio
            }
        
        # Buffer statistics
        if self.replay_buffer:
            stats.update({
                'buffer_size': len(self.replay_buffer),
                'buffer_can_sample': self.replay_buffer.can_sample()
            })
        
        return stats
    
    async def run_automatic_training_cycle(self) -> Dict[str, Any]:
        """
        Run complete automatic training cycle
        
        Returns:
            Results of training cycle
        """
        try:
            # Check training triggers
            triggers = await self.check_training_triggers()
            activated_triggers = [t for t in triggers if t.is_triggered()]
            
            if not activated_triggers:
                return {
                    'training_triggered': False,
                    'trigger_status': [
                        {
                            'type': t.trigger_type,
                            'threshold': t.threshold_value,
                            'current': t.current_value,
                            'triggered': t.is_triggered()
                        }
                        for t in triggers
                    ]
                }
            
            # Start training session
            primary_trigger = activated_triggers[0]  # Use first trigger
            session = await self.start_training_session(primary_trigger)
            
            # Run training
            training_result = await self.run_training_session()
            
            # Create new model version
            new_version = await self.create_model_version(session, training_result)
            
            # Evaluate performance
            is_better = await self.evaluate_model_performance(new_version)
            
            result = {
                'training_triggered': True,
                'session_success': session.success,
                'session_id': session.session_id,
                'trigger_reason': primary_trigger.trigger_type,
                'episodes_completed': training_result['episodes_completed'],
                'new_model_version': new_version.version_id,
                'performance_improved': is_better
            }
            
            # Activate new version if better
            if is_better:
                await self.activate_model_version(new_version.version_id)
                result['model_activated'] = True
            else:
                result['model_activated'] = False
                self.logger.info("New model version not activated due to insufficient improvement")
            
            # Cleanup old versions
            cleaned_count = await self.cleanup_old_versions()
            result['versions_cleaned'] = cleaned_count
            
            return result
            
        except Exception as e:
            self.logger.error("Automatic training cycle failed", error=str(e))
            raise ContinuousLearningError(f"Training cycle failed: {str(e)}")
    
    async def check_and_trigger_training(self) -> Optional[Dict[str, Any]]:
        """
        Check if training should be triggered and run if necessary
        
        Returns:
            Training results if triggered, None otherwise
        """
        # Skip if already training
        if self.is_training:
            self.logger.debug("Training already in progress, skipping trigger check")
            return None
        
        # Check if experience collector suggests training
        if (self.experience_collector and 
            self.experience_collector.should_trigger_training(self.config.training_trigger_threshold)):
            
            self.logger.info("Training triggered by experience collector",
                           buffer_size=self.experience_collector.get_buffer_size(),
                           threshold=self.config.training_trigger_threshold)
            
            return await self.run_automatic_training_cycle()
        
        return None


class ContinuousLearningError(Exception):
    """Raised when continuous learning operations fail"""
    pass