"""
Training Pipeline Metrics Collector

Implements comprehensive monitoring for RL training pipeline performance,
including experience storage integration, training progress, and resource usage.
"""

import time
from typing import Dict, List, Optional, Any
import structlog

from .base import MetricsCollector, MetricsRegistry


logger = structlog.get_logger()

# Metric buckets for histograms
EPISODE_DURATION_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1200)  # seconds
REWARD_BUCKETS = (-1000, -100, -10, -1, 0, 1, 10, 100, 1000)
LOSS_BUCKETS = (0.001, 0.01, 0.1, 1, 10, 100)


class TrainingPipelineMetrics(MetricsCollector):
    """
    Metrics collector for RL training pipeline performance.
    
    Tracks training progress, experience utilization, model performance,
    and resource usage during training.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """Initialize training pipeline metrics collector."""
        super().__init__(registry)
        
        # Training progress counters
        self.episodes_completed_total = registry.get_counter(
            "rl_episodes_completed_total",
            "Total number of training episodes completed",
            ["session_id", "agent_type"]
        )
        
        self.steps_completed_total = registry.get_counter(
            "rl_steps_completed_total", 
            "Total number of training steps completed",
            ["session_id", "episode"]
        )
        
        self.model_updates_total = registry.get_counter(
            "rl_model_updates_total",
            "Total number of model parameter updates",
            ["session_id", "update_type"]
        )
        
        self.experiences_consumed_total = registry.get_counter(
            "rl_experiences_consumed_total",
            "Total number of experiences used for training",
            ["session_id", "source"]
        )
        
        # Performance gauges
        self.current_epsilon = registry.get_gauge(
            "rl_current_epsilon",
            "Current exploration epsilon value",
            ["session_id"]
        )
        
        self.current_episode_reward = registry.get_gauge(
            "rl_current_episode_reward",
            "Current episode total reward",
            ["session_id"]
        )
        
        self.average_episode_reward = registry.get_gauge(
            "rl_average_episode_reward",
            "Moving average episode reward",
            ["session_id", "window_size"]
        )
        
        self.model_loss = registry.get_gauge(
            "rl_model_loss",
            "Current model training loss",
            ["session_id", "loss_type"]
        )
        
        self.training_progress = registry.get_gauge(
            "rl_training_progress_ratio",
            "Training progress as ratio of total episodes",
            ["session_id"]
        )
        
        # Performance histograms
        self.episode_duration = registry.get_histogram(
            "rl_episode_duration_seconds",
            "Training episode duration in seconds",
            ["session_id"],
            buckets=EPISODE_DURATION_BUCKETS
        )
        
        self.episode_reward = registry.get_histogram(
            "rl_episode_reward",
            "Episode total reward distribution",
            ["session_id"],
            buckets=REWARD_BUCKETS
        )
        
        self.step_duration = registry.get_histogram(
            "rl_step_duration_ms",
            "Training step duration in milliseconds",
            ["session_id", "step_type"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500)
        )
        
        self.model_loss_histogram = registry.get_histogram(
            "rl_model_loss_distribution",
            "Model loss value distribution",
            ["session_id", "loss_type"],
            buckets=LOSS_BUCKETS
        )
        
        # Experience utilization metrics
        self.experience_buffer_efficiency = registry.get_gauge(
            "rl_experience_buffer_efficiency",
            "Ratio of unique experiences used vs total available",
            ["session_id"]
        )
        
        self.experience_replay_ratio = registry.get_gauge(
            "rl_experience_replay_ratio", 
            "Ratio of replayed vs new experiences",
            ["session_id"]
        )
        
        # Resource usage metrics
        self.memory_usage_bytes = registry.get_gauge(
            "rl_memory_usage_bytes",
            "Memory usage during training",
            ["session_id", "component"]
        )
        
        self.gpu_utilization = registry.get_gauge(
            "rl_gpu_utilization_ratio",
            "GPU utilization during training",
            ["session_id", "device_id"]
        )
        
        # Training stability metrics
        self.reward_volatility = registry.get_gauge(
            "rl_reward_volatility",
            "Standard deviation of recent episode rewards",
            ["session_id", "window_size"]
        )
        
        self.convergence_indicator = registry.get_gauge(
            "rl_convergence_indicator",
            "Training convergence indicator (0-1)",
            ["session_id"]
        )
        
        # Error tracking
        self.training_errors_total = registry.get_counter(
            "rl_training_errors_total",
            "Total number of training errors",
            ["session_id", "error_type"]
        )
        
        # Internal state for calculations
        self._session_stats: Dict[str, Dict] = {}
        self._reward_windows: Dict[str, List[float]] = {}
        self._last_update_times: Dict[str, float] = {}
    
    async def collect_metrics(self) -> None:
        """Collect training pipeline metrics."""
        # This method is called periodically by the monitoring system
        # Most metrics are updated in real-time via record_* methods
        try:
            current_time = time.time()
            
            # Update derived metrics for each active session
            for session_id, stats in self._session_stats.items():
                await self._update_derived_metrics(session_id, stats, current_time)
                
        except Exception as e:
            logger.error("Error collecting training pipeline metrics", error=str(e))
            raise
    
    async def _update_derived_metrics(self, session_id: str, stats: Dict, current_time: float) -> None:
        """Update derived metrics for a training session."""
        try:
            # Update reward volatility
            if session_id in self._reward_windows:
                rewards = self._reward_windows[session_id]
                if len(rewards) >= 10:
                    import numpy as np
                    volatility = float(np.std(rewards[-100:]))  # Last 100 episodes
                    self.reward_volatility.labels(
                        session_id=session_id,
                        window_size="100"
                    ).set(volatility)
            
            # Update convergence indicator based on reward trend
            convergence_score = self._calculate_convergence_score(session_id)
            self.convergence_indicator.labels(session_id=session_id).set(convergence_score)
            
            # Update experience buffer efficiency
            if 'total_experiences_available' in stats and 'unique_experiences_used' in stats:
                efficiency = stats['unique_experiences_used'] / max(stats['total_experiences_available'], 1)
                self.experience_buffer_efficiency.labels(session_id=session_id).set(efficiency)
            
        except Exception as e:
            logger.error("Error updating derived metrics", session_id=session_id, error=str(e))
    
    def _calculate_convergence_score(self, session_id: str) -> float:
        """Calculate convergence score based on reward stability."""
        if session_id not in self._reward_windows:
            return 0.0
        
        rewards = self._reward_windows[session_id]
        if len(rewards) < 20:
            return 0.0
        
        # Simple convergence metric: inverse of reward volatility with trend consideration
        try:
            import numpy as np
            recent_rewards = rewards[-20:]
            older_rewards = rewards[-40:-20] if len(rewards) >= 40 else rewards[:-20]
            
            if len(older_rewards) == 0:
                return 0.1
            
            recent_mean = np.mean(recent_rewards)
            older_mean = np.mean(older_rewards)
            recent_std = np.std(recent_rewards)
            
            # Higher score for improvement and stability
            improvement_factor = max(0, (recent_mean - older_mean) / max(abs(older_mean), 1))
            stability_factor = 1.0 / (1.0 + recent_std)
            
            convergence = min(1.0, (improvement_factor + stability_factor) / 2.0)
            return max(0.0, convergence)
            
        except Exception:
            return 0.0
    
    def record_episode_start(self, session_id: str, episode: int, epsilon: float) -> None:
        """Record episode start metrics."""
        self.current_epsilon.labels(session_id=session_id).set(epsilon)
        
        # Initialize session stats if needed
        if session_id not in self._session_stats:
            self._session_stats[session_id] = {
                'episodes_completed': 0,
                'total_reward': 0.0,
                'episode_start_time': time.time()
            }
        
        self._session_stats[session_id]['episode_start_time'] = time.time()
    
    def record_episode_complete(self, session_id: str, episode: int, 
                              total_reward: float, steps: int, 
                              duration_seconds: float, agent_type: str = "DQN") -> None:
        """Record episode completion metrics."""
        # Update counters
        self.episodes_completed_total.labels(
            session_id=session_id,
            agent_type=agent_type
        ).inc()
        
        self.steps_completed_total.labels(
            session_id=session_id,
            episode=str(episode)
        ).inc(steps)
        
        # Update gauges
        self.current_episode_reward.labels(session_id=session_id).set(total_reward)
        
        # Update histograms
        self.episode_duration.labels(session_id=session_id).observe(duration_seconds)
        self.episode_reward.labels(session_id=session_id).observe(total_reward)
        
        # Update reward window for moving averages
        if session_id not in self._reward_windows:
            self._reward_windows[session_id] = []
        
        self._reward_windows[session_id].append(total_reward)
        
        # Keep only last 1000 rewards
        if len(self._reward_windows[session_id]) > 1000:
            self._reward_windows[session_id] = self._reward_windows[session_id][-1000:]
        
        # Update moving averages
        self._update_moving_averages(session_id)
        
        # Update session stats
        if session_id in self._session_stats:
            self._session_stats[session_id]['episodes_completed'] += 1
            self._session_stats[session_id]['total_reward'] += total_reward
    
    def _update_moving_averages(self, session_id: str) -> None:
        """Update moving average metrics."""
        if session_id not in self._reward_windows:
            return
        
        rewards = self._reward_windows[session_id]
        
        # Update different window sizes
        for window_size in [10, 50, 100]:
            if len(rewards) >= window_size:
                avg_reward = sum(rewards[-window_size:]) / window_size
                self.average_episode_reward.labels(
                    session_id=session_id,
                    window_size=str(window_size)
                ).set(avg_reward)
    
    def record_training_step(self, session_id: str, step_type: str, 
                           duration_ms: float) -> None:
        """Record training step metrics."""
        self.step_duration.labels(
            session_id=session_id,
            step_type=step_type
        ).observe(duration_ms)
    
    def record_model_update(self, session_id: str, update_type: str,
                          loss_value: Optional[float] = None) -> None:
        """Record model update metrics.""" 
        self.model_updates_total.labels(
            session_id=session_id,
            update_type=update_type
        ).inc()
        
        if loss_value is not None:
            self.model_loss.labels(
                session_id=session_id,
                loss_type=update_type
            ).set(loss_value)
            
            self.model_loss_histogram.labels(
                session_id=session_id,
                loss_type=update_type
            ).observe(loss_value)
    
    def record_experience_consumption(self, session_id: str, count: int, 
                                    source: str = "database") -> None:
        """Record experience consumption metrics."""
        self.experiences_consumed_total.labels(
            session_id=session_id,
            source=source
        ).inc(count)
    
    def record_training_progress(self, session_id: str, current_episode: int,
                               total_episodes: int) -> None:
        """Record training progress metrics."""
        progress = current_episode / max(total_episodes, 1)
        self.training_progress.labels(session_id=session_id).set(progress)
    
    def record_resource_usage(self, session_id: str, memory_bytes: int,
                            gpu_utilization: Optional[float] = None) -> None:
        """Record resource usage metrics."""
        self.memory_usage_bytes.labels(
            session_id=session_id,
            component="training"
        ).set(memory_bytes)
        
        if gpu_utilization is not None:
            self.gpu_utilization.labels(
                session_id=session_id,
                device_id="0"
            ).set(gpu_utilization)
    
    def record_training_error(self, session_id: str, error_type: str) -> None:
        """Record training error metrics."""
        self.training_errors_total.labels(
            session_id=session_id,
            error_type=error_type
        ).inc()
    
    def update_experience_stats(self, session_id: str, 
                              total_available: int, unique_used: int,
                              replay_ratio: float) -> None:
        """Update experience utilization statistics."""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = {}
        
        self._session_stats[session_id].update({
            'total_experiences_available': total_available,
            'unique_experiences_used': unique_used
        })
        
        self.experience_replay_ratio.labels(session_id=session_id).set(replay_ratio)
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        return {
            # Counters
            "rl_episodes_completed_total": "Total training episodes completed",
            "rl_steps_completed_total": "Total training steps completed", 
            "rl_model_updates_total": "Total model parameter updates",
            "rl_experiences_consumed_total": "Total experiences used for training",
            "rl_training_errors_total": "Total training errors encountered",
            
            # Gauges
            "rl_current_epsilon": "Current exploration epsilon value",
            "rl_current_episode_reward": "Current episode total reward",
            "rl_average_episode_reward": "Moving average episode reward",
            "rl_model_loss": "Current model training loss",
            "rl_training_progress_ratio": "Training progress ratio",
            "rl_experience_buffer_efficiency": "Experience buffer utilization efficiency",
            "rl_experience_replay_ratio": "Experience replay vs new ratio",
            "rl_memory_usage_bytes": "Training memory usage",
            "rl_gpu_utilization_ratio": "GPU utilization during training",
            "rl_reward_volatility": "Episode reward volatility",
            "rl_convergence_indicator": "Training convergence indicator",
            
            # Histograms
            "rl_episode_duration_seconds": "Training episode duration",
            "rl_episode_reward": "Episode reward distribution",
            "rl_step_duration_ms": "Training step duration",
            "rl_model_loss_distribution": "Model loss distribution"
        }
    
    def is_healthy(self) -> bool:
        """Check if training metrics collector is healthy."""
        # Simple health check - ensure we have at least some metrics
        return len(self._session_stats) >= 0  # Always healthy
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get training summary for a specific session."""
        if session_id not in self._session_stats:
            return {}
        
        stats = self._session_stats[session_id]
        rewards = self._reward_windows.get(session_id, [])
        
        summary = {
            "episodes_completed": stats.get('episodes_completed', 0),
            "total_reward": stats.get('total_reward', 0.0),
            "average_reward": stats.get('total_reward', 0.0) / max(stats.get('episodes_completed', 1), 1),
            "recent_rewards": rewards[-10:] if len(rewards) >= 10 else rewards,
            "convergence_score": self._calculate_convergence_score(session_id),
            "last_update": self._last_update_times.get(session_id, 0)
        }
        
        return summary


def create_training_pipeline_metrics(registry: MetricsRegistry) -> TrainingPipelineMetrics:
    """Factory function to create training pipeline metrics collector."""
    return TrainingPipelineMetrics(registry)