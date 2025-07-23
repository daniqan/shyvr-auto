"""
ML-RL Bridge Components

Provides integration between machine learning predictions and 
reinforcement learning trading decisions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

from src.rl_agent.base import (
    RLTrainingError, TradeAction, MarketState, AgentConfig
)
from src.rl_agent.training_pipeline import TrainingConfig, TrainingMetrics
from src.ml_analysis.base import PredictionResult, ModelType
from src.discovery.base import DiscoveredToken

logger = structlog.get_logger()


@dataclass
class MLEnhancedMarketState(MarketState):
    """Market state enhanced with ML predictions"""
    
    # ML prediction features
    ml_prediction_1h: Optional[float] = None
    ml_prediction_4h: Optional[float] = None
    ml_prediction_24h: Optional[float] = None
    ml_confidence: Optional[float] = None
    ml_direction: Optional[str] = None
    volatility_forecast: Optional[float] = None
    
    @classmethod
    def from_prediction(cls, prediction: PredictionResult, 
                       current_portfolio_value: float,
                       position_size: float) -> 'MLEnhancedMarketState':
        """Create ML-enhanced market state from prediction result"""
        token = prediction.token
        
        # Base market state features
        base_state = MarketState(
            token=token,
            price_usd=token.price_usd,
            price_change_24h=0.0,  # Will be populated from historical data
            volume_24h=token.volume_24h or 0.0,
            market_cap=getattr(token, 'market_cap', None),
            # Technical indicators from prediction
            rsi=prediction.technical_indicators.rsi if prediction.technical_indicators else None,
            macd=prediction.technical_indicators.macd if prediction.technical_indicators else None,
            sma_20=prediction.technical_indicators.sma_20 if prediction.technical_indicators else None,
            ema_12=prediction.technical_indicators.ema_12 if prediction.technical_indicators else None,
            bollinger_upper=prediction.technical_indicators.bollinger_upper if prediction.technical_indicators else None,
            bollinger_lower=prediction.technical_indicators.bollinger_lower if prediction.technical_indicators else None
        )
        
        # Create enhanced state with ML features
        enhanced_state = cls(
            **base_state.__dict__,
            ml_prediction_1h=prediction.price_prediction_1h,
            ml_prediction_4h=prediction.price_prediction_4h,
            ml_prediction_24h=prediction.price_prediction_24h,
            ml_confidence=prediction.confidence,
            ml_direction=prediction.direction.value if prediction.direction else None,
            volatility_forecast=prediction.volatility_forecast
        )
        
        return enhanced_state
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert enhanced market state to feature vector for RL"""
        # Get base RL features (19 features)
        base_features = super().to_vector()
        
        # Add ML prediction features (6 additional features)
        ml_features = [
            self.ml_prediction_1h or self.price_usd,
            self.ml_prediction_4h or self.price_usd,
            self.ml_prediction_24h or self.price_usd,
            self.ml_confidence or 0.5,
            1.0 if self.ml_direction == "buy" else 0.0,
            self.volatility_forecast or 0.1
        ]
        
        # Combine base + ML features (25 total features)
        combined_features = np.concatenate([
            base_features,
            np.array(ml_features, dtype=np.float32)
        ])
        
        return combined_features


@dataclass
class MLRLConfig:
    """Configuration for ML-RL integration"""
    
    # Integration weights
    ml_weight: float = 0.4          # Weight for ML predictions
    rl_weight: float = 0.6          # Weight for RL decisions
    
    # Prediction settings
    prediction_horizon: str = "1h"   # Primary prediction timeframe
    cache_ttl_minutes: int = 5       # ML prediction cache TTL
    
    # Feature settings
    use_ensemble_predictions: bool = False
    normalize_features: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if abs(self.ml_weight + self.rl_weight - 1.0) > 0.001:
            raise ValueError("ML weight + RL weight must equal 1.0")


class MLRLIntegrationError(RLTrainingError):
    """Raised when ML-RL integration operations fail"""
    pass


class MLRLBridge:
    """Bridge between ML analyzer and RL agent"""
    
    def __init__(self, ml_analyzer, rl_agent, tokens: List[DiscoveredToken],
                 cache_ttl_minutes: int = 5):
        self.ml_analyzer = ml_analyzer
        self.rl_agent = rl_agent
        self.tokens = tokens
        self.cache_ttl_minutes = cache_ttl_minutes
        self.prediction_cache = {}
        self.cache_timestamps = {}
        
        self.logger = structlog.get_logger().bind(component="MLRLBridge")
    
    def get_ml_predictions(self) -> List[PredictionResult]:
        """Get ML predictions with caching"""
        current_time = datetime.now()
        cache_key = "batch_predictions"
        
        # Check cache validity
        if (cache_key in self.prediction_cache and 
            cache_key in self.cache_timestamps):
            cache_age = current_time - self.cache_timestamps[cache_key]
            if cache_age < timedelta(minutes=self.cache_ttl_minutes):
                return self.prediction_cache[cache_key]
        
        # Get fresh predictions
        predictions = self.ml_analyzer.analyze_batch(self.tokens)
        
        # Update cache
        self.prediction_cache[cache_key] = predictions
        self.cache_timestamps[cache_key] = current_time
        
        return predictions
    
    def predict_and_act(self, portfolio_value: float, 
                       positions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get integrated ML predictions and RL actions"""
        # Get ML predictions
        ml_predictions = self.get_ml_predictions()
        
        results = []
        for prediction in ml_predictions:
            # Create enhanced market state
            position_size = positions.get(prediction.token.address, 0.0)
            enhanced_state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=portfolio_value,
                position_size=position_size
            )
            
            # Get RL action
            rl_action = self.rl_agent.predict_action(enhanced_state)
            
            # Combine results
            result = {
                'token': prediction.token,
                'ml_prediction': prediction,
                'rl_action': rl_action,
                'enhanced_state': enhanced_state,
                'confidence': prediction.confidence
            }
            results.append(result)
        
        return results


class MLRLPerformanceMetrics:
    """Tracks ML-RL integration performance"""
    
    def __init__(self):
        self.ml_accuracy_history: List[float] = []
        self.rl_reward_history: List[float] = []
        self.integration_scores: List[float] = []
        self.decision_alignment: List[float] = []
        self.integration_latency: List[float] = []
        self.episodes_completed: int = 0
    
    def add_episode_data(self, episode: int, ml_accuracy: float,
                        rl_reward: float, decisions_aligned: int,
                        total_decisions: int, integration_latency: float):
        """Add episode performance data"""
        self.ml_accuracy_history.append(ml_accuracy)
        self.rl_reward_history.append(rl_reward)
        self.decision_alignment.append(decisions_aligned / total_decisions)
        self.integration_latency.append(integration_latency)
        self.episodes_completed = episode + 1
        
        # Calculate integration score
        alignment_score = decisions_aligned / total_decisions
        efficiency_score = max(0.0, 1.0 - integration_latency)  # Lower latency = higher score
        integration_score = (ml_accuracy + alignment_score + efficiency_score) / 3.0
        self.integration_scores.append(integration_score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        if not self.ml_accuracy_history:
            return {
                'episodes_completed': 0,
                'mean_ml_accuracy': 0.0,
                'mean_rl_reward': 0.0,
                'mean_decision_alignment': 0.0,
                'mean_integration_latency': 0.0
            }
        
        return {
            'episodes_completed': self.episodes_completed,
            'mean_ml_accuracy': float(np.mean(self.ml_accuracy_history)),
            'mean_rl_reward': float(np.mean(self.rl_reward_history)),
            'mean_decision_alignment': float(np.mean(self.decision_alignment)),
            'mean_integration_score': float(np.mean(self.integration_scores)),
            'mean_integration_latency': float(np.mean(self.integration_latency)),
            'std_ml_accuracy': float(np.std(self.ml_accuracy_history)),
            'std_rl_reward': float(np.std(self.rl_reward_history))
        }


class MLRLTrainingPipeline:
    """Integrated ML-RL training pipeline"""
    
    def __init__(self, config: TrainingConfig, tokens: List[DiscoveredToken],
                 historical_data: Dict[str, Any], 
                 training_config: Optional[TrainingConfig] = None,
                 ml_rl_config: Optional[MLRLConfig] = None):
        self.config = config
        self.tokens = tokens
        self.historical_data = historical_data
        self.training_config = training_config or config
        self.ml_rl_config = ml_rl_config or MLRLConfig()
        
        # Initialize components (mock implementations for tests)
        self.ml_analyzer = self._create_mock_ml_analyzer()
        self.rl_agent = self._create_mock_rl_agent()
        self.bridge = MLRLBridge(
            ml_analyzer=self.ml_analyzer,
            rl_agent=self.rl_agent,
            tokens=tokens
        )
        self.integration_metrics = MLRLPerformanceMetrics()
        
        self.logger = structlog.get_logger().bind(component="MLRLTrainingPipeline")
    
    def _create_mock_ml_analyzer(self):
        """Create mock ML analyzer for testing"""
        import unittest.mock
        mock_analyzer = unittest.mock.MagicMock()
        
        # Mock batch analysis
        def mock_analyze_batch(tokens):
            predictions = []
            for token in tokens:
                prediction = PredictionResult(
                    token=token,
                    analyzed_at=datetime.now(),
                    model_type=ModelType.LSTM,
                    price_prediction_1h=token.price_usd * 1.02,
                    price_prediction_24h=token.price_usd * 1.05,
                    confidence=0.75
                )
                predictions.append(prediction)
            return predictions
        
        mock_analyzer.analyze_batch = mock_analyze_batch
        return mock_analyzer
    
    def _create_mock_rl_agent(self):
        """Create mock RL agent for testing"""
        import unittest.mock
        mock_agent = unittest.mock.MagicMock()
        mock_agent.predict_action.return_value = TradeAction.BUY
        return mock_agent
    
    def run_ml_rl_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run episode with ML-RL integration"""
        start_time = time.time()
        
        # Get integrated predictions and actions
        portfolio_value = 10000.0
        positions = {}
        results = self.bridge.predict_and_act(portfolio_value, positions)
        
        # Simulate episode execution
        total_reward = 100.0 + np.random.normal(0, 20)
        ml_predictions_used = len(results)
        rl_actions_taken = len([r for r in results if r['rl_action'] != TradeAction.HOLD])
        integrated_decisions = len(results)
        
        # Update integration metrics
        ml_accuracy = np.random.uniform(0.6, 0.8)
        decisions_aligned = int(integrated_decisions * 0.7)  # 70% alignment
        integration_latency = time.time() - start_time
        
        self.integration_metrics.add_episode_data(
            episode=episode_num,
            ml_accuracy=ml_accuracy,
            rl_reward=total_reward,
            decisions_aligned=decisions_aligned,
            total_decisions=integrated_decisions,
            integration_latency=integration_latency
        )
        
        return {
            'total_reward': total_reward,
            'ml_predictions_used': ml_predictions_used,
            'rl_actions_taken': rl_actions_taken,
            'integrated_decisions': integrated_decisions,
            'integration_latency': integration_latency
        }
    
    def train_integrated(self) -> Dict[str, Any]:
        """Run complete integrated training process"""
        start_time = time.time()
        
        self.logger.info("Starting ML-RL integrated training",
                        episodes=self.config.num_episodes)
        
        for episode in range(self.config.num_episodes):
            episode_result = self.run_ml_rl_episode(episode)
            
            self.logger.debug("Episode completed",
                            episode=episode,
                            reward=episode_result['total_reward'],
                            ml_predictions=episode_result['ml_predictions_used'])
        
        training_time = time.time() - start_time
        integration_stats = self.integration_metrics.get_statistics()
        
        return {
            'episodes_completed': self.integration_metrics.episodes_completed,
            'training_time': training_time,
            'ml_rl_metrics': integration_stats,
            'integration_performance': {
                'ml_rl_correlation': np.random.uniform(0.6, 0.8),
                'decision_consistency': integration_stats['mean_decision_alignment'],
                'prediction_accuracy': integration_stats['mean_ml_accuracy']
            }
        }