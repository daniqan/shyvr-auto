"""
Ensemble Weight Manager
Manages sentiment-based weight optimization for transformer ensemble models
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import structlog
from dataclasses import dataclass
from collections import defaultdict

from .base import ModelType
from .market_data import FearGreedIndexClient, MarketSentimentData
from src.activity_logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)


logger = structlog.get_logger()


@dataclass
class SentimentWeightConfig:
    """Configuration for sentiment-based weight adjustments"""
    extreme_fear_threshold: float = 15.0
    fear_threshold: float = 30.0
    cautious_threshold: float = 40.0
    neutral_min: float = 40.0
    neutral_max: float = 60.0
    optimistic_threshold: float = 70.0
    greed_threshold: float = 85.0
    
    # Weight adjustment multipliers
    extreme_adjustments: Dict[ModelType, float] = None
    fear_adjustments: Dict[ModelType, float] = None
    neutral_adjustments: Dict[ModelType, float] = None
    greed_adjustments: Dict[ModelType, float] = None
    
    def __post_init__(self):
        if self.extreme_adjustments is None:
            self.extreme_adjustments = {
                ModelType.LSTM: 1.4,  # Conservative models preferred in extreme fear
                ModelType.TRANSFORMER: 0.8,
                ModelType.ITRANSFORMER: 0.8,
                ModelType.PATCHTST: 0.8,
                ModelType.TIMESMIXER: 1.3,
                ModelType.TIMESFM: 0.8
            }
        
        if self.fear_adjustments is None:
            self.fear_adjustments = {
                ModelType.LSTM: 1.2,
                ModelType.TRANSFORMER: 1.0,
                ModelType.ITRANSFORMER: 1.1,
                ModelType.PATCHTST: 1.0,
                ModelType.TIMESMIXER: 1.0,
                ModelType.TIMESFM: 1.0
            }
        
        if self.neutral_adjustments is None:
            self.neutral_adjustments = {
                ModelType.LSTM: 1.0,
                ModelType.TRANSFORMER: 1.0,
                ModelType.ITRANSFORMER: 1.0,
                ModelType.PATCHTST: 1.0,
                ModelType.TIMESMIXER: 1.0,
                ModelType.TIMESFM: 1.0
            }
        
        if self.greed_adjustments is None:
            self.greed_adjustments = {
                ModelType.LSTM: 0.7,  # Aggressive models preferred in extreme greed
                ModelType.TRANSFORMER: 1.4,
                ModelType.ITRANSFORMER: 1.4,
                ModelType.PATCHTST: 1.4,
                ModelType.TIMESMIXER: 1.1,
                ModelType.TIMESFM: 1.4
            }


@dataclass
class SentimentRegimeAnalysis:
    """Analysis of sentiment regime patterns"""
    regime: str
    confidence: float
    duration_minutes: int
    volatility_level: str
    predicted_transition: Optional[str]
    historical_performance: Dict[ModelType, float]
    optimal_weights: Dict[ModelType, float]


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization process"""
    original_weights: Dict[ModelType, float]
    optimized_weights: Dict[ModelType, float]
    sentiment_regime: str
    optimization_confidence: float
    expected_improvement: float
    risk_adjustment: float
    processing_time_ms: float


class EnsembleWeightManager:
    """
    Manages sentiment-based weight optimization for ensemble models
    Provides sophisticated algorithms for weight adjustment based on market sentiment
    """
    
    def __init__(self, config: Optional[SentimentWeightConfig] = None):
        self.config = config or SentimentWeightConfig()
        self.logger = structlog.get_logger().bind(component="EnsembleWeightManager")
        
        # Fear & Greed client
        self._fear_greed_client = FearGreedIndexClient()
        
        # Historical sentiment tracking
        self._sentiment_history: List[Tuple[datetime, str, float]] = []
        self._max_history_hours = 72  # 3 days of sentiment history
        
        # Performance tracking per sentiment regime
        self._regime_performance: Dict[str, Dict[ModelType, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Weight optimization cache
        self._optimization_cache: Dict[str, Tuple[datetime, WeightOptimizationResult]] = {}
        self._cache_ttl_minutes = 5  # Cache optimizations for 5 minutes
        
        # Regime transition detection
        self._regime_transitions: List[Tuple[datetime, str, str]] = []  # (timestamp, from_regime, to_regime)
        self._transition_smoothing_minutes = 15  # Smooth transitions over 15 minutes
        
    async def optimize_weights(self, 
                              current_weights: Dict[ModelType, float],
                              market_volatility: float = 0.5,
                              confidence_threshold: float = 0.7) -> WeightOptimizationResult:
        """
        Optimize ensemble weights based on current market sentiment
        
        Args:
            current_weights: Current model weights
            market_volatility: Current market volatility (0.0-1.0)
            confidence_threshold: Minimum confidence for optimization
            
        Returns:
            WeightOptimizationResult with optimized weights
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{hash(tuple(sorted(current_weights.items())))}{market_volatility}"
            if cache_key in self._optimization_cache:
                cached_time, cached_result = self._optimization_cache[cache_key]
                if (start_time - cached_time).total_seconds() < self._cache_ttl_minutes * 60:
                    self.logger.debug("Returning cached weight optimization")
                    return cached_result
            
            # Get current sentiment regime
            sentiment_analysis = await self._analyze_sentiment_regime()
            
            # Calculate base sentiment adjustments
            sentiment_weights = self._calculate_sentiment_weights(
                current_weights, 
                sentiment_analysis.regime,
                sentiment_analysis.confidence
            )
            
            # Apply volatility adjustments
            volatility_weights = self._apply_volatility_adjustments(
                sentiment_weights, 
                market_volatility
            )
            
            # Apply historical performance adjustments
            performance_weights = await self._apply_performance_adjustments(
                volatility_weights,
                sentiment_analysis.regime
            )
            
            # Smooth regime transitions
            final_weights = await self._smooth_regime_transitions(
                performance_weights,
                current_weights
            )
            
            # Calculate optimization confidence and expected improvement
            optimization_confidence = self._calculate_optimization_confidence(
                sentiment_analysis,
                market_volatility
            )
            
            expected_improvement = self._estimate_improvement(
                current_weights,
                final_weights,
                sentiment_analysis
            )
            
            # Risk adjustment factor
            risk_adjustment = self._calculate_risk_adjustment(
                sentiment_analysis.regime,
                market_volatility
            )
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = WeightOptimizationResult(
                original_weights=current_weights.copy(),
                optimized_weights=final_weights,
                sentiment_regime=sentiment_analysis.regime,
                optimization_confidence=optimization_confidence,
                expected_improvement=expected_improvement,
                risk_adjustment=risk_adjustment,
                processing_time_ms=processing_time_ms
            )
            
            # Cache the result
            self._optimization_cache[cache_key] = (start_time, result)
            
            # Log optimization
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.EXECUTE,
                source="ensemble_weight_manager",
                event_type="weight_optimization",
                title=f"Ensemble weights optimized for {sentiment_analysis.regime} sentiment",
                severity=ActivitySeverity.INFO,
                metadata={
                    "sentiment_regime": sentiment_analysis.regime,
                    "optimization_confidence": optimization_confidence,
                    "expected_improvement": expected_improvement,
                    "processing_time_ms": processing_time_ms,
                    "original_weights": current_weights,
                    "optimized_weights": final_weights
                }
            )
            
            self.logger.info("Weight optimization completed",
                           sentiment_regime=sentiment_analysis.regime,
                           confidence=optimization_confidence,
                           improvement=expected_improvement,
                           processing_time_ms=processing_time_ms)
            
            return result
            
        except Exception as e:
            self.logger.error("Weight optimization failed", error=str(e))
            # Return original weights on failure
            return WeightOptimizationResult(
                original_weights=current_weights.copy(),
                optimized_weights=current_weights.copy(),
                sentiment_regime="neutral",
                optimization_confidence=0.0,
                expected_improvement=0.0,
                risk_adjustment=1.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _analyze_sentiment_regime(self) -> SentimentRegimeAnalysis:
        """Analyze current sentiment regime with confidence scoring"""
        try:
            # Get current sentiment data
            sentiment_data = await self._fear_greed_client.get_market_data()
            fear_greed_value = sentiment_data.fear_greed_index
            
            # Classify regime
            regime = self._classify_sentiment_regime(fear_greed_value)
            
            # Update sentiment history
            current_time = datetime.now()
            self._sentiment_history.append((current_time, regime, fear_greed_value))
            
            # Clean old history
            cutoff_time = current_time - timedelta(hours=self._max_history_hours)
            self._sentiment_history = [
                (ts, r, v) for ts, r, v in self._sentiment_history 
                if ts > cutoff_time
            ]
            
            # Calculate confidence based on recent stability
            confidence = self._calculate_regime_confidence()
            
            # Estimate regime duration
            duration_minutes = self._estimate_regime_duration(regime)
            
            # Predict potential transitions
            predicted_transition = self._predict_regime_transition(regime, fear_greed_value)
            
            # Get historical performance for this regime
            historical_performance = self._get_regime_performance(regime)
            
            # Calculate optimal weights for this regime
            optimal_weights = self._calculate_optimal_regime_weights(regime, historical_performance)
            
            return SentimentRegimeAnalysis(
                regime=regime,
                confidence=confidence,
                duration_minutes=duration_minutes,
                volatility_level=sentiment_data.volatility_regime,
                predicted_transition=predicted_transition,
                historical_performance=historical_performance,
                optimal_weights=optimal_weights
            )
            
        except Exception as e:
            self.logger.error("Sentiment regime analysis failed", error=str(e))
            # Return neutral analysis on failure
            return SentimentRegimeAnalysis(
                regime="neutral",
                confidence=0.5,
                duration_minutes=30,
                volatility_level="medium",
                predicted_transition=None,
                historical_performance={},
                optimal_weights={}
            )
    
    def _classify_sentiment_regime(self, fear_greed_value: float) -> str:
        """Classify sentiment into refined regimes"""
        if fear_greed_value <= self.config.extreme_fear_threshold:
            return "extreme_fear"
        elif fear_greed_value <= self.config.fear_threshold:
            return "fear"
        elif fear_greed_value <= self.config.cautious_threshold:
            return "cautious"
        elif fear_greed_value <= self.config.neutral_max:
            return "neutral"
        elif fear_greed_value <= self.config.optimistic_threshold:
            return "optimistic"
        elif fear_greed_value <= self.config.greed_threshold:
            return "greed"
        else:
            return "extreme_greed"
    
    def _calculate_sentiment_weights(self, 
                                   current_weights: Dict[ModelType, float],
                                   regime: str,
                                   confidence: float) -> Dict[ModelType, float]:
        """Calculate base sentiment-adjusted weights (environment-aware)"""
        # Handle single-model case (development environment)
        if len(current_weights) == 1:
            model_type = next(iter(current_weights.keys()))
            self.logger.debug("Single model detected, maintaining weight", 
                            model_type=model_type.value)
            return {model_type: 1.0}
        
        adjustment_map = {
            "extreme_fear": self.config.extreme_adjustments,
            "fear": self.config.fear_adjustments,
            "cautious": self.config.fear_adjustments,  # Use fear adjustments for cautious
            "neutral": self.config.neutral_adjustments,
            "optimistic": self.config.greed_adjustments,  # Use greed adjustments for optimistic
            "greed": self.config.greed_adjustments,
            "extreme_greed": self.config.greed_adjustments
        }
        
        adjustments = adjustment_map.get(regime, self.config.neutral_adjustments)
        
        # Apply confidence weighting - lower confidence means less extreme adjustments
        adjusted_weights = {}
        for model_type, weight in current_weights.items():
            adjustment = adjustments.get(model_type, 1.0)
            # Blend adjustment with neutral (1.0) based on confidence
            blended_adjustment = 1.0 + (adjustment - 1.0) * confidence
            adjusted_weights[model_type] = weight * blended_adjustment
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _apply_volatility_adjustments(self, 
                                    weights: Dict[ModelType, float],
                                    volatility: float) -> Dict[ModelType, float]:
        """Apply volatility-based adjustments to weights (environment-aware)"""
        # Handle single-model case (development environment)
        if len(weights) == 1:
            model_type = next(iter(weights.keys()))
            self.logger.debug("Single model detected, skipping volatility adjustment", 
                            model_type=model_type.value)
            return {model_type: 1.0}
            
        # High volatility favors models that handle volatility well
        volatility_preferences = {
            ModelType.LSTM: max(0.7, 1.0 - volatility * 0.4),  # LSTM struggles with high volatility
            ModelType.TRANSFORMER: 1.0 + volatility * 0.2,  # Transformers handle volatility better
            ModelType.ITRANSFORMER: 1.0 + volatility * 0.3,  # Best for volatility clustering
            ModelType.PATCHTST: 1.0,  # Stable under volatility
            ModelType.TIMESMIXER: 1.0 + volatility * 0.4,  # Excellent volatility decomposition
            ModelType.TIMESFM: 1.0 + volatility * 0.2  # Foundation model handles volatility well
        }
        
        adjusted_weights = {}
        for model_type, weight in weights.items():
            preference = volatility_preferences.get(model_type, 1.0)
            adjusted_weights[model_type] = weight * preference
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    async def _apply_performance_adjustments(self, 
                                           weights: Dict[ModelType, float],
                                           regime: str) -> Dict[ModelType, float]:
        """Apply historical performance adjustments for the current regime (environment-aware)"""
        try:
            # Handle single-model case (development environment)
            if len(weights) == 1:
                model_type = next(iter(weights.keys()))
                self.logger.debug("Single model detected, skipping performance adjustment", 
                                model_type=model_type.value)
                return {model_type: 1.0}
                
            performance_data = self._regime_performance.get(regime, {})
            
            if not performance_data:
                return weights  # No historical data, return as-is
            
            # Calculate performance multipliers based on historical success
            performance_multipliers = {}
            for model_type in weights.keys():
                historical_scores = performance_data.get(model_type, [])
                if historical_scores:
                    avg_performance = np.mean(historical_scores)
                    # Convert to multiplier (0.5x to 1.5x based on performance)
                    multiplier = 0.5 + avg_performance
                    performance_multipliers[model_type] = multiplier
                else:
                    performance_multipliers[model_type] = 1.0
            
            # Apply performance adjustments
            adjusted_weights = {}
            for model_type, weight in weights.items():
                multiplier = performance_multipliers.get(model_type, 1.0)
                adjusted_weights[model_type] = weight * multiplier
            
            # Normalize weights
            total = sum(adjusted_weights.values())
            if total > 0:
                adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.warning("Performance adjustment failed", error=str(e))
            return weights
    
    async def _smooth_regime_transitions(self, 
                                       target_weights: Dict[ModelType, float],
                                       current_weights: Dict[ModelType, float]) -> Dict[ModelType, float]:
        """Smooth weight transitions to prevent abrupt changes (environment-aware)"""
        try:
            # Handle single-model case (development environment)
            if len(target_weights) == 1 or len(current_weights) == 1:
                model_type = next(iter(target_weights.keys()))
                self.logger.debug("Single model detected, skipping transition smoothing", 
                                model_type=model_type.value)
                return {model_type: 1.0}
                
            # Check if we're in a regime transition period
            current_time = datetime.now()
            recent_transitions = [
                (ts, from_r, to_r) for ts, from_r, to_r in self._regime_transitions
                if (current_time - ts).total_seconds() < self._transition_smoothing_minutes * 60
            ]
            
            if not recent_transitions:
                return target_weights  # No recent transitions, use target weights
            
            # Calculate smoothing factor based on time since transition
            latest_transition = recent_transitions[-1]
            time_since_transition = (current_time - latest_transition[0]).total_seconds() / 60  # minutes
            
            # Smoothing factor: 0.0 (use current) to 1.0 (use target) over transition period
            smoothing_factor = min(1.0, time_since_transition / self._transition_smoothing_minutes)
            
            # Blend current and target weights
            smoothed_weights = {}
            for model_type in current_weights.keys():
                current_weight = current_weights.get(model_type, 0.0)
                target_weight = target_weights.get(model_type, 0.0)
                smoothed_weight = current_weight + (target_weight - current_weight) * smoothing_factor
                smoothed_weights[model_type] = smoothed_weight
            
            # Normalize weights
            total = sum(smoothed_weights.values())
            if total > 0:
                smoothed_weights = {k: v / total for k, v in smoothed_weights.items()}
            
            return smoothed_weights
            
        except Exception as e:
            self.logger.warning("Transition smoothing failed", error=str(e))
            return target_weights
    
    def _calculate_regime_confidence(self) -> float:
        """Calculate confidence in current regime classification"""
        if len(self._sentiment_history) < 3:
            return 0.5  # Low confidence with insufficient data
        
        # Look at recent sentiment values (last hour)
        current_time = datetime.now()
        recent_history = [
            (ts, regime, value) for ts, regime, value in self._sentiment_history
            if (current_time - ts).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_history) < 2:
            return 0.5
        
        # Calculate regime stability (how consistent the regime has been)
        regimes = [regime for _, regime, _ in recent_history]
        most_common_regime = max(set(regimes), key=regimes.count)
        stability = regimes.count(most_common_regime) / len(regimes)
        
        # Calculate value consistency (how close the values are)
        values = [value for _, _, value in recent_history]
        value_std = np.std(values)
        value_consistency = max(0.0, 1.0 - value_std / 25.0)  # Normalize by typical volatility
        
        # Combine stability and consistency
        confidence = (stability * 0.7 + value_consistency * 0.3)
        return min(1.0, max(0.1, confidence))
    
    def _estimate_regime_duration(self, current_regime: str) -> int:
        """Estimate how long the current regime might last (in minutes)"""
        if len(self._sentiment_history) < 5:
            return 30  # Default assumption
        
        # Find the start of the current regime
        current_time = datetime.now()
        regime_start = current_time
        
        for i in range(len(self._sentiment_history) - 1, -1, -1):
            ts, regime, _ = self._sentiment_history[i]
            if regime == current_regime:
                regime_start = ts
            else:
                break
        
        # Current regime duration
        current_duration = (current_time - regime_start).total_seconds() / 60
        
        # Historical average for this regime type
        historical_durations = []
        in_regime = False
        regime_start_ts = None
        
        for ts, regime, _ in self._sentiment_history:
            if regime == current_regime and not in_regime:
                in_regime = True
                regime_start_ts = ts
            elif regime != current_regime and in_regime:
                in_regime = False
                if regime_start_ts:
                    duration = (ts - regime_start_ts).total_seconds() / 60
                    historical_durations.append(duration)
        
        if historical_durations:
            avg_duration = np.mean(historical_durations)
            remaining_duration = max(0, avg_duration - current_duration)
            return int(remaining_duration)
        else:
            # Default estimates based on regime type
            default_durations = {
                "extreme_fear": 120,  # 2 hours
                "fear": 180,  # 3 hours
                "cautious": 240,  # 4 hours
                "neutral": 360,  # 6 hours
                "optimistic": 240,  # 4 hours
                "greed": 180,  # 3 hours
                "extreme_greed": 120  # 2 hours
            }
            return default_durations.get(current_regime, 180)
    
    def _predict_regime_transition(self, current_regime: str, current_value: float) -> Optional[str]:
        """Predict likely next regime transition"""
        # Simple prediction based on proximity to thresholds
        if current_regime == "extreme_fear" and current_value > 12:
            return "fear"
        elif current_regime == "fear" and current_value > 28:
            return "cautious"
        elif current_regime == "cautious" and current_value > 38:
            return "neutral"
        elif current_regime == "neutral":
            if current_value < 42:
                return "cautious"
            elif current_value > 58:
                return "optimistic"
        elif current_regime == "optimistic" and current_value > 68:
            return "greed"
        elif current_regime == "greed" and current_value > 83:
            return "extreme_greed"
        elif current_regime == "extreme_greed" and current_value < 87:
            return "greed"
        
        return None
    
    def _get_regime_performance(self, regime: str) -> Dict[ModelType, float]:
        """Get historical performance data for a regime"""
        performance_data = self._regime_performance.get(regime, {})
        avg_performance = {}
        
        for model_type, scores in performance_data.items():
            if scores:
                avg_performance[model_type] = np.mean(scores)
            else:
                avg_performance[model_type] = 0.5  # Neutral performance
        
        return avg_performance
    
    def _calculate_optimal_regime_weights(self, 
                                        regime: str, 
                                        historical_performance: Dict[ModelType, float]) -> Dict[ModelType, float]:
        """Calculate theoretically optimal weights for a regime"""
        if not historical_performance:
            # Return equal weights if no performance data
            model_types = [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.ITRANSFORMER, 
                          ModelType.PATCHTST, ModelType.TIMESMIXER, ModelType.TIMESFM]
            return {mt: 1.0 / len(model_types) for mt in model_types}
        
        # Use performance scores as weights (higher performance = higher weight)
        total_performance = sum(historical_performance.values())
        if total_performance > 0:
            optimal_weights = {
                model_type: performance / total_performance
                for model_type, performance in historical_performance.items()
            }
        else:
            # Fallback to equal weights
            optimal_weights = {
                model_type: 1.0 / len(historical_performance)
                for model_type in historical_performance.keys()
            }
        
        return optimal_weights
    
    def _calculate_optimization_confidence(self, 
                                         sentiment_analysis: SentimentRegimeAnalysis,
                                         market_volatility: float) -> float:
        """Calculate confidence in the optimization"""
        # Base confidence from sentiment analysis
        base_confidence = sentiment_analysis.confidence
        
        # Reduce confidence in high volatility conditions
        volatility_penalty = min(0.3, market_volatility * 0.5)
        
        # Boost confidence if we have good historical data
        historical_boost = 0.0
        if sentiment_analysis.historical_performance:
            performance_count = sum(1 for scores in sentiment_analysis.historical_performance.values() if scores)
            historical_boost = min(0.2, performance_count * 0.05)
        
        confidence = base_confidence - volatility_penalty + historical_boost
        return min(1.0, max(0.1, confidence))
    
    def _estimate_improvement(self, 
                            original_weights: Dict[ModelType, float],
                            optimized_weights: Dict[ModelType, float],
                            sentiment_analysis: SentimentRegimeAnalysis) -> float:
        """Estimate expected improvement from optimization"""
        try:
            # Calculate weighted change in allocations
            total_change = 0.0
            for model_type in original_weights.keys():
                original = original_weights.get(model_type, 0.0)
                optimized = optimized_weights.get(model_type, 0.0)
                change = abs(optimized - original)
                total_change += change
            
            # Scale by historical performance differential
            performance_differential = 0.0
            if sentiment_analysis.historical_performance:
                performances = list(sentiment_analysis.historical_performance.values())
                if performances:
                    performance_differential = max(performances) - min(performances)
            
            # Expected improvement is proportional to change magnitude and performance differential
            expected_improvement = total_change * performance_differential
            return min(0.5, expected_improvement)  # Cap at 50% improvement
            
        except Exception as e:
            self.logger.warning("Improvement estimation failed", error=str(e))
            return 0.05  # Conservative default
    
    def _calculate_risk_adjustment(self, regime: str, volatility: float) -> float:
        """Calculate risk adjustment factor"""
        # Higher risk in extreme regimes and high volatility
        regime_risk = {
            "extreme_fear": 1.3,
            "fear": 1.1,
            "cautious": 1.05,
            "neutral": 1.0,
            "optimistic": 1.05,
            "greed": 1.1,
            "extreme_greed": 1.3
        }
        
        base_risk = regime_risk.get(regime, 1.0)
        volatility_risk = 1.0 + volatility * 0.3
        
        return base_risk * volatility_risk
    
    async def record_performance(self, 
                               model_type: ModelType, 
                               performance_score: float,
                               sentiment_regime: str):
        """Record model performance for a specific sentiment regime"""
        try:
            self._regime_performance[sentiment_regime][model_type].append(performance_score)
            
            # Keep only recent performance data (last 100 scores per regime/model)
            if len(self._regime_performance[sentiment_regime][model_type]) > 100:
                self._regime_performance[sentiment_regime][model_type] = \
                    self._regime_performance[sentiment_regime][model_type][-100:]
            
            self.logger.debug("Performance recorded",
                            model_type=model_type.value,
                            regime=sentiment_regime,
                            score=performance_score)
            
        except Exception as e:
            self.logger.error("Failed to record performance", error=str(e))
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about sentiment regimes and performance"""
        try:
            current_time = datetime.now()
            
            # Regime distribution over time
            regime_counts = defaultdict(int)
            for _, regime, _ in self._sentiment_history:
                regime_counts[regime] += 1
            
            # Performance statistics per regime
            performance_stats = {}
            for regime, model_performance in self._regime_performance.items():
                regime_stats = {}
                for model_type, scores in model_performance.items():
                    if scores:
                        regime_stats[model_type.value] = {
                            "count": len(scores),
                            "mean": float(np.mean(scores)),
                            "std": float(np.std(scores)),
                            "min": float(np.min(scores)),
                            "max": float(np.max(scores))
                        }
                performance_stats[regime] = regime_stats
            
            return {
                "sentiment_history_length": len(self._sentiment_history),
                "regime_distribution": dict(regime_counts),
                "performance_statistics": performance_stats,
                "cache_size": len(self._optimization_cache),
                "transition_history": len(self._regime_transitions),
                "last_updated": current_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to generate regime statistics", error=str(e))
            return {"error": str(e)}