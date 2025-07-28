"""
XAI Trading Integration Bridge.

This module provides integration between the XAI explanation system
and trading modes, enabling real-time explanation of trading decisions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict

from .factory import ExplainerFactory
from .data_models import ExplanationData
from ..monitoring.base import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class TradingExplanation:
    """Container for trading decision explanations."""
    
    decision_id: str
    timestamp: str
    decision_type: str  # 'buy', 'sell', 'hold'
    symbol: str
    explanation_data: ExplanationData
    model_type: str  # 'ml_model', 'rl_agent'
    confidence: float
    metadata: Dict[str, Any]


class TradingExplanationManager:
    """
    Manages XAI explanations for trading decisions.
    
    Coordinates between trading modes and XAI explainers to provide
    real-time explanations of trading decisions with caching and storage.
    """
    
    def __init__(
        self,
        explainer_factory: Optional[ExplainerFactory] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        cache_size: int = 1000,
        explanation_timeout: float = 5.0
    ):
        """
        Initialize trading explanation manager.
        
        Args:
            explainer_factory: Factory for creating explainers
            metrics_collector: Metrics collector for monitoring
            cache_size: Maximum number of explanations to cache
            explanation_timeout: Timeout for explanation generation
        """
        self.explainer_factory = explainer_factory or ExplainerFactory()
        self.metrics_collector = metrics_collector
        self.cache_size = cache_size
        self.explanation_timeout = explanation_timeout
        
        # Explanation cache and storage
        self._explanation_cache: Dict[str, TradingExplanation] = {}
        self._explainer_cache: Dict[str, Any] = {}
        
        # Configuration
        self._enabled = True
        self._default_explainer_type = 'permutation'
        self._fallback_explainer_types = ['lime', 'gradient']
        
        logger.info(f"Initialized TradingExplanationManager with cache_size={cache_size}")
    
    async def explain_trading_decision(
        self,
        decision_id: str,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        decision_type: str,
        symbol: str,
        model_type: str = 'ml_model',
        explainer_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TradingExplanation]:
        """
        Generate explanation for a trading decision.
        
        Args:
            decision_id: Unique identifier for the decision
            model: The ML/RL model that made the decision
            feature_data: Input features used for the decision
            feature_names: Names of the features
            decision_type: Type of decision ('buy', 'sell', 'hold')
            symbol: Trading symbol
            model_type: Type of model ('ml_model', 'rl_agent')
            explainer_type: Specific explainer to use
            metadata: Additional metadata
            
        Returns:
            TradingExplanation object or None if explanation failed
        """
        if not self._enabled:
            return None
        
        try:
            # Check cache first
            if decision_id in self._explanation_cache:
                logger.debug(f"Using cached explanation for decision {decision_id}")
                return self._explanation_cache[decision_id]
            
            # Generate explanation with timeout
            explanation_data = await asyncio.wait_for(
                self._generate_explanation(
                    model=model,
                    feature_data=feature_data,
                    feature_names=feature_names,
                    explainer_type=explainer_type or self._default_explainer_type
                ),
                timeout=self.explanation_timeout
            )
            
            if explanation_data is None:
                logger.warning(f"Failed to generate explanation for decision {decision_id}")
                return None
            
            # Create trading explanation
            trading_explanation = TradingExplanation(
                decision_id=decision_id,
                timestamp=datetime.utcnow().isoformat() + 'Z',
                decision_type=decision_type,
                symbol=symbol,
                explanation_data=explanation_data,
                model_type=model_type,
                confidence=explanation_data.confidence_score or 0.5,
                metadata=metadata or {}
            )
            
            # Cache the explanation
            self._cache_explanation(decision_id, trading_explanation)
            
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter(
                    'xai_explanations_generated_total',
                    {'explainer_type': explanation_data.explanation_type, 'symbol': symbol}
                )
            
            logger.info(f"Generated {explanation_data.explanation_type} explanation for {decision_type} decision on {symbol}")
            return trading_explanation
            
        except asyncio.TimeoutError:
            logger.warning(f"Explanation generation timed out for decision {decision_id}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter('xai_explanation_timeouts_total')
            return None
            
        except Exception as e:
            logger.error(f"Error generating explanation for decision {decision_id}: {str(e)}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter('xai_explanation_errors_total')
            return None
    
    async def _generate_explanation(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        explainer_type: str
    ) -> Optional[ExplanationData]:
        """
        Generate explanation using specified explainer type with fallbacks.
        
        Args:
            model: The model to explain
            feature_data: Input features
            feature_names: Feature names
            explainer_type: Type of explainer to use
            
        Returns:
            ExplanationData or None if all explainers fail
        """
        explainer_types_to_try = [explainer_type] + [
            t for t in self._fallback_explainer_types if t != explainer_type
        ]
        
        for exp_type in explainer_types_to_try:
            try:
                if not self.explainer_factory.is_supported(exp_type):
                    continue
                
                # Get or create explainer
                explainer_key = f"{exp_type}_{id(model)}_{hash(tuple(feature_names))}"
                if explainer_key not in self._explainer_cache:
                    explainer = self.explainer_factory.create_explainer(
                        explainer_type=exp_type,
                        model=model,
                        feature_names=feature_names
                    )
                    self._explainer_cache[explainer_key] = explainer
                else:
                    explainer = self._explainer_cache[explainer_key]
                
                # Generate explanation
                explanation_data = explainer.explain_instance(feature_data)
                logger.debug(f"Successfully generated {exp_type} explanation")
                return explanation_data
                
            except Exception as e:
                logger.warning(f"Failed to generate {exp_type} explanation: {str(e)}")
                continue
        
        logger.error("All explainer types failed to generate explanation")
        return None
    
    def _cache_explanation(self, decision_id: str, explanation: TradingExplanation) -> None:
        """Cache explanation with size limit."""
        if len(self._explanation_cache) >= self.cache_size:
            # Remove oldest explanation
            oldest_key = min(
                self._explanation_cache.keys(),
                key=lambda k: self._explanation_cache[k].timestamp
            )
            del self._explanation_cache[oldest_key]
        
        self._explanation_cache[decision_id] = explanation
    
    def get_explanation(self, decision_id: str) -> Optional[TradingExplanation]:
        """Get cached explanation by decision ID."""
        return self._explanation_cache.get(decision_id)
    
    def get_recent_explanations(
        self,
        symbol: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 100
    ) -> List[TradingExplanation]:
        """
        Get recent explanations with optional filtering.
        
        Args:
            symbol: Filter by trading symbol
            decision_type: Filter by decision type
            limit: Maximum number of explanations to return
            
        Returns:
            List of recent trading explanations
        """
        explanations = list(self._explanation_cache.values())
        
        # Apply filters
        if symbol:
            explanations = [e for e in explanations if e.symbol == symbol]
        if decision_type:
            explanations = [e for e in explanations if e.decision_type == decision_type]
        
        # Sort by timestamp (newest first) and limit
        explanations.sort(key=lambda e: e.timestamp, reverse=True)
        return explanations[:limit]
    
    def get_feature_importance_summary(
        self,
        symbol: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, float]:
        """
        Get aggregated feature importance across recent decisions.
        
        Args:
            symbol: Filter by trading symbol
            hours_back: Number of hours to look back
            
        Returns:
            Dictionary of feature names to aggregated importance scores
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        cutoff_iso = cutoff_time.isoformat() + 'Z'
        
        # Filter explanations
        explanations = [
            e for e in self._explanation_cache.values()
            if e.timestamp >= cutoff_iso and (not symbol or e.symbol == symbol)
        ]
        
        if not explanations:
            return {}
        
        # Aggregate feature importance
        feature_sums: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}
        
        for explanation in explanations:
            for feature, importance in explanation.explanation_data.feature_importance.items():
                feature_sums[feature] = feature_sums.get(feature, 0.0) + abs(importance)
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Calculate averages
        return {
            feature: feature_sums[feature] / feature_counts[feature]
            for feature in feature_sums
        }
    
    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._explanation_cache.clear()
        self._explainer_cache.clear()
        logger.info("Cleared explanation cache")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable explanation generation."""
        self._enabled = enabled
        logger.info(f"XAI explanation generation {'enabled' if enabled else 'disabled'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'explanation_cache_size': len(self._explanation_cache),
            'explainer_cache_size': len(self._explainer_cache),
            'cache_limit': self.cache_size,
            'enabled': self._enabled,
            'default_explainer_type': self._default_explainer_type
        }
    
    def to_dict(self, explanation: TradingExplanation) -> Dict[str, Any]:
        """Convert TradingExplanation to dictionary for serialization."""
        result = asdict(explanation)
        result['explanation_data'] = explanation.explanation_data.to_dict()
        return result