"""
XAI Trading Integration Bridge.

This module provides integration between the XAI explanation system
and trading modes, enabling real-time explanation of trading decisions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
from dataclasses import dataclass, asdict
import torch
from torch import Tensor

from .factory import ExplainerFactory
from .data_models import ExplanationData
from ..monitoring.base import MetricsCollector
from .transformers.attention_explainer import AttentionExplainer
from .transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
from .transformers.cross_attention_analyzer import CrossAttentionAnalyzer

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
    attention_data: Optional[Dict[str, Any]] = None


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
        
        # Transformer-specific configuration
        self._transformer_explainer_types = ['attention', 'temporal_attention', 'cross_attention']
        self._attention_cache = {}  # Cache for attention explainers
        self._enable_attention_heatmaps = True
        self._attention_cache_size = 100
        
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
        metadata: Optional[Dict[str, Any]] = None,
        enable_attention_analysis: bool = True
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
                    explainer_type=explainer_type or self._default_explainer_type,
                    enable_attention_analysis=enable_attention_analysis
                ),
                timeout=self.explanation_timeout
            )
            
            if explanation_data is None:
                logger.warning(f"Failed to generate explanation for decision {decision_id}")
                return None
            
            # Generate attention data for transformer models if enabled
            attention_data = None
            if enable_attention_analysis and self._is_transformer_model(model):
                attention_data = await self._generate_attention_analysis(
                    model=model,
                    feature_data=feature_data,
                    feature_names=feature_names,
                    symbol=symbol
                )
            
            # Create trading explanation
            trading_explanation = TradingExplanation(
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                decision_type=decision_type,
                symbol=symbol,
                explanation_data=explanation_data,
                model_type=model_type,
                confidence=explanation_data.confidence_score or 0.5,
                metadata=metadata or {},
                attention_data=attention_data
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
        explainer_type: str,
        enable_attention_analysis: bool = True
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
    
    def _is_transformer_model(self, model: Any) -> bool:
        """Check if model is a transformer model that supports attention analysis."""
        model_type = getattr(model, 'model_type', None)
        if model_type:
            return model_type.lower() in ['itransformer', 'patchtst', 'timesmixer', 'timesfm', 'transformerpredictor']
        
        # Check model class name as fallback
        model_class_name = model.__class__.__name__.lower()
        transformer_indicators = ['transformer', 'attention', 'itransformer', 'patchtst', 'timesmixer', 'timesfm']
        return any(indicator in model_class_name for indicator in transformer_indicators)
    
    async def _generate_attention_analysis(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Generate attention-based analysis for transformer models."""
        try:
            attention_data = {}
            
            # Basic attention weight extraction
            if hasattr(model, 'get_attention_weights') or hasattr(model, 'attention_weights'):
                attention_weights = await self._extract_attention_weights(model, feature_data)
                if attention_weights is not None:
                    attention_data['attention_weights'] = attention_weights
                    attention_data['attention_heatmap'] = self._generate_attention_heatmap(attention_weights)
            
            # Temporal attention analysis for time-series patterns
            temporal_patterns = await self._analyze_temporal_attention(
                model, feature_data, feature_names
            )
            if temporal_patterns:
                attention_data['temporal_patterns'] = temporal_patterns
            
            # Cross-asset attention analysis (if applicable)
            if self._has_multi_asset_data(feature_names):
                cross_asset_patterns = await self._analyze_cross_asset_attention(
                    model, feature_data, feature_names, symbol
                )
                if cross_asset_patterns:
                    attention_data['cross_asset_patterns'] = cross_asset_patterns
            
            return attention_data if attention_data else None
            
        except Exception as e:
            logger.warning(f"Failed to generate attention analysis: {str(e)}")
            return None
    
    async def _extract_attention_weights(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]]
    ) -> Optional[np.ndarray]:
        """Extract attention weights from transformer model."""
        try:
            if hasattr(model, 'get_attention_weights'):
                return model.get_attention_weights(feature_data)
            elif hasattr(model, 'attention_weights'):
                # Some models store attention weights as attributes
                return model.attention_weights
            elif hasattr(model, 'predict_with_attention'):
                _, attention_weights = model.predict_with_attention(feature_data)
                return attention_weights
            
            # Fallback: try to extract from forward pass
            if hasattr(model, 'forward_with_attention'):
                with torch.no_grad():
                    if isinstance(feature_data, (list, np.ndarray)):
                        feature_tensor = torch.FloatTensor(feature_data).unsqueeze(0)
                    else:
                        feature_tensor = feature_data
                    
                    _, attention_weights = model.forward_with_attention(feature_tensor)
                    return attention_weights.cpu().numpy()
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract attention weights: {str(e)}")
            return None
    
    def _generate_attention_heatmap(
        self,
        attention_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Generate attention heatmap data for visualization."""
        try:
            if attention_weights.ndim == 3:  # [num_heads, seq_len, seq_len]
                # Average across heads for simplified heatmap
                avg_attention = np.mean(attention_weights, axis=0)
            elif attention_weights.ndim == 2:  # [seq_len, seq_len]
                avg_attention = attention_weights
            else:
                logger.warning(f"Unexpected attention weights shape: {attention_weights.shape}")
                return {}
            
            return {
                'heatmap_data': avg_attention.tolist(),
                'shape': avg_attention.shape,
                'max_attention': float(np.max(avg_attention)),
                'min_attention': float(np.min(avg_attention)),
                'attention_entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8))),
                'attention_sparsity': float(np.mean(avg_attention < 0.1))
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate attention heatmap: {str(e)}")
            return {}
    
    async def _analyze_temporal_attention(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Analyze temporal attention patterns in time-series data."""
        try:
            # Create temporal attention analyzer if not cached
            analyzer_key = f"temporal_{id(model)}"
            if analyzer_key not in self._attention_cache:
                try:
                    analyzer = TemporalAttentionAnalyzer(model, feature_names)
                    self._attention_cache[analyzer_key] = analyzer
                except Exception as e:
                    logger.debug(f"Could not create temporal attention analyzer: {str(e)}")
                    return None
            
            analyzer = self._attention_cache[analyzer_key]
            
            # Analyze temporal patterns
            temporal_analysis = analyzer.analyze_temporal_patterns(
                feature_data, include_seasonality=True
            )
            
            return {
                'recency_bias': temporal_analysis.get('recency_bias', 0.0),
                'periodic_patterns': temporal_analysis.get('periodic_patterns', []),
                'trend_attention': temporal_analysis.get('trend_attention', 0.0),
                'volatility_focus': temporal_analysis.get('volatility_focus', 0.0)
            }
            
        except Exception as e:
            logger.debug(f"Temporal attention analysis failed: {str(e)}")
            return None
    
    async def _analyze_cross_asset_attention(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze cross-asset attention patterns for multi-asset models."""
        try:
            # Create cross-asset attention analyzer if not cached
            analyzer_key = f"cross_asset_{id(model)}"
            if analyzer_key not in self._attention_cache:
                try:
                    analyzer = CrossAttentionAnalyzer(model, feature_names)
                    self._attention_cache[analyzer_key] = analyzer
                except Exception as e:
                    logger.debug(f"Could not create cross-asset attention analyzer: {str(e)}")
                    return None
            
            analyzer = self._attention_cache[analyzer_key]
            
            # Analyze cross-asset patterns
            cross_asset_analysis = analyzer.analyze_cross_asset_attention(
                feature_data, primary_asset=symbol
            )
            
            return {
                'asset_correlations': cross_asset_analysis.get('asset_correlations', {}),
                'lead_lag_relationships': cross_asset_analysis.get('lead_lag_relationships', {}),
                'arbitrage_patterns': cross_asset_analysis.get('arbitrage_patterns', []),
                'cross_asset_influence': cross_asset_analysis.get('cross_asset_influence', 0.0)
            }
            
        except Exception as e:
            logger.debug(f"Cross-asset attention analysis failed: {str(e)}")
            return None
    
    def _has_multi_asset_data(self, feature_names: List[str]) -> bool:
        """Check if feature names indicate multi-asset data."""
        asset_indicators = ['btc', 'eth', 'sol', 'cross_asset', 'correlation', 'arbitrage']
        return any(
            any(indicator in feature_name.lower() for indicator in asset_indicators)
            for feature_name in feature_names
        )
    
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
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
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
        if explanation.attention_data:
            result['attention_data'] = explanation.attention_data
        return result
    
    def get_attention_statistics(
        self,
        symbol: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get attention statistics for transformer models."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        cutoff_iso = cutoff_time.isoformat() + 'Z'
        
        # Filter explanations with attention data
        explanations = [
            e for e in self._explanation_cache.values()
            if e.timestamp >= cutoff_iso 
            and (not symbol or e.symbol == symbol)
            and e.attention_data is not None
        ]
        
        if not explanations:
            return {}
        
        # Aggregate attention statistics
        attention_entropies = []
        attention_sparsities = []
        recency_biases = []
        
        for explanation in explanations:
            attention_data = explanation.attention_data
            
            if 'attention_heatmap' in attention_data:
                heatmap = attention_data['attention_heatmap']
                if 'attention_entropy' in heatmap:
                    attention_entropies.append(heatmap['attention_entropy'])
                if 'attention_sparsity' in heatmap:
                    attention_sparsities.append(heatmap['attention_sparsity'])
            
            if 'temporal_patterns' in attention_data:
                temporal = attention_data['temporal_patterns']
                if 'recency_bias' in temporal:
                    recency_biases.append(temporal['recency_bias'])
        
        stats = {
            'total_transformer_explanations': len(explanations),
            'attention_coverage': len(explanations) / len(self._explanation_cache) if self._explanation_cache else 0.0
        }
        
        if attention_entropies:
            stats['average_attention_entropy'] = np.mean(attention_entropies)
            stats['attention_entropy_std'] = np.std(attention_entropies)
        
        if attention_sparsities:
            stats['average_attention_sparsity'] = np.mean(attention_sparsities)
            stats['attention_sparsity_std'] = np.std(attention_sparsities)
        
        if recency_biases:
            stats['average_recency_bias'] = np.mean(recency_biases)
            stats['recency_bias_std'] = np.std(recency_biases)
        
        return stats
    
    def clear_attention_cache(self) -> None:
        """Clear the attention analyzer cache."""
        self._attention_cache.clear()
        logger.info("Cleared attention analyzer cache")
    
    def set_attention_analysis_enabled(self, enabled: bool) -> None:
        """Enable or disable attention analysis for transformer models."""
        self._enable_attention_heatmaps = enabled
        logger.info(f"Attention analysis {'enabled' if enabled else 'disabled'}")