"""
TemporalAttentionAnalyzer - Time-Series Importance Pattern Analysis

This module provides the TemporalAttentionAnalyzer class for identifying and 
analyzing temporal patterns in attention weights to understand time-series
importance for trading decisions.
"""

import logging
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import torch
from torch import Tensor

from ..base import BaseExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class TemporalAttentionAnalyzer(BaseExplainer):
    """
    Temporal attention analyzer for time-series forecasting models.
    
    Analyzes temporal patterns in attention weights to identify recency bias,
    periodic patterns, regime transitions, and time window importance.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TemporalAttentionAnalyzer.
        
        Args:
            model: Transformer model with attention capability
            feature_names: List of feature names (should include temporal info)
            config: Optional configuration dictionary
        """
        # Set default config
        default_config = {
            'analyze_recency_bias': True,
            'detect_periodic_patterns': True,
            'identify_regime_transitions': True,
            'time_window_analysis': True,
            'seasonal_attention_patterns': True,
            'min_period': 2,
            'max_period': 168,  # 1 week for hourly data
            'regime_threshold': 0.3
        }
        
        final_config = {**default_config, **(config or {})}
        super().__init__(model, feature_names, final_config)
        
        # Parse temporal information from feature names
        self.temporal_info = self._parse_temporal_features()
    
    def _parse_temporal_features(self) -> Dict[str, Any]:
        """Parse temporal information from feature names."""
        temporal_info = {
            'has_timestamps': False,
            'timestamp_pattern': None,
            'sequence_length': len(self.feature_names),
            'assets': set(),
            'time_indices': []
        }
        
        # Common timestamp patterns
        patterns = [
            r'.*_t_(\d+)',           # feature_t_0, feature_t_1
            r'.*_(\d{2}:\d{2})',     # price_14:30
            r'.*_(\d{2}_\d{2})',     # price_14_30
            r'.*_(\d{4}\d{2}\d{2}_\d{4})',  # price_20240101_1430
            r'.*_(\w{3}_\d{2})',     # price_Mon_14
        ]
        
        for pattern in patterns:
            matches = []
            for feature_name in self.feature_names:
                match = re.search(pattern, feature_name)
                if match:
                    matches.append(match.group(1))
                    # Extract asset name (assume it's before the first underscore)
                    asset = feature_name.split('_')[0]
                    temporal_info['assets'].add(asset)
            
            if matches:
                temporal_info['has_timestamps'] = True
                temporal_info['timestamp_pattern'] = pattern
                temporal_info['time_indices'] = matches
                break
        
        return temporal_info
    
    def validate_input(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model and feature names for temporal analysis.
        
        Args:
            model: The model to validate
            feature_names: List of feature names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation
        if not isinstance(feature_names, list) or len(feature_names) == 0:
            return False, "feature_names must be a non-empty list"
        
        # Check for attention capability
        if not hasattr(model, 'get_attention_weights'):
            return False, "Model must have 'get_attention_weights' method"
        
        return True, None
    
    def analyze_recency_bias(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze recency bias in attention patterns.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with recency bias analysis
        """
        if not hasattr(self.model, 'get_attention_weights'):
            raise ValueError("Model must have get_attention_weights method")
        
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        seq_len = attention_weights.shape[-1]
        
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        # Compute recency score
        position_weights = torch.arange(seq_len, dtype=torch.float32)
        
        # For each query position, compute correlation with position indices
        recency_scores = []
        for query_pos in range(seq_len):
            attention_values = avg_attention[query_pos]
            if attention_values.sum() > 0:
                corr = torch.corrcoef(torch.stack([attention_values, position_weights]))[0, 1]
                if not torch.isnan(corr):
                    recency_scores.append(float(corr))
        
        overall_recency_score = np.mean(recency_scores) if recency_scores else 0.0
        
        # Compute temporal weights (average attention to each position)
        temporal_weights = avg_attention.mean(dim=0)  # Average over query positions
        temporal_weights = temporal_weights / temporal_weights.sum().clamp(min=1e-8)
        
        # Compute bias strength (how concentrated is attention on recent positions)
        recent_positions = seq_len // 4  # Last 25% of positions
        recent_attention = temporal_weights[-recent_positions:].sum()
        bias_strength = float(recent_attention)
        
        # Check for temporal structure
        temporal_structure_strength = self._compute_temporal_structure_strength(avg_attention)
        
        return {
            "recency_score": float(np.clip(overall_recency_score, 0.0, 1.0)),
            "temporal_weights": temporal_weights.cpu().numpy().tolist(),
            "bias_strength": bias_strength,
            "temporal_structure_strength": temporal_structure_strength,
            "analysis_summary": {
                "strong_recency_bias": overall_recency_score > 0.7,
                "recent_focus_percentage": bias_strength * 100
            }
        }
    
    def _compute_temporal_structure_strength(self, attention: Tensor) -> float:
        """Compute strength of temporal structure in attention."""
        # Compute autocorrelation of attention patterns
        seq_len = attention.shape[0]
        
        # Average attention pattern
        avg_pattern = attention.mean(dim=0)
        
        # Compute autocorrelation at different lags
        autocorrs = []
        for lag in range(1, min(seq_len // 4, 10)):
            if seq_len - lag > 0:
                corr = torch.corrcoef(torch.stack([
                    avg_pattern[:-lag],
                    avg_pattern[lag:]
                ]))[0, 1]
                if not torch.isnan(corr):
                    autocorrs.append(float(corr.abs()))
        
        return float(np.mean(autocorrs)) if autocorrs else 0.0
    
    def detect_periodic_patterns(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Detect periodic patterns in attention.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with periodic pattern analysis
        """
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        seq_len = attention_weights.shape[-1]
        
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        # Compute periodicity for different periods
        min_period = self.config.get('min_period', 2)
        max_period = min(self.config.get('max_period', 168), seq_len // 2)
        
        period_strengths = {}
        dominant_period = None
        max_strength = 0.0
        
        for period in range(min_period, max_period + 1):
            strength = self._compute_period_strength(avg_attention, period)
            period_strengths[period] = strength
            
            if strength > max_strength:
                max_strength = strength
                dominant_period = period
        
        # Analyze seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(avg_attention, seq_len)
        
        return {
            "dominant_period": dominant_period if dominant_period else min_period,
            "periodicity_strength": max_strength,
            "period_strengths": period_strengths,
            "seasonal_patterns": seasonal_patterns,
            "analysis_summary": {
                "has_strong_periodicity": max_strength > 0.5,
                "likely_daily_pattern": 24 in period_strengths and period_strengths.get(24, 0) > 0.3,
                "likely_weekly_pattern": 168 in period_strengths and period_strengths.get(168, 0) > 0.3
            }
        }
    
    def _compute_period_strength(self, attention: Tensor, period: int) -> float:
        """Compute strength of a specific period in attention pattern."""
        seq_len = attention.shape[0]
        
        # Average attention across query positions
        avg_attention = attention.mean(dim=0)
        
        # Compute correlation between positions separated by the period
        correlations = []
        for start in range(period):
            positions = list(range(start, seq_len, period))
            if len(positions) >= 2:
                values = avg_attention[positions]
                if values.std() > 1e-6:  # Avoid division by zero
                    # Compute autocorrelation
                    for i in range(len(values) - 1):
                        for j in range(i + 1, len(values)):
                            corr = torch.corrcoef(torch.stack([
                                values[i:i+1],
                                values[j:j+1]
                            ]))[0, 1]
                            if not torch.isnan(corr):
                                correlations.append(float(corr.abs()))
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _analyze_seasonal_patterns(self, attention: Tensor, seq_len: int) -> Dict[str, Any]:
        """Analyze seasonal attention patterns."""
        seasonal_info = {}
        
        # Weekly pattern (if sequence is long enough)
        if seq_len >= 168:  # 1 week of hourly data
            weekly_strength = self._compute_period_strength(attention, 168)
            seasonal_info["weekly_pattern"] = {
                "strength": weekly_strength,
                "detected": weekly_strength > 0.3
            }
        
        # Daily pattern
        if seq_len >= 24:
            daily_strength = self._compute_period_strength(attention, 24)
            seasonal_info["daily_pattern"] = {
                "strength": daily_strength,
                "detected": daily_strength > 0.3
            }
        
        # Compute overall seasonal strength
        strengths = [info["strength"] for info in seasonal_info.values()]
        seasonal_strength = float(np.mean(strengths)) if strengths else 0.0
        
        seasonal_info["seasonal_strength"] = seasonal_strength
        
        return seasonal_info
    
    def identify_regime_transitions(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Identify regime transitions in attention patterns.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with regime transition analysis
        """
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        seq_len = attention_weights.shape[-1]
        
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        # Detect transitions by analyzing attention shift
        transition_points = []
        regime_threshold = self.config.get('regime_threshold', 0.3)
        
        # Sliding window analysis
        window_size = max(seq_len // 10, 3)
        
        for i in range(window_size, seq_len - window_size):
            # Compare attention patterns before and after position i
            before_pattern = avg_attention[:i].mean(dim=0)
            after_pattern = avg_attention[i:].mean(dim=0)
            
            # Compute KL divergence between patterns
            kl_divergence = self._compute_kl_divergence(before_pattern, after_pattern)
            
            if kl_divergence > regime_threshold:
                transition_points.append(i)
        
        # Filter nearby transition points
        filtered_transitions = self._filter_nearby_transitions(transition_points, min_distance=window_size)
        
        # Compute regime stability
        regime_stability = self._compute_regime_stability(avg_attention, filtered_transitions)
        
        # Compute attention shift magnitude
        shift_magnitudes = []
        for tp in filtered_transitions:
            if tp > 0 and tp < seq_len - 1:
                before = avg_attention[:tp].mean()
                after = avg_attention[tp:].mean()
                shift_magnitudes.append(float((after - before).abs()))
        
        avg_shift_magnitude = float(np.mean(shift_magnitudes)) if shift_magnitudes else 0.0
        
        return {
            "transition_points": filtered_transitions,
            "regime_stability": regime_stability,
            "attention_shift_magnitude": avg_shift_magnitude,
            "analysis_summary": {
                "num_regimes": len(filtered_transitions) + 1,
                "has_clear_regimes": len(filtered_transitions) > 0 and avg_shift_magnitude > 0.1,
                "regime_change_frequency": len(filtered_transitions) / seq_len if seq_len > 0 else 0.0
            }
        }
    
    def _compute_kl_divergence(self, p: Tensor, q: Tensor) -> float:
        """Compute KL divergence between two probability distributions."""
        # Normalize to probability distributions
        p = p / p.sum().clamp(min=1e-8)
        q = q / q.sum().clamp(min=1e-8)
        
        # Compute KL divergence
        kl = (p * torch.log((p / q.clamp(min=1e-8)).clamp(min=1e-8))).sum()
        
        return float(kl) if not torch.isnan(kl) else 0.0
    
    def _filter_nearby_transitions(self, transitions: List[int], min_distance: int) -> List[int]:
        """Filter out transition points that are too close to each other."""
        if not transitions:
            return []
        
        filtered = [transitions[0]]
        
        for tp in transitions[1:]:
            if tp - filtered[-1] >= min_distance:
                filtered.append(tp)
        
        return filtered
    
    def _compute_regime_stability(self, attention: Tensor, transitions: List[int]) -> float:
        """Compute stability within each regime."""
        seq_len = attention.shape[0]
        
        if not transitions:
            # Single regime - compute overall stability
            return 1.0 - float(attention.std())
        
        # Compute stability for each regime
        regime_boundaries = [0] + transitions + [seq_len]
        stabilities = []
        
        for i in range(len(regime_boundaries) - 1):
            start, end = regime_boundaries[i], regime_boundaries[i + 1]
            if end - start > 1:
                regime_attention = attention[start:end]
                stability = 1.0 - float(regime_attention.std())
                stabilities.append(stability)
        
        return float(np.mean(stabilities)) if stabilities else 0.0
    
    def analyze_time_windows(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze different time window importances.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with time window analysis
        """
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        seq_len = attention_weights.shape[-1]
        
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        # Define time windows
        windows = {
            "last_24h": min(24, seq_len),
            "last_48h": min(48, seq_len),
            "last_72h": min(72, seq_len),
            "last_week": min(168, seq_len)
        }
        
        # Compute importance for each window
        window_importances = {}
        temporal_weights = avg_attention.mean(dim=0)  # Average over query positions
        
        for window_name, window_size in windows.items():
            if window_size > 0:
                window_attention = temporal_weights[-window_size:].sum()
                window_importances[window_name] = float(window_attention)
        
        # Find dominant window
        dominant_window = max(window_importances.keys(), 
                            key=lambda k: window_importances[k])
        
        # Compute temporal focus distribution
        total_attention = temporal_weights.sum()
        focus_distribution = {}
        for window_name, importance in window_importances.items():
            focus_distribution[window_name] = float(importance / total_attention.clamp(min=1e-8))
        
        return {
            "window_importances": window_importances,
            "dominant_window": dominant_window,
            "temporal_focus_distribution": focus_distribution,
            "analysis_summary": {
                "recent_focus_strong": window_importances.get("last_24h", 0) > 0.5,
                "temporal_decay": self._compute_temporal_decay(temporal_weights)
            }
        }
    
    def _compute_temporal_decay(self, temporal_weights: Tensor) -> float:
        """Compute temporal decay rate in attention weights."""
        seq_len = len(temporal_weights)
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Fit exponential decay
        log_weights = torch.log(temporal_weights.clamp(min=1e-8))
        
        # Compute correlation with position (negative correlation = decay)
        corr = torch.corrcoef(torch.stack([log_weights, positions]))[0, 1]
        
        return float(-corr) if not torch.isnan(corr) else 0.0
    
    def analyze_seasonal_attention_patterns(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in attention.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with seasonal analysis
        """
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        seq_len = attention_weights.shape[-1]
        
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        return self._analyze_seasonal_patterns(avg_attention, seq_len)
    
    def get_feature_importance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get temporal-weighted feature importance.
        
        Args:
            instance: Input instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Get base attention-based importance
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        # Compute temporal-weighted importance
        temporal_weights = attention_weights.mean(dim=(0, 1, 2))  # [seq]
        temporal_weights = temporal_weights / temporal_weights.sum().clamp(min=1e-8)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(temporal_weights):
                feature_importance[feature_name] = float(temporal_weights[i])
            else:
                feature_importance[feature_name] = 0.0
        
        return feature_importance
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> ExplanationData:
        """
        Explain instance with temporal attention analysis.
        
        Args:
            instance: Single input instance to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            ExplanationData object containing the explanation
        """
        start_time = time.time()
        
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Handle single timestamp case
        if len(instance) == 1:
            feature_importance = {self.feature_names[0]: 1.0}
            explanation_metadata = {
                "insufficient_temporal_data": True,
                "sequence_length": 1,
                "temporal_analysis": "Limited due to single timestamp"
            }
            
            return ExplanationData(
                feature_importance=feature_importance,
                explanation_type="temporal_attention",
                instance_data=instance.tolist(),
                model_prediction=0.5,  # Default prediction
                explanation_metadata=explanation_metadata,
                feature_names=self.feature_names.copy()
            )
        
        # Get model prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(instance.reshape(1, -1))
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0] if len(prediction) == 1 else prediction
        else:
            prediction = 0.5  # Fallback
        
        # Get feature importance
        feature_importance = self.get_feature_importance(instance)
        
        # Perform temporal analyses
        analyses = {}
        
        if self.config.get('analyze_recency_bias', True):
            analyses["recency_analysis"] = self.analyze_recency_bias(instance)
        
        if self.config.get('detect_periodic_patterns', True):
            analyses["periodic_patterns"] = self.detect_periodic_patterns(instance)
        
        if self.config.get('identify_regime_transitions', True):
            analyses["regime_transitions"] = self.identify_regime_transitions(instance)
        
        if self.config.get('time_window_analysis', True):
            analyses["time_window_analysis"] = self.analyze_time_windows(instance)
        
        if self.config.get('seasonal_attention_patterns', True):
            seasonal_analysis = self.analyze_seasonal_attention_patterns(instance)
            analyses["seasonal_patterns"] = seasonal_analysis
        
        # Add trading pattern analysis for multi-asset scenarios
        if len(self.temporal_info['assets']) > 1:
            analyses["trading_pattern_analysis"] = self._analyze_trading_patterns(instance)
        
        # Add cross-asset temporal analysis for multi-asset data
        if len(self.temporal_info['assets']) > 1:
            analyses["cross_asset_temporal_analysis"] = self._analyze_cross_asset_temporal_patterns(instance)
        
        # Prepare metadata
        explanation_metadata = {
            **analyses,
            "temporal_config": self.config.copy(),
            "temporal_info": self.temporal_info,
            "model_type": self.model.__class__.__name__,
            "explanation_time_ms": (time.time() - start_time) * 1000
        }
        
        return ExplanationData(
            feature_importance=feature_importance,
            explanation_type="temporal_attention",
            instance_data=instance.tolist(),
            model_prediction=prediction,
            explanation_metadata=explanation_metadata,
            feature_names=self.feature_names.copy()
        )
    
    def _analyze_trading_patterns(self, instance: np.ndarray) -> Dict[str, Any]:
        """Analyze trading-specific temporal patterns."""
        attention_weights = self.model.get_attention_weights(instance)
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        # Look for trend reversal patterns
        trend_reversal_indicators = []
        seq_len = attention_weights.shape[-1]
        
        # Simple heuristic: high attention to positions that could indicate reversals
        avg_attention = attention_weights.mean(dim=(0, 1, 2))  # [seq]
        
        # Find peaks in attention (potential reversal points)
        for i in range(1, seq_len - 1):
            if (avg_attention[i] > avg_attention[i-1] and 
                avg_attention[i] > avg_attention[i+1] and
                avg_attention[i] > avg_attention.mean() + avg_attention.std()):
                trend_reversal_indicators.append(i)
        
        return {
            "trend_reversal_indicators": trend_reversal_indicators,
        }
    
    def _analyze_cross_asset_temporal_patterns(self, instance: np.ndarray) -> Dict[str, Any]:
        """Analyze cross-asset temporal patterns."""
        cross_asset_analysis = {}
        
        # Group features by asset
        asset_features = {}
        for i, feature_name in enumerate(self.feature_names):
            asset = feature_name.split('_')[0]
            if asset not in asset_features:
                asset_features[asset] = []
            asset_features[asset].append(i)
        
        # Analyze temporal patterns for each asset
        for asset, feature_indices in asset_features.items():
            if len(feature_indices) > 1:
                # Compute attention weights for this asset's features
                attention_weights = self.model.get_attention_weights(instance)
                if not isinstance(attention_weights, torch.Tensor):
                    attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
                
                asset_attention = attention_weights[:, :, feature_indices, :][:, :, :, feature_indices]
                avg_asset_attention = asset_attention.mean(dim=(0, 1))
                
                # Analyze recency bias for this asset
                recency_score = self._compute_recency_score(avg_asset_attention)
                
                cross_asset_analysis[f"{asset}_temporal_patterns"] = {
                    "recency_score": recency_score,
                    "feature_count": len(feature_indices)
                }
        
        return cross_asset_analysis
    
    def _compute_recency_score(self, attention: Tensor) -> float:
        """Compute recency score for attention matrix."""
        seq_len = attention.shape[0]
        position_weights = torch.arange(seq_len, dtype=torch.float32)
        
        # Average attention across query positions
        avg_attention = attention.mean(dim=0)
        
        # Compute correlation with position indices
        if avg_attention.sum() > 0:
            corr = torch.corrcoef(torch.stack([avg_attention, position_weights]))[0, 1]
            return float(corr) if not torch.isnan(corr) else 0.0
        
        return 0.0
    
    def explain(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        **kwargs
    ) -> List[ExplanationData]:
        """
        Explain multiple instances.
        
        Args:
            instances: Input instances to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            List of ExplanationData objects, one per instance
        """
        if isinstance(instances, list):
            instances = np.array(instances)
        
        if len(instances.shape) == 1:
            instances = instances.reshape(1, -1)
        
        explanations = []
        for instance in instances:
            explanation = self.explain_instance(instance, **kwargs)
            explanations.append(explanation)
        
        return explanations
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        return {
            "explainer_type": "temporal_attention",
            "temporal_analysis_capabilities": [
                "recency_bias",
                "periodic_patterns", 
                "regime_transitions",
                "time_window_analysis",
                "seasonal_patterns"
            ],
            "supported_patterns": [
                "daily_cycles",
                "weekly_cycles", 
                "trend_reversals",
                "regime_changes",
                "momentum_shifts"
            ],
            "temporal_config": self.config.copy(),
            "temporal_info": self.temporal_info,
            "performance_requirements": {
                "analysis_time_ms": "<500"
            }
        }