"""
CrossAttentionAnalyzer - Multivariate Relationship Analysis

This module provides the CrossAttentionAnalyzer class for analyzing cross-attention
patterns between different assets and features to understand multivariate 
relationships in trading decisions.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor

from ..base import BaseExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class CrossAttentionAnalyzer(BaseExplainer):
    """
    Cross-attention analyzer for multivariate relationship analysis.
    
    Analyzes cross-attention patterns between different assets and features
    to understand correlations, lead-lag relationships, arbitrage patterns,
    and momentum spillovers in trading scenarios.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CrossAttentionAnalyzer.
        
        Args:
            model: Transformer model with cross-attention capability
            feature_names: List of feature names (should include asset info)
            config: Optional configuration dictionary
        """
        # Set default config
        default_config = {
            'analyze_asset_correlations': True,
            'detect_lead_lag_relationships': True,
            'identify_arbitrage_patterns': True,
            'cross_asset_momentum': True,
            'asset_list': [],
            'correlation_threshold': 0.3,
            'lead_lag_max_steps': 5,
            'arbitrage_threshold': 0.5,
            'real_time_mode': False,
            'latency_optimization': False
        }
        
        final_config = {**default_config, **(config or {})}
        super().__init__(model, feature_names, final_config)
        
        # Parse asset information from feature names
        self.asset_info = self._parse_asset_features()
        
        # Auto-detect assets if not provided in config
        if not self.config.get('asset_list'):
            self.config['asset_list'] = list(self.asset_info['assets'])
    
    def _parse_asset_features(self) -> Dict[str, Any]:
        """Parse asset information from feature names."""
        asset_info = {
            'assets': set(),
            'asset_feature_map': {},
            'feature_asset_map': {},
            'sequence_length_per_asset': {},
            'total_assets': 0
        }
        
        # Parse feature names to extract asset information
        for idx, feature_name in enumerate(self.feature_names):
            # Assume format: ASSET_feature_time or similar
            parts = feature_name.split('_')
            if len(parts) >= 1:
                asset = parts[0]
                asset_info['assets'].add(asset)
                asset_info['feature_asset_map'][idx] = asset
                
                if asset not in asset_info['asset_feature_map']:
                    asset_info['asset_feature_map'][asset] = []
                asset_info['asset_feature_map'][asset].append(idx)
        
        # Compute sequence length per asset
        for asset, feature_indices in asset_info['asset_feature_map'].items():
            asset_info['sequence_length_per_asset'][asset] = len(feature_indices)
        
        asset_info['total_assets'] = len(asset_info['assets'])
        
        return asset_info
    
    def validate_input(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model and feature names for cross-attention analysis.
        
        Args:
            model: The model to validate
            feature_names: List of feature names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation
        if not isinstance(feature_names, list) or len(feature_names) == 0:
            return False, "feature_names must be a non-empty list"
        
        # Check for cross-attention capability
        required_method = 'get_cross_attention_weights'
        if not hasattr(model, required_method):
            # Fallback to regular attention if cross-attention not available
            if not hasattr(model, 'get_attention_weights'):
                return False, f"Model must have '{required_method}' or 'get_attention_weights' method"
        
        return True, None
    
    def analyze_asset_correlations(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cross-asset correlations from attention patterns.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with correlation analysis
        """
        # Get cross-attention weights
        if hasattr(self.model, 'get_cross_attention_weights'):
            attention_weights = self.model.get_cross_attention_weights(instance)
        else:
            attention_weights = self.model.get_attention_weights(instance)
        
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        # Handle NaN values
        if torch.isnan(attention_weights).any():
            logger.warning("NaN values detected in attention weights")
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        num_assets = len(self.asset_info['assets'])
        assets = list(self.asset_info['assets'])
        
        # Create correlation matrix
        correlation_matrix = torch.zeros(num_assets, num_assets)
        dominant_correlations = []
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j:
                    # Get feature indices for both assets
                    indices1 = self.asset_info['asset_feature_map'].get(asset1, [])
                    indices2 = self.asset_info['asset_feature_map'].get(asset2, [])
                    
                    if indices1 and indices2:
                        # Extract cross-attention between asset groups
                        cross_att = attention_weights[:, :, indices1, :][:, :, :, indices2]
                        correlation_strength = float(cross_att.mean())
                        correlation_matrix[i, j] = correlation_strength
                        
                        # Record strong correlations
                        threshold = self.config.get('correlation_threshold', 0.3)
                        if correlation_strength > threshold:
                            dominant_correlations.append({
                                'asset1': asset1,
                                'asset2': asset2, 
                                'strength': correlation_strength
                            })
        
        # Overall correlation strength
        total_correlation = float(correlation_matrix.mean())
        
        return {
            "correlation_matrix": correlation_matrix.cpu().numpy(),
            "dominant_correlations": dominant_correlations,
            "correlation_strength": total_correlation,
            "analysis_summary": {
                "num_strong_correlations": len(dominant_correlations),
                "most_correlated_pair": max(dominant_correlations, key=lambda x: x['strength']) if dominant_correlations else None,
                "average_correlation": total_correlation
            }
        }
    
    def detect_lead_lag_relationships(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Detect lead-lag relationships between assets.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with lead-lag analysis
        """
        # Get attention weights
        if hasattr(self.model, 'get_cross_attention_weights'):
            attention_weights = self.model.get_cross_attention_weights(instance)
        else:
            attention_weights = self.model.get_attention_weights(instance)
        
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        assets = list(self.asset_info['assets'])
        lead_lag_pairs = []
        lag_times = []
        max_lag_steps = self.config.get('lead_lag_max_steps', 5)
        
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    # Get feature indices
                    indices1 = self.asset_info['asset_feature_map'].get(asset1, [])
                    indices2 = self.asset_info['asset_feature_map'].get(asset2, [])
                    
                    if indices1 and indices2:
                        # Analyze lead-lag relationship
                        lead_lag_info = self._compute_lead_lag(
                            attention_weights, indices1, indices2, asset1, asset2, max_lag_steps
                        )
                        
                        if lead_lag_info['strength'] > 0.3:  # Threshold for significance
                            lead_lag_pairs.append({
                                'leader': lead_lag_info['leader'],
                                'follower': lead_lag_info['follower'],
                                'strength': lead_lag_info['strength']
                            })
                            
                            lag_times.append({
                                'leader': lead_lag_info['leader'],
                                'follower': lead_lag_info['follower'],
                                'lag_steps': lead_lag_info['lag_steps']
                            })
        
        # Compute overall relationship strength
        relationship_strength = float(np.mean([pair['strength'] for pair in lead_lag_pairs])) if lead_lag_pairs else 0.0
        
        return {
            "lead_lag_pairs": lead_lag_pairs,
            "lag_times": lag_times,
            "relationship_strength": relationship_strength,
            "analysis_summary": {
                "num_relationships": len(lead_lag_pairs),
                "strongest_relationship": max(lead_lag_pairs, key=lambda x: x['strength']) if lead_lag_pairs else None,
                "average_lag_time": float(np.mean([lt['lag_steps'] for lt in lag_times])) if lag_times else 0.0
            }
        }
    
    def _compute_lead_lag(
        self, 
        attention_weights: Tensor, 
        indices1: List[int], 
        indices2: List[int],
        asset1: str,
        asset2: str,
        max_lag: int
    ) -> Dict[str, Any]:
        """Compute lead-lag relationship between two asset groups."""
        
        # Extract cross-attention between asset groups
        cross_att = attention_weights[:, :, indices2, :][:, :, :, indices1]  # asset2 attending to asset1
        avg_cross_att = cross_att.mean(dim=(0, 1))  # [len(indices2), len(indices1)]
        
        # Check for temporal lag patterns
        best_lag = 0
        best_strength = 0.0
        leader = asset1
        follower = asset2
        
        seq_len1 = len(indices1)
        seq_len2 = len(indices2)
        
        # Look for diagonal patterns with different lags
        for lag in range(-max_lag, max_lag + 1):
            strength = 0.0
            count = 0
            
            for i in range(seq_len2):
                j = i + lag
                if 0 <= j < seq_len1:
                    strength += float(avg_cross_att[i, j])
                    count += 1
            
            if count > 0:
                strength /= count
                if strength > best_strength:
                    best_strength = strength
                    best_lag = lag
                    if lag > 0:
                        leader = asset1
                        follower = asset2
                    elif lag < 0:
                        leader = asset2
                        follower = asset1
                        best_lag = abs(lag)
        
        return {
            'leader': leader,
            'follower': follower,
            'lag_steps': best_lag,
            'strength': best_strength
        }
    
    def identify_arbitrage_patterns(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Identify arbitrage opportunities in cross-attention.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with arbitrage analysis
        """
        # Get attention weights
        if hasattr(self.model, 'get_cross_attention_weights'):
            attention_weights = self.model.get_cross_attention_weights(instance)
        else:
            attention_weights = self.model.get_attention_weights(instance)
        
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        assets = list(self.asset_info['assets'])
        arbitrage_triangles = []
        arbitrage_threshold = self.config.get('arbitrage_threshold', 0.5)
        
        # Look for triangular arbitrage patterns (3-asset cycles)
        if len(assets) >= 3:
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    for k in range(j + 1, len(assets)):
                        triangle = [assets[i], assets[j], assets[k]]
                        triangle_strength = self._compute_arbitrage_triangle_strength(
                            attention_weights, triangle, arbitrage_threshold
                        )
                        
                        if triangle_strength > arbitrage_threshold:
                            arbitrage_triangles.append({
                                'assets': triangle,
                                'strength': triangle_strength,
                                'type': 'triangular_arbitrage'
                            })
        
        # Compute overall arbitrage strength
        arbitrage_strength = float(np.mean([t['strength'] for t in arbitrage_triangles])) if arbitrage_triangles else 0.0
        
        # Estimate profit potential (simplified heuristic)
        profit_potential = self._estimate_profit_potential(arbitrage_triangles)
        
        return {
            "arbitrage_triangles": arbitrage_triangles,
            "arbitrage_strength": arbitrage_strength,
            "profit_potential": profit_potential,
            "analysis_summary": {
                "num_opportunities": len(arbitrage_triangles),
                "strongest_opportunity": max(arbitrage_triangles, key=lambda x: x['strength']) if arbitrage_triangles else None,
                "estimated_profit_score": profit_potential
            }
        }
    
    def _compute_arbitrage_triangle_strength(
        self, 
        attention_weights: Tensor, 
        triangle: List[str], 
        threshold: float
    ) -> float:
        """Compute strength of triangular arbitrage pattern."""
        
        strengths = []
        
        # Check all possible cycles in the triangle
        for i in range(3):
            asset1 = triangle[i]
            asset2 = triangle[(i + 1) % 3]
            
            indices1 = self.asset_info['asset_feature_map'].get(asset1, [])
            indices2 = self.asset_info['asset_feature_map'].get(asset2, [])
            
            if indices1 and indices2:
                # Compute attention from asset1 to asset2
                cross_att = attention_weights[:, :, indices1, :][:, :, :, indices2]
                strength = float(cross_att.mean())
                strengths.append(strength)
        
        # Triangle strength is minimum of all edges (weakest link)
        return float(min(strengths)) if strengths else 0.0
    
    def _estimate_profit_potential(self, triangles: List[Dict[str, Any]]) -> float:
        """Estimate profit potential from arbitrage opportunities."""
        if not triangles:
            return 0.0
        
        # Simple heuristic: higher strength triangles have higher profit potential
        total_potential = sum(triangle['strength'] for triangle in triangles)
        return float(total_potential / len(triangles))
    
    def analyze_cross_asset_momentum(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cross-asset momentum patterns.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with momentum analysis
        """
        # Get attention weights
        if hasattr(self.model, 'get_cross_attention_weights'):
            attention_weights = self.model.get_cross_attention_weights(instance)
        else:
            attention_weights = self.model.get_attention_weights(instance)
        
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        assets = list(self.asset_info['assets'])
        momentum_spillovers = []
        momentum_leaders = []
        
        # Analyze momentum spillover between assets
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    spillover_strength = self._compute_momentum_spillover(
                        attention_weights, asset1, asset2
                    )
                    
                    if spillover_strength > 0.3:  # Threshold for significance
                        momentum_spillovers.append({
                            'from': asset1,
                            'to': asset2,
                            'strength': spillover_strength
                        })
        
        # Identify momentum leaders
        spillover_out = {}
        spillover_in = {}
        
        for spillover in momentum_spillovers:
            from_asset = spillover['from']
            to_asset = spillover['to']
            strength = spillover['strength']
            
            spillover_out[from_asset] = spillover_out.get(from_asset, 0) + strength
            spillover_in[to_asset] = spillover_in.get(to_asset, 0) + strength
        
        # Leaders have high outward spillover and low inward spillover
        for asset in assets:
            out_strength = spillover_out.get(asset, 0)
            in_strength = spillover_in.get(asset, 0)
            leadership_score = out_strength - 0.5 * in_strength
            
            if leadership_score > 0.2:
                momentum_leaders.append({
                    'asset': asset,
                    'leadership_score': leadership_score,
                    'outward_spillover': out_strength,
                    'inward_spillover': in_strength
                })
        
        # Sort leaders by score
        momentum_leaders.sort(key=lambda x: x['leadership_score'], reverse=True)
        
        # Overall momentum strength
        momentum_strength = float(np.mean([s['strength'] for s in momentum_spillovers])) if momentum_spillovers else 0.0
        
        return {
            "momentum_spillovers": momentum_spillovers,
            "momentum_leaders": momentum_leaders,
            "momentum_strength": momentum_strength,
            "analysis_summary": {
                "num_spillovers": len(momentum_spillovers),
                "top_leader": momentum_leaders[0] if momentum_leaders else None,
                "average_spillover_strength": momentum_strength
            }
        }
    
    def _compute_momentum_spillover(
        self, 
        attention_weights: Tensor, 
        from_asset: str, 
        to_asset: str
    ) -> float:
        """Compute momentum spillover strength between two assets."""
        
        from_indices = self.asset_info['asset_feature_map'].get(from_asset, [])
        to_indices = self.asset_info['asset_feature_map'].get(to_asset, [])
        
        if not from_indices or not to_indices:
            return 0.0
        
        # Focus on recent positions (momentum is typically short-term)
        recent_portion = max(1, len(to_indices) // 4)  # Last 25% of positions
        
        recent_to_indices = to_indices[-recent_portion:]
        recent_from_indices = from_indices[-recent_portion:]
        
        # Compute attention from recent 'to' positions to recent 'from' positions
        cross_att = attention_weights[:, :, recent_to_indices, :][:, :, :, recent_from_indices]
        spillover_strength = float(cross_att.mean())
        
        return spillover_strength
    
    def get_feature_importance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance considering cross-asset relationships.
        
        Args:
            instance: Input instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Get cross-attention weights
        if hasattr(self.model, 'get_cross_attention_weights'):
            attention_weights = self.model.get_cross_attention_weights(instance)
        else:
            attention_weights = self.model.get_attention_weights(instance)
        
        if not isinstance(attention_weights, torch.Tensor):
            attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
        
        # Compute cross-asset weighted importance
        # Average over batch, heads, and query positions
        importance_scores = attention_weights.mean(dim=(0, 1, 2))  # [seq]
        
        # Normalize
        importance_scores = importance_scores / importance_scores.sum().clamp(min=1e-8)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(importance_scores):
                feature_importance[feature_name] = float(importance_scores[i])
            else:
                feature_importance[feature_name] = 0.0
        
        return feature_importance
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> ExplanationData:
        """
        Explain instance with cross-attention analysis.
        
        Args:
            instance: Single input instance to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            ExplanationData object containing the explanation
        """
        start_time = time.time()
        
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Handle single asset case
        if len(self.asset_info['assets']) == 1:
            feature_importance = self.get_feature_importance(instance)
            explanation_metadata = {
                "single_asset_limitation": True,
                "detected_assets": list(self.asset_info['assets']),
                "cross_asset_analysis": "Limited due to single asset"
            }
            
            return ExplanationData(
                feature_importance=feature_importance,
                explanation_type="cross_attention",
                instance_data=instance.tolist(),
                model_prediction=0.5,  # Default
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
        
        # Perform cross-attention analyses
        analyses = {}
        
        try:
            if self.config.get('analyze_asset_correlations', True):
                analyses["cross_asset_correlations"] = self.analyze_asset_correlations(instance)
        except Exception as e:
            logger.warning(f"Failed to analyze asset correlations: {str(e)}")
            analyses["cross_asset_correlations"] = {"error": str(e)}
        
        try:
            if self.config.get('detect_lead_lag_relationships', True):
                analyses["lead_lag_relationships"] = self.detect_lead_lag_relationships(instance)
        except Exception as e:
            logger.warning(f"Failed to detect lead-lag relationships: {str(e)}")
            analyses["lead_lag_relationships"] = {"error": str(e)}
        
        try:
            if self.config.get('identify_arbitrage_patterns', True):
                analyses["arbitrage_patterns"] = self.identify_arbitrage_patterns(instance)
        except Exception as e:
            logger.warning(f"Failed to identify arbitrage patterns: {str(e)}")
            analyses["arbitrage_patterns"] = {"error": str(e)}
        
        try:
            if self.config.get('cross_asset_momentum', True):
                analyses["momentum_analysis"] = self.analyze_cross_asset_momentum(instance)
        except Exception as e:
            logger.warning(f"Failed to analyze cross-asset momentum: {str(e)}")
            analyses["momentum_analysis"] = {"error": str(e)}
        
        # Add portfolio analysis for trading
        analyses["portfolio_analysis"] = self._analyze_portfolio_implications(instance)
        
        # Handle mismatched dimensions
        config_assets = set(self.config.get('asset_list', []))
        detected_assets = self.asset_info['assets']
        if config_assets != detected_assets:
            analyses["auto_detected_assets"] = list(detected_assets)
            analyses["config_assets"] = list(config_assets)
        
        # Check for NaN values in attention
        try:
            if hasattr(self.model, 'get_cross_attention_weights'):
                attention_weights = self.model.get_cross_attention_weights(instance)
            else:
                attention_weights = self.model.get_attention_weights(instance)
            
            if isinstance(attention_weights, torch.Tensor) and torch.isnan(attention_weights).any():
                analyses["nan_values_detected"] = True
        except Exception:
            pass
        
        # Prepare metadata
        explanation_metadata = {
            **analyses,
            "cross_attention_config": self.config.copy(),
            "asset_info": self.asset_info,
            "model_type": self.model.__class__.__name__,
            "explanation_time_ms": (time.time() - start_time) * 1000
        }
        
        return ExplanationData(
            feature_importance=feature_importance,
            explanation_type="cross_attention",
            instance_data=instance.tolist(),
            model_prediction=prediction,
            explanation_metadata=explanation_metadata,
            feature_names=self.feature_names.copy()
        )
    
    def _analyze_portfolio_implications(self, instance: np.ndarray) -> Dict[str, Any]:
        """Analyze portfolio-level implications."""
        assets = list(self.asset_info['assets'])
        
        # Compute asset weights based on attention
        asset_weights = {}
        total_features = len(self.feature_names)
        
        for asset in assets:
            feature_count = len(self.asset_info['asset_feature_map'].get(asset, []))
            asset_weights[asset] = feature_count / total_features
        
        # Simple diversification score (higher is more diversified)
        diversification_score = 1.0 - sum(w**2 for w in asset_weights.values())
        
        # Generate rebalancing signals (simplified)
        rebalancing_signals = {}
        for asset in assets:
            # Higher attention weight suggests higher importance
            weight = asset_weights[asset]
            if weight > 1.0 / len(assets) * 1.2:  # 20% above equal weight
                rebalancing_signals[asset] = "increase"
            elif weight < 1.0 / len(assets) * 0.8:  # 20% below equal weight
                rebalancing_signals[asset] = "decrease"
            else:
                rebalancing_signals[asset] = "hold"
        
        return {
            "asset_weights": asset_weights,
            "rebalancing_signals": rebalancing_signals,
            "diversification_score": diversification_score,
            "portfolio_summary": {
                "num_assets": len(assets),
                "most_important_asset": max(asset_weights.keys(), key=lambda k: asset_weights[k]),
                "diversification_level": "high" if diversification_score > 0.7 else "medium" if diversification_score > 0.4 else "low"
            }
        }
    
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
            "explainer_type": "cross_attention",
            "cross_attention_capabilities": [
                "asset_correlations",
                "lead_lag_detection",
                "arbitrage_analysis",
                "momentum_spillovers",
                "portfolio_analysis"
            ],
            "supported_assets": list(self.asset_info['assets']),
            "asset_info": self.asset_info,
            "cross_attention_config": self.config.copy(),
            "performance_requirements": {
                "analysis_time_ms": "<500",
                "real_time_analysis_ms": "<100"
            }
        }