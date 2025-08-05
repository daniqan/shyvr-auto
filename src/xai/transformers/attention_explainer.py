"""
AttentionExplainer - Attention Weight Analysis for Transformer Models

This module provides the AttentionExplainer class for extracting and analyzing
attention weights from transformer models to provide interpretable explanations
for trading decisions.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor

from ..base import GradientBasedExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class AttentionExplainer(GradientBasedExplainer):
    """
    Attention-based explainer for transformer models.
    
    Extracts and analyzes attention weights from transformer models to provide
    interpretable explanations for predictions. Supports all transformer model
    types used in the RLTE system.
    """
    
    SUPPORTED_MODEL_TYPES = [
        'iTransformer',
        'PatchTST', 
        'TimesMixer',
        'TimesFM',
        'TransformerPredictor'
    ]
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AttentionExplainer.
        
        Args:
            model: Transformer model with attention capability
            feature_names: List of feature names
            config: Optional configuration dictionary
        """
        # Set default config
        default_config = {
            'extract_attention': True,
            'attention_heads': 'all',
            'attention_layers': 'last',
            'attention_rollout': True,
            'normalize_attention': True,
            'performance_mode': True
        }
        
        final_config = {**default_config, **(config or {})}
        super().__init__(model, feature_names, final_config)
        
        # Store original config for metadata
        self._original_config = config or {}
        
        # Validate that model supports attention extraction
        self._validate_attention_support()
    
    def _validate_attention_support(self) -> None:
        """
        Validate that the model supports attention weight extraction.
        
        Raises:
            ValueError: If model doesn't support attention extraction
        """
        required_methods = ['get_attention_weights']
        
        for method in required_methods:
            if not hasattr(self.model, method):
                raise ValueError(
                    f"Model must have '{method}' method for attention-based explanations. "
                    f"Supported model types: {self.SUPPORTED_MODEL_TYPES}"
                )
    
    def validate_input(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model and feature names for attention explainer.
        
        Args:
            model: The model to validate
            feature_names: List of feature names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check feature names but be more lenient than base class
        if not isinstance(feature_names, list):
            return False, "feature_names must be a list"
        
        if len(feature_names) == 0:
            # Allow empty feature names during init but warn
            logger.warning("Empty feature names provided - explanations may fail")
        
        if not all(isinstance(name, str) for name in feature_names):
            return False, "All feature names must be strings"
        
        # Check for attention capability
        if not hasattr(model, 'get_attention_weights'):
            return False, "Model must have 'get_attention_weights' method for attention analysis"
        
        # Check model type if available
        model_name = model.__class__.__name__
        if model_name not in self.SUPPORTED_MODEL_TYPES and model_name != 'Mock':
            logger.warning(f"Model type '{model_name}' not in supported types: {self.SUPPORTED_MODEL_TYPES}")
        
        return True, None
    
    def extract_attention_weights(self, instance: np.ndarray) -> Tensor:
        """
        Extract attention weights from the model for given instance.
        
        Args:
            instance: Input instance
            
        Returns:
            Attention weights tensor [batch, heads, seq_len, seq_len]
        """
        # Convert input to appropriate format for model
        if isinstance(instance, np.ndarray):
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)  # Add batch dimension
        
        # Extract attention weights from model
        try:
            attention_weights = self.model.get_attention_weights(instance)
            
            if not isinstance(attention_weights, torch.Tensor):
                attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
            
            # Validate attention shape
            if len(attention_weights.shape) != 4:
                raise ValueError(f"Expected 4D attention weights [batch, heads, seq, seq], got shape {attention_weights.shape}")
            
            # Handle NaN values
            if torch.isnan(attention_weights).any():
                logger.warning("NaN values detected in attention weights, replacing with zeros")
                attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Failed to extract attention weights: {str(e)}")
            raise ValueError(f"Could not extract attention weights from model: {str(e)}")
    
    def compute_attention_rollout(self, instance: np.ndarray) -> Tensor:
        """
        Compute attention rollout for long-range dependency tracking.
        
        Args:
            instance: Input instance
            
        Returns:
            Attention rollout tensor [seq_len, seq_len]
        """
        # For models with multiple layers, we need layer-wise attention
        if hasattr(self.model, 'get_layer_attention_weights'):
            layer_attentions = self.model.get_layer_attention_weights(instance)
            
            if not isinstance(layer_attentions, list):
                layer_attentions = [layer_attentions]
            
            # Convert to tensors
            layer_tensors = []
            for layer_att in layer_attentions:
                if not isinstance(layer_att, torch.Tensor):
                    layer_att = torch.tensor(layer_att, dtype=torch.float32)
                layer_tensors.append(layer_att)
            
            # Compute rollout through layers
            rollout = None
            for layer_att in layer_tensors:
                # Average over heads: [batch, heads, seq, seq] -> [batch, seq, seq]
                layer_att_avg = layer_att.mean(dim=1)
                
                if rollout is None:
                    rollout = layer_att_avg
                else:
                    # Matrix multiplication for rollout
                    rollout = torch.matmul(rollout, layer_att_avg)
            
            # Remove batch dimension and normalize
            rollout = rollout.squeeze(0)  # [seq, seq]
            rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
        else:
            # Single layer attention - just average over heads
            attention_weights = self.extract_attention_weights(instance)
            rollout = attention_weights.mean(dim=1).squeeze(0)  # [seq, seq]
            rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Ensure values are between 0 and 1
        rollout = torch.clamp(rollout, 0.0, 1.0)
        
        return rollout
    
    def analyze_attention_heads(self, instance: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Analyze each attention head separately.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with head-wise analysis
        """
        attention_weights = self.extract_attention_weights(instance)
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        head_analysis = {}
        
        for head_idx in range(num_heads):
            head_attention = attention_weights[0, head_idx]  # [seq_len, seq_len]
            
            # Compute head specialization metrics
            attention_entropy = self._compute_attention_entropy(head_attention)
            attention_focus = self._compute_attention_focus(head_attention)
            
            # Identify attention patterns
            pattern_type = self._identify_attention_pattern(head_attention)
            
            # Compute importance scores for this head
            importance_scores = head_attention.sum(dim=0)  # Sum over query positions
            importance_scores = importance_scores / importance_scores.sum()
            
            head_analysis[f"head_{head_idx}"] = {
                "attention_pattern": head_attention.cpu().numpy().tolist(),
                "specialization": {
                    "entropy": float(attention_entropy),
                    "focus": float(attention_focus),
                    "pattern_type": pattern_type
                },
                "importance_scores": importance_scores.cpu().numpy().tolist()
            }
        
        return head_analysis
    
    def _compute_attention_entropy(self, attention: Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Average attention across query positions
        avg_attention = attention.mean(dim=0)
        avg_attention = avg_attention / avg_attention.sum().clamp(min=1e-8)
        
        # Compute entropy
        entropy = -(avg_attention * torch.log(avg_attention.clamp(min=1e-8))).sum()
        return float(entropy)
    
    def _compute_attention_focus(self, attention: Tensor) -> float:
        """Compute attention focus (inverse of dispersion)."""
        # Average attention across query positions
        avg_attention = attention.mean(dim=0)
        
        # Compute Gini coefficient as focus measure
        sorted_att, _ = torch.sort(avg_attention)
        n = len(sorted_att)
        index = torch.arange(1, n + 1, dtype=torch.float32)
        gini = (2 * (index * sorted_att).sum()) / (n * sorted_att.sum()) - (n + 1) / n
        
        return float(gini)
    
    def _identify_attention_pattern(self, attention: Tensor) -> str:
        """Identify the type of attention pattern."""
        seq_len = attention.shape[0]
        
        # Check for diagonal pattern (local attention)
        diagonal_strength = torch.diag(attention).mean()
        
        # Check for recency bias (attention to recent positions)
        recency_weights = torch.arange(seq_len, dtype=torch.float32)
        recency_corr = torch.corrcoef(torch.stack([
            attention.mean(dim=0),
            recency_weights
        ]))[0, 1]
        
        # Check for broadcast pattern (one position attends to all)
        broadcast_strength = attention.max(dim=1)[0].mean()
        
        if diagonal_strength > 0.5:
            return "local"
        elif abs(recency_corr) > 0.7:
            return "recency_biased"
        elif broadcast_strength > 0.8:
            return "broadcast"
        else:
            return "global"
    
    def analyze_cross_asset_attention(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cross-asset attention patterns for multivariate inputs.
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary with cross-asset analysis
        """
        attention_weights = self.extract_attention_weights(instance)
        seq_len = attention_weights.shape[-1]
        
        # Try to identify asset groups from feature names
        asset_groups = self._identify_asset_groups()
        
        if len(asset_groups) <= 1:
            return {"message": "Single asset detected, limited cross-asset analysis"}
        
        cross_asset_analysis = {}
        
        # Analyze attention between asset groups
        for asset1, indices1 in asset_groups.items():
            for asset2, indices2 in asset_groups.items():
                if asset1 != asset2:
                    # Extract attention between asset groups
                    cross_attention = attention_weights[:, :, indices1, :][:, :, :, indices2]
                    avg_cross_attention = cross_attention.mean()
                    
                    cross_asset_analysis[f"{asset1}_to_{asset2}"] = {
                        "average_attention": float(avg_cross_attention),
                        "attention_strength": "strong" if avg_cross_attention > 0.3 else "weak"
                    }
        
        return cross_asset_analysis
    
    def _identify_asset_groups(self) -> Dict[str, List[int]]:
        """Identify asset groups from feature names."""
        asset_groups = {}
        
        for idx, feature_name in enumerate(self.feature_names):
            # Try to extract asset name from feature name
            parts = feature_name.split('_')
            if len(parts) >= 2:
                asset_name = parts[0]  # Assume first part is asset name
                if asset_name not in asset_groups:
                    asset_groups[asset_name] = []
                asset_groups[asset_name].append(idx)
            else:
                # Fallback: treat as single asset
                if "UNKNOWN" not in asset_groups:
                    asset_groups["UNKNOWN"] = []
                asset_groups["UNKNOWN"].append(idx)
        
        return asset_groups
    
    def compute_attention_confidence(self, attention_weights: Tensor) -> float:
        """
        Compute confidence score based on attention distribution.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Confidence score between 0 and 1
        """
        # Average over batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq, seq]
        
        # Compute entropy of attention distribution
        attention_flat = avg_attention.flatten()
        attention_prob = attention_flat / attention_flat.sum().clamp(min=1e-8)
        
        entropy = -(attention_prob * torch.log(attention_prob.clamp(min=1e-8))).sum()
        max_entropy = torch.log(torch.tensor(len(attention_flat), dtype=torch.float32))
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - (entropy / max_entropy)
        
        return float(torch.clamp(confidence, 0.0, 1.0))
    
    def get_feature_importance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores from attention weights.
        
        Args:
            instance: Input instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(instance)
        
        # Compute importance by averaging attention across heads and queries
        # [batch, heads, seq, seq] -> [seq]
        importance_scores = attention_weights.mean(dim=(0, 1, 2))  # Average over batch, heads, queries
        
        # Normalize to sum to 1
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
        Explain a single instance using attention weights.
        
        Args:
            instance: Single input instance to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            ExplanationData object containing the explanation
        """
        start_time = time.time()
        
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Check for empty input
        if len(instance) == 0 or len(self.feature_names) == 0:
            raise ValueError("Cannot explain empty input instance or model with empty feature names")
        
        # Get model prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(instance.reshape(1, -1))
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0] if len(prediction) == 1 else prediction
        else:
            prediction = 0.5  # Fallback
        
        # Get feature importance from attention
        feature_importance = self.get_feature_importance(instance)
        
        # Extract attention weights for metadata
        attention_weights = self.extract_attention_weights(instance)
        
        # Compute attention rollout if enabled
        attention_rollout = None
        if self.config.get('attention_rollout', True):
            try:
                attention_rollout = self.compute_attention_rollout(instance)
                attention_rollout = attention_rollout.cpu().numpy().tolist()
            except Exception as e:
                logger.warning(f"Could not compute attention rollout: {str(e)}")
        
        # Analyze attention heads
        head_analysis = self.analyze_attention_heads(instance)
        
        # Analyze cross-asset attention for multivariate data
        cross_asset_analysis = self.analyze_cross_asset_attention(instance)
        
        # Compute confidence score
        confidence_score = self.compute_attention_confidence(attention_weights)
        
        # Prepare metadata
        explanation_metadata = {
            "attention_weights": attention_weights.cpu().numpy().tolist(),
            "head_analysis": head_analysis,
            "cross_asset_analysis": cross_asset_analysis,
            "attention_config": self.config.copy(),
            "model_type": self.model.__class__.__name__,
            "explanation_time_ms": (time.time() - start_time) * 1000
        }
        
        if attention_rollout is not None:
            explanation_metadata["attention_rollout"] = attention_rollout
        
        return ExplanationData(
            feature_importance=feature_importance,
            explanation_type="attention",
            instance_data=instance.tolist(),
            model_prediction=prediction,
            confidence_score=confidence_score,
            explanation_metadata=explanation_metadata,
            feature_names=self.feature_names.copy()
        )
    
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
    
    def compute_gradients(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradients using attention weights as proxy.
        
        Args:
            instance: Input instance
            target_class: Target class (not used for attention-based gradients)
            
        Returns:
            Gradient-like array based on attention weights
        """
        # For attention explainer, use attention weights as gradient proxy
        feature_importance = self.get_feature_importance(instance)
        gradients = np.array([feature_importance.get(name, 0.0) for name in self.feature_names])
        
        return gradients
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        return {
            "explainer_type": "attention",
            "supported_models": self.SUPPORTED_MODEL_TYPES,
            "attention_config": self._original_config.copy(),
            "capabilities": [
                "attention_weight_extraction",
                "attention_rollout",
                "head_analysis", 
                "cross_asset_analysis",
                "confidence_scoring"
            ],
            "performance_requirements": {
                "inference_overhead_ms": "<10",
                "explanation_time_ms": "<500"
            }
        }