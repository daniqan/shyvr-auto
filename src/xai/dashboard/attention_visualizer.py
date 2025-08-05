"""
Attention Pattern Visualizer for Transformer Models.

This module provides basic visualization components for transformer attention patterns
in trading decisions. This is a foundational structure that can be extended with
full dashboard implementation.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Basic attention pattern visualizer for transformer models.
    
    Provides foundational structure for creating attention heatmaps,
    head analysis charts, and temporal attention patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize attention visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_colormap = self.config.get('colormap', 'viridis')
        self.max_display_heads = self.config.get('max_display_heads', 8)
        self.max_sequence_length = self.config.get('max_sequence_length', 96)
        
        logger.info("Initialized AttentionVisualizer")
    
    def create_attention_heatmap_data(
        self,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Create data structure for attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights array
            feature_names: Optional feature names
            timestamps: Optional timestamps
            
        Returns:
            Heatmap data structure
        """
        try:
            # Ensure proper shape
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]  # Take first batch
            
            # Average across heads for main heatmap
            if attention_weights.ndim == 3:
                avg_attention = np.mean(attention_weights, axis=0)
            else:
                avg_attention = attention_weights
            
            # Prepare labels
            seq_len = avg_attention.shape[0]
            
            if timestamps and len(timestamps) == seq_len:
                time_labels = [ts.strftime('%H:%M') for ts in timestamps]
            else:
                time_labels = [f"T-{seq_len-i}" for i in range(seq_len)]
            
            if feature_names and len(feature_names) == seq_len:
                feature_labels = feature_names
            else:
                feature_labels = [f"F{i}" for i in range(seq_len)]
            
            # Calculate statistics
            max_attention = float(np.max(avg_attention))
            min_attention = float(np.min(avg_attention))
            mean_attention = float(np.mean(avg_attention))
            std_attention = float(np.std(avg_attention))
            
            return {
                'heatmap_data': avg_attention.tolist(),
                'x_labels': time_labels,
                'y_labels': feature_labels,
                'shape': avg_attention.shape,
                'statistics': {
                    'max': max_attention,
                    'min': min_attention,
                    'mean': mean_attention,
                    'std': std_attention
                },
                'colormap': self.default_colormap,
                'title': 'Attention Pattern Heatmap'
            }
            
        except Exception as e:
            logger.error(f"Error creating attention heatmap data: {str(e)}")
            return {}
    
    def create_head_analysis_data(
        self,
        attention_weights: np.ndarray,
        head_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create data structure for attention head analysis.
        
        Args:
            attention_weights: Attention weights array
            head_names: Optional head names
            
        Returns:
            Head analysis data structure
        """
        try:
            if attention_weights.ndim < 3:
                return {}
            
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]  # Take first batch
            
            num_heads = min(attention_weights.shape[0], self.max_display_heads)
            head_data = []
            
            for head_idx in range(num_heads):
                head_attention = attention_weights[head_idx]
                
                # Calculate head statistics
                entropy = self._calculate_entropy(head_attention)
                sparsity = self._calculate_sparsity(head_attention)
                focus_strength = np.max(head_attention) / (np.mean(head_attention) + 1e-8)
                
                head_name = head_names[head_idx] if head_names and head_idx < len(head_names) else f"Head {head_idx}"
                
                head_data.append({
                    'head_name': head_name,
                    'head_index': head_idx,
                    'entropy': float(entropy),
                    'sparsity': float(sparsity),
                    'focus_strength': float(focus_strength),
                    'attention_pattern': head_attention.tolist()
                })
            
            return {
                'head_data': head_data,
                'num_heads': num_heads,
                'total_heads': attention_weights.shape[0],
                'analysis_type': 'head_specialization'
            }
            
        except Exception as e:
            logger.error(f"Error creating head analysis data: {str(e)}")
            return {}
    
    def create_temporal_attention_data(
        self,
        attention_weights: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Create data structure for temporal attention visualization.
        
        Args:
            attention_weights: Attention weights array
            timestamps: Optional timestamps
            
        Returns:
            Temporal attention data structure
        """
        try:
            # Average across heads and features to get temporal pattern
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]  # Take first batch
            
            if attention_weights.ndim == 3:
                # Average across heads
                temporal_attention = np.mean(attention_weights, axis=0)
                # Average across destination positions to get source importance
                temporal_attention = np.mean(temporal_attention, axis=1)
            else:
                temporal_attention = np.mean(attention_weights, axis=1)
            
            # Prepare time labels
            seq_len = len(temporal_attention)
            if timestamps and len(timestamps) == seq_len:
                time_labels = [ts.strftime('%H:%M:%S') for ts in timestamps]
                time_values = [ts.timestamp() for ts in timestamps]
            else:
                time_labels = [f"T-{seq_len-i}" for i in range(seq_len)]
                time_values = list(range(seq_len))
            
            # Calculate temporal statistics
            max_idx = np.argmax(temporal_attention)
            recency_weight = np.mean(temporal_attention[-seq_len//4:]) if seq_len >= 4 else temporal_attention[-1]
            early_weight = np.mean(temporal_attention[:seq_len//4]) if seq_len >= 4 else temporal_attention[0]
            recency_bias = recency_weight / (recency_weight + early_weight + 1e-8)
            
            return {
                'temporal_values': temporal_attention.tolist(),
                'time_labels': time_labels,
                'time_values': time_values,
                'peak_attention_time': time_labels[max_idx],
                'peak_attention_value': float(temporal_attention[max_idx]),
                'recency_bias': float(recency_bias),
                'statistics': {
                    'mean': float(np.mean(temporal_attention)),
                    'std': float(np.std(temporal_attention)),
                    'max': float(np.max(temporal_attention)),
                    'min': float(np.min(temporal_attention))
                },
                'chart_type': 'line',
                'title': 'Temporal Attention Pattern'
            }
            
        except Exception as e:
            logger.error(f"Error creating temporal attention data: {str(e)}")
            return {}
    
    def create_feature_importance_chart_data(
        self,
        feature_importance: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Create data structure for feature importance visualization.
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Optional feature names
            top_k: Number of top features to display
            
        Returns:
            Feature importance chart data
        """
        try:
            if not feature_importance:
                return {}
            
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Take top K features
            top_features = sorted_features[:top_k]
            
            feature_names_list = [name for name, _ in top_features]
            importance_values = [importance for _, importance in top_features]
            
            # Separate positive and negative importances
            positive_values = [max(0, val) for val in importance_values]
            negative_values = [min(0, val) for val in importance_values]
            
            return {
                'feature_names': feature_names_list,
                'importance_values': importance_values,
                'positive_values': positive_values,
                'negative_values': negative_values,
                'total_features': len(feature_importance),
                'displayed_features': len(top_features),
                'chart_type': 'horizontal_bar',
                'title': f'Top {len(top_features)} Feature Importance'
            }
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart data: {str(e)}")
            return {}
    
    def create_attention_summary_data(
        self,
        attention_weights: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
        temporal_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create summary data for attention analysis dashboard.
        
        Args:
            attention_weights: Attention weights array
            feature_importance: Optional feature importance scores
            temporal_data: Optional temporal analysis data
            
        Returns:
            Attention summary data
        """
        try:
            # Basic attention statistics
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]  # Take first batch
            
            avg_attention = np.mean(attention_weights, axis=0) if attention_weights.ndim == 3 else attention_weights
            
            # Calculate key metrics
            entropy = self._calculate_entropy(avg_attention)
            sparsity = self._calculate_sparsity(avg_attention)
            concentration = self._calculate_concentration(avg_attention)
            
            # Head specialization if multi-head
            head_specialization = 0.0
            if attention_weights.ndim == 3:
                head_specialization = self._calculate_head_specialization(attention_weights)
            
            summary = {
                'attention_entropy': float(entropy),
                'attention_sparsity': float(sparsity),
                'attention_concentration': float(concentration),
                'head_specialization': float(head_specialization),
                'sequence_length': int(avg_attention.shape[0]),
                'num_heads': int(attention_weights.shape[0]) if attention_weights.ndim == 3 else 1,
                'peak_attention': float(np.max(avg_attention)),
                'attention_spread': float(np.std(avg_attention))
            }
            
            # Add feature importance summary if available
            if feature_importance:
                summary.update({
                    'top_feature': max(feature_importance.items(), key=lambda x: abs(x[1]))[0],
                    'top_feature_importance': max(feature_importance.values(), key=abs),
                    'num_features': len(feature_importance),
                    'importance_concentration': float(np.std(list(feature_importance.values())))
                })
            
            # Add temporal summary if available
            if temporal_data:
                summary.update({
                    'recency_bias': temporal_data.get('recency_bias', 0.0),
                    'peak_attention_time': temporal_data.get('peak_attention_time', 'Unknown'),
                    'temporal_spread': temporal_data.get('statistics', {}).get('std', 0.0)
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating attention summary data: {str(e)}")
            return {}
    
    def export_visualization_config(
        self,
        attention_data: Dict[str, Any],
        output_format: str = 'json'
    ) -> str:
        """
        Export visualization configuration for external dashboards.
        
        Args:
            attention_data: Attention visualization data
            output_format: Output format ('json', 'plotly', 'matplotlib')
            
        Returns:
            Serialized visualization configuration
        """
        try:
            if output_format == 'json':
                return json.dumps(attention_data, indent=2)
            elif output_format == 'plotly':
                return self._create_plotly_config(attention_data)
            elif output_format == 'matplotlib':
                return self._create_matplotlib_config(attention_data)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error exporting visualization config: {str(e)}")
            return "{}"
    
    # Helper methods
    
    def _calculate_entropy(self, attention_weights: np.ndarray) -> float:
        """Calculate attention entropy."""
        flat_attention = attention_weights.flatten()
        flat_attention = flat_attention / (np.sum(flat_attention) + 1e-8)
        entropy = -np.sum(flat_attention * np.log(flat_attention + 1e-8))
        return entropy
    
    def _calculate_sparsity(self, attention_weights: np.ndarray) -> float:
        """Calculate attention sparsity."""
        threshold = 0.1 * np.mean(attention_weights)
        sparsity = np.mean(attention_weights < threshold)
        return sparsity
    
    def _calculate_concentration(self, attention_weights: np.ndarray) -> float:
        """Calculate attention concentration (how focused the attention is)."""
        max_attention = np.max(attention_weights)
        mean_attention = np.mean(attention_weights)
        concentration = max_attention / (mean_attention + 1e-8)
        return concentration
    
    def _calculate_head_specialization(self, attention_weights: np.ndarray) -> float:
        """Calculate head specialization score."""
        if attention_weights.ndim < 3:
            return 0.0
        
        num_heads = attention_weights.shape[0]
        head_patterns = []
        
        for head in range(num_heads):
            head_pattern = np.mean(attention_weights[head], axis=0)
            head_patterns.append(head_pattern)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                corr = np.corrcoef(head_patterns[i], head_patterns[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        # Lower correlation means higher specialization
        avg_correlation = np.mean(correlations) if correlations else 0.0
        specialization = max(0.0, 1.0 - avg_correlation)
        return specialization
    
    def _create_plotly_config(self, attention_data: Dict[str, Any]) -> str:
        """Create Plotly configuration."""
        # Basic Plotly config structure
        config = {
            'data': [],
            'layout': {
                'title': attention_data.get('title', 'Attention Visualization'),
                'showlegend': True
            }
        }
        
        if 'heatmap_data' in attention_data:
            config['data'].append({
                'type': 'heatmap',
                'z': attention_data['heatmap_data'],
                'x': attention_data.get('x_labels', []),
                'y': attention_data.get('y_labels', []),
                'colorscale': attention_data.get('colormap', 'Viridis')
            })
        
        return json.dumps(config, indent=2)
    
    def _create_matplotlib_config(self, attention_data: Dict[str, Any]) -> str:
        """Create Matplotlib configuration."""
        # Basic Matplotlib config structure
        config = {
            'figure_params': {
                'figsize': [12, 8],
                'dpi': 100
            },
            'plots': []
        }
        
        if 'heatmap_data' in attention_data:
            config['plots'].append({
                'type': 'imshow',
                'data': attention_data['heatmap_data'],
                'cmap': attention_data.get('colormap', 'viridis'),
                'aspect': 'auto',
                'title': attention_data.get('title', 'Attention Heatmap')
            })
        
        return json.dumps(config, indent=2)