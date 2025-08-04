"""
Multi-Head Attention mechanism optimized for time-series prediction
Includes Flash Attention optimization and XAI visualization utilities
"""

import math
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism optimized for time-series forecasting
    
    Features:
    - Flash Attention optimization for memory efficiency
    - Variable sequence length handling
    - Causal and non-causal masking support
    - Time-series specific optimizations
    - XAI-friendly attention weight extraction
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_flash_attention: bool = True, max_seq_length: int = 1000,
                 bias: bool = True):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        self.max_seq_length = max_seq_length
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(d_model, d_model, bias=bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.d_k)
        
        # Initialize parameters
        self._init_parameters()
        
        logger.debug("MultiHeadAttention initialized",
                    d_model=d_model,
                    n_heads=n_heads,
                    use_flash_attention=use_flash_attention)
    
    def _init_parameters(self):
        """Initialize parameters with appropriate scaling"""
        # Xavier/Glorot initialization for better gradient flow
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            attention_mask: Padding mask [batch_size, seq_len] or [batch_size, seq_len_q, seq_len_k]
            causal_mask: Causal mask [seq_len_q, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        if seq_len_k != seq_len_v:
            raise ValueError(f"Key length ({seq_len_k}) must equal value length ({seq_len_v})")
        
        # Linear projections and reshape for multi-head attention
        Q = self.query_projection(query)  # [batch, seq_len_q, d_model]
        K = self.key_projection(key)      # [batch, seq_len_k, d_model]
        V = self.value_projection(value)  # [batch, seq_len_v, d_model]
        
        # Reshape for multi-head: [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention mechanism
        if self.use_flash_attention and seq_len_q > 64:  # Use flash attention for longer sequences
            output, attention_weights = self._flash_attention(Q, K, V, attention_mask, causal_mask)
        else:
            output, attention_weights = self._standard_attention(Q, K, V, attention_mask, causal_mask)
        
        # Reshape output: [batch, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        output = self.output_projection(output)
        
        return output, attention_weights
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention"""
        # Compute attention scores: [batch, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply masks
        scores = self._apply_masks(scores, attention_mask, causal_mask)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch, n_heads, seq_len_q, d_k]
        
        return output, attention_weights
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None,
                        causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flash Attention implementation for memory efficiency
        Simplified version - in production, would use optimized CUDA kernels
        """
        batch_size, n_heads, seq_len_q, d_k = Q.shape
        seq_len_k = K.shape[2]
        
        # For our implementation, we'll use chunked computation to simulate flash attention benefits
        chunk_size = min(64, seq_len_q)  # Process in chunks
        
        output = torch.zeros_like(Q)
        attention_weights = torch.zeros(batch_size, n_heads, seq_len_q, seq_len_k, 
                                      device=Q.device, dtype=Q.dtype)
        
        for i in range(0, seq_len_q, chunk_size):
            end_i = min(i + chunk_size, seq_len_q)
            
            # Process chunk
            Q_chunk = Q[:, :, i:end_i, :]  # [batch, n_heads, chunk_size, d_k]
            
            # Compute scores for this chunk
            scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale
            
            # Apply masks to chunk
            if attention_mask is not None:
                mask_chunk = self._get_mask_chunk(attention_mask, i, end_i, seq_len_k)
                scores_chunk = scores_chunk.masked_fill(~mask_chunk, float('-inf'))
            
            if causal_mask is not None:
                causal_chunk = causal_mask[i:end_i, :]
                scores_chunk = scores_chunk.masked_fill(~causal_chunk, float('-inf'))
            
            # Softmax and attention for chunk
            attn_chunk = F.softmax(scores_chunk, dim=-1)
            attn_chunk = self.dropout(attn_chunk)
            
            # Apply attention to values
            output_chunk = torch.matmul(attn_chunk, V)
            
            # Store results
            output[:, :, i:end_i, :] = output_chunk
            attention_weights[:, :, i:end_i, :] = attn_chunk
        
        return output, attention_weights
    
    def _apply_masks(self, scores: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None,
                    causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention and causal masks to scores"""
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Padding mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # Custom mask: [batch, seq_len_q, seq_len_k] -> [batch, 1, seq_len_q, seq_len_k]
                mask = attention_mask.unsqueeze(1)
            else:
                raise ValueError(f"Attention mask must have 2 or 3 dimensions, got {attention_mask.dim()}")
            
            scores = scores.masked_fill(~mask, float('-inf'))
        
        if causal_mask is not None:
            # Causal mask: [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        return scores
    
    def _get_mask_chunk(self, attention_mask: torch.Tensor, start_i: int, end_i: int, 
                       seq_len_k: int) -> torch.Tensor:
        """Get mask chunk for flash attention"""
        if attention_mask.dim() == 2:
            # Padding mask
            return attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, end_i - start_i, seq_len_k)
        else:
            # Custom mask
            return attention_mask[:, start_i:end_i, :].unsqueeze(1)


class AttentionVisualization:
    """
    Utilities for visualizing and interpreting attention patterns
    Designed for XAI (Explainable AI) requirements in financial trading
    """
    
    def __init__(self):
        self.logger = structlog.get_logger().bind(component="AttentionVisualization")
    
    def create_attention_heatmap(self, attention_weights: torch.Tensor, 
                               head_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        """
        Create heatmap data for attention visualization
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            head_idx: Which attention head to visualize
            batch_idx: Which batch item to visualize
            
        Returns:
            Dictionary with heatmap data and metadata
        """
        if attention_weights.dim() != 4:
            raise ValueError(f"Expected 4D attention weights, got {attention_weights.dim()}D")
        
        batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        
        if head_idx >= n_heads:
            raise ValueError(f"head_idx ({head_idx}) must be < n_heads ({n_heads})")
        
        if batch_idx >= batch_size:
            raise ValueError(f"batch_idx ({batch_idx}) must be < batch_size ({batch_size})")
        
        # Extract attention matrix for specific head and batch
        attention_matrix = attention_weights[batch_idx, head_idx].detach().cpu().numpy()
        
        return {
            'attention_matrix': attention_matrix,
            'row_labels': [f't-{seq_len_q-1-i}' for i in range(seq_len_q)],  # Time labels
            'col_labels': [f't-{seq_len_k-1-i}' for i in range(seq_len_k)],
            'head_idx': head_idx,
            'batch_idx': batch_idx,
            'shape': attention_matrix.shape
        }
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Analyze attention patterns for interpretability
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            
        Returns:
            Dictionary with pattern analysis metrics
        """
        batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        
        # Convert to numpy for analysis
        weights_np = attention_weights.detach().cpu().numpy()
        
        # Calculate entropy (attention concentration)
        entropy = -np.sum(weights_np * np.log(weights_np + 1e-8), axis=-1)
        avg_entropy = np.mean(entropy)
        
        # Calculate sparsity (how focused the attention is)
        max_attention = np.max(weights_np, axis=-1)
        sparsity = np.mean(max_attention)
        
        # Calculate locality (how much attention focuses on nearby positions)
        if seq_len_q == seq_len_k:  # Self-attention case
            distance_matrix = np.abs(np.arange(seq_len_q)[:, None] - np.arange(seq_len_k)[None, :])
            weighted_distance = np.sum(weights_np * distance_matrix[None, None, :, :], axis=(-2, -1))
            locality = 1.0 / (1.0 + np.mean(weighted_distance) / seq_len_q)
        else:
            locality = 0.5  # Neutral for cross-attention
        
        # Calculate head diversity (how different attention heads are)
        head_similarities = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                # Correlation between heads
                head_i = weights_np[:, i].flatten()
                head_j = weights_np[:, j].flatten()
                corr = np.corrcoef(head_i, head_j)[0, 1]
                if not np.isnan(corr):
                    head_similarities.append(abs(corr))
        
        head_diversity = 1.0 - np.mean(head_similarities) if head_similarities else 1.0
        
        return {
            'entropy': float(avg_entropy),
            'sparsity': float(sparsity),
            'locality': float(locality),
            'head_diversity': float(head_diversity)
        }
    
    def get_head_specialization(self, attention_weights: torch.Tensor) -> Dict[int, Dict[str, Any]]:
        """
        Analyze what each attention head specializes in
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            
        Returns:
            Dictionary mapping head index to specialization info
        """
        batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        weights_np = attention_weights.detach().cpu().numpy()
        
        head_specialization = {}
        
        for head_idx in range(n_heads):
            head_weights = weights_np[:, head_idx]  # [batch, seq_len_q, seq_len_k]
            
            # Average across batches
            avg_head_weights = np.mean(head_weights, axis=0)
            
            # Analyze attention pattern
            max_positions = np.unravel_index(np.argmax(avg_head_weights), avg_head_weights.shape)
            concentration = np.max(avg_head_weights, axis=-1).mean()
            
            # Determine focus type
            if seq_len_q == seq_len_k:  # Self-attention
                # Check if it's focusing on recent positions (recency bias)
                recent_attention = np.mean(avg_head_weights[:, -5:])  # Last 5 positions
                distant_attention = np.mean(avg_head_weights[:, :-5])  # Earlier positions
                
                if recent_attention > distant_attention * 1.5:
                    focus_type = "recent_focus"
                elif distant_attention > recent_attention * 1.5:
                    focus_type = "long_range_focus"
                else:
                    focus_type = "distributed_focus"
            else:
                focus_type = "cross_attention"
            
            # Find dominant positions (top 5)
            flat_weights = avg_head_weights.flatten()
            top_indices = np.argsort(flat_weights)[-5:]
            dominant_positions = [np.unravel_index(idx, avg_head_weights.shape) for idx in top_indices]
            
            head_specialization[head_idx] = {
                'focus_type': focus_type,
                'concentration_score': float(concentration),
                'dominant_positions': [(int(pos[0]), int(pos[1])) for pos in dominant_positions]
            }
        
        return head_specialization
    
    def extract_temporal_patterns(self, attention_weights: torch.Tensor,
                                sequence_timestamps: List[int]) -> Dict[str, float]:
        """
        Extract temporal patterns from attention for time-series analysis
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            sequence_timestamps: List of timestamps for each position
            
        Returns:
            Dictionary with temporal pattern metrics
        """
        batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        weights_np = attention_weights.detach().cpu().numpy()
        
        if len(sequence_timestamps) != seq_len_k:
            raise ValueError(f"Timestamps length ({len(sequence_timestamps)}) must match seq_len_k ({seq_len_k})")
        
        # Calculate recency bias
        time_distances = np.array(sequence_timestamps)
        if seq_len_q == seq_len_k:
            # For self-attention, calculate how much attention goes to recent vs distant
            recent_mask = time_distances >= np.percentile(time_distances, 80)  # Recent 20%
            distant_mask = time_distances <= np.percentile(time_distances, 20)  # Distant 20%
            
            recent_attention = np.mean(weights_np[:, :, :, recent_mask])
            distant_attention = np.mean(weights_np[:, :, :, distant_mask])
            
            recency_bias = recent_attention / (recent_attention + distant_attention + 1e-8)
        else:
            recency_bias = 0.5  # Neutral for cross-attention
        
        # Calculate periodicity (simplified)
        # Look for periodic patterns in attention weights
        avg_weights = np.mean(weights_np, axis=(0, 1, 2))  # Average across all dimensions
        
        # Simple periodicity detection using autocorrelation
        if len(avg_weights) > 10:
            autocorr = np.correlate(avg_weights, avg_weights, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (excluding the first peak at lag 0)
            if len(autocorr) > 1:
                peak_idx = np.argmax(autocorr[1:]) + 1
                periodicity = float(peak_idx)
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0
        
        # Calculate long-range dependencies
        if seq_len_q == seq_len_k:
            # Create distance matrix
            distance_matrix = np.abs(np.arange(seq_len_q)[:, None] - np.arange(seq_len_k)[None, :])
            long_range_mask = distance_matrix > seq_len_q // 2
            
            long_range_attention = np.mean(weights_np[:, :, :, :][..., long_range_mask])
            total_attention = np.mean(weights_np)
            
            long_range_dependencies = long_range_attention / (total_attention + 1e-8)
        else:
            long_range_dependencies = 0.5  # Neutral for cross-attention
        
        return {
            'recency_bias': float(np.clip(recency_bias, 0, 1)),
            'periodicity': periodicity,
            'long_range_dependencies': float(np.clip(long_range_dependencies, 0, 1))
        }
    
    def analyze_cross_asset_attention(self, attention_weights: torch.Tensor,
                                    asset_names: List[str]) -> Dict[str, Any]:
        """
        Analyze attention patterns across multiple assets
        
        Args:
            attention_weights: Attention weights [n_assets, n_heads, seq_len, seq_len]
            asset_names: Names of assets (BTC, ETH, SOL, etc.)
            
        Returns:
            Dictionary with cross-asset attention analysis
        """
        n_assets, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        
        if len(asset_names) != n_assets:
            raise ValueError(f"Asset names length ({len(asset_names)}) must match n_assets ({n_assets})")
        
        weights_np = attention_weights.detach().cpu().numpy()
        
        # Calculate correlation matrix between assets
        asset_vectors = []
        for asset_idx in range(n_assets):
            # Flatten attention pattern for this asset
            asset_vector = weights_np[asset_idx].flatten()
            asset_vectors.append(asset_vector)
        
        correlation_matrix = np.corrcoef(asset_vectors)
        
        # Identify leading indicators (assets that others pay attention to)
        # This is a simplified heuristic - in practice would need more sophisticated analysis
        leading_scores = {}
        for i, asset in enumerate(asset_names):
            # Assets with higher average correlation might be leading indicators
            leading_scores[asset] = float(np.mean(correlation_matrix[i]))
        
        # Sort by leading score
        leading_indicators = sorted(leading_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze attention flow patterns
        attention_flow = {}
        for i, source_asset in enumerate(asset_names):
            flow_pattern = {}
            for j, target_asset in enumerate(asset_names):
                if i != j:
                    flow_strength = correlation_matrix[i, j]
                    flow_pattern[target_asset] = float(flow_strength)
            attention_flow[source_asset] = flow_pattern
        
        return {
            'correlation_matrix': torch.tensor(correlation_matrix),
            'leading_indicators': leading_indicators,
            'attention_flow': attention_flow
        }
    
    def generate_explanations(self, attention_weights: torch.Tensor,
                            feature_names: List[str],
                            prediction_timestamp: int) -> Dict[str, Any]:
        """
        Generate human-readable explanations of attention patterns
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            feature_names: Names of features/time positions
            prediction_timestamp: Index of the position being predicted
            
        Returns:
            Dictionary with human-readable explanations
        """
        batch_size, n_heads, seq_len_q, seq_len_k = attention_weights.shape
        
        if len(feature_names) != seq_len_k:
            raise ValueError(f"Feature names length ({len(feature_names)}) must match seq_len_k ({seq_len_k})")
        
        weights_np = attention_weights.detach().cpu().numpy()
        
        # Focus on the prediction timestamp
        if prediction_timestamp >= seq_len_q:
            prediction_timestamp = seq_len_q - 1
        
        # Get attention weights for the prediction position
        pred_attention = weights_np[:, :, prediction_timestamp, :]  # [batch, n_heads, seq_len_k]
        
        # Average across batches and heads for overall importance
        avg_attention = np.mean(pred_attention, axis=(0, 1))
        
        # Get top features
        top_indices = np.argsort(avg_attention)[-10:][::-1]  # Top 10 features
        
        most_important_features = []
        for idx in top_indices:
            importance_score = avg_attention[idx]
            if importance_score > 0.01:  # Only include meaningful features
                most_important_features.append({
                    'feature_name': feature_names[idx],
                    'importance_score': float(importance_score),
                    'time_position': int(idx)
                })
        
        # Analyze temporal focus
        if seq_len_q == seq_len_k:
            recent_attention = np.mean(avg_attention[-5:])  # Last 5 positions
            mid_attention = np.mean(avg_attention[len(avg_attention)//2:len(avg_attention)//2+5])
            distant_attention = np.mean(avg_attention[:5])  # First 5 positions
            
            if recent_attention > max(mid_attention, distant_attention) * 1.2:
                temporal_focus = "Recent data is most important for this prediction"
            elif distant_attention > max(mid_attention, recent_attention) * 1.2:
                temporal_focus = "Historical patterns are driving this prediction"
            else:
                temporal_focus = "Model considers both recent and historical information"
        else:
            temporal_focus = "Cross-attention pattern detected"
        
        # Generate summary
        top_feature = most_important_features[0]['feature_name'] if most_important_features else "Unknown"
        attention_summary = f"Primary focus on {top_feature} with {len(most_important_features)} key factors identified"
        
        return {
            'most_important_features': most_important_features,
            'temporal_focus': temporal_focus,
            'attention_summary': attention_summary
        }
    
    def generate_compliance_report(self, attention_weights: torch.Tensor,
                                 decision_outcome: str,
                                 confidence_score: float) -> Dict[str, Any]:
        """
        Generate regulatory compliance report for attention-based decisions
        
        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            decision_outcome: Trading decision (BUY/SELL/HOLD)
            confidence_score: Model confidence in decision
            
        Returns:
            Dictionary with compliance information
        """
        patterns = self.analyze_attention_patterns(attention_weights)
        
        # Calculate transparency score based on attention patterns
        # Higher entropy and lower sparsity = more transparent (considers many factors)
        max_entropy = math.log(attention_weights.shape[-1])
        normalized_entropy = patterns['entropy'] / max_entropy
        transparency_score = (normalized_entropy + (1 - patterns['sparsity'])) / 2
        
        # Generate decision rationale
        if patterns['sparsity'] > 0.8:
            rationale = f"Decision based on highly focused analysis of key market indicators. "
        elif patterns['sparsity'] < 0.3:
            rationale = f"Decision considers a broad range of market factors with distributed attention. "
        else:
            rationale = f"Decision balances focused analysis with comprehensive market assessment. "
        
        rationale += f"Model shows {'high' if confidence_score > 0.8 else 'moderate' if confidence_score > 0.6 else 'low'} confidence in {decision_outcome} recommendation."
        
        # Identify key factors
        key_factors = []
        if patterns['locality'] > 0.7:
            key_factors.append("Recent market movements")
        if patterns['head_diversity'] > 0.6:
            key_factors.append("Multiple analytical perspectives")
        if transparency_score > 0.6:
            key_factors.append("Comprehensive factor analysis")
        
        # Identify risk factors
        risk_factors = []
        if confidence_score < 0.6:
            risk_factors.append("Lower model confidence")
        if patterns['sparsity'] > 0.9:
            risk_factors.append("Very narrow focus may miss important signals")
        if patterns['head_diversity'] < 0.3:
            risk_factors.append("Limited analytical diversity")
        
        return {
            'decision_rationale': rationale,
            'key_factors': key_factors,
            'risk_factors': risk_factors,
            'model_confidence': confidence_score,
            'transparency_score': float(transparency_score),
            'attention_metrics': patterns
        }