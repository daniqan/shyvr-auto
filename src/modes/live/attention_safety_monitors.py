"""
Attention Safety Monitors for Phase 3.2.3.3 RLTE Transformer Integration

This module implements comprehensive attention anomaly detection monitors for
the safety systems, including:

- Attention weight instability detection
- Cross-asset attention anomaly monitoring
- Temporal attention drift detection
- Feature attention collapse detection
- Attention distribution entropy changes
- Head disagreement pattern detection
- Attention sparsity violation monitoring

Each monitor follows the TDD approach and integrates with the existing safety
infrastructure to provide real-time anomaly detection with appropriate safety
actions.
"""

import asyncio
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import deque
from scipy import stats
from scipy.spatial.distance import cosine
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = structlog.get_logger(__name__)


class AttentionAnomalyType(Enum):
    """Types of attention anomalies for safety monitoring"""
    ATTENTION_WEIGHT_INSTABILITY = "attention_weight_instability"
    CROSS_ASSET_ANOMALY = "cross_asset_anomaly"
    TEMPORAL_DRIFT = "temporal_drift"
    FEATURE_COLLAPSE = "feature_collapse"
    ENTROPY_CHANGE = "entropy_change"
    HEAD_DISAGREEMENT = "head_disagreement"
    SPARSITY_VIOLATION = "sparsity_violation"


class SafetyAction(Enum):
    """Safety actions to take when anomalies are detected"""
    EMERGENCY_STOP = "emergency_stop"
    POSITION_REDUCTION = "position_reduction"
    ALERT_GENERATION = "alert_generation"
    MODEL_FALLBACK = "model_fallback"


@dataclass
class AttentionAnomalyResult:
    """Result of attention anomaly detection"""
    anomaly_detected: bool
    anomaly_type: AttentionAnomalyType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    recommended_action: SafetyAction
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseAttentionMonitor:
    """Base class for all attention monitors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.baseline_patterns = None
        self.history = deque(maxlen=1000)  # Store recent patterns
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the monitor"""
        self.is_initialized = True
        self.logger.info("Monitor initialized", monitor=self.__class__.__name__)
        return {"success": True, "monitor": self.__class__.__name__}
    
    async def get_monitoring_capabilities(self) -> Dict[str, Any]:
        """Get monitoring capabilities"""
        return {
            "monitor_type": self.__class__.__name__,
            "is_initialized": self.is_initialized,
            "supports_real_time": True,
            "config": self.config
        }
    
    def _validate_attention_weights(self, attention_weights: torch.Tensor) -> bool:
        """Validate attention weights format and values"""
        if not isinstance(attention_weights, torch.Tensor):
            return False
        
        if len(attention_weights.shape) != 4:
            return False
        
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return False
        
        # Check if weights are properly normalized (sum to 1 along last dimension)
        sums = attention_weights.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5):
            return False
        
        return True


class AttentionWeightStabilityMonitor(BaseAttentionMonitor):
    """Monitor for attention weight instability detection"""
    
    def __init__(
        self,
        stability_threshold: float = 0.3,
        window_size: int = 10,
        detection_sensitivity: float = 0.8,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.detection_sensitivity = detection_sensitivity
        self.stability_history = deque(maxlen=window_size)
    
    async def detect_attention_instability(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect attention weight instability"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute stability score
        stability_score = self._compute_stability_score(attention_weights)
        self.stability_history.append(stability_score)
        
        # Compute instability indicators
        variance_score = self._compute_variance_instability(attention_weights)
        temporal_instability = self._compute_temporal_instability()
        pattern_consistency = self._compute_pattern_consistency(attention_weights)
        
        # Combined instability score
        combined_instability = (
            (1.0 - stability_score) * 0.4 +
            variance_score * 0.3 +
            temporal_instability * 0.2 +
            (1.0 - pattern_consistency) * 0.1
        )
        
        # Determine if anomaly detected
        anomaly_detected = combined_instability > self.stability_threshold
        
        # Determine severity and recommended action
        if combined_instability > 0.9:
            severity = combined_instability
            recommended_action = SafetyAction.EMERGENCY_STOP
        elif combined_instability > 0.6:
            severity = combined_instability
            recommended_action = SafetyAction.POSITION_REDUCTION
        else:
            severity = combined_instability
            recommended_action = SafetyAction.ALERT_GENERATION
        
        # Compute confidence based on detection consistency
        confidence = min(1.0, self.detection_sensitivity * combined_instability)
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.ATTENTION_WEIGHT_INSTABILITY,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "stability_score": stability_score,
                "variance_score": variance_score,
                "temporal_instability": temporal_instability,
                "pattern_consistency": pattern_consistency,
                "combined_instability": combined_instability,
                "window_size": len(self.stability_history),
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_stability_score(self, attention_weights: torch.Tensor) -> float:
        """Compute overall stability score"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute coefficient of variation for each attention head
        cv_scores = []
        for h in range(heads):
            head_weights = attention_weights[0, h]  # [seq, seq]
            
            # Compute coefficient of variation across sequence positions
            for i in range(seq_len):
                attention_dist = head_weights[i]
                mean_val = torch.mean(attention_dist)
                std_val = torch.std(attention_dist)
                cv = std_val / (mean_val + 1e-8)
                cv_scores.append(cv.item())
        
        # Stability is inverse of mean coefficient of variation
        mean_cv = np.mean(cv_scores)
        stability_score = 1.0 / (1.0 + mean_cv)
        
        return stability_score
    
    def _compute_variance_instability(self, attention_weights: torch.Tensor) -> float:
        """Compute variance-based instability"""
        # Compute variance across the attention matrix
        variance = torch.var(attention_weights, dim=-1)
        mean_variance = torch.mean(variance).item()
        
        # Normalize variance to 0-1 range
        # High variance indicates instability
        return min(1.0, mean_variance * 10.0)
    
    def _compute_temporal_instability(self) -> float:
        """Compute temporal instability based on stability history"""
        if len(self.stability_history) < 3:
            return 0.0
        
        # Compute variance in stability scores over time
        stability_values = list(self.stability_history)
        stability_variance = np.var(stability_values)
        
        # High variance in stability indicates temporal instability
        return min(1.0, stability_variance * 20.0)
    
    def _compute_pattern_consistency(self, attention_weights: torch.Tensor) -> float:
        """Compute pattern consistency across attention heads"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        if heads < 2:
            return 1.0
        
        # Compute pairwise cosine similarities between attention heads
        similarities = []
        for h1 in range(heads):
            for h2 in range(h1 + 1, heads):
                head1 = attention_weights[0, h1].flatten()
                head2 = attention_weights[0, h2].flatten()
                
                similarity = 1.0 - cosine(head1.cpu().numpy(), head2.cpu().numpy())
                similarities.append(similarity)
        
        # Pattern consistency is mean similarity
        return np.mean(similarities)


class CrossAssetAttentionMonitor(BaseAttentionMonitor):
    """Monitor for cross-asset attention anomalies"""
    
    def __init__(
        self,
        correlation_threshold: float = 0.8,
        asset_mapping: Optional[Dict[str, Tuple[int, int]]] = None,
        window_size: int = 20,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.correlation_threshold = correlation_threshold
        self.asset_mapping = asset_mapping or {
            "BTC": (0, 40), "ETH": (40, 80), "SOL": (80, 120)
        }
        self.window_size = window_size
        self.correlation_history = deque(maxlen=window_size)
    
    async def detect_cross_asset_anomalies(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect cross-asset attention anomalies"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        # Compute cross-asset correlation matrix
        correlation_matrix = self._compute_cross_asset_correlations(attention_weights)
        self.correlation_history.append(correlation_matrix)
        
        # Detect unexpected correlations
        unexpected_correlations = self._detect_unexpected_correlations(correlation_matrix)
        correlation_strength = self._compute_correlation_strength(correlation_matrix)
        temporal_correlation_change = self._compute_temporal_correlation_change()
        
        # Combined anomaly score
        anomaly_score = (
            len(unexpected_correlations) / len(self.asset_mapping) * 0.5 +
            correlation_strength * 0.3 +
            temporal_correlation_change * 0.2
        )
        
        anomaly_detected = anomaly_score > 0.5  # Threshold for cross-asset anomalies
        
        # Determine severity and action
        if anomaly_score > 0.8:
            severity = anomaly_score
            recommended_action = SafetyAction.POSITION_REDUCTION
        elif anomaly_score > 0.6:
            severity = anomaly_score
            recommended_action = SafetyAction.MODEL_FALLBACK
        else:
            severity = anomaly_score
            recommended_action = SafetyAction.ALERT_GENERATION
        
        confidence = min(1.0, anomaly_score * 1.2)
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.CROSS_ASSET_ANOMALY,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "correlation_matrix": correlation_matrix.tolist(),
                "unexpected_correlations": unexpected_correlations,
                "correlation_strength": correlation_strength,
                "temporal_change": temporal_correlation_change,
                "anomaly_score": anomaly_score,
                "asset_mapping": self.asset_mapping,
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_cross_asset_correlations(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute cross-asset attention correlation matrix"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        n_assets = len(self.asset_mapping)
        
        # Initialize correlation matrix
        correlation_matrix = np.zeros((n_assets, n_assets))
        asset_names = list(self.asset_mapping.keys())
        
        # Compute pairwise correlations
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names):
                if i != j:
                    start1, end1 = self.asset_mapping[asset1]
                    start2, end2 = self.asset_mapping[asset2]
                    
                    # Extract cross-asset attention
                    cross_attention = attention_weights[0, :, start1:end1, start2:end2]
                    correlation = torch.mean(cross_attention).item()
                    correlation_matrix[i, j] = correlation
                else:
                    correlation_matrix[i, j] = 1.0
        
        return correlation_matrix
    
    def _detect_unexpected_correlations(self, correlation_matrix: np.ndarray) -> List[str]:
        """Detect unexpected high correlations"""
        unexpected = []
        asset_names = list(self.asset_mapping.keys())
        
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                correlation = correlation_matrix[i, j]
                if correlation > self.correlation_threshold:
                    unexpected.append(f"{asset_names[i]}-{asset_names[j]}")
        
        return unexpected
    
    def _compute_correlation_strength(self, correlation_matrix: np.ndarray) -> float:
        """Compute overall correlation strength"""
        # Exclude diagonal elements
        off_diagonal = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
        return np.mean(off_diagonal)
    
    def _compute_temporal_correlation_change(self) -> float:
        """Compute temporal change in correlations"""
        if len(self.correlation_history) < 2:
            return 0.0
        
        current = self.correlation_history[-1]
        previous = self.correlation_history[-2]
        
        # Compute change in correlation matrix
        change = np.abs(current - previous)
        return np.mean(change)


class TemporalAttentionDriftMonitor(BaseAttentionMonitor):
    """Monitor for temporal attention drift detection"""
    
    def __init__(
        self,
        drift_threshold: float = 0.3,
        horizon_change_threshold: float = 0.5,
        detection_window: int = 50,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.drift_threshold = drift_threshold
        self.horizon_change_threshold = horizon_change_threshold
        self.detection_window = detection_window
        self.temporal_patterns = deque(maxlen=detection_window)
    
    async def detect_temporal_drift(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect temporal attention drift"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        # Compute temporal attention distribution
        temporal_dist = self._compute_temporal_distribution(attention_weights)
        self.temporal_patterns.append(temporal_dist)
        
        # Compute drift metrics
        recency_bias = self._compute_recency_bias(temporal_dist)
        horizon_change = self._compute_horizon_change(attention_weights)
        drift_magnitude = self._compute_drift_magnitude()
        
        # Combined drift score
        drift_score = (
            recency_bias * 0.4 +
            horizon_change * 0.4 +
            drift_magnitude * 0.2
        )
        
        anomaly_detected = drift_score > self.drift_threshold
        
        # Determine severity and action
        if drift_score > 0.8:
            severity = drift_score
            recommended_action = SafetyAction.MODEL_FALLBACK
        elif drift_score > 0.6:
            severity = drift_score
            recommended_action = SafetyAction.POSITION_REDUCTION
        else:
            severity = drift_score
            recommended_action = SafetyAction.ALERT_GENERATION
        
        confidence = min(1.0, drift_score * 1.3)
        
        # Determine drift direction
        drift_direction = "recent" if recency_bias > 0.7 else "past" if recency_bias < 0.3 else "stable"
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.TEMPORAL_DRIFT,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "recency_bias": recency_bias,
                "horizon_change": horizon_change,
                "drift_magnitude": drift_magnitude,
                "drift_score": drift_score,
                "drift_direction": drift_direction,
                "temporal_distribution": temporal_dist.tolist(),
                "pattern_history_length": len(self.temporal_patterns),
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_temporal_distribution(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute temporal attention distribution"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Average attention to each time step across all queries, heads, and batches
        temporal_attention = torch.mean(attention_weights, dim=(0, 1, 2))
        
        # Normalize to probability distribution
        temporal_dist = temporal_attention / torch.sum(temporal_attention)
        
        return temporal_dist.cpu().numpy()
    
    def _compute_recency_bias(self, temporal_dist: np.ndarray) -> float:
        """Compute recency bias score"""
        seq_len = len(temporal_dist)
        positions = np.arange(seq_len)
        
        # Weighted average position (higher = more recent focus)
        weighted_position = np.sum(positions * temporal_dist)
        
        # Normalize by sequence length
        recency_score = weighted_position / (seq_len - 1)
        
        return recency_score
    
    def _compute_horizon_change(self, attention_weights: torch.Tensor) -> float:
        """Compute attention horizon change"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute effective attention window size
        window_sizes = []
        for h in range(heads):
            for i in range(seq_len):
                attention_dist = attention_weights[0, h, i]
                
                # Find positions that contain 80% of attention mass
                sorted_indices = torch.argsort(attention_dist, descending=True)
                cumsum = torch.cumsum(attention_dist[sorted_indices], dim=0)
                threshold_idx = torch.where(cumsum >= 0.8)[0]
                
                if len(threshold_idx) > 0:
                    effective_window = threshold_idx[0].item() + 1
                else:
                    effective_window = seq_len
                
                window_sizes.append(effective_window)
        
        current_avg_window = np.mean(window_sizes)
        
        # Compare with expected baseline (simplified)
        expected_window = seq_len * 0.3  # Expect ~30% of sequence
        horizon_change = abs(current_avg_window - expected_window) / seq_len
        
        return min(1.0, horizon_change)
    
    def _compute_drift_magnitude(self) -> float:
        """Compute drift magnitude based on temporal pattern history"""
        if len(self.temporal_patterns) < 2:
            return 0.0
        
        current = self.temporal_patterns[-1]
        previous = self.temporal_patterns[-2]
        
        # Compute L1 distance between distributions
        drift_magnitude = np.sum(np.abs(current - previous))
        
        return min(1.0, drift_magnitude * 2.0)


class FeatureAttentionCollapseMonitor(BaseAttentionMonitor):
    """Monitor for feature attention collapse detection"""
    
    def __init__(
        self,
        collapse_threshold: float = 0.8,
        diversity_threshold: float = 0.3,
        feature_mapping: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.collapse_threshold = collapse_threshold
        self.diversity_threshold = diversity_threshold
        self.feature_mapping = feature_mapping or {
            "price": 0, "volume": 1, "volatility": 2, "momentum": 3, "sentiment": 4
        }
    
    async def detect_feature_collapse(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect feature attention collapse"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        # Compute feature importance from attention
        feature_importance = self._compute_feature_importance(attention_weights)
        
        # Detect collapse indicators
        diversity_score = self._compute_feature_diversity(feature_importance)
        dominant_feature = self._identify_dominant_feature(feature_importance)
        collapse_indicator = self._compute_collapse_indicator(feature_importance)
        
        # Combined collapse score
        collapse_score = (
            (1.0 - diversity_score) * 0.5 +
            collapse_indicator * 0.5
        )
        
        anomaly_detected = collapse_score > 0.6  # Lower threshold for feature collapse
        
        # Determine severity and action
        if collapse_score > 0.9:
            severity = collapse_score
            recommended_action = SafetyAction.EMERGENCY_STOP
        elif collapse_score > 0.7:
            severity = collapse_score
            recommended_action = SafetyAction.MODEL_FALLBACK
        else:
            severity = collapse_score
            recommended_action = SafetyAction.POSITION_REDUCTION
        
        confidence = min(1.0, collapse_score * 1.1)
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.FEATURE_COLLAPSE,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "feature_importance": feature_importance.tolist(),
                "diversity_score": diversity_score,
                "dominant_feature": dominant_feature,
                "collapse_indicator": collapse_indicator,
                "collapse_score": collapse_score,
                "feature_mapping": self.feature_mapping,
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_feature_importance(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute feature importance from attention weights"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        n_features = len(self.feature_mapping)
        
        # Simplified feature mapping - assume features repeat across sequence
        features_per_timestep = n_features
        
        if seq_len % features_per_timestep != 0:
            # Fallback: map first n_features positions to features
            feature_attention = torch.mean(attention_weights[:, :, :, :n_features], dim=(0, 1, 2))
        else:
            # Map sequence positions to features
            seq_per_feature = seq_len // features_per_timestep
            feature_attention = torch.zeros(n_features)
            
            for f in range(n_features):
                feature_positions = torch.arange(f, seq_len, features_per_timestep)
                feature_attention[f] = torch.mean(attention_weights[:, :, :, feature_positions])
        
        # Normalize to sum to 1
        feature_importance = feature_attention / torch.sum(feature_attention)
        
        return feature_importance.cpu().numpy()
    
    def _compute_feature_diversity(self, feature_importance: np.ndarray) -> float:
        """Compute feature diversity using entropy"""
        # Compute Shannon entropy
        entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-8))
        max_entropy = np.log(len(feature_importance))
        
        # Normalized entropy as diversity score
        diversity = entropy / max_entropy
        
        return diversity
    
    def _identify_dominant_feature(self, feature_importance: np.ndarray) -> str:
        """Identify the dominant feature"""
        feature_names = list(self.feature_mapping.keys())
        dominant_idx = np.argmax(feature_importance)
        
        if dominant_idx < len(feature_names):
            return feature_names[dominant_idx]
        else:
            return f"feature_{dominant_idx}"
    
    def _compute_collapse_indicator(self, feature_importance: np.ndarray) -> float:
        """Compute collapse indicator based on maximum feature importance"""
        max_importance = np.max(feature_importance)
        
        # High max importance indicates collapse
        return max_importance


class AttentionEntropyMonitor(BaseAttentionMonitor):
    """Monitor for attention distribution entropy changes"""
    
    def __init__(
        self,
        entropy_range: Tuple[float, float] = (1.0, 4.0),
        warning_threshold: float = 0.2,
        critical_threshold: float = 0.4,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.entropy_range = entropy_range
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.entropy_history = deque(maxlen=100)
    
    async def detect_entropy_anomalies(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect attention entropy anomalies"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        # Compute entropy metrics
        entropy_metrics = self._compute_entropy_metrics(attention_weights)
        self.entropy_history.append(entropy_metrics["mean_entropy"])
        
        # Detect entropy anomalies
        entropy_deviation = self._compute_entropy_deviation(entropy_metrics["mean_entropy"])
        entropy_trend = self._compute_entropy_trend()
        
        # Combined anomaly score
        anomaly_score = max(entropy_deviation, entropy_trend * 0.5)
        
        anomaly_detected = anomaly_score > self.warning_threshold
        
        # Determine severity and action
        if anomaly_score > self.critical_threshold:
            severity = anomaly_score
            if entropy_metrics["mean_entropy"] < self.entropy_range[0]:
                recommended_action = SafetyAction.POSITION_REDUCTION  # Over-focused
            else:
                recommended_action = SafetyAction.MODEL_FALLBACK  # Too uniform
        else:
            severity = anomaly_score
            recommended_action = SafetyAction.ALERT_GENERATION
        
        confidence = min(1.0, anomaly_score * 2.0)
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.ENTROPY_CHANGE,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "entropy_metrics": entropy_metrics,
                "entropy_deviation": entropy_deviation,
                "entropy_trend": entropy_trend,
                "anomaly_score": anomaly_score,
                "entropy_range": self.entropy_range,
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_entropy_metrics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute attention entropy metrics"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute entropy for each head
        head_entropies = []
        for h in range(heads):
            for i in range(seq_len):
                attention_dist = attention_weights[0, h, i]
                # Add small epsilon to avoid log(0)
                attention_dist = attention_dist + 1e-12
                entropy = -torch.sum(attention_dist * torch.log(attention_dist))
                head_entropies.append(entropy.item())
        
        return {
            "mean_entropy": np.mean(head_entropies),
            "entropy_std": np.std(head_entropies),
            "min_entropy": np.min(head_entropies),
            "max_entropy": np.max(head_entropies)
        }
    
    def _compute_entropy_deviation(self, current_entropy: float) -> float:
        """Compute deviation from normal entropy range"""
        min_entropy, max_entropy = self.entropy_range
        
        if current_entropy < min_entropy:
            deviation = (min_entropy - current_entropy) / min_entropy
        elif current_entropy > max_entropy:
            deviation = (current_entropy - max_entropy) / max_entropy
        else:
            deviation = 0.0
        
        return min(1.0, deviation)
    
    def _compute_entropy_trend(self) -> float:
        """Compute entropy trend over time"""
        if len(self.entropy_history) < 5:
            return 0.0
        
        recent_entropies = list(self.entropy_history)[-5:]
        
        # Compute trend using linear regression slope
        x = np.arange(len(recent_entropies))
        slope, _, _, _, _ = stats.linregress(x, recent_entropies)
        
        # Normalize slope to 0-1 range
        trend_strength = min(1.0, abs(slope) * 2.0)
        
        return trend_strength


class HeadDisagreementMonitor(BaseAttentionMonitor):
    """Monitor for attention head disagreement patterns"""
    
    def __init__(
        self,
        agreement_threshold: float = 0.5,
        disagreement_severity_threshold: float = 0.7,
        consensus_requirement: float = 0.6,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.agreement_threshold = agreement_threshold
        self.disagreement_severity_threshold = disagreement_severity_threshold
        self.consensus_requirement = consensus_requirement
        self.agreement_history = deque(maxlen=50)
    
    async def detect_head_disagreements(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect head disagreement patterns"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute head agreement metrics
        agreement_score = self._compute_head_agreement(attention_weights)
        self.agreement_history.append(agreement_score)
        
        # Identify conflicting heads
        conflicting_heads = self._identify_conflicting_heads(attention_weights)
        consensus_strength = self._compute_consensus_strength(attention_weights)
        agreement_trend = self._compute_agreement_trend()
        
        # Combined disagreement score
        disagreement_score = (
            (1.0 - agreement_score) * 0.4 +
            len(conflicting_heads) / heads * 0.3 +
            (1.0 - consensus_strength) * 0.2 +
            (1.0 - agreement_trend) * 0.1
        )
        
        anomaly_detected = disagreement_score > 0.5
        
        # Determine severity and action
        if disagreement_score > self.disagreement_severity_threshold:
            severity = disagreement_score
            recommended_action = SafetyAction.MODEL_FALLBACK
        elif disagreement_score > 0.5:
            severity = disagreement_score
            recommended_action = SafetyAction.POSITION_REDUCTION
        else:
            severity = disagreement_score
            recommended_action = SafetyAction.ALERT_GENERATION
        
        confidence = min(1.0, disagreement_score * 1.2)
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.HEAD_DISAGREEMENT,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "agreement_score": agreement_score,
                "conflicting_heads": conflicting_heads,
                "consensus_strength": consensus_strength,
                "agreement_trend": agreement_trend,
                "disagreement_score": disagreement_score,
                "head_count": heads,
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_head_agreement(self, attention_weights: torch.Tensor) -> float:
        """Compute overall head agreement score"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        if heads < 2:
            return 1.0
        
        # Compute pairwise similarities between heads
        similarities = []
        for h1 in range(heads):
            for h2 in range(h1 + 1, heads):
                head1 = attention_weights[0, h1].flatten()
                head2 = attention_weights[0, h2].flatten()
                
                # Cosine similarity
                similarity = 1.0 - cosine(head1.cpu().numpy(), head2.cpu().numpy())
                similarities.append(max(0.0, similarity))  # Ensure non-negative
        
        return np.mean(similarities)
    
    def _identify_conflicting_heads(self, attention_weights: torch.Tensor) -> List[int]:
        """Identify heads that are in conflict with the majority"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        if heads < 3:
            return []
        
        conflicting_heads = []
        
        # Compute each head's similarity to the average pattern
        avg_pattern = torch.mean(attention_weights[0], dim=0).flatten()
        
        for h in range(heads):
            head_pattern = attention_weights[0, h].flatten()
            similarity = 1.0 - cosine(head_pattern.cpu().numpy(), avg_pattern.cpu().numpy())
            
            if similarity < self.agreement_threshold:
                conflicting_heads.append(h)
        
        return conflicting_heads
    
    def _compute_consensus_strength(self, attention_weights: torch.Tensor) -> float:
        """Compute consensus strength across heads"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        # Compute standard deviation across heads for each position
        head_std = torch.std(attention_weights[0], dim=0)
        mean_std = torch.mean(head_std).item()
        
        # Consensus strength is inverse of standard deviation
        consensus_strength = 1.0 / (1.0 + mean_std * 10.0)
        
        return consensus_strength
    
    def _compute_agreement_trend(self) -> float:
        """Compute trend in head agreement over time"""
        if len(self.agreement_history) < 5:
            return 1.0
        
        recent_agreements = list(self.agreement_history)[-5:]
        
        # Simple trend: compare current to average
        current_agreement = recent_agreements[-1]
        avg_agreement = np.mean(recent_agreements[:-1])
        
        # Agreement trend (1.0 = stable/improving, 0.0 = deteriorating)
        if avg_agreement > 0:
            trend = current_agreement / avg_agreement
        else:
            trend = 1.0
        
        return min(1.0, trend)


class AttentionSparsityMonitor(BaseAttentionMonitor):
    """Monitor for attention sparsity violations"""
    
    def __init__(
        self,
        sparsity_range: Tuple[float, float] = (0.1, 0.8),
        violation_threshold: float = 0.9,
        warning_threshold: float = 0.75,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.sparsity_range = sparsity_range
        self.violation_threshold = violation_threshold
        self.warning_threshold = warning_threshold
        self.sparsity_history = deque(maxlen=50)
    
    async def detect_sparsity_violations(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AttentionAnomalyResult:
        """Detect attention sparsity violations"""
        if not self._validate_attention_weights(attention_weights):
            raise ValueError("Invalid attention weights format")
        
        # Compute sparsity metrics
        sparsity_metrics = self._compute_sparsity_metrics(attention_weights)
        self.sparsity_history.append(sparsity_metrics["mean_sparsity"])
        
        # Detect sparsity violations
        sparsity_violation = self._compute_sparsity_violation(sparsity_metrics["mean_sparsity"])
        sparsity_trend = self._compute_sparsity_trend()
        
        # Combined violation score
        violation_score = max(sparsity_violation, sparsity_trend * 0.3)
        
        anomaly_detected = violation_score > 0.5
        
        # Determine severity and action based on sparsity type
        if violation_score > self.violation_threshold:
            severity = violation_score
            recommended_action = SafetyAction.EMERGENCY_STOP
        elif violation_score > self.warning_threshold:
            severity = violation_score
            recommended_action = SafetyAction.POSITION_REDUCTION
        else:
            severity = violation_score
            recommended_action = SafetyAction.ALERT_GENERATION
        
        confidence = min(1.0, violation_score * 1.1)
        
        # Determine violation type
        if sparsity_metrics["mean_sparsity"] > self.sparsity_range[1]:
            violation_type = "too_sparse"
        elif sparsity_metrics["mean_sparsity"] < self.sparsity_range[0]:
            violation_type = "too_dense"
        else:
            violation_type = "none"
        
        return AttentionAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=AttentionAnomalyType.SPARSITY_VIOLATION,
            severity=severity,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata={
                "sparsity_metrics": sparsity_metrics,
                "sparsity_violation": sparsity_violation,
                "sparsity_trend": sparsity_trend,
                "violation_score": violation_score,
                "violation_type": violation_type,
                "sparsity_range": self.sparsity_range,
                "input_metadata": metadata or {}
            }
        )
    
    def _compute_sparsity_metrics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute attention sparsity metrics"""
        batch_size, heads, seq_len, _ = attention_weights.shape
        
        sparsity_scores = []
        
        for h in range(heads):
            for i in range(seq_len):
                attention_dist = attention_weights[0, h, i]
                
                # Compute Gini coefficient as sparsity measure
                sorted_attention = torch.sort(attention_dist)[0]
                n = len(sorted_attention)
                cumsum = torch.cumsum(sorted_attention, dim=0)
                gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
                sparsity_scores.append(gini.item())
        
        return {
            "mean_sparsity": np.mean(sparsity_scores),
            "sparsity_std": np.std(sparsity_scores),
            "min_sparsity": np.min(sparsity_scores),
            "max_sparsity": np.max(sparsity_scores)
        }
    
    def _compute_sparsity_violation(self, current_sparsity: float) -> float:
        """Compute sparsity violation score"""
        min_sparsity, max_sparsity = self.sparsity_range
        
        if current_sparsity > max_sparsity:
            violation = (current_sparsity - max_sparsity) / (1.0 - max_sparsity)
        elif current_sparsity < min_sparsity:
            violation = (min_sparsity - current_sparsity) / min_sparsity
        else:
            violation = 0.0
        
        return min(1.0, violation)
    
    def _compute_sparsity_trend(self) -> float:
        """Compute sparsity trend over time"""
        if len(self.sparsity_history) < 5:
            return 0.0
        
        recent_sparsities = list(self.sparsity_history)[-5:]
        
        # Compute trend using variance
        sparsity_variance = np.var(recent_sparsities)
        
        # High variance indicates unstable sparsity
        return min(1.0, sparsity_variance * 5.0)


class AttentionAnomalyDetector:
    """Integrated attention anomaly detection system"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        monitors: Optional[Dict[str, BaseAttentionMonitor]] = None
    ):
        self.config = config or {}
        self.monitors = monitors or {}
        self.is_initialized = False
        self.detection_history = deque(maxlen=1000)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the integrated anomaly detection system"""
        if not self.monitors:
            # Initialize default monitors
            self.monitors = {
                "stability": AttentionWeightStabilityMonitor(config=self.config),
                "cross_asset": CrossAssetAttentionMonitor(config=self.config),
                "temporal": TemporalAttentionDriftMonitor(config=self.config),
                "feature_collapse": FeatureAttentionCollapseMonitor(config=self.config),
                "entropy": AttentionEntropyMonitor(config=self.config),
                "head_disagreement": HeadDisagreementMonitor(config=self.config),
                "sparsity": AttentionSparsityMonitor(config=self.config)
            }
        
        # Initialize all monitors
        for monitor_name, monitor in self.monitors.items():
            try:
                await monitor.initialize()
                self.logger.info("Monitor initialized", monitor=monitor_name)
            except Exception as e:
                self.logger.error("Failed to initialize monitor", monitor=monitor_name, error=str(e))
                raise
        
        self.is_initialized = True
        self.logger.info("Attention anomaly detector initialized", monitor_count=len(self.monitors))
        
        return {
            "success": True,
            "active_monitors": list(self.monitors.keys()),
            "config": self.config
        }
    
    async def get_monitoring_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive monitoring capabilities"""
        capabilities = {
            "detector_type": "AttentionAnomalyDetector",
            "is_initialized": self.is_initialized,
            "real_time_monitoring": True,
            "emergency_stop_enabled": True,
            "active_monitors": []
        }
        
        for monitor_name, monitor in self.monitors.items():
            monitor_caps = await monitor.get_monitoring_capabilities()
            capabilities["active_monitors"].append({
                "name": monitor_name,
                "capabilities": monitor_caps
            })
        
        return capabilities
    
    async def detect_all_anomalies(
        self,
        attention_weights: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Any] = None
    ) -> "ComprehensiveAnomalyResult":
        """Run comprehensive anomaly detection across all monitors"""
        if not self.is_initialized:
            raise RuntimeError("AttentionAnomalyDetector must be initialized before use")
        
        start_time = datetime.now()
        
        # Run all monitors in parallel
        detection_tasks = []
        for monitor_name, monitor in self.monitors.items():
            if monitor_name == "stability":
                task = monitor.detect_attention_instability(attention_weights, metadata)
            elif monitor_name == "cross_asset":
                task = monitor.detect_cross_asset_anomalies(attention_weights, metadata)
            elif monitor_name == "temporal":
                task = monitor.detect_temporal_drift(attention_weights, metadata)
            elif monitor_name == "feature_collapse":
                task = monitor.detect_feature_collapse(attention_weights, metadata)
            elif monitor_name == "entropy":
                task = monitor.detect_entropy_anomalies(attention_weights, metadata)
            elif monitor_name == "head_disagreement":
                task = monitor.detect_head_disagreements(attention_weights, metadata)
            elif monitor_name == "sparsity":
                task = monitor.detect_sparsity_violations(attention_weights, metadata)
            else:
                continue
            
            detection_tasks.append((monitor_name, task))
        
        # Wait for all detections to complete
        results = {}
        for monitor_name, task in detection_tasks:
            try:
                result = await task
                results[monitor_name] = result
            except Exception as e:
                self.logger.error("Monitor detection failed", monitor=monitor_name, error=str(e))
                # Create error result
                results[monitor_name] = AttentionAnomalyResult(
                    anomaly_detected=False,
                    anomaly_type=AttentionAnomalyType.ATTENTION_WEIGHT_INSTABILITY,
                    severity=0.0,
                    confidence=0.0,
                    recommended_action=SafetyAction.ALERT_GENERATION,
                    metadata={"error": str(e)}
                )
        
        # Aggregate results
        comprehensive_result = self._aggregate_results(results, start_time)
        
        # Store in history
        self.detection_history.append(comprehensive_result)
        
        return comprehensive_result
    
    def _aggregate_results(
        self,
        results: Dict[str, AttentionAnomalyResult],
        start_time: datetime
    ) -> "ComprehensiveAnomalyResult":
        """Aggregate individual monitor results into comprehensive result"""
        detected_anomalies = [result for result in results.values() if result.anomaly_detected]
        
        if not detected_anomalies:
            overall_severity = 0.0
            recommended_action = SafetyAction.ALERT_GENERATION
            emergency_stop_required = False
        else:
            # Compute overall severity (weighted average)
            severity_weights = {
                AttentionAnomalyType.ATTENTION_WEIGHT_INSTABILITY: 0.8,
                AttentionAnomalyType.FEATURE_COLLAPSE: 0.9,
                AttentionAnomalyType.SPARSITY_VIOLATION: 0.7,
                AttentionAnomalyType.CROSS_ASSET_ANOMALY: 0.6,
                AttentionAnomalyType.TEMPORAL_DRIFT: 0.5,
                AttentionAnomalyType.HEAD_DISAGREEMENT: 0.4,
                AttentionAnomalyType.ENTROPY_CHANGE: 0.3
            }
            
            weighted_severity = 0.0
            total_weight = 0.0
            
            for result in detected_anomalies:
                weight = severity_weights.get(result.anomaly_type, 0.5)
                weighted_severity += result.severity * weight
                total_weight += weight
            
            overall_severity = weighted_severity / total_weight if total_weight > 0 else 0.0
            
            # Determine recommended action based on highest severity
            max_severity_result = max(detected_anomalies, key=lambda r: r.severity)
            recommended_action = max_severity_result.recommended_action
            
            # Emergency stop required if any critical anomaly
            emergency_stop_required = any(
                result.recommended_action == SafetyAction.EMERGENCY_STOP
                for result in detected_anomalies
            )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000  # ms
        
        return ComprehensiveAnomalyResult(
            detected_anomalies=detected_anomalies,
            overall_severity=overall_severity,
            recommended_action=recommended_action,
            emergency_stop_required=emergency_stop_required,
            individual_results=results,
            processing_time_ms=processing_time,
            timestamp=start_time
        )


@dataclass
class ComprehensiveAnomalyResult:
    """Result of comprehensive attention anomaly detection"""
    detected_anomalies: List[AttentionAnomalyResult]
    overall_severity: float
    recommended_action: SafetyAction
    emergency_stop_required: bool
    individual_results: Dict[str, AttentionAnomalyResult]
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)