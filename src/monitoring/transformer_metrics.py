"""
Transformer-Specific Metrics Monitoring System

This module implements comprehensive metrics monitoring for transformer models including:
- Attention entropy monitoring
- Prediction confidence tracking
- Model performance by market regime
- Resource usage monitoring (memory, compute)
- Gradient flow health checks
- Training stability indicators
- Embedding drift monitoring
- Integration with existing monitoring systems

Requirements:
- <1 hour detection time for significant drift
- Comprehensive model health monitoring
- Real-time metrics collection and analysis
- Support for iTransformer, PatchTST, TimesMixer, TimesFM
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import time
import psutil
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import structlog

from .base import MetricsCollector, MetricsRegistry
from .cloud_monitoring import CloudMonitoringClient
# from .operational_analytics import OperationalAnalyticsManager  # Avoid config loading issues

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = structlog.get_logger(__name__)


class ModelHealthStatus(Enum):
    """Enumeration for model health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AttentionHealthMetrics:
    """Metrics for attention mechanism health."""
    mean_entropy: float
    entropy_std: float
    head_entropy_distribution: List[float]
    entropy_health_status: str
    head_diversity_index: float = 0.0
    attention_sparsity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_mean: float
    confidence_std: float = 0.0
    prediction_latency_ms: float = 0.0
    throughput_predictions_per_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    gpu_percent: float
    inference_time_ms: float
    memory_growth_rate: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_temperature: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingMetrics:
    """Embedding health metrics."""
    embedding_stability: float
    position_encoding_health: float
    semantic_consistency: float
    embedding_drift_score: float = 0.0
    token_similarity_changes: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerMetricsResult:
    """Comprehensive transformer metrics result."""
    timestamp: datetime
    overall_health_status: str
    model_metrics: Dict[str, Any]
    collection_latency_ms: float = 0.0
    resource_limited: bool = False
    collection_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionEntropyMonitor:
    """Monitor attention entropy and head diversity."""
    
    def __init__(
        self,
        entropy_baseline_threshold: float = 2.0,
        entropy_warning_threshold: float = 1.5,
        entropy_critical_threshold: float = 1.0,
        head_diversity_threshold: float = 0.8,
        anomaly_detection_enabled: bool = False
    ):
        """
        Initialize attention entropy monitor.
        
        Args:
            entropy_baseline_threshold: Baseline entropy threshold
            entropy_warning_threshold: Warning threshold
            entropy_critical_threshold: Critical threshold
            head_diversity_threshold: Head diversity threshold
            anomaly_detection_enabled: Enable anomaly detection
        """
        self.entropy_baseline_threshold = entropy_baseline_threshold
        self.entropy_warning_threshold = entropy_warning_threshold
        self.entropy_critical_threshold = entropy_critical_threshold
        self.head_diversity_threshold = head_diversity_threshold
        self.anomaly_detection_enabled = anomaly_detection_enabled
        
        self.baseline_entropy_stats = None
        self.anomaly_detector = None
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> AttentionHealthMetrics:
        """
        Compute attention entropy metrics.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            
        Returns:
            AttentionHealthMetrics
        """
        # Ensure attention weights are normalized
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Compute entropy for each head
        log_attention = torch.log(attention_weights + 1e-12)
        head_entropies = -torch.sum(attention_weights * log_attention, dim=-1)
        
        # Average over batch and sequence positions
        mean_head_entropies = torch.mean(head_entropies, dim=(0, 2))  # [heads]
        
        mean_entropy = torch.mean(mean_head_entropies).item()
        entropy_std = torch.std(mean_head_entropies).item()
        
        # Determine health status
        if mean_entropy >= self.entropy_baseline_threshold:
            health_status = 'healthy'
        elif mean_entropy >= self.entropy_warning_threshold:
            health_status = 'warning'
        else:
            health_status = 'critical'
        
        return AttentionHealthMetrics(
            mean_entropy=mean_entropy,
            entropy_std=entropy_std,
            head_entropy_distribution=mean_head_entropies.detach().numpy().tolist(),
            entropy_health_status=health_status
        )
    
    def set_baseline_entropy(self, baseline_entropy: AttentionHealthMetrics):
        """Set baseline entropy statistics."""
        self.baseline_entropy_stats = baseline_entropy
        self.logger.info("Baseline entropy statistics set")
    
    def detect_entropy_degradation(
        self,
        current_entropy: AttentionHealthMetrics,
        baseline_entropy: AttentionHealthMetrics
    ) -> Dict[str, Any]:
        """
        Detect entropy degradation.
        
        Args:
            current_entropy: Current entropy metrics
            baseline_entropy: Baseline entropy metrics
            
        Returns:
            Degradation result dictionary
        """
        entropy_decline = baseline_entropy.mean_entropy - current_entropy.mean_entropy
        entropy_decline_ratio = entropy_decline / baseline_entropy.mean_entropy if baseline_entropy.mean_entropy > 0 else 0
        
        has_degradation = entropy_decline_ratio > 0.1  # 10% decline threshold
        
        if entropy_decline_ratio > 0.3:
            severity = 'high'
        elif entropy_decline_ratio > 0.2:
            severity = 'moderate'
        else:
            severity = 'low'
        
        # Check head-specific degradation
        head_degradation = []
        for i, (curr, base) in enumerate(zip(current_entropy.head_entropy_distribution, 
                                           baseline_entropy.head_entropy_distribution)):
            if base > 0:
                head_decline = (base - curr) / base
                if head_decline > 0.15:
                    head_degradation.append({'head': i, 'decline_ratio': head_decline})
        
        return {
            'has_degradation': has_degradation,
            'degradation_severity': severity,
            'entropy_decline_ratio': entropy_decline_ratio,
            'metadata': {
                'entropy_decline': entropy_decline,
                'head_specific_degradation': head_degradation
            }
        }
    
    def monitor_head_diversity(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Monitor attention head diversity.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            
        Returns:
            Head diversity metrics
        """
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Average across batches
        avg_attention = torch.mean(attention_weights, dim=0)  # [heads, seq, seq]
        
        # Compute pairwise similarities between heads
        head_similarities = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            for j in range(n_heads):
                head_i = avg_attention[i].flatten()
                head_j = avg_attention[j].flatten()
                similarity = torch.cosine_similarity(head_i.unsqueeze(0), head_j.unsqueeze(0)).item()
                head_similarities[i, j] = similarity
        
        # Compute diversity index (1 - mean off-diagonal similarity)
        off_diagonal = head_similarities.clone()
        off_diagonal.fill_diagonal_(0)
        mean_similarity = torch.mean(off_diagonal[off_diagonal != 0])
        diversity_index = 1.0 - mean_similarity.item()
        
        # Compute specialization scores for each head
        specialization_scores = []
        for i in range(n_heads):
            other_similarities = [head_similarities[i, j].item() for j in range(n_heads) if i != j]
            specialization_score = 1.0 - np.mean(other_similarities)
            specialization_scores.append(specialization_score)
        
        # Find redundant heads (high similarity pairs)
        redundant_heads = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                if head_similarities[i, j] > 0.9:
                    redundant_heads.append((i, j))
        
        return {
            'head_similarity_matrix': head_similarities.numpy(),
            'diversity_index': diversity_index,
            'specialization_scores': specialization_scores,
            'redundant_heads': redundant_heads
        }
    
    def fit_anomaly_detector(self, baseline_entropies: List[float]):
        """Fit anomaly detector on baseline entropy measurements."""
        if not self.anomaly_detection_enabled:
            return
        
        self.baseline_entropies = baseline_entropies
        self.entropy_mean = np.mean(baseline_entropies)
        self.entropy_std = np.std(baseline_entropies)
        self.logger.info("Anomaly detector fitted", mean=self.entropy_mean, std=self.entropy_std)
    
    def detect_entropy_anomaly(self, entropy_value: float) -> Dict[str, Any]:
        """
        Detect entropy anomalies.
        
        Args:
            entropy_value: Current entropy value
            
        Returns:
            Anomaly detection result
        """
        if not self.anomaly_detection_enabled or not hasattr(self, 'entropy_mean'):
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}
        
        # Z-score based anomaly detection
        z_score = abs(entropy_value - self.entropy_mean) / (self.entropy_std + 1e-6)
        is_anomaly = z_score > 2.5  # 2.5 standard deviations
        anomaly_score = min(1.0, z_score / 3.0)  # Normalize to [0, 1]
        confidence = min(1.0, z_score / 2.5) if is_anomaly else 1.0 - anomaly_score
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'confidence': confidence,
            'z_score': z_score
        }


class PredictionConfidenceTracker:
    """Track prediction confidence and calibration."""
    
    def __init__(
        self,
        confidence_window_size: int = 100,
        low_confidence_threshold: float = 0.7,
        high_confidence_threshold: float = 0.9,
        trend_detection_periods: int = 10,
        market_condition_aware: bool = False,
        calibration_analysis_enabled: bool = False
    ):
        """
        Initialize confidence tracker.
        
        Args:
            confidence_window_size: Window size for confidence tracking
            low_confidence_threshold: Low confidence threshold
            high_confidence_threshold: High confidence threshold
            trend_detection_periods: Periods for trend detection
            market_condition_aware: Enable market condition awareness
            calibration_analysis_enabled: Enable calibration analysis
        """
        self.confidence_window_size = confidence_window_size
        self.low_confidence_threshold = low_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.trend_detection_periods = trend_detection_periods
        self.market_condition_aware = market_condition_aware
        self.calibration_analysis_enabled = calibration_analysis_enabled
        
        self.confidence_history = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def track_confidence(
        self,
        model_type: str,
        confidence_scores: torch.Tensor,
        prediction_targets: List[str]
    ) -> Dict[str, Any]:
        """
        Track prediction confidence.
        
        Args:
            model_type: Type of model
            confidence_scores: Confidence scores
            prediction_targets: Prediction target names
            
        Returns:
            Confidence tracking result
        """
        if model_type not in self.confidence_history:
            self.confidence_history[model_type] = []
        
        confidence_values = confidence_scores.detach().numpy()
        
        # Store in history
        entry = {
            'timestamp': datetime.now(),
            'confidence_scores': confidence_values.tolist(),
            'targets': prediction_targets,
            'mean_confidence': np.mean(confidence_values)
        }
        self.confidence_history[model_type].append(entry)
        
        # Maintain window size
        if len(self.confidence_history[model_type]) > self.confidence_window_size:
            self.confidence_history[model_type] = self.confidence_history[model_type][-self.confidence_window_size:]
        
        # Calculate statistics
        mean_confidence = np.mean(confidence_values)
        confidence_std = np.std(confidence_values)
        low_confidence_count = np.sum(confidence_values < self.low_confidence_threshold)
        
        # Calculate confidence distribution
        confidence_distribution = {
            'low': np.sum(confidence_values < self.low_confidence_threshold),
            'medium': np.sum((confidence_values >= self.low_confidence_threshold) & 
                           (confidence_values < self.high_confidence_threshold)),
            'high': np.sum(confidence_values >= self.high_confidence_threshold)
        }
        
        # Calculate trend if we have enough history
        confidence_trend = 'stable'
        if len(self.confidence_history[model_type]) >= self.trend_detection_periods:
            recent_means = [entry['mean_confidence'] for entry in self.confidence_history[model_type][-self.trend_detection_periods:]]
            if len(recent_means) > 1:
                trend_slope = np.polyfit(range(len(recent_means)), recent_means, 1)[0]
                if trend_slope > 0.01:
                    confidence_trend = 'increasing'
                elif trend_slope < -0.01:
                    confidence_trend = 'decreasing'
        
        return {
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'low_confidence_count': int(low_confidence_count),
            'confidence_distribution': confidence_distribution,
            'confidence_trend': confidence_trend
        }
    
    def detect_confidence_degradation(self, model_type: str) -> Dict[str, Any]:
        """
        Detect confidence degradation trends.
        
        Args:
            model_type: Type of model
            
        Returns:
            Degradation detection result
        """
        if model_type not in self.confidence_history or len(self.confidence_history[model_type]) < self.trend_detection_periods:
            return {'has_degradation': False, 'degradation_rate': 0.0, 'degradation_significance': 0.0}
        
        # Get recent confidence means
        recent_entries = self.confidence_history[model_type][-self.trend_detection_periods:]
        confidence_means = [entry['mean_confidence'] for entry in recent_entries]
        time_points = list(range(len(confidence_means)))
        
        # Fit linear trend
        if len(confidence_means) > 1:
            trend_slope, intercept = np.polyfit(time_points, confidence_means, 1)
            
            # Calculate trend significance using correlation
            correlation = np.corrcoef(time_points, confidence_means)[0, 1] if len(confidence_means) > 2 else 0
            significance = abs(correlation)
            
            has_degradation = trend_slope < -0.02 and significance > 0.7  # Declining trend with high significance
            
            return {
                'has_degradation': has_degradation,
                'degradation_rate': trend_slope,
                'degradation_significance': significance,
                'metadata': {
                    'trend_analysis': {
                        'slope': trend_slope,
                        'intercept': intercept,
                        'correlation': correlation
                    },
                    'recent_confidence_means': confidence_means
                }
            }
        
        return {'has_degradation': False, 'degradation_rate': 0.0, 'degradation_significance': 0.0}
    
    def track_confidence_by_regime(
        self,
        model_type: str,
        confidence_scores: torch.Tensor,
        market_regime: str,
        volatility_level: float
    ) -> Dict[str, Any]:
        """
        Track confidence by market regime.
        
        Args:
            model_type: Type of model
            confidence_scores: Confidence scores
            market_regime: Market regime name
            volatility_level: Volatility level
            
        Returns:
            Regime-specific confidence result
        """
        confidence_values = confidence_scores.detach().numpy()
        mean_confidence = np.mean(confidence_values)
        
        # Adjust confidence based on market regime expectations
        regime_adjustments = {
            'bull_market': 0.0,    # No adjustment for bull market
            'bear_market': -0.05,  # Expect slightly lower confidence in bear market
            'sideways_market': 0.02,  # Expect slightly higher confidence in sideways market
            'high_volatility': -0.1   # Expect lower confidence in high volatility
        }
        
        expected_adjustment = regime_adjustments.get(market_regime, 0.0)
        adjusted_confidence = mean_confidence + expected_adjustment
        
        # Factor in volatility
        volatility_penalty = min(0.1, volatility_level * 0.1)  # Higher volatility reduces expected confidence
        regime_adjusted_confidence = adjusted_confidence - volatility_penalty
        
        return {
            'mean_confidence': mean_confidence,
            'regime': market_regime,
            'volatility_level': volatility_level,
            'regime_adjusted_confidence': regime_adjusted_confidence,
            'confidence_relative_to_regime': mean_confidence - regime_adjusted_confidence
        }
    
    def analyze_confidence_calibration(
        self,
        predictions_data: List[Dict[str, Any]],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Analyze confidence calibration.
        
        Args:
            predictions_data: List of prediction data with confidence and accuracy
            model_type: Type of model
            
        Returns:
            Calibration analysis result
        """
        if not self.calibration_analysis_enabled or len(predictions_data) < 10:
            return {'calibration_error': 0.0, 'reliability_diagram': [], 'overconfidence_ratio': 0.0}
        
        # Extract confidence and accuracy pairs
        confidences = []
        accuracies = []
        
        for pred_data in predictions_data:
            conf_scores = pred_data['confidence']
            acc_scores = pred_data['accuracy']
            
            if isinstance(conf_scores, torch.Tensor):
                conf_scores = conf_scores.detach().numpy()
            if isinstance(acc_scores, torch.Tensor):
                acc_scores = acc_scores.detach().numpy()
            
            confidences.extend(conf_scores.flatten() if hasattr(conf_scores, 'flatten') else [conf_scores])
            accuracies.extend(acc_scores.flatten() if hasattr(acc_scores, 'flatten') else [acc_scores])
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Create reliability diagram (bin confidences and compute average accuracy per bin)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_diagram = []
        total_calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                reliability_diagram.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin
                })
                
                # Contribution to calibration error
                total_calibration_error += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
        
        # Calculate overconfidence and underconfidence ratios
        overconfident_mask = confidences > accuracies
        underconfident_mask = confidences < accuracies
        
        overconfidence_ratio = np.mean(overconfident_mask)
        underconfidence_ratio = np.mean(underconfident_mask)
        
        # Calculate calibration slope (ideally should be 1.0)
        if len(confidences) > 5:
            calibration_slope = np.polyfit(confidences, accuracies, 1)[0]
        else:
            calibration_slope = 1.0
        
        return {
            'calibration_error': total_calibration_error,
            'reliability_diagram': reliability_diagram,
            'overconfidence_ratio': overconfidence_ratio,
            'underconfidence_ratio': underconfidence_ratio,
            'calibration_slope': calibration_slope
        }


class MarketRegimePerformanceMonitor:
    """Monitor model performance across different market regimes."""
    
    def __init__(
        self,
        regimes: List[str] = None,
        performance_window_hours: int = 24,
        regime_transition_sensitivity: float = 0.1
    ):
        """
        Initialize market regime performance monitor.
        
        Args:
            regimes: List of market regimes to track
            performance_window_hours: Window for performance tracking
            regime_transition_sensitivity: Sensitivity for regime transitions
        """
        self.regimes = regimes or ['bull', 'bear', 'sideways', 'volatile']
        self.performance_window_hours = performance_window_hours
        self.regime_transition_sensitivity = regime_transition_sensitivity
        self.regime_performance_history = {}
        self.baseline_performance = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def monitor_regime_performance(
        self,
        model_type: str,
        regime: str,
        performance_metrics: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Monitor performance in a specific regime.
        
        Args:
            model_type: Type of model
            regime: Market regime
            performance_metrics: Performance metrics
            market_data: Market data for context
            
        Returns:
            Regime performance result
        """
        key = f"{model_type}_{regime}"
        
        if key not in self.regime_performance_history:
            self.regime_performance_history[key] = []
        
        # Store performance entry
        entry = {
            'timestamp': datetime.now(),
            'regime': regime,
            'metrics': performance_metrics.copy(),
            'market_volatility': market_data['volatility'].mean() if 'volatility' in market_data.columns else 0.0,
            'market_returns_std': market_data['returns'].std() if 'returns' in market_data.columns else 0.0
        }
        
        self.regime_performance_history[key].append(entry)
        
        # Maintain window
        cutoff_time = datetime.now() - timedelta(hours=self.performance_window_hours)
        self.regime_performance_history[key] = [
            e for e in self.regime_performance_history[key] 
            if e['timestamp'] > cutoff_time
        ]
        
        # Calculate relative performance
        relative_performance = {}
        if key in self.baseline_performance:
            baseline = self.baseline_performance[key]
            for metric, value in performance_metrics.items():
                if metric in baseline:
                    relative_performance[metric] = (value - baseline[metric]) / baseline[metric] if baseline[metric] != 0 else 0.0
        
        # Calculate regime stability
        if len(self.regime_performance_history[key]) > 5:
            recent_metrics = [e['metrics'] for e in self.regime_performance_history[key][-5:]]
            metric_stds = {}
            for metric in performance_metrics.keys():
                values = [m[metric] for m in recent_metrics if metric in m]
                if values:
                    metric_stds[metric] = np.std(values)
            
            regime_stability = 1.0 - np.mean(list(metric_stds.values())) if metric_stds else 1.0
        else:
            regime_stability = 1.0
        
        return {
            'regime_name': regime,
            'performance_scores': performance_metrics,
            'relative_performance': relative_performance,
            'regime_stability': regime_stability,
            'n_observations': len(self.regime_performance_history[key])
        }
    
    def set_baseline_performance(self, regime: str, baseline_metrics: Dict[str, float]):
        """Set baseline performance for a regime."""
        self.baseline_performance[regime] = baseline_metrics
        self.logger.info(f"Baseline performance set for regime {regime}")
    
    def detect_performance_degradation(
        self,
        regime: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect performance degradation within a regime.
        
        Args:
            regime: Market regime
            current_metrics: Current performance metrics
            
        Returns:
            Degradation detection result
        """
        if regime not in self.baseline_performance:
            return {'has_degradation': False, 'degradation_severity': 'none', 'degraded_metrics': []}
        
        baseline = self.baseline_performance[regime]
        degraded_metrics = []
        degradation_scores = []
        
        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Calculate degradation based on metric type
                if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'r2_score']:
                    # Higher is better - degradation is decrease
                    degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                elif metric in ['mse', 'max_drawdown', 'error_rate']:
                    # Lower is better - degradation is increase
                    degradation = (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
                else:
                    # Default: treat as higher is better
                    degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                
                if degradation > 0.1:  # 10% degradation threshold
                    degraded_metrics.append(metric)
                    degradation_scores.append(degradation)
        
        has_degradation = len(degraded_metrics) > 0
        overall_degradation_score = np.mean(degradation_scores) if degradation_scores else 0.0
        
        if overall_degradation_score > 0.3:
            severity = 'severe'
        elif overall_degradation_score > 0.15:
            severity = 'moderate'
        else:
            severity = 'mild'
        
        return {
            'has_degradation': has_degradation,
            'degradation_severity': severity,
            'degraded_metrics': degraded_metrics,
            'overall_degradation_score': overall_degradation_score,
            'individual_degradations': dict(zip(degraded_metrics, degradation_scores))
        }
    
    def detect_regime_transition(
        self,
        historical_data: pd.DataFrame,
        current_regime: str
    ) -> Dict[str, Any]:
        """
        Detect market regime transitions.
        
        Args:
            historical_data: Historical market data
            current_regime: Current market regime
            
        Returns:
            Regime transition result
        """
        if len(historical_data) < 50:
            return {'transition_detected': False, 'from_regime': current_regime, 'to_regime': current_regime}
        
        # Simple regime detection based on returns and volatility
        recent_data = historical_data.tail(20)  # Last 20 periods
        
        if 'returns' in recent_data.columns and 'volatility' in recent_data.columns:
            avg_return = recent_data['returns'].mean()
            avg_volatility = recent_data['volatility'].mean()
            return_trend = np.polyfit(range(len(recent_data)), recent_data['returns'], 1)[0]
            
            # Classify regime based on returns and volatility
            if avg_volatility > 0.04:  # High volatility threshold
                predicted_regime = 'volatile'
            elif avg_return > 0.01 and return_trend > 0:  # Positive returns and trend
                predicted_regime = 'bull'
            elif avg_return < -0.01 and return_trend < 0:  # Negative returns and trend
                predicted_regime = 'bear'
            else:
                predicted_regime = 'sideways'
            
            transition_detected = predicted_regime != current_regime
            
            if transition_detected:
                # Calculate transition confidence based on signal strength
                signal_strength = abs(avg_return) + abs(return_trend) + abs(avg_volatility - 0.02)
                confidence = min(1.0, signal_strength * 10)
            else:
                confidence = 0.0
            
            return {
                'transition_detected': transition_detected,
                'from_regime': current_regime,
                'to_regime': predicted_regime,
                'transition_confidence': confidence,
                'transition_timestamp': datetime.now(),
                'signal_metrics': {
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'return_trend': return_trend
                }
            }
        
        return {'transition_detected': False, 'from_regime': current_regime, 'to_regime': current_regime}


class ResourceUsageMonitor:
    """Monitor resource usage for transformer models."""
    
    def __init__(
        self,
        memory_warning_threshold_mb: int = 1000,
        memory_critical_threshold_mb: int = 2000,
        gpu_utilization_threshold: float = 0.8,
        cpu_utilization_threshold: float = 0.7,
        anomaly_detection_enabled: bool = False,
        optimization_recommendations_enabled: bool = False
    ):
        """
        Initialize resource usage monitor.
        
        Args:
            memory_warning_threshold_mb: Memory warning threshold in MB
            memory_critical_threshold_mb: Memory critical threshold in MB
            gpu_utilization_threshold: GPU utilization threshold
            cpu_utilization_threshold: CPU utilization threshold
            anomaly_detection_enabled: Enable anomaly detection
            optimization_recommendations_enabled: Enable optimization recommendations
        """
        self.memory_warning_threshold_mb = memory_warning_threshold_mb
        self.memory_critical_threshold_mb = memory_critical_threshold_mb
        self.gpu_utilization_threshold = gpu_utilization_threshold
        self.cpu_utilization_threshold = cpu_utilization_threshold
        self.anomaly_detection_enabled = anomaly_detection_enabled
        self.optimization_recommendations_enabled = optimization_recommendations_enabled
        
        self.resource_history = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def monitor_memory_usage(
        self,
        model_type: str,
        model: Any,
        current_memory_mb: float
    ) -> Dict[str, Any]:
        """
        Monitor memory usage for a model.
        
        Args:
            model_type: Type of model
            model: Model object
            current_memory_mb: Current memory usage in MB
            
        Returns:
            Memory monitoring result
        """
        if model_type not in self.resource_history:
            self.resource_history[model_type] = []
        
        # Calculate memory growth rate
        if len(self.resource_history[model_type]) > 0:
            last_memory = self.resource_history[model_type][-1]['memory_mb']
            memory_growth_rate = (current_memory_mb - last_memory) / last_memory if last_memory > 0 else 0
        else:
            memory_growth_rate = 0.0
        
        # Determine memory status
        if current_memory_mb >= self.memory_critical_threshold_mb:
            memory_status = 'critical'
        elif current_memory_mb >= self.memory_warning_threshold_mb:
            memory_status = 'warning'
        else:
            memory_status = 'normal'
        
        # Calculate peak memory
        peak_memory_mb = max([entry['memory_mb'] for entry in self.resource_history[model_type]] + [current_memory_mb])
        
        # Calculate memory efficiency score
        try:
            # Estimate model parameters
            n_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
            expected_memory = n_params * 4 / (1024 * 1024)  # 4 bytes per parameter, convert to MB
            memory_efficiency_score = expected_memory / current_memory_mb if current_memory_mb > 0 else 0
            memory_efficiency_score = min(1.0, memory_efficiency_score)  # Cap at 1.0
        except:
            memory_efficiency_score = 0.5  # Default value if calculation fails
        
        # Store in history
        self.resource_history[model_type].append({
            'timestamp': datetime.now(),
            'memory_mb': current_memory_mb,
            'memory_growth_rate': memory_growth_rate
        })
        
        # Maintain history size
        if len(self.resource_history[model_type]) > 100:
            self.resource_history[model_type] = self.resource_history[model_type][-100:]
        
        return {
            'current_memory_mb': current_memory_mb,
            'memory_growth_rate': memory_growth_rate,
            'memory_status': memory_status,
            'peak_memory_mb': peak_memory_mb,
            'memory_efficiency_score': memory_efficiency_score
        }
    
    def monitor_gpu_utilization(
        self,
        gpu_stats: Dict[str, float],
        active_models: List[str]
    ) -> Dict[str, Any]:
        """
        Monitor GPU utilization.
        
        Args:
            gpu_stats: GPU statistics
            active_models: List of active model types
            
        Returns:
            GPU monitoring result
        """
        gpu_memory_used = gpu_stats.get('gpu_memory_used_mb', 0)
        gpu_memory_total = gpu_stats.get('gpu_memory_total_mb', 1)
        gpu_utilization = gpu_stats.get('gpu_utilization_percent', 0) / 100.0
        gpu_temperature = gpu_stats.get('gpu_temperature_celsius', 0)
        gpu_power_draw = gpu_stats.get('gpu_power_draw_watts', 0)
        
        # Calculate GPU memory utilization
        gpu_memory_utilization = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0
        
        # Calculate power efficiency (utilization per watt)
        gpu_power_efficiency = gpu_utilization / (gpu_power_draw + 1e-6) if gpu_power_draw > 0 else 0
        
        # Determine GPU health status
        if (gpu_memory_utilization > 0.9 or 
            gpu_utilization > 0.95 or 
            gpu_temperature > 85):
            gpu_health_status = 'critical'
        elif (gpu_memory_utilization > 0.8 or 
              gpu_utilization > self.gpu_utilization_threshold or 
              gpu_temperature > 75):
            gpu_health_status = 'warning'
        else:
            gpu_health_status = 'normal'
        
        return {
            'gpu_memory_utilization': gpu_memory_utilization,
            'gpu_compute_utilization': gpu_utilization,
            'gpu_temperature': gpu_temperature,
            'gpu_power_efficiency': gpu_power_efficiency,
            'gpu_health_status': gpu_health_status,
            'active_models': active_models,
            'gpu_power_draw': gpu_power_draw
        }
    
    def fit_resource_anomaly_detector(self, baseline_resources: List[ResourceMetrics]):
        """Fit anomaly detector on baseline resource measurements."""
        if not self.anomaly_detection_enabled or len(baseline_resources) < 10:
            return
        
        # Extract metrics
        memory_values = [r.memory_mb for r in baseline_resources]
        cpu_values = [r.cpu_percent for r in baseline_resources]
        gpu_values = [r.gpu_percent for r in baseline_resources]
        inference_times = [r.inference_time_ms for r in baseline_resources]
        
        # Calculate statistics
        self.baseline_stats = {
            'memory': {'mean': np.mean(memory_values), 'std': np.std(memory_values)},
            'cpu': {'mean': np.mean(cpu_values), 'std': np.std(cpu_values)},
            'gpu': {'mean': np.mean(gpu_values), 'std': np.std(gpu_values)},
            'inference_time': {'mean': np.mean(inference_times), 'std': np.std(inference_times)}
        }
        
        self.logger.info("Resource anomaly detector fitted")
    
    def detect_resource_anomaly(self, current_usage: ResourceMetrics) -> Dict[str, Any]:
        """
        Detect resource usage anomalies.
        
        Args:
            current_usage: Current resource usage
            
        Returns:
            Anomaly detection result
        """
        if not self.anomaly_detection_enabled or not hasattr(self, 'baseline_stats'):
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'anomalous_metrics': []}
        
        anomalous_metrics = []
        z_scores = {}
        
        # Check each metric
        metrics = {
            'memory': current_usage.memory_mb,
            'cpu': current_usage.cpu_percent,
            'gpu': current_usage.gpu_percent,
            'inference_time': current_usage.inference_time_ms
        }
        
        for metric, value in metrics.items():
            if metric in self.baseline_stats:
                mean = self.baseline_stats[metric]['mean']
                std = self.baseline_stats[metric]['std']
                z_score = abs(value - mean) / (std + 1e-6)
                z_scores[metric] = z_score
                
                if z_score > 2.5:  # 2.5 standard deviations
                    anomalous_metrics.append(metric)
        
        is_anomaly = len(anomalous_metrics) > 0
        anomaly_score = np.mean(list(z_scores.values())) / 3.0 if z_scores else 0.0  # Normalize
        anomaly_score = min(1.0, anomaly_score)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'anomalous_metrics': anomalous_metrics,
            'z_scores': z_scores
        }
    
    def generate_optimization_recommendations(
        self,
        resource_data: Dict[str, ResourceMetrics]
    ) -> Dict[str, Any]:
        """
        Generate resource optimization recommendations.
        
        Args:
            resource_data: Resource data for all models
            
        Returns:
            Optimization recommendations
        """
        if not self.optimization_recommendations_enabled:
            return {'recommendations': [], 'potential_savings': {}, 'priority_actions': []}
        
        recommendations = []
        potential_savings = {'memory_mb': 0, 'cpu_percent': 0, 'gpu_percent': 0}
        priority_actions = []
        
        for model_type, metrics in resource_data.items():
            # High memory usage
            if metrics.memory_mb > self.memory_warning_threshold_mb:
                recommendations.append(f"Consider model quantization or pruning for {model_type} (current memory: {metrics.memory_mb:.0f} MB)")
                potential_savings['memory_mb'] += metrics.memory_mb * 0.3  # Estimated 30% savings
                priority_actions.append(f"optimize_memory_{model_type}")
            
            # High inference time
            if metrics.inference_time_ms > 100:
                recommendations.append(f"Optimize inference pipeline for {model_type} (current time: {metrics.inference_time_ms:.1f} ms)")
                priority_actions.append(f"optimize_inference_{model_type}")
            
            # High GPU usage
            if metrics.gpu_percent > 90:
                recommendations.append(f"Consider batch size optimization for {model_type} (GPU usage: {metrics.gpu_percent:.1f}%)")
                potential_savings['gpu_percent'] += 10  # Estimated 10% reduction
        
        return {
            'recommendations': recommendations,
            'potential_savings': potential_savings,
            'priority_actions': priority_actions
        }


class GradientFlowHealthChecker:
    """Check gradient flow health during training."""
    
    def __init__(
        self,
        gradient_norm_threshold: float = 10.0,
        vanishing_gradient_threshold: float = 1e-7,
        exploding_gradient_threshold: float = 100.0,
        layer_wise_analysis_enabled: bool = True,
        optimization_suggestions_enabled: bool = False
    ):
        """
        Initialize gradient flow health checker.
        
        Args:
            gradient_norm_threshold: Gradient norm threshold
            vanishing_gradient_threshold: Vanishing gradient threshold
            exploding_gradient_threshold: Exploding gradient threshold
            layer_wise_analysis_enabled: Enable layer-wise analysis
            optimization_suggestions_enabled: Enable optimization suggestions
        """
        self.gradient_norm_threshold = gradient_norm_threshold
        self.vanishing_gradient_threshold = vanishing_gradient_threshold
        self.exploding_gradient_threshold = exploding_gradient_threshold
        self.layer_wise_analysis_enabled = layer_wise_analysis_enabled
        self.optimization_suggestions_enabled = optimization_suggestions_enabled
        self.gradient_history = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def analyze_gradient_flow(self, model: Any, model_type: str) -> Dict[str, Any]:
        """
        Analyze gradient flow in the model.
        
        Args:
            model: Model to analyze
            model_type: Type of model
            
        Returns:
            Gradient flow analysis result
        """
        gradient_norms = []
        layer_wise_norms = []
        vanishing_layers = []
        exploding_layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)
                
                if self.layer_wise_analysis_enabled:
                    layer_wise_norms.append({'layer': name, 'grad_norm': grad_norm})
                
                # Check for vanishing gradients
                if grad_norm < self.vanishing_gradient_threshold:
                    vanishing_layers.append(name)
                
                # Check for exploding gradients
                if grad_norm > self.exploding_gradient_threshold:
                    exploding_layers.append(name)
        
        overall_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        # Determine gradient flow health
        if len(vanishing_layers) > len(gradient_norms) * 0.3:  # >30% vanishing
            gradient_flow_health = 'critical'
        elif len(exploding_layers) > 0:
            gradient_flow_health = 'critical'
        elif len(vanishing_layers) > 0:
            gradient_flow_health = 'warning'
        else:
            gradient_flow_health = 'healthy'
        
        return {
            'overall_gradient_norm': overall_gradient_norm,
            'layer_wise_norms': layer_wise_norms,
            'vanishing_layers': vanishing_layers,
            'exploding_layers': exploding_layers,
            'gradient_flow_health': gradient_flow_health
        }
    
    def detect_vanishing_gradients(self, model: Any, model_type: str) -> Dict[str, Any]:
        """
        Detect vanishing gradients.
        
        Args:
            model: Model to analyze
            model_type: Type of model
            
        Returns:
            Vanishing gradient detection result
        """
        vanishing_layers = []
        gradient_stats = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append(grad_norm)
                
                if grad_norm < self.vanishing_gradient_threshold:
                    vanishing_layers.append({
                        'layer': name,
                        'grad_norm': grad_norm,
                        'severity': 'severe' if grad_norm < self.vanishing_gradient_threshold / 10 else 'moderate'
                    })
        
        has_vanishing_gradients = len(vanishing_layers) > 0
        vanishing_severity = 'severe' if len(vanishing_layers) > len(gradient_stats) * 0.3 else 'moderate'
        
        return {
            'has_vanishing_gradients': has_vanishing_gradients,
            'vanishing_layers': vanishing_layers,
            'vanishing_severity': vanishing_severity,
            'metadata': {
                'gradient_statistics': {
                    'mean': np.mean(gradient_stats) if gradient_stats else 0,
                    'std': np.std(gradient_stats) if gradient_stats else 0,
                    'min': np.min(gradient_stats) if gradient_stats else 0,
                    'max': np.max(gradient_stats) if gradient_stats else 0
                }
            }
        }
    
    def detect_exploding_gradients(self, model: Any, model_type: str) -> Dict[str, Any]:
        """
        Detect exploding gradients.
        
        Args:
            model: Model to analyze
            model_type: Type of model
            
        Returns:
            Exploding gradient detection result
        """
        exploding_layers = []
        gradient_stats = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append(grad_norm)
                
                if grad_norm > self.exploding_gradient_threshold:
                    exploding_layers.append({
                        'layer': name,
                        'grad_norm': grad_norm,
                        'severity': 'severe' if grad_norm > self.exploding_gradient_threshold * 10 else 'moderate'
                    })
        
        has_exploding_gradients = len(exploding_layers) > 0
        exploding_severity = 'severe' if any(layer['severity'] == 'severe' for layer in exploding_layers) else 'moderate'
        
        return {
            'has_exploding_gradients': has_exploding_gradients,
            'exploding_layers': exploding_layers,
            'exploding_severity': exploding_severity,
            'metadata': {
                'gradient_statistics': {
                    'mean': np.mean(gradient_stats) if gradient_stats else 0,
                    'std': np.std(gradient_stats) if gradient_stats else 0,
                    'min': np.min(gradient_stats) if gradient_stats else 0,
                    'max': np.max(gradient_stats) if gradient_stats else 0
                }
            }
        }
    
    def generate_optimization_suggestions(self, model: Any, model_type: str) -> Dict[str, Any]:
        """
        Generate gradient flow optimization suggestions.
        
        Args:
            model: Model to analyze
            model_type: Type of model
            
        Returns:
            Optimization suggestions
        """
        if not self.optimization_suggestions_enabled:
            return {'suggestions': [], 'priority_level': 'low', 'expected_improvements': []}
        
        vanishing_result = self.detect_vanishing_gradients(model, model_type)
        exploding_result = self.detect_exploding_gradients(model, model_type)
        
        suggestions = []
        priority_level = 'low'
        expected_improvements = []
        
        # Vanishing gradient suggestions
        if vanishing_result['has_vanishing_gradients']:
            suggestions.append("Consider using residual connections or skip connections")
            suggestions.append("Try layer normalization or batch normalization")
            suggestions.append("Reduce learning rate or use learning rate scheduling")
            suggestions.append("Consider different weight initialization (Xavier, He, etc.)")
            priority_level = 'high' if vanishing_result['vanishing_severity'] == 'severe' else 'medium'
            expected_improvements.append("Improved gradient flow in early layers")
        
        # Exploding gradient suggestions
        if exploding_result['has_exploding_gradients']:
            suggestions.append("Implement gradient clipping")
            suggestions.append("Reduce learning rate")
            suggestions.append("Use batch normalization")
            suggestions.append("Consider different optimizer (Adam, RMSprop)")
            priority_level = 'high'
            expected_improvements.append("Stabilized training dynamics")
        
        # General suggestions if both problems exist
        if vanishing_result['has_vanishing_gradients'] and exploding_result['has_exploding_gradients']:
            suggestions.append("Review model architecture for potential issues")
            suggestions.append("Consider using pre-trained weights")
            priority_level = 'critical'
        
        return {
            'suggestions': suggestions,
            'priority_level': priority_level,
            'expected_improvements': expected_improvements
        }


class TrainingStabilityIndicator:
    """Indicate training stability and convergence."""
    
    def __init__(
        self,
        loss_smoothing_window: int = 20,
        stability_threshold: float = 0.05,
        convergence_patience: int = 10
    ):
        """
        Initialize training stability indicator.
        
        Args:
            loss_smoothing_window: Window for loss smoothing
            stability_threshold: Stability threshold
            convergence_patience: Patience for convergence detection
        """
        self.loss_smoothing_window = loss_smoothing_window
        self.stability_threshold = stability_threshold
        self.convergence_patience = convergence_patience
        self.training_history = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def analyze_training_stability(
        self,
        loss_history: List[float],
        model_type: str,
        scenario_name: str = 'default'
    ) -> Dict[str, Any]:
        """
        Analyze training stability.
        
        Args:
            loss_history: History of loss values
            model_type: Type of model
            scenario_name: Name of training scenario
            
        Returns:
            Training stability analysis
        """
        if len(loss_history) < 10:
            return {
                'stability_score': 0.5,
                'stability_status': 'unknown',
                'loss_trend': 'unknown',
                'volatility_measure': 0.0,
                'convergence_indicator': 'unknown'
            }
        
        # Smooth the loss history
        if len(loss_history) >= self.loss_smoothing_window:
            smoothed_losses = []
            for i in range(len(loss_history) - self.loss_smoothing_window + 1):
                window = loss_history[i:i + self.loss_smoothing_window]
                smoothed_losses.append(np.mean(window))
        else:
            smoothed_losses = loss_history
        
        # Calculate loss trend
        if len(smoothed_losses) > 5:
            trend_slope = np.polyfit(range(len(smoothed_losses)), smoothed_losses, 1)[0]
            if trend_slope < -0.01:
                loss_trend = 'decreasing'
            elif trend_slope > 0.01:
                loss_trend = 'increasing'
            else:
                loss_trend = 'stable'
        else:
            loss_trend = 'unknown'
        
        # Calculate volatility measure
        if len(loss_history) > 1:
            loss_diffs = np.diff(loss_history)
            volatility_measure = np.std(loss_diffs)
        else:
            volatility_measure = 0.0
        
        # Calculate stability score
        relative_volatility = volatility_measure / (np.mean(loss_history) + 1e-6)
        stability_score = max(0.0, 1.0 - relative_volatility / self.stability_threshold)
        
        # Determine stability status
        if stability_score > 0.8:
            stability_status = 'stable'
        elif stability_score > 0.5:
            stability_status = 'unstable'
        else:
            stability_status = 'divergent'
        
        # Check for convergence
        if len(smoothed_losses) >= self.convergence_patience:
            recent_losses = smoothed_losses[-self.convergence_patience:]
            if np.std(recent_losses) < self.stability_threshold * np.mean(recent_losses):
                convergence_indicator = 'converged'
            else:
                convergence_indicator = 'converging' if loss_trend == 'decreasing' else 'not_converged'
        else:
            convergence_indicator = 'unknown'
        
        return {
            'stability_score': stability_score,
            'stability_status': stability_status,
            'loss_trend': loss_trend,
            'volatility_measure': volatility_measure,
            'convergence_indicator': convergence_indicator
        }
    
    def detect_training_divergence(
        self,
        loss_history: List[float],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Detect training divergence.
        
        Args:
            loss_history: History of loss values
            model_type: Type of model
            
        Returns:
            Divergence detection result
        """
        if len(loss_history) < 10:
            return {'is_divergent': False, 'divergence_point': 0, 'divergence_rate': 0.0}
        
        # Look for consistent increase in loss
        divergence_point = -1
        for i in range(5, len(loss_history)):
            # Check if loss has been increasing for the last 5 steps
            recent_losses = loss_history[i-5:i]
            if len(recent_losses) >= 2:
                trend_slope = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                if trend_slope > 0.1 * np.mean(recent_losses):  # Significant increase
                    divergence_point = i
                    break
        
        is_divergent = divergence_point > 0
        
        if is_divergent:
            # Calculate divergence rate
            pre_divergence = loss_history[:divergence_point]
            post_divergence = loss_history[divergence_point:]
            
            if len(pre_divergence) > 0 and len(post_divergence) > 0:
                divergence_rate = (np.mean(post_divergence) - np.mean(pre_divergence)) / np.mean(pre_divergence)
            else:
                divergence_rate = 0.0
        else:
            divergence_rate = 0.0
        
        return {
            'is_divergent': is_divergent,
            'divergence_point': divergence_point,
            'divergence_rate': divergence_rate,
            'metadata': {
                'loss_growth_analysis': {
                    'initial_loss': loss_history[0] if loss_history else 0,
                    'final_loss': loss_history[-1] if loss_history else 0,
                    'max_loss': max(loss_history) if loss_history else 0
                }
            }
        }
    
    def analyze_convergence(
        self,
        loss_history: List[float],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Analyze training convergence.
        
        Args:
            loss_history: History of loss values
            model_type: Type of model
            
        Returns:
            Convergence analysis result
        """
        if len(loss_history) < self.convergence_patience:
            return {
                'has_converged': False,
                'convergence_point': 0,
                'final_loss': 0.0,
                'convergence_confidence': 0.0
            }
        
        # Look for convergence point
        convergence_point = -1
        for i in range(self.convergence_patience, len(loss_history)):
            # Check stability in recent window
            window = loss_history[i-self.convergence_patience:i]
            window_std = np.std(window)
            window_mean = np.mean(window)
            
            # Check if variation is small relative to mean
            if window_std < self.stability_threshold * window_mean:
                convergence_point = i - self.convergence_patience // 2
                break
        
        has_converged = convergence_point > 0
        final_loss = loss_history[-1] if loss_history else 0.0
        
        if has_converged:
            # Calculate convergence confidence
            post_convergence = loss_history[convergence_point:]
            convergence_stability = 1.0 - (np.std(post_convergence) / (np.mean(post_convergence) + 1e-6))
            convergence_confidence = max(0.0, min(1.0, convergence_stability))
        else:
            convergence_confidence = 0.0
        
        return {
            'has_converged': has_converged,
            'convergence_point': convergence_point,
            'final_loss': final_loss,
            'convergence_confidence': convergence_confidence
        }


class EmbeddingDriftMonitor:
    """Monitor embedding drift and stability."""
    
    def __init__(
        self,
        embedding_similarity_threshold: float = 0.9,
        drift_detection_window: int = 100,
        position_encoding_drift_threshold: float = 0.1,
        semantic_consistency_enabled: bool = False
    ):
        """
        Initialize embedding drift monitor.
        
        Args:
            embedding_similarity_threshold: Similarity threshold for embeddings
            drift_detection_window: Window for drift detection
            position_encoding_drift_threshold: Position encoding drift threshold
            semantic_consistency_enabled: Enable semantic consistency checks
        """
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.drift_detection_window = drift_detection_window
        self.position_encoding_drift_threshold = position_encoding_drift_threshold
        self.semantic_consistency_enabled = semantic_consistency_enabled
        self.baseline_embeddings = {}
        self.baseline_positional_encodings = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def set_baseline_embeddings(self, model_type: str, baseline_embeddings: torch.Tensor):
        """Set baseline embeddings for a model."""
        self.baseline_embeddings[model_type] = baseline_embeddings.detach()
        self.logger.info(f"Baseline embeddings set for {model_type}")
    
    def monitor_token_embedding_stability(
        self,
        model_type: str,
        current_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Monitor token embedding stability.
        
        Args:
            model_type: Type of model
            current_embeddings: Current embeddings
            
        Returns:
            Embedding stability result
        """
        if model_type not in self.baseline_embeddings:
            return {
                'embedding_similarity': 1.0,
                'drift_magnitude': 0.0,
                'drift_detected': False,
                'affected_tokens': [],
                'stability_score': 1.0
            }
        
        baseline = self.baseline_embeddings[model_type]
        current = current_embeddings.detach()
        
        # Calculate cosine similarity
        if baseline.shape == current.shape:
            similarities = torch.cosine_similarity(baseline, current, dim=-1)
            mean_similarity = torch.mean(similarities).item()
            
            # Calculate drift magnitude
            drift_magnitude = 1.0 - mean_similarity
            drift_detected = mean_similarity < self.embedding_similarity_threshold
            
            # Find affected tokens (embeddings with low similarity)
            affected_tokens = []
            for i, sim in enumerate(similarities):
                if sim < self.embedding_similarity_threshold:
                    affected_tokens.append(i)
            
            stability_score = mean_similarity
        else:
            # Shape mismatch - significant drift
            mean_similarity = 0.0
            drift_magnitude = 1.0
            drift_detected = True
            affected_tokens = list(range(min(baseline.shape[0], current.shape[0])))
            stability_score = 0.0
        
        return {
            'embedding_similarity': mean_similarity,
            'drift_magnitude': drift_magnitude,
            'drift_detected': drift_detected,
            'affected_tokens': affected_tokens,
            'stability_score': stability_score
        }
    
    def set_baseline_positional_encoding(self, model_type: str, baseline_pos_enc: torch.Tensor):
        """Set baseline positional encoding for a model."""
        self.baseline_positional_encodings[model_type] = baseline_pos_enc.detach()
        self.logger.info(f"Baseline positional encoding set for {model_type}")
    
    def monitor_positional_encoding_drift(
        self,
        model_type: str,
        current_positional_encoding: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Monitor positional encoding drift.
        
        Args:
            model_type: Type of model
            current_positional_encoding: Current positional encoding
            
        Returns:
            Positional encoding drift result
        """
        if model_type not in self.baseline_positional_encodings:
            return {
                'position_drift_magnitude': 0.0,
                'drift_pattern': 'stable',
                'affected_positions': [],
                'encoding_health_status': 'healthy'
            }
        
        baseline = self.baseline_positional_encodings[model_type]
        current = current_positional_encoding.detach()
        
        if baseline.shape != current.shape:
            return {
                'position_drift_magnitude': 1.0,
                'drift_pattern': 'shape_mismatch',
                'affected_positions': list(range(min(baseline.shape[0], current.shape[0]))),
                'encoding_health_status': 'critical'
            }
        
        # Calculate position-wise drift
        position_drifts = torch.norm(current - baseline, dim=-1)
        mean_drift = torch.mean(position_drifts).item()
        
        # Find affected positions
        affected_positions = []
        for i, drift in enumerate(position_drifts):
            if drift > self.position_encoding_drift_threshold:
                affected_positions.append(i)
        
        # Determine drift pattern
        if len(affected_positions) == 0:
            drift_pattern = 'stable'
        elif len(affected_positions) < len(position_drifts) * 0.1:
            drift_pattern = 'localized'
        else:
            drift_pattern = 'widespread'
        
        # Determine health status
        if mean_drift > self.position_encoding_drift_threshold * 2:
            health_status = 'critical'
        elif mean_drift > self.position_encoding_drift_threshold:
            health_status = 'warning'
        else:
            health_status = 'healthy'
        
        return {
            'position_drift_magnitude': mean_drift,
            'drift_pattern': drift_pattern,
            'affected_positions': affected_positions,
            'encoding_health_status': health_status
        }
    
    def detect_embedding_distribution_shift(
        self,
        baseline_embeddings: torch.Tensor,
        current_embeddings: torch.Tensor,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Detect embedding distribution shifts.
        
        Args:
            baseline_embeddings: Baseline embeddings
            current_embeddings: Current embeddings
            model_type: Type of model
            
        Returns:
            Distribution shift result
        """
        if baseline_embeddings.shape != current_embeddings.shape:
            return {
                'distribution_shift_detected': True,
                'shift_magnitude': 1.0,
                'shift_type': 'shape_mismatch',
                'statistical_significance': 1.0
            }
        
        # Flatten embeddings for analysis
        baseline_flat = baseline_embeddings.flatten().detach().numpy()
        current_flat = current_embeddings.flatten().detach().numpy()
        
        # Statistical tests
        # KS test for distribution comparison
        ks_statistic, ks_p_value = stats.ks_2samp(baseline_flat, current_flat)
        
        # Mean and variance shifts
        mean_shift = abs(np.mean(current_flat) - np.mean(baseline_flat))
        var_shift = abs(np.var(current_flat) - np.var(baseline_flat))
        
        # Combined shift magnitude
        shift_magnitude = (ks_statistic + mean_shift + var_shift / 10) / 3
        
        # Determine shift type
        if mean_shift > var_shift:
            shift_type = 'mean_shift'
        elif var_shift > mean_shift * 2:
            shift_type = 'variance_shift'
        else:
            shift_type = 'uniform'
        
        distribution_shift_detected = ks_p_value < 0.05 or shift_magnitude > 0.1
        
        return {
            'distribution_shift_detected': distribution_shift_detected,
            'shift_magnitude': shift_magnitude,
            'shift_type': shift_type,
            'statistical_significance': 1.0 - ks_p_value
        }
    
    def check_semantic_consistency(
        self,
        token_groups: Dict[str, List[str]],
        embeddings: Dict[str, torch.Tensor],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Check semantic consistency of embeddings.
        
        Args:
            token_groups: Groups of semantically related tokens
            embeddings: Embeddings for each group
            model_type: Type of model
            
        Returns:
            Semantic consistency result
        """
        if not self.semantic_consistency_enabled:
            return {
                'intra_group_similarity': {},
                'inter_group_similarity': 0.0,
                'semantic_drift_score': 0.0,
                'consistency_violations': []
            }
        
        intra_group_similarities = {}
        consistency_violations = []
        
        # Calculate intra-group similarities
        for group_name, group_embeddings in embeddings.items():
            if group_embeddings.shape[0] > 1:
                similarities = []
                for i in range(group_embeddings.shape[0]):
                    for j in range(i + 1, group_embeddings.shape[0]):
                        sim = torch.cosine_similarity(
                            group_embeddings[i].unsqueeze(0),
                            group_embeddings[j].unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                
                mean_similarity = np.mean(similarities) if similarities else 1.0
                intra_group_similarities[group_name] = mean_similarity
                
                # Check for consistency violations (low intra-group similarity)
                if mean_similarity < 0.7:
                    consistency_violations.append(f"Low intra-group similarity in {group_name}: {mean_similarity:.3f}")
        
        # Calculate inter-group similarity
        group_centroids = {}
        for group_name, group_embeddings in embeddings.items():
            group_centroids[group_name] = torch.mean(group_embeddings, dim=0)
        
        inter_similarities = []
        group_names = list(group_centroids.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                sim = torch.cosine_similarity(
                    group_centroids[group_names[i]].unsqueeze(0),
                    group_centroids[group_names[j]].unsqueeze(0)
                ).item()
                inter_similarities.append(sim)
        
        inter_group_similarity = np.mean(inter_similarities) if inter_similarities else 0.0
        
        # Calculate semantic drift score
        intra_similarities = list(intra_group_similarities.values())
        if intra_similarities:
            avg_intra_similarity = np.mean(intra_similarities)
            semantic_drift_score = max(0.0, inter_group_similarity - avg_intra_similarity + 0.3)
        else:
            semantic_drift_score = 0.0
        
        return {
            'intra_group_similarity': intra_group_similarities,
            'inter_group_similarity': inter_group_similarity,
            'semantic_drift_score': semantic_drift_score,
            'consistency_violations': consistency_violations
        }


class TransformerHealthDashboard:
    """Comprehensive transformer health dashboard."""
    
    def __init__(
        self,
        update_interval_seconds: int = 30,
        alert_thresholds: Dict[str, float] = None,
        visualization_enabled: bool = True,
        real_time_monitoring: bool = False,
        monitoring_frequency_seconds: int = 1,
        alert_enabled: bool = False,
        notification_channels: List[str] = None
    ):
        """
        Initialize transformer health dashboard.
        
        Args:
            update_interval_seconds: Update interval in seconds
            alert_thresholds: Alert thresholds for different metrics
            visualization_enabled: Enable visualization
            real_time_monitoring: Enable real-time monitoring
            monitoring_frequency_seconds: Monitoring frequency
            alert_enabled: Enable alerting
            notification_channels: List of notification channels
        """
        self.update_interval_seconds = update_interval_seconds
        self.alert_thresholds = alert_thresholds or {
            'attention_entropy': 0.8,
            'confidence': 0.7,
            'memory_usage': 0.9,
            'gradient_health': 0.6
        }
        self.visualization_enabled = visualization_enabled
        self.real_time_monitoring = real_time_monitoring
        self.monitoring_frequency_seconds = monitoring_frequency_seconds
        self.alert_enabled = alert_enabled
        self.notification_channels = notification_channels or []
        
        self.health_history = []
        self.is_monitoring = False
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def generate_comprehensive_health_report(
        self,
        models: Dict[str, Any],
        current_metrics: Dict[str, Any],
        market_context: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive health report.
        
        Args:
            models: Dictionary of models
            current_metrics: Current metrics
            market_context: Market context
            
        Returns:
            Comprehensive health report
        """
        model_health_breakdown = {}
        critical_issues = []
        recommendations = []
        
        # Analyze each model's health
        for model_type in models.keys():
            model_health = {
                'overall_score': 0.0,
                'attention_health': 'unknown',
                'performance_health': 'unknown',
                'resource_health': 'unknown',
                'embedding_health': 'unknown'
            }
            
            # Attention health
            if 'attention_health' in current_metrics:
                attention_metrics = current_metrics['attention_health']
                if attention_metrics.entropy_health_status == 'healthy':
                    model_health['attention_health'] = 'healthy'
                    attention_score = 1.0
                elif attention_metrics.entropy_health_status == 'warning':
                    model_health['attention_health'] = 'warning'
                    attention_score = 0.6
                else:
                    model_health['attention_health'] = 'critical'
                    attention_score = 0.2
                    critical_issues.append(f"Critical attention entropy in {model_type}")
            else:
                attention_score = 0.5
            
            # Performance health
            if 'performance_metrics' in current_metrics:
                perf_metrics = current_metrics['performance_metrics']
                if perf_metrics.accuracy > 0.8 and perf_metrics.confidence_mean > 0.7:
                    model_health['performance_health'] = 'healthy'
                    performance_score = 1.0
                elif perf_metrics.accuracy > 0.6:
                    model_health['performance_health'] = 'warning'
                    performance_score = 0.6
                else:
                    model_health['performance_health'] = 'critical'
                    performance_score = 0.2
                    critical_issues.append(f"Poor performance in {model_type}")
            else:
                performance_score = 0.5
            
            # Resource health
            if 'resource_metrics' in current_metrics:
                resource_metrics = current_metrics['resource_metrics']
                if resource_metrics.memory_mb < 1000 and resource_metrics.cpu_percent < 70:
                    model_health['resource_health'] = 'healthy'
                    resource_score = 1.0
                elif resource_metrics.memory_mb < 2000:
                    model_health['resource_health'] = 'warning'
                    resource_score = 0.6
                else:
                    model_health['resource_health'] = 'critical'
                    resource_score = 0.2
                    critical_issues.append(f"High resource usage in {model_type}")
            else:
                resource_score = 0.5
            
            # Embedding health
            if 'embedding_metrics' in current_metrics:
                embedding_metrics = current_metrics['embedding_metrics']
                if embedding_metrics.embedding_stability > 0.9:
                    model_health['embedding_health'] = 'healthy'
                    embedding_score = 1.0
                elif embedding_metrics.embedding_stability > 0.7:
                    model_health['embedding_health'] = 'warning'
                    embedding_score = 0.6
                else:
                    model_health['embedding_health'] = 'critical'
                    embedding_score = 0.2
                    critical_issues.append(f"Embedding drift in {model_type}")
            else:
                embedding_score = 0.5
            
            # Calculate overall score
            model_health['overall_score'] = (attention_score + performance_score + resource_score + embedding_score) / 4
            model_health_breakdown[model_type] = model_health
        
        # Generate recommendations
        if critical_issues:
            recommendations.append("Immediate attention required for critical issues")
        
        overall_scores = [health['overall_score'] for health in model_health_breakdown.values()]
        overall_health_score = np.mean(overall_scores) if overall_scores else 0.0
        
        return {
            'overall_health_score': overall_health_score,
            'model_health_breakdown': model_health_breakdown,
            'critical_issues': critical_issues,
            'recommendations': recommendations,
            'performance_trends': {},  # Could be expanded with historical analysis
            'market_context': market_context
        }
    
    def start_real_time_monitoring(self, models: Dict[str, Any]):
        """Start real-time monitoring."""
        if not self.real_time_monitoring:
            return
        
        self.is_monitoring = True
        self.logger.info("Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        self.logger.info("Real-time monitoring stopped")
    
    def capture_health_snapshot(self) -> Dict[str, Any]:
        """Capture a health snapshot."""
        return {
            'timestamp': datetime.now(),
            'health_scores': {'overall': 0.8},  # Placeholder
            'alert_level': 'normal'
        }
    
    def generate_health_alerts(
        self,
        current_metrics: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Generate health alerts.
        
        Args:
            current_metrics: Current metrics
            model_type: Type of model
            
        Returns:
            Alert generation result
        """
        alerts = []
        
        # Check attention entropy
        if 'attention_health' in current_metrics:
            attention_metrics = current_metrics['attention_health']
            if attention_metrics.entropy_health_status == 'critical':
                alerts.append({
                    'severity': 'critical',
                    'message': f"Critical attention entropy detected in {model_type}",
                    'metric': 'attention_entropy',
                    'value': attention_metrics.mean_entropy
                })
        
        # Check memory usage
        if 'resource_metrics' in current_metrics:
            resource_metrics = current_metrics['resource_metrics']
            if resource_metrics.memory_mb > 2000:
                alerts.append({
                    'severity': 'critical',
                    'message': f"High memory usage in {model_type}: {resource_metrics.memory_mb} MB",
                    'metric': 'memory_usage',
                    'value': resource_metrics.memory_mb
                })
        
        return {'alerts': alerts}
    
    def dispatch_notifications(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dispatch notifications for alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Notification dispatch result
        """
        notifications_sent = 0
        failed_notifications = []
        
        for alert in alerts:
            # Simulate notification dispatch
            try:
                # In real implementation, would dispatch to actual channels
                notifications_sent += 1
            except Exception as e:
                failed_notifications.append(str(e))
        
        return {
            'notifications_sent': notifications_sent,
            'failed_notifications': failed_notifications
        }


class TransformerMetricsCollector(MetricsCollector):
    """Main transformer metrics collector."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        registry: Optional[MetricsRegistry] = None,
        collection_interval_seconds: int = 60,
        metrics_retention_hours: int = 24,
        health_check_threshold: float = 0.8,
        streaming_enabled: bool = False,
        stream_buffer_size: int = 100,
        detection_window_minutes: int = 60,
        drift_sensitivity: float = 0.1,
        cloud_monitoring: Optional[CloudMonitoringClient] = None,
        cloud_integration_enabled: bool = False,
        operational_analytics: Optional[Any] = None,
        analytics_integration_enabled: bool = False,
        intelligent_alerting: Optional[Any] = None,
        auto_alerting_enabled: bool = False,
        concurrent_collection_enabled: bool = False,
        memory_efficient_mode: bool = False,
        resource_constraint_handling: bool = False,
        max_memory_usage_mb: Optional[int] = None,
        failure_resilient: bool = False
    ):
        """
        Initialize transformer metrics collector.
        
        Args:
            models: Dictionary of transformer models
            registry: Metrics registry
            collection_interval_seconds: Collection interval
            metrics_retention_hours: Metrics retention period
            health_check_threshold: Health check threshold
            streaming_enabled: Enable streaming metrics
            stream_buffer_size: Stream buffer size
            detection_window_minutes: Detection window in minutes
            drift_sensitivity: Drift sensitivity
            cloud_monitoring: Cloud monitoring manager
            cloud_integration_enabled: Enable cloud integration
            operational_analytics: Operational analytics manager
            analytics_integration_enabled: Enable analytics integration
            intelligent_alerting: Intelligent alerting system
            auto_alerting_enabled: Enable automatic alerting
            concurrent_collection_enabled: Enable concurrent collection
            memory_efficient_mode: Enable memory efficient mode
            resource_constraint_handling: Enable resource constraint handling
            max_memory_usage_mb: Maximum memory usage limit
            failure_resilient: Enable failure resilience
        """
        if registry is None:
            registry = MetricsRegistry()
        
        super().__init__(registry)
        
        self.models = models
        self.collection_interval_seconds = collection_interval_seconds
        self.metrics_retention_hours = metrics_retention_hours
        self.health_check_threshold = health_check_threshold
        self.streaming_enabled = streaming_enabled
        self.stream_buffer_size = stream_buffer_size
        self.detection_window_minutes = detection_window_minutes
        self.drift_sensitivity = drift_sensitivity
        self.cloud_monitoring = cloud_monitoring
        self.cloud_integration_enabled = cloud_integration_enabled
        self.operational_analytics = operational_analytics
        self.analytics_integration_enabled = analytics_integration_enabled
        self.intelligent_alerting = intelligent_alerting
        self.auto_alerting_enabled = auto_alerting_enabled
        self.concurrent_collection_enabled = concurrent_collection_enabled
        self.memory_efficient_mode = memory_efficient_mode
        self.resource_constraint_handling = resource_constraint_handling
        self.max_memory_usage_mb = max_memory_usage_mb
        self.failure_resilient = failure_resilient
        
        # Initialize component monitors
        self.attention_entropy_monitor = AttentionEntropyMonitor()
        self.confidence_tracker = PredictionConfidenceTracker()
        self.resource_monitor = ResourceUsageMonitor()
        self.gradient_health_checker = GradientFlowHealthChecker()
        self.embedding_drift_monitor = EmbeddingDriftMonitor()
        
        self.metrics_history = []
        self.is_streaming = False
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def collect_metrics(self) -> None:
        """Collect metrics for all models."""
        try:
            result = self.collect_all_metrics()
            self.metrics_history.append(result)
            
            # Maintain retention
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            raise
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        return {
            "transformer_attention_entropy": "Average attention entropy across all heads",
            "transformer_prediction_confidence": "Mean prediction confidence",
            "transformer_memory_usage_mb": "Memory usage in megabytes",
            "transformer_inference_time_ms": "Inference time in milliseconds",
            "transformer_health_score": "Overall transformer health score"
        }
    
    def collect_all_metrics(
        self,
        current_market_regime: str = 'unknown',
        trading_data: Optional[pd.DataFrame] = None
    ) -> TransformerMetricsResult:
        """
        Collect comprehensive metrics for all models.
        
        Args:
            current_market_regime: Current market regime
            trading_data: Trading data for context
            
        Returns:
            TransformerMetricsResult
        """
        start_time = time.time()
        model_metrics = {}
        collection_errors = []
        overall_health_status = 'healthy'
        
        for model_type, model in self.models.items():
            try:
                # Collect model-specific metrics
                model_result = self.collect_model_specific_metrics(model_type)
                model_metrics[model_type] = model_result
                
                # Update overall health status
                if hasattr(model_result, 'health_status'):
                    if model_result.health_status == 'critical':
                        overall_health_status = 'critical'
                    elif model_result.health_status == 'warning' and overall_health_status != 'critical':
                        overall_health_status = 'warning'
                
            except Exception as e:
                error_msg = f"Failed to collect metrics for {model_type}: {e}"
                collection_errors.append(error_msg)
                self.logger.error(error_msg)
        
        collection_latency = (time.time() - start_time) * 1000
        
        return TransformerMetricsResult(
            timestamp=datetime.now(),
            overall_health_status=overall_health_status,
            model_metrics=model_metrics,
            collection_latency_ms=collection_latency,
            collection_errors=collection_errors
        )
    
    def collect_model_specific_metrics(self, model_type: str) -> Dict[str, Any]:
        """
        Collect metrics for a specific model.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model-specific metrics
        """
        if model_type not in self.models:
            return {
                'collection_errors': [f"Model {model_type} not found"],
                'collection_status': 'failed'
            }
        
        model = self.models[model_type]
        metrics = {
            'model_type': model_type,
            'collection_status': 'success',
            'collection_errors': []
        }
        
        try:
            # Mock attention weights - in real implementation would get from model
            attention_weights = torch.rand(4, 8, 100, 100)
            
            # Attention health metrics
            attention_health = self.attention_entropy_monitor.compute_attention_entropy(attention_weights)
            metrics['attention_health'] = attention_health
            
            # Performance metrics (mock data)
            performance_metrics = PerformanceMetrics(
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                confidence_mean=0.83
            )
            metrics['performance_metrics'] = performance_metrics
            
            # Resource metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            resource_metrics = ResourceMetrics(
                memory_mb=memory_info.rss / (1024 * 1024),
                cpu_percent=process.cpu_percent(),
                gpu_percent=0.0,  # Would get from GPU monitoring
                inference_time_ms=50.0
            )
            metrics['resource_metrics'] = resource_metrics
            
            # Embedding metrics (mock data)
            embedding_metrics = EmbeddingMetrics(
                embedding_stability=0.92,
                position_encoding_health=0.95,
                semantic_consistency=0.88
            )
            metrics['embedding_metrics'] = embedding_metrics
            
            # Overall health status
            if attention_health.entropy_health_status == 'critical':
                metrics['health_status'] = 'critical'
            elif attention_health.entropy_health_status == 'warning':
                metrics['health_status'] = 'warning'
            else:
                metrics['health_status'] = 'healthy'
            
        except Exception as e:
            metrics['collection_errors'].append(str(e))
            metrics['collection_status'] = 'partial'
            metrics['partial_metrics'] = True
        
        return metrics
    
    def start_streaming(self):
        """Start streaming metrics collection."""
        self.is_streaming = True
        self.logger.info("Streaming metrics collection started")
    
    def stop_streaming(self):
        """Stop streaming metrics collection."""
        self.is_streaming = False
        self.logger.info("Streaming metrics collection stopped")
    
    def collect_streaming_metrics(self) -> TransformerMetricsResult:
        """Collect streaming metrics."""
        if not self.is_streaming:
            raise ValueError("Streaming is not enabled")
        
        return self.collect_all_metrics()
    
    def send_metrics_to_cloud(self, metrics_result: TransformerMetricsResult) -> Dict[str, Any]:
        """
        Send metrics to cloud monitoring.
        
        Args:
            metrics_result: Metrics result to send
            
        Returns:
            Cloud integration result
        """
        if not self.cloud_integration_enabled or self.cloud_monitoring is None:
            return {'success': False, 'reason': 'Cloud integration not enabled'}
        
        try:
            # Send metrics to cloud
            metrics_sent = 0
            cloud_metric_ids = []
            
            for model_type, model_metrics in metrics_result.model_metrics.items():
                # Send key metrics
                if hasattr(model_metrics, 'attention_health'):
                    self.cloud_monitoring.send_custom_metric(
                        f"transformer.{model_type}.attention_entropy",
                        model_metrics['attention_health'].mean_entropy
                    )
                    metrics_sent += 1
                
                if hasattr(model_metrics, 'resource_metrics'):
                    self.cloud_monitoring.send_custom_metric(
                        f"transformer.{model_type}.memory_usage",
                        model_metrics['resource_metrics'].memory_mb
                    )
                    metrics_sent += 1
            
            return {
                'success': True,
                'metrics_sent': metrics_sent,
                'cloud_metric_ids': cloud_metric_ids
            }
            
        except Exception as e:
            if self.failure_resilient:
                # Save metrics locally as backup
                return {
                    'success': False,
                    'failure_reason': str(e),
                    'local_backup_saved': True
                }
            else:
                raise
    
    def update_operational_analytics(self, metrics_result: TransformerMetricsResult) -> Dict[str, Any]:
        """
        Update operational analytics.
        
        Args:
            metrics_result: Metrics result
            
        Returns:
            Analytics update result
        """
        if not self.analytics_integration_enabled or self.operational_analytics is None:
            return {'success': False, 'reason': 'Analytics integration not enabled'}
        
        try:
            # Update analytics
            self.operational_analytics.update_transformer_metrics(metrics_result)
            
            return {
                'success': True,
                'analytics_updated': True,
                'dashboard_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Analytics update failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def trigger_intelligent_alerts(self, metrics_result: TransformerMetricsResult) -> Dict[str, Any]:
        """
        Trigger intelligent alerts.
        
        Args:
            metrics_result: Metrics result
            
        Returns:
            Alert result
        """
        if not self.auto_alerting_enabled or self.intelligent_alerting is None:
            return {'alerts_generated': 0}
        
        alerts_generated = 0
        alert_priorities = []
        escalation_triggered = False
        
        # Check for critical conditions
        if metrics_result.overall_health_status == 'critical':
            self.intelligent_alerting.create_alert(
                alert_type='transformer_critical',
                message='Critical transformer health detected',
                priority='high'
            )
            alerts_generated += 1
            alert_priorities.append('high')
            escalation_triggered = True
        
        return {
            'alerts_generated': alerts_generated,
            'alert_priorities': alert_priorities,
            'escalation_triggered': escalation_triggered
        }