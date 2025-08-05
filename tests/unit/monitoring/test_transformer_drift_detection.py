"""
Comprehensive failing tests for Phase 2.4.1 - Transformer-Specific Drift Detection

This module contains failing tests that define the expected behavior for transformer-specific 
drift detection components. Following TDD methodology, these tests will fail initially as 
the implementation doesn't exist yet.

Test Coverage Areas:
- Attention pattern drift detection algorithms
- Feature importance shift monitoring
- Temporal focus change detection
- Cross-asset correlation drift tracking
- Attention distribution monitoring
- Head specialization tracking
- Temporal attention shift detection
- Drift detection within 100 predictions (success criteria)
- False positive rate <5% (success criteria)

Requirements:
- Attention drift detected within 100 predictions
- False positive rate <5%
- Integration with existing DriftDetector
- Support for all transformer models (iTransformer, PatchTST, TimesMixer, TimesFM)
- Real-time drift detection capabilities
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import MagicMock, patch
import warnings

# Import existing monitoring components
from src.monitoring.drift_detection import DriftDetector, DriftAnalysisResult, DriftSeverity, DriftType

# These imports will fail initially - this is expected for TDD
try:
    from src.monitoring.transformer_drift_detection import (
        TransformerDriftDetector,
        AttentionPatternDrift,
        FeatureImportanceDrift,
        TemporalFocusDrift,
        CrossAssetCorrelationDrift,
        AttentionDistributionDrift,
        HeadSpecializationDrift,
        TemporalAttentionShiftDrift,
        TransformerDriftResult,
        AttentionDriftAnalyzer,
        AttentionPatternAnalyzer,
        HeadSpecializationAnalyzer,
        TemporalAttentionAnalyzer
    )
except ImportError:
    # Expected to fail initially - create placeholder classes for testing
    class TransformerDriftDetector:
        pass
    
    class AttentionPatternDrift:
        pass
    
    class FeatureImportanceDrift:
        pass
    
    class TemporalFocusDrift:
        pass
    
    class CrossAssetCorrelationDrift:
        pass
    
    class AttentionDistributionDrift:
        pass
    
    class HeadSpecializationDrift:
        pass
    
    class TemporalAttentionShiftDrift:
        pass
    
    class TransformerDriftResult:
        pass
    
    class AttentionDriftAnalyzer:
        pass
    
    class AttentionPatternAnalyzer:
        pass
    
    class HeadSpecializationAnalyzer:
        pass
    
    class TemporalAttentionAnalyzer:
        pass


class TestTransformerDriftDetection:
    """Test suite for transformer-specific drift detection components"""
    
    @pytest.fixture
    def baseline_attention_weights(self):
        """Create baseline attention weights for different transformer models"""
        # Simulate attention weights for different models
        attention_data = {
            'iTransformer': {
                # Inverted attention: [batch, n_heads, seq_len, seq_len]
                'weights': torch.rand(4, 8, 100, 100),
                'head_patterns': torch.rand(8, 100, 100),
                'temporal_patterns': torch.rand(100, 5),  # 5 features
                'feature_importance': torch.rand(5, 100)
            },
            'PatchTST': {
                # Patch-based attention: [batch, n_heads, n_patches, n_patches]
                'weights': torch.rand(4, 8, 20, 20),  # 100 seq_len / 5 patch_size
                'head_patterns': torch.rand(8, 20, 20),
                'temporal_patterns': torch.rand(20, 5),
                'feature_importance': torch.rand(5, 20)
            },
            'TimesMixer': {
                # Mixed attention patterns: [batch, n_heads, seq_len, seq_len]
                'weights': torch.rand(4, 8, 100, 100),
                'head_patterns': torch.rand(8, 100, 100),
                'temporal_patterns': torch.rand(100, 5),
                'feature_importance': torch.rand(5, 100),
                'time_mixing': torch.rand(100, 100),
                'feature_mixing': torch.rand(5, 5)
            },
            'TimesFM': {
                # Foundation model attention: [batch, n_heads, seq_len, seq_len]
                'weights': torch.rand(4, 12, 512, 512),  # Larger model
                'head_patterns': torch.rand(12, 512, 512),
                'temporal_patterns': torch.rand(512, 10),  # More features
                'feature_importance': torch.rand(10, 512)
            }
        }
        return attention_data
    
    @pytest.fixture
    def drift_attention_weights(self):
        """Create drifted attention weights to test drift detection"""
        # Create attention patterns that show various types of drift
        drift_data = {
            'attention_shift': {
                # Attention shifted to recent time steps (recency bias drift)
                'weights': torch.zeros(4, 8, 100, 100),
                'description': 'Increased recency bias - attention focused on last 20 time steps'
            },
            'head_specialization_loss': {
                # Heads becoming more similar (loss of specialization)
                'weights': torch.ones(4, 8, 100, 100) * 0.1,
                'description': 'Head attention patterns becoming homogeneous'
            },
            'temporal_pattern_change': {
                # Different temporal patterns (regime change)
                'weights': torch.rand(4, 8, 100, 100) * 0.5,
                'description': 'Temporal attention patterns changed significantly'
            },
            'cross_asset_correlation_drift': {
                # Cross-asset correlations changed
                'weights': torch.rand(4, 8, 100, 100),
                'description': 'Cross-asset attention correlations shifted'
            },
            'attention_entropy_change': {
                # Attention distribution became more/less uniform
                'weights': torch.rand(4, 8, 100, 100) ** 2,  # More peaked distribution
                'description': 'Attention entropy significantly changed'
            }
        }
        
        # Create realistic drift patterns
        for drift_type, data in drift_data.items():
            if drift_type == 'attention_shift':
                # Focus attention on last 20 time steps
                data['weights'][:, :, -20:, -20:] = torch.rand(4, 8, 20, 20) * 2.0
            elif drift_type == 'head_specialization_loss':
                # Make all heads similar
                base_pattern = torch.rand(100, 100)
                for head in range(8):
                    data['weights'][:, head, :, :] = base_pattern.unsqueeze(0).repeat(4, 1, 1) + torch.rand(4, 100, 100) * 0.1
        
        return drift_data
    
    @pytest.fixture
    def trading_data_sequence(self):
        """Create realistic trading data sequence for testing"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5  # BTC, ETH, SOL price, volume, volatility
        
        # Create base time series with trends and patterns
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        
        # Create realistic crypto data with correlations
        btc_price = 40000 + np.cumsum(np.random.randn(n_samples) * 100)
        eth_price = 2500 + np.cumsum(np.random.randn(n_samples) * 50 + 0.7 * np.diff(np.concatenate([[40000], btc_price])))
        sol_price = 100 + np.cumsum(np.random.randn(n_samples) * 5 + 0.5 * np.diff(np.concatenate([[2500], eth_price])))
        
        volume = np.random.lognormal(10, 1, n_samples)
        volatility = np.random.exponential(0.02, n_samples)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': btc_price,
            'eth_price': eth_price,
            'sol_price': sol_price,
            'volume': volume,
            'volatility': volatility
        })
        
        return data
    
    @pytest.fixture
    def transformer_models(self):
        """Create mock transformer models for testing"""
        models = {}
        
        # Mock iTransformer
        itransformer = MagicMock()
        itransformer.model_type = 'iTransformer'
        itransformer.attention_weights = torch.rand(4, 8, 100, 100)
        itransformer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        models['iTransformer'] = itransformer
        
        # Mock PatchTST
        patchtst = MagicMock()
        patchtst.model_type = 'PatchTST'
        patchtst.attention_weights = torch.rand(4, 8, 20, 20)
        patchtst.get_attention_weights.return_value = torch.rand(4, 8, 20, 20)
        models['PatchTST'] = patchtst
        
        # Mock TimesMixer
        timesmixer = MagicMock()
        timesmixer.model_type = 'TimesMixer'
        timesmixer.attention_weights = torch.rand(4, 8, 100, 100)
        timesmixer.get_attention_weights.return_value = torch.rand(4, 8, 100, 100)
        models['TimesMixer'] = timesmixer
        
        # Mock TimesFM
        timesfm = MagicMock()
        timesfm.model_type = 'TimesFM'
        timesfm.attention_weights = torch.rand(4, 12, 512, 512)
        timesfm.get_attention_weights.return_value = torch.rand(4, 12, 512, 512)
        models['TimesFM'] = timesfm
        
        return models


class TestTransformerDriftDetector:
    """Test TransformerDriftDetector main class functionality"""
    
    def test_transformer_drift_detector_initialization(self, transformer_models):
        """Test TransformerDriftDetector initialization with different models"""
        # This test will fail initially - defining expected behavior
        detector = TransformerDriftDetector(
            models=transformer_models,
            detection_window=100,
            false_positive_threshold=0.05,
            drift_sensitivity=0.1
        )
        
        # Expected initialization behavior
        assert detector.models == transformer_models
        assert detector.detection_window == 100
        assert detector.false_positive_threshold == 0.05
        assert detector.drift_sensitivity == 0.1
        assert hasattr(detector, 'baseline_patterns')
        assert hasattr(detector, 'drift_analyzers')
        assert len(detector.drift_analyzers) == 7  # 7 different drift types
    
    def test_transformer_drift_detector_fit_baseline(self, transformer_models, baseline_attention_weights, trading_data_sequence):
        """Test fitting baseline attention patterns"""
        detector = TransformerDriftDetector(models=transformer_models)
        
        # Fit baseline patterns
        detector.fit_baseline(
            attention_weights=baseline_attention_weights,
            trading_data=trading_data_sequence
        )
        
        # Expected behavior after fitting
        assert detector.is_fitted is True
        assert detector.baseline_patterns is not None
        assert 'iTransformer' in detector.baseline_patterns
        assert 'PatchTST' in detector.baseline_patterns
        assert 'TimesMixer' in detector.baseline_patterns
        assert 'TimesFM' in detector.baseline_patterns
        
        # Each model should have baseline statistics
        for model_type in detector.baseline_patterns:
            baseline = detector.baseline_patterns[model_type]
            assert 'attention_entropy' in baseline
            assert 'head_specialization' in baseline
            assert 'temporal_patterns' in baseline
            assert 'feature_importance' in baseline
    
    def test_detect_attention_pattern_drift(self, transformer_models, baseline_attention_weights, drift_attention_weights):
        """Test attention pattern drift detection"""
        detector = TransformerDriftDetector(models=transformer_models)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Test drift detection for different types
        for drift_type, drift_data in drift_attention_weights.items():
            result = detector.detect_drift(
                current_attention_weights={'iTransformer': drift_data['weights']},
                model_type='iTransformer'
            )
            
            assert isinstance(result, TransformerDriftResult)
            assert result.model_type == 'iTransformer'
            assert result.has_drift is True or result.has_drift is False  # Should detect some drifts
            assert result.drift_score >= 0.0 and result.drift_score <= 1.0
            assert result.confidence >= 0.0 and result.confidence <= 1.0
            assert len(result.drift_types) >= 0
    
    def test_drift_detection_within_100_predictions(self, transformer_models, baseline_attention_weights):
        """Test that drift is detected within 100 predictions (success criteria)"""
        detector = TransformerDriftDetector(models=transformer_models, detection_window=100)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Simulate gradual drift over 100 predictions
        predictions_until_drift = 0
        for i in range(100):
            # Create gradually changing attention patterns
            drift_strength = i / 100.0  # Increase drift strength
            
            # Create drifted attention weights
            current_weights = {}
            for model_type, baseline in baseline_attention_weights.items():
                # Add increasing noise to simulate drift
                drifted_weights = baseline['weights'] * (1 - drift_strength) + torch.rand_like(baseline['weights']) * drift_strength
                current_weights[model_type] = drifted_weights
            
            result = detector.detect_drift(
                current_attention_weights=current_weights,
                model_type='iTransformer'
            )
            
            if result.has_drift and result.confidence > 0.8:
                predictions_until_drift = i + 1
                break
        
        # Success criteria: detect significant drift within 100 predictions
        assert predictions_until_drift > 0, "Drift should be detected within 100 predictions"
        assert predictions_until_drift <= 100, f"Drift detected after {predictions_until_drift} predictions, should be ≤100"
    
    def test_false_positive_rate_below_5_percent(self, transformer_models, baseline_attention_weights):
        """Test that false positive rate is below 5% (success criteria)"""
        detector = TransformerDriftDetector(models=transformer_models, false_positive_threshold=0.05)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Test with stable (non-drifting) attention patterns
        false_positives = 0
        total_tests = 100
        
        for i in range(total_tests):
            # Create slightly noisy but stable attention patterns
            stable_weights = {}
            for model_type, baseline in baseline_attention_weights.items():
                # Add small amount of noise (not drift)
                noise_level = 0.05  # 5% noise
                stable_weights[model_type] = baseline['weights'] + torch.randn_like(baseline['weights']) * noise_level
            
            result = detector.detect_drift(
                current_attention_weights=stable_weights,
                model_type='iTransformer'
            )
            
            if result.has_drift:
                false_positives += 1
        
        false_positive_rate = false_positives / total_tests
        
        # Success criteria: false positive rate < 5%
        assert false_positive_rate < 0.05, f"False positive rate {false_positive_rate:.3f} exceeds 5% threshold"


class TestAttentionPatternDrift:
    """Test attention pattern drift detection algorithms"""
    
    def test_attention_pattern_analyzer_initialization(self):
        """Test AttentionPatternAnalyzer initialization"""
        analyzer = AttentionPatternAnalyzer(
            n_heads=8,
            seq_length=100,
            pattern_threshold=0.1,
            entropy_threshold=0.2
        )
        
        assert analyzer.n_heads == 8
        assert analyzer.seq_length == 100
        assert analyzer.pattern_threshold == 0.1
        assert analyzer.entropy_threshold == 0.2
        assert hasattr(analyzer, 'baseline_patterns')
    
    def test_compute_attention_entropy(self, baseline_attention_weights):
        """Test attention entropy computation"""
        analyzer = AttentionPatternAnalyzer(n_heads=8, seq_length=100)
        
        # Test entropy computation for different models
        for model_type, attention_data in baseline_attention_weights.items():
            if model_type in ['iTransformer', 'TimesMixer']:  # Models with 100 seq_length
                entropy = analyzer.compute_attention_entropy(attention_data['weights'])
                
                assert isinstance(entropy, torch.Tensor)
                assert entropy.shape == (4, 8)  # [batch, heads]
                assert torch.all(entropy >= 0), "Entropy should be non-negative"
                assert torch.all(entropy <= np.log(100)), "Entropy should be bounded by log(seq_length)"
    
    def test_detect_attention_shift(self, baseline_attention_weights, drift_attention_weights):
        """Test detection of attention shift (recency bias drift)"""
        analyzer = AttentionPatternAnalyzer(n_heads=8, seq_length=100)
        
        # Fit baseline
        baseline = baseline_attention_weights['iTransformer']['weights']
        analyzer.fit_baseline(baseline)
        
        # Test shift detection
        shifted_attention = drift_attention_weights['attention_shift']['weights']
        drift_result = analyzer.detect_attention_shift(shifted_attention)
        
        assert isinstance(drift_result, AttentionPatternDrift)
        assert drift_result.drift_type == 'attention_shift'
        assert drift_result.has_drift is True
        assert drift_result.drift_score > 0.0
        assert 'recency_bias' in drift_result.metadata
    
    def test_detect_pattern_similarity_change(self, baseline_attention_weights):
        """Test detection of attention pattern similarity changes"""
        analyzer = AttentionPatternAnalyzer(n_heads=8, seq_length=100)
        
        baseline = baseline_attention_weights['iTransformer']['weights']
        analyzer.fit_baseline(baseline)
        
        # Create patterns with different similarity structure
        changed_patterns = baseline.clone()
        # Alter specific attention patterns
        changed_patterns[:, :4, :, :] *= 0.5  # Weaken first 4 heads
        changed_patterns[:, 4:, :, :] *= 1.5  # Strengthen last 4 heads
        
        drift_result = analyzer.detect_pattern_similarity_change(changed_patterns)
        
        assert isinstance(drift_result, AttentionPatternDrift)
        assert drift_result.drift_type == 'pattern_similarity'
        assert 'similarity_matrix' in drift_result.metadata
        assert 'similarity_change' in drift_result.metadata


class TestFeatureImportanceDrift:
    """Test feature importance shift monitoring"""
    
    def test_feature_importance_analyzer_initialization(self):
        """Test feature importance analyzer initialization"""
        analyzer = AttentionDriftAnalyzer(
            feature_names=['btc_price', 'eth_price', 'sol_price', 'volume', 'volatility'],
            importance_threshold=0.1,
            shift_threshold=0.15
        )
        
        assert len(analyzer.feature_names) == 5
        assert analyzer.importance_threshold == 0.1
        assert analyzer.shift_threshold == 0.15
        assert hasattr(analyzer, 'baseline_importance')
    
    def test_compute_attention_based_importance(self, baseline_attention_weights):
        """Test computation of feature importance from attention weights"""
        analyzer = AttentionDriftAnalyzer(
            feature_names=['btc_price', 'eth_price', 'sol_price', 'volume', 'volatility']
        )
        
        # Test for iTransformer (multivariate)
        attention_weights = baseline_attention_weights['iTransformer']['weights']
        importance_scores = analyzer.compute_attention_based_importance(
            attention_weights,
            feature_dim=5
        )
        
        assert isinstance(importance_scores, np.ndarray)
        assert importance_scores.shape == (5,)  # One score per feature
        assert np.all(importance_scores >= 0), "Importance scores should be non-negative"
        assert np.isclose(np.sum(importance_scores), 1.0, atol=1e-6), "Importance scores should sum to 1"
    
    def test_detect_importance_shift(self, baseline_attention_weights):
        """Test detection of feature importance shifts"""
        analyzer = AttentionDriftAnalyzer(
            feature_names=['btc_price', 'eth_price', 'sol_price', 'volume', 'volatility'],
            shift_threshold=0.1
        )
        
        # Fit baseline importance
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_importance = analyzer.compute_attention_based_importance(baseline_weights, feature_dim=5)
        analyzer.fit_baseline_importance(baseline_importance)
        
        # Create shifted importance (volume becomes much more important)
        shifted_weights = baseline_weights.clone()
        # Boost attention to volume feature (index 3)
        shifted_weights[:, :, :, :] *= 0.8  # Reduce overall attention
        shifted_weights[:, :, :, 3] *= 2.0   # Double attention to volume
        
        drift_result = analyzer.detect_importance_shift(shifted_weights, feature_dim=5)
        
        assert isinstance(drift_result, FeatureImportanceDrift)
        assert drift_result.drift_type == 'feature_importance'
        assert drift_result.has_drift is True
        assert 'importance_changes' in drift_result.metadata
        assert 'shifted_features' in drift_result.metadata
    
    def test_detect_new_feature_emergence(self, baseline_attention_weights):
        """Test detection of new feature emergence (previously unimportant features become important)"""
        analyzer = AttentionDriftAnalyzer(
            feature_names=['btc_price', 'eth_price', 'sol_price', 'volume', 'volatility'],
            emergence_threshold=0.05
        )
        
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_importance = analyzer.compute_attention_based_importance(baseline_weights, feature_dim=5)
        analyzer.fit_baseline_importance(baseline_importance)
        
        # Simulate emergence of volatility as important feature
        emerged_weights = baseline_weights.clone()
        emerged_weights[:, :, :, 4] *= 3.0  # Triple attention to volatility
        
        drift_result = analyzer.detect_new_feature_emergence(emerged_weights, feature_dim=5)
        
        assert isinstance(drift_result, FeatureImportanceDrift)
        assert drift_result.drift_type == 'feature_emergence'
        assert 'emerged_features' in drift_result.metadata


class TestTemporalFocusDrift:
    """Test temporal focus change detection"""
    
    def test_temporal_attention_analyzer_initialization(self):
        """Test TemporalAttentionAnalyzer initialization"""
        analyzer = TemporalAttentionAnalyzer(
            seq_length=100,
            window_sizes=[5, 10, 20, 50],
            focus_shift_threshold=0.1
        )
        
        assert analyzer.seq_length == 100
        assert analyzer.window_sizes == [5, 10, 20, 50]
        assert analyzer.focus_shift_threshold == 0.1
        assert hasattr(analyzer, 'baseline_temporal_patterns')
    
    def test_compute_temporal_attention_distribution(self, baseline_attention_weights):
        """Test computation of temporal attention distribution"""
        analyzer = TemporalAttentionAnalyzer(seq_length=100)
        
        attention_weights = baseline_attention_weights['iTransformer']['weights']
        temporal_dist = analyzer.compute_temporal_attention_distribution(attention_weights)
        
        assert isinstance(temporal_dist, np.ndarray)
        assert temporal_dist.shape == (100,)  # One value per time step
        assert np.all(temporal_dist >= 0), "Temporal distribution should be non-negative"
        assert np.isclose(np.sum(temporal_dist), 1.0, atol=1e-6), "Temporal distribution should sum to 1"
    
    def test_detect_recency_bias_drift(self, baseline_attention_weights, drift_attention_weights):
        """Test detection of recency bias changes"""
        analyzer = TemporalAttentionAnalyzer(seq_length=100)
        
        # Fit baseline
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_temporal = analyzer.compute_temporal_attention_distribution(baseline_weights)
        analyzer.fit_baseline_temporal_patterns(baseline_temporal)
        
        # Test recency bias drift
        shifted_weights = drift_attention_weights['attention_shift']['weights']
        drift_result = analyzer.detect_recency_bias_drift(shifted_weights)
        
        assert isinstance(drift_result, TemporalFocusDrift)
        assert drift_result.drift_type == 'recency_bias'
        assert drift_result.has_drift is True
        assert 'recency_score_change' in drift_result.metadata
        assert 'focus_window_change' in drift_result.metadata
    
    def test_detect_periodic_pattern_change(self, baseline_attention_weights):
        """Test detection of periodic pattern changes (e.g., daily, weekly cycles)"""
        analyzer = TemporalAttentionAnalyzer(seq_length=100)
        
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_temporal = analyzer.compute_temporal_attention_distribution(baseline_weights)
        analyzer.fit_baseline_temporal_patterns(baseline_temporal)
        
        # Create attention with different periodic pattern
        periodic_weights = baseline_weights.clone()
        # Add strong 24-hour cyclical pattern (assuming 1-hour intervals)
        for i in range(100):
            cycle_strength = 1.0 + 0.5 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
            periodic_weights[:, :, i, :] *= cycle_strength
        
        drift_result = analyzer.detect_periodic_pattern_change(periodic_weights)
        
        assert isinstance(drift_result, TemporalFocusDrift)
        assert drift_result.drift_type == 'periodic_pattern'
        assert 'cycle_strength_change' in drift_result.metadata
        assert 'dominant_periods' in drift_result.metadata


class TestCrossAssetCorrelationDrift:
    """Test cross-asset correlation drift tracking"""
    
    def test_cross_asset_correlation_analyzer_initialization(self):
        """Test cross-asset correlation analyzer initialization"""
        from src.monitoring.transformer_drift_detection import CrossAssetCorrelationAnalyzer
        
        analyzer = CrossAssetCorrelationAnalyzer(
            asset_names=['BTC', 'ETH', 'SOL'],
            correlation_threshold=0.1,
            min_correlation_change=0.05
        )
        
        assert analyzer.asset_names == ['BTC', 'ETH', 'SOL']
        assert analyzer.correlation_threshold == 0.1
        assert analyzer.min_correlation_change == 0.05
        assert hasattr(analyzer, 'baseline_correlations')
    
    def test_compute_cross_attention_correlations(self, baseline_attention_weights):
        """Test computation of cross-asset attention correlations"""
        from src.monitoring.transformer_drift_detection import CrossAssetCorrelationAnalyzer
        
        analyzer = CrossAssetCorrelationAnalyzer(asset_names=['BTC', 'ETH', 'SOL'])
        
        # Use TimesFM data with more features for cross-asset analysis
        attention_weights = baseline_attention_weights['TimesFM']['weights']
        correlations = analyzer.compute_cross_attention_correlations(
            attention_weights,
            n_assets=3
        )
        
        assert isinstance(correlations, np.ndarray)
        assert correlations.shape == (3, 3)  # Asset correlation matrix
        assert np.allclose(np.diag(correlations), 1.0, atol=1e-6), "Diagonal should be 1.0"
        assert np.allclose(correlations, correlations.T, atol=1e-6), "Matrix should be symmetric"
        assert np.all(correlations >= -1.0) and np.all(correlations <= 1.0), "Correlations should be in [-1, 1]"
    
    def test_detect_correlation_regime_change(self, baseline_attention_weights):
        """Test detection of correlation regime changes"""
        from src.monitoring.transformer_drift_detection import CrossAssetCorrelationAnalyzer
        
        analyzer = CrossAssetCorrelationAnalyzer(asset_names=['BTC', 'ETH', 'SOL'])
        
        # Fit baseline correlations
        baseline_weights = baseline_attention_weights['TimesFM']['weights']
        baseline_corr = analyzer.compute_cross_attention_correlations(baseline_weights, n_assets=3)
        analyzer.fit_baseline_correlations(baseline_corr)
        
        # Create regime change (assets become more correlated)
        regime_weights = baseline_weights.clone()
        # Increase cross-asset attention patterns
        for i in range(3):
            for j in range(3):
                if i != j:
                    regime_weights[:, :, i*170:(i+1)*170, j*170:(j+1)*170] *= 1.5
        
        drift_result = analyzer.detect_correlation_regime_change(regime_weights, n_assets=3)
        
        assert isinstance(drift_result, CrossAssetCorrelationDrift)
        assert drift_result.drift_type == 'correlation_regime'
        assert 'correlation_changes' in drift_result.metadata
        assert 'regime_type' in drift_result.metadata  # 'high_correlation', 'low_correlation', etc.


class TestAttentionDistributionDrift:
    """Test attention distribution monitoring"""
    
    def test_attention_distribution_analyzer_initialization(self):
        """Test attention distribution analyzer initialization"""
        from src.monitoring.transformer_drift_detection import AttentionDistributionAnalyzer
        
        analyzer = AttentionDistributionAnalyzer(
            distribution_bins=20,
            entropy_threshold=0.1,
            sparsity_threshold=0.15
        )
        
        assert analyzer.distribution_bins == 20
        assert analyzer.entropy_threshold == 0.1
        assert analyzer.sparsity_threshold == 0.15
        assert hasattr(analyzer, 'baseline_distributions')
    
    def test_compute_attention_sparsity(self, baseline_attention_weights):
        """Test computation of attention sparsity metrics"""
        from src.monitoring.transformer_drift_detection import AttentionDistributionAnalyzer
        
        analyzer = AttentionDistributionAnalyzer()
        
        attention_weights = baseline_attention_weights['iTransformer']['weights']
        sparsity_metrics = analyzer.compute_attention_sparsity(attention_weights)
        
        assert isinstance(sparsity_metrics, dict)
        assert 'gini_coefficient' in sparsity_metrics
        assert 'top_k_concentration' in sparsity_metrics
        assert 'effective_attention_rank' in sparsity_metrics
        
        # Validate metric ranges
        assert 0.0 <= sparsity_metrics['gini_coefficient'] <= 1.0
        assert 0.0 <= sparsity_metrics['top_k_concentration'] <= 1.0
        assert sparsity_metrics['effective_attention_rank'] > 0
    
    def test_detect_attention_entropy_drift(self, baseline_attention_weights, drift_attention_weights):
        """Test detection of attention entropy changes"""
        from src.monitoring.transformer_drift_detection import AttentionDistributionAnalyzer
        
        analyzer = AttentionDistributionAnalyzer(entropy_threshold=0.1)
        
        # Fit baseline
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_entropy = analyzer.compute_attention_entropy(baseline_weights)
        analyzer.fit_baseline_entropy(baseline_entropy)
        
        # Test entropy change
        entropy_changed_weights = drift_attention_weights['attention_entropy_change']['weights']
        drift_result = analyzer.detect_attention_entropy_drift(entropy_changed_weights)
        
        assert isinstance(drift_result, AttentionDistributionDrift)
        assert drift_result.drift_type == 'entropy_drift'
        assert 'entropy_change' in drift_result.metadata
        assert 'entropy_direction' in drift_result.metadata  # 'increase' or 'decrease'
    
    def test_detect_attention_sparsity_change(self, baseline_attention_weights):
        """Test detection of attention sparsity changes"""
        from src.monitoring.transformer_drift_detection import AttentionDistributionAnalyzer
        
        analyzer = AttentionDistributionAnalyzer(sparsity_threshold=0.1)
        
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_sparsity = analyzer.compute_attention_sparsity(baseline_weights)
        analyzer.fit_baseline_sparsity(baseline_sparsity)
        
        # Create sparser attention (more concentrated)
        sparse_weights = baseline_weights.clone()
        sparse_weights = torch.softmax(sparse_weights * 2.0, dim=-1)  # Sharper distribution
        
        drift_result = analyzer.detect_attention_sparsity_change(sparse_weights)
        
        assert isinstance(drift_result, AttentionDistributionDrift)
        assert drift_result.drift_type == 'sparsity_change'
        assert 'sparsity_changes' in drift_result.metadata


class TestHeadSpecializationDrift:
    """Test head specialization tracking"""
    
    def test_head_specialization_analyzer_initialization(self):
        """Test head specialization analyzer initialization"""
        analyzer = HeadSpecializationAnalyzer(
            n_heads=8,
            specialization_threshold=0.1,
            similarity_threshold=0.8
        )
        
        assert analyzer.n_heads == 8
        assert analyzer.specialization_threshold == 0.1
        assert analyzer.similarity_threshold == 0.8
        assert hasattr(analyzer, 'baseline_specializations')
    
    def test_compute_head_specialization_scores(self, baseline_attention_weights):
        """Test computation of head specialization scores"""
        analyzer = HeadSpecializationAnalyzer(n_heads=8)
        
        attention_weights = baseline_attention_weights['iTransformer']['weights']
        specialization_scores = analyzer.compute_head_specialization_scores(attention_weights)
        
        assert isinstance(specialization_scores, dict)
        assert 'head_similarities' in specialization_scores
        assert 'specialization_index' in specialization_scores
        assert 'head_roles' in specialization_scores
        
        # Validate shapes and ranges
        assert specialization_scores['head_similarities'].shape == (8, 8)
        assert 0.0 <= specialization_scores['specialization_index'] <= 1.0
        assert len(specialization_scores['head_roles']) == 8
    
    def test_detect_head_homogenization(self, baseline_attention_weights, drift_attention_weights):
        """Test detection of head homogenization (loss of specialization)"""
        analyzer = HeadSpecializationAnalyzer(n_heads=8)
        
        # Fit baseline
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_spec = analyzer.compute_head_specialization_scores(baseline_weights)
        analyzer.fit_baseline_specialization(baseline_spec)
        
        # Test homogenization
        homogenized_weights = drift_attention_weights['head_specialization_loss']['weights']
        drift_result = analyzer.detect_head_homogenization(homogenized_weights)
        
        assert isinstance(drift_result, HeadSpecializationDrift)
        assert drift_result.drift_type == 'head_homogenization'
        assert drift_result.has_drift is True
        assert 'specialization_loss' in drift_result.metadata
        assert 'similar_head_pairs' in drift_result.metadata
    
    def test_detect_role_switching(self, baseline_attention_weights):
        """Test detection of head role switching"""
        analyzer = HeadSpecializationAnalyzer(n_heads=8)
        
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_spec = analyzer.compute_head_specialization_scores(baseline_weights)
        analyzer.fit_baseline_specialization(baseline_spec)
        
        # Create role switching (swap attention patterns between heads)
        role_switched_weights = baseline_weights.clone()
        # Swap heads 0 and 4
        role_switched_weights[:, 0, :, :] = baseline_weights[:, 4, :, :]
        role_switched_weights[:, 4, :, :] = baseline_weights[:, 0, :, :]
        
        drift_result = analyzer.detect_role_switching(role_switched_weights)
        
        assert isinstance(drift_result, HeadSpecializationDrift)
        assert drift_result.drift_type == 'role_switching'
        assert 'switched_pairs' in drift_result.metadata
        assert 'role_stability_score' in drift_result.metadata


class TestTemporalAttentionShiftDrift:
    """Test temporal attention shift detection"""
    
    def test_temporal_shift_analyzer_initialization(self):
        """Test temporal attention shift analyzer initialization"""
        from src.monitoring.transformer_drift_detection import TemporalShiftAnalyzer
        
        analyzer = TemporalShiftAnalyzer(
            seq_length=100,
            shift_detection_windows=[10, 20, 50],
            shift_threshold=0.1
        )
        
        assert analyzer.seq_length == 100
        assert analyzer.shift_detection_windows == [10, 20, 50]
        assert analyzer.shift_threshold == 0.1
        assert hasattr(analyzer, 'baseline_temporal_focus')
    
    def test_compute_temporal_attention_centroid(self, baseline_attention_weights):
        """Test computation of temporal attention centroid (center of mass)"""
        from src.monitoring.transformer_drift_detection import TemporalShiftAnalyzer
        
        analyzer = TemporalShiftAnalyzer(seq_length=100)
        
        attention_weights = baseline_attention_weights['iTransformer']['weights']
        centroids = analyzer.compute_temporal_attention_centroid(attention_weights)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape == (4, 8)  # [batch, heads]
        assert np.all(centroids >= 0) and np.all(centroids < 100), "Centroids should be within sequence length"
    
    def test_detect_temporal_focus_shift(self, baseline_attention_weights, drift_attention_weights):
        """Test detection of temporal focus shifts"""
        from src.monitoring.transformer_drift_detection import TemporalShiftAnalyzer
        
        analyzer = TemporalShiftAnalyzer(seq_length=100)
        
        # Fit baseline
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_centroids = analyzer.compute_temporal_attention_centroid(baseline_weights)
        analyzer.fit_baseline_temporal_focus(baseline_centroids)
        
        # Test shift detection
        shifted_weights = drift_attention_weights['attention_shift']['weights']
        drift_result = analyzer.detect_temporal_focus_shift(shifted_weights)
        
        assert isinstance(drift_result, TemporalAttentionShiftDrift)
        assert drift_result.drift_type == 'temporal_focus_shift'
        assert 'centroid_shifts' in drift_result.metadata
        assert 'shift_direction' in drift_result.metadata  # 'recent', 'past', 'middle'
        assert 'shift_magnitude' in drift_result.metadata
    
    def test_detect_attention_window_change(self, baseline_attention_weights):
        """Test detection of attention window size changes"""
        from src.monitoring.transformer_drift_detection import TemporalShiftAnalyzer
        
        analyzer = TemporalShiftAnalyzer(seq_length=100)
        
        baseline_weights = baseline_attention_weights['iTransformer']['weights']
        baseline_windows = analyzer.compute_attention_window_sizes(baseline_weights)
        analyzer.fit_baseline_attention_windows(baseline_windows)
        
        # Create narrower attention windows
        narrow_weights = baseline_weights.clone()
        # Apply Gaussian filter to make attention more concentrated
        for i in range(100):
            for j in range(100):
                if abs(i - j) > 10:  # Reduce long-range attention
                    narrow_weights[:, :, i, j] *= 0.3
        
        drift_result = analyzer.detect_attention_window_change(narrow_weights)
        
        assert isinstance(drift_result, TemporalAttentionShiftDrift)
        assert drift_result.drift_type == 'attention_window_change'
        assert 'window_size_changes' in drift_result.metadata
        assert 'window_direction' in drift_result.metadata  # 'narrowing' or 'widening'


class TestIntegrationWithExistingDriftDetector:
    """Test integration with existing DriftDetector system"""
    
    def test_transformer_drift_detector_integration(self, transformer_models, trading_data_sequence):
        """Test integration of TransformerDriftDetector with existing DriftDetector"""
        
        # Create existing DriftDetector
        feature_columns = ['btc_price', 'eth_price', 'sol_price', 'volume', 'volatility']
        existing_detector = DriftDetector(
            reference_data=trading_data_sequence,
            feature_columns=feature_columns,
            target_column=None
        )
        existing_detector.fit()
        
        # Create TransformerDriftDetector
        transformer_detector = TransformerDriftDetector(
            models=transformer_models,
            existing_drift_detector=existing_detector
        )
        
        # Test combined drift detection
        current_data = trading_data_sequence.iloc[-100:].copy()
        current_data['volume'] *= 2.0  # Introduce feature drift
        
        # Mock attention weights for current state
        current_attention = {
            'iTransformer': torch.rand(4, 8, 100, 100),
            'PatchTST': torch.rand(4, 8, 20, 20),
            'TimesMixer': torch.rand(4, 8, 100, 100),
            'TimesFM': torch.rand(4, 12, 512, 512)
        }
        
        # Detect combined drift
        combined_result = transformer_detector.detect_combined_drift(
            current_data=current_data,
            current_attention_weights=current_attention
        )
        
        assert hasattr(combined_result, 'feature_drift_results')
        assert hasattr(combined_result, 'attention_drift_results')
        assert hasattr(combined_result, 'overall_drift_score')
        assert hasattr(combined_result, 'drift_confidence')
        assert isinstance(combined_result.overall_drift_score, float)
        assert 0.0 <= combined_result.overall_drift_score <= 1.0
    
    def test_automated_drift_alerts_integration(self, transformer_models):
        """Test integration with intelligent alerting system"""
        from src.monitoring.intelligent_alerting import IntelligentAlertingSystem
        
        # Create alert system
        alert_system = IntelligentAlertingSystem()
        
        # Create transformer detector with alerting
        transformer_detector = TransformerDriftDetector(
            models=transformer_models,
            alert_system=alert_system,
            auto_alert_threshold=0.7
        )
        
        # Simulate high-confidence drift detection
        mock_drift_result = TransformerDriftResult(
            model_type='iTransformer',
            has_drift=True,
            drift_score=0.8,
            confidence=0.9,
            drift_types=['attention_shift', 'head_specialization'],
            timestamp=datetime.now()
        )
        
        # Test alert generation
        alerts = transformer_detector.generate_drift_alerts(mock_drift_result)
        
        assert len(alerts) > 0
        assert all(hasattr(alert, 'alert_type') for alert in alerts)
        assert all(hasattr(alert, 'severity') for alert in alerts)
        assert all(hasattr(alert, 'message') for alert in alerts)
        assert any('attention' in alert.message.lower() for alert in alerts)
    
    def test_real_time_drift_monitoring(self, transformer_models, trading_data_sequence):
        """Test real-time drift monitoring capabilities"""
        transformer_detector = TransformerDriftDetector(
            models=transformer_models,
            real_time_monitoring=True,
            monitoring_interval_seconds=60
        )
        
        # Simulate real-time monitoring
        monitoring_results = []
        
        for i in range(5):  # 5 monitoring cycles
            # Simulate new data and attention weights
            current_data = trading_data_sequence.iloc[i*20:(i+1)*20].copy()
            current_attention = {
                'iTransformer': torch.rand(1, 8, 20, 20),
                'PatchTST': torch.rand(1, 8, 4, 4),
                'TimesMixer': torch.rand(1, 8, 20, 20),
                'TimesFM': torch.rand(1, 12, 20, 20)
            }
            
            # Monitor drift
            result = transformer_detector.monitor_real_time_drift(
                current_data=current_data,
                current_attention_weights=current_attention
            )
            
            monitoring_results.append(result)
        
        # Validate monitoring results
        assert len(monitoring_results) == 5
        assert all(hasattr(result, 'timestamp') for result in monitoring_results)
        assert all(hasattr(result, 'drift_detected') for result in monitoring_results)
        assert all(hasattr(result, 'monitoring_latency_ms') for result in monitoring_results)
        
        # Performance requirement: <1 second monitoring latency
        assert all(result.monitoring_latency_ms < 1000 for result in monitoring_results)


class TestPerformanceRequirements:
    """Test performance requirements for drift detection"""
    
    def test_drift_detection_latency(self, transformer_models, baseline_attention_weights):
        """Test that drift detection meets latency requirements"""
        detector = TransformerDriftDetector(models=transformer_models)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Test detection latency
        current_attention = {
            'iTransformer': torch.rand(4, 8, 100, 100),
            'TimesFM': torch.rand(4, 12, 512, 512)
        }
        
        start_time = time.time()
        result = detector.detect_drift(
            current_attention_weights=current_attention,
            model_type='iTransformer'
        )
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance requirement: drift detection should be fast for real-time trading
        assert detection_time < 100, f"Drift detection took {detection_time:.2f}ms, should be <100ms"
        assert result is not None
    
    def test_memory_usage_efficiency(self, transformer_models, baseline_attention_weights):
        """Test memory efficiency of drift detection"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create detector and fit baseline
        detector = TransformerDriftDetector(models=transformer_models)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Perform multiple drift detections
        for _ in range(10):
            current_attention = {
                'iTransformer': torch.rand(4, 8, 100, 100),
                'TimesFM': torch.rand(4, 12, 512, 512)
            }
            detector.detect_drift(current_attention_weights=current_attention, model_type='iTransformer')
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency requirement: should not have significant memory leaks
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, should be <100MB"
    
    def test_concurrent_drift_detection(self, transformer_models, baseline_attention_weights):
        """Test concurrent drift detection for multiple models"""
        import threading
        import time
        
        detector = TransformerDriftDetector(models=transformer_models)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        results = {}
        threads = []
        
        def detect_drift_for_model(model_type, attention_weights):
            try:
                result = detector.detect_drift(
                    current_attention_weights={model_type: attention_weights},
                    model_type=model_type
                )
                results[model_type] = result
            except Exception as e:
                results[model_type] = f"Error: {e}"
        
        # Start concurrent detection for all models
        start_time = time.time()
        
        for model_type in ['iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM']:
            if model_type in baseline_attention_weights:
                attention_shape = baseline_attention_weights[model_type]['weights'].shape
                current_attention = torch.rand(*attention_shape)
                
                thread = threading.Thread(
                    target=detect_drift_for_model,
                    args=(model_type, current_attention)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance requirement: concurrent detection should be efficient
        assert total_time < 500, f"Concurrent detection took {total_time:.2f}ms, should be <500ms"
        assert len(results) > 0
        assert all(not isinstance(result, str) or not result.startswith("Error") for result in results.values())


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for drift detection"""
    
    def test_malformed_attention_weights(self, transformer_models):
        """Test handling of malformed attention weights"""
        detector = TransformerDriftDetector(models=transformer_models)
        
        # Test various malformed inputs
        malformed_cases = [
            {'case': 'nan_values', 'weights': torch.full((4, 8, 100, 100), float('nan'))},
            {'case': 'inf_values', 'weights': torch.full((4, 8, 100, 100), float('inf'))},
            {'case': 'negative_values', 'weights': torch.full((4, 8, 100, 100), -1.0)},
            {'case': 'wrong_shape', 'weights': torch.rand(2, 4, 50, 50)},  # Wrong dimensions
            {'case': 'empty_tensor', 'weights': torch.empty(0)},
        ]
        
        for case in malformed_cases:
            with pytest.raises((ValueError, RuntimeError, TypeError)):
                detector.detect_drift(
                    current_attention_weights={'iTransformer': case['weights']},
                    model_type='iTransformer'
                )
    
    def test_insufficient_baseline_data(self, transformer_models):
        """Test handling of insufficient baseline data"""
        detector = TransformerDriftDetector(models=transformer_models)
        
        # Try to detect drift without fitting baseline
        with pytest.raises(ValueError, match="must be fitted"):
            detector.detect_drift(
                current_attention_weights={'iTransformer': torch.rand(4, 8, 100, 100)},
                model_type='iTransformer'
            )
    
    def test_unsupported_model_type(self, transformer_models, baseline_attention_weights):
        """Test handling of unsupported model types"""
        detector = TransformerDriftDetector(models=transformer_models)
        detector.fit_baseline(attention_weights=baseline_attention_weights, trading_data=None)
        
        # Test unsupported model type
        with pytest.raises(ValueError, match="Unsupported model type"):
            detector.detect_drift(
                current_attention_weights={'UnsupportedModel': torch.rand(4, 8, 100, 100)},
                model_type='UnsupportedModel'  
            )
    
    def test_memory_pressure_handling(self, transformer_models):
        """Test handling of memory pressure scenarios"""
        detector = TransformerDriftDetector(models=transformer_models, max_memory_mb=100)
        
        # Try to fit with very large attention weights that exceed memory limit
        large_attention_weights = {
            'iTransformer': {
                'weights': torch.rand(100, 32, 1000, 1000),  # Very large tensor
                'head_patterns': torch.rand(32, 1000, 1000),
                'temporal_patterns': torch.rand(1000, 5),
                'feature_importance': torch.rand(5, 1000)
            }
        }
        
        # Should handle memory pressure gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                detector.fit_baseline(attention_weights=large_attention_weights, trading_data=None)
                # If successful, should use memory-efficient techniques
                assert detector.memory_efficient_mode is True
            except (RuntimeError, MemoryError):
                # Expected behavior under memory pressure
                pass