"""
Tests for TemporalAttentionAnalyzer - Time-Series Importance Pattern Analysis

Following TDD principles, these tests define the expected behavior
of temporal attention analysis for time-series forecasting BEFORE implementation.

These tests MUST FAIL initially as TemporalAttentionAnalyzer doesn't exist yet.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

# Import existing XAI framework components
from src.xai.base import BaseExplainer
from src.xai.data_models import ExplanationData


class TestTemporalAttentionAnalyzer:
    """
    Test suite for TemporalAttentionAnalyzer following TDD principles.
    
    TemporalAttentionAnalyzer should identify and analyze temporal patterns
    in attention weights to understand time-series importance for trading decisions.
    """
    
    def test_temporal_attention_analyzer_import_fails_initially(self):
        """Test that TemporalAttentionAnalyzer import fails initially (TDD)."""
        # This test should FAIL initially - TemporalAttentionAnalyzer doesn't exist yet
        with pytest.raises(ImportError):
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
    
    def test_temporal_attention_analyzer_initialization(self):
        """Test TemporalAttentionAnalyzer initialization."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        # Mock transformer model
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 8, 64, 64))
        
        # Time-series feature names with timestamps
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(64, 0, -1)]
        feature_names = [f"price_{ts.strftime('%H:%M')}" for ts in timestamps]
        
        config = {
            'analyze_recency_bias': True,
            'detect_periodic_patterns': True,
            'identify_regime_transitions': True,
            'time_window_analysis': True,
            'seasonal_attention_patterns': True
        }
        
        analyzer = TemporalAttentionAnalyzer(
            model=mock_model,
            feature_names=feature_names,
            config=config
        )
        
        assert analyzer.model == mock_model
        assert analyzer.feature_names == feature_names
        assert analyzer.config['analyze_recency_bias'] is True
        assert analyzer.config['detect_periodic_patterns'] is True
    
    def test_temporal_attention_analyzer_inherits_from_base_explainer(self):
        """Test that TemporalAttentionAnalyzer inherits from BaseExplainer."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        assert issubclass(TemporalAttentionAnalyzer, BaseExplainer)
    
    def test_analyze_recency_bias(self):
        """Test analysis of recency bias in attention patterns."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 64
        
        # Create attention with strong recency bias (recent timestamps get more attention)
        recency_attention = torch.zeros(1, 1, seq_len, seq_len)
        for i in range(seq_len):
            # More attention to recent positions (higher indices)
            recency_attention[0, 0, i, max(0, i-5):] = torch.softmax(
                torch.linspace(0, 2, min(6, seq_len - max(0, i-5))), dim=0
            )
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=recency_attention)
        
        # Create timestamps (oldest to newest)
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(seq_len, 0, -1)]
        feature_names = [f"price_{ts.strftime('%Y%m%d_%H%M')}" for ts in timestamps]
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should detect recency bias
        input_instance = np.random.randn(seq_len)
        recency_analysis = analyzer.analyze_recency_bias(input_instance)
        
        assert isinstance(recency_analysis, dict)
        assert "recency_score" in recency_analysis
        assert "temporal_weights" in recency_analysis
        assert "bias_strength" in recency_analysis
        
        # Should show strong recency bias (score close to 1.0)
        assert recency_analysis["recency_score"] > 0.7
        assert 0 <= recency_analysis["recency_score"] <= 1
    
    def test_detect_periodic_patterns(self):
        """Test detection of periodic patterns in attention."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 168  # 1 week of hourly data
        
        # Create attention with 24-hour periodic pattern
        periodic_attention = torch.zeros(1, 1, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                # Higher attention to same hour of day patterns
                if i % 24 == j % 24:
                    periodic_attention[0, 0, i, j] = 0.8
                else:
                    periodic_attention[0, 0, i, j] = 0.2 / (seq_len - 1)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=periodic_attention)
        
        # Create hourly timestamps
        base_time = datetime.now() - timedelta(hours=seq_len)
        timestamps = [base_time + timedelta(hours=i) for i in range(seq_len)]
        feature_names = [f"price_{ts.strftime('%d_%H')}" for ts in timestamps]
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should detect 24-hour periodicity
        input_instance = np.random.randn(seq_len)
        periodic_analysis = analyzer.detect_periodic_patterns(input_instance)
        
        assert isinstance(periodic_analysis, dict)
        assert "dominant_period" in periodic_analysis
        assert "periodicity_strength" in periodic_analysis
        assert "seasonal_patterns" in periodic_analysis
        
        # Should identify 24-hour cycle
        assert abs(periodic_analysis["dominant_period"] - 24) <= 2  # Allow some tolerance
        assert periodic_analysis["periodicity_strength"] > 0.5
    
    def test_identify_regime_transitions(self):
        """Test identification of market regime transitions in attention."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 100
        
        # Create attention that shows regime transition at position 50
        regime_attention = torch.zeros(1, 1, seq_len, seq_len)
        
        # Before regime change: attention focuses on positions 0-25
        for i in range(50):
            regime_attention[0, 0, i, :25] = torch.softmax(torch.randn(25), dim=0)
        
        # After regime change: attention focuses on positions 75-100
        for i in range(50, seq_len):
            regime_attention[0, 0, i, 75:] = torch.softmax(torch.randn(seq_len - 75), dim=0)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=regime_attention)
        
        feature_names = [f"market_state_{i}" for i in range(seq_len)]
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should identify regime transition
        input_instance = np.random.randn(seq_len)
        regime_analysis = analyzer.identify_regime_transitions(input_instance)
        
        assert isinstance(regime_analysis, dict)
        assert "transition_points" in regime_analysis
        assert "regime_stability" in regime_analysis
        assert "attention_shift_magnitude" in regime_analysis
        
        # Should detect transition around position 50
        transition_points = regime_analysis["transition_points"]
        assert len(transition_points) > 0
        assert any(abs(tp - 50) < 10 for tp in transition_points)
    
    def test_analyze_time_windows(self):
        """Test analysis of different time window importances."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 72  # 3 days of hourly data
        
        # Create attention with different window preferences
        window_attention = torch.zeros(1, 1, seq_len, seq_len)
        
        # Attention focused on last 24 hours (positions 48-72)
        for i in range(seq_len):
            window_attention[0, 0, i, 48:] = torch.softmax(torch.randn(24), dim=0)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=window_attention)
        
        # Create timestamps for 3 days
        base_time = datetime.now() - timedelta(hours=seq_len)
        timestamps = [base_time + timedelta(hours=i) for i in range(seq_len)]
        feature_names = [f"data_{ts.strftime('%d_%H')}" for ts in timestamps]
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should analyze different time windows
        input_instance = np.random.randn(seq_len)
        window_analysis = analyzer.analyze_time_windows(input_instance)
        
        assert isinstance(window_analysis, dict)
        assert "window_importances" in window_analysis
        assert "dominant_window" in window_analysis
        assert "temporal_focus_distribution" in window_analysis
        
        # Should identify last 24 hours as most important
        window_importances = window_analysis["window_importances"]
        assert "last_24h" in window_importances
        assert "last_48h" in window_importances
        assert "last_72h" in window_importances
        
        # Last 24h should have highest importance
        assert window_importances["last_24h"] > window_importances["last_48h"]
    
    def test_temporal_attention_analyzer_explain_instance(self):
        """Test explaining instance with temporal attention analysis."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 48
        mock_attention = torch.randn(1, 4, seq_len, seq_len)
        mock_prediction = np.array([0.75])  # Bullish prediction
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        mock_model.predict = Mock(return_value=mock_prediction)
        
        # Create timestamped feature names
        base_time = datetime.now() - timedelta(hours=seq_len)
        timestamps = [base_time + timedelta(hours=i) for i in range(seq_len)]
        feature_names = [f"BTC_price_{ts.strftime('%m%d_%H')}" for ts in timestamps]
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should generate temporal explanation
        input_instance = np.random.randn(seq_len)
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        assert explanation.explanation_type == "temporal_attention"
        assert len(explanation.feature_importance) == seq_len
        
        # Should include temporal analysis metadata
        metadata = explanation.explanation_metadata
        assert "recency_analysis" in metadata
        assert "periodic_patterns" in metadata
        assert "regime_transitions" in metadata
        assert "time_window_analysis" in metadata
    
    def test_get_temporal_feature_importance(self):
        """Test getting temporal-weighted feature importance."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 32
        
        # Create attention with clear temporal pattern
        temporal_attention = torch.zeros(1, 1, seq_len, seq_len)
        
        # Strong attention to recent 5 positions for all queries
        for i in range(seq_len):
            temporal_attention[0, 0, i, -5:] = torch.softmax(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float), dim=0)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=temporal_attention)
        
        feature_names = [f"price_t_{i}" for i in range(seq_len)]
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should compute temporal-weighted importance
        input_instance = np.random.randn(seq_len)
        importance = analyzer.get_feature_importance(input_instance)
        
        assert isinstance(importance, dict)
        assert len(importance) == seq_len
        
        # Recent features should have higher importance
        recent_importance = importance[f"price_t_{seq_len-1}"]  # Most recent
        older_importance = importance[f"price_t_0"]  # Oldest
        
        assert recent_importance > older_importance
    
    def test_analyze_seasonal_attention_patterns(self):
        """Test analysis of seasonal patterns in attention."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        # Use 7 days of hourly data (168 hours)
        seq_len = 168
        
        # Create attention with weekly seasonal pattern
        seasonal_attention = torch.zeros(1, 1, seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Higher attention to same day of week
                day_i = i // 24
                day_j = j // 24
                if day_i % 7 == day_j % 7:
                    seasonal_attention[0, 0, i, j] = 0.7
                else:
                    seasonal_attention[0, 0, i, j] = 0.3 / (seq_len - 1)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=seasonal_attention)
        
        # Create feature names with day-of-week info
        base_time = datetime.now() - timedelta(hours=seq_len)
        feature_names = []
        for i in range(seq_len):
            ts = base_time + timedelta(hours=i)
            feature_names.append(f"price_{ts.strftime('%a_%H')}")  # Mon_14, Tue_15, etc.
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should detect seasonal patterns
        input_instance = np.random.randn(seq_len)
        seasonal_analysis = analyzer.analyze_seasonal_attention_patterns(input_instance)
        
        assert isinstance(seasonal_analysis, dict)
        assert "weekly_pattern" in seasonal_analysis
        assert "daily_pattern" in seasonal_analysis
        assert "seasonal_strength" in seasonal_analysis
        
        # Should detect strong weekly pattern
        assert seasonal_analysis["seasonal_strength"] > 0.5
    
    def test_validate_input_for_temporal_analysis(self):
        """Test input validation for temporal analysis."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        analyzer = TemporalAttentionAnalyzer(Mock(), ["feature_1"])
        
        # Should require model with temporal attention capability
        valid_model = Mock()
        valid_model.get_attention_weights = Mock()
        is_valid, error = analyzer.validate_input(valid_model, ["feature_1"])
        assert is_valid
        
        # Should validate feature names have temporal structure
        temporal_features = [f"BTC_price_{i:02d}:00" for i in range(24)]
        is_valid, error = analyzer.validate_input(valid_model, temporal_features)
        assert is_valid
        
        # Should reject non-temporal feature names
        non_temporal_features = ["random_feature_1", "random_feature_2"]
        is_valid, error = analyzer.validate_input(valid_model, non_temporal_features)
        # Should still be valid but may issue warning about optimal temporal naming
        assert is_valid
    
    def test_get_explanation_metadata(self):
        """Test getting temporal analysis metadata."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        config = {
            'analyze_recency_bias': True,
            'detect_periodic_patterns': True,
            'time_window_analysis': True
        }
        
        analyzer = TemporalAttentionAnalyzer(Mock(), ["f1", "f2"], config)
        metadata = analyzer.get_explanation_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata["explainer_type"] == "temporal_attention"
        assert "temporal_analysis_capabilities" in metadata
        assert "supported_patterns" in metadata
        
        capabilities = metadata["temporal_analysis_capabilities"]
        assert "recency_bias" in capabilities
        assert "periodic_patterns" in capabilities
        assert "regime_transitions" in capabilities
    
    def test_performance_requirements_temporal_analysis(self):
        """Test that temporal analysis meets performance requirements."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        import time
        
        seq_len = 64
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 8, seq_len, seq_len))
        mock_model.predict = Mock(return_value=np.array([0.6]))
        
        analyzer = TemporalAttentionAnalyzer(mock_model, [f"price_t_{i}" for i in range(seq_len)])
        
        # Should perform temporal analysis in <500ms
        input_instance = np.random.randn(seq_len)
        
        start_time = time.time()
        explanation = analyzer.explain_instance(input_instance)
        end_time = time.time()
        
        analysis_time = (end_time - start_time) * 1000  # Convert to ms
        assert analysis_time < 500, f"Temporal analysis took {analysis_time}ms, should be <500ms"


class TestTemporalAttentionAnalyzerErrorHandling:
    """Test error handling and edge cases for TemporalAttentionAnalyzer."""
    
    def test_handles_irregular_time_series(self):
        """Test handling of irregular time series data."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        # Mock model
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 20, 20))
        
        # Irregular timestamps (missing some hours)
        irregular_times = [
            datetime.now() - timedelta(hours=i) 
            for i in [0, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 48, 50, 55, 60, 65, 70, 75, 80]
        ]
        feature_names = [f"price_{ts.strftime('%H:%M')}" for ts in irregular_times]
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should handle irregular timestamps gracefully
        input_instance = np.random.randn(20)
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        assert explanation.explanation_type == "temporal_attention"
    
    def test_handles_single_timestamp(self):
        """Test handling of single timestamp (no temporal patterns possible)."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 1, 1, 1))
        
        analyzer = TemporalAttentionAnalyzer(mock_model, ["single_feature"])
        
        # Should handle gracefully but with limited temporal analysis
        input_instance = np.array([1.0])
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        # Temporal analysis should be limited or indicate insufficient data
        assert "insufficient_temporal_data" in explanation.explanation_metadata
    
    def test_handles_attention_with_no_temporal_structure(self):
        """Test handling when attention weights show no temporal structure."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 32
        
        # Create completely random attention (no temporal structure)
        random_attention = torch.randn(1, 4, seq_len, seq_len)
        random_attention = torch.softmax(random_attention, dim=-1)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=random_attention)
        
        feature_names = [f"price_t_{i}" for i in range(seq_len)]
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should analyze but report low temporal structure
        input_instance = np.random.randn(seq_len)
        temporal_analysis = analyzer.analyze_recency_bias(input_instance)
        
        # Should indicate weak temporal patterns
        assert temporal_analysis["recency_score"] < 0.7  # Random should have low recency bias
        assert "temporal_structure_strength" in temporal_analysis
        assert temporal_analysis["temporal_structure_strength"] < 0.5


class TestTemporalAttentionAnalyzerIntegration:
    """Test integration with trading systems and existing XAI framework."""
    
    def test_integration_with_trading_signals(self):
        """Test integration with trading signal interpretation."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 48  # 2 days of hourly data
        
        # Create attention pattern indicating trend reversal focus
        reversal_attention = torch.zeros(1, 1, seq_len, seq_len)
        
        # Model focuses on positions around trend changes (positions 12, 24, 36)
        trend_change_positions = [12, 24, 36]
        for i in range(seq_len):
            attention_weights = torch.zeros(seq_len)
            for pos in trend_change_positions:
                if pos < seq_len:
                    attention_weights[max(0, pos-2):min(seq_len, pos+3)] = 0.8
            reversal_attention[0, 0, i, :] = torch.softmax(attention_weights, dim=0)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=reversal_attention)
        mock_model.predict = Mock(return_value=np.array([0.8]))  # Strong BUY signal
        
        # Create realistic trading feature names
        base_time = datetime.now() - timedelta(hours=seq_len)
        feature_names = []
        for i in range(seq_len):
            ts = base_time + timedelta(hours=i)
            feature_names.append(f"BTC_price_{ts.strftime('%m%d_%H')}")
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should provide trading-relevant temporal analysis
        input_instance = np.random.randn(seq_len)
        explanation = analyzer.explain_instance(input_instance)
        
        assert explanation.model_prediction == 0.8
        
        # Should identify trend reversal patterns
        metadata = explanation.explanation_metadata
        assert "trading_pattern_analysis" in metadata
        
        trading_patterns = metadata["trading_pattern_analysis"]
        assert "trend_reversal_indicators" in trading_patterns
        assert len(trading_patterns["trend_reversal_indicators"]) > 0
    
    def test_multi_asset_temporal_analysis(self):
        """Test temporal analysis for multi-asset trading scenarios."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        # 3 assets × 24 hours = 72 features
        seq_len, num_assets = 24, 3
        total_features = seq_len * num_assets
        
        mock_attention = torch.randn(1, 8, total_features, total_features)
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        # Create multi-asset feature names
        assets = ['BTC', 'ETH', 'SOL']
        feature_names = []
        base_time = datetime.now() - timedelta(hours=seq_len)
        
        for asset in assets:
            for i in range(seq_len):
                ts = base_time + timedelta(hours=i)
                feature_names.append(f"{asset}_price_{ts.strftime('%H')}")
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should handle multi-asset temporal analysis
        input_instance = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_instance)
        
        assert len(explanation.feature_importance) == total_features
        
        # Should provide cross-asset temporal analysis
        metadata = explanation.explanation_metadata
        assert "cross_asset_temporal_analysis" in metadata
        
        cross_asset_analysis = metadata["cross_asset_temporal_analysis"]
        for asset in assets:
            assert f"{asset}_temporal_patterns" in cross_asset_analysis
    
    def test_integration_with_existing_xai_framework(self):
        """Test integration with existing XAI framework components."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        # Should be compatible with ExplanationData serialization
        seq_len = 16
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 2, seq_len, seq_len))
        mock_model.predict = Mock(return_value=np.array([0.65]))
        
        analyzer = TemporalAttentionAnalyzer(mock_model, [f"f_{i}" for i in range(seq_len)])
        
        input_instance = np.random.randn(seq_len)
        explanation = analyzer.explain_instance(input_instance)
        
        # Should be serializable
        json_str = explanation.to_json()
        reconstructed = ExplanationData.from_json(json_str)
        
        assert reconstructed.explanation_type == "temporal_attention"
        assert len(reconstructed.feature_importance) == seq_len
        
        # Should work with explanation merging
        explanation2 = analyzer.explain_instance(input_instance)
        merged = ExplanationData.merge_explanations([explanation, explanation2])
        
        assert merged.explanation_type == "merged"
        assert len(merged.feature_importance) == seq_len