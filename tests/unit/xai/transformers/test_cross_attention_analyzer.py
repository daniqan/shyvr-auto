"""
Tests for CrossAttentionAnalyzer - Multivariate Relationship Analysis

Following TDD principles, these tests define the expected behavior
of cross-attention analysis for multivariate relationships BEFORE implementation.

These tests MUST FAIL initially as CrossAttentionAnalyzer doesn't exist yet.
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


class TestCrossAttentionAnalyzer:
    """
    Test suite for CrossAttentionAnalyzer following TDD principles.
    
    CrossAttentionAnalyzer should analyze cross-attention patterns between
    different assets and features to understand multivariate relationships
    in trading decisions.
    """
    
    def test_cross_attention_analyzer_import_fails_initially(self):
        """Test that CrossAttentionAnalyzer import fails initially (TDD)."""
        # This test should FAIL initially - CrossAttentionAnalyzer doesn't exist yet
        with pytest.raises(ImportError):
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
    
    def test_cross_attention_analyzer_inherits_from_base_explainer(self):
        """Test that CrossAttentionAnalyzer inherits from BaseExplainer."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        assert issubclass(CrossAttentionAnalyzer, BaseExplainer)
    
    def test_cross_attention_analyzer_initialization(self):
        """Test CrossAttentionAnalyzer initialization with multivariate data."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Mock transformer model with cross-attention capability
        num_assets, seq_len = 3, 32
        total_features = num_assets * seq_len
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 8, total_features, total_features)
        )
        
        # Multi-asset feature names
        feature_names = []
        assets = ['BTC', 'ETH', 'SOL']
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        config = {
            'analyze_asset_correlations': True,
            'detect_lead_lag_relationships': True,
            'identify_arbitrage_patterns': True,
            'cross_asset_momentum': True,
            'asset_list': assets
        }
        
        analyzer = CrossAttentionAnalyzer(
            model=mock_model,
            feature_names=feature_names,
            config=config
        )
        
        assert analyzer.model == mock_model
        assert analyzer.feature_names == feature_names
        assert analyzer.config['analyze_asset_correlations'] is True
        assert analyzer.config['asset_list'] == assets
    
    def test_analyze_asset_correlations(self):
        """Test analysis of cross-asset correlations from attention patterns."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 3, 24
        assets = ['BTC', 'ETH', 'SOL']
        total_features = num_assets * seq_len
        
        # Create cross-attention showing BTC → ETH correlation
        cross_attention = torch.zeros(1, 1, total_features, total_features)
        
        # BTC features (0:24) strongly attend to ETH features (24:48)
        for i in range(seq_len):
            for j in range(seq_len):
                # BTC_t_i attends to ETH_t_j with correlation strength
                btc_idx = i
                eth_idx = seq_len + j
                cross_attention[0, 0, btc_idx, eth_idx] = 0.8 if abs(i - j) <= 2 else 0.1
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=cross_attention)
        
        # Create feature names
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(mock_model, feature_names, {'asset_list': assets})
        
        # Should analyze cross-asset correlations
        input_instance = np.random.randn(total_features)
        correlation_analysis = analyzer.analyze_asset_correlations(input_instance)
        
        assert isinstance(correlation_analysis, dict)
        assert "correlation_matrix" in correlation_analysis
        assert "dominant_correlations" in correlation_analysis
        assert "correlation_strength" in correlation_analysis
        
        # Should detect BTC-ETH correlation
        correlations = correlation_analysis["dominant_correlations"]
        assert any("BTC" in corr and "ETH" in corr for corr in correlations)
        
        # Correlation matrix should be symmetric
        corr_matrix = correlation_analysis["correlation_matrix"]
        assert isinstance(corr_matrix, (torch.Tensor, np.ndarray))
        assert corr_matrix.shape == (num_assets, num_assets)
    
    def test_detect_lead_lag_relationships(self):
        """Test detection of lead-lag relationships between assets."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 2, 32
        assets = ['BTC', 'ETH']
        total_features = num_assets * seq_len
        
        # Create attention pattern showing BTC leads ETH by 2 time steps
        lead_lag_attention = torch.zeros(1, 1, total_features, total_features)
        
        # ETH at time t attends to BTC at time t-2 (lag relationship)
        for eth_t in range(seq_len):
            for btc_t in range(seq_len):
                eth_idx = seq_len + eth_t  # ETH features start at index seq_len
                btc_idx = btc_t
                
                # ETH_t attends strongly to BTC_{t-2}
                if btc_t == max(0, eth_t - 2):
                    lead_lag_attention[0, 0, eth_idx, btc_idx] = 0.9
                else:
                    lead_lag_attention[0, 0, eth_idx, btc_idx] = 0.1 / (seq_len - 1)
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=lead_lag_attention)
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(mock_model, feature_names, {'asset_list': assets})
        
        # Should detect lead-lag relationship
        input_instance = np.random.randn(total_features)
        lead_lag_analysis = analyzer.detect_lead_lag_relationships(input_instance)
        
        assert isinstance(lead_lag_analysis, dict)
        assert "lead_lag_pairs" in lead_lag_analysis
        assert "lag_times" in lead_lag_analysis
        assert "relationship_strength" in lead_lag_analysis
        
        # Should identify BTC leads ETH
        lead_lag_pairs = lead_lag_analysis["lead_lag_pairs"]
        assert any(
            pair['leader'] == 'BTC' and pair['follower'] == 'ETH' 
            for pair in lead_lag_pairs
        )
        
        # Should identify 2-step lag
        lag_times = lead_lag_analysis["lag_times"]
        btc_eth_lag = next(
            (lag['lag_steps'] for lag in lag_times 
             if lag['leader'] == 'BTC' and lag['follower'] == 'ETH'), 
            None
        )
        assert btc_eth_lag is not None
        assert abs(btc_eth_lag - 2) <= 1  # Allow some tolerance
    
    def test_identify_arbitrage_patterns(self):
        """Test identification of arbitrage opportunities in cross-attention."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 3, 16
        assets = ['BTC', 'ETH', 'SOL']
        total_features = num_assets * seq_len
        
        # Create attention pattern showing triangular arbitrage pattern
        # BTC → ETH → SOL → BTC
        arbitrage_attention = torch.zeros(1, 1, total_features, total_features)
        
        for t in range(seq_len):
            # BTC attends to ETH
            btc_idx = t
            eth_idx = seq_len + t
            arbitrage_attention[0, 0, btc_idx, eth_idx] = 0.7
            
            # ETH attends to SOL
            sol_idx = 2 * seq_len + t
            arbitrage_attention[0, 0, eth_idx, sol_idx] = 0.7
            
            # SOL attends back to BTC (completing the triangle)
            arbitrage_attention[0, 0, sol_idx, btc_idx] = 0.7
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=arbitrage_attention)
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(mock_model, feature_names, {'asset_list': assets})
        
        # Should identify arbitrage patterns
        input_instance = np.random.randn(total_features)
        arbitrage_analysis = analyzer.identify_arbitrage_patterns(input_instance)
        
        assert isinstance(arbitrage_analysis, dict)
        assert "arbitrage_triangles" in arbitrage_analysis
        assert "arbitrage_strength" in arbitrage_analysis
        assert "profit_potential" in arbitrage_analysis
        
        # Should detect BTC-ETH-SOL triangle
        triangles = arbitrage_analysis["arbitrage_triangles"]
        assert len(triangles) > 0
        
        # Should find triangle involving all three assets
        for triangle in triangles:
            triangle_assets = set(triangle["assets"])
            if triangle_assets == set(assets):
                assert triangle["strength"] > 0.5
                break
        else:
            pytest.fail("Expected to find BTC-ETH-SOL arbitrage triangle")
    
    def test_analyze_cross_asset_momentum(self):
        """Test analysis of cross-asset momentum patterns."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 2, 20
        assets = ['BTC', 'ETH']
        total_features = num_assets * seq_len
        
        # Create momentum spillover pattern: BTC momentum → ETH momentum
        momentum_attention = torch.zeros(1, 1, total_features, total_features)
        
        # Recent ETH positions attend to recent BTC positions (momentum spillover)
        for eth_t in range(15, seq_len):  # Recent ETH positions
            for btc_t in range(10, seq_len):  # Recent BTC positions
                eth_idx = seq_len + eth_t
                btc_idx = btc_t
                
                # Stronger attention for more recent BTC momentum
                attention_strength = 0.8 if btc_t >= 15 else 0.4
                momentum_attention[0, 0, eth_idx, btc_idx] = attention_strength
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=momentum_attention)
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(mock_model, feature_names, {'asset_list': assets})
        
        # Should analyze momentum spillover
        input_instance = np.random.randn(total_features)
        momentum_analysis = analyzer.analyze_cross_asset_momentum(input_instance)
        
        assert isinstance(momentum_analysis, dict)
        assert "momentum_spillovers" in momentum_analysis
        assert "momentum_leaders" in momentum_analysis
        assert "momentum_strength" in momentum_analysis
        
        # Should identify BTC as momentum leader
        momentum_leaders = momentum_analysis["momentum_leaders"]
        assert "BTC" in [leader["asset"] for leader in momentum_leaders]
        
        # Should detect BTC → ETH spillover
        spillovers = momentum_analysis["momentum_spillovers"]
        assert any(
            spill["from"] == "BTC" and spill["to"] == "ETH"
            for spill in spillovers
        )
    
    def test_cross_attention_analyzer_explain_instance(self):
        """Test explaining instance with cross-attention analysis."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 3, 16
        assets = ['BTC', 'ETH', 'SOL']
        total_features = num_assets * seq_len
        
        mock_cross_attention = torch.randn(1, 4, total_features, total_features)
        mock_prediction = np.array([0.85])  # Strong bullish signal
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=mock_cross_attention)
        mock_model.predict = Mock(return_value=mock_prediction)
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_close_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': assets}
        )
        
        # Should generate cross-attention explanation
        input_instance = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        assert explanation.explanation_type == "cross_attention"
        assert len(explanation.feature_importance) == total_features
        assert explanation.model_prediction == 0.85
        
        # Should include cross-attention analysis metadata
        metadata = explanation.explanation_metadata
        assert "cross_asset_correlations" in metadata
        assert "lead_lag_relationships" in metadata
        assert "arbitrage_patterns" in metadata
        assert "momentum_analysis" in metadata
    
    def test_get_cross_asset_feature_importance(self):
        """Test getting feature importance considering cross-asset relationships."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 2, 12
        assets = ['BTC', 'ETH']
        total_features = num_assets * seq_len
        
        # Create cross-attention where BTC features are more important
        cross_attention = torch.zeros(1, 1, total_features, total_features)
        
        # All positions attend more strongly to BTC features
        for i in range(total_features):
            for j in range(seq_len):  # BTC features (first seq_len positions)
                cross_attention[0, 0, i, j] = 0.7 / seq_len
            for j in range(seq_len, total_features):  # ETH features
                cross_attention[0, 0, i, j] = 0.3 / seq_len
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=cross_attention)
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(mock_model, feature_names, {'asset_list': assets})
        
        # Should compute cross-asset weighted importance
        input_instance = np.random.randn(total_features)
        importance = analyzer.get_feature_importance(input_instance)
        
        assert isinstance(importance, dict)
        assert len(importance) == total_features
        
        # BTC features should have higher importance on average
        btc_importance = np.mean([importance[f"BTC_price_t_{t}"] for t in range(seq_len)])
        eth_importance = np.mean([importance[f"ETH_price_t_{t}"] for t in range(seq_len)])
        
        assert btc_importance > eth_importance
    
    def test_validate_input_for_cross_attention(self):
        """Test input validation for cross-attention analysis."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        analyzer = CrossAttentionAnalyzer(Mock(), ["feature_1"], {'asset_list': ['BTC']})
        
        # Should require model with cross-attention capability
        valid_model = Mock()
        valid_model.get_cross_attention_weights = Mock()
        is_valid, error = analyzer.validate_input(valid_model, ["BTC_price_t_0"])
        assert is_valid
        
        # Should require multivariate features for meaningful analysis
        single_asset_features = ["BTC_price_t_0"]
        is_valid, error = analyzer.validate_input(valid_model, single_asset_features)
        # Should warn but still be valid
        assert is_valid
        
        # Should validate asset list consistency
        multi_asset_features = ["BTC_price_t_0", "ETH_price_t_0", "SOL_price_t_0"]
        config = {'asset_list': ['BTC', 'ETH']}  # Missing SOL
        analyzer_inconsistent = CrossAttentionAnalyzer(valid_model, multi_asset_features, config)
        
        is_valid, error = analyzer_inconsistent.validate_input(valid_model, multi_asset_features)
        # Should detect inconsistency but still be valid (will auto-detect assets)
        assert is_valid
    
    def test_get_explanation_metadata(self):
        """Test getting cross-attention analysis metadata."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        config = {
            'analyze_asset_correlations': True,
            'detect_lead_lag_relationships': True,
            'asset_list': ['BTC', 'ETH', 'SOL']
        }
        
        analyzer = CrossAttentionAnalyzer(Mock(), ["f1", "f2"], config)
        metadata = analyzer.get_explanation_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata["explainer_type"] == "cross_attention"
        assert "cross_attention_capabilities" in metadata
        assert "supported_assets" in metadata
        assert metadata["supported_assets"] == ['BTC', 'ETH', 'SOL']
        
        capabilities = metadata["cross_attention_capabilities"]
        assert "asset_correlations" in capabilities
        assert "lead_lag_detection" in capabilities
        assert "arbitrage_analysis" in capabilities
    
    def test_performance_requirements_cross_attention(self):
        """Test that cross-attention analysis meets performance requirements."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        import time
        
        num_assets, seq_len = 3, 32
        total_features = num_assets * seq_len
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 8, total_features, total_features)
        )
        mock_model.predict = Mock(return_value=np.array([0.7]))
        
        assets = ['BTC', 'ETH', 'SOL']
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': assets}
        )
        
        # Should perform cross-attention analysis in <500ms
        input_instance = np.random.randn(total_features)
        
        start_time = time.time()
        explanation = analyzer.explain_instance(input_instance)
        end_time = time.time()
        
        analysis_time = (end_time - start_time) * 1000  # Convert to ms
        assert analysis_time < 500, f"Cross-attention analysis took {analysis_time}ms, should be <500ms"


class TestCrossAttentionAnalyzerErrorHandling:
    """Test error handling and edge cases for CrossAttentionAnalyzer."""
    
    def test_handles_single_asset_gracefully(self):
        """Test handling when only single asset is present."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        seq_len = 16
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 4, seq_len, seq_len)
        )
        
        # Single asset features
        feature_names = [f"BTC_price_t_{t}" for t in range(seq_len)]
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': ['BTC']}
        )
        
        # Should handle gracefully but indicate limited cross-asset analysis
        input_instance = np.random.randn(seq_len)
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        # Should indicate single asset limitation
        assert "single_asset_limitation" in explanation.explanation_metadata
    
    def test_handles_mismatched_asset_dimensions(self):
        """Test handling when asset dimensions don't match expectations."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Mismatched dimensions: 2 assets but 3×seq_len features
        seq_len = 10
        total_features = 3 * seq_len  # 3 assets worth of features
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 4, total_features, total_features)
        )
        
        feature_names = []
        for asset in ['BTC', 'ETH', 'SOL']:  # 3 assets
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        # Config only lists 2 assets
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': ['BTC', 'ETH']}  # Missing SOL
        )
        
        # Should detect and auto-correct asset list
        input_instance = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        # Should include auto-detected assets in metadata
        assert "auto_detected_assets" in explanation.explanation_metadata
    
    def test_handles_attention_with_nan_values(self):
        """Test handling of NaN values in cross-attention weights."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        num_assets, seq_len = 2, 12
        total_features = num_assets * seq_len
        
        # Create cross-attention with NaN values
        cross_attention = torch.randn(1, 2, total_features, total_features)
        cross_attention[0, 0, :5, :] = float('nan')  # Inject NaN
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=cross_attention)
        
        feature_names = []
        for asset in ['BTC', 'ETH']:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': ['BTC', 'ETH']}
        )
        
        # Should handle NaN gracefully
        input_instance = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_instance)
        
        # Should not contain NaN in final importance scores
        for importance in explanation.feature_importance.values():
            assert not np.isnan(importance)
        
        # Should indicate NaN handling in metadata
        assert "nan_values_detected" in explanation.explanation_metadata


class TestCrossAttentionAnalyzerIntegration:
    """Test integration with trading systems and existing XAI framework."""
    
    def test_integration_with_trading_strategies(self):
        """Test integration with multi-asset trading strategies."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Portfolio with major crypto assets
        assets = ['BTC', 'ETH', 'SOL', 'ADA']
        seq_len = 24  # Daily data
        total_features = len(assets) * seq_len
        
        # Create attention showing portfolio rebalancing signals
        portfolio_attention = torch.randn(1, 8, total_features, total_features)
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=portfolio_attention)
        mock_model.predict = Mock(return_value=np.array([0.6, 0.7, 0.4, 0.5]))  # Multi-asset predictions
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': assets}
        )
        
        # Should provide portfolio-relevant analysis
        input_instance = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_instance)
        
        assert len(explanation.feature_importance) == total_features
        
        # Should provide portfolio analysis
        metadata = explanation.explanation_metadata
        assert "portfolio_analysis" in metadata
        
        portfolio_info = metadata["portfolio_analysis"]
        assert "asset_weights" in portfolio_info
        assert "rebalancing_signals" in portfolio_info
        assert "diversification_score" in portfolio_info
    
    def test_real_time_cross_attention_analysis(self):
        """Test real-time cross-attention analysis for live trading."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Streaming data scenario
        assets = ['BTC', 'ETH']
        seq_len = 60  # 1 hour of minute data
        total_features = len(assets) * seq_len
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 4, total_features, total_features)
        )
        
        feature_names = []
        for asset in assets:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_min_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {
                'asset_list': assets,
                'real_time_mode': True,
                'latency_optimization': True
            }
        )
        
        # Should handle streaming updates efficiently
        input_instance = np.random.randn(total_features)
        
        # Simulate multiple rapid updates
        import time
        start_time = time.time()
        
        for _ in range(5):  # 5 rapid updates
            explanation = analyzer.explain_instance(input_instance)
            # Modify last few features to simulate new data
            input_instance[-10:] = np.random.randn(10)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / 5
        
        # Should maintain <100ms per analysis for real-time trading
        assert avg_time < 100, f"Average analysis time {avg_time}ms should be <100ms for real-time trading"
    
    def test_integration_with_existing_xai_components(self):
        """Test integration with existing XAI framework components."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Should work with ExplanationData merging
        num_assets, seq_len = 2, 16
        total_features = num_assets * seq_len
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(
            return_value=torch.randn(1, 2, total_features, total_features)
        )
        mock_model.predict = Mock(return_value=np.array([0.75]))
        
        feature_names = []
        for asset in ['BTC', 'ETH']:
            for t in range(seq_len):
                feature_names.append(f"{asset}_price_t_{t}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': ['BTC', 'ETH']}
        )
        
        input_instance = np.random.randn(total_features)
        
        # Generate multiple explanations
        explanation1 = analyzer.explain_instance(input_instance)
        explanation2 = analyzer.explain_instance(input_instance + 0.1)
        
        # Should be compatible with ExplanationData merging
        merged = ExplanationData.merge_explanations([explanation1, explanation2])
        
        assert merged.explanation_type == "merged"
        assert len(merged.feature_importance) == total_features
        assert "cross_attention" in merged.explanation_metadata["merged_from"]
        
        # Should be serializable
        json_str = explanation1.to_json()
        reconstructed = ExplanationData.from_json(json_str)
        
        assert reconstructed.explanation_type == "cross_attention"
        assert len(reconstructed.feature_importance) == total_features