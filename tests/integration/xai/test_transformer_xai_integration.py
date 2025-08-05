"""
Integration Tests for Transformer XAI Framework Integration

Following TDD principles, these tests define the expected behavior
of complete XAI integration with transformer models BEFORE implementation.

These tests MUST FAIL initially as the integrated system doesn't exist yet.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio

# Import existing XAI framework components
from src.xai.base import BaseExplainer
from src.xai.data_models import ExplanationData
from src.xai.trading_integration import TradingExplanationManager, TradingExplanation
from src.xai.factory import ExplainerFactory


class TestTransformerXAIIntegration:
    """
    Integration test suite for complete Transformer XAI system.
    
    Tests the full pipeline from transformer models to trading explanations,
    including performance requirements and regulatory compliance.
    """
    
    def test_transformer_xai_components_import_fail_initially(self):
        """Test that new XAI components don't exist yet (TDD)."""
        # These imports should FAIL initially
        with pytest.raises(ImportError):
            from src.xai.transformers.attention_explainer import AttentionExplainer
        
        with pytest.raises(ImportError):
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        
        with pytest.raises(ImportError):
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
    
    def test_explainer_factory_supports_transformer_explainers(self):
        """Test that ExplainerFactory can create transformer explainers."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("Transformer XAI components not implemented yet")
        
        factory = ExplainerFactory()
        
        # Should support new transformer explainer types
        supported_types = factory.get_supported_explainer_types()
        assert "attention" in supported_types
        assert "temporal_attention" in supported_types
        assert "cross_attention" in supported_types
        
        # Mock transformer model
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 32, 32))
        mock_model.__class__.__name__ = "iTransformerPredictor"
        
        feature_names = [f"BTC_price_t_{i}" for i in range(32)]
        
        # Should create attention explainer
        attention_explainer = factory.create_explainer(
            explainer_type="attention",
            model=mock_model,
            feature_names=feature_names
        )
        assert isinstance(attention_explainer, AttentionExplainer)
        
        # Should create temporal attention analyzer
        temporal_explainer = factory.create_explainer(
            explainer_type="temporal_attention",
            model=mock_model,
            feature_names=feature_names
        )
        assert isinstance(temporal_explainer, TemporalAttentionAnalyzer)
        
        # Should create cross attention analyzer for multi-asset
        multi_asset_features = []
        for asset in ['BTC', 'ETH']:
            for i in range(16):
                multi_asset_features.append(f"{asset}_price_t_{i}")
        
        cross_explainer = factory.create_explainer(
            explainer_type="cross_attention",
            model=mock_model,
            feature_names=multi_asset_features,
            config={'asset_list': ['BTC', 'ETH']}
        )
        assert isinstance(cross_explainer, CrossAttentionAnalyzer)
    
    def test_trading_explanation_manager_supports_transformer_explanations(self):
        """Test TradingExplanationManager integration with transformer explainers."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Create enhanced factory with transformer support
        factory = ExplainerFactory()
        manager = TradingExplanationManager(explainer_factory=factory)
        
        # Mock transformer model
        seq_len = 48
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 8, seq_len, seq_len))
        mock_model.predict = Mock(return_value=np.array([0.8]))
        mock_model.__class__.__name__ = "iTransformerPredictor"
        
        feature_names = [f"BTC_price_{i:02d}h" for i in range(seq_len)]
        feature_data = np.random.randn(seq_len)
        
        # Should create attention-based trading explanation
        explanation = asyncio.run(manager.explain_trading_decision(
            decision_id="test_decision_001",
            model=mock_model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="BTC",
            model_type="transformer",
            explainer_type="attention",
            metadata={"transformer_type": "iTransformer"}
        ))
        
        assert isinstance(explanation, TradingExplanation)
        assert explanation.explanation_data.explanation_type == "attention"
        assert explanation.decision_type == "buy"
        assert explanation.symbol == "BTC"
        assert explanation.model_type == "transformer"
        
        # Should include attention-specific metadata
        assert "attention_weights" in explanation.explanation_data.explanation_metadata
    
    def test_attention_weight_extraction_from_all_transformer_models(self):
        """Test attention weight extraction from all transformer model types."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Test all transformer types
        transformer_types = [
            ("iTransformerPredictor", (1, 8, 64, 64)),
            ("PatchTSTPredictor", (1, 6, 32, 32)),
            ("TimesMixerPredictor", (1, 4, 48, 48)),
            ("TransformerPredictor", (1, 12, 128, 128)),
            ("TimesFMWrapper", (1, 16, 256, 256))
        ]
        
        for model_type, attention_shape in transformer_types:
            # Mock specific transformer model
            mock_model = Mock()
            mock_model.__class__.__name__ = model_type
            mock_model.get_attention_weights = Mock(return_value=torch.randn(*attention_shape))
            mock_model.predict = Mock(return_value=np.array([0.7]))
            
            seq_len = attention_shape[2]
            feature_names = [f"price_t_{i}" for i in range(seq_len)]
            
            explainer = AttentionExplainer(mock_model, feature_names)
            
            # Should extract attention weights successfully
            input_data = np.random.randn(seq_len)
            attention_weights = explainer.extract_attention_weights(input_data)
            
            assert attention_weights.shape == attention_shape
            
            # Should generate explanation
            explanation = explainer.explain_instance(input_data)
            assert isinstance(explanation, ExplanationData)
            assert explanation.explanation_type == "attention"
            
            # Should include model-specific metadata
            metadata = explanation.explanation_metadata
            assert "model_type" in metadata
            assert metadata["model_type"] == model_type
    
    def test_attention_rollout_for_long_range_dependencies(self):
        """Test attention rollout computation for tracking long-range dependencies."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock multi-layer transformer (6 layers)
        num_layers, num_heads, seq_len = 6, 8, 64
        
        # Create attention with progressive focus (each layer focuses further back)
        layer_attentions = []
        for layer in range(num_layers):
            attention = torch.zeros(1, num_heads, seq_len, seq_len)
            # Each layer focuses on positions further back in time
            focus_range = min(10 + layer * 5, seq_len)
            for i in range(seq_len):
                start_pos = max(0, i - focus_range)
                attention[0, :, i, start_pos:i+1] = torch.softmax(
                    torch.randn(num_heads, i - start_pos + 1), dim=1
                )
            layer_attentions.append(attention)
        
        mock_model = Mock()
        mock_model.get_layer_attention_weights = Mock(return_value=layer_attentions)
        mock_model.get_attention_weights = Mock(return_value=layer_attentions[-1])  # Last layer
        
        feature_names = [f"market_state_t_{i}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should compute attention rollout showing long-range dependencies
        input_data = np.random.randn(seq_len)
        rollout_attention = explainer.compute_attention_rollout(input_data)
        
        assert rollout_attention.shape == (seq_len, seq_len)
        
        # Earlier positions should receive some attention from later positions
        # (indicating long-range dependencies)
        early_to_late_attention = rollout_attention[-10:, :10].sum()
        assert early_to_late_attention > 0.1, "Should show long-range dependencies"
        
        # Rollout should preserve attention conservation (rows sum to 1)
        row_sums = rollout_attention.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    
    def test_head_wise_attention_heatmap_generation(self):
        """Test generation of head-wise attention heatmaps."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        num_heads, seq_len = 8, 32
        mock_attention = torch.randn(1, num_heads, seq_len, seq_len)
        
        # Create specialized attention patterns for different heads
        for head in range(num_heads):
            if head == 0:  # Recency head
                for i in range(seq_len):
                    mock_attention[0, head, i, max(0, i-5):i+1] = torch.softmax(
                        torch.tensor([1, 2, 3, 4, 5][:i+1-max(0, i-5)], dtype=torch.float), dim=0
                    )
            elif head == 1:  # Long-range head
                for i in range(seq_len):
                    mock_attention[0, head, i, :i//2] = torch.softmax(
                        torch.randn(i//2), dim=0
                    ) if i > 0 else torch.tensor([1.0])
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        feature_names = [f"price_t_{i:02d}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should generate head-wise heatmaps
        input_data = np.random.randn(seq_len)
        head_heatmaps = explainer.generate_head_wise_heatmaps(input_data)
        
        assert isinstance(head_heatmaps, dict)
        assert len(head_heatmaps) == num_heads
        
        # Each head should have heatmap data
        for head_idx in range(num_heads):
            head_key = f"head_{head_idx}"
            assert head_key in head_heatmaps
            
            heatmap_data = head_heatmaps[head_key]
            assert "heatmap_matrix" in heatmap_data
            assert "specialization" in heatmap_data
            assert "attention_pattern_type" in heatmap_data
            
            # Heatmap matrix should match attention dimensions
            heatmap_matrix = heatmap_data["heatmap_matrix"]
            assert heatmap_matrix.shape == (seq_len, seq_len)
        
        # Head 0 should be identified as "recency" focused
        assert head_heatmaps["head_0"]["specialization"] == "recency_focused"
        
        # Head 1 should be identified as "long_range" focused
        assert head_heatmaps["head_1"]["specialization"] == "long_range_focused"
    
    def test_temporal_pattern_analysis_for_trading_signals(self):
        """Test temporal attention pattern analysis for trading signal interpretation."""
        try:
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("TemporalAttentionAnalyzer not implemented yet")
        
        seq_len = 72  # 3 days of hourly data
        
        # Create attention pattern showing pre-breakout focus
        # Model attends to consolidation period before breakout
        breakout_attention = torch.zeros(1, 4, seq_len, seq_len)
        
        # Consolidation period: hours 24-48
        # Breakout period: hours 60-72
        for query_pos in range(60, seq_len):  # Recent positions (during breakout)
            # Strong attention to consolidation period
            consolidation_weights = torch.zeros(seq_len)
            consolidation_weights[24:48] = torch.softmax(torch.randn(24), dim=0) * 0.8
            # Some attention to recent momentum
            consolidation_weights[56:query_pos] = torch.softmax(torch.randn(query_pos-56), dim=0) * 0.2
            
            breakout_attention[0, :, query_pos, :] = consolidation_weights.unsqueeze(0).repeat(4, 1)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=breakout_attention)
        mock_model.predict = Mock(return_value=np.array([0.85]))  # Strong BUY signal
        
        # Create realistic trading feature names
        base_time = datetime.now() - timedelta(hours=seq_len)
        feature_names = []
        for h in range(seq_len):
            timestamp = base_time + timedelta(hours=h)
            feature_names.append(f"BTC_price_{timestamp.strftime('%m%d_%H')}")
        
        analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        # Should identify trading pattern
        input_data = np.random.randn(seq_len)
        explanation = analyzer.explain_instance(input_data)
        
        assert explanation.model_prediction == 0.85
        
        # Should detect consolidation-breakout pattern
        metadata = explanation.explanation_metadata
        assert "trading_pattern_analysis" in metadata
        
        pattern_analysis = metadata["trading_pattern_analysis"]
        assert "pattern_type" in pattern_analysis
        assert pattern_analysis["pattern_type"] in ["consolidation_breakout", "pre_breakout_accumulation"]
        
        # Should identify key time periods
        assert "key_periods" in pattern_analysis
        key_periods = pattern_analysis["key_periods"]
        
        # Should identify consolidation period
        consolidation_period = next(
            (period for period in key_periods if period["type"] == "consolidation"), 
            None
        )
        assert consolidation_period is not None
        assert 24 <= consolidation_period["start_hour"] <= 30
        assert 45 <= consolidation_period["end_hour"] <= 50
    
    def test_cross_asset_attention_analysis_for_portfolio_decisions(self):
        """Test cross-asset attention analysis for portfolio trading decisions."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        # Portfolio with correlated assets
        assets = ['BTC', 'ETH', 'SOL']
        seq_len = 48  # 2 days of hourly data
        total_features = len(assets) * seq_len
        
        # Create cross-attention showing BTC leading the market
        portfolio_attention = torch.zeros(1, 6, total_features, total_features)
        
        # ETH and SOL features attend strongly to BTC features with 1-2 hour lag
        for asset_idx, asset in enumerate(['ETH', 'SOL']):
            asset_start = (asset_idx + 1) * seq_len  # ETH starts at 48, SOL at 96
            
            for t in range(seq_len):
                feature_idx = asset_start + t
                
                # Attend to BTC features from 1-2 hours ago
                btc_lag_start = max(0, t - 2)
                btc_lag_end = max(1, t)
                
                attention_weights = torch.zeros(total_features)
                attention_weights[btc_lag_start:btc_lag_end] = torch.softmax(
                    torch.randn(btc_lag_end - btc_lag_start), dim=0
                ) * 0.7
                
                # Some attention to own past
                own_past_start = max(asset_start, asset_start + t - 5)
                attention_weights[own_past_start:asset_start + t] = torch.softmax(
                    torch.randn(asset_start + t - own_past_start), dim=0
                ) * 0.3
                
                portfolio_attention[0, :, feature_idx, :] = attention_weights.unsqueeze(0).repeat(6, 1)
        
        mock_model = Mock()
        mock_model.get_cross_attention_weights = Mock(return_value=portfolio_attention)
        mock_model.predict = Mock(return_value=np.array([0.7, 0.75, 0.65]))  # BUY signals for all
        
        # Create multi-asset feature names
        feature_names = []
        base_time = datetime.now() - timedelta(hours=seq_len)
        for asset in assets:
            for h in range(seq_len):
                timestamp = base_time + timedelta(hours=h)
                feature_names.append(f"{asset}_{timestamp.strftime('%m%d_%H')}")
        
        analyzer = CrossAttentionAnalyzer(
            mock_model, 
            feature_names, 
            {'asset_list': assets}
        )
        
        # Should identify portfolio dynamics
        input_data = np.random.randn(total_features)
        explanation = analyzer.explain_instance(input_data)
        
        # Should detect BTC leadership
        metadata = explanation.explanation_metadata
        assert "lead_lag_relationships" in metadata
        
        lead_lag = metadata["lead_lag_relationships"]
        btc_leadership = any(
            pair["leader"] == "BTC" and pair["follower"] in ["ETH", "SOL"]
            for pair in lead_lag["lead_lag_pairs"]
        )
        assert btc_leadership, "Should detect BTC market leadership"
        
        # Should provide portfolio allocation insights
        assert "portfolio_analysis" in metadata
        portfolio_info = metadata["portfolio_analysis"]
        assert "recommended_weights" in portfolio_info
        assert "risk_correlation" in portfolio_info
    
    def test_performance_requirements_integration(self):
        """Test that integrated XAI system meets performance requirements."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
        except ImportError:
            pytest.skip("Transformer XAI components not implemented yet")
        
        import time
        
        # Test with realistic trading scenario
        seq_len = 64
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 8, seq_len, seq_len))
        mock_model.predict = Mock(return_value=np.array([0.8]))
        mock_model.__class__.__name__ = "iTransformerPredictor"
        
        feature_names = [f"BTC_price_t_{i}" for i in range(seq_len)]
        input_data = np.random.randn(seq_len)
        
        # Test inference overhead requirement (<10ms)
        base_inference_time = 50  # Simulated base inference time in ms
        
        # Attention explanation should add <10ms
        attention_explainer = AttentionExplainer(mock_model, feature_names)
        
        start_time = time.time()
        attention_explanation = attention_explainer.explain_instance(input_data)
        attention_time = (time.time() - start_time) * 1000
        
        assert attention_time < 10, f"Attention explanation added {attention_time}ms, should be <10ms"
        
        # Full explanation should be <500ms
        temporal_analyzer = TemporalAttentionAnalyzer(mock_model, feature_names)
        
        start_time = time.time()
        temporal_explanation = temporal_analyzer.explain_instance(input_data)
        full_explanation_time = (time.time() - start_time) * 1000
        
        assert full_explanation_time < 500, f"Full explanation took {full_explanation_time}ms, should be <500ms"
        
        # Combined system should maintain real-time performance
        factory = ExplainerFactory()
        manager = TradingExplanationManager(explainer_factory=factory)
        
        start_time = time.time()
        trading_explanation = asyncio.run(manager.explain_trading_decision(
            decision_id="perf_test_001",
            model=mock_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="BTC",
            model_type="transformer",
            explainer_type="attention"
        ))
        end_to_end_time = (time.time() - start_time) * 1000
        
        assert end_to_end_time < 1000, f"End-to-end explanation took {end_to_end_time}ms, should be <1000ms"
    
    def test_trading_explanation_manager_integration_with_transformers(self):
        """Test comprehensive TradingExplanationManager integration."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Enhanced trading explanation manager
        factory = ExplainerFactory()
        manager = TradingExplanationManager(
            explainer_factory=factory,
            cache_size=100,
            explanation_timeout=5.0
        )
        
        # Mock transformer ensemble (multiple models)
        models = {}
        model_types = ["iTransformerPredictor", "PatchTSTPredictor", "TimesMixerPredictor"]
        
        for model_type in model_types:
            mock_model = Mock()
            mock_model.__class__.__name__ = model_type
            mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 32, 32))
            mock_model.predict = Mock(return_value=np.array([0.6 + np.random.random() * 0.3]))
            models[model_type] = mock_model
        
        feature_names = [f"BTC_price_t_{i:02d}" for i in range(32)]
        feature_data = np.random.randn(32)
        
        # Should handle multiple transformer models
        explanations = {}
        for model_type, model in models.items():
            explanation = asyncio.run(manager.explain_trading_decision(
                decision_id=f"ensemble_test_{model_type}",
                model=model,
                feature_data=feature_data,
                feature_names=feature_names,
                decision_type="buy",
                symbol="BTC",
                model_type="transformer",
                explainer_type="attention",
                metadata={"model_variant": model_type}
            ))
            explanations[model_type] = explanation
        
        # All explanations should be successful
        assert len(explanations) == len(model_types)
        
        for model_type, explanation in explanations.items():
            assert isinstance(explanation, TradingExplanation)
            assert explanation.explanation_data.explanation_type == "attention"
            assert explanation.metadata["model_variant"] == model_type
        
        # Should support explanation comparison
        recent_explanations = manager.get_recent_explanations(symbol="BTC", limit=10)
        assert len(recent_explanations) == len(model_types)
        
        # Should provide aggregated feature importance
        feature_importance_summary = manager.get_feature_importance_summary(symbol="BTC")
        assert len(feature_importance_summary) == len(feature_names)
        
        # Should handle caching
        cached_explanation = manager.get_explanation("ensemble_test_iTransformerPredictor")
        assert cached_explanation is not None
        assert cached_explanation.explanation_data.explanation_type == "attention"
    
    def test_regulatory_compliance_for_attention_explanations(self):
        """Test regulatory compliance features for attention-based explanations."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 48
        mock_attention = torch.randn(1, 8, seq_len, seq_len)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        mock_model.predict = Mock(return_value=np.array([0.8]))
        mock_model.__class__.__name__ = "iTransformerPredictor"
        
        feature_names = [f"BTC_market_indicator_{i}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should generate regulatory compliance report
        input_data = np.random.randn(seq_len)
        explanation = explainer.explain_instance(input_data)
        
        # Should include compliance metadata
        metadata = explanation.explanation_metadata
        assert "regulatory_compliance" in metadata
        
        compliance_info = metadata["regulatory_compliance"]
        assert "explainability_score" in compliance_info
        assert "model_transparency" in compliance_info
        assert "decision_auditability" in compliance_info
        assert "bias_analysis" in compliance_info
        
        # Explainability score should be high for attention-based methods
        explainability_score = compliance_info["explainability_score"]
        assert 0 <= explainability_score <= 1
        assert explainability_score > 0.7, "Attention-based explanations should have high explainability"
        
        # Should include human-readable summary
        assert "human_readable_summary" in compliance_info
        summary = compliance_info["human_readable_summary"]
        assert isinstance(summary, str)
        assert len(summary) > 50  # Should be a meaningful summary
        assert "attention" in summary.lower()
        
        # Should track decision factors
        assert "key_decision_factors" in compliance_info
        key_factors = compliance_info["key_decision_factors"]
        assert isinstance(key_factors, list)
        assert len(key_factors) > 0
        
        # Each factor should have required compliance fields
        for factor in key_factors:
            assert "factor_name" in factor
            assert "importance_score" in factor
            assert "confidence_level" in factor
            assert "time_relevance" in factor


class TestTransformerXAIPerformanceBenchmarks:
    """Performance benchmark tests for Transformer XAI integration."""
    
    def test_large_sequence_performance(self):
        """Test performance with large sequence lengths (up to 2048)."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        import time
        
        # Test with increasing sequence lengths
        sequence_lengths = [64, 128, 256, 512, 1024]
        performance_results = {}
        
        for seq_len in sequence_lengths:
            mock_model = Mock()
            mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 8, seq_len, seq_len))
            mock_model.predict = Mock(return_value=np.array([0.7]))
            
            feature_names = [f"feature_{i}" for i in range(seq_len)]
            explainer = AttentionExplainer(mock_model, feature_names)
            
            # Measure explanation time
            input_data = np.random.randn(seq_len)
            
            start_time = time.time()
            explanation = explainer.explain_instance(input_data)
            explanation_time = (time.time() - start_time) * 1000
            
            performance_results[seq_len] = explanation_time
            
            # Should maintain reasonable performance even for large sequences
            max_time = 500 + (seq_len / 256) * 500  # Allow scaling with sequence length
            assert explanation_time < max_time, f"Seq len {seq_len}: {explanation_time}ms > {max_time}ms"
        
        # Performance should scale sub-quadratically (better than O(n²))
        # This assumes Flash Attention or similar optimizations
        scaling_factor = performance_results[1024] / performance_results[256]
        expected_quadratic_scaling = (1024 / 256) ** 2  # 16x for quadratic
        
        assert scaling_factor < expected_quadratic_scaling * 0.7, \
            f"Performance scaling {scaling_factor} should be better than quadratic {expected_quadratic_scaling}"
    
    def test_multi_asset_scalability(self):
        """Test scalability with multiple assets and complex portfolios."""
        try:
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("CrossAttentionAnalyzer not implemented yet")
        
        import time
        
        # Test with increasing number of assets
        asset_configs = [
            (['BTC', 'ETH'], 32),
            (['BTC', 'ETH', 'SOL'], 32),
            (['BTC', 'ETH', 'SOL', 'ADA', 'DOT'], 24),
            (['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE'], 16)
        ]
        
        for assets, seq_len in asset_configs:
            total_features = len(assets) * seq_len
            
            mock_model = Mock()
            mock_model.get_cross_attention_weights = Mock(
                return_value=torch.randn(1, 8, total_features, total_features)
            )
            
            feature_names = []
            for asset in assets:
                for t in range(seq_len):
                    feature_names.append(f"{asset}_price_t_{t}")
            
            analyzer = CrossAttentionAnalyzer(
                mock_model, 
                feature_names, 
                {'asset_list': assets}
            )
            
            # Should handle large portfolios efficiently
            input_data = np.random.randn(total_features)
            
            start_time = time.time()
            explanation = analyzer.explain_instance(input_data)
            analysis_time = (time.time() - start_time) * 1000
            
            # Should maintain reasonable performance for portfolio analysis
            max_time = 1000 + len(assets) * 100  # Allow scaling with number of assets
            assert analysis_time < max_time, \
                f"Portfolio analysis with {len(assets)} assets took {analysis_time}ms > {max_time}ms"
            
            # Should successfully analyze all cross-asset relationships
            metadata = explanation.explanation_metadata
            assert "cross_asset_correlations" in metadata
            
            correlation_matrix = metadata["cross_asset_correlations"]["correlation_matrix"]
            assert correlation_matrix.shape == (len(assets), len(assets))
    
    def test_concurrent_explanation_performance(self):
        """Test performance with concurrent explanation requests."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        import asyncio
        import time
        
        # Simulate concurrent trading decisions requiring explanations
        num_concurrent_requests = 10
        seq_len = 64
        
        async def generate_explanation(request_id):
            mock_model = Mock()
            mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, seq_len, seq_len))
            mock_model.predict = Mock(return_value=np.array([0.6 + np.random.random() * 0.3]))
            
            feature_names = [f"market_data_{i}" for i in range(seq_len)]
            explainer = AttentionExplainer(mock_model, feature_names)
            
            input_data = np.random.randn(seq_len)
            
            # Add small delay to simulate real model inference
            await asyncio.sleep(0.01)
            
            explanation = explainer.explain_instance(input_data)
            return request_id, explanation
        
        # Run concurrent explanations
        start_time = time.time()
        
        tasks = [generate_explanation(i) for i in range(num_concurrent_requests)]
        results = asyncio.run(asyncio.gather(*tasks))
        
        total_time = (time.time() - start_time) * 1000
        avg_time_per_request = total_time / num_concurrent_requests
        
        # Should handle concurrent requests efficiently
        assert len(results) == num_concurrent_requests
        assert avg_time_per_request < 200, \
            f"Average time per concurrent request {avg_time_per_request}ms should be <200ms"
        
        # All explanations should be successful
        for request_id, explanation in results:
            assert isinstance(explanation, ExplanationData)
            assert explanation.explanation_type == "attention"


class TestTransformerXAIErrorHandlingIntegration:
    """Test error handling and robustness of integrated Transformer XAI system."""
    
    def test_graceful_degradation_on_attention_extraction_failure(self):
        """Test graceful degradation when attention extraction fails."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock model that fails attention extraction
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(side_effect=RuntimeError("Attention extraction failed"))
        mock_model.predict = Mock(return_value=np.array([0.7]))
        
        feature_names = [f"feature_{i}" for i in range(32)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should handle gracefully and provide fallback explanation
        input_data = np.random.randn(32)
        
        # Should not raise exception but provide degraded explanation
        explanation = explainer.explain_instance(input_data)
        
        assert isinstance(explanation, ExplanationData)
        # Should indicate degraded explanation
        assert "attention_extraction_failed" in explanation.explanation_metadata
        assert explanation.explanation_metadata["attention_extraction_failed"] is True
        
        # Should still provide some form of feature importance
        assert len(explanation.feature_importance) == 32
    
    def test_handling_corrupted_attention_weights(self):
        """Test handling of corrupted or invalid attention weights."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 32
        
        # Test various corruption scenarios
        corrupted_attentions = [
            torch.full((1, 4, seq_len, seq_len), float('inf')),  # Infinite values
            torch.full((1, 4, seq_len, seq_len), float('nan')),  # NaN values
            torch.zeros(1, 4, seq_len, seq_len),  # All zeros
            torch.randn(1, 4, seq_len, seq_len) * 1e10,  # Extremely large values
        ]
        
        for corrupted_attention in corrupted_attentions:
            mock_model = Mock()
            mock_model.get_attention_weights = Mock(return_value=corrupted_attention)
            mock_model.predict = Mock(return_value=np.array([0.5]))
            
            feature_names = [f"feature_{i}" for i in range(seq_len)]
            explainer = AttentionExplainer(mock_model, feature_names)
            
            # Should handle corrupted attention gracefully
            input_data = np.random.randn(seq_len)
            explanation = explainer.explain_instance(input_data)
            
            assert isinstance(explanation, ExplanationData)
            
            # Should not contain invalid values in final importance scores
            for importance in explanation.feature_importance.values():
                assert np.isfinite(importance), f"Feature importance should be finite, got {importance}"
                assert not np.isnan(importance), f"Feature importance should not be NaN"
            
            # Should indicate attention cleaning in metadata
            assert "attention_weights_cleaned" in explanation.explanation_metadata
    
    def test_model_compatibility_validation(self):
        """Test validation of model compatibility with different XAI components."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
            from src.xai.transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
            from src.xai.transformers.cross_attention_analyzer import CrossAttentionAnalyzer
        except ImportError:
            pytest.skip("Transformer XAI components not implemented yet")
        
        # Test incompatible models
        incompatible_models = [
            Mock(),  # No attention methods
            Mock(spec=['predict']),  # Only predict method
            Mock(spec=['get_attention_weights', 'predict']),  # Missing required methods for specific analyzers
        ]
        
        feature_names = [f"feature_{i}" for i in range(16)]
        
        for model in incompatible_models:
            # AttentionExplainer should validate model compatibility
            try:
                explainer = AttentionExplainer(model, feature_names)
                is_valid, error = explainer.validate_input(model, feature_names)
                
                if not is_valid:
                    # Should provide clear error message
                    assert isinstance(error, str)
                    assert len(error) > 10
                    assert "attention" in error.lower()
                
            except (ValueError, AttributeError) as e:
                # Should raise informative error
                assert "attention" in str(e).lower() or "method" in str(e).lower()
        
        # Test compatible model
        compatible_model = Mock()
        compatible_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 16, 16))
        compatible_model.predict = Mock(return_value=np.array([0.6]))
        
        explainer = AttentionExplainer(compatible_model, feature_names)
        is_valid, error = explainer.validate_input(compatible_model, feature_names)
        
        assert is_valid
        assert error is None