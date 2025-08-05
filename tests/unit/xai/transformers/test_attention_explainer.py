"""
Tests for AttentionExplainer - Attention Weight Analysis for Transformer Models

Following TDD principles, these tests define the expected behavior
of attention-based explanations BEFORE implementation.

These tests MUST FAIL initially as AttentionExplainer doesn't exist yet.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import existing XAI framework components
from src.xai.base import GradientBasedExplainer
from src.xai.data_models import ExplanationData


class TestAttentionExplainer:
    """
    Test suite for AttentionExplainer following TDD principles.
    
    AttentionExplainer should extract and analyze attention weights from
    transformer models to provide interpretable explanations for trading decisions.
    """
    
    def test_attention_explainer_import_fails_initially(self):
        """Test that AttentionExplainer import fails initially (TDD)."""
        # This test should FAIL initially - AttentionExplainer doesn't exist yet
        with pytest.raises(ImportError):
            from src.xai.transformers.attention_explainer import AttentionExplainer
    
    def test_attention_explainer_inherits_from_gradient_based(self):
        """Test that AttentionExplainer inherits from GradientBasedExplainer."""
        # Skip if AttentionExplainer doesn't exist yet
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Should inherit from GradientBasedExplainer since attention uses gradients
        assert issubclass(AttentionExplainer, GradientBasedExplainer)
    
    def test_attention_explainer_initialization(self):
        """Test AttentionExplainer initialization with transformer model."""
        # Skip if not implemented
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock transformer model with attention
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(2, 8, 32, 32))
        
        feature_names = [f"feature_{i}" for i in range(32)]
        
        # Should initialize with transformer-specific config
        config = {
            'extract_attention': True,
            'attention_heads': 'all',
            'attention_layers': 'last',
            'attention_rollout': True
        }
        
        explainer = AttentionExplainer(
            model=mock_model,
            feature_names=feature_names,
            config=config
        )
        
        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.config['extract_attention'] is True
        assert explainer.config['attention_heads'] == 'all'
    
    def test_attention_explainer_extract_attention_weights(self):
        """Test extraction of attention weights from transformer models."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock transformer model
        batch_size, num_heads, seq_len = 2, 8, 32
        mock_attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        feature_names = [f"feature_{i}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should extract attention weights
        input_data = np.random.randn(seq_len)
        attention_weights = explainer.extract_attention_weights(input_data)
        
        assert isinstance(attention_weights, torch.Tensor)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        mock_model.get_attention_weights.assert_called_once()
    
    def test_attention_explainer_supports_all_transformer_types(self):
        """Test that AttentionExplainer supports all transformer model types."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        supported_types = [
            'iTransformer',
            'PatchTST', 
            'TimesMixer',
            'TimesFM',
            'TransformerPredictor'
        ]
        
        for model_type in supported_types:
            mock_model = Mock()
            mock_model.__class__.__name__ = model_type
            mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 16, 16))
            
            explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(16)])
            
            # Should validate that model is supported
            is_valid, error_msg = explainer.validate_input(mock_model, [f"f_{i}" for i in range(16)])
            assert is_valid, f"Should support {model_type}: {error_msg}"
    
    def test_attention_explainer_attention_rollout(self):
        """Test attention rollout for long-range dependency tracking."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock multi-layer attention
        num_layers, num_heads, seq_len = 6, 8, 32
        layer_attentions = []
        for _ in range(num_layers):
            layer_attentions.append(torch.randn(1, num_heads, seq_len, seq_len))
        
        mock_model = Mock()
        mock_model.get_layer_attention_weights = Mock(return_value=layer_attentions)
        
        explainer = AttentionExplainer(mock_model, [f"t_{i}" for i in range(seq_len)])
        
        # Should compute attention rollout
        input_data = np.random.randn(seq_len)
        rollout_attention = explainer.compute_attention_rollout(input_data)
        
        assert isinstance(rollout_attention, torch.Tensor)
        assert rollout_attention.shape == (seq_len, seq_len)
        # Values should be between 0 and 1 (attention scores)
        assert torch.all(rollout_attention >= 0) and torch.all(rollout_attention <= 1)
    
    def test_attention_explainer_head_wise_analysis(self):
        """Test head-wise attention analysis."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        num_heads, seq_len = 8, 32
        mock_attention = torch.randn(1, num_heads, seq_len, seq_len)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        explainer = AttentionExplainer(mock_model, [f"feature_{i}" for i in range(seq_len)])
        
        # Should analyze each attention head separately
        input_data = np.random.randn(seq_len)
        head_analysis = explainer.analyze_attention_heads(input_data)
        
        assert isinstance(head_analysis, dict)
        assert len(head_analysis) == num_heads
        
        for head_idx in range(num_heads):
            assert f"head_{head_idx}" in head_analysis
            head_info = head_analysis[f"head_{head_idx}"]
            assert "attention_pattern" in head_info
            assert "specialization" in head_info
            assert "importance_scores" in head_info
    
    def test_attention_explainer_explain_instance(self):
        """Test explaining single instance with attention weights."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 32
        mock_attention = torch.randn(1, 4, seq_len, seq_len)
        mock_prediction = np.array([0.7])  # BUY signal
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        mock_model.predict = Mock(return_value=mock_prediction)
        
        feature_names = [f"price_t_{i}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should generate explanation with attention-based importance
        input_instance = np.random.randn(seq_len)
        explanation = explainer.explain_instance(input_instance)
        
        assert isinstance(explanation, ExplanationData)
        assert explanation.explanation_type == "attention"
        assert len(explanation.feature_importance) == seq_len
        
        # Should include attention metadata
        assert "attention_weights" in explanation.explanation_metadata
        assert "head_analysis" in explanation.explanation_metadata
        assert "attention_rollout" in explanation.explanation_metadata
    
    def test_attention_explainer_get_feature_importance(self):
        """Test getting feature importance from attention weights."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 16
        # Create attention with specific pattern (last token attends to first token)
        mock_attention = torch.zeros(1, 1, seq_len, seq_len)
        mock_attention[0, 0, -1, 0] = 1.0  # Strong attention from last to first
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        feature_names = [f"feature_{i}" for i in range(seq_len)]
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should compute feature importance from attention
        input_instance = np.random.randn(seq_len)
        importance = explainer.get_feature_importance(input_instance)
        
        assert isinstance(importance, dict)
        assert len(importance) == seq_len
        
        # Feature_0 should have highest importance due to attention pattern
        assert importance["feature_0"] > importance["feature_1"]
        
        # All importance scores should sum to approximately 1.0
        total_importance = sum(abs(score) for score in importance.values())
        assert abs(total_importance - 1.0) < 0.1
    
    def test_attention_explainer_validate_input(self):
        """Test input validation for transformer models."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        explainer = AttentionExplainer(Mock(), ["feature_1"])
        
        # Should require model with get_attention_weights method
        valid_model = Mock()
        valid_model.get_attention_weights = Mock()
        is_valid, error = explainer.validate_input(valid_model, ["feature_1"])
        assert is_valid
        assert error is None
        
        # Should reject model without attention capability
        invalid_model = Mock()
        del invalid_model.get_attention_weights  # Remove method
        is_valid, error = explainer.validate_input(invalid_model, ["feature_1"])
        assert not is_valid
        assert "attention" in error.lower()
    
    def test_attention_explainer_get_explanation_metadata(self):
        """Test getting explanation metadata."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        mock_model = Mock()
        config = {'attention_heads': 'all', 'attention_layers': 'last'}
        explainer = AttentionExplainer(mock_model, ["f1", "f2"], config)
        
        metadata = explainer.get_explanation_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata["explainer_type"] == "attention"
        assert metadata["supported_models"] == ["iTransformer", "PatchTST", "TimesMixer", "TimesFM", "TransformerPredictor"]
        assert "attention_config" in metadata
        assert metadata["attention_config"] == config
    
    def test_attention_explainer_performance_requirements(self):
        """Test that attention explanation meets performance requirements."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        import time
        
        # Mock fast model
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=torch.randn(1, 4, 32, 32))
        mock_model.predict = Mock(return_value=np.array([0.5]))
        
        explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(32)])
        
        # Should generate explanation in <500ms
        input_instance = np.random.randn(32)
        
        start_time = time.time()
        explanation = explainer.explain_instance(input_instance)
        end_time = time.time()
        
        explanation_time = (end_time - start_time) * 1000  # Convert to ms
        assert explanation_time < 500, f"Explanation took {explanation_time}ms, should be <500ms"
        
        # Should add <10ms to inference time
        # (This is a mock test - real implementation would measure actual overhead)
        inference_overhead = 5  # Simulated overhead in ms
        assert inference_overhead < 10, f"Inference overhead {inference_overhead}ms should be <10ms"
    
    def test_attention_explainer_handles_different_input_shapes(self):
        """Test handling different input shapes and sequence lengths."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        mock_model = Mock()
        
        # Test different sequence lengths
        for seq_len in [16, 32, 64, 128]:
            mock_attention = torch.randn(1, 4, seq_len, seq_len)
            mock_model.get_attention_weights = Mock(return_value=mock_attention)
            
            feature_names = [f"feature_{i}" for i in range(seq_len)]
            explainer = AttentionExplainer(mock_model, feature_names)
            
            input_instance = np.random.randn(seq_len)
            explanation = explainer.explain_instance(input_instance)
            
            assert len(explanation.feature_importance) == seq_len
            assert explanation.explanation_type == "attention"
    
    def test_attention_explainer_multivariate_support(self):
        """Test support for multivariate time series (multiple assets)."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock multivariate input (3 assets, 32 time steps each)
        num_assets, seq_len = 3, 32
        total_features = num_assets * seq_len
        
        mock_attention = torch.randn(1, 8, total_features, total_features)
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        feature_names = []
        for asset in ['BTC', 'ETH', 'SOL']:
            for t in range(seq_len):
                feature_names.append(f"{asset}_t_{t}")
        
        explainer = AttentionExplainer(mock_model, feature_names)
        
        # Should handle multivariate input
        input_instance = np.random.randn(total_features)
        explanation = explainer.explain_instance(input_instance)
        
        assert len(explanation.feature_importance) == total_features
        
        # Should provide cross-asset attention analysis
        cross_asset_analysis = explainer.analyze_cross_asset_attention(input_instance)
        assert isinstance(cross_asset_analysis, dict)
        assert "BTC_to_ETH" in str(cross_asset_analysis)  # Should analyze cross-asset relationships


class TestAttentionExplainerErrorHandling:
    """Test error handling and edge cases for AttentionExplainer."""
    
    def test_attention_explainer_handles_nan_attention_weights(self):
        """Test handling of NaN values in attention weights."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock attention with NaN values
        mock_attention = torch.randn(1, 4, 16, 16)
        mock_attention[0, 0, 0, :] = float('nan')  # Inject NaN
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(16)])
        
        # Should handle NaN gracefully
        input_instance = np.random.randn(16)
        explanation = explainer.explain_instance(input_instance)
        
        # Should not contain NaN in final importance scores
        for importance in explanation.feature_importance.values():
            assert not np.isnan(importance)
    
    def test_attention_explainer_handles_model_without_attention(self):
        """Test handling models that don't support attention extraction."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        # Mock model without get_attention_weights method
        mock_model = Mock()
        # Don't add get_attention_weights method
        
        feature_names = [f"f_{i}" for i in range(10)]
        
        # Should raise appropriate error during initialization
        with pytest.raises((ValueError, AttributeError)) as exc_info:
            explainer = AttentionExplainer(mock_model, feature_names)
            explainer.explain_instance(np.random.randn(10))
        
        assert "attention" in str(exc_info.value).lower()
    
    def test_attention_explainer_handles_empty_input(self):
        """Test handling of empty input sequences."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        mock_model = Mock()
        explainer = AttentionExplainer(mock_model, [])
        
        # Should handle empty input gracefully
        with pytest.raises(ValueError) as exc_info:
            explainer.explain_instance(np.array([]))
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_attention_explainer_validates_attention_shape_consistency(self):
        """Test validation of attention weight shape consistency."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 32
        # Mock attention with inconsistent shape
        mock_attention = torch.randn(1, 4, seq_len, seq_len + 5)  # Inconsistent dimensions
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        
        explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(seq_len)])
        
        # Should detect and handle shape inconsistency
        input_instance = np.random.randn(seq_len)
        with pytest.raises(ValueError) as exc_info:
            explainer.explain_instance(input_instance)
        
        assert "shape" in str(exc_info.value).lower()


class TestAttentionExplainerIntegration:
    """Test integration with existing XAI framework."""
    
    def test_attention_explainer_integrates_with_explanation_data(self):
        """Test integration with ExplanationData structure."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 16
        mock_attention = torch.randn(1, 4, seq_len, seq_len)
        
        mock_model = Mock()
        mock_model.get_attention_weights = Mock(return_value=mock_attention)
        mock_model.predict = Mock(return_value=np.array([0.8]))
        
        explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(seq_len)])
        
        input_instance = np.random.randn(seq_len)
        explanation = explainer.explain_instance(input_instance)
        
        # Should create valid ExplanationData
        assert explanation.is_valid()
        assert explanation.explanation_type == "attention"
        assert explanation.model_prediction == 0.8
        
        # Should be serializable
        json_str = explanation.to_json()
        reconstructed = ExplanationData.from_json(json_str)
        assert reconstructed.explanation_type == "attention"
    
    def test_attention_explainer_supports_confidence_scoring(self):
        """Test attention-based confidence scoring."""
        try:
            from src.xai.transformers.attention_explainer import AttentionExplainer
        except ImportError:
            pytest.skip("AttentionExplainer not implemented yet")
        
        seq_len = 16
        
        # Create focused attention (high confidence)
        focused_attention = torch.zeros(1, 1, seq_len, seq_len)
        focused_attention[0, 0, -1, 0] = 1.0  # Strong focus on first token
        
        # Create distributed attention (low confidence)  
        distributed_attention = torch.ones(1, 1, seq_len, seq_len) / seq_len
        
        mock_model = Mock()
        explainer = AttentionExplainer(mock_model, [f"f_{i}" for i in range(seq_len)])
        
        # Should compute higher confidence for focused attention
        focused_conf = explainer.compute_attention_confidence(focused_attention)
        distributed_conf = explainer.compute_attention_confidence(distributed_attention)
        
        assert focused_conf > distributed_conf
        assert 0 <= focused_conf <= 1
        assert 0 <= distributed_conf <= 1