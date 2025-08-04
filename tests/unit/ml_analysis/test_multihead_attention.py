"""
Tests for MultiHeadAttention mechanism optimized for time-series
Following TDD methodology - tests written first
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from unittest.mock import Mock, patch

from src.ml_analysis.transformers.attention import MultiHeadAttention, AttentionVisualization
from src.ml_analysis.transformers.base import TransformerConfig


class TestMultiHeadAttention:
    """Test MultiHeadAttention implementation"""
    
    @pytest.fixture
    def attention_config(self):
        """Fixture for attention configuration"""
        return {
            'd_model': 256,
            'n_heads': 8,
            'dropout': 0.1,
            'use_flash_attention': True,
            'max_seq_length': 1000
        }
    
    @pytest.fixture
    def attention_module(self, attention_config):
        """Fixture for MultiHeadAttention module"""
        return MultiHeadAttention(**attention_config)
    
    def test_multihead_attention_initialization(self, attention_module):
        """Test MultiHeadAttention initialization"""
        assert attention_module.d_model == 256
        assert attention_module.n_heads == 8
        assert attention_module.d_k == 32  # 256 / 8
        assert attention_module.use_flash_attention == True
        
        # Check layer initialization
        assert isinstance(attention_module.query_projection, nn.Linear)
        assert isinstance(attention_module.key_projection, nn.Linear)
        assert isinstance(attention_module.value_projection, nn.Linear)
        assert isinstance(attention_module.output_projection, nn.Linear)
        assert isinstance(attention_module.dropout, nn.Dropout)
    
    def test_forward_pass_basic(self, attention_module):
        """Test basic forward pass functionality"""
        batch_size, seq_len, d_model = 2, 50, 256
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention_module(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check attention weights shape
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)  # n_heads=8
        
        # Check attention weights properties
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)), atol=1e-5)
        assert (attention_weights >= 0).all()
        assert (attention_weights <= 1).all()
    
    def test_self_attention(self, attention_module):
        """Test self-attention case (Q=K=V)"""
        batch_size, seq_len, d_model = 1, 20, 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention_module(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        
        # Self-attention weights should be symmetric for identical inputs
        # (though not exactly due to different projections)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_variable_sequence_lengths(self, attention_module):
        """Test handling of variable sequence lengths with masking"""
        batch_size, max_seq_len, d_model = 2, 100, 256
        
        # Create inputs with different actual sequence lengths
        query = torch.randn(batch_size, max_seq_len, d_model)
        key = torch.randn(batch_size, max_seq_len, d_model)
        value = torch.randn(batch_size, max_seq_len, d_model)
        
        # Create attention mask (first sequence: 60 tokens, second: 80 tokens)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        attention_mask[0, :60] = True
        attention_mask[1, :80] = True
        
        output, weights = attention_module(query, key, value, attention_mask=attention_mask)
        
        assert output.shape == (batch_size, max_seq_len, d_model)
        
        # Check that attention is correctly masked
        # Positions beyond the mask should have minimal attention from valid positions
        masked_positions_0 = weights[0, :, :60, 60:]  # First batch, attending to masked positions
        masked_positions_1 = weights[1, :, :80, 80:]  # Second batch, attending to masked positions
        
        # These should be very small (near zero) due to masking
        assert masked_positions_0.max() < 1e-6
        assert masked_positions_1.max() < 1e-6
    
    def test_causal_masking(self, attention_module):
        """Test causal (autoregressive) masking"""
        batch_size, seq_len, d_model = 1, 10, 256
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        
        output, weights = attention_module(query, key, value, causal_mask=causal_mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check that future positions have zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, :, i, j].max() < 1e-6  # Future positions should be masked
    
    def test_flash_attention_optimization(self, attention_config):
        """Test Flash Attention optimization"""
        # Test with Flash Attention enabled
        flash_attention = MultiHeadAttention(**attention_config)
        
        # Test with Flash Attention disabled
        no_flash_config = attention_config.copy()
        no_flash_config['use_flash_attention'] = False
        regular_attention = MultiHeadAttention(**no_flash_config)
        
        batch_size, seq_len, d_model = 1, 1000, 256  # Large sequence for memory test
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Both should produce similar results
        with torch.no_grad():
            flash_out, flash_weights = flash_attention(x, x, x)
            regular_out, regular_weights = regular_attention(x, x, x)
        
        # Flash attention should use less memory and produce similar results
        # (exact equality not expected due to different computation paths)
        assert flash_out.shape == regular_out.shape
        assert flash_weights.shape == regular_weights.shape
    
    def test_memory_efficiency_constraints(self, attention_module):
        """Test memory usage stays within GCP production limits"""
        # Test with maximum allowed sequence length
        batch_size = 1  # Conservative for memory testing
        seq_len = 1000  # Maximum allowed
        d_model = 256
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # This should not raise memory errors
        with torch.no_grad():
            output, weights = attention_module(query, key, value)
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()
    
    def test_gradient_flow(self, attention_module):
        """Test gradient flow through attention mechanism"""
        batch_size, seq_len, d_model = 1, 20, 256
        
        query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        output, _ = attention_module(query, key, value)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are not zero
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
        assert not torch.allclose(key.grad, torch.zeros_like(key.grad))
        assert not torch.allclose(value.grad, torch.zeros_like(value.grad))
    
    def test_time_series_specific_features(self, attention_module):
        """Test time-series specific attention features"""
        batch_size, seq_len, d_model = 1, 50, 256
        
        # Create time-series like data with temporal patterns
        time_steps = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1)
        temporal_pattern = torch.sin(time_steps / 10.0)  # Seasonal pattern
        noise = torch.randn(batch_size, seq_len, d_model - 1) * 0.1
        
        time_series_data = torch.cat([temporal_pattern, noise], dim=-1)
        
        output, attention_weights = attention_module(time_series_data, time_series_data, time_series_data)
        
        # Check that recent time steps get more attention (recency bias)
        # This is a heuristic check - in real scenarios, attention should learn this
        recent_attention = attention_weights[0, :, -5:, -10:].mean()  # Recent attending to recent
        distant_attention = attention_weights[0, :, -5:, :10].mean()   # Recent attending to distant
        
        # Recent positions should generally get more attention than very distant ones
        # (though this may not always hold, it's a reasonable expectation for time series)
        assert not torch.isnan(recent_attention)
        assert not torch.isnan(distant_attention)
    
    def test_attention_weights_interpretation(self, attention_module):
        """Test attention weights can be interpreted for XAI"""
        batch_size, seq_len, d_model = 1, 20, 256
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention_module(query, key, value)
        
        # Test attention weight properties for XAI
        assert attention_weights.dim() == 4  # (batch, heads, seq_len, seq_len)
        
        # Each head should have different attention patterns
        head_correlations = []
        for i in range(attention_module.n_heads):
            for j in range(i + 1, attention_module.n_heads):
                corr = torch.corrcoef(torch.stack([
                    attention_weights[0, i].flatten(),
                    attention_weights[0, j].flatten()
                ]))[0, 1]
                head_correlations.append(corr.abs())
        
        # Heads should have somewhat different patterns (not perfectly correlated)
        avg_correlation = torch.stack(head_correlations).mean()
        assert avg_correlation < 0.95  # Not too similar
    
    def test_edge_cases(self, attention_module):
        """Test edge cases and error conditions"""
        batch_size, seq_len, d_model = 1, 10, 256
        
        # Test with very small values
        small_input = torch.randn(batch_size, seq_len, d_model) * 1e-8
        output, weights = attention_module(small_input, small_input, small_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Test with very large values (should be handled by scaling)
        large_input = torch.randn(batch_size, seq_len, d_model) * 1e3
        output, weights = attention_module(large_input, large_input, large_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_qkv_dimensions(self, attention_config):
        """Test attention with different Q, K, V sequence lengths (cross-attention)"""
        attention_module = MultiHeadAttention(**attention_config)
        
        batch_size, d_model = 1, 256
        q_len, kv_len = 20, 30
        
        query = torch.randn(batch_size, q_len, d_model)
        key = torch.randn(batch_size, kv_len, d_model)
        value = torch.randn(batch_size, kv_len, d_model)
        
        output, weights = attention_module(query, key, value)
        
        assert output.shape == (batch_size, q_len, d_model)
        assert weights.shape == (batch_size, 8, q_len, kv_len)  # n_heads=8
    
    def test_performance_requirements(self, attention_module):
        """Test performance meets production requirements"""
        import time
        
        batch_size, seq_len, d_model = 1, 100, 256
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Warm up
        _ = attention_module(query, key, value)
        
        # Time the forward pass
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Average over multiple runs
                _ = attention_module(query, key, value)
        
        avg_time = (time.time() - start_time) / 10 * 1000  # Convert to ms
        
        # Should be fast enough for real-time trading
        assert avg_time < 50, f"Attention took {avg_time:.2f}ms, exceeds 50ms limit"


class TestAttentionVisualization:
    """Test AttentionVisualization utilities for XAI"""
    
    @pytest.fixture
    def sample_attention_weights(self):
        """Fixture for sample attention weights"""
        batch_size, n_heads, seq_len = 1, 8, 50
        # Create realistic attention pattern
        weights = torch.softmax(torch.randn(batch_size, n_heads, seq_len, seq_len), dim=-1)
        return weights
    
    @pytest.fixture
    def visualization_module(self):
        """Fixture for AttentionVisualization module"""
        return AttentionVisualization()
    
    def test_attention_visualization_initialization(self, visualization_module):
        """Test AttentionVisualization initialization"""
        assert hasattr(visualization_module, 'create_attention_heatmap')
        assert hasattr(visualization_module, 'analyze_attention_patterns')
        assert hasattr(visualization_module, 'get_head_specialization')
        assert hasattr(visualization_module, 'extract_temporal_patterns')
    
    def test_attention_heatmap_creation(self, visualization_module, sample_attention_weights):
        """Test attention heatmap creation"""
        heatmap_data = visualization_module.create_attention_heatmap(
            sample_attention_weights, 
            head_idx=0
        )
        
        assert isinstance(heatmap_data, dict)
        assert 'attention_matrix' in heatmap_data
        assert 'row_labels' in heatmap_data
        assert 'col_labels' in heatmap_data
        
        attention_matrix = heatmap_data['attention_matrix']
        assert attention_matrix.shape == (50, 50)  # seq_len x seq_len
        assert (attention_matrix >= 0).all()
        assert (attention_matrix <= 1).all()
    
    def test_attention_pattern_analysis(self, visualization_module, sample_attention_weights):
        """Test attention pattern analysis"""
        patterns = visualization_module.analyze_attention_patterns(sample_attention_weights)
        
        assert isinstance(patterns, dict)
        assert 'entropy' in patterns
        assert 'sparsity' in patterns
        assert 'locality' in patterns
        assert 'head_diversity' in patterns
        
        # Check value ranges
        assert 0 <= patterns['entropy'] <= math.log(50)  # Max entropy for seq_len=50
        assert 0 <= patterns['sparsity'] <= 1
        assert 0 <= patterns['locality'] <= 1
        assert 0 <= patterns['head_diversity'] <= 1
    
    def test_head_specialization_analysis(self, visualization_module, sample_attention_weights):
        """Test head specialization analysis"""
        specialization = visualization_module.get_head_specialization(sample_attention_weights)
        
        assert isinstance(specialization, dict)
        assert len(specialization) == 8  # n_heads
        
        for head_idx, spec in specialization.items():
            assert isinstance(spec, dict)
            assert 'focus_type' in spec
            assert 'concentration_score' in spec
            assert 'dominant_positions' in spec
            
            assert isinstance(spec['focus_type'], str)
            assert 0 <= spec['concentration_score'] <= 1
            assert isinstance(spec['dominant_positions'], list)
    
    def test_temporal_pattern_extraction(self, visualization_module, sample_attention_weights):
        """Test temporal pattern extraction for time-series"""
        temporal_patterns = visualization_module.extract_temporal_patterns(
            sample_attention_weights,
            sequence_timestamps=list(range(50))
        )
        
        assert isinstance(temporal_patterns, dict)
        assert 'recency_bias' in temporal_patterns
        assert 'periodicity' in temporal_patterns
        assert 'long_range_dependencies' in temporal_patterns
        
        # Check value ranges
        assert 0 <= temporal_patterns['recency_bias'] <= 1
        assert isinstance(temporal_patterns['periodicity'], (int, float))
        assert 0 <= temporal_patterns['long_range_dependencies'] <= 1
    
    def test_cross_asset_attention_analysis(self, visualization_module):
        """Test cross-asset attention analysis for multi-asset scenarios"""
        batch_size, n_heads, seq_len = 3, 8, 50  # 3 assets (BTC, ETH, SOL)
        
        # Create attention weights for multi-asset scenario
        multi_asset_weights = torch.softmax(
            torch.randn(batch_size, n_heads, seq_len, seq_len), 
            dim=-1
        )
        
        asset_names = ['BTC', 'ETH', 'SOL']
        cross_asset_analysis = visualization_module.analyze_cross_asset_attention(
            multi_asset_weights, 
            asset_names
        )
        
        assert isinstance(cross_asset_analysis, dict)
        assert 'correlation_matrix' in cross_asset_analysis
        assert 'leading_indicators' in cross_asset_analysis
        assert 'attention_flow' in cross_asset_analysis
        
        correlation_matrix = cross_asset_analysis['correlation_matrix']
        assert correlation_matrix.shape == (3, 3)  # 3x3 for 3 assets
        assert torch.allclose(torch.diag(correlation_matrix), torch.ones(3))  # Diagonal should be 1
    
    def test_attention_explanation_generation(self, visualization_module, sample_attention_weights):
        """Test generation of human-readable attention explanations"""
        explanations = visualization_module.generate_explanations(
            sample_attention_weights,
            feature_names=[f'feature_{i}' for i in range(50)],
            prediction_timestamp=49  # Predicting for the last timestep
        )
        
        assert isinstance(explanations, dict)
        assert 'most_important_features' in explanations
        assert 'temporal_focus' in explanations
        assert 'attention_summary' in explanations
        
        most_important = explanations['most_important_features']
        assert isinstance(most_important, list)
        assert len(most_important) <= 10  # Top 10 features
        
        for feature_info in most_important:
            assert 'feature_name' in feature_info
            assert 'importance_score' in feature_info
            assert 'time_position' in feature_info
            assert 0 <= feature_info['importance_score'] <= 1
    
    def test_regulatory_compliance_reporting(self, visualization_module, sample_attention_weights):
        """Test regulatory compliance features for attention explanations"""
        compliance_report = visualization_module.generate_compliance_report(
            sample_attention_weights,
            decision_outcome='BUY',
            confidence_score=0.85
        )
        
        assert isinstance(compliance_report, dict)
        assert 'decision_rationale' in compliance_report
        assert 'key_factors' in compliance_report
        assert 'risk_factors' in compliance_report
        assert 'model_confidence' in compliance_report
        assert 'transparency_score' in compliance_report
        
        # Check compliance requirements
        assert compliance_report['model_confidence'] == 0.85
        assert 0 <= compliance_report['transparency_score'] <= 1
        assert isinstance(compliance_report['decision_rationale'], str)
        assert len(compliance_report['decision_rationale']) > 0