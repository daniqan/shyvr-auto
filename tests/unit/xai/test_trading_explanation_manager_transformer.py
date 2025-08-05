"""
Comprehensive failing tests for Phase 2.3.4 - Transformer-specific TradingExplanationManager enhancements.

This module contains test-driven development (TDD) tests for transformer-specific explanation
generation in the TradingExplanationManager class. These tests are designed to FAIL initially
as the enhanced implementation does not exist yet.

Key Test Categories:
1. Transformer-specific explanation generation in TradingExplanationManager
2. Attention-based trading narratives
3. Integration with InterpretableTradingSignalGenerator
4. Performance requirements (<500ms for full explanations)
5. Multi-asset explanation generation
6. Explanation caching and optimization
7. Transformer model type detection and handling
8. Attention pattern analysis integration
9. Real-time explanation generation for trading
10. Error handling and fallback mechanisms

CRITICAL: All tests written using TDD methodology - they MUST fail initially!
"""

import pytest
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
import json

# Import existing classes
from src.xai.trading_integration import TradingExplanationManager, TradingExplanation
from src.xai.data_models import ExplanationData

# Import the classes we'll be testing (these don't exist yet - tests will fail!)
try:
    from src.xai.transformers.interpretable_trading_signals import (
        InterpretableTradingSignalGenerator,
        TradingSignalExplanation,
        MarketEventAttribution,
        TemporalContribution,
        AttentionBasedFeatureImportance
    )
except ImportError:
    # Expected to fail - these classes don't exist yet
    InterpretableTradingSignalGenerator = None
    TradingSignalExplanation = None
    MarketEventAttribution = None
    TemporalContribution = None
    AttentionBasedFeatureImportance = None


class TestTradingExplanationManagerTransformerIntegration:
    """
    Test suite for transformer-specific enhancements to TradingExplanationManager.
    
    These tests define the expected behavior for integrating attention-based
    explanations into the existing trading explanation system.
    """
    
    @pytest.fixture
    def mock_transformer_model(self):
        """Create a mock transformer model with attention capabilities."""
        model = Mock()
        model.__class__.__name__ = "iTransformerPredictor"
        model.model_type = "itransformer"
        
        # Mock attention weights [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.rand(1, 8, 96, 96)
        model.get_attention_weights.return_value = attention_weights
        model.get_layer_attention_weights.return_value = [attention_weights]
        
        # Mock predictions
        model.predict.return_value = np.array([0.75])
        
        # Mock attention-specific methods
        model.forward_with_attention = Mock(return_value=(torch.tensor([0.75]), attention_weights))
        model.predict_with_attention = Mock(return_value=(np.array([0.75]), attention_weights))
        
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Create comprehensive feature names for testing."""
        return [
            'price_return_1h', 'price_return_4h', 'price_return_24h',
            'volume_sma_12', 'volume_ema_24', 'volatility_garch','rsi_14', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'funding_rate', 'basis_spread', 'cross_exchange_arb',
            'market_regime_bull', 'market_regime_bear', 'market_regime_sideways',
            'time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin',
            'correlation_btc_eth', 'correlation_btc_sol', 'stress_indicator'
        ]
    
    @pytest.fixture
    def sample_market_events(self):
        """Create sample market events for testing."""
        base_time = datetime.now()
        return [
            {
                'timestamp': base_time - timedelta(hours=2),
                'event': 'Fed_Rate_Decision',
                'impact': 0.8,
                'event_type': 'monetary_policy'
            },
            {
                'timestamp': base_time - timedelta(hours=6),
                'event': 'BTC_ETF_Approval',
                'impact': 0.9,
                'event_type': 'regulatory'
            },
            {
                'timestamp': base_time - timedelta(hours=12),
                'event': 'Whale_Transaction',
                'impact': 0.6,
                'event_type': 'market_structure'
            }
        ]
    
    @pytest.fixture
    def enhanced_manager(self, mock_transformer_model):
        """Create enhanced TradingExplanationManager with transformer support."""
        manager = TradingExplanationManager(
            cache_size=100,
            explanation_timeout=0.5
        )
        # This enhancement should be added to the manager
        return manager
    
    def test_transformer_model_detection(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test that TradingExplanationManager correctly identifies transformer models."""
        # This test will FAIL - enhanced transformer detection doesn't exist
        
        # Test transformer model detection
        is_transformer = enhanced_manager._is_transformer_model(mock_transformer_model)
        assert is_transformer == True
        
        # Test model type identification
        model_type = enhanced_manager._get_transformer_model_type(mock_transformer_model)
        assert model_type == "itransformer"
        
        # Test attention capability detection
        has_attention = enhanced_manager._supports_attention_analysis(mock_transformer_model)
        assert has_attention == True
    
    def test_attention_based_explanation_generation(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test generation of attention-based explanations for transformer models."""
        # This test will FAIL - method doesn't exist yet
        
        input_data = np.random.randn(len(feature_names))
        
        # Test attention-based explanation generation
        explanation_future = enhanced_manager.generate_attention_based_explanation(
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            include_temporal_analysis=True,
            include_market_events=True
        )
        
        # Should return a coroutine/future
        assert asyncio.iscoroutine(explanation_future) or hasattr(explanation_future, '__await__')
        
        # Execute the async method
        explanation = asyncio.run(explanation_future) if asyncio.iscoroutine(explanation_future) else explanation_future
        
        # Expected enhanced explanation structure
        assert isinstance(explanation, dict)
        assert 'attention_analysis' in explanation
        assert 'temporal_contributions' in explanation
        assert 'trading_narrative' in explanation
        assert 'confidence_breakdown' in explanation
        
        # Attention analysis should include key components
        attention_analysis = explanation['attention_analysis']
        assert 'attention_weights' in attention_analysis
        assert 'head_analysis' in attention_analysis
        assert 'attention_entropy' in attention_analysis
        assert 'attention_sparsity' in attention_analysis
    
    @pytest.mark.asyncio
    async def test_enhanced_explain_trading_decision_with_attention(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test enhanced explain_trading_decision method with attention analysis."""
        # This test will FAIL - enhanced method doesn't exist
        
        decision_id = "test_decision_001"
        input_data = np.random.randn(len(feature_names))
        
        # Test enhanced trading decision explanation
        explanation = await enhanced_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            enable_interpretable_signals=True,  # New parameter
            include_market_context=True         # New parameter
        )
        
        # Should return enhanced TradingExplanation
        assert isinstance(explanation, TradingExplanation)
        assert explanation.attention_data is not None
        
        # Should include interpretable trading signals
        assert 'interpretable_signals' in explanation.metadata
        interpretable_signals = explanation.metadata['interpretable_signals']
        assert isinstance(interpretable_signals, dict)
        assert 'signal_interpretation' in interpretable_signals
        assert 'feature_importance' in interpretable_signals
        assert 'temporal_analysis' in interpretable_signals
    
    def test_attention_based_trading_narratives(self, enhanced_manager, mock_transformer_model, feature_names, sample_market_events):
        """Test generation of attention-based trading narratives."""
        # This test will FAIL - method doesn't exist yet
        
        input_data = np.random.randn(len(feature_names))
        
        # Test narrative generation with attention patterns
        narrative = enhanced_manager.generate_attention_based_narrative(
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            market_events=sample_market_events,
            prediction_confidence=0.85
        )
        
        # Expected narrative structure
        assert isinstance(narrative, dict)
        assert 'executive_summary' in narrative
        assert 'attention_insights' in narrative
        assert 'temporal_analysis' in narrative
        assert 'market_event_analysis' in narrative
        assert 'risk_assessment' in narrative
        assert 'recommendation' in narrative
        
        # Attention insights should be detailed
        attention_insights = narrative['attention_insights']
        assert isinstance(attention_insights, str)
        assert len(attention_insights) >= 200
        assert 'attention' in attention_insights.lower()
        
        # Should mention specific features and time periods
        assert any(feature.lower() in narrative['attention_insights'].lower() 
                  for feature in feature_names[:5])
        
        # Market event analysis should reference events
        market_analysis = narrative['market_event_analysis']
        assert any(event['event'].lower() in market_analysis.lower() 
                  for event in sample_market_events)
    
    def test_performance_requirement_transformer_explanations(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test that transformer explanations meet <500ms performance requirement."""
        # This test will FAIL - enhanced performance optimization doesn't exist
        
        input_data = np.random.randn(len(feature_names))
        decision_id = "perf_test_001"
        
        # Measure explanation generation time
        start_time = time.time()
        
        explanation_task = enhanced_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            enable_interpretable_signals=True,
            include_full_narrative=True
        )
        
        # Execute async operation
        explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
        
        end_time = time.time()
        explanation_time = end_time - start_time
        
        # Must meet <500ms requirement
        assert explanation_time < 0.5
        assert explanation is not None
        
        # Should still provide comprehensive analysis
        assert explanation.attention_data is not None
        assert 'interpretable_signals' in explanation.metadata
        assert len(explanation.explanation_data.feature_importance) == len(feature_names)
    
    def test_multi_asset_transformer_explanations(self, enhanced_manager, feature_names):
        """Test explanation generation for multi-asset transformer scenarios."""
        # This test will FAIL - multi-asset support doesn't exist yet
        
        # Mock multi-asset transformer
        multi_asset_model = Mock()
        multi_asset_model.__class__.__name__ = "iTransformerPredictor"
        multi_asset_model.model_type = "itransformer"
        multi_asset_model.get_attention_weights.return_value = torch.rand(1, 8, 96, 96)
        multi_asset_model.predict.return_value = np.array([0.8, 0.3, 0.6])  # BTC, ETH, SOL
        
        # Multi-asset feature names
        assets = ['BTC', 'ETH', 'SOL']
        multi_feature_names = []
        for asset in assets:
            for feature in feature_names:
                multi_feature_names.append(f"{asset}_{feature}")
        
        input_data = np.random.randn(len(multi_feature_names))
        
        # Test multi-asset explanation generation
        explanations = enhanced_manager.generate_multi_asset_explanations(
            model=multi_asset_model,
            feature_data=input_data,
            feature_names=multi_feature_names,
            asset_names=assets,
            decision_id="multi_asset_001"
        )
        
        # Should generate explanation for each asset
        assert isinstance(explanations, dict)
        assert len(explanations) == 3
        
        for asset in assets:
            assert asset in explanations
            explanation = explanations[asset]
            assert isinstance(explanation, TradingExplanation)
            assert explanation.symbol == asset
            assert explanation.attention_data is not None
            
            # Should include cross-asset influences
            assert 'cross_asset_influences' in explanation.metadata
            cross_influences = explanation.metadata['cross_asset_influences']
            assert isinstance(cross_influences, dict)
            other_assets = [a for a in assets if a != asset]
            for other_asset in other_assets:
                assert other_asset in cross_influences
    
    def test_explanation_caching_with_attention_patterns(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test caching mechanism for attention-based explanations."""
        # This test will FAIL - enhanced caching doesn't exist yet
        
        input_data = np.random.randn(len(feature_names))
        decision_id = "cache_test_001"
        
        # First explanation (should cache attention patterns)
        start_time1 = time.time()
        explanation1_task = enhanced_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            cache_attention_patterns=True  # New parameter
        )
        explanation1 = asyncio.run(explanation1_task) if asyncio.iscoroutine(explanation1_task) else explanation1_task
        time1 = time.time() - start_time1
        
        # Second explanation with same data (should use cached attention)
        start_time2 = time.time()
        explanation2_task = enhanced_manager.explain_trading_decision(
            decision_id=decision_id + "_cached",
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            cache_attention_patterns=True
        )
        explanation2 = asyncio.run(explanation2_task) if asyncio.iscoroutine(explanation2_task) else explanation2_task
        time2 = time.time() - start_time2
        
        # Cached explanation should be faster
        assert time2 < time1 * 0.7  # At least 30% faster
        
        # Should have consistent attention patterns
        assert explanation1.attention_data is not None
        assert explanation2.attention_data is not None
        
        # Check cache statistics
        cache_stats = enhanced_manager.get_attention_cache_statistics()
        assert 'attention_cache_hits' in cache_stats
        assert 'attention_cache_misses' in cache_stats
        assert cache_stats['attention_cache_hits'] >= 1
    
    def test_transformer_model_type_specific_handling(self, enhanced_manager, feature_names):
        """Test handling of different transformer model types."""
        # This test will FAIL - model-specific handling doesn't exist yet
        
        transformer_models = {
            'itransformer': {
                'class_name': 'iTransformerPredictor',
                'model_type': 'itransformer',
                'num_heads': 8,
                'special_features': ['multivariate_attention', 'variate_embeddings']
            },
            'patchtst': {
                'class_name': 'PatchTSTPredictor',
                'model_type': 'patchtst',
                'num_heads': 8,
                'special_features': ['patch_attention', 'channel_independence']
            },
            'timesmixer': {
                'class_name': 'TimesMixerPredictor',
                'model_type': 'timesmixer',
                'num_heads': 4,
                'special_features': ['decomposable_mixing', 'multi_scale_patterns']
            },
            'timesfm': {
                'class_name': 'TimesFMWrapper',
                'model_type': 'timesfm',
                'num_heads': 12,
                'special_features': ['foundation_model', 'zero_shot_prediction']
            }
        }
        
        for model_name, config in transformer_models.items():
            # Mock model
            model = Mock()
            model.__class__.__name__ = config['class_name']
            model.model_type = config['model_type']
            model.get_attention_weights.return_value = torch.rand(1, config['num_heads'], 96, 96)
            model.predict.return_value = np.array([0.75])
            
            input_data = np.random.randn(len(feature_names))
            
            # Test model-specific explanation generation
            explanation_task = enhanced_manager.explain_trading_decision(
                decision_id=f"test_{model_name}",
                model=model,
                feature_data=input_data,
                feature_names=feature_names,
                decision_type="BUY",
                symbol="BTC",
                enable_attention_analysis=True
            )
            
            explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
            
            # Should handle model-specific features
            assert explanation.model_type == config['model_type']
            assert explanation.attention_data is not None
            
            # Should include model-specific analysis
            model_specific_analysis = explanation.metadata.get('model_specific_analysis', {})
            assert model_name in model_specific_analysis or config['model_type'] in str(model_specific_analysis)
            
            # Should handle special features
            for special_feature in config['special_features']:
                feature_handled = any(
                    special_feature in str(value).lower() 
                    for value in explanation.metadata.values() 
                    if isinstance(value, (str, dict))
                )
                # Not strict requirement but good to have
    
    def test_real_time_explanation_streaming(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test real-time explanation generation for live trading."""
        # This test will FAIL - streaming capability doesn't exist yet
        
        # Mock streaming data
        streaming_data = []
        for i in range(5):
            streaming_data.append({
                'timestamp': datetime.now() - timedelta(seconds=i*10),
                'feature_data': np.random.randn(len(feature_names)),
                'decision_type': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'symbol': 'BTC'
            })
        
        # Test streaming explanation generation
        explanation_stream = enhanced_manager.generate_streaming_explanations(
            model=mock_transformer_model,
            streaming_data=streaming_data,
            feature_names=feature_names,
            max_latency_ms=100  # Real-time requirement
        )
        
        # Should return async generator or similar
        assert hasattr(explanation_stream, '__aiter__') or hasattr(explanation_stream, '__iter__')
        
        # Process stream
        explanations = []
        if hasattr(explanation_stream, '__aiter__'):
            async def collect_explanations():
                async for explanation in explanation_stream:
                    explanations.append(explanation)
                    if len(explanations) >= 3:  # Collect first 3
                        break
            asyncio.run(collect_explanations())
        else:
            for explanation in explanation_stream:
                explanations.append(explanation)
                if len(explanations) >= 3:
                    break
        
        # Should generate explanations in real-time
        assert len(explanations) >= 3
        for explanation in explanations:
            assert isinstance(explanation, TradingExplanation)
            assert explanation.attention_data is not None
            
        # Should maintain temporal ordering
        timestamps = [exp.timestamp for exp in explanations]
        assert timestamps == sorted(timestamps, reverse=True)  # Most recent first
    
    def test_attention_pattern_anomaly_detection(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test detection of anomalous attention patterns."""
        # This test will FAIL - anomaly detection doesn't exist yet
        
        # Normal attention pattern
        normal_attention = torch.rand(1, 8, 96, 96)
        mock_transformer_model.get_attention_weights.return_value = normal_attention
        
        input_data = np.random.randn(len(feature_names))
        
        # Test normal pattern detection
        normal_explanation_task = enhanced_manager.explain_trading_decision(
            decision_id="normal_pattern",
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_anomaly_detection=True  # New parameter
        )
        
        normal_explanation = asyncio.run(normal_explanation_task) if asyncio.iscoroutine(normal_explanation_task) else normal_explanation_task
        
        # Should not detect anomaly
        assert 'attention_anomaly' in normal_explanation.metadata
        assert normal_explanation.metadata['attention_anomaly']['is_anomalous'] == False
        
        # Create anomalous attention pattern (all attention on one position)
        anomalous_attention = torch.zeros(1, 8, 96, 96)
        anomalous_attention[0, :, 0, 0] = 1.0  # All attention on first position
        mock_transformer_model.get_attention_weights.return_value = anomalous_attention
        
        # Test anomalous pattern detection
        anomalous_explanation_task = enhanced_manager.explain_trading_decision(
            decision_id="anomalous_pattern",
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_anomaly_detection=True
        )
        
        anomalous_explanation = asyncio.run(anomalous_explanation_task) if asyncio.iscoroutine(anomalous_explanation_task) else anomalous_explanation_task
        
        # Should detect anomaly
        assert anomalous_explanation.metadata['attention_anomaly']['is_anomalous'] == True
        assert 'anomaly_score' in anomalous_explanation.metadata['attention_anomaly']
        assert 'anomaly_description' in anomalous_explanation.metadata['attention_anomaly']
        
        # Confidence should be reduced for anomalous patterns
        assert anomalous_explanation.confidence < normal_explanation.confidence
    
    def test_regulatory_compliance_attention_explanations(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test regulatory compliance features for attention-based explanations."""
        # This test will FAIL - regulatory compliance features don't exist yet
        
        input_data = np.random.randn(len(feature_names))
        
        # Test regulatory-compliant explanation generation
        explanation_task = enhanced_manager.explain_trading_decision(
            decision_id="regulatory_test",
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            regulatory_compliance='MiFID_II'  # New parameter
        )
        
        explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
        
        # Should include regulatory compliance metadata
        assert 'regulatory_compliance' in explanation.metadata
        compliance_data = explanation.metadata['regulatory_compliance']
        
        # MiFID II specific requirements
        assert compliance_data['framework'] == 'MiFID_II'
        assert 'decision_rationale' in compliance_data
        assert 'risk_disclosure' in compliance_data
        assert 'model_transparency' in compliance_data
        assert 'audit_trail' in compliance_data
        
        # Decision rationale should be human-readable
        rationale = compliance_data['decision_rationale']
        assert isinstance(rationale, str)
        assert len(rationale) >= 200  # Comprehensive explanation
        
        # Should include attention-specific disclosures
        model_transparency = compliance_data['model_transparency']
        assert 'attention_mechanism' in model_transparency
        assert 'feature_importance_method' in model_transparency
        assert 'uncertainty_estimation' in model_transparency
    
    def test_explanation_export_and_serialization(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test export and serialization of attention-based explanations."""
        # This test will FAIL - export functionality doesn't exist yet
        
        input_data = np.random.randn(len(feature_names))
        
        explanation_task = enhanced_manager.explain_trading_decision(
            decision_id="export_test",
            model=mock_transformer_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True
        )
        
        explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
        
        # Test JSON export
        json_export = enhanced_manager.export_explanation_to_json(explanation)
        assert isinstance(json_export, str)
        
        # Should be valid JSON
        parsed_json = json.loads(json_export)
        assert 'decision_id' in parsed_json
        assert 'attention_data' in parsed_json
        assert 'explanation_data' in parsed_json
        
        # Test structured export for reporting
        structured_export = enhanced_manager.export_explanation_for_reporting(
            explanation, 
            format='structured_dict'
        )
        assert isinstance(structured_export, dict)
        assert 'executive_summary' in structured_export
        assert 'technical_analysis' in structured_export
        assert 'attention_insights' in structured_export
        assert 'risk_assessment' in structured_export
        
        # Test PDF-ready export
        pdf_export = enhanced_manager.export_explanation_for_reporting(
            explanation,
            format='pdf_ready'
        )
        assert isinstance(pdf_export, dict)
        assert 'title' in pdf_export
        assert 'sections' in pdf_export
        assert len(pdf_export['sections']) >= 4  # Multiple sections for comprehensive report
    
    def test_error_handling_transformer_failures(self, enhanced_manager, feature_names):
        """Test error handling when transformer models fail."""
        # This test will FAIL - enhanced error handling doesn't exist yet
        
        # Mock failing transformer model
        failing_model = Mock()
        failing_model.__class__.__name__ = "iTransformerPredictor"
        failing_model.model_type = "itransformer"
        failing_model.get_attention_weights.side_effect = RuntimeError("Model inference failed")
        failing_model.predict.side_effect = RuntimeError("Prediction failed")
        
        input_data = np.random.randn(len(feature_names))
        
        # Test graceful failure handling
        explanation_task = enhanced_manager.explain_trading_decision(
            decision_id="failure_test",
            model=failing_model,
            feature_data=input_data,
            feature_names=feature_names,
            decision_type="BUY",
            symbol="BTC",
            enable_attention_analysis=True,
            fallback_to_basic_explanation=True  # New parameter
        )
        
        explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
        
        # Should still return explanation (fallback)
        assert explanation is not None
        assert isinstance(explanation, TradingExplanation)
        
        # Should indicate failure in metadata
        assert 'transformer_failure' in explanation.metadata
        assert explanation.metadata['transformer_failure']['failed'] == True
        assert 'error_message' in explanation.metadata['transformer_failure']
        
        # Should provide basic explanation
        assert explanation.explanation_data is not None
        assert len(explanation.explanation_data.feature_importance) > 0
        
        # Confidence should be reduced due to failure
        assert explanation.confidence <= 0.5
        
        # Should log error for monitoring
        assert 'fallback_explanation_used' in explanation.metadata
    
    def test_attention_drift_detection_integration(self, enhanced_manager, mock_transformer_model, feature_names):
        """Test integration with attention drift detection systems."""
        # This test will FAIL - drift detection integration doesn't exist yet
        
        input_data = np.random.randn(len(feature_names))
        
        # Generate multiple explanations to establish baseline
        explanations = []
        for i in range(5):
            explanation_task = enhanced_manager.explain_trading_decision(
                decision_id=f"drift_test_{i}",
                model=mock_transformer_model,
                feature_data=input_data + np.random.normal(0, 0.1, input_data.shape),  # Slight variation
                feature_names=feature_names,
                decision_type="BUY",
                symbol="BTC",
                enable_attention_analysis=True,
                enable_drift_detection=True  # New parameter
            )
            
            explanation = asyncio.run(explanation_task) if asyncio.iscoroutine(explanation_task) else explanation_task
            explanations.append(explanation)
        
        # Should track attention patterns over time
        for explanation in explanations:
            assert 'attention_drift_analysis' in explanation.metadata
            drift_analysis = explanation.metadata['attention_drift_analysis']
            assert 'drift_score' in drift_analysis
            assert 'baseline_deviation' in drift_analysis
            assert 'drift_status' in drift_analysis  # 'stable', 'minor_drift', 'significant_drift'
        
        # Should provide drift statistics
        drift_stats = enhanced_manager.get_attention_drift_statistics(
            symbol="BTC",
            time_window_hours=1
        )
        assert 'average_drift_score' in drift_stats
        assert 'max_drift_score' in drift_stats
        assert 'drift_trend' in drift_stats
        assert 'attention_stability' in drift_stats


class TestTransformerExplanationUtilities:
    """Test suite for transformer-specific explanation utilities."""
    
    def test_attention_pattern_visualization_data(self):
        """Test generation of attention pattern visualization data."""
        # This test will FAIL - visualization utilities don't exist yet
        
        manager = TradingExplanationManager()
        
        # Mock attention weights
        attention_weights = torch.rand(1, 8, 96, 96)
        feature_names = [f"feature_{i}" for i in range(21)]
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(96, 0, -1)]
        
        # Test visualization data generation
        viz_data = manager.generate_attention_visualization_data(
            attention_weights=attention_weights,
            feature_names=feature_names,
            timestamps=timestamps,
            top_k_features=10
        )
        
        # Expected visualization structure
        assert isinstance(viz_data, dict)
        assert 'attention_heatmap' in viz_data
        assert 'feature_importance_chart' in viz_data
        assert 'temporal_attention_chart' in viz_data
        assert 'head_analysis_chart' in viz_data
        
        # Attention heatmap should be properly formatted
        heatmap = viz_data['attention_heatmap']
        assert 'data' in heatmap
        assert 'labels' in heatmap
        assert len(heatmap['data']) == 96  # Time dimension
        assert len(heatmap['data'][0]) == 96  # Time dimension
        
        # Feature importance should include top features
        feature_chart = viz_data['feature_importance_chart']
        assert 'features' in feature_chart
        assert 'importance_scores' in feature_chart
        assert len(feature_chart['features']) <= 10  # Top k
    
    def test_attention_summary_statistics(self):
        """Test calculation of attention summary statistics."""
        # This test will FAIL - statistics utilities don't exist yet
        
        manager = TradingExplanationManager()
        
        # Mock attention weights for multiple decisions
        attention_data = []
        for i in range(10):
            attention_data.append({
                'decision_id': f"decision_{i}",
                'attention_weights': torch.rand(1, 8, 96, 96),
                'timestamp': datetime.now() - timedelta(minutes=i*5),
                'symbol': 'BTC'
            })
        
        # Test summary statistics calculation
        summary_stats = manager.calculate_attention_summary_statistics(
            attention_data=attention_data,
            time_window_hours=1
        )
        
        # Expected statistics
        assert isinstance(summary_stats, dict)
        assert 'average_attention_entropy' in summary_stats
        assert 'attention_sparsity_trend' in summary_stats
        assert 'head_specialization_stability' in summary_stats
        assert 'temporal_focus_patterns' in summary_stats
        assert 'attention_consistency_score' in summary_stats
        
        # Statistics should be within expected ranges
        assert 0 <= summary_stats['attention_sparsity_trend'] <= 1
        assert 0 <= summary_stats['attention_consistency_score'] <= 1
        assert summary_stats['average_attention_entropy'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])