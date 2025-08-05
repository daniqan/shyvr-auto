"""
Comprehensive failing tests for Phase 2.3.4 - Interpretable Trading Signals.

This module contains test-driven development (TDD) tests for the InterpretableTradingSignalGenerator
class and related components. These tests are designed to FAIL initially as the implementation
does not exist yet. The tests define the expected behavior and will guide implementation.

Key Test Categories:
1. InterpretableTradingSignalGenerator core functionality
2. Attention-based feature importance calculation
3. Temporal contribution analysis (which time periods drive decisions)
4. Market event attribution (link attention spikes to market events)
5. Trading signal interpretation (BUY/SELL/HOLD with rationale)
6. Confidence score generation based on attention patterns
7. Narrative generation for human-readable explanations
8. Integration with transformer models (iTransformer, PatchTST, TimesMixer, TimesFM)
9. Performance requirements (<500ms for full explanations)
10. Error handling and edge cases

CRITICAL: All tests written using TDD methodology - they MUST fail initially!
"""

import pytest
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

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

from src.xai.data_models import ExplanationData


class TestInterpretableTradingSignalGenerator:
    """
    Test suite for InterpretableTradingSignalGenerator class.
    
    These tests define the expected behavior for interpreting transformer-based
    trading decisions using attention patterns.
    """
    
    @pytest.fixture
    def mock_transformer_model(self):
        """Create a mock transformer model with attention capabilities."""
        model = Mock()
        model.__class__.__name__ = "iTransformerPredictor"
        model.model_type = "itransformer"
        
        # Mock attention weights [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.rand(1, 8, 96, 96)  # 8 heads, 96 timesteps
        model.get_attention_weights.return_value = attention_weights
        
        # Mock prediction
        model.predict.return_value = np.array([0.75])  # BUY signal
        
        # Mock attention patterns
        model.get_layer_attention_weights.return_value = [attention_weights]
        
        return model
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for testing."""
        timestamps = [
            datetime.now() - timedelta(hours=i) for i in range(96, 0, -1)
        ]
        
        return {
            'timestamps': timestamps,
            'prices': np.random.uniform(40000, 50000, 96),  # BTC price range
            'volumes': np.random.uniform(100, 1000, 96),
            'volatility': np.random.uniform(0.01, 0.05, 96),
            'market_events': [
                {'timestamp': timestamps[20], 'event': 'Fed_Rate_Decision', 'impact': 0.8},
                {'timestamp': timestamps[50], 'event': 'BTC_ETF_News', 'impact': 0.6},
                {'timestamp': timestamps[80], 'event': 'Whale_Transaction', 'impact': 0.4}
            ]
        }
    
    @pytest.fixture
    def feature_names(self):
        """Create comprehensive feature names for testing."""
        return [
            'price_return_1h', 'price_return_4h', 'price_return_24h',
            'volume_sma_12', 'volume_ema_24', 'volatility_garch',
            'rsi_14', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'funding_rate', 'basis_spread', 'cross_exchange_arb',
            'market_regime_bull', 'market_regime_bear', 'market_regime_sideways',
            'time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin',
            'correlation_btc_eth', 'correlation_btc_sol', 'stress_indicator'
        ]
    
    def test_interpretable_signal_generator_initialization(self, mock_transformer_model, feature_names):
        """Test InterpretableTradingSignalGenerator initialization."""
        # This test will FAIL initially - class doesn't exist
        with pytest.raises((ImportError, NameError, TypeError)):
            generator = InterpretableTradingSignalGenerator(
                model=mock_transformer_model,
                feature_names=feature_names,
                config={
                    'explanation_timeout': 0.5,  # 500ms requirement
                    'enable_market_events': True,
                    'enable_temporal_analysis': True,
                    'confidence_threshold': 0.7,
                    'narrative_generation': True
                }
            )
    
    def test_attention_based_feature_importance_calculation(self, mock_transformer_model, feature_names):
        """Test calculation of feature importance from attention patterns."""
        # This test will FAIL - method doesn't exist
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Sample input data
        input_data = np.random.randn(96, len(feature_names))
        
        # Test attention-based feature importance
        importance = generator.calculate_attention_based_importance(
            input_data=input_data,
            attention_weights=mock_transformer_model.get_attention_weights.return_value
        )
        
        # Expected behavior
        assert isinstance(importance, AttentionBasedFeatureImportance)
        assert len(importance.feature_scores) == len(feature_names)
        assert all(0 <= score <= 1 for score in importance.feature_scores.values())
        assert importance.total_importance == pytest.approx(1.0, rel=1e-3)
        
        # Most important features should be identified
        top_features = importance.get_top_features(n=5)
        assert len(top_features) == 5
        assert all(isinstance(f, str) for f in top_features)
    
    def test_temporal_contribution_analysis(self, mock_transformer_model, feature_names, sample_market_data):
        """Test analysis of which time periods drive trading decisions."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        input_data = np.random.randn(96, len(feature_names))
        timestamps = sample_market_data['timestamps']
        
        # Test temporal contribution analysis
        temporal_contrib = generator.analyze_temporal_contributions(
            input_data=input_data,
            timestamps=timestamps,
            attention_weights=mock_transformer_model.get_attention_weights.return_value
        )
        
        # Expected behavior
        assert isinstance(temporal_contrib, TemporalContribution)
        assert len(temporal_contrib.time_importance) == 96
        assert all(0 <= score <= 1 for score in temporal_contrib.time_importance)
        
        # Should identify critical time windows
        critical_windows = temporal_contrib.get_critical_time_windows(threshold=0.8)
        assert isinstance(critical_windows, list)
        assert all('start_time' in window and 'end_time' in window for window in critical_windows)
        
        # Should provide time-based explanations
        time_explanation = temporal_contrib.get_time_based_explanation()
        assert isinstance(time_explanation, str)
        assert len(time_explanation) > 50  # Meaningful explanation
    
    def test_market_event_attribution(self, mock_transformer_model, feature_names, sample_market_data):
        """Test linking attention spikes to specific market events."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names,
            config={'enable_market_events': True}
        )
        
        input_data = np.random.randn(96, len(feature_names))
        market_events = sample_market_data['market_events']
        timestamps = sample_market_data['timestamps']
        
        # Test market event attribution
        event_attribution = generator.attribute_market_events(
            input_data=input_data,
            timestamps=timestamps,
            market_events=market_events,
            attention_weights=mock_transformer_model.get_attention_weights.return_value
        )
        
        # Expected behavior
        assert isinstance(event_attribution, MarketEventAttribution)
        assert len(event_attribution.event_influences) == len(market_events)
        
        # Each event should have influence score
        for event_influence in event_attribution.event_influences:
            assert 'event' in event_influence
            assert 'influence_score' in event_influence
            assert 'attention_spike' in event_influence
            assert 0 <= event_influence['influence_score'] <= 1
        
        # Should identify most influential events
        top_events = event_attribution.get_most_influential_events(n=2)
        assert len(top_events) <= 2
        assert all('event' in event for event in top_events)
    
    def test_trading_signal_interpretation_buy_signal(self, mock_transformer_model, feature_names):
        """Test interpretation of BUY trading signals with rationale."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Mock strong BUY signal
        mock_transformer_model.predict.return_value = np.array([0.85])
        input_data = np.random.randn(96, len(feature_names))
        
        # Test signal interpretation
        signal_explanation = generator.interpret_trading_signal(
            input_data=input_data,
            prediction_value=0.85,
            decision_threshold_buy=0.7,
            decision_threshold_sell=0.3
        )
        
        # Expected behavior
        assert isinstance(signal_explanation, TradingSignalExplanation)
        assert signal_explanation.signal_type == "BUY"
        assert signal_explanation.confidence_score >= 0.7
        assert isinstance(signal_explanation.rationale, str)
        assert len(signal_explanation.rationale) > 100  # Detailed rationale
        
        # Should include key supporting factors
        assert 'supporting_factors' in signal_explanation.metadata
        assert len(signal_explanation.metadata['supporting_factors']) >= 3
        
        # Should include risk factors
        assert 'risk_factors' in signal_explanation.metadata
        assert isinstance(signal_explanation.metadata['risk_factors'], list)
    
    def test_trading_signal_interpretation_sell_signal(self, mock_transformer_model, feature_names):
        """Test interpretation of SELL trading signals with rationale."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Mock strong SELL signal
        mock_transformer_model.predict.return_value = np.array([0.15])
        input_data = np.random.randn(96, len(feature_names))
        
        signal_explanation = generator.interpret_trading_signal(
            input_data=input_data,
            prediction_value=0.15,
            decision_threshold_buy=0.7,
            decision_threshold_sell=0.3
        )
        
        assert signal_explanation.signal_type == "SELL"
        assert signal_explanation.confidence_score >= 0.7  # High confidence in sell
        assert "sell" in signal_explanation.rationale.lower() or "bearish" in signal_explanation.rationale.lower()
    
    def test_trading_signal_interpretation_hold_signal(self, mock_transformer_model, feature_names):
        """Test interpretation of HOLD trading signals with rationale."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Mock neutral HOLD signal
        mock_transformer_model.predict.return_value = np.array([0.5])
        input_data = np.random.randn(96, len(feature_names))
        
        signal_explanation = generator.interpret_trading_signal(
            input_data=input_data,
            prediction_value=0.5,
            decision_threshold_buy=0.7,
            decision_threshold_sell=0.3
        )
        
        assert signal_explanation.signal_type == "HOLD"
        assert "hold" in signal_explanation.rationale.lower() or "neutral" in signal_explanation.rationale.lower()
        assert 'uncertainty_factors' in signal_explanation.metadata
    
    def test_confidence_score_generation_high_attention_focus(self, mock_transformer_model, feature_names):
        """Test confidence score generation based on attention patterns - high focus case."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Create focused attention pattern (high confidence)
        focused_attention = torch.zeros(1, 8, 96, 96)
        focused_attention[0, :, :10, :10] = 0.8  # Focus on first 10 timesteps
        focused_attention[0, :, 10:, 10:] = 0.1  # Low attention elsewhere
        
        input_data = np.random.randn(96, len(feature_names))
        
        confidence = generator.calculate_attention_confidence(
            attention_weights=focused_attention,
            prediction_value=0.85,
            input_data=input_data
        )
        
        # Focused attention should yield high confidence
        assert confidence >= 0.8
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_confidence_score_generation_dispersed_attention(self, mock_transformer_model, feature_names):
        """Test confidence score generation based on attention patterns - dispersed case."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Create dispersed attention pattern (low confidence)
        dispersed_attention = torch.ones(1, 8, 96, 96) * 0.01  # Uniform low attention
        
        input_data = np.random.randn(96, len(feature_names))
        
        confidence = generator.calculate_attention_confidence(
            attention_weights=dispersed_attention,
            prediction_value=0.85,
            input_data=input_data
        )
        
        # Dispersed attention should yield lower confidence
        assert confidence <= 0.5
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_narrative_generation_comprehensive(self, mock_transformer_model, feature_names, sample_market_data):
        """Test generation of human-readable explanations."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names,
            config={'narrative_generation': True}
        )
        
        input_data = np.random.randn(96, len(feature_names))
        
        # Test comprehensive narrative generation
        narrative = generator.generate_trading_narrative(
            input_data=input_data,
            prediction_value=0.8,
            timestamps=sample_market_data['timestamps'],
            market_events=sample_market_data['market_events']
        )
        
        # Expected narrative structure
        assert isinstance(narrative, dict)
        assert 'executive_summary' in narrative
        assert 'detailed_analysis' in narrative
        assert 'risk_assessment' in narrative
        assert 'recommendation' in narrative
        
        # Executive summary should be concise
        assert 50 <= len(narrative['executive_summary']) <= 200
        
        # Detailed analysis should be comprehensive
        assert len(narrative['detailed_analysis']) >= 300
        
        # Should mention specific features and time periods
        detailed_text = narrative['detailed_analysis'].lower()
        assert any(feature.lower() in detailed_text for feature in feature_names[:5])
        
        # Risk assessment should be present
        assert len(narrative['risk_assessment']) >= 100
        assert ('risk' in narrative['risk_assessment'].lower() or 
                'caution' in narrative['risk_assessment'].lower())
    
    def test_integration_with_itransformer(self):
        """Test integration with iTransformer model."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock iTransformer model
        itransformer = Mock()
        itransformer.__class__.__name__ = "iTransformerPredictor"
        itransformer.model_type = "itransformer"
        itransformer.get_attention_weights.return_value = torch.rand(1, 8, 96, 96)
        itransformer.predict.return_value = np.array([0.75])
        
        feature_names = ['feature_' + str(i) for i in range(20)]
        
        generator = InterpretableTradingSignalGenerator(
            model=itransformer,
            feature_names=feature_names
        )
        
        # Should successfully create explanations for iTransformer
        assert generator.model_type == "itransformer"
        assert generator.supports_multivariate_attention == True
    
    def test_integration_with_patchtst(self):
        """Test integration with PatchTST model."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock PatchTST model
        patchtst = Mock()
        patchtst.__class__.__name__ = "PatchTSTPredictor"
        patchtst.model_type = "patchtst"
        patchtst.get_attention_weights.return_value = torch.rand(1, 8, 24, 24)  # Patch-based
        patchtst.predict.return_value = np.array([0.65])
        
        feature_names = ['feature_' + str(i) for i in range(20)]
        
        generator = InterpretableTradingSignalGenerator(
            model=patchtst,
            feature_names=feature_names
        )
        
        # Should handle patch-based attention
        assert generator.model_type == "patchtst"
        assert generator.uses_patch_attention == True
    
    def test_integration_with_timesmixer(self):
        """Test integration with TimesMixer model."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock TimesMixer model
        timesmixer = Mock()
        timesmixer.__class__.__name__ = "TimesMixerPredictor"
        timesmixer.model_type = "timesmixer"
        timesmixer.get_attention_weights.return_value = torch.rand(1, 4, 96, 96)  # Fewer heads
        timesmixer.predict.return_value = np.array([0.55])
        
        feature_names = ['feature_' + str(i) for i in range(20)]
        
        generator = InterpretableTradingSignalGenerator(
            model=timesmixer,
            feature_names=feature_names
        )
        
        # Should handle mixing-based attention
        assert generator.model_type == "timesmixer"
        assert generator.uses_decomposable_mixing == True
    
    def test_integration_with_timesfm(self):
        """Test integration with TimesFM foundation model."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock TimesFM model
        timesfm = Mock()
        timesfm.__class__.__name__ = "TimesFMWrapper"
        timesfm.model_type = "timesfm"
        timesfm.get_attention_weights.return_value = torch.rand(1, 12, 96, 96)  # More heads
        timesfm.predict.return_value = np.array([0.70])
        
        feature_names = ['feature_' + str(i) for i in range(20)]
        
        generator = InterpretableTradingSignalGenerator(
            model=timesfm,
            feature_names=feature_names
        )
        
        # Should handle foundation model attention
        assert generator.model_type == "timesfm"
        assert generator.is_foundation_model == True
    
    def test_performance_requirement_explanation_time(self, mock_transformer_model, feature_names):
        """Test that full explanations complete within 500ms requirement."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names,
            config={'explanation_timeout': 0.5}
        )
        
        input_data = np.random.randn(96, len(feature_names))
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(96, 0, -1)]
        
        # Measure explanation time
        start_time = time.time()
        
        explanation = generator.generate_full_explanation(
            input_data=input_data,
            timestamps=timestamps,
            include_narrative=True,
            include_market_events=True,
            include_temporal_analysis=True
        )
        
        end_time = time.time()
        explanation_time = end_time - start_time
        
        # Must complete within 500ms
        assert explanation_time < 0.5
        assert explanation is not None
        assert isinstance(explanation, dict)
        
        # Should include all components
        assert 'signal_interpretation' in explanation
        assert 'feature_importance' in explanation
        assert 'temporal_contributions' in explanation
        assert 'confidence_score' in explanation
        assert 'narrative' in explanation
    
    def test_multi_asset_explanation_generation(self, feature_names):
        """Test explanation generation for multi-asset scenarios."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock multi-asset transformer
        multi_asset_model = Mock()
        multi_asset_model.__class__.__name__ = "iTransformerPredictor" 
        multi_asset_model.model_type = "itransformer"
        multi_asset_model.get_attention_weights.return_value = torch.rand(1, 8, 96, 96)
        multi_asset_model.predict.return_value = np.array([0.8, 0.3, 0.6])  # BTC, ETH, SOL
        
        # Multi-asset feature names
        multi_feature_names = []
        assets = ['BTC', 'ETH', 'SOL']
        for asset in assets:
            for feature in feature_names:
                multi_feature_names.append(f"{asset}_{feature}")
        
        generator = InterpretableTradingSignalGenerator(
            model=multi_asset_model,
            feature_names=multi_feature_names,
            config={'multi_asset_mode': True}
        )
        
        input_data = np.random.randn(96, len(multi_feature_names))
        
        # Test multi-asset explanation
        explanations = generator.generate_multi_asset_explanations(
            input_data=input_data,
            asset_names=['BTC', 'ETH', 'SOL']
        )
        
        # Should have explanation for each asset
        assert len(explanations) == 3
        for i, asset in enumerate(['BTC', 'ETH', 'SOL']):
            assert asset in explanations
            assert 'signal_type' in explanations[asset]
            assert 'confidence_score' in explanations[asset]
            assert 'cross_asset_influences' in explanations[asset]
    
    def test_error_handling_invalid_model(self, feature_names):
        """Test error handling for invalid model types."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        # Mock invalid model (no attention capabilities)
        invalid_model = Mock()
        invalid_model.__class__.__name__ = "LinearRegression"
        del invalid_model.get_attention_weights  # Remove attention method
        
        # Should raise appropriate error
        with pytest.raises((ValueError, AttributeError)):
            generator = InterpretableTradingSignalGenerator(
                model=invalid_model,
                feature_names=feature_names
            )
    
    def test_error_handling_corrupted_attention_weights(self, mock_transformer_model, feature_names):
        """Test error handling for corrupted attention weights."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Test with NaN attention weights
        corrupted_attention = torch.full((1, 8, 96, 96), float('nan'))
        input_data = np.random.randn(96, len(feature_names))
        
        # Should handle corrupted data gracefully
        explanation = generator.generate_full_explanation(
            input_data=input_data,
            custom_attention_weights=corrupted_attention
        )
        
        # Should return fallback explanation
        assert explanation is not None
        assert 'error_handled' in explanation.get('metadata', {})
        assert explanation['confidence_score'] <= 0.3  # Low confidence for corrupted data
    
    def test_error_handling_mismatched_dimensions(self, mock_transformer_model, feature_names):
        """Test error handling for dimension mismatches."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names
        )
        
        # Wrong input data dimensions
        wrong_input_data = np.random.randn(48, len(feature_names))  # Wrong sequence length
        
        # Should handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            generator.generate_full_explanation(input_data=wrong_input_data)
    
    def test_caching_mechanism_for_repeated_explanations(self, mock_transformer_model, feature_names):
        """Test caching mechanism for performance optimization."""
        if InterpretableTradingSignalGenerator is None:
            pytest.skip("InterpretableTradingSignalGenerator not implemented yet")
        
        generator = InterpretableTradingSignalGenerator(
            model=mock_transformer_model,
            feature_names=feature_names,
            config={'enable_caching': True, 'cache_size': 100}
        )
        
        input_data = np.random.randn(96, len(feature_names))
        
        # First explanation (should cache)
        start_time1 = time.time()
        explanation1 = generator.generate_full_explanation(input_data=input_data)
        time1 = time.time() - start_time1
        
        # Second explanation with same data (should use cache)
        start_time2 = time.time()
        explanation2 = generator.generate_full_explanation(input_data=input_data)
        time2 = time.time() - start_time2
        
        # Cached explanation should be faster
        assert time2 < time1 * 0.5  # At least 50% faster
        assert explanation1['signal_interpretation']['signal_type'] == explanation2['signal_interpretation']['signal_type']
        
        # Should have cache statistics
        cache_stats = generator.get_cache_statistics()
        assert 'cache_hits' in cache_stats
        assert 'cache_misses' in cache_stats
        assert cache_stats['cache_hits'] >= 1


class TestTradingSignalExplanation:
    """Test suite for TradingSignalExplanation data structure."""
    
    def test_trading_signal_explanation_structure(self):
        """Test TradingSignalExplanation data structure."""
        if TradingSignalExplanation is None:
            pytest.skip("TradingSignalExplanation not implemented yet")
        
        explanation = TradingSignalExplanation(
            signal_type="BUY",
            confidence_score=0.85,
            rationale="Strong bullish momentum detected based on attention patterns",
            supporting_factors=["High volume", "Positive funding rates", "Bullish RSI"],
            risk_factors=["High volatility", "Macro uncertainty"],
            timestamp=datetime.now(),
            metadata={'model_type': 'itransformer'}
        )
        
        assert explanation.signal_type == "BUY"
        assert explanation.confidence_score == 0.85
        assert len(explanation.supporting_factors) == 3
        assert len(explanation.risk_factors) == 2
        assert isinstance(explanation.timestamp, datetime)


class TestMarketEventAttribution:
    """Test suite for MarketEventAttribution functionality."""
    
    def test_market_event_attribution_structure(self):
        """Test MarketEventAttribution data structure."""
        if MarketEventAttribution is None:
            pytest.skip("MarketEventAttribution not implemented yet")
        
        attribution = MarketEventAttribution(
            event_influences=[
                {
                    'event': 'Fed_Rate_Decision',
                    'influence_score': 0.8,
                    'attention_spike': True,
                    'time_impact': '2h_window'
                },
                {
                    'event': 'BTC_ETF_News',
                    'influence_score': 0.6,
                    'attention_spike': True,
                    'time_impact': '4h_window'
                }
            ],
            total_event_influence=0.7,
            background_influence=0.3
        )
        
        assert len(attribution.event_influences) == 2
        assert attribution.total_event_influence == 0.7
        assert attribution.background_influence == 0.3
        
        # Test most influential events method
        top_events = attribution.get_most_influential_events(n=1)
        assert len(top_events) == 1
        assert top_events[0]['event'] == 'Fed_Rate_Decision'


class TestTemporalContribution:
    """Test suite for TemporalContribution analysis."""
    
    def test_temporal_contribution_structure(self):
        """Test TemporalContribution data structure."""
        if TemporalContribution is None:
            pytest.skip("TemporalContribution not implemented yet")
        
        time_importance = np.random.rand(96)  # 96 timesteps
        
        contribution = TemporalContribution(
            time_importance=time_importance,
            critical_windows=[
                {'start_time': datetime.now() - timedelta(hours=4),
                 'end_time': datetime.now() - timedelta(hours=2),
                 'importance_score': 0.9}
            ],
            recency_bias=0.7,
            temporal_patterns=['morning_volatility', 'afternoon_consolidation']
        )
        
        assert len(contribution.time_importance) == 96
        assert contribution.recency_bias == 0.7
        assert len(contribution.critical_windows) == 1
        assert len(contribution.temporal_patterns) == 2


class TestAttentionBasedFeatureImportance:
    """Test suite for AttentionBasedFeatureImportance calculation."""
    
    def test_attention_based_feature_importance_structure(self):
        """Test AttentionBasedFeatureImportance data structure."""
        if AttentionBasedFeatureImportance is None:
            pytest.skip("AttentionBasedFeatureImportance not implemented yet")
        
        feature_scores = {
            'price_return_1h': 0.25,
            'volume_sma_12': 0.20,
            'rsi_14': 0.15,
            'funding_rate': 0.12,
            'volatility_garch': 0.10,
            'macd_signal': 0.08,
            'bollinger_upper': 0.06,
            'basis_spread': 0.04
        }
        
        importance = AttentionBasedFeatureImportance(
            feature_scores=feature_scores,
            attention_entropy=2.3,
            attention_sparsity=0.15,
            head_specialization=0.8
        )
        
        assert abs(sum(importance.feature_scores.values()) - 1.0) < 1e-6
        assert importance.attention_entropy == 2.3
        assert importance.attention_sparsity == 0.15
        
        # Test top features method
        top_features = importance.get_top_features(n=3)
        assert len(top_features) == 3
        assert top_features[0] == 'price_return_1h'  # Highest score
        assert top_features[1] == 'volume_sma_12'   # Second highest
        assert top_features[2] == 'rsi_14'          # Third highest


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])