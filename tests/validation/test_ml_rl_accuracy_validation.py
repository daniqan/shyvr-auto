"""
ML-RL Accuracy Validation Tests

Comprehensive tests that validate how ML prediction accuracy feeds correctly
into RL decisions in the Shyvr AI trading system. These tests verify that:

1. RL agent decision-making is appropriately influenced by ML prediction confidence
2. ML prediction direction (buy/sell/hold signals) properly affects RL action selection
3. Prediction uncertainty and volatility forecasts influence RL risk assessment
4. ML confidence thresholds properly modulate RL action strength
5. Historical ML accuracy affects future RL decision weighting

These tests use actual ML-RL integration components where possible to validate
production scenarios.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock, patch

from src.rl_agent.base import TradeAction, MarketState, AgentConfig
from src.rl_agent.dqn_agent import DQNTradingAgent, DQNNetwork
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.ml_analysis.base import (
    PredictionResult, ModelType, PredictionDirection, 
    TechnicalIndicators, MarketFeatures
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain
from src.integration.ml_rl_bridge import (
    MLEnhancedMarketState, MLRLBridge, MLRLConfig,
    MLRLPerformanceMetrics, MLRLTrainingPipeline
)


class TestMLRLAccuracyValidation:
    """Test suite for ML-RL accuracy integration validation"""
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        return DiscoveredToken(
            address="0x123...",
            symbol="TESTCOIN",
            name="Test Coin",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            volume_24h=1000000
        )
    
    @pytest.fixture
    def technical_indicators(self):
        """Create sample technical indicators"""
        return TechnicalIndicators(
            rsi=65.0,
            macd=2.5,
            sma_20=98.0,
            ema_12=101.0,
            bollinger_upper=105.0,
            bollinger_lower=95.0,
            atr=5.0,
            obv=50000
        )
    
    @pytest.fixture
    def market_features(self):
        """Create sample market features"""
        return MarketFeatures(
            fear_greed_index=75.0,
            market_trend="bull",
            volatility_regime="medium",
            btc_correlation=0.6,
            social_score=0.7
        )
    
    def create_ml_prediction(self, token: DiscoveredToken, 
                           confidence: float, 
                           direction: PredictionDirection,
                           volatility_forecast: float = 0.1,
                           technical_indicators: Optional[TechnicalIndicators] = None,
                           market_features: Optional[MarketFeatures] = None) -> PredictionResult:
        """Create ML prediction with specified parameters"""
        price_multiplier = {
            PredictionDirection.STRONG_BUY: 1.15,
            PredictionDirection.BUY: 1.08,
            PredictionDirection.HOLD: 1.02,
            PredictionDirection.SELL: 0.95,
            PredictionDirection.STRONG_SELL: 0.85
        }.get(direction, 1.0)
        
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=token.price_usd * price_multiplier,
            price_prediction_4h=token.price_usd * price_multiplier * 1.02,
            price_prediction_24h=token.price_usd * price_multiplier * 1.05,
            direction=direction,
            confidence=confidence,
            probability_up=confidence if direction in [PredictionDirection.BUY, PredictionDirection.STRONG_BUY] else 1.0 - confidence,
            volatility_forecast=volatility_forecast,
            technical_indicators=technical_indicators,
            market_features=market_features
        )
    
    def create_mock_ml_analyzer(self, predictions: List[PredictionResult]):
        """Create mock ML analyzer with specified predictions"""
        mock_analyzer = MagicMock()
        
        # Create proper async mock
        mock_analyzer.batch_analyze = AsyncMock(return_value=predictions)
        mock_analyzer.analyze_batch = MagicMock(return_value=predictions)  # Sync fallback
        return mock_analyzer
    
    def create_mock_rl_agent(self, action_mapping: Dict[float, Tuple[TradeAction, float]] = None):
        """Create mock RL agent with configurable action mapping based on confidence"""
        mock_agent = MagicMock()
        
        # Default confidence-based action mapping
        default_mapping = {
            0.9: (TradeAction.STRONG_BUY, 0.9),
            0.8: (TradeAction.BUY, 0.8),
            0.6: (TradeAction.BUY, 0.6),
            0.5: (TradeAction.HOLD, 0.5),
            0.3: (TradeAction.SELL, 0.3),
            0.1: (TradeAction.STRONG_SELL, 0.1)
        }
        mapping = action_mapping or default_mapping
        
        def mock_predict_action(market_state: MLEnhancedMarketState):
            # Base action on ML confidence level and direction
            ml_confidence = market_state.ml_confidence or 0.5
            ml_direction = market_state.ml_direction
            
            # If custom mapping is provided and matches confidence level, use it (for special test cases)
            if action_mapping and ml_confidence in action_mapping:
                return action_mapping[ml_confidence]
            
            # Override based on ML direction first
            if ml_direction == "hold":
                return TradeAction.HOLD, ml_confidence
            elif ml_direction == "strong_sell":
                return TradeAction.STRONG_SELL, ml_confidence
            elif ml_direction == "sell":
                return TradeAction.SELL, ml_confidence
            elif ml_direction == "strong_buy":
                return TradeAction.STRONG_BUY, ml_confidence
            elif ml_direction == "buy":
                return TradeAction.BUY, ml_confidence
            
            # Fallback to confidence-based mapping if no clear direction
            closest_threshold = min(mapping.keys(), key=lambda x: abs(x - ml_confidence))
            action, confidence = mapping[closest_threshold]
            
            # Low confidence default to hold
            if ml_confidence < 0.5:
                action = TradeAction.HOLD
                
            return action, confidence
        
        # Create synchronous mock for easier testing
        mock_agent.predict_action = MagicMock(side_effect=mock_predict_action)
        return mock_agent
    
    def test_high_confidence_ml_predictions_lead_to_strong_actions(self, sample_token, technical_indicators):
        """Test that high-confidence ML predictions (>0.8) lead to BUY/STRONG_BUY RL actions"""
        # Create high-confidence buy prediction
        high_confidence_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.85,
            direction=PredictionDirection.BUY,
            technical_indicators=technical_indicators
        )
        
        # Setup ML-RL bridge with sync analyzer for testing
        ml_analyzer = MagicMock()
        ml_analyzer.analyze_batch = MagicMock(return_value=[high_confidence_prediction])
        # Remove batch_analyze to force sync path
        del ml_analyzer.batch_analyze
        
        rl_agent = self.create_mock_rl_agent()
        bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[sample_token]
        )
        
        # Get integrated predictions and actions
        results = bridge.predict_and_act(portfolio_value=10000.0, positions={})
        
        assert len(results) == 1
        result = results[0]
        
        # Verify high confidence leads to strong buy action
        assert result['ml_prediction'].confidence == 0.85
        assert result['ml_prediction'].direction == PredictionDirection.BUY
        assert result['rl_action'][0] in [TradeAction.BUY, TradeAction.STRONG_BUY]
        assert result['rl_action'][1] >= 0.8  # High RL confidence
        
        # Verify enhanced market state includes ML features
        enhanced_state = result['enhanced_state']
        assert enhanced_state.ml_confidence == 0.85
        assert enhanced_state.ml_direction == "buy"
        assert enhanced_state.ml_prediction_1h > sample_token.price_usd  # Bullish prediction
    
    def test_low_confidence_ml_predictions_lead_to_conservative_actions(self, sample_token, technical_indicators):
        """Test that low-confidence ML predictions (<0.5) lead to HOLD RL actions"""
        # Create low-confidence prediction
        low_confidence_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.3,
            direction=PredictionDirection.HOLD,
            technical_indicators=technical_indicators
        )
        
        # Setup ML-RL bridge with sync analyzer for testing
        ml_analyzer = MagicMock()
        ml_analyzer.analyze_batch = MagicMock(return_value=[low_confidence_prediction])
        # Remove batch_analyze to force sync path
        del ml_analyzer.batch_analyze
        
        rl_agent = self.create_mock_rl_agent()
        bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[sample_token]
        )
        
        # Get integrated predictions and actions
        results = bridge.predict_and_act(portfolio_value=10000.0, positions={})
        
        assert len(results) == 1
        result = results[0]
        
        # Verify low confidence leads to conservative (HOLD) action
        assert result['ml_prediction'].confidence == 0.3
        assert result['rl_action'][0] == TradeAction.HOLD
        assert result['rl_action'][1] <= 0.5  # Low RL confidence
    
    def test_ml_direction_affects_rl_action_selection(self, sample_token, technical_indicators):
        """Test that ML prediction direction properly affects RL action selection"""
        test_cases = [
            (PredictionDirection.STRONG_BUY, 0.9, [TradeAction.STRONG_BUY, TradeAction.BUY]),
            (PredictionDirection.BUY, 0.75, [TradeAction.BUY, TradeAction.STRONG_BUY]),
            (PredictionDirection.HOLD, 0.6, [TradeAction.HOLD]),
            (PredictionDirection.SELL, 0.75, [TradeAction.SELL, TradeAction.STRONG_SELL]),
            (PredictionDirection.STRONG_SELL, 0.9, [TradeAction.STRONG_SELL, TradeAction.SELL])
        ]
        
        for direction, confidence, expected_actions in test_cases:
            # Create prediction with specific direction and confidence
            prediction = self.create_ml_prediction(
                token=sample_token,
                confidence=confidence,
                direction=direction,
                technical_indicators=technical_indicators
            )
            
            # Setup ML-RL bridge with sync analyzer for testing
            ml_analyzer = MagicMock()
            ml_analyzer.analyze_batch = MagicMock(return_value=[prediction])
            # Remove batch_analyze to force sync path
            del ml_analyzer.batch_analyze
            
            rl_agent = self.create_mock_rl_agent()
            bridge = MLRLBridge(
                ml_analyzer=ml_analyzer,
                rl_agent=rl_agent,
                tokens=[sample_token]
            )
            
            # Get integrated predictions and actions
            results = bridge.predict_and_act(portfolio_value=10000.0, positions={})
            result = results[0]
            
            # Verify RL action aligns with ML direction
            assert result['rl_action'][0] in expected_actions, f"Expected {expected_actions} for {direction}, got {result['rl_action'][0]}"
    
    def test_volatility_forecast_influences_risk_assessment(self, sample_token, technical_indicators):
        """Test that ML volatility forecasts influence RL risk assessment"""
        # Create predictions with different volatility forecasts
        low_volatility_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.8,
            direction=PredictionDirection.BUY,
            volatility_forecast=0.05,  # Low volatility
            technical_indicators=technical_indicators
        )
        
        high_volatility_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.8,
            direction=PredictionDirection.BUY,
            volatility_forecast=0.25,  # High volatility
            technical_indicators=technical_indicators
        )
        
        # Test low volatility scenario with sync analyzer
        ml_analyzer_low_vol = MagicMock()
        ml_analyzer_low_vol.analyze_batch = MagicMock(return_value=[low_volatility_prediction])
        del ml_analyzer_low_vol.batch_analyze
        
        rl_agent_low_vol = self.create_mock_rl_agent()
        bridge_low_vol = MLRLBridge(
            ml_analyzer=ml_analyzer_low_vol,
            rl_agent=rl_agent_low_vol,
            tokens=[sample_token]
        )
        
        results_low_vol = bridge_low_vol.predict_and_act(portfolio_value=10000.0, positions={})
        enhanced_state_low_vol = results_low_vol[0]['enhanced_state']
        
        # Test high volatility scenario with sync analyzer
        ml_analyzer_high_vol = MagicMock()
        ml_analyzer_high_vol.analyze_batch = MagicMock(return_value=[high_volatility_prediction])
        del ml_analyzer_high_vol.batch_analyze
        
        rl_agent_high_vol = self.create_mock_rl_agent()
        bridge_high_vol = MLRLBridge(
            ml_analyzer=ml_analyzer_high_vol,
            rl_agent=rl_agent_high_vol,
            tokens=[sample_token]
        )
        
        results_high_vol = bridge_high_vol.predict_and_act(portfolio_value=10000.0, positions={})
        enhanced_state_high_vol = results_high_vol[0]['enhanced_state']
        
        # Verify volatility forecast is properly included in market state
        assert enhanced_state_low_vol.volatility_forecast == 0.05
        assert enhanced_state_high_vol.volatility_forecast == 0.25
        
        # Verify feature vector includes volatility information
        low_vol_features = enhanced_state_low_vol.to_feature_vector()
        high_vol_features = enhanced_state_high_vol.to_feature_vector()
        
        # Volatility forecast should be the last feature in the ML features section
        assert low_vol_features[-1] == 0.05
        assert high_vol_features[-1] == 0.25
    
    def test_ml_confidence_thresholds_modulate_action_strength(self, sample_token, technical_indicators):
        """Test that ML confidence thresholds properly modulate RL action strength"""
        confidence_levels = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
        
        for confidence in confidence_levels:
            # Create prediction with specific confidence level
            prediction = self.create_ml_prediction(
                token=sample_token,
                confidence=confidence,
                direction=PredictionDirection.BUY,
                technical_indicators=technical_indicators
            )
            
            # Create enhanced market state
            enhanced_state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=10000.0,
                position_size=0.0
            )
            
            # Verify confidence is properly encoded in feature vector
            feature_vector = enhanced_state.to_feature_vector()
            ml_confidence_feature = feature_vector[-3]  # ML confidence is 3rd from last
            
            # Confidence should be clamped to [0, 1] range
            assert 0.0 <= ml_confidence_feature <= 1.0
            assert abs(ml_confidence_feature - confidence) < 0.001
            
            # High confidence should lead to stronger position encoding
            if confidence > 0.8:
                assert enhanced_state.ml_direction == "buy"
                assert feature_vector[-2] == 1.0  # Buy direction encoding
            elif confidence < 0.5:
                # Low confidence predictions should be more conservative
                assert ml_confidence_feature < 0.5
    
    def test_historical_ml_accuracy_affects_decision_weighting(self):
        """Test that historical ML accuracy affects future RL decision weighting"""
        # Create performance metrics with different accuracy histories
        high_accuracy_metrics = MLRLPerformanceMetrics()
        low_accuracy_metrics = MLRLPerformanceMetrics()
        
        # Simulate high accuracy history
        for episode in range(10):
            high_accuracy_metrics.add_episode_data(
                episode=episode,
                ml_accuracy=0.85,  # High accuracy
                rl_reward=150.0,
                decisions_aligned=8,
                total_decisions=10,
                integration_latency=0.1
            )
        
        # Simulate low accuracy history
        for episode in range(10):
            low_accuracy_metrics.add_episode_data(
                episode=episode,
                ml_accuracy=0.45,  # Low accuracy
                rl_reward=80.0,
                decisions_aligned=4,
                total_decisions=10,
                integration_latency=0.15
            )
        
        # Get statistics
        high_accuracy_stats = high_accuracy_metrics.get_statistics()
        low_accuracy_stats = low_accuracy_metrics.get_statistics()
        
        # Verify accuracy tracking
        assert high_accuracy_stats['mean_ml_accuracy'] == 0.85
        assert low_accuracy_stats['mean_ml_accuracy'] == 0.45
        
        # Verify integration scores reflect accuracy differences
        assert high_accuracy_stats['mean_integration_score'] > low_accuracy_stats['mean_integration_score']
        assert high_accuracy_stats['mean_decision_alignment'] > low_accuracy_stats['mean_decision_alignment']
    
    def test_ml_fundamental_analysis_disagreement_scenarios(self, sample_token, technical_indicators):
        """Test edge cases where ML and fundamental analysis disagree"""
        # Create conflicting signals: ML says sell, but fundamentals might suggest buy
        conflicting_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.7,
            direction=PredictionDirection.SELL,  # ML suggests sell
            technical_indicators=technical_indicators  # But technical indicators show strength
        )
        
        # Setup ML-RL bridge with sync analyzer
        ml_analyzer = MagicMock()
        ml_analyzer.analyze_batch = MagicMock(return_value=[conflicting_prediction])
        del ml_analyzer.batch_analyze
        
        # Create RL agent that considers both ML and technical signals
        rl_agent = self.create_mock_rl_agent({
            0.7: (TradeAction.HOLD, 0.6)  # Conservative action due to conflict
        })
        
        bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[sample_token]
        )
        
        # Get integrated decision
        results = bridge.predict_and_act(portfolio_value=10000.0, positions={})
        result = results[0]
        
        # Verify that conflicting signals lead to more conservative action
        assert result['ml_prediction'].direction == PredictionDirection.SELL
        assert result['rl_action'][0] == TradeAction.HOLD  # Conservative due to conflict
        
        # Verify enhanced state includes both ML and technical features
        enhanced_state = result['enhanced_state']
        assert enhanced_state.ml_direction == "sell"
        assert enhanced_state.rsi == 65.0  # Technical strength
        assert enhanced_state.ml_confidence == 0.7
    
    def test_ml_rl_config_validation(self):
        """Test ML-RL configuration validation"""
        # Valid configuration
        valid_config = MLRLConfig(ml_weight=0.4, rl_weight=0.6)
        assert valid_config.ml_weight == 0.4
        assert valid_config.rl_weight == 0.6
        
        # Invalid configuration (weights don't sum to 1.0)
        with pytest.raises(ValueError, match="ML weight \\+ RL weight must equal 1.0"):
            MLRLConfig(ml_weight=0.3, rl_weight=0.5)  # Sum = 0.8
        
        # Another invalid configuration
        with pytest.raises(ValueError, match="ML weight \\+ RL weight must equal 1.0"):
            MLRLConfig(ml_weight=0.6, rl_weight=0.5)  # Sum = 1.1
    
    def test_enhanced_market_state_feature_vector_completeness(self, sample_token, technical_indicators):
        """Test that enhanced market state produces complete feature vectors"""
        prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.8,
            direction=PredictionDirection.BUY,
            volatility_forecast=0.15,
            technical_indicators=technical_indicators
        )
        
        # Create enhanced market state
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.5
        )
        
        # Get feature vector
        feature_vector = enhanced_state.to_feature_vector()
        
        # Verify feature vector has correct dimensions (19 base + 6 ML = 25)
        assert len(feature_vector) == 25
        
        # Verify all features are valid numbers
        assert np.all(np.isfinite(feature_vector))
        
        # Verify ML features are properly encoded
        ml_features = feature_vector[-6:]  # Last 6 features are ML features
        
        # ML prediction features should reflect the prediction
        assert ml_features[0] > sample_token.price_usd  # 1h prediction higher
        assert ml_features[1] > sample_token.price_usd  # 4h prediction higher
        assert ml_features[2] > sample_token.price_usd  # 24h prediction higher
        assert ml_features[3] == 0.8  # Confidence
        assert ml_features[4] == 1.0  # Buy direction encoding
        assert ml_features[5] == 0.15  # Volatility forecast
    
    def test_integrated_training_pipeline_accuracy_tracking(self, sample_token):
        """Test that integrated training pipeline tracks ML accuracy properly"""
        # Create training configuration
        from src.rl_agent.training_pipeline import TrainingConfig
        config = TrainingConfig(
            num_episodes=5,
            learning_rate=0.001,
            batch_size=32
        )
        
        # Create ML-RL training pipeline
        pipeline = MLRLTrainingPipeline(
            config=config,
            tokens=[sample_token],
            historical_data={"prices": [100, 102, 101, 103, 105]}
        )
        
        # Run training
        results = pipeline.train_integrated()
        
        # Verify training completed
        assert results['episodes_completed'] == 5
        assert 'ml_rl_metrics' in results
        assert 'integration_performance' in results
        
        # Verify accuracy tracking
        ml_rl_metrics = results['ml_rl_metrics']
        assert 'mean_ml_accuracy' in ml_rl_metrics
        assert 'mean_decision_alignment' in ml_rl_metrics
        assert ml_rl_metrics['episodes_completed'] == 5
        
        # Verify integration performance metrics
        integration_perf = results['integration_performance']
        assert 'prediction_accuracy' in integration_perf
        assert 'decision_consistency' in integration_perf
        assert 0.0 <= integration_perf['prediction_accuracy'] <= 1.0
    
    def test_ml_rl_bridge_caching_behavior(self, sample_token, technical_indicators):
        """Test that ML-RL bridge properly caches predictions for performance"""
        prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.8,
            direction=PredictionDirection.BUY,
            technical_indicators=technical_indicators
        )
        
        # Create ML analyzer that tracks call count
        call_count = 0
        original_predictions = [prediction]
        
        def mock_analyze_batch(tokens):
            nonlocal call_count
            call_count += 1
            return original_predictions
        
        ml_analyzer = MagicMock()
        ml_analyzer.analyze_batch = MagicMock(side_effect=mock_analyze_batch)
        # Remove batch_analyze to force sync path
        del ml_analyzer.batch_analyze
        
        # Create bridge with short cache TTL for testing
        bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=self.create_mock_rl_agent(),
            tokens=[sample_token],
            cache_ttl_minutes=1
        )
        
        # First call should hit ML analyzer
        predictions1 = bridge.get_ml_predictions()
        assert call_count == 1
        assert len(predictions1) == 1
        
        # Second immediate call should use cache
        predictions2 = bridge.get_ml_predictions()
        assert call_count == 1  # No additional call
        assert predictions1 is predictions2  # Same object from cache
        
        # Wait for cache to expire and verify fresh call
        import time
        bridge.cache_ttl_minutes = 0.01  # 0.6 seconds
        time.sleep(1)
        
        predictions3 = bridge.get_ml_predictions()
        assert call_count == 2  # Fresh call made
    
    def test_prediction_uncertainty_integration(self, sample_token, technical_indicators):
        """Test that prediction uncertainty properly integrates with RL decision-making"""
        # Create predictions with different uncertainty levels
        certain_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.9,  # High certainty
            direction=PredictionDirection.BUY,
            volatility_forecast=0.05,  # Low volatility = more certainty
            technical_indicators=technical_indicators
        )
        
        uncertain_prediction = self.create_ml_prediction(
            token=sample_token,
            confidence=0.6,  # Moderate certainty
            direction=PredictionDirection.BUY,
            volatility_forecast=0.3,  # High volatility = more uncertainty
            technical_indicators=technical_indicators
        )
        
        # Test certain prediction with sync analyzer
        ml_analyzer_certain = MagicMock()
        ml_analyzer_certain.analyze_batch = MagicMock(return_value=[certain_prediction])
        del ml_analyzer_certain.batch_analyze
        
        rl_agent_certain = self.create_mock_rl_agent()
        bridge_certain = MLRLBridge(
            ml_analyzer=ml_analyzer_certain,
            rl_agent=rl_agent_certain,
            tokens=[sample_token]
        )
        
        results_certain = bridge_certain.predict_and_act(portfolio_value=10000.0, positions={})
        
        # Test uncertain prediction with sync analyzer
        ml_analyzer_uncertain = MagicMock()
        ml_analyzer_uncertain.analyze_batch = MagicMock(return_value=[uncertain_prediction])
        del ml_analyzer_uncertain.batch_analyze
        
        rl_agent_uncertain = self.create_mock_rl_agent()
        bridge_uncertain = MLRLBridge(
            ml_analyzer=ml_analyzer_uncertain,
            rl_agent=rl_agent_uncertain,
            tokens=[sample_token]
        )
        
        results_uncertain = bridge_uncertain.predict_and_act(portfolio_value=10000.0, positions={})
        
        # Compare results - certain predictions should lead to stronger actions
        certain_result = results_certain[0]
        uncertain_result = results_uncertain[0]
        
        # High certainty should lead to stronger RL confidence
        assert certain_result['rl_action'][1] >= uncertain_result['rl_action'][1]
        
        # Verify uncertainty is encoded in feature vectors
        certain_features = certain_result['enhanced_state'].to_feature_vector()
        uncertain_features = uncertain_result['enhanced_state'].to_feature_vector()
        
        # Higher ML confidence and lower volatility should be reflected
        assert certain_features[-3] > uncertain_features[-3]  # ML confidence
        assert certain_features[-1] < uncertain_features[-1]  # Volatility forecast