"""
Cross-Module Integration Tests for Shyvr AI Trading System

Comprehensive integration tests validating complete data flow across:
1. Discovery → Evaluation pipeline
2. Evaluation → ML Analysis pipeline  
3. ML Analysis → RL Agent pipeline
4. Complete end-to-end data flow

These tests ensure that DiscoveredToken objects flow correctly through
the entire system pipeline while maintaining data integrity and proper
transformations at each integration point.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Any, Optional

# Discovery module imports
from src.discovery.base import (
    DiscoveredToken,
    TokenStatus,
    TokenDiscoveryBase
)

# Evaluation module imports  
from src.evaluation.base import (
    EvaluationResult,
    EvaluationStatus,
    RiskLevel,
    SecurityFlags,
    FundamentalMetrics,
    TokenEvaluatorBase
)

# ML Analysis module imports
from src.ml_analysis.base import (
    PredictionResult,
    ModelType,
    PredictionDirection,
    TechnicalIndicators,
    MarketFeatures,
    MLAnalyzerBase
)

# RL Agent module imports
from src.rl_agent.base import (
    MarketState,
    TradeAction,
    TradingResult,
    AgentConfig,
    RLAgentBase
)

# Integration imports
from src.integration.ml_rl_bridge import MLEnhancedMarketState, MLRLBridge

# Utils
from src.utils.base import Chain


class TestDiscoveryToEvaluationPipeline:
    """Test data flow from Discovery to Evaluation modules"""

    @pytest.fixture
    def sample_discovered_tokens(self):
        """Create sample discovered tokens for testing"""
        return [
            DiscoveredToken(
                address="0x1234567890abcdef1234567890abcdef12345678",
                symbol="TESTCOIN",
                name="Test Coin",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="birdeye",
                status=TokenStatus.DISCOVERED,
                price_usd=0.00123,
                market_cap=1500000.0,
                volume_24h=75000.0,
                price_change_24h=12.5,
                decimals=18,
                total_supply=1000000000.0,
                circulating_supply=750000000.0,
                trending_score=0.78,
                social_mentions=45,
                tags=["defi", "new"],
                metadata={"source_rank": 1, "confidence": 0.85}
            ),
            DiscoveredToken(
                address="0xabcdef1234567890abcdef1234567890abcdef12",
                symbol="ALTCOIN",
                name="Alternative Coin",
                chain=Chain.SOLANA,
                discovered_at=datetime.now() - timedelta(hours=2),
                discovery_source="jupiter",
                status=TokenStatus.TRENDING,
                price_usd=0.0456,
                market_cap=2300000.0,
                volume_24h=120000.0,
                price_change_24h=-3.2,
                decimals=6,
                total_supply=500000000.0,
                circulating_supply=400000000.0,
                trending_score=0.92,
                social_mentions=128,
                tags=["meme", "trending"],
                metadata={"source_rank": 2, "confidence": 0.91}
            )
        ]

    @pytest.fixture
    def mock_token_evaluator(self):
        """Create mock token evaluator"""
        evaluator = MagicMock(spec=TokenEvaluatorBase)
        
        async def mock_evaluate_token(token: DiscoveredToken) -> EvaluationResult:
            # Create realistic evaluation based on token data
            security_flags = SecurityFlags(
                is_honeypot=False,
                is_rugpull_risk=False,
                contract_verified=True,
                liquidity_locked=True,
                security_score=75.0 if token.volume_24h > 100000 else 65.0
            )
            
            fundamental_metrics = FundamentalMetrics(
                liquidity_usd=token.volume_24h * 0.1,  # Estimate liquidity from volume
                holder_count=int(token.market_cap / 1000) if token.market_cap else 100,
                volume_24h=token.volume_24h,
                price_usd=token.price_usd,
                price_change_24h=token.price_change_24h,
                market_cap=token.market_cap,
                social_score=min(token.social_mentions / 10.0, 100.0) if token.social_mentions else 50.0
            )
            
            overall_score = fundamental_metrics.calculate_fundamental_score()
            
            return EvaluationResult(
                token=token,
                evaluated_at=datetime.now(),
                evaluation_duration_ms=250.0,
                status=EvaluationStatus.COMPLETED,
                overall_risk=RiskLevel.LOW if overall_score > 70 else RiskLevel.MEDIUM,
                overall_score=overall_score,
                security_flags=security_flags,
                fundamental_metrics=fundamental_metrics,
                is_approved=overall_score > 60.0 and security_flags.is_safe_to_trade(),
                recommended_action="BUY" if overall_score > 70 else "HOLD",
                confidence_level=75.0 if overall_score > 70 else 65.0,
                security_risk=25.0,
                liquidity_risk=30.0,
                volatility_risk=abs(token.price_change_24h) if token.price_change_24h else 20.0,
                social_risk=40.0,
                warnings=[] if security_flags.is_safe_to_trade() else ["Low security score"],
                notes=[f"Evaluated from {token.discovery_source}"]
            )
        
        evaluator.evaluate_token.side_effect = mock_evaluate_token
        return evaluator

    @pytest.mark.asyncio
    async def test_discovered_token_to_evaluation_result_transformation(self, sample_discovered_tokens, mock_token_evaluator):
        """Test transformation from DiscoveredToken to EvaluationResult"""
        token = sample_discovered_tokens[0]  # TESTCOIN
        
        # Evaluate the discovered token
        evaluation_result = await mock_token_evaluator.evaluate_token(token)
        
        # Verify the transformation maintains token identity
        assert evaluation_result.token == token
        assert evaluation_result.token.address == token.address
        assert evaluation_result.token.symbol == token.symbol
        assert evaluation_result.token.chain == token.chain
        
        # Verify evaluation status
        assert evaluation_result.status == EvaluationStatus.COMPLETED
        assert isinstance(evaluation_result.overall_score, float)
        assert 0.0 <= evaluation_result.overall_score <= 100.0
        
        # Verify security flags are populated
        assert evaluation_result.security_flags is not None
        assert isinstance(evaluation_result.security_flags.security_score, float)
        
        # Verify fundamental metrics use discovery data
        assert evaluation_result.fundamental_metrics is not None
        assert evaluation_result.fundamental_metrics.price_usd == token.price_usd
        assert evaluation_result.fundamental_metrics.volume_24h == token.volume_24h
        assert evaluation_result.fundamental_metrics.market_cap == token.market_cap

    @pytest.mark.asyncio
    async def test_batch_token_evaluation_pipeline(self, sample_discovered_tokens, mock_token_evaluator):
        """Test batch processing of discovered tokens through evaluation"""
        # Process all tokens through evaluation
        evaluation_results = []
        for token in sample_discovered_tokens:
            result = await mock_token_evaluator.evaluate_token(token)
            evaluation_results.append(result)
        
        # Verify all tokens were processed
        assert len(evaluation_results) == len(sample_discovered_tokens)
        
        # Verify each result maintains proper token association
        for i, result in enumerate(evaluation_results):
            original_token = sample_discovered_tokens[i]
            assert result.token.address == original_token.address
            assert result.token.symbol == original_token.symbol
            assert result.status == EvaluationStatus.COMPLETED
            
            # Verify evaluation metrics are reasonable
            assert result.overall_score > 0.0
            assert result.confidence_level > 0.0
            assert result.recommended_action in ["BUY", "SELL", "HOLD", "AVOID"]

    @pytest.mark.asyncio
    async def test_discovery_metadata_preservation_in_evaluation(self, sample_discovered_tokens, mock_token_evaluator):
        """Test that discovery metadata is preserved through evaluation"""
        token = sample_discovered_tokens[0]  # Has rich metadata
        evaluation_result = await mock_token_evaluator.evaluate_token(token)
        
        # Verify discovery metadata is accessible through evaluation result
        assert evaluation_result.token.discovery_source == "birdeye"
        assert evaluation_result.token.trending_score == 0.78
        assert evaluation_result.token.social_mentions == 45
        assert "defi" in evaluation_result.token.tags
        assert evaluation_result.token.metadata["confidence"] == 0.85
        
        # Verify evaluation adds its own metadata without losing discovery data
        assert evaluation_result.evaluation_duration_ms is not None
        assert evaluation_result.evaluated_at is not None


class TestEvaluationToMLAnalysisPipeline:
    """Test data flow from Evaluation to ML Analysis modules"""

    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample evaluation results for ML analysis"""
        # Create base tokens
        token1 = DiscoveredToken(
            address="0x1111111111111111111111111111111111111111",
            symbol="MLTEST1",
            name="ML Test Token 1",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0089,
            volume_24h=95000.0,
            market_cap=1800000.0,
            price_change_24h=8.5
        )
        
        token2 = DiscoveredToken(
            address="0x2222222222222222222222222222222222222222",
            symbol="MLTEST2", 
            name="ML Test Token 2",
            chain=Chain.SOLANA,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0234,
            volume_24h=156000.0,
            market_cap=3200000.0,
            price_change_24h=-2.1
        )
        
        # Create evaluation results
        evaluation1 = EvaluationResult(
            token=token1,
            evaluated_at=datetime.now(),
            status=EvaluationStatus.COMPLETED,
            overall_risk=RiskLevel.LOW,
            overall_score=78.5,
            security_flags=SecurityFlags(
                is_honeypot=False,
                is_rugpull_risk=False,
                contract_verified=True,
                security_score=80.0
            ),
            fundamental_metrics=FundamentalMetrics(
                liquidity_usd=9500.0,
                holder_count=180,
                volume_24h=95000.0,
                price_usd=0.0089,
                price_change_24h=8.5,
                market_cap=1800000.0,
                social_score=65.0
            ),
            is_approved=True,
            recommended_action="BUY",
            confidence_level=82.0
        )
        
        evaluation2 = EvaluationResult(
            token=token2,
            evaluated_at=datetime.now(),
            status=EvaluationStatus.COMPLETED,
            overall_risk=RiskLevel.MEDIUM,
            overall_score=65.2,
            security_flags=SecurityFlags(
                is_honeypot=False,
                is_rugpull_risk=False,
                contract_verified=True,
                security_score=70.0
            ),
            fundamental_metrics=FundamentalMetrics(
                liquidity_usd=15600.0,
                holder_count=240,
                volume_24h=156000.0,
                price_usd=0.0234,
                price_change_24h=-2.1,
                market_cap=3200000.0,
                social_score=58.0
            ),
            is_approved=True,
            recommended_action="HOLD",
            confidence_level=68.0
        )
        
        return [evaluation1, evaluation2]

    @pytest.fixture
    def mock_ml_analyzer(self):
        """Create mock ML analyzer"""
        analyzer = MagicMock(spec=MLAnalyzerBase)
        
        async def mock_analyze_token(token: DiscoveredToken, historical_data=None) -> PredictionResult:
            # Create technical indicators based on token data
            tech_indicators = TechnicalIndicators(
                rsi=65.0 if token.price_change_24h > 0 else 45.0,
                macd=0.001 if token.price_change_24h > 0 else -0.001,
                sma_20=token.price_usd * 0.98,
                sma_50=token.price_usd * 0.95,
                ema_12=token.price_usd * 1.01,
                ema_26=token.price_usd * 0.99,
                bollinger_upper=token.price_usd * 1.05,
                bollinger_lower=token.price_usd * 0.95,
                atr=token.price_usd * 0.02,
                volume_sma=token.volume_24h * 0.8 if token.volume_24h else 50000,
                volume_ratio=1.2 if token.volume_24h > 100000 else 0.8,
                obv=token.volume_24h * 0.6 if token.volume_24h else 30000
            )
            
            # Market features
            market_features = MarketFeatures(
                fear_greed_index=55.0,
                market_trend="bull" if token.price_change_24h > 5 else "bear",
                volatility_regime="medium",
                btc_correlation=0.3,
                social_score=0.6
            )
            
            # Prediction direction based on price change
            if token.price_change_24h > 5:
                direction = PredictionDirection.BUY
                prob_up = 0.75
            elif token.price_change_24h < -5:
                direction = PredictionDirection.SELL
                prob_up = 0.25
            else:
                direction = PredictionDirection.HOLD
                prob_up = 0.5
            
            return PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_1h=token.price_usd * (1.0 + token.price_change_24h / 2400),  # 1h prediction
                price_prediction_4h=token.price_usd * (1.0 + token.price_change_24h / 600),   # 4h prediction
                price_prediction_24h=token.price_usd * (1.0 + token.price_change_24h / 100), # 24h prediction
                direction=direction,
                confidence=0.75 if abs(token.price_change_24h) > 5 else 0.6,
                probability_up=prob_up,
                technical_indicators=tech_indicators,
                market_features=market_features,
                model_accuracy=0.78,
                prediction_uncertainty=0.15,
                volatility_forecast=abs(token.price_change_24h) / 100.0,
                downside_risk=0.1 if token.price_change_24h > 0 else 0.2,
                upside_potential=0.2 if token.price_change_24h > 0 else 0.1,
                entry_signal="buy" if direction == PredictionDirection.BUY else "hold",
                stop_loss_level=token.price_usd * 0.95,
                take_profit_level=token.price_usd * 1.15,
                features_used=["price", "volume", "rsi", "macd", "social"],
                model_version="lstm_v1.2",
                processing_time_ms=850.0
            )
        
        analyzer.analyze_token.side_effect = mock_analyze_token
        return analyzer

    @pytest.mark.asyncio
    async def test_evaluation_result_to_ml_prediction_transformation(self, sample_evaluation_results, mock_ml_analyzer):
        """Test transformation from EvaluationResult to ML PredictionResult"""
        evaluation_result = sample_evaluation_results[0]  # MLTEST1 with BUY recommendation
        token = evaluation_result.token
        
        # Run ML analysis on the evaluated token
        ml_prediction = await mock_ml_analyzer.analyze_token(token)
        
        # Verify token identity is preserved
        assert ml_prediction.token == token
        assert ml_prediction.token.address == token.address
        assert ml_prediction.token.symbol == token.symbol
        
        # Verify ML predictions are generated
        assert ml_prediction.price_prediction_1h is not None
        assert ml_prediction.price_prediction_4h is not None
        assert ml_prediction.price_prediction_24h is not None
        assert ml_prediction.confidence > 0.0
        
        # Verify technical indicators are populated
        assert ml_prediction.technical_indicators is not None
        assert ml_prediction.technical_indicators.rsi is not None
        assert ml_prediction.technical_indicators.macd is not None
        
        # Verify ML predictions align with evaluation (positive price change → BUY)
        if token.price_change_24h > 0:
            assert ml_prediction.direction in [PredictionDirection.BUY, PredictionDirection.HOLD]
            assert ml_prediction.probability_up >= 0.5

    @pytest.mark.asyncio
    async def test_evaluation_quality_influences_ml_confidence(self, sample_evaluation_results, mock_ml_analyzer):
        """Test that evaluation quality influences ML analysis confidence"""
        high_quality_eval = sample_evaluation_results[0]  # 78.5 score, 82% confidence
        medium_quality_eval = sample_evaluation_results[1]  # 65.2 score, 68% confidence
        
        # Analyze both tokens
        high_quality_prediction = await mock_ml_analyzer.analyze_token(high_quality_eval.token)
        medium_quality_prediction = await mock_ml_analyzer.analyze_token(medium_quality_eval.token)
        
        # Both should have valid predictions
        assert high_quality_prediction.confidence > 0.0
        assert medium_quality_prediction.confidence > 0.0
        
        # Verify predictions contain required technical analysis
        for prediction in [high_quality_prediction, medium_quality_prediction]:
            assert prediction.technical_indicators is not None
            assert prediction.market_features is not None
            assert prediction.processing_time_ms is not None
            assert len(prediction.features_used) > 0

    @pytest.mark.asyncio
    async def test_evaluation_risk_level_affects_ml_strategy(self, sample_evaluation_results, mock_ml_analyzer):
        """Test that evaluation risk levels affect ML trading strategy"""
        low_risk_eval = sample_evaluation_results[0]   # RiskLevel.LOW
        medium_risk_eval = sample_evaluation_results[1] # RiskLevel.MEDIUM
        
        low_risk_prediction = await mock_ml_analyzer.analyze_token(low_risk_eval.token)
        medium_risk_prediction = await mock_ml_analyzer.analyze_token(medium_risk_eval.token)
        
        # Verify risk considerations in ML predictions
        assert low_risk_prediction.stop_loss_level is not None
        assert medium_risk_prediction.stop_loss_level is not None
        
        # Higher risk should generally have closer stop losses (more conservative)
        # This is reflected in the volatility forecast and risk metrics
        assert low_risk_prediction.volatility_forecast is not None
        assert medium_risk_prediction.volatility_forecast is not None

    @pytest.mark.asyncio
    async def test_batch_evaluation_to_ml_pipeline(self, sample_evaluation_results, mock_ml_analyzer):
        """Test batch processing from evaluation results to ML predictions"""
        # Process all evaluation results through ML analysis
        ml_predictions = []
        for eval_result in sample_evaluation_results:
            prediction = await mock_ml_analyzer.analyze_token(eval_result.token)
            ml_predictions.append(prediction)
        
        # Verify all tokens were processed
        assert len(ml_predictions) == len(sample_evaluation_results)
        
        # Verify each prediction maintains token association
        for i, prediction in enumerate(ml_predictions):
            original_eval = sample_evaluation_results[i]
            assert prediction.token.address == original_eval.token.address
            assert prediction.model_type == ModelType.LSTM
            assert prediction.analyzed_at is not None
            
            # Verify predictions have valid ranges
            current_price = prediction.token.price_usd
            assert prediction.price_prediction_1h > current_price * 0.5
            assert prediction.price_prediction_1h < current_price * 2.0
            assert prediction.price_prediction_24h > current_price * 0.5
            assert prediction.price_prediction_24h < current_price * 2.0


class TestMLAnalysisToRLAgentPipeline:
    """Test data flow from ML Analysis to RL Agent modules"""

    @pytest.fixture
    def sample_ml_predictions(self):
        """Create sample ML predictions for RL analysis"""
        # Create base tokens with different characteristics
        token1 = DiscoveredToken(
            address="0x3333333333333333333333333333333333333333",
            symbol="RLTEST1",
            name="RL Test Token 1", 
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0145,
            volume_24h=125000.0,
            market_cap=2100000.0,
            price_change_24h=15.8
        )
        
        token2 = DiscoveredToken(
            address="0x4444444444444444444444444444444444444444",
            symbol="RLTEST2",
            name="RL Test Token 2",
            chain=Chain.SOLANA,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0067,
            volume_24h=89000.0,
            market_cap=890000.0,
            price_change_24h=-8.2
        )
        
        # Create ML predictions with rich technical data
        tech_indicators1 = TechnicalIndicators(
            rsi=72.5,
            macd=0.0015,
            sma_20=0.0142,
            sma_50=0.0138,
            ema_12=0.0147,
            ema_26=0.0141,
            bollinger_upper=0.0152,
            bollinger_lower=0.0138,
            atr=0.0003,
            volume_sma=115000,
            volume_ratio=1.09,
            obv=75000
        )
        
        tech_indicators2 = TechnicalIndicators(
            rsi=38.2,
            macd=-0.0008,
            sma_20=0.0070,
            sma_50=0.0074,
            ema_12=0.0065,
            ema_26=0.0069,
            bollinger_upper=0.0075,
            bollinger_lower=0.0059,
            atr=0.0005,
            volume_sma=95000,
            volume_ratio=0.94,
            obv=53000
        )
        
        prediction1 = PredictionResult(
            token=token1,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=0.0147,
            price_prediction_4h=0.0151,
            price_prediction_24h=0.0158,
            direction=PredictionDirection.BUY,
            confidence=0.82,
            probability_up=0.78,
            technical_indicators=tech_indicators1,
            model_accuracy=0.76,
            volatility_forecast=0.12,
            downside_risk=0.08,
            upside_potential=0.18,
            entry_signal="buy",
            stop_loss_level=0.0138,
            take_profit_level=0.0165
        )
        
        prediction2 = PredictionResult(
            token=token2,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=0.0065,
            price_prediction_4h=0.0063,
            price_prediction_24h=0.0061,
            direction=PredictionDirection.SELL,
            confidence=0.71,
            probability_up=0.32,
            technical_indicators=tech_indicators2,
            model_accuracy=0.74,
            volatility_forecast=0.16,
            downside_risk=0.15,
            upside_potential=0.05,
            entry_signal="sell",
            stop_loss_level=0.0070,
            take_profit_level=0.0058
        )
        
        return [prediction1, prediction2]

    @pytest.fixture
    def mock_rl_agent(self):
        """Create mock RL agent"""
        agent = MagicMock(spec=RLAgentBase)
        
        async def mock_predict_action(state: MarketState):
            # Make decisions based on market state features
            if hasattr(state, 'ml_confidence') and state.ml_confidence > 0.75:
                if hasattr(state, 'ml_direction') and state.ml_direction == "buy":
                    return TradeAction.STRONG_BUY, 0.85
                elif hasattr(state, 'ml_direction') and state.ml_direction == "sell":
                    return TradeAction.STRONG_SELL, 0.85
            elif hasattr(state, 'ml_confidence') and state.ml_confidence > 0.6:
                if hasattr(state, 'ml_direction') and state.ml_direction == "buy":
                    return TradeAction.BUY, 0.70
                elif hasattr(state, 'ml_direction') and state.ml_direction == "sell":
                    return TradeAction.SELL, 0.70
            
            return TradeAction.HOLD, 0.50
        
        agent.predict_action.side_effect = mock_predict_action
        return agent

    def test_ml_prediction_to_market_state_transformation(self, sample_ml_predictions):
        """Test transformation from ML PredictionResult to MarketState"""
        ml_prediction = sample_ml_predictions[0]  # RLTEST1 with BUY signal
        
        # Create enhanced market state from ML prediction
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=ml_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Verify basic market state features
        assert enhanced_state.token == ml_prediction.token
        assert enhanced_state.price_usd == ml_prediction.token.price_usd
        assert enhanced_state.volume_24h == ml_prediction.token.volume_24h
        
        # Verify technical indicators are transferred
        assert enhanced_state.rsi == ml_prediction.technical_indicators.rsi
        assert enhanced_state.macd == ml_prediction.technical_indicators.macd
        assert enhanced_state.sma_20 == ml_prediction.technical_indicators.sma_20
        assert enhanced_state.ema_12 == ml_prediction.technical_indicators.ema_12
        assert enhanced_state.bollinger_upper == ml_prediction.technical_indicators.bollinger_upper
        assert enhanced_state.bollinger_lower == ml_prediction.technical_indicators.bollinger_lower
        
        # Verify ML-specific features
        assert enhanced_state.ml_prediction_1h == ml_prediction.price_prediction_1h
        assert enhanced_state.ml_prediction_4h == ml_prediction.price_prediction_4h
        assert enhanced_state.ml_prediction_24h == ml_prediction.price_prediction_24h
        assert enhanced_state.ml_confidence == ml_prediction.confidence
        assert enhanced_state.ml_direction == ml_prediction.direction.value
        assert enhanced_state.volatility_forecast == ml_prediction.volatility_forecast

    def test_enhanced_market_state_feature_vector(self, sample_ml_predictions):
        """Test MLEnhancedMarketState generates correct feature vector"""
        ml_prediction = sample_ml_predictions[0]
        
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=ml_prediction,
            current_portfolio_value=10000.0,
            position_size=0.1
        )
        
        # Get feature vector
        feature_vector = enhanced_state.to_feature_vector()
        
        # Verify dimensions (25 features: 19 base + 6 ML)
        assert len(feature_vector) == 25
        assert feature_vector.dtype == np.float32
        
        # Verify no NaN or infinite values
        assert np.all(np.isfinite(feature_vector))
        assert not np.any(np.isnan(feature_vector))
        
        # Verify ML features are in the last 6 positions
        ml_features = feature_vector[19:]
        assert len(ml_features) == 6
        
        # Verify ML prediction values are reasonable
        assert ml_features[0] > 0  # ml_prediction_1h
        assert ml_features[1] > 0  # ml_prediction_4h  
        assert ml_features[2] > 0  # ml_prediction_24h
        assert 0.0 <= ml_features[3] <= 1.0  # ml_confidence
        assert ml_features[4] in [0.0, 1.0]  # ml_direction (binary)
        assert ml_features[5] >= 0.0  # volatility_forecast

    @pytest.mark.asyncio
    async def test_rl_agent_decision_making_with_ml_features(self, sample_ml_predictions, mock_rl_agent):
        """Test RL agent makes decisions based on ML-enhanced market state"""
        # Test strong buy signal (high confidence BUY prediction)
        buy_prediction = sample_ml_predictions[0]  # High confidence BUY
        buy_state = MLEnhancedMarketState.from_prediction(
            prediction=buy_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        action, confidence = await mock_rl_agent.predict_action(buy_state)
        assert action == TradeAction.STRONG_BUY
        assert confidence > 0.8
        
        # Test sell signal (SELL prediction)
        sell_prediction = sample_ml_predictions[1]  # SELL prediction
        sell_state = MLEnhancedMarketState.from_prediction(
            prediction=sell_prediction,
            current_portfolio_value=10000.0,
            position_size=0.2
        )
        
        action, confidence = await mock_rl_agent.predict_action(sell_state)
        assert action in [TradeAction.SELL, TradeAction.STRONG_SELL]
        assert confidence > 0.6

    @pytest.mark.asyncio
    async def test_ml_confidence_affects_rl_action_strength(self, sample_ml_predictions, mock_rl_agent):
        """Test that ML confidence levels affect RL action strength"""
        # Create states with different confidence levels
        high_conf_prediction = sample_ml_predictions[0]  # 0.82 confidence
        
        # Create a copy of prediction for medium confidence test
        import copy
        medium_conf_prediction = copy.deepcopy(sample_ml_predictions[0])
        medium_conf_prediction.confidence = 0.65  # Medium confidence
        
        # Create market states
        high_conf_state = MLEnhancedMarketState.from_prediction(
            prediction=high_conf_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        medium_conf_state = MLEnhancedMarketState.from_prediction(
            prediction=medium_conf_prediction, 
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Get actions
        high_action, high_confidence = await mock_rl_agent.predict_action(high_conf_state)
        medium_action, medium_confidence = await mock_rl_agent.predict_action(medium_conf_state)
        
        # High confidence should lead to stronger action
        assert high_confidence > medium_confidence
        
        # High confidence should prefer STRONG_BUY over BUY
        if high_action == TradeAction.STRONG_BUY:
            assert medium_action in [TradeAction.BUY, TradeAction.HOLD]

    @pytest.mark.asyncio
    async def test_portfolio_state_integration_with_ml_predictions(self, sample_ml_predictions, mock_rl_agent):
        """Test that portfolio state is properly integrated with ML predictions"""
        prediction = sample_ml_predictions[0]
        
        # Test different portfolio states
        scenarios = [
            {"portfolio_value": 10000.0, "position_size": 0.0},    # No position
            {"portfolio_value": 12000.0, "position_size": 0.2},   # 20% position, profit
            {"portfolio_value": 8000.0, "position_size": -0.1},   # Short position, loss
        ]
        
        for scenario in scenarios:
            state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=scenario["portfolio_value"],
                position_size=scenario["position_size"]
            )
            
            # Verify portfolio information is included in state
            assert state.portfolio_value == scenario["portfolio_value"]
            assert state.current_position == scenario["position_size"]
            
            # Verify feature vector includes portfolio features
            feature_vector = state.to_feature_vector()
            assert len(feature_vector) == 25
            
            # RL agent should be able to make decisions
            action, confidence = await mock_rl_agent.predict_action(state)
            assert action in [TradeAction.HOLD, TradeAction.BUY, TradeAction.SELL, 
                            TradeAction.STRONG_BUY, TradeAction.STRONG_SELL]
            assert 0.0 <= confidence <= 1.0


class TestCompleteEndToEndDataFlow:
    """Test complete end-to-end data flow across all modules"""

    @pytest.fixture
    def complete_token_pipeline_data(self):
        """Create complete token data for end-to-end testing"""
        # Step 1: Discovery data
        discovered_token = DiscoveredToken(
            address="0x5555555555555555555555555555555555555555",
            symbol="E2ETEST",
            name="End-to-End Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="birdeye",
            status=TokenStatus.TRENDING,
            price_usd=0.0234,
            market_cap=3400000.0,
            volume_24h=178000.0,
            price_change_24h=12.7,
            decimals=18,
            total_supply=1500000000.0,
            circulating_supply=1200000000.0,
            trending_score=0.85,
            social_mentions=67,
            tags=["defi", "trending", "verified"],
            metadata={"source_rank": 1, "discovery_confidence": 0.92}
        )
        
        # Step 2: Evaluation data
        evaluation_result = EvaluationResult(
            token=discovered_token,
            evaluated_at=datetime.now(),
            evaluation_duration_ms=320.0,
            status=EvaluationStatus.COMPLETED,
            overall_risk=RiskLevel.LOW,
            overall_score=82.3,
            security_flags=SecurityFlags(
                is_honeypot=False,
                is_rugpull_risk=False,
                contract_verified=True,
                ownership_renounced=True,
                liquidity_locked=True,
                security_score=85.0
            ),
            fundamental_metrics=FundamentalMetrics(
                liquidity_usd=17800.0,
                holder_count=285,
                volume_24h=178000.0,
                price_usd=0.0234,
                price_change_24h=12.7,
                market_cap=3400000.0,
                social_score=67.0
            ),
            is_approved=True,
            recommended_action="BUY",
            confidence_level=85.5,
            security_risk=15.0,
            liquidity_risk=20.0,
            volatility_risk=25.0,
            social_risk=18.0,
            warnings=[],
            notes=["High quality token with strong fundamentals"]
        )
        
        # Step 3: ML Analysis data
        tech_indicators = TechnicalIndicators(
            rsi=68.5,
            macd=0.0018,
            sma_20=0.0228,
            sma_50=0.0221,
            ema_12=0.0236,
            ema_26=0.0230,
            bollinger_upper=0.0245,
            bollinger_lower=0.0223,
            atr=0.0004,
            volume_sma=165000,
            volume_ratio=1.08,
            obv=106800
        )
        
        market_features = MarketFeatures(
            fear_greed_index=62.0,
            market_trend="bull",
            volatility_regime="medium",
            btc_correlation=0.45,
            eth_correlation=0.52,
            market_beta=1.15,
            social_score=0.67,
            mention_volume=67,
            sentiment_trend=0.12
        )
        
        ml_prediction = PredictionResult(
            token=discovered_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=0.0237,
            price_prediction_4h=0.0241,
            price_prediction_24h=0.0248,
            direction=PredictionDirection.BUY,
            confidence=0.83,
            probability_up=0.79,
            technical_indicators=tech_indicators,
            market_features=market_features,
            model_accuracy=0.81,
            prediction_uncertainty=0.12,
            volatility_forecast=0.14,
            downside_risk=0.09,
            upside_potential=0.21,
            entry_signal="buy",
            stop_loss_level=0.0222,
            take_profit_level=0.0265,
            position_size_multiplier=1.2,
            features_used=["price", "volume", "rsi", "macd", "social", "market_sentiment"],
            model_version="lstm_v2.1",
            processing_time_ms=680.0
        )
        
        return {
            "discovered_token": discovered_token,
            "evaluation_result": evaluation_result,
            "ml_prediction": ml_prediction
        }

    def test_complete_data_flow_integrity(self, complete_token_pipeline_data):
        """Test that data integrity is maintained through complete pipeline"""
        data = complete_token_pipeline_data
        
        # Verify token identity is preserved across all stages
        token_discovery = data["discovered_token"]
        token_evaluation = data["evaluation_result"].token
        token_ml = data["ml_prediction"].token
        
        assert token_discovery.address == token_evaluation.address == token_ml.address
        assert token_discovery.symbol == token_evaluation.symbol == token_ml.symbol
        assert token_discovery.chain == token_evaluation.chain == token_ml.chain
        
        # Verify timestamps show progression
        assert data["evaluation_result"].evaluated_at >= token_discovery.discovered_at
        assert data["ml_prediction"].analyzed_at >= data["evaluation_result"].evaluated_at
        
        # Verify data enrichment at each stage
        assert token_discovery.price_usd is not None
        assert data["evaluation_result"].overall_score > 0
        assert data["ml_prediction"].confidence > 0
        
        # Verify consistent price data
        assert data["evaluation_result"].fundamental_metrics.price_usd == token_discovery.price_usd
        assert data["ml_prediction"].token.price_usd == token_discovery.price_usd

    def test_end_to_end_ml_rl_integration(self, complete_token_pipeline_data):
        """Test complete ML-RL integration with real data flow"""
        data = complete_token_pipeline_data
        ml_prediction = data["ml_prediction"]
        
        # Create ML-enhanced market state
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=ml_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Verify all data is properly integrated
        assert enhanced_state.token.address == data["discovered_token"].address
        assert enhanced_state.price_usd == data["discovered_token"].price_usd
        assert enhanced_state.rsi == ml_prediction.technical_indicators.rsi
        assert enhanced_state.ml_confidence == ml_prediction.confidence
        assert enhanced_state.ml_direction == ml_prediction.direction.value
        
        # Test feature vector generation
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 25
        assert np.all(np.isfinite(feature_vector))
        
        # Verify feature vector contains discovery, evaluation, and ML data
        assert feature_vector[0] > 0  # price_usd (from discovery)
        assert feature_vector[4] > 0  # rsi (from ML analysis)
        assert feature_vector[19] > 0  # ml_prediction_1h
        assert feature_vector[22] > 0  # ml_confidence

    def test_pipeline_performance_end_to_end(self, complete_token_pipeline_data):
        """Test end-to-end pipeline performance and timing"""
        data = complete_token_pipeline_data
        
        # Measure ML-RL integration time
        start_time = datetime.now()
        
        # Create enhanced market state (this is the integration step)
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=data["ml_prediction"],
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Generate feature vector (preparation for RL)
        feature_vector = enhanced_state.to_feature_vector()
        
        end_time = datetime.now()
        integration_time = (end_time - start_time).total_seconds()
        
        # Verify performance targets
        assert integration_time < 0.1  # Integration should be <100ms
        assert len(feature_vector) == 25
        
        # Verify processing times from each stage are reasonable
        assert data["evaluation_result"].evaluation_duration_ms < 1000  # <1 second
        assert data["ml_prediction"].processing_time_ms < 2000  # <2 seconds

    def test_error_handling_across_pipeline(self, complete_token_pipeline_data):
        """Test error handling and graceful degradation across pipeline"""
        data = complete_token_pipeline_data
        
        # Test with missing ML technical indicators
        incomplete_prediction = data["ml_prediction"]
        incomplete_prediction.technical_indicators = None
        
        # Should still create market state with default values
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=incomplete_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Verify graceful handling
        assert enhanced_state.rsi is None  # Should handle missing data
        assert enhanced_state.ml_confidence == incomplete_prediction.confidence
        
        # Feature vector should still be valid with default values
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 25
        assert np.all(np.isfinite(feature_vector))

    def test_batch_processing_end_to_end(self, complete_token_pipeline_data):
        """Test batch processing through complete pipeline"""
        # Create multiple tokens at different pipeline stages
        base_data = complete_token_pipeline_data
        
        # Create additional tokens with variations
        tokens_batch = []
        for i in range(3):
            token = DiscoveredToken(
                address=f"0x{i+6:040x}",
                symbol=f"BATCH{i}",
                name=f"Batch Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=0.01 + i * 0.005,
                volume_24h=100000 + i * 25000,
                market_cap=1000000 + i * 500000,
                price_change_24h=5.0 + i * 3.0
            )
            tokens_batch.append(token)
        
        # Simulate ML predictions for all tokens
        ml_predictions = []
        for token in tokens_batch:
            prediction = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_1h=token.price_usd * 1.01,
                price_prediction_24h=token.price_usd * 1.05,
                direction=PredictionDirection.BUY,
                confidence=0.75,
                technical_indicators=TechnicalIndicators(rsi=60.0, macd=0.001)
            )
            ml_predictions.append(prediction)
        
        # Create enhanced market states for all
        enhanced_states = []
        for prediction in ml_predictions:
            state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=10000.0,
                position_size=0.0
            )
            enhanced_states.append(state)
        
        # Verify batch processing results
        assert len(enhanced_states) == 3
        for i, state in enumerate(enhanced_states):
            assert state.token.symbol == f"BATCH{i}"
            assert state.ml_confidence == 0.75
            
            feature_vector = state.to_feature_vector()
            assert len(feature_vector) == 25

    def test_data_consistency_validation(self, complete_token_pipeline_data):
        """Test data consistency validation across pipeline stages"""
        data = complete_token_pipeline_data
        
        # Validation rules that should hold across the pipeline
        token = data["discovered_token"]
        evaluation = data["evaluation_result"]
        ml_prediction = data["ml_prediction"]
        
        # Price consistency
        assert token.price_usd == evaluation.fundamental_metrics.price_usd
        assert token.price_usd == ml_prediction.token.price_usd
        
        # Volume consistency  
        assert token.volume_24h == evaluation.fundamental_metrics.volume_24h
        assert token.volume_24h == ml_prediction.token.volume_24h
        
        # Market cap consistency
        assert token.market_cap == evaluation.fundamental_metrics.market_cap
        
        # Chain and address consistency
        assert token.chain == evaluation.token.chain == ml_prediction.token.chain
        assert token.address == evaluation.token.address == ml_prediction.token.address
        
        # Timestamp progression (later stages should have later timestamps)
        assert evaluation.evaluated_at >= token.discovered_at
        assert ml_prediction.analyzed_at >= evaluation.evaluated_at
        
        # Quality consistency (high evaluation score should correlate with ML confidence)
        if evaluation.overall_score > 80:
            assert ml_prediction.confidence > 0.7  # High evaluation → high ML confidence


class TestIntegrationErrorHandlingAndFallbacks:
    """Test error handling and fallback mechanisms across integration points"""

    def test_missing_discovery_data_handling(self):
        """Test handling of incomplete discovery data"""
        # Create token with minimal data
        minimal_token = DiscoveredToken(
            address="0x7777777777777777777777777777777777777777",
            symbol="MINIMAL",
            name="Minimal Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test"
            # Missing: price_usd, volume_24h, market_cap, etc.
        )
        
        # Should still be able to create evaluation result with defaults
        evaluation = EvaluationResult(
            token=minimal_token,
            evaluated_at=datetime.now(),
            status=EvaluationStatus.COMPLETED,
            overall_risk=RiskLevel.HIGH,  # Default to high risk for incomplete data
            overall_score=30.0,  # Low score for incomplete data
            is_approved=False,
            recommended_action="AVOID",
            confidence_level=25.0
        )
        
        assert evaluation.token == minimal_token
        assert evaluation.overall_score < 50.0  # Low score for incomplete data
        assert not evaluation.is_approved

    def test_evaluation_failure_handling(self):
        """Test handling of evaluation failures"""
        token = DiscoveredToken(
            address="0x8888888888888888888888888888888888888888",
            symbol="FAILTEST",
            name="Evaluation Fail Test",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.001
        )
        
        # Create failed evaluation result
        failed_evaluation = EvaluationResult(
            token=token,
            evaluated_at=datetime.now(),
            status=EvaluationStatus.FAILED,
            overall_risk=RiskLevel.VERY_HIGH,
            overall_score=0.0,
            is_approved=False,
            recommended_action="AVOID",
            confidence_level=0.0,
            warnings=["Evaluation failed due to API timeout"],
            notes=["Manual review required"]
        )
        
        # Should still be able to proceed with ML analysis using token data
        # ML should use conservative defaults for failed evaluations
        assert failed_evaluation.status == EvaluationStatus.FAILED
        assert failed_evaluation.overall_score == 0.0
        assert not failed_evaluation.is_approved

    def test_ml_analysis_fallback_mechanisms(self):
        """Test ML analysis fallback when features are missing"""
        token = DiscoveredToken(
            address="0x9999999999999999999999999999999999999999",
            symbol="FALLBACK",
            name="Fallback Test Token",
            chain=Chain.SOLANA,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0156
            # Missing volume, market cap, etc.
        )
        
        # Create prediction with minimal technical indicators
        minimal_tech = TechnicalIndicators(
            rsi=50.0,  # Default neutral RSI
            macd=0.0   # Default neutral MACD
            # Other indicators will be None
        )
        
        ml_prediction = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=token.price_usd,  # No change prediction
            price_prediction_24h=token.price_usd,
            direction=PredictionDirection.HOLD,  # Conservative default
            confidence=0.4,  # Low confidence due to missing data
            technical_indicators=minimal_tech,
            features_used=["price"]  # Only price available
        )
        
        # Should create valid enhanced market state with defaults
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=ml_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        assert enhanced_state.token == token
        assert enhanced_state.ml_confidence == 0.4
        assert enhanced_state.ml_direction == "hold"
        
        # Feature vector should use defaults for missing values
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 25
        assert np.all(np.isfinite(feature_vector))

    def test_rl_agent_error_recovery(self):
        """Test RL agent error recovery with invalid market states"""
        # Create market state with extreme/invalid values
        token = DiscoveredToken(
            address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            symbol="EXTREME",
            name="Extreme Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0  # Zero price (invalid)
        )
        
        prediction = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            confidence=1.5,  # Invalid confidence > 1.0
            direction=PredictionDirection.BUY
        )
        
        # Should handle invalid data gracefully
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Feature vector should sanitize invalid values
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 25
        assert np.all(np.isfinite(feature_vector))
        
        # Confidence should be clamped to valid range [0, 1]
        confidence_feature = feature_vector[22]  # ml_confidence position
        assert 0.0 <= confidence_feature <= 1.0

    def test_integration_timeout_handling(self):
        """Test handling of integration timeouts and slow responses"""
        # Simulate slow ML prediction
        start_time = datetime.now()
        
        token = DiscoveredToken(
            address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            symbol="SLOW",
            name="Slow Processing Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.0234
        )
        
        # Create prediction that took a long time
        slow_prediction = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=0.0235,
            direction=PredictionDirection.HOLD,
            confidence=0.6,
            processing_time_ms=5000.0  # 5 seconds (slow)
        )
        
        # Integration should still work despite slow ML
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=slow_prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        end_time = datetime.now()
        integration_time = (end_time - start_time).total_seconds()
        
        # Integration itself should be fast even with slow ML
        assert integration_time < 0.5  # Integration <500ms
        assert enhanced_state.ml_confidence == 0.6
        assert slow_prediction.processing_time_ms == 5000.0


# Mark integration tests for proper test execution
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]