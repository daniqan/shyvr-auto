"""
Unit tests for ML-enhanced token evaluator
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken
from src.evaluation.base import EvaluationResult, EvaluationStatus, RiskLevel
from src.evaluation.ml_evaluator import MLEnhancedEvaluator
from src.ml_analysis.base import (
    ModelType, PredictionDirection, PredictionResult, TechnicalIndicators, MarketFeatures
)
from src.ml_analysis.model_manager import ModelManager


class TestMLEnhancedEvaluator:
    """Test ML-enhanced evaluator"""
    
    @pytest.fixture
    def evaluator_config(self):
        """Evaluator configuration for testing"""
        return {
            'ml_weight': 0.4,
            'confidence_threshold': 0.6,
            'risk_adjustment_factor': 1.2,
            'batch_size': 5
        }
    
    @pytest.fixture
    def mock_model_manager(self):
        """Mock model manager for testing"""
        manager = MagicMock(spec=ModelManager)
        manager.health_check = AsyncMock(return_value={
            'overall_healthy': True,
            'ensemble_available': True
        })
        manager.get_model_performance = MagicMock(return_value={
            'weights': {'lstm': 1.0},
            'performance': {'lstm': {'accuracy': 0.85}},
            'cache_stats': {'size': 0}
        })
        return manager
    
    @pytest.fixture
    def ml_evaluator(self, mock_model_manager, evaluator_config):
        """Create ML evaluator for testing"""
        return MLEnhancedEvaluator(mock_model_manager, evaluator_config)
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        return DiscoveredToken(
            address="0x123456789abcdef",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            market_cap=1000000.0,
            volume_24h=50000.0,
            price_change_24h=5.0
        )
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        prices = [1.50 + i * 0.001 for i in range(100)]
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 for i in range(100)]
        })
    
    @pytest.fixture
    def sample_ml_prediction(self, sample_token):
        """Create sample ML prediction"""
        return PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=1.55,
            price_prediction_4h=1.60,
            price_prediction_24h=1.75,
            direction=PredictionDirection.BUY,
            confidence=0.8,
            probability_up=0.75,
            upside_potential=0.15,
            downside_risk=0.05,
            model_accuracy=0.85,
            prediction_uncertainty=0.1,
            volatility_forecast=0.2,
            technical_indicators=TechnicalIndicators(rsi=65.0, volume_ratio=1.2),
            market_features=MarketFeatures(fear_greed_index=75.0, market_trend="bull")
        )
    
    def test_ml_evaluator_initialization(self, ml_evaluator, evaluator_config):
        """Test ML evaluator initialization"""
        assert ml_evaluator.ml_weight == evaluator_config['ml_weight']
        assert ml_evaluator.confidence_threshold == evaluator_config['confidence_threshold']
        assert ml_evaluator.risk_adjustment_factor == evaluator_config['risk_adjustment_factor']
        assert isinstance(ml_evaluator.model_manager, MagicMock)
    
    @pytest.mark.asyncio
    async def test_evaluate_token_success(self, ml_evaluator, sample_token, 
                                        sample_historical_data, sample_ml_prediction):
        """Test successful token evaluation"""
        # Mock ML analysis
        ml_evaluator.model_manager.analyze_token = AsyncMock(return_value=sample_ml_prediction)
        
        # Create historical data dict
        historical_data = {sample_token.address: sample_historical_data}
        
        # Evaluate token
        result = await ml_evaluator.evaluate_token(sample_token, historical_data)
        
        # Check result
        assert isinstance(result, EvaluationResult)
        assert result.token == sample_token
        assert result.status == EvaluationStatus.COMPLETED
        assert result.overall_score > 0
        assert result.is_approved is not None
        assert result.recommended_action in ["BUY", "SELL", "HOLD", "AVOID"]
        assert result.evaluation_duration_ms is not None
        
        # Check ML prediction is stored
        assert 'ml_prediction' in result.metadata
        assert 'ml_confidence' in result.metadata
        assert 'ml_direction' in result.metadata
        
        # Check that notes include ML insights
        assert len(result.notes) > 0
        ml_notes = [note for note in result.notes if 'ML model' in note]
        assert len(ml_notes) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_token_ml_failure(self, ml_evaluator, sample_token):
        """Test token evaluation when ML analysis fails"""
        # Mock ML analysis failure
        ml_evaluator.model_manager.analyze_token = AsyncMock(side_effect=Exception("ML analysis failed"))
        
        # Evaluate token
        result = await ml_evaluator.evaluate_token(sample_token)
        
        # Should still complete with fundamental analysis only
        assert result.status == EvaluationStatus.COMPLETED
        assert result.overall_score > 0
        assert 'ml_prediction' not in result.metadata
    
    @pytest.mark.asyncio
    async def test_evaluate_token_model_manager_unhealthy(self, ml_evaluator, sample_token):
        """Test evaluation when model manager is unhealthy"""
        # Mock unhealthy model manager
        ml_evaluator.model_manager.health_check = AsyncMock(return_value={
            'overall_healthy': False
        })
        
        # Evaluate token
        result = await ml_evaluator.evaluate_token(sample_token)
        
        # Should complete with fundamental analysis only
        assert result.status == EvaluationStatus.COMPLETED
        assert 'ml_prediction' not in result.metadata
    
    @pytest.mark.asyncio
    async def test_run_ml_analysis_success(self, ml_evaluator, sample_token, 
                                         sample_historical_data, sample_ml_prediction):
        """Test successful ML analysis"""
        # Mock successful analysis
        ml_evaluator.model_manager.analyze_token = AsyncMock(return_value=sample_ml_prediction)
        
        historical_data = {sample_token.address: sample_historical_data}
        
        # Run ML analysis
        result = await ml_evaluator._run_ml_analysis(sample_token, historical_data)
        
        assert result == sample_ml_prediction
        ml_evaluator.model_manager.analyze_token.assert_called_once_with(
            sample_token, sample_historical_data, use_ensemble=True
        )
    
    @pytest.mark.asyncio
    async def test_run_ml_analysis_no_data(self, ml_evaluator, sample_token, sample_ml_prediction):
        """Test ML analysis with no historical data"""
        ml_evaluator.model_manager.analyze_token = AsyncMock(return_value=sample_ml_prediction)
        
        # Run without historical data
        result = await ml_evaluator._run_ml_analysis(sample_token, None)
        
        assert result == sample_ml_prediction
        ml_evaluator.model_manager.analyze_token.assert_called_once_with(
            sample_token, None, use_ensemble=True
        )
    
    @pytest.mark.asyncio
    async def test_run_fundamental_analysis(self, ml_evaluator, sample_token):
        """Test fundamental analysis"""
        metrics = await ml_evaluator._run_fundamental_analysis(sample_token)
        
        assert metrics.price_usd == sample_token.price_usd
        assert metrics.price_change_24h == sample_token.price_change_24h
        assert metrics.market_cap == sample_token.market_cap
        assert metrics.volume_24h == sample_token.volume_24h
        assert metrics.volume_to_liquidity_ratio is not None
        assert metrics.holder_count is not None
        assert metrics.liquidity_usd is not None
    
    @pytest.mark.asyncio
    async def test_combine_analyses(self, ml_evaluator, sample_token, sample_ml_prediction):
        """Test combining ML and fundamental analyses"""
        # Create initial result
        result = EvaluationResult(
            token=sample_token,
            evaluated_at=datetime.now(),
            status=EvaluationStatus.IN_PROGRESS
        )
        
        # Create fundamental metrics
        from src.evaluation.base import FundamentalMetrics
        fundamental_metrics = FundamentalMetrics(
            price_usd=1.50,
            market_cap=1000000.0,
            volume_24h=50000.0
        )
        
        # Combine analyses
        combined_result = await ml_evaluator._combine_analyses(
            result, sample_ml_prediction, fundamental_metrics
        )
        
        assert combined_result.fundamental_metrics == fundamental_metrics
        assert combined_result.security_flags is not None
        assert 'ml_prediction' in combined_result.metadata
        assert len(combined_result.notes) > 0
    
    def test_calculate_final_evaluation(self, ml_evaluator, sample_token):
        """Test final evaluation calculation"""
        # Create result with ML prediction
        result = EvaluationResult(
            token=sample_token,
            evaluated_at=datetime.now(),
            metadata={
                'ml_prediction': {
                    'confidence': 0.8,
                    'direction': 'buy',
                    'price_prediction_24h': 1.75
                }
            }
        )
        
        # Add fundamental metrics
        from src.evaluation.base import FundamentalMetrics, SecurityFlags
        result.fundamental_metrics = FundamentalMetrics(
            price_usd=1.50,
            market_cap=1000000.0,
            volume_24h=50000.0,
            liquidity_usd=100000.0,
            holder_count=500
        )
        result.security_flags = SecurityFlags(security_score=70.0)
        
        # Calculate final evaluation
        final_result = ml_evaluator._calculate_final_evaluation(result)
        
        assert final_result.overall_score > 0
        assert final_result.security_risk >= 0
        assert final_result.liquidity_risk >= 0
        assert final_result.volatility_risk >= 0
        assert final_result.social_risk >= 0
        assert final_result.overall_risk in list(RiskLevel)
        assert final_result.recommended_action in ["BUY", "SELL", "HOLD", "AVOID"]
        assert 0 <= final_result.confidence_level <= 100
    
    def test_calculate_security_risk(self, ml_evaluator, sample_token):
        """Test security risk calculation"""
        result = EvaluationResult(token=sample_token, evaluated_at=datetime.now())
        
        # Test with no security flags
        risk = ml_evaluator._calculate_security_risk(result)
        assert risk == 80.0  # High risk if no security data
        
        # Test with security flags
        from src.evaluation.base import SecurityFlags
        result.security_flags = SecurityFlags(
            is_honeypot=False,
            contract_verified=True,
            ownership_renounced=True,
            liquidity_locked=True,
            security_score=80.0
        )
        
        risk = ml_evaluator._calculate_security_risk(result)
        assert 0 <= risk <= 100
        assert risk < 50  # Should be low risk with good security
    
    def test_calculate_liquidity_risk(self, ml_evaluator, sample_token):
        """Test liquidity risk calculation"""
        result = EvaluationResult(token=sample_token, evaluated_at=datetime.now())
        
        # Test with no fundamental metrics
        risk = ml_evaluator._calculate_liquidity_risk(result)
        assert risk == 70.0
        
        # Test with high liquidity
        from src.evaluation.base import FundamentalMetrics
        result.fundamental_metrics = FundamentalMetrics(liquidity_usd=2000000.0)
        risk = ml_evaluator._calculate_liquidity_risk(result)
        assert risk == 10.0  # Low risk for high liquidity
        
        # Test with low liquidity
        result.fundamental_metrics.liquidity_usd = 5000.0
        risk = ml_evaluator._calculate_liquidity_risk(result)
        assert risk == 80.0  # High risk for low liquidity
    
    def test_calculate_volatility_risk(self, ml_evaluator, sample_token):
        """Test volatility risk calculation"""
        result = EvaluationResult(token=sample_token, evaluated_at=datetime.now())
        
        # Test with no ML prediction
        risk = ml_evaluator._calculate_volatility_risk(result)
        assert risk == 50.0  # Default moderate risk
        
        # Test with ML prediction
        result.metadata = {
            'ml_prediction': {
                'prediction_uncertainty': 0.2,
                'volatility_forecast': 0.3
            }
        }
        risk = ml_evaluator._calculate_volatility_risk(result)
        assert 10.0 <= risk <= 90.0
    
    def test_determine_recommendation(self, ml_evaluator, sample_token):
        """Test recommendation determination"""
        result = EvaluationResult(
            token=sample_token,
            evaluated_at=datetime.now(),
            overall_score=80.0,
            overall_risk=RiskLevel.LOW
        )
        
        # Test high score, low risk
        action, confidence = ml_evaluator._determine_recommendation(result)
        assert action == "BUY"
        assert confidence > 50
        
        # Test low score
        result.overall_score = 25.0
        result.overall_risk = RiskLevel.HIGH
        action, confidence = ml_evaluator._determine_recommendation(result)
        assert action == "AVOID"
        
        # Test with ML prediction override
        result.overall_score = 60.0
        result.overall_risk = RiskLevel.MEDIUM
        result.metadata = {
            'ml_prediction': {
                'confidence': 0.9,
                'direction': 'strong_buy'
            }
        }
        action, confidence = ml_evaluator._determine_recommendation(result)
        assert action == "BUY"
        assert confidence >= 75.0
    
    @pytest.mark.asyncio
    async def test_batch_evaluate_success(self, ml_evaluator, sample_ml_prediction):
        """Test successful batch evaluation"""
        # Create multiple tokens
        tokens = []
        for i in range(8):  # More than batch size (5)
            token = DiscoveredToken(
                address=f"0x{i:040x}",
                chain=Chain.ETHEREUM,
                symbol=f"TEST{i}",
                name=f"Test Token {i}",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.1,
                market_cap=1000000.0
            )
            tokens.append(token)
        
        # Mock ML analysis
        ml_evaluator.model_manager.analyze_token = AsyncMock(return_value=sample_ml_prediction)
        
        # Run batch evaluation
        results = await ml_evaluator.batch_evaluate(tokens)
        
        assert len(results) == len(tokens)
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert all(r.status == EvaluationStatus.COMPLETED for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_evaluate_with_failures(self, ml_evaluator, sample_token):
        """Test batch evaluation with some failures"""
        tokens = [sample_token]
        
        # Mock evaluation failure
        with patch.object(ml_evaluator, 'evaluate_token', 
                         side_effect=Exception("Evaluation failed")):
            results = await ml_evaluator.batch_evaluate(tokens)
        
        assert len(results) == 1
        assert results[0].status == EvaluationStatus.FAILED
        assert len(results[0].warnings) > 0
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ml_evaluator):
        """Test successful health check"""
        health = await ml_evaluator.health_check()
        
        assert health['evaluator_healthy'] is True
        assert health['ml_models_healthy'] is True
        assert health['ml_ensemble_available'] is True
        assert 'config' in health
        assert 'model_performance' in health
        
        # Check config values
        config = health['config']
        assert config['ml_weight'] == 0.4
        assert config['confidence_threshold'] == 0.6
        assert config['risk_adjustment_factor'] == 1.2
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ml_evaluator):
        """Test health check failure"""
        # Mock health check failure
        ml_evaluator.model_manager.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        
        health = await ml_evaluator.health_check()
        
        assert health['evaluator_healthy'] is False
        assert 'error' in health
    
    def test_ml_weight_configuration(self, mock_model_manager):
        """Test different ML weight configurations"""
        # Test high ML weight
        evaluator = MLEnhancedEvaluator(mock_model_manager, {'ml_weight': 0.8})
        assert evaluator.ml_weight == 0.8
        
        # Test low ML weight
        evaluator = MLEnhancedEvaluator(mock_model_manager, {'ml_weight': 0.2})
        assert evaluator.ml_weight == 0.2
        
        # Test default weight
        evaluator = MLEnhancedEvaluator(mock_model_manager, {})
        assert evaluator.ml_weight == 0.4
    
    @pytest.mark.asyncio
    async def test_different_ml_signals(self, ml_evaluator, sample_token):
        """Test evaluation with different ML signals"""
        test_cases = [
            (PredictionDirection.STRONG_BUY, 0.9, "strong buy signal"),
            (PredictionDirection.BUY, 0.7, "buy signal"),
            (PredictionDirection.HOLD, 0.6, "hold signal"),
            (PredictionDirection.SELL, 0.7, "sell signal"),
            (PredictionDirection.STRONG_SELL, 0.9, "strong sell signal")
        ]
        
        for direction, confidence, description in test_cases:
            # Create ML prediction with specific direction
            ml_prediction = PredictionResult(
                token=sample_token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                direction=direction,
                confidence=confidence,
                price_prediction_24h=1.60
            )
            
            # Mock ML analysis
            ml_evaluator.model_manager.analyze_token = AsyncMock(return_value=ml_prediction)
            
            # Evaluate token
            result = await ml_evaluator.evaluate_token(sample_token)
            
            # Check that ML signal is reflected in results
            assert result.status == EvaluationStatus.COMPLETED
            assert 'ml_prediction' in result.metadata
            assert result.metadata['ml_direction'] == direction.value
            
            # Check notes mention the ML signal
            ml_notes = [note for note in result.notes if 'ML model' in note.lower()]
            assert len(ml_notes) > 0, f"No ML notes found for {description}"


class TestMLEvaluatorIntegration:
    """Integration tests for ML evaluator with real components"""
    
    @pytest.mark.asyncio
    async def test_integration_with_real_model_manager(self):
        """Test integration with actual model manager (mocked models)"""
        # Create temporary directory for models
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model manager config
            manager_config = {
                'model_dir': temp_dir,
                'cache_ttl_minutes': 5,
                'lstm': {
                    'sequence_length': 10,
                    'hidden_size': 16,
                    'num_epochs': 1
                }
            }
            
            # Create real model manager
            model_manager = ModelManager(manager_config)
            
            # Create evaluator
            evaluator = MLEnhancedEvaluator(model_manager)
            
            # Create sample token
            token = DiscoveredToken(
                address="0x123456789abcdef",
                chain=Chain.ETHEREUM,
                symbol="TEST",
                name="Test Token",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.50,
                market_cap=1000000.0
            )
            
            # Evaluate token (will use untrained models, should fallback gracefully)
            result = await evaluator.evaluate_token(token)
            
            # Should complete successfully even with untrained models
            assert isinstance(result, EvaluationResult)
            assert result.status == EvaluationStatus.COMPLETED
            assert result.token == token
    
    @pytest.mark.asyncio
    async def test_error_handling_comprehensive(self):
        """Test comprehensive error handling"""
        # Create evaluator with failing model manager
        failing_manager = MagicMock()
        failing_manager.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        failing_manager.analyze_token = AsyncMock(side_effect=Exception("Analysis failed"))
        
        evaluator = MLEnhancedEvaluator(failing_manager)
        
        token = DiscoveredToken(
            address="0x123456789abcdef",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        # Should handle all failures gracefully
        result = await evaluator.evaluate_token(token)
        
        # Should not fail completely, but fall back to fundamental analysis
        assert isinstance(result, EvaluationResult)
        assert result.token == token