"""
Unit tests for ML analysis base classes and data structures
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken, TokenStatus
from src.ml_analysis.base import (
    ModelType, PredictionDirection, TechnicalIndicators, MarketFeatures,
    PredictionResult, MLAnalyzerBase, MLAnalysisError, ModelNotTrainedError,
    FeatureEngineeringError, PredictionError
)


class TestTechnicalIndicators:
    """Test TechnicalIndicators data class"""
    
    def test_technical_indicators_creation(self):
        """Test basic technical indicators creation"""
        indicators = TechnicalIndicators(
            sma_20=100.0,
            rsi=65.0,
            macd=2.5,
            volume_ratio=1.2
        )
        
        assert indicators.sma_20 == 100.0
        assert indicators.rsi == 65.0
        assert indicators.macd == 2.5
        assert indicators.volume_ratio == 1.2
        
        # Default values
        assert indicators.sma_50 is None
        assert indicators.atr is None
    
    def test_to_feature_vector(self):
        """Test conversion to feature vector"""
        indicators = TechnicalIndicators(
            sma_20=100.0,
            sma_50=105.0,
            rsi=65.0,
            macd=2.5,
            volume_ratio=1.2,
            price_momentum=5.0
        )
        
        vector = indicators.to_feature_vector()
        
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == 17  # All indicator features
        
        # Check specific values
        assert vector[0] == 100.0  # sma_20
        assert vector[1] == 105.0  # sma_50
        assert vector[4] == 65.0   # rsi
        assert vector[13] == 1.2   # volume_ratio
        assert vector[15] == 5.0   # price_momentum
    
    def test_feature_vector_with_defaults(self):
        """Test feature vector with default values"""
        indicators = TechnicalIndicators()
        vector = indicators.to_feature_vector()
        
        # Should use default values
        assert vector[0] == 0.0    # sma_20 default
        assert vector[4] == 50.0   # rsi default
        assert vector[13] == 1.0   # volume_ratio default


class TestMarketFeatures:
    """Test MarketFeatures data class"""
    
    def test_market_features_creation(self):
        """Test basic market features creation"""
        features = MarketFeatures(
            fear_greed_index=75.0,
            market_trend="bull",
            btc_correlation=0.8,
            social_score=0.7,
            mention_volume=500
        )
        
        assert features.fear_greed_index == 75.0
        assert features.market_trend == "bull"
        assert features.btc_correlation == 0.8
        assert features.social_score == 0.7
        assert features.mention_volume == 500
    
    def test_to_feature_vector(self):
        """Test conversion to feature vector"""
        features = MarketFeatures(
            fear_greed_index=75.0,
            market_trend="bull",
            volatility_regime="high",
            btc_correlation=0.8,
            eth_correlation=0.6,
            market_beta=1.2,
            social_score=0.7,
            mention_volume=500,
            sentiment_trend=0.1
        )
        
        vector = features.to_feature_vector()
        
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == 9  # All market features
        
        # Check specific values
        assert vector[0] == 75.0  # fear_greed_index
        assert vector[1] == 1.0   # is_bull_market
        assert vector[2] == 1.0   # is_high_volatility
        assert vector[3] == 0.8   # btc_correlation
        assert vector[6] == 0.7   # social_score
        assert vector[7] == 0.5   # mention_volume normalized (500/1000)


class TestPredictionResult:
    """Test PredictionResult data class"""
    
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
            price_usd=1.50
        )
    
    @pytest.fixture
    def sample_technical_indicators(self):
        """Create sample technical indicators"""
        return TechnicalIndicators(
            sma_20=100.0,
            rsi=65.0,
            macd=2.5,
            volume_ratio=1.2
        )
    
    @pytest.fixture
    def sample_market_features(self):
        """Create sample market features"""
        return MarketFeatures(
            fear_greed_index=75.0,
            market_trend="bull",
            btc_correlation=0.8
        )
    
    def test_prediction_result_creation(self, sample_token, sample_technical_indicators, sample_market_features):
        """Test basic prediction result creation"""
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=1.55,
            price_prediction_4h=1.60,
            price_prediction_24h=1.75,
            direction=PredictionDirection.BUY,
            confidence=0.8,
            probability_up=0.75,
            technical_indicators=sample_technical_indicators,
            market_features=sample_market_features
        )
        
        assert result.token == sample_token
        assert result.model_type == ModelType.LSTM
        assert result.price_prediction_1h == 1.55
        assert result.price_prediction_24h == 1.75
        assert result.direction == PredictionDirection.BUY
        assert result.confidence == 0.8
        assert result.probability_up == 0.75
        assert result.technical_indicators == sample_technical_indicators
        assert result.market_features == sample_market_features
    
    def test_get_expected_return(self, sample_token):
        """Test expected return calculation"""
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=1.65,  # 10% increase
            price_prediction_24h=1.80  # 20% increase
        )
        
        # Test 1h return
        return_1h = result.get_expected_return("1h")
        assert abs(return_1h - 0.1) < 0.001  # 10% return
        
        # Test 24h return  
        return_24h = result.get_expected_return("24h")
        assert abs(return_24h - 0.2) < 0.001  # 20% return
        
        # Test invalid timeframe
        invalid_return = result.get_expected_return("7d")
        assert invalid_return is None
    
    def test_get_expected_return_no_price(self):
        """Test expected return with no token price"""
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=None  # No price
        )
        
        result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_24h=1.80
        )
        
        expected_return = result.get_expected_return("24h")
        assert expected_return is None
    
    def test_get_risk_reward_ratio(self, sample_token):
        """Test risk/reward ratio calculation"""
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            upside_potential=0.20,  # 20% upside
            downside_risk=0.10      # 10% downside
        )
        
        ratio = result.get_risk_reward_ratio()
        assert ratio == 2.0  # 20% / 10% = 2:1
    
    def test_get_risk_reward_ratio_edge_cases(self, sample_token):
        """Test risk/reward ratio edge cases"""
        # Zero downside risk
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            upside_potential=0.20,
            downside_risk=0.0
        )
        
        ratio = result.get_risk_reward_ratio()
        assert ratio == float('inf')
        
        # No risk/reward data
        result_empty = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM
        )
        
        ratio_empty = result_empty.get_risk_reward_ratio()
        assert ratio_empty is None
    
    def test_is_buy_signal(self, sample_token):
        """Test buy signal detection"""
        # Strong buy signal
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.STRONG_BUY,
            confidence=0.8,
            probability_up=0.75
        )
        
        assert result.is_buy_signal()
        assert result.is_buy_signal(min_confidence=0.7)
        assert not result.is_buy_signal(min_confidence=0.9)  # Too high confidence requirement
        
        # Regular buy signal
        result_buy = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.BUY,
            confidence=0.7,
            probability_up=0.65
        )
        
        assert result_buy.is_buy_signal()
        
        # Not a buy signal (hold)
        result_hold = PredictionResult(
            token=sample_token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.HOLD,
            confidence=0.8,
            probability_up=0.55
        )
        
        assert not result_hold.is_buy_signal()
    
    def test_to_dict(self, sample_token):
        """Test dictionary serialization"""
        result = PredictionResult(
            token=sample_token,
            analyzed_at=datetime(2025, 1, 1, 12, 0, 0),
            model_type=ModelType.LSTM,
            price_prediction_1h=1.55,
            price_prediction_24h=1.75,
            direction=PredictionDirection.BUY,
            confidence=0.8,
            entry_signal="BUY",
            stop_loss_level=1.40,
            features_used=["rsi", "macd"],
            model_version="1.0.0"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["token_address"] == "0x123456789abcdef"
        assert result_dict["chain"] == "ethereum"
        assert result_dict["analyzed_at"] == "2025-01-01T12:00:00"
        assert result_dict["model_type"] == "lstm"
        assert result_dict["price_prediction_1h"] == 1.55
        assert result_dict["price_prediction_24h"] == 1.75
        assert result_dict["direction"] == "buy"
        assert result_dict["confidence"] == 0.8
        assert result_dict["entry_signal"] == "BUY"
        assert result_dict["stop_loss_level"] == 1.40
        assert result_dict["features_used"] == ["rsi", "macd"]
        assert result_dict["model_version"] == "1.0.0"


class MockMLAnalyzer(MLAnalyzerBase):
    """Mock ML analyzer for testing base class"""
    
    def __init__(self, should_fail=False):
        super().__init__(ModelType.LSTM)
        self.should_fail = should_fail
        self._is_trained = True  # Mock as trained
        self._model = MagicMock()  # Mock model
    
    async def analyze_token(self, token, historical_data=None):
        if self.should_fail:
            raise PredictionError("Mock analysis failed")
        
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=self.model_type,
            direction=PredictionDirection.BUY,
            confidence=0.7
        )
    
    async def train_model(self, training_data):
        return not self.should_fail
    
    def get_required_features(self):
        return ["close", "volume", "rsi"]


class TestMLAnalyzerBase:
    """Test MLAnalyzerBase abstract class"""
    
    @pytest.mark.asyncio
    async def test_batch_analyze_success(self):
        """Test successful batch analysis"""
        analyzer = MockMLAnalyzer()
        
        tokens = [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test"
            ),
            DiscoveredToken(
                address="0x456",
                chain=Chain.ETHEREUM,
                symbol="TEST2",
                name="Test Token 2",
                discovered_at=datetime.now(),
                discovery_source="test"
            )
        ]
        
        results = await analyzer.batch_analyze(tokens)
        
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)
        assert results[0].token.address == "0x123"
        assert results[1].token.address == "0x456"
    
    @pytest.mark.asyncio
    async def test_batch_analyze_with_failures(self):
        """Test batch analysis with some failures"""
        analyzer = MockMLAnalyzer(should_fail=True)
        
        tokens = [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test"
            )
        ]
        
        results = await analyzer.batch_analyze(tokens)
        
        # Should still return a result even if analysis fails
        assert len(results) == 1
        assert results[0].confidence == 0.0
        assert "error" in results[0].features_used
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality"""
        analyzer = MockMLAnalyzer()
        
        # Should be healthy when trained and model exists
        is_healthy = await analyzer.health_check()
        assert is_healthy
        
        # Should be unhealthy when not trained
        analyzer._is_trained = False
        is_healthy = await analyzer.health_check()
        assert not is_healthy
    
    def test_is_model_trained(self):
        """Test model training status check"""
        analyzer = MockMLAnalyzer()
        
        assert analyzer.is_model_trained()
        
        analyzer._is_trained = False
        assert not analyzer.is_model_trained()
    
    def test_get_required_features(self):
        """Test required features specification"""
        analyzer = MockMLAnalyzer()
        features = analyzer.get_required_features()
        
        assert isinstance(features, list)
        assert "close" in features
        assert "volume" in features
        assert "rsi" in features


class TestMLAnalysisExceptions:
    """Test ML analysis exception classes"""
    
    def test_ml_analysis_error(self):
        """Test base MLAnalysisError"""
        error = MLAnalysisError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_model_not_trained_error(self):
        """Test ModelNotTrainedError inheritance"""
        error = ModelNotTrainedError("Model not trained")
        assert str(error) == "Model not trained"
        assert isinstance(error, MLAnalysisError)
        assert isinstance(error, Exception)
    
    def test_feature_engineering_error(self):
        """Test FeatureEngineeringError inheritance"""
        error = FeatureEngineeringError("Feature engineering failed")
        assert str(error) == "Feature engineering failed"
        assert isinstance(error, MLAnalysisError)
    
    def test_prediction_error(self):
        """Test PredictionError inheritance"""
        error = PredictionError("Prediction failed")
        assert str(error) == "Prediction failed"
        assert isinstance(error, MLAnalysisError)


class TestModelTypeEnum:
    """Test ModelType enumeration"""
    
    def test_model_type_values(self):
        """Test ModelType enum values"""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.TRANSFORMER.value == "transformer"
        assert ModelType.LINEAR_REGRESSION.value == "linear_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.ENSEMBLE.value == "ensemble"
    
    def test_model_type_membership(self):
        """Test ModelType enum membership"""
        assert ModelType.LSTM in ModelType
        assert "invalid_model" not in [mt.value for mt in ModelType]


class TestPredictionDirectionEnum:
    """Test PredictionDirection enumeration"""
    
    def test_prediction_direction_values(self):
        """Test PredictionDirection enum values"""
        assert PredictionDirection.STRONG_BUY.value == "strong_buy"
        assert PredictionDirection.BUY.value == "buy"
        assert PredictionDirection.HOLD.value == "hold"
        assert PredictionDirection.SELL.value == "sell"
        assert PredictionDirection.STRONG_SELL.value == "strong_sell"
    
    def test_prediction_direction_ordering(self):
        """Test prediction direction logical ordering"""
        directions = [
            PredictionDirection.STRONG_SELL,
            PredictionDirection.SELL,
            PredictionDirection.HOLD,
            PredictionDirection.BUY,
            PredictionDirection.STRONG_BUY
        ]
        
        # All should be unique
        assert len(set(directions)) == len(directions)
        
        # Verify they're all valid enum members
        for direction in directions:
            assert direction in PredictionDirection