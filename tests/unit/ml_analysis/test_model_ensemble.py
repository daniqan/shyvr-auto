"""
Unit tests for Model Ensemble functionality
Tests ensemble prediction, model loading, and combination strategies
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.ml_analysis.model_ensemble import (
    EnsemblePredictor, EnsembleConfig, EnsembleStrategy, EnsemblePrediction
)
from src.ml_analysis.base import (
    PredictionResult, ModelType, PredictionDirection, TechnicalIndicators, MarketFeatures
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


@pytest.fixture
def sample_token():
    """Create a sample token for testing"""
    return DiscoveredToken(
        chain=Chain.ETHEREUM,
        address="0x1234567890123456789012345678901234567890",
        symbol="TEST",
        name="Test Token",
        price_usd=100.0,
        market_cap=1000000.0,
        volume_24h=50000.0
    )


@pytest.fixture
def sample_lstm_prediction(sample_token):
    """Create a sample LSTM prediction"""
    return PredictionResult(
        token=sample_token,
        analyzed_at=datetime.now(),
        model_type=ModelType.LSTM,
        price_prediction_1h=101.0,
        price_prediction_4h=102.0,
        price_prediction_24h=105.0,
        direction=PredictionDirection.BUY,
        confidence=0.8,
        probability_up=0.7,
        model_accuracy=0.75,
        technical_indicators=TechnicalIndicators(sma_20=99.0, rsi=60.0),
        market_features=MarketFeatures(fear_greed_index=65.0),
        features_used=["price", "volume", "technical"],
        model_version="lstm_1.0"
    )


@pytest.fixture
def sample_transformer_prediction(sample_token):
    """Create a sample Transformer prediction"""
    return PredictionResult(
        token=sample_token,
        analyzed_at=datetime.now(),
        model_type=ModelType.TRANSFORMER,
        price_prediction_1h=99.0,
        price_prediction_4h=98.0,
        price_prediction_24h=95.0,
        direction=PredictionDirection.SELL,
        confidence=0.6,
        probability_up=0.3,
        model_accuracy=0.65,
        technical_indicators=TechnicalIndicators(sma_20=101.0, rsi=40.0),
        market_features=MarketFeatures(fear_greed_index=35.0),
        features_used=["price", "volume", "attention"],
        model_version="transformer_1.0"
    )


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'price': 100 + np.random.randn(1000) * 5,
        'volume': 1000 + np.random.randn(1000) * 100,
        'returns': np.random.randn(1000) * 0.02
    }
    return pd.DataFrame(data)


class TestEnsembleConfig:
    """Test EnsembleConfig functionality"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EnsembleConfig()

        assert config.lstm_weight == 0.6
        assert config.transformer_weight == 0.4
        assert config.strategy == EnsembleStrategy.WEIGHTED_AVERAGE
        assert config.checkpoint_dir == "tmp/checkpoints"
        assert config.meta_learner_type == "random_forest"

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0"""
        config = EnsembleConfig(lstm_weight=0.8, transformer_weight=0.6)

        # Should be normalized to sum to 1.0
        assert abs(config.lstm_weight + config.transformer_weight - 1.0) < 1e-6
        assert abs(config.lstm_weight - 0.8/1.4) < 1e-6  # 0.8 / (0.8 + 0.6)
        assert abs(config.transformer_weight - 0.6/1.4) < 1e-6

    def test_custom_meta_learner_params(self):
        """Test custom meta-learner parameters"""
        custom_params = {"n_estimators": 200, "max_depth": 10}
        config = EnsembleConfig(meta_learner_params=custom_params)

        assert config.meta_learner_params == custom_params


class TestEnsemblePredictor:
    """Test EnsemblePredictor functionality"""

    @pytest.fixture
    def ensemble_config(self):
        """Create test ensemble configuration"""
        return EnsembleConfig(
            lstm_weight=0.7,
            transformer_weight=0.3,
            strategy=EnsembleStrategy.WEIGHTED_AVERAGE
        )

    @pytest.fixture
    def ensemble_predictor(self, ensemble_config):
        """Create ensemble predictor with mocked models"""
        with patch('src.ml_analysis.model_ensemble.LSTMPricePredictor') as mock_lstm, \
             patch('src.ml_analysis.model_ensemble.TransformerPredictor') as mock_transformer:

            # Mock the models
            mock_lstm_instance = Mock()
            mock_lstm_instance.is_model_trained.return_value = True
            mock_lstm.return_value = mock_lstm_instance

            mock_transformer_instance = Mock()
            mock_transformer_instance.is_model_trained.return_value = True
            mock_transformer.return_value = mock_transformer_instance

            predictor = EnsemblePredictor(ensemble_config)
            predictor._models_loaded = True
            predictor._is_trained = True

            return predictor

    @pytest.mark.asyncio
    async def test_simple_average_combination(self, ensemble_predictor, sample_token,
                                            sample_lstm_prediction, sample_transformer_prediction):
        """Test simple average combination strategy"""
        ensemble_predictor.ensemble_config.strategy = EnsembleStrategy.SIMPLE_AVERAGE

        result = await ensemble_predictor._simple_average_combination(
            sample_token, sample_lstm_prediction, sample_transformer_prediction
        )

        assert isinstance(result, EnsemblePrediction)
        assert result.strategy_used == EnsembleStrategy.SIMPLE_AVERAGE

        # Check averaged predictions
        expected_1h = (101.0 + 99.0) / 2
        expected_4h = (102.0 + 98.0) / 2
        expected_24h = (105.0 + 95.0) / 2

        assert abs(result.ensemble_result.price_prediction_1h - expected_1h) < 1e-6
        assert abs(result.ensemble_result.price_prediction_4h - expected_4h) < 1e-6
        assert abs(result.ensemble_result.price_prediction_24h - expected_24h) < 1e-6

        # Check averaged confidence
        expected_confidence = (0.8 + 0.6) / 2
        assert abs(result.ensemble_result.confidence - expected_confidence) < 1e-6

    @pytest.mark.asyncio
    async def test_weighted_average_combination(self, ensemble_predictor, sample_token,
                                              sample_lstm_prediction, sample_transformer_prediction):
        """Test weighted average combination strategy"""
        # Config has lstm_weight=0.7, transformer_weight=0.3
        result = await ensemble_predictor._weighted_average_combination(
            sample_token, sample_lstm_prediction, sample_transformer_prediction
        )

        assert isinstance(result, EnsemblePrediction)
        assert result.strategy_used == EnsembleStrategy.WEIGHTED_AVERAGE

        # Check weighted predictions
        expected_1h = 101.0 * 0.7 + 99.0 * 0.3
        expected_4h = 102.0 * 0.7 + 98.0 * 0.3
        expected_24h = 105.0 * 0.7 + 95.0 * 0.3

        assert abs(result.ensemble_result.price_prediction_1h - expected_1h) < 1e-6
        assert abs(result.ensemble_result.price_prediction_4h - expected_4h) < 1e-6
        assert abs(result.ensemble_result.price_prediction_24h - expected_24h) < 1e-6

        # Check weighted confidence
        expected_confidence = 0.8 * 0.7 + 0.6 * 0.3
        assert abs(result.ensemble_result.confidence - expected_confidence) < 1e-6

    @pytest.mark.asyncio
    async def test_confidence_weighted_combination(self, ensemble_predictor, sample_token,
                                                 sample_lstm_prediction, sample_transformer_prediction):
        """Test confidence-weighted combination strategy"""
        result = await ensemble_predictor._confidence_weighted_combination(
            sample_token, sample_lstm_prediction, sample_transformer_prediction
        )

        assert isinstance(result, EnsemblePrediction)
        assert result.strategy_used == EnsembleStrategy.CONFIDENCE_WEIGHTED

        # Weights should be based on confidence: 0.8 and 0.6, normalized
        total_conf = 0.8 + 0.6
        lstm_weight = 0.8 / total_conf
        transformer_weight = 0.6 / total_conf

        expected_1h = 101.0 * lstm_weight + 99.0 * transformer_weight
        assert abs(result.ensemble_result.price_prediction_1h - expected_1h) < 1e-6

    @pytest.mark.asyncio
    async def test_single_model_prediction(self, ensemble_predictor, sample_token, sample_lstm_prediction):
        """Test ensemble with only one model available"""
        result = await ensemble_predictor._weighted_average_combination(
            sample_token, sample_lstm_prediction, None
        )

        assert isinstance(result, EnsemblePrediction)

        # Should return LSTM prediction unchanged
        assert result.ensemble_result.price_prediction_1h == 101.0
        assert result.ensemble_result.price_prediction_4h == 102.0
        assert result.ensemble_result.price_prediction_24h == 105.0
        assert result.ensemble_result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_no_predictions_error(self, ensemble_predictor, sample_token):
        """Test error handling when no predictions available"""
        with pytest.raises(Exception):
            await ensemble_predictor._simple_average_combination(sample_token, None, None)

    def test_prediction_agreement_calculation(self, ensemble_predictor,
                                            sample_lstm_prediction, sample_transformer_prediction):
        """Test prediction agreement calculation"""
        predictions = [sample_lstm_prediction, sample_transformer_prediction]
        agreement = ensemble_predictor._calculate_prediction_agreement(predictions)

        assert 0.0 <= agreement <= 1.0

        # Single prediction should have perfect agreement
        single_agreement = ensemble_predictor._calculate_prediction_agreement([sample_lstm_prediction])
        assert single_agreement == 1.0

    def test_direction_determination(self, ensemble_predictor):
        """Test ensemble direction determination"""
        # Test strong buy
        direction = ensemble_predictor._determine_ensemble_direction(110.0, 100.0)
        assert direction == PredictionDirection.STRONG_BUY

        # Test buy
        direction = ensemble_predictor._determine_ensemble_direction(103.0, 100.0)
        assert direction == PredictionDirection.BUY

        # Test hold
        direction = ensemble_predictor._determine_ensemble_direction(101.0, 100.0)
        assert direction == PredictionDirection.HOLD

        # Test sell
        direction = ensemble_predictor._determine_ensemble_direction(97.0, 100.0)
        assert direction == PredictionDirection.SELL

        # Test strong sell
        direction = ensemble_predictor._determine_ensemble_direction(85.0, 100.0)
        assert direction == PredictionDirection.STRONG_SELL

    def test_weight_and_strategy_setters(self, ensemble_predictor):
        """Test weight and strategy setter methods"""
        # Test weight setting
        ensemble_predictor.set_weights(0.3, 0.7)
        assert abs(ensemble_predictor.ensemble_config.lstm_weight - 0.3) < 1e-6
        assert abs(ensemble_predictor.ensemble_config.transformer_weight - 0.7) < 1e-6

        # Test strategy setting
        ensemble_predictor.set_strategy(EnsembleStrategy.CONFIDENCE_WEIGHTED)
        assert ensemble_predictor.ensemble_config.strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED

    @pytest.mark.asyncio
    async def test_analyze_token_integration(self, sample_token, sample_lstm_prediction,
                                           sample_transformer_prediction):
        """Test full analyze_token integration"""
        with patch('src.ml_analysis.model_ensemble.LSTMPricePredictor') as mock_lstm, \
             patch('src.ml_analysis.model_ensemble.TransformerPredictor') as mock_transformer, \
             patch('src.activity_logging.activity_logger.activity_logger') as mock_logger:

            # Setup mocks
            mock_lstm_instance = AsyncMock()
            mock_lstm_instance.is_model_trained.return_value = True
            mock_lstm_instance.analyze_token.return_value = sample_lstm_prediction
            mock_lstm.return_value = mock_lstm_instance

            mock_transformer_instance = AsyncMock()
            mock_transformer_instance.is_model_trained.return_value = True
            mock_transformer_instance.analyze_token.return_value = sample_transformer_prediction
            mock_transformer.return_value = mock_transformer_instance

            mock_logger.log_activity = AsyncMock()

            # Create ensemble
            config = EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
            ensemble = EnsemblePredictor(config)
            ensemble._models_loaded = True
            ensemble._is_trained = True

            # Test analyze_token
            result = await ensemble.analyze_token(sample_token)

            assert isinstance(result, PredictionResult)
            assert result.model_type == ModelType.ENSEMBLE
            assert result.token == sample_token

            # Verify models were called
            mock_lstm_instance.analyze_token.assert_called_once_with(sample_token, None)
            mock_transformer_instance.analyze_token.assert_called_once_with(sample_token, None)

            # Verify activity logging
            assert mock_logger.log_activity.call_count >= 1

    @pytest.mark.asyncio
    async def test_model_loading_with_checkpoints(self):
        """Test model loading from checkpoints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake checkpoint files
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            lstm_checkpoint = checkpoint_dir / "lstm_best.pt"
            transformer_checkpoint = checkpoint_dir / "transformer_best.pt"

            # Create fake checkpoint data
            torch.save({"model_state": "fake_state"}, lstm_checkpoint)
            torch.save({"model_state": "fake_state"}, transformer_checkpoint)

            with patch('src.ml_analysis.model_ensemble.LSTMPricePredictor') as mock_lstm, \
                 patch('src.ml_analysis.model_ensemble.TransformerPredictor') as mock_transformer:

                # Setup mocks
                mock_lstm_instance = Mock()
                mock_lstm_instance.load_model.return_value = True
                mock_lstm.return_value = mock_lstm_instance

                mock_transformer_instance = Mock()
                mock_transformer_instance.load_model.return_value = True
                mock_transformer.return_value = mock_transformer_instance

                # Create ensemble with custom checkpoint dir
                config = EnsembleConfig(checkpoint_dir=str(checkpoint_dir))
                ensemble = EnsemblePredictor(config)

                # Test loading
                results = await ensemble.load_models()

                assert results["lstm"] == True
                assert results["transformer"] == True
                assert ensemble._models_loaded == True

    @pytest.mark.asyncio
    async def test_training_functionality(self, sample_training_data):
        """Test ensemble training functionality"""
        with patch('src.ml_analysis.model_ensemble.LSTMPricePredictor') as mock_lstm, \
             patch('src.ml_analysis.model_ensemble.TransformerPredictor') as mock_transformer:

            # Setup mocks
            mock_lstm_instance = AsyncMock()
            mock_lstm_instance.train_model.return_value = True
            mock_lstm.return_value = mock_lstm_instance

            mock_transformer_instance = AsyncMock()
            mock_transformer_instance.train_model.return_value = True
            mock_transformer.return_value = mock_transformer_instance

            # Create ensemble
            config = EnsembleConfig(strategy=EnsembleStrategy.STACKING)
            ensemble = EnsemblePredictor(config)

            # Test training
            success = await ensemble.train_model(sample_training_data)

            assert success == True
            assert ensemble._is_trained == True
            assert ensemble._models_loaded == True

            # Verify individual model training was called
            mock_lstm_instance.train_model.assert_called_once_with(sample_training_data)
            mock_transformer_instance.train_model.assert_called_once_with(sample_training_data)

    @pytest.mark.asyncio
    async def test_health_check(self, ensemble_predictor):
        """Test ensemble health check"""
        # Mock healthy models
        ensemble_predictor._lstm_model.health_check = AsyncMock(return_value=True)
        ensemble_predictor._transformer_model.health_check = AsyncMock(return_value=True)

        health = await ensemble_predictor.health_check()
        assert health == True

        # Test with one unhealthy model
        ensemble_predictor._lstm_model.health_check = AsyncMock(return_value=False)
        health = await ensemble_predictor.health_check()
        assert health == True  # Still healthy if one model works

        # Test with both unhealthy
        ensemble_predictor._transformer_model.health_check = AsyncMock(return_value=False)
        health = await ensemble_predictor.health_check()
        assert health == False

    def test_get_performance_metrics(self, ensemble_predictor):
        """Test performance metrics retrieval"""
        metrics = ensemble_predictor.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "strategy" in metrics
        assert "models_loaded" in metrics
        assert "lstm_weight" in metrics
        assert "transformer_weight" in metrics
        assert "lstm_trained" in metrics
        assert "transformer_trained" in metrics

    def test_get_required_features(self, ensemble_predictor):
        """Test required features aggregation"""
        # Mock feature requirements
        ensemble_predictor._lstm_model.get_required_features.return_value = ["price", "volume"]
        ensemble_predictor._transformer_model.get_required_features.return_value = ["price", "attention"]

        features = ensemble_predictor.get_required_features()

        assert isinstance(features, list)
        assert "price" in features
        assert len(set(features)) >= 2  # Should have unique features from both models


class TestEnsembleHelperFunctions:
    """Test helper functions in ensemble module"""

    @pytest.fixture
    def ensemble_predictor(self):
        """Create basic ensemble predictor for testing"""
        with patch('src.ml_analysis.model_ensemble.LSTMPricePredictor'), \
             patch('src.ml_analysis.model_ensemble.TransformerPredictor'):
            return EnsemblePredictor()

    def test_average_values(self, ensemble_predictor):
        """Test _average_values helper function"""
        # Test with valid values
        result = ensemble_predictor._average_values([1.0, 2.0, 3.0])
        assert abs(result - 2.0) < 1e-6

        # Test with None values
        result = ensemble_predictor._average_values([1.0, None, 3.0])
        assert abs(result - 2.0) < 1e-6

        # Test with all None values
        result = ensemble_predictor._average_values([None, None, None])
        assert result is None

    def test_weighted_average_values(self, ensemble_predictor):
        """Test _weighted_average_values helper function"""
        # Test with valid values and weights
        result = ensemble_predictor._weighted_average_values([1.0, 2.0, 3.0], [0.5, 0.3, 0.2])
        expected = (1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(result - expected) < 1e-6

        # Test with None values
        result = ensemble_predictor._weighted_average_values([1.0, None, 3.0], [0.5, 0.3, 0.2])
        expected = (1.0 * 0.5 + 3.0 * 0.2) / (0.5 + 0.2)
        assert abs(result - expected) < 1e-6

        # Test with all None values
        result = ensemble_predictor._weighted_average_values([None, None], [0.5, 0.5])
        assert result is None

    def test_prepare_meta_features(self, ensemble_predictor, sample_lstm_prediction,
                                 sample_transformer_prediction, sample_token):
        """Test meta-feature preparation for stacking"""
        predictions = [sample_lstm_prediction, sample_transformer_prediction]
        features = ensemble_predictor._prepare_meta_features(predictions, sample_token)

        assert isinstance(features, list)
        assert len(features) == 30  # Target size
        assert all(isinstance(f, float) for f in features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])