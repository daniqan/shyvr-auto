"""
Unit tests for model manager
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken
from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import (
    ModelType, PredictionDirection, PredictionResult,
    MLAnalysisError
)


class TestModelManager:
    """Test ModelManager class"""
    
    @pytest.fixture
    def manager_config(self):
        """Model manager configuration for testing"""
        return {
            'cache_ttl_minutes': 5,
            'lstm': {
                'sequence_length': 20,
                'hidden_size': 32,
                'num_epochs': 3  # Fewer epochs for tests
            }
        }
    
    @pytest.fixture
    def model_manager(self, manager_config):
        """Create model manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager_config['model_dir'] = temp_dir
            return ModelManager(manager_config)
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        return [
            DiscoveredToken(
                address="0x123456789abcdef",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=100.0,
                market_cap=1000000.0
            ),
            DiscoveredToken(
                address="0x987654321fedcba",
                chain=Chain.SOLANA,
                symbol="TEST2", 
                name="Test Token 2",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=50.0,
                market_cap=500000.0
            )
        ]
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data for testing"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
        prices = [100.0 + i * 0.1 for i in range(100)]  # Slowly increasing
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 for i in range(100)]
        })
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization"""
        assert isinstance(model_manager._models, dict)
        assert ModelType.LSTM in model_manager._models
        assert isinstance(model_manager._model_weights, dict)
        assert isinstance(model_manager._model_performance, dict)
        assert len(model_manager._ensemble_cache) == 0
    
    def test_model_weights_initialization(self, model_manager):
        """Test model weights are properly initialized"""
        assert ModelType.LSTM in model_manager._model_weights
        assert model_manager._model_weights[ModelType.LSTM] == 1.0
        
        # Check performance tracking initialization
        assert ModelType.LSTM in model_manager._model_performance
        perf = model_manager._model_performance[ModelType.LSTM]
        assert perf['accuracy'] == 0.0
        assert perf['predictions_made'] == 0
        assert 'last_updated' in perf
    
    @pytest.mark.asyncio
    async def test_analyze_token_single_model(self, model_manager, sample_tokens, sample_historical_data):
        """Test token analysis with single model"""
        token = sample_tokens[0]
        
        # Mock the LSTM model to return a prediction
        mock_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.BUY,
            confidence=0.8,
            price_prediction_24h=110.0
        )
        
        with patch.object(model_manager._models[ModelType.LSTM], 'analyze_token', return_value=mock_result):
            result = await model_manager.analyze_token(token, sample_historical_data, use_ensemble=False)
            
            assert isinstance(result, PredictionResult)
            assert result.token == token
            assert result.direction == PredictionDirection.BUY
            assert result.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_token_caching(self, model_manager, sample_tokens, sample_historical_data):
        """Test token analysis caching"""
        token = sample_tokens[0]
        
        mock_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.BUY,
            confidence=0.8
        )
        
        with patch.object(model_manager._models[ModelType.LSTM], 'analyze_token', return_value=mock_result) as mock_analyze:
            # First call
            result1 = await model_manager.analyze_token(token, sample_historical_data, use_ensemble=False)
            
            # Second call should use cache
            result2 = await model_manager.analyze_token(token, sample_historical_data, use_ensemble=False)
            
            # Should only call analyze_token once due to caching
            assert mock_analyze.call_count == 1
            assert result1 is result2  # Same cached object
    
    @pytest.mark.asyncio
    async def test_batch_analyze_success(self, model_manager, sample_tokens, sample_historical_data):
        """Test successful batch analysis"""
        mock_results = []
        for token in sample_tokens:
            mock_results.append(PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                direction=PredictionDirection.BUY,
                confidence=0.7
            ))
        
        historical_data = {token.address: sample_historical_data for token in sample_tokens}
        
        with patch.object(model_manager, 'analyze_token', side_effect=mock_results):
            results = await model_manager.batch_analyze(sample_tokens, historical_data)
            
            assert len(results) == len(sample_tokens)
            assert all(isinstance(r, PredictionResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_analyze_with_failures(self, model_manager, sample_tokens):
        """Test batch analysis with some failures"""
        def mock_analyze_with_failure(token, *args, **kwargs):
            if token.address == sample_tokens[0].address:
                raise MLAnalysisError("Mock analysis failed")
            return PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                direction=PredictionDirection.HOLD,
                confidence=0.5
            )
        
        with patch.object(model_manager, 'analyze_token', side_effect=mock_analyze_with_failure):
            results = await model_manager.batch_analyze(sample_tokens)
            
            # Should only return successful results
            assert len(results) == 1
            assert results[0].token.address == sample_tokens[1].address
    
    @pytest.mark.asyncio
    async def test_train_models_success(self, model_manager, sample_historical_data):
        """Test successful model training"""
        # Mock successful training
        with patch.object(model_manager._models[ModelType.LSTM], 'train_model', return_value=True) as mock_train:
            results = await model_manager.train_models(sample_historical_data, [ModelType.LSTM])
            
            assert results[ModelType.LSTM] is True
            mock_train.assert_called_once_with(sample_historical_data)
    
    @pytest.mark.asyncio
    async def test_train_models_failure(self, model_manager, sample_historical_data):
        """Test model training failure"""
        # Mock training failure
        with patch.object(model_manager._models[ModelType.LSTM], 'train_model', return_value=False):
            results = await model_manager.train_models(sample_historical_data, [ModelType.LSTM])
            
            assert results[ModelType.LSTM] is False
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, model_manager, sample_tokens):
        """Test ensemble prediction with multiple models"""
        token = sample_tokens[0]
        
        # Create mock predictions from multiple models
        mock_predictions = {
            ModelType.LSTM: PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_24h=110.0,
                confidence=0.8,
                probability_up=0.7
            )
        }
        
        # Mock all models as trained
        for model in model_manager._models.values():
            model.is_model_trained = MagicMock(return_value=True)
            model.analyze_token = AsyncMock(return_value=mock_predictions[ModelType.LSTM])
        
        result = await model_manager._ensemble_prediction(token, None)
        
        assert isinstance(result, PredictionResult)
        assert result.model_type == ModelType.ENSEMBLE
        assert result.token == token
    
    def test_combine_predictions(self, model_manager, sample_tokens):
        """Test prediction combination for ensemble"""
        token = sample_tokens[0]
        
        # Create multiple predictions
        predictions = {
            ModelType.LSTM: PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_24h=110.0,
                confidence=0.8,
                probability_up=0.7
            )
        }
        
        ensemble_result = model_manager._combine_predictions(predictions, token)
        
        assert isinstance(ensemble_result, PredictionResult)
        assert ensemble_result.model_type == ModelType.ENSEMBLE
        assert ensemble_result.token == token
        assert ensemble_result.price_prediction_24h is not None
        assert 0 <= ensemble_result.confidence <= 1
        assert 0 <= ensemble_result.probability_up <= 1
    
    def test_get_best_model(self, model_manager):
        """Test best model selection"""
        # All models untrained initially
        best_model = model_manager._get_best_model()
        assert best_model is not None
        
        # Mock one model as trained with high performance
        model_manager._models[ModelType.LSTM].is_model_trained = MagicMock(return_value=True)
        model_manager._model_performance[ModelType.LSTM]['accuracy'] = 0.9
        
        best_model = model_manager._get_best_model()
        assert best_model == model_manager._models[ModelType.LSTM]
    
    @pytest.mark.asyncio
    async def test_update_performance_tracking(self, model_manager, sample_tokens):
        """Test performance tracking updates"""
        token = sample_tokens[0]
        
        result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            model_accuracy=0.85,
            confidence=0.8
        )
        
        initial_count = model_manager._model_performance[ModelType.LSTM]['predictions_made']
        
        await model_manager._update_performance_tracking(result)
        
        # Check that performance was updated
        perf = model_manager._model_performance[ModelType.LSTM]
        assert perf['predictions_made'] == initial_count + 1
        assert perf['accuracy'] == 0.85  # First prediction sets accuracy
    
    @pytest.mark.asyncio
    async def test_update_model_weights(self, model_manager):
        """Test model weight updates"""
        # Set some performance data
        model_manager._model_performance[ModelType.LSTM] = {
            'accuracy': 0.8,
            'predictions_made': 50,
            'last_updated': datetime.now().timestamp()
        }
        
        initial_weight = model_manager._model_weights[ModelType.LSTM]
        
        await model_manager._update_model_weights()
        
        # Weight should be updated based on performance
        new_weight = model_manager._model_weights[ModelType.LSTM]
        assert new_weight >= 0.1  # Minimum weight
        
        # With good accuracy and experience, weight should be high
        assert new_weight > 0.5
    
    @pytest.mark.asyncio
    async def test_save_and_load_models(self, model_manager):
        """Test model saving and loading"""
        # Mock model with save/load capabilities
        mock_model = MagicMock()
        mock_model.save_model = MagicMock(return_value=True)
        mock_model.load_model = MagicMock(return_value=True)
        model_manager._models[ModelType.LSTM] = mock_model
        
        # Test saving
        await model_manager._save_model(ModelType.LSTM)
        mock_model.save_model.assert_called_once()
        
        # Create the expected file to simulate successful save
        model_file = model_manager.model_dir / "lstm_model.pt"
        model_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        model_file.touch()  # Create empty file
        
        # Test loading
        load_results = await model_manager.load_models([ModelType.LSTM])
        assert load_results[ModelType.LSTM] is True
        mock_model.load_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, model_manager):
        """Test health check functionality"""
        # Mock model health check
        model_manager._models[ModelType.LSTM].health_check = AsyncMock(return_value=True)
        model_manager._models[ModelType.LSTM].is_model_trained = MagicMock(return_value=True)
        
        health_status = await model_manager.health_check()
        
        assert isinstance(health_status, dict)
        assert 'overall_healthy' in health_status
        assert 'models' in health_status
        assert 'ensemble_available' in health_status
        assert 'model_weights' in health_status
        assert 'performance' in health_status
        
        # With one healthy model
        assert health_status['overall_healthy'] is True
        assert health_status['ensemble_available'] is False  # Need >1 healthy model
        
        # Check model-specific status
        lstm_status = health_status['models']['lstm']
        assert lstm_status['healthy'] is True
        assert lstm_status['trained'] is True
    
    def test_get_model_performance(self, model_manager):
        """Test model performance retrieval"""
        performance = model_manager.get_model_performance()
        
        assert isinstance(performance, dict)
        assert 'weights' in performance
        assert 'performance' in performance
        assert 'cache_stats' in performance
        
        # Check cache stats
        cache_stats = performance['cache_stats']
        assert 'size' in cache_stats
        assert 'ttl_minutes' in cache_stats
        assert cache_stats['ttl_minutes'] == 5  # From config
    
    def test_clear_cache(self, model_manager):
        """Test cache clearing"""
        # Add something to cache
        model_manager._ensemble_cache["test_key"] = (datetime.now(), MagicMock())
        
        assert len(model_manager._ensemble_cache) == 1
        
        # Clear cache
        model_manager.clear_cache()
        
        assert len(model_manager._ensemble_cache) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_token_ensemble_fallback(self, model_manager, sample_tokens):
        """Test ensemble analysis fallback to single model"""
        token = sample_tokens[0]
        
        # Mock single model result
        mock_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            direction=PredictionDirection.BUY,
            confidence=0.8
        )
        
        # Mock get_best_model to return LSTM
        with patch.object(model_manager, '_get_best_model') as mock_best:
            mock_best.return_value = model_manager._models[ModelType.LSTM]
            
            with patch.object(model_manager._models[ModelType.LSTM], 'analyze_token', return_value=mock_result):
                # Request ensemble but only have 1 model
                result = await model_manager.analyze_token(token, None, use_ensemble=True)
                
                assert isinstance(result, PredictionResult)
                assert result.direction == PredictionDirection.BUY
    
    @pytest.mark.asyncio
    async def test_analyze_token_error_handling(self, model_manager, sample_tokens):
        """Test error handling in token analysis"""
        token = sample_tokens[0]
        
        # Mock model to raise exception
        with patch.object(model_manager, '_get_best_model') as mock_best:
            mock_model = MagicMock()
            mock_model.analyze_token = AsyncMock(side_effect=Exception("Mock error"))
            mock_best.return_value = mock_model
            
            with pytest.raises(MLAnalysisError):
                await model_manager.analyze_token(token, None, use_ensemble=False)
    
    @pytest.mark.asyncio
    async def test_train_models_invalid_type(self, model_manager, sample_historical_data):
        """Test training with invalid model type"""
        # Try to train a model type that doesn't exist
        fake_model_type = MagicMock()
        fake_model_type.value = "fake_model"
        
        results = await model_manager.train_models(sample_historical_data, [fake_model_type])
        
        # Should return empty results for invalid model types
        assert len(results) == 0
    
    def test_model_directory_creation(self):
        """Test model directory is created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            config = {'model_dir': str(model_dir)}
            
            # Directory shouldn't exist initially
            assert not model_dir.exists()
            
            # Creating manager should create directory
            manager = ModelManager(config)
            assert model_dir.exists()
            assert model_dir.is_dir()