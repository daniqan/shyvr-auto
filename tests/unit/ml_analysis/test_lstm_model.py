"""
Unit tests for LSTM price prediction model
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken
from src.ml_analysis.lstm_model import LSTMNetwork, LSTMPricePredictor
from src.ml_analysis.base import (
    ModelType, PredictionDirection, PredictionResult,
    ModelNotTrainedError, PredictionError
)


class TestLSTMNetwork:
    """Test LSTM neural network architecture"""
    
    def test_lstm_network_initialization(self):
        """Test LSTM network initialization"""
        input_size = 20
        hidden_size = 64
        num_layers = 2
        output_size = 3
        
        model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        
        assert model.hidden_size == hidden_size
        assert model.num_layers == num_layers
        
        # Check if all layers are present
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'uncertainty_layer')
    
    def test_lstm_network_forward_pass(self):
        """Test LSTM network forward pass"""
        input_size = 20
        hidden_size = 64
        sequence_length = 50
        batch_size = 4
        
        model = LSTMNetwork(input_size=input_size, hidden_size=hidden_size)
        
        # Create sample input
        x = torch.randn(batch_size, sequence_length, input_size)
        
        # Forward pass
        predictions, uncertainty = model(x)
        
        # Check output shapes
        assert predictions.shape == (batch_size, 3)  # 3 time horizons
        assert uncertainty.shape == (batch_size, 3)  # 3 uncertainty estimates
        
        # Check output types
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(uncertainty, torch.Tensor)
    
    def test_lstm_network_with_different_parameters(self):
        """Test LSTM network with different parameter configurations"""
        configs = [
            {'input_size': 10, 'hidden_size': 32, 'num_layers': 1},
            {'input_size': 50, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3},
            {'input_size': 25, 'hidden_size': 256, 'num_layers': 2, 'output_size': 5}
        ]
        
        for config in configs:
            model = LSTMNetwork(**config)
            
            # Test forward pass
            batch_size = 2
            sequence_length = 20
            input_size = config['input_size']
            output_size = config.get('output_size', 3)
            
            x = torch.randn(batch_size, sequence_length, input_size)
            predictions, uncertainty = model(x)
            
            assert predictions.shape == (batch_size, output_size)
            assert uncertainty.shape == (batch_size, output_size)


class TestLSTMPricePredictor:
    """Test LSTM price prediction analyzer"""
    
    @pytest.fixture
    def lstm_config(self):
        """LSTM configuration for testing"""
        return {
            'sequence_length': 20,  # Shorter for tests
            'hidden_size': 32,      # Smaller for tests
            'num_layers': 1,        # Simpler for tests
            'num_epochs': 5,        # Fewer epochs for tests
            'batch_size': 4
        }
    
    @pytest.fixture
    def lstm_predictor(self, lstm_config):
        """Create LSTM predictor for testing"""
        return LSTMPricePredictor(lstm_config)
    
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
            price_usd=100.0,
            market_cap=1000000.0,
            volume_24h=50000.0
        )
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate 100 hours of data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
        
        # Generate realistic price series
        prices = [100.0]
        for _ in range(99):
            change = np.random.normal(0, 0.01)  # 1% volatility
            prices.append(prices[-1] * (1 + change))
        
        volumes = np.random.uniform(1000, 5000, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': volumes
        })
    
    def test_lstm_predictor_initialization(self, lstm_predictor, lstm_config):
        """Test LSTM predictor initialization"""
        assert lstm_predictor.model_type == ModelType.LSTM
        assert lstm_predictor.sequence_length == lstm_config['sequence_length']
        assert lstm_predictor.hidden_size == lstm_config['hidden_size']
        assert lstm_predictor.num_layers == lstm_config['num_layers']
        assert lstm_predictor._model is None
        assert not lstm_predictor.is_model_trained()
    
    @pytest.mark.asyncio
    async def test_analyze_token_untrained_model(self, lstm_predictor, sample_token, sample_historical_data):
        """Test analyze_token with untrained model"""
        with pytest.raises(ModelNotTrainedError):
            await lstm_predictor.analyze_token(sample_token, sample_historical_data)
    
    @pytest.mark.asyncio
    async def test_analyze_token_insufficient_data(self, lstm_predictor, sample_token):
        """Test analyze_token with insufficient historical data"""
        # Mock model as trained
        lstm_predictor._is_trained = True
        lstm_predictor._model = MagicMock()
        
        # Create insufficient data
        short_data = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'volume': [1000, 1100, 1200]
        })
        
        result = await lstm_predictor.analyze_token(sample_token, short_data)
        
        # Should return fallback result
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.HOLD
        assert result.confidence == 0.0
        assert "insufficient_data" in result.features_used
    
    @pytest.mark.asyncio
    async def test_train_model_insufficient_data(self, lstm_predictor):
        """Test model training with insufficient data"""
        # Create minimal training data
        small_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        success = await lstm_predictor.train_model(small_data)
        assert not success
        assert not lstm_predictor.is_model_trained()
    
    @pytest.mark.asyncio 
    async def test_train_model_success(self, lstm_predictor, sample_historical_data):
        """Test successful model training"""
        # Extend data to have sufficient samples for training
        extended_data = pd.concat([sample_historical_data] * 5, ignore_index=True)
        
        # Add more realistic columns for training
        extended_data['open'] = extended_data['close'].shift(1).fillna(extended_data['close'])
        extended_data['high'] = extended_data['close'] * 1.01
        extended_data['low'] = extended_data['close'] * 0.99
        
        success = await lstm_predictor.train_model(extended_data)
        
        assert success
        assert lstm_predictor.is_model_trained()
        assert lstm_predictor._model is not None
        assert len(lstm_predictor._training_history) > 0
    
    def test_get_required_features(self, lstm_predictor):
        """Test required features specification"""
        features = lstm_predictor.get_required_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = ["close", "volume", "rsi", "macd"]
        for feature in expected_features:
            assert feature in features
    
    @pytest.mark.asyncio
    async def test_prepare_features(self, lstm_predictor, sample_token, sample_historical_data):
        """Test feature preparation for prediction"""
        # Mock feature engineer
        with patch.object(lstm_predictor._feature_engineer, 'calculate_technical_indicators') as mock_tech, \
             patch.object(lstm_predictor._feature_engineer, 'calculate_market_features') as mock_market:
            
            # Mock technical indicators
            mock_tech.return_value = MagicMock()
            mock_tech.return_value.to_feature_vector.return_value = np.ones(17, dtype=np.float32)
            
            # Mock market features
            mock_market.return_value = MagicMock()
            mock_market.return_value.to_feature_vector.return_value = np.ones(9, dtype=np.float32)
            
            features = await lstm_predictor._prepare_features(sample_token, sample_historical_data)
            
            assert isinstance(features, np.ndarray)
            assert features.shape[0] == 1  # Batch size 1
            assert features.shape[1] == lstm_predictor.sequence_length
            assert features.shape[2] > 0  # Feature dimension
    
    @pytest.mark.asyncio
    async def test_generate_predictions(self, lstm_predictor):
        """Test prediction generation with mock model"""
        # Create mock model
        mock_model = MagicMock()
        mock_predictions = torch.tensor([[1.1, 1.2, 1.3]])  # 3 time horizons
        mock_uncertainty = torch.tensor([[0.1, 0.2, 0.3]])
        mock_model.return_value = (mock_predictions, mock_uncertainty)
        
        lstm_predictor._model = mock_model
        lstm_predictor._device = torch.device('cpu')
        
        # Create sample features
        features = np.random.randn(1, 20, 10)
        
        predictions, uncertainty = await lstm_predictor._generate_predictions(features)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(uncertainty, np.ndarray)
        assert len(predictions) == 3  # 3 time horizons
        assert len(uncertainty) == 3
    
    def test_analyze_prediction_direction(self, lstm_predictor):
        """Test prediction direction analysis"""
        current_price = 100.0
        
        # Test strong buy signal
        strong_buy_predictions = np.array([105.0, 110.0, 115.0])  # >10% increase
        low_uncertainty = np.array([0.1, 0.1, 0.1])
        
        direction, confidence = lstm_predictor._analyze_prediction_direction(
            strong_buy_predictions, low_uncertainty, current_price
        )
        
        assert direction == PredictionDirection.STRONG_BUY
        assert confidence > 0.5
        
        # Test buy signal
        buy_predictions = np.array([101.0, 102.0, 103.0])  # 3% increase
        direction, confidence = lstm_predictor._analyze_prediction_direction(
            buy_predictions, low_uncertainty, current_price
        )
        
        assert direction == PredictionDirection.BUY
        
        # Test hold signal
        hold_predictions = np.array([100.5, 100.8, 101.0])  # Small change
        direction, confidence = lstm_predictor._analyze_prediction_direction(
            hold_predictions, low_uncertainty, current_price
        )
        
        assert direction == PredictionDirection.HOLD
        
        # Test with no current price
        direction, confidence = lstm_predictor._analyze_prediction_direction(
            strong_buy_predictions, low_uncertainty, None
        )
        
        assert direction == PredictionDirection.HOLD
        assert confidence == 0.0
    
    def test_calculate_probability_up(self, lstm_predictor):
        """Test probability up calculation"""
        current_price = 100.0
        
        # All predictions above current price
        all_up_predictions = np.array([101.0, 102.0, 103.0])
        prob_up = lstm_predictor._calculate_probability_up(all_up_predictions, current_price)
        assert prob_up == 1.0
        
        # Mixed predictions
        mixed_predictions = np.array([99.0, 101.0, 102.0])
        prob_mixed = lstm_predictor._calculate_probability_up(mixed_predictions, current_price)
        assert prob_mixed == 2.0/3.0  # 2 out of 3 above current
        
        # No current price
        prob_none = lstm_predictor._calculate_probability_up(all_up_predictions, None)
        assert prob_none == 0.5
    
    def test_generate_trading_signals(self, lstm_predictor):
        """Test trading signal generation"""
        current_price = 100.0
        predictions = np.array([102.0, 105.0, 110.0])  # Strong upward trend
        low_uncertainty = np.array([0.1, 0.1, 0.1])
        
        entry_signal, stop_loss, take_profit = lstm_predictor._generate_trading_signals(
            predictions, current_price, low_uncertainty
        )
        
        assert entry_signal == "BUY"
        assert stop_loss == current_price * 0.95  # 5% stop loss
        assert take_profit == current_price * 1.15  # 15% take profit
        
        # Test with downward predictions
        down_predictions = np.array([98.0, 95.0, 90.0])
        entry_signal_down, stop_loss_down, take_profit_down = lstm_predictor._generate_trading_signals(
            down_predictions, current_price, low_uncertainty
        )
        
        assert entry_signal_down == "SELL"
        
        # Test with no current price
        signals_none = lstm_predictor._generate_trading_signals(
            predictions, None, low_uncertainty
        )
        
        assert all(s is None for s in signals_none)
    
    def test_calculate_position_multiplier(self, lstm_predictor):
        """Test position size multiplier calculation"""
        high_confidence = 0.9
        low_uncertainty = np.array([0.1, 0.1, 0.1])
        
        multiplier = lstm_predictor._calculate_position_multiplier(high_confidence, low_uncertainty)
        
        assert isinstance(multiplier, float)
        assert 0.1 <= multiplier <= 2.0  # Should be clamped
        
        # High confidence + low uncertainty should give higher multiplier
        low_confidence = 0.3
        high_uncertainty = np.array([0.8, 0.8, 0.8])
        
        low_multiplier = lstm_predictor._calculate_position_multiplier(low_confidence, high_uncertainty)
        
        assert multiplier > low_multiplier
    
    def test_save_and_load_model(self, lstm_predictor):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pt"
            
            # Try to save untrained model
            success = lstm_predictor.save_model(str(model_path))
            assert not success  # Should fail for untrained model
            
            # Mock trained model with matching configuration
            lstm_predictor._is_trained = True
            lstm_predictor._model = LSTMNetwork(
                input_size=10, 
                hidden_size=lstm_predictor.hidden_size,  # Use predictor's config
                num_layers=lstm_predictor.num_layers,
                dropout=lstm_predictor.dropout
            )
            lstm_predictor._feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5",
                                          "feature6", "feature7", "feature8", "feature9", "feature10"]
            lstm_predictor._training_history = [0.5, 0.4, 0.3]
            
            # Save model
            success = lstm_predictor.save_model(str(model_path))
            assert success
            assert model_path.exists()
            
            # Create new predictor with same config and load model
            new_predictor = LSTMPricePredictor(lstm_predictor.config)
            load_success = new_predictor.load_model(str(model_path))
            
            assert load_success
            assert new_predictor.is_model_trained()
            assert new_predictor._feature_names == lstm_predictor._feature_names
            assert new_predictor._training_history == lstm_predictor._training_history
    
    def test_load_nonexistent_model(self, lstm_predictor):
        """Test loading non-existent model"""
        nonexistent_path = "/tmp/nonexistent_model.pt"
        success = lstm_predictor.load_model(nonexistent_path)
        assert not success
    
    @pytest.mark.asyncio
    async def test_health_check(self, lstm_predictor):
        """Test health check functionality"""
        # Untrained model should be unhealthy
        is_healthy = await lstm_predictor.health_check()
        assert not is_healthy
        
        # Mock trained model
        lstm_predictor._is_trained = True
        lstm_predictor._model = MagicMock()
        
        # Trained model should be healthy
        is_healthy = await lstm_predictor.health_check()
        assert is_healthy
    
    def test_create_fallback_result(self, lstm_predictor, sample_token):
        """Test fallback result creation"""
        start_time = datetime.now()
        
        result = lstm_predictor._create_fallback_result(sample_token, start_time)
        
        assert isinstance(result, PredictionResult)
        assert result.token == sample_token
        assert result.model_type == ModelType.LSTM
        assert result.direction == PredictionDirection.HOLD
        assert result.confidence == 0.0
        assert result.probability_up == 0.5
        assert "insufficient_data" in result.features_used
    
    def test_get_model_accuracy(self, lstm_predictor):
        """Test model accuracy calculation"""
        # No training history
        accuracy = lstm_predictor._get_model_accuracy()
        assert accuracy is None
        
        # With training history
        lstm_predictor._training_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        accuracy = lstm_predictor._get_model_accuracy()
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        # Accuracy should be based on final loss (lower loss = higher accuracy)
        assert accuracy > 0.5  # Final loss was 0.2, so accuracy should be > 0.5