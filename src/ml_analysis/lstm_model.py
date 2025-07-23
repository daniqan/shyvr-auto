"""
LSTM Price Prediction Model
Advanced neural network for cryptocurrency price forecasting
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import structlog

from src.discovery.base import DiscoveredToken
from .base import (
    MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection,
    TechnicalIndicators, MarketFeatures, ModelNotTrainedError, PredictionError
)
from .feature_engineer import FeatureEngineer


logger = structlog.get_logger()


class LSTMNetwork(nn.Module):
    """LSTM Neural Network for price prediction"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,  # 1h, 4h, 24h predictions
                 dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
        # Uncertainty estimation layer
        self.uncertainty_layer = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        last_output = attn_output[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        
        # Predictions
        predictions = self.fc2(out)
        
        # Uncertainty estimates (log variance)
        uncertainty = self.uncertainty_layer(out)
        
        return predictions, uncertainty


class LSTMPricePredictor(MLAnalyzerBase):
    """LSTM-based price prediction analyzer"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ModelType.LSTM, config)
        
        # Model configuration
        self.sequence_length = self.config.get('sequence_length', 50)
        self.hidden_size = self.config.get('hidden_size', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.num_epochs = self.config.get('num_epochs', 100)
        
        # Model components
        self._model: Optional[LSTMNetwork] = None
        self._scaler = None
        self._feature_engineer = FeatureEngineer()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model metadata
        self._model_version = "1.0.0"
        self._training_history = []
        self._feature_names = []
        
        self.logger.info("LSTM Predictor initialized", 
                        device=str(self._device),
                        config=self.config)
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Analyze a token and generate LSTM price predictions"""
        start_time = datetime.now()
        
        if not self.is_model_trained():
            raise ModelNotTrainedError("LSTM model must be trained before analysis")
        
        try:
            # Calculate technical indicators and features
            if historical_data is None or len(historical_data) < self.sequence_length:
                self.logger.warning("Insufficient historical data for LSTM prediction",
                                  token=token.address,
                                  data_length=len(historical_data) if historical_data is not None else 0)
                return self._create_fallback_result(token, start_time)
            
            # Prepare features
            features = await self._prepare_features(token, historical_data)
            
            # Generate predictions
            predictions, uncertainty = await self._generate_predictions(features)
            
            # Calculate technical indicators for result
            tech_indicators = await self._feature_engineer.calculate_technical_indicators(
                token, historical_data
            )
            market_features = await self._feature_engineer.calculate_market_features()
            
            # Determine direction and confidence
            direction, confidence = self._analyze_prediction_direction(
                predictions, uncertainty, token.price_usd
            )
            
            # Calculate trading signals
            entry_signal, stop_loss, take_profit = self._generate_trading_signals(
                predictions, token.price_usd, uncertainty
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = PredictionResult(
                token=token,
                analyzed_at=start_time,
                model_type=self.model_type,
                price_prediction_1h=float(predictions[0]),
                price_prediction_4h=float(predictions[1]),
                price_prediction_24h=float(predictions[2]),
                direction=direction,
                confidence=confidence,
                probability_up=self._calculate_probability_up(predictions, token.price_usd),
                technical_indicators=tech_indicators,
                market_features=market_features,
                model_accuracy=self._get_model_accuracy(),
                prediction_uncertainty=float(uncertainty.mean()),
                volatility_forecast=self._calculate_volatility_forecast(historical_data),
                downside_risk=self._calculate_downside_risk(predictions, token.price_usd),
                upside_potential=self._calculate_upside_potential(predictions, token.price_usd),
                entry_signal=entry_signal,
                stop_loss_level=stop_loss,
                take_profit_level=take_profit,
                position_size_multiplier=self._calculate_position_multiplier(confidence, uncertainty),
                features_used=self._feature_names,
                model_version=self._model_version,
                processing_time_ms=processing_time,
            )
            
            self.logger.info("LSTM analysis completed",
                           token=token.address,
                           direction=direction.value,
                           confidence=confidence,
                           processing_time=processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error("LSTM analysis failed", 
                            token=token.address, 
                            error=str(e))
            raise PredictionError(f"LSTM analysis failed: {str(e)}")
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """Train the LSTM model on historical data"""
        try:
            self.logger.info("Starting LSTM model training", 
                           data_shape=training_data.shape)
            
            # Prepare training data
            X, y, feature_names = await self._prepare_training_data(training_data)
            
            if len(X) < 100:  # Need sufficient training data
                self.logger.error("Insufficient training data", samples=len(X))
                return False
            
            # Initialize model
            input_size = X.shape[2]  # Number of features
            self._model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self._device)
            
            self._feature_names = feature_names
            
            # Prepare data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X).to(self._device),
                torch.FloatTensor(y).to(self._device)
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Initialize optimizer and loss function
            optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            self._model.train()
            training_losses = []
            
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    predictions, uncertainty = self._model(batch_X)
                    
                    # Calculate loss (predictions + uncertainty regularization)
                    pred_loss = criterion(predictions, batch_y)
                    uncertainty_loss = torch.mean(torch.exp(-uncertainty) * (predictions - batch_y)**2 + uncertainty)
                    
                    total_loss = pred_loss + 0.1 * uncertainty_loss
                    
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                training_losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info("Training progress", 
                                   epoch=epoch + 1, 
                                   loss=avg_loss)
            
            # Save training history
            self._training_history = training_losses
            self._is_trained = True
            
            self.logger.info("LSTM model training completed",
                           epochs=self.num_epochs,
                           final_loss=training_losses[-1],
                           model_parameters=sum(p.numel() for p in self._model.parameters()))
            
            return True
            
        except Exception as e:
            self.logger.error("LSTM model training failed", error=str(e))
            return False
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for LSTM model"""
        return [
            # Price data
            "open", "high", "low", "close", "volume",
            # Technical indicators
            "sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd",
            # Volume indicators  
            "volume_ratio", "obv",
            # Volatility indicators
            "atr", "bollinger_width", "volatility_score",
            # Market features
            "market_sentiment", "btc_correlation"
        ]
    
    async def _prepare_features(self, token: DiscoveredToken, 
                              historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        # Calculate technical indicators
        tech_indicators = await self._feature_engineer.calculate_technical_indicators(
            token, historical_data
        )
        market_features = await self._feature_engineer.calculate_market_features()
        
        # Create feature matrix for the sequence
        feature_vectors = []
        
        # Use the last sequence_length rows for prediction
        recent_data = historical_data.tail(self.sequence_length)
        
        for _, row in recent_data.iterrows():
            # Combine price, technical, and market features
            price_features = [
                row.get('open', 0), row.get('high', 0), 
                row.get('low', 0), row.get('close', 0), 
                row.get('volume', 0)
            ]
            
            tech_vector = tech_indicators.to_feature_vector()
            market_vector = market_features.to_feature_vector()
            
            combined = np.concatenate([price_features, tech_vector, market_vector])
            feature_vectors.append(combined)
        
        # Normalize features if scaler is available
        feature_matrix = np.array(feature_vectors)
        if self._scaler is not None:
            feature_matrix = self._scaler.transform(feature_matrix)
        
        return feature_matrix.reshape(1, self.sequence_length, -1)  # Batch size 1
    
    async def _generate_predictions(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate price predictions using trained model"""
        self._model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self._device)
            predictions, uncertainty = self._model(features_tensor)
            
            # Convert back to numpy
            predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
            uncertainty = uncertainty.cpu().numpy()[0]
        
        return predictions, uncertainty
    
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for LSTM"""
        # This is a simplified version - in production, you'd process multiple tokens
        # For now, return placeholder values
        
        # Create sequences and targets from price data
        sequences = []
        targets = []
        
        # Use sliding window approach
        for i in range(len(data) - self.sequence_length - 3):  # -3 for 3 prediction horizons
            # Input sequence
            seq_data = data.iloc[i:i + self.sequence_length]
            
            # Target prices (1h, 4h, 24h later - simplified as next 3 values)
            target_prices = data['close'].iloc[i + self.sequence_length:i + self.sequence_length + 3].values
            
            if len(target_prices) == 3:  # Ensure we have all targets
                # Create feature sequence (simplified)
                seq_features = seq_data[['open', 'high', 'low', 'close', 'volume']].values
                sequences.append(seq_features)
                targets.append(target_prices)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        feature_names = ['open', 'high', 'low', 'close', 'volume']
        
        self.logger.info("Training data prepared", 
                        sequences=len(X), 
                        sequence_length=self.sequence_length,
                        features=len(feature_names))
        
        return X, y, feature_names
    
    def _create_fallback_result(self, token: DiscoveredToken, start_time: datetime) -> PredictionResult:
        """Create a fallback result when prediction fails"""
        return PredictionResult(
            token=token,
            analyzed_at=start_time,
            model_type=self.model_type,
            direction=PredictionDirection.HOLD,
            confidence=0.0,
            probability_up=0.5,
            model_accuracy=0.0,
            features_used=["insufficient_data"],
            model_version=self._model_version,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
    
    def _analyze_prediction_direction(self, predictions: np.ndarray, 
                                    uncertainty: np.ndarray, 
                                    current_price: Optional[float]) -> Tuple[PredictionDirection, float]:
        """Analyze prediction direction and confidence"""
        if current_price is None:
            return PredictionDirection.HOLD, 0.0
        
        # Use 24h prediction for direction
        predicted_price = predictions[2]  # 24h prediction
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Base confidence on uncertainty (lower uncertainty = higher confidence)
        base_confidence = 1.0 / (1.0 + float(uncertainty[2]))  # 24h uncertainty
        
        # Determine direction
        if price_change_pct > 0.1:  # >10% increase
            return PredictionDirection.STRONG_BUY, min(base_confidence * 0.9, 0.95)
        elif price_change_pct > 0.02:  # >2% increase
            return PredictionDirection.BUY, min(base_confidence * 0.7, 0.85)
        elif price_change_pct < -0.1:  # >10% decrease
            return PredictionDirection.STRONG_SELL, min(base_confidence * 0.9, 0.95)
        elif price_change_pct < -0.02:  # >2% decrease
            return PredictionDirection.SELL, min(base_confidence * 0.7, 0.85)
        else:
            return PredictionDirection.HOLD, min(base_confidence * 0.5, 0.6)
    
    def _calculate_probability_up(self, predictions: np.ndarray, current_price: Optional[float]) -> float:
        """Calculate probability of price going up"""
        if current_price is None:
            return 0.5
        
        # Count how many predictions are above current price
        above_current = np.sum(predictions > current_price)
        return float(above_current) / len(predictions)
    
    def _generate_trading_signals(self, predictions: np.ndarray, 
                                current_price: Optional[float],
                                uncertainty: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Generate trading signals based on predictions"""
        if current_price is None:
            return None, None, None
        
        predicted_24h = predictions[2]
        expected_return = (predicted_24h - current_price) / current_price
        confidence = 1.0 / (1.0 + float(uncertainty[2]))
        
        # Entry signal
        entry_signal = None
        if expected_return > 0.05 and confidence > 0.7:
            entry_signal = "BUY"
        elif expected_return < -0.05 and confidence > 0.7:
            entry_signal = "SELL"
        else:
            entry_signal = "WAIT"
        
        # Stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if entry_signal == "BUY":
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = current_price * 1.15  # 15% take profit
        elif entry_signal == "SELL":
            stop_loss = current_price * 1.05  # 5% stop loss for short
            take_profit = current_price * 0.85  # 15% take profit for short
        
        return entry_signal, stop_loss, take_profit
    
    def _calculate_volatility_forecast(self, historical_data: Optional[pd.DataFrame]) -> Optional[float]:
        """Calculate volatility forecast"""
        if historical_data is None or len(historical_data) < 20:
            return None
        
        returns = historical_data['close'].pct_change().dropna()
        recent_volatility = returns.tail(20).std()
        
        return float(recent_volatility) if not pd.isna(recent_volatility) else None
    
    def _calculate_downside_risk(self, predictions: np.ndarray, current_price: Optional[float]) -> Optional[float]:
        """Calculate downside risk estimate"""
        if current_price is None:
            return None
        
        min_prediction = np.min(predictions)
        downside = (current_price - min_prediction) / current_price
        
        return max(float(downside), 0.0)
    
    def _calculate_upside_potential(self, predictions: np.ndarray, current_price: Optional[float]) -> Optional[float]:
        """Calculate upside potential estimate"""
        if current_price is None:
            return None
        
        max_prediction = np.max(predictions)
        upside = (max_prediction - current_price) / current_price
        
        return max(float(upside), 0.0)
    
    def _calculate_position_multiplier(self, confidence: float, uncertainty: np.ndarray) -> float:
        """Calculate position size multiplier based on confidence and uncertainty"""
        # Lower multiplier for high uncertainty or low confidence
        uncertainty_factor = 1.0 / (1.0 + float(uncertainty.mean()))
        
        multiplier = confidence * uncertainty_factor
        
        # Clamp between 0.1 and 2.0
        return max(0.1, min(2.0, multiplier))
    
    def _get_model_accuracy(self) -> Optional[float]:
        """Get model accuracy from training history"""
        if not self._training_history:
            return None
        
        # Convert loss to approximate accuracy (simplified)
        final_loss = self._training_history[-1]
        accuracy = max(0.0, 1.0 - final_loss)
        
        return min(accuracy, 1.0)
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        if not self.is_model_trained():
            return False
        
        try:
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'model_config': self.config,
                'feature_names': self._feature_names,
                'model_version': self._model_version,
                'training_history': self._training_history,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length,
            }, filepath)
            
            self.logger.info("Model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save model", error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.error("Model file not found", filepath=filepath)
                return False
            
            checkpoint = torch.load(filepath, map_location=self._device)
            
            # Restore model configuration
            self.hidden_size = checkpoint.get('hidden_size', self.hidden_size)
            self.num_layers = checkpoint.get('num_layers', self.num_layers)
            self.dropout = checkpoint.get('dropout', self.dropout)
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            
            # Reconstruct model with saved configuration
            input_size = len(checkpoint['feature_names'])
            self._model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self._device)
            
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._feature_names = checkpoint['feature_names']
            self._model_version = checkpoint['model_version']
            self._training_history = checkpoint['training_history']
            self._is_trained = True
            
            self.logger.info("Model loaded", 
                           filepath=filepath,
                           version=self._model_version)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            return False