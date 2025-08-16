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
from sklearn.preprocessing import StandardScaler

from src.discovery.base import DiscoveredToken
from .base import (
    MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection,
    TechnicalIndicators, MarketFeatures, ModelNotTrainedError, PredictionError
)
from .feature_engineer import FeatureEngineer
from .market_data import CoinGeckoClient, MarketDataError, APIRateLimitError, DataNotAvailableError


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
        self._scaler = StandardScaler()
        self._feature_engineer = FeatureEngineer()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Market data client
        coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self._coingecko_client = CoinGeckoClient(api_key=coingecko_api_key, cache_ttl=300)
        
        # Model metadata
        self._model_version = "2.0.0"  # Updated version for real data integration
        self._training_history = []
        self._feature_names = []
        self._scaler_fitted = False
        
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
    
    async def train_model(self, training_data: pd.DataFrame, coin_id: str = "bitcoin") -> bool:
        """Train the LSTM model on historical data"""
        try:
            self.logger.info("Starting LSTM model training", 
                           data_shape=training_data.shape,
                           coin_id=coin_id)
            
            # Prepare training data using real market data
            X, y, feature_names = await self._prepare_training_data(training_data, coin_id)
            
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
        """Prepare features for prediction using the same enhanced features as training"""
        try:
            # Prepare enhanced features similar to training
            feature_data = await self._prepare_enhanced_features(historical_data)
            
            # Use the last sequence_length rows for prediction
            if len(feature_data) < self.sequence_length:
                raise MarketDataError(f"Insufficient data for prediction: need {self.sequence_length} rows, got {len(feature_data)}")
            
            recent_features = feature_data.tail(self.sequence_length)
            feature_matrix = recent_features.values
            
            # Normalize features using the fitted scaler
            if self._scaler_fitted:
                feature_matrix = self._scaler.transform(feature_matrix)
            else:
                self.logger.warning("Scaler not fitted, using raw features for prediction")
            
            return feature_matrix.reshape(1, self.sequence_length, -1)  # Batch size 1
            
        except Exception as e:
            self.logger.error("Failed to prepare features for prediction", error=str(e))
            raise PredictionError(f"Failed to prepare features: {str(e)}")
    
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
    
    def prepare_training_from_corpus(self, data: pd.DataFrame, 
                                    sequence_length: Optional[int] = None,
                                    prediction_horizons: List[int] = [1, 4, 24]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from unified corpus format
        
        Args:
            data: DataFrame from corpus with all features
            sequence_length: Length of input sequences (uses self.sequence_length if not provided)
            prediction_horizons: Hours ahead to predict [1h, 4h, 24h]
            
        Returns:
            X: Input sequences [n_samples, sequence_length, n_features]
            y: Target values [n_samples, n_targets] 
            feature_names: List of feature names used
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Identify metadata columns to exclude
        metadata_cols = ['id', 'ohlcv_id', 'token_id', 'timestamp', 'feature_version',
                        'calculated_at', 'data_source', 'granularity', 'created_at', 'updated_at',
                        'collection_timestamp']
        
        # Get all numeric columns except metadata
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in metadata_cols]
        
        # Ensure we have essential columns
        if 'close' not in data.columns:
            raise ValueError("Missing 'close' price column in corpus data")
            
        # Select feature columns - use all available numeric features
        feature_data = data[feature_cols].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        max_horizon = max(prediction_horizons)
        
        for i in range(sequence_length, len(data) - max_horizon):
            # Input sequence
            seq_features = feature_data.iloc[i-sequence_length:i].values
            
            # Target prices at different horizons
            current_price = data['close'].iloc[i-1]
            target_values = []
            
            for horizon in prediction_horizons:
                if i + horizon - 1 < len(data):
                    future_price = data['close'].iloc[i + horizon - 1]
                    # Store actual price (model calculates returns internally if needed)
                    target_values.append(future_price)
                else:
                    target_values.append(data['close'].iloc[-1])
            
            sequences.append(seq_features)
            targets.append(target_values)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Get feature names
        feature_names = list(feature_data.columns)
        
        # Normalize features if scaler is fitted
        if self._scaler_fitted:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_normalized = self._scaler.transform(X_reshaped)
            X = X_normalized.reshape(X.shape)
        else:
            # Fit scaler on this data
            X_reshaped = X.reshape(-1, X.shape[-1])
            self._scaler.fit(X_reshaped)
            self._scaler_fitted = True
            X_normalized = self._scaler.transform(X_reshaped)
            X = X_normalized.reshape(X.shape)
        
        self.logger.info("Corpus training data prepared",
                        sequences=len(X),
                        sequence_length=sequence_length,
                        features=len(feature_names),
                        horizons=prediction_horizons)
        
        return X, y, feature_names
    
    async def _prepare_training_data(self, data: pd.DataFrame, coin_id: str = "bitcoin") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for LSTM using real market data"""
        try:
            # Get extended historical data from CoinGecko for better training
            self.logger.info("Fetching extended historical data for training", coin_id=coin_id)
            
            # Get 90 days of historical data for training
            extended_data = await self._coingecko_client.get_ohlcv_data(coin_id, days=90)
            
            if len(extended_data) < self.sequence_length + 24:  # Need data for 24h predictions
                raise MarketDataError(f"Insufficient historical data: got {len(extended_data)} records, need at least {self.sequence_length + 24}")
            
            # Prepare features with technical indicators
            feature_data = await self._prepare_enhanced_features(extended_data)
            
            # Create sequences and targets
            sequences = []
            targets = []
            
            # Use sliding window approach
            for i in range(len(feature_data) - self.sequence_length - 24):  # -24 for 24h prediction
                # Input sequence
                seq_features = feature_data.iloc[i:i + self.sequence_length].values
                
                # Target prices at different horizons (1h, 4h, 24h later)
                base_idx = i + self.sequence_length
                target_1h = extended_data['close'].iloc[base_idx + 1] if base_idx + 1 < len(extended_data) else extended_data['close'].iloc[-1]
                target_4h = extended_data['close'].iloc[base_idx + 4] if base_idx + 4 < len(extended_data) else extended_data['close'].iloc[-1]
                target_24h = extended_data['close'].iloc[base_idx + 24] if base_idx + 24 < len(extended_data) else extended_data['close'].iloc[-1]
                
                sequences.append(seq_features)
                targets.append([target_1h, target_4h, target_24h])
            
            X = np.array(sequences, dtype=np.float32)
            y = np.array(targets, dtype=np.float32)
            
            # Get feature names
            feature_names = list(feature_data.columns)
            
            # Normalize features
            if not self._scaler_fitted:
                # Fit scaler on training data
                X_reshaped = X.reshape(-1, X.shape[-1])
                self._scaler.fit(X_reshaped)
                self._scaler_fitted = True
            
            # Transform features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_normalized = self._scaler.transform(X_reshaped)
            X = X_normalized.reshape(X.shape)
            
            self.logger.info("Real training data prepared", 
                            sequences=len(X), 
                            sequence_length=self.sequence_length,
                            features=len(feature_names),
                            coin_id=coin_id)
            
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error("Failed to prepare real training data", error=str(e))
            raise MarketDataError(f"Failed to prepare training data: {str(e)}")
    
    async def _prepare_enhanced_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features with technical indicators"""
        try:
            df = ohlcv_data.copy()
            
            # Basic OHLCV features
            features = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Price-based features
            features['price_change'] = df['close'].pct_change()
            features['high_low_ratio'] = df['high'] / df['low']
            features['volume_price_ratio'] = df['volume'] / df['close']
            
            # Moving averages
            features['sma_5'] = df['close'].rolling(window=5).mean()
            features['sma_10'] = df['close'].rolling(window=10).mean()
            features['sma_20'] = df['close'].rolling(window=20).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(df['close'], window=14)
            features['macd'] = features['ema_12'] - features['ema_26']
            features['atr'] = self._calculate_atr(df, window=14)
            
            # Bollinger Bands
            bb_window = 20
            bb_std = 2
            bb_mean = df['close'].rolling(window=bb_window).mean()
            bb_std_val = df['close'].rolling(window=bb_window).std()
            features['bollinger_upper'] = bb_mean + (bb_std_val * bb_std)
            features['bollinger_lower'] = bb_mean - (bb_std_val * bb_std)
            features['bollinger_width'] = features['bollinger_upper'] - features['bollinger_lower']
            features['bollinger_position'] = (df['close'] - features['bollinger_lower']) / features['bollinger_width']
            
            # Volume indicators
            features['volume_sma'] = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['obv'] = self._calculate_obv(df)
            
            # Volatility indicators
            features['returns'] = df['close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(window=50).mean()
            
            # Time-based features
            features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
            # Fill NaN values
            features = features.fillna(method='forward').fillna(method='backward').fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to prepare enhanced features", error=str(e))
            raise MarketDataError(f"Failed to prepare enhanced features: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.where(df['close'] > df['close'].shift(), df['volume'], 
                      np.where(df['close'] < df['close'].shift(), -df['volume'], 0))
        return pd.Series(obv, index=df.index).cumsum()
    
    async def _prepare_training_data_real(self, data: pd.DataFrame, coin_id: str = "bitcoin") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Alias for the new real data preparation method (for testing compatibility)"""
        return await self._prepare_training_data(data, coin_id)
    
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
    
    async def close(self):
        """Close resources and cleanup"""
        if self._coingecko_client:
            await self._coingecko_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Alias for compatibility with test expectations
LSTMAnalyzer = LSTMPricePredictor