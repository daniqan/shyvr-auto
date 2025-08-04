"""
Basic Transformer Predictor for time-series forecasting
Implementation follows the existing pattern of LSTMPricePredictor
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import structlog

from src.discovery.base import DiscoveredToken
from src.ml_analysis.base import (
    MLAnalyzerBase, ModelType, PredictionResult, PredictionDirection,
    TechnicalIndicators, MarketFeatures, ModelNotTrainedError
)
from .base import TransformerBase, TransformerConfig
from .positional_encodings import create_positional_encoding
from .temporal_embeddings import FinancialTemporalEmbedding


logger = structlog.get_logger()


class TransformerNetwork(TransformerBase):
    """
    Transformer network for time-series prediction
    Based on the standard Transformer architecture adapted for financial data
    """
    
    def __init__(self, config: TransformerConfig):
        # TransformerBase expects (model_type, config_dict), but we're inheriting from it
        # Create a dummy config dict from the TransformerConfig
        config_dict = {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'max_seq_length': config.max_seq_length,
            'activation': config.activation,
        }
        super().__init__(ModelType.TRANSFORMER, config_dict)
        
        # Input projection - use a reasonable input size for now
        input_dim = getattr(config, 'input_dim', 20)
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Positional encoding
        from .positional_encodings import PositionalEncodingConfig
        pos_config = PositionalEncodingConfig(
            d_model=config.d_model,
            max_length=config.max_seq_length,
            dropout=config.dropout
        )
        self.pos_encoding = create_positional_encoding('sinusoidal', pos_config)
        
        # Temporal embeddings for financial features
        from .temporal_embeddings import TemporalEmbeddingConfig
        temporal_config = TemporalEmbeddingConfig(
            d_model=config.d_model,
            max_sequence_length=config.max_seq_length,
            dropout=config.dropout
        )
        self.temporal_embedding = FinancialTemporalEmbedding(temporal_config)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Output heads for different prediction horizons
        self.output_heads = nn.ModuleDict({
            '1h': nn.Linear(config.d_model, 1),
            '4h': nn.Linear(config.d_model, 1),
            '24h': nn.Linear(config.d_model, 1)
        })
        
        # Confidence estimator
        self.confidence_head = nn.Linear(config.d_model, 1)
        
        # Direction classifier
        self.direction_head = nn.Linear(config.d_model, 5)  # 5 direction classes
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            timestamps: Optional timestamp tensor for temporal embeddings
            
        Returns:
            Dictionary containing predictions for different horizons
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Add temporal embeddings if timestamps provided
        if timestamps is not None:
            temporal_emb = self.temporal_embedding(timestamps)
            x = x + temporal_emb
        
        # Apply transformer encoder
        memory = self.transformer_encoder(x)
        
        # Use last token for prediction (similar to GPT style)
        last_hidden = memory[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Generate predictions for different horizons
        outputs = {}
        for horizon, head in self.output_heads.items():
            outputs[f'price_{horizon}'] = head(last_hidden)
        
        # Generate confidence and direction predictions
        outputs['confidence'] = torch.sigmoid(self.confidence_head(last_hidden))
        outputs['direction_logits'] = self.direction_head(last_hidden)
        outputs['direction_probs'] = torch.softmax(outputs['direction_logits'], dim=-1)
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing attention weights from different layers
        """
        # This is a simplified implementation
        # In a full implementation, you would modify the transformer encoder
        # to return attention weights
        return {'layer_0_attention': torch.ones(1, self.config.n_heads, x.size(1), x.size(1))}


class TransformerPredictor(MLAnalyzerBase):
    """
    Transformer-based price predictor following the same interface as LSTMPricePredictor
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ModelType.TRANSFORMER, config)
        
        # Parse configuration
        self.model_config = self._parse_config(config or {})
        
        # Initialize model
        self.model = TransformerNetwork(self.model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self._setup_training()
        
        # Feature engineering
        self.sequence_length = self.model_config.max_seq_length
        self.required_features = self._get_required_features()
    
    def _parse_config(self, config: Dict) -> TransformerConfig:
        """Parse configuration dictionary into TransformerConfig"""
        return TransformerConfig(
            d_model=config.get('d_model', 128),
            n_heads=config.get('nhead', config.get('n_heads', 8)),  # Support both naming conventions
            n_layers=config.get('num_layers', config.get('n_layers', 4)),
            d_ff=config.get('dim_feedforward', config.get('d_ff', 512)),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            max_seq_length=config.get('max_seq_length', 100)
        )
    
    def _setup_training(self):
        """Setup training components"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        self.criterion = nn.MSELoss()
        self.direction_criterion = nn.CrossEntropyLoss()
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for this model"""
        return self.required_features
    
    def _get_required_features(self) -> List[str]:
        """Define required features for Transformer model"""
        return [
            # Price features
            'close', 'open', 'high', 'low', 'volume',
            # Technical indicators
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
            'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'atr', 'volume_ratio', 'price_momentum', 'volatility_score',
            # Market features
            'fear_greed_index', 'btc_correlation', 'market_beta',
            # Temporal features
            'hour_of_day', 'day_of_week', 'month_of_year',
            'trading_session', 'market_regime'
        ]
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Analyze a token and generate ML predictions
        
        Args:
            token: Token to analyze
            historical_data: Historical price data
            
        Returns:
            PredictionResult with predictions
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Transformer model is not trained")
        
        if historical_data is None or len(historical_data) < self.sequence_length:
            self.logger.warning("Insufficient data for prediction", 
                              token=token.address,
                              data_length=len(historical_data) if historical_data is not None else 0,
                              required_length=self.sequence_length)
            return self._create_default_prediction(token)
        
        try:
            # Prepare input sequence
            input_sequence = self._prepare_sequence(historical_data, token)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                
                # Extract predictions
                price_1h = outputs['price_1h'].item()
                price_4h = outputs['price_4h'].item()
                price_24h = outputs['price_24h'].item()
                confidence = outputs['confidence'].item()
                direction_probs = outputs['direction_probs'].cpu().numpy().flatten()
            
            # Convert relative predictions to absolute prices
            current_price = token.price_usd or historical_data['close'].iloc[-1]
            price_pred_1h = current_price * (1 + price_1h)
            price_pred_4h = current_price * (1 + price_4h)
            price_pred_24h = current_price * (1 + price_24h)
            
            # Determine direction from direction probabilities
            direction_idx = np.argmax(direction_probs)
            directions = [PredictionDirection.STRONG_SELL, PredictionDirection.SELL, 
                         PredictionDirection.HOLD, PredictionDirection.BUY, 
                         PredictionDirection.STRONG_BUY]
            direction = directions[direction_idx]
            
            # Calculate probability of upward movement
            prob_up = direction_probs[3] + direction_probs[4]  # BUY + STRONG_BUY
            
            # Create technical indicators (simplified)
            tech_indicators = TechnicalIndicators(
                rsi=50.0,  # Placeholder - would be calculated from data
                volume_ratio=1.0,
                price_momentum=price_24h * 100,  # Convert to percentage
                volatility_score=confidence
            )
            
            # Create prediction result
            result = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.TRANSFORMER,
                price_prediction_1h=price_pred_1h,
                price_prediction_4h=price_pred_4h,
                price_prediction_24h=price_pred_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                technical_indicators=tech_indicators,
                model_accuracy=0.75,  # Placeholder
                features_used=['transformer_sequence', 'attention_weights'],
                model_version='transformer_v1.0'
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Transformer prediction failed", 
                            token=token.address, 
                            error=str(e))
            return self._create_default_prediction(token)
    
    def _prepare_sequence(self, data: pd.DataFrame, token: DiscoveredToken) -> np.ndarray:
        """
        Prepare input sequence for the model
        
        Args:
            data: Historical price data
            token: Token information
            
        Returns:
            Numpy array of shape (sequence_length, input_dim)
        """
        # Take the last sequence_length rows
        sequence_data = data.tail(self.sequence_length).copy()
        
        # Basic feature extraction (simplified for now)
        features = []
        
        for _, row in sequence_data.iterrows():
            row_features = [
                row.get('close', 0) / 100.0,  # Normalized price
                row.get('volume', 0) / 1000000.0,  # Normalized volume
                row.get('high', 0) / 100.0,
                row.get('low', 0) / 100.0,
                row.get('open', 0) / 100.0,
                # Add more features as needed
                0.5,  # Placeholder for additional features
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5
            ]
            features.append(row_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_default_prediction(self, token: DiscoveredToken) -> PredictionResult:
        """Create a default prediction when analysis fails"""
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.TRANSFORMER,
            direction=PredictionDirection.HOLD,
            confidence=0.1,
            probability_up=0.5,
            features_used=['default'],
            model_version='transformer_v1.0'
        )
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the Transformer model on historical data
        
        Args:
            training_data: Historical training data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info("Starting Transformer model training", 
                           data_shape=training_data.shape)
            
            # Prepare training data
            sequences, targets = self._prepare_training_data(training_data)
            
            if len(sequences) < 10:  # Minimum training samples
                self.logger.error("Insufficient training data", samples=len(sequences))
                return False
            
            # Update input dimension based on actual data
            input_dim = sequences.shape[2]
            # Set input_dim as attribute for the TransformerNetwork
            setattr(self.model_config, 'input_dim', input_dim)
            self.model = TransformerNetwork(self.model_config)
            self.model.to(self.device)
            self._setup_training()
            
            # Convert to tensors
            X_train = torch.FloatTensor(sequences).to(self.device)
            y_train = torch.FloatTensor(targets).to(self.device)
            
            # Training loop (simplified)
            self.model.train()
            epochs = self.config.get('epochs', 50)
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                outputs = self.model(X_train)
                
                # Calculate loss for price predictions
                loss = 0
                for i, horizon in enumerate(['1h', '4h', '24h']):
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        loss += self.criterion(outputs[pred_key].squeeze(), y_train[:, i])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"Training epoch {epoch}, loss: {loss.item():.4f}")
            
            self._is_trained = True
            self.logger.info("Transformer model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("Transformer model training failed", error=str(e))
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data in the format expected by the model
        
        Args:
            data: Raw training data
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        # Create sliding windows
        for i in range(self.sequence_length, len(data) - 24):  # Need 24 hours ahead for targets
            # Input sequence
            seq_data = data.iloc[i-self.sequence_length:i]
            sequence = self._prepare_sequence(seq_data, None)
            
            # Target values (price changes)
            current_price = data.iloc[i]['close']
            target_1h = (data.iloc[i+1]['close'] - current_price) / current_price
            target_4h = (data.iloc[i+4]['close'] - current_price) / current_price if i+4 < len(data) else 0
            target_24h = (data.iloc[i+24]['close'] - current_price) / current_price if i+24 < len(data) else 0
            
            sequences.append(sequence)
            targets.append([target_1h, target_4h, target_24h])
        
        return np.array(sequences), np.array(targets)
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if not self._is_trained:
                self.logger.warning("Attempting to save untrained model")
                return False
            
            state = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config.__dict__,
                'is_trained': self._is_trained,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
            }
            
            torch.save(state, filepath)
            self.logger.info("Transformer model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save Transformer model", error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Update config
            self.model_config = TransformerConfig(**state['config'])
            
            # Recreate model with loaded config
            self.model = TransformerNetwork(self.model_config)
            self.model.to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            
            self._is_trained = state.get('is_trained', False)
            
            # Load optimizer state if available
            if state.get('optimizer_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.logger.info("Transformer model loaded", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load Transformer model", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check if the analyzer is healthy and ready"""
        try:
            if not self._is_trained:
                return False
            
            # Quick forward pass test
            input_dim = getattr(self.model_config, 'input_dim', 20)
            dummy_input = torch.randn(1, self.sequence_length, input_dim).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(dummy_input)
                # Check that all expected outputs are present
                expected_keys = ['price_1h', 'price_4h', 'price_24h', 'confidence', 'direction_probs']
                return all(key in outputs for key in expected_keys)
                
        except Exception as e:
            self.logger.error("Transformer health check failed", error=str(e))
            return False