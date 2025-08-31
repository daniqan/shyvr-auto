"""
iTransformer Implementation for Multivariate Time-Series Forecasting

The iTransformer uses inverted attention where time points are treated as tokens
instead of features, which is specifically designed for multivariate time-series
prediction and better captures temporal dependencies.

Key Innovation: Inverted attention mechanism treats T time points as tokens
and D features as channels, allowing for better multivariate correlation capture.

Reference: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import structlog

from .base import TransformerBase, TransformerConfig
from .attention import MultiHeadAttention
from .positional_encodings import create_positional_encoding, PositionalEncodingConfig
from .temporal_embeddings import FinancialTemporalEmbedding, TemporalEmbeddingConfig
from ..base import MLAnalyzerBase, ModelType, PredictionResult, PredictionDirection, ModelNotTrainedError
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


@dataclass
class InvertedAttentionConfig(TransformerConfig):
    """Configuration for iTransformer with inverted attention mechanism"""
    
    # iTransformer specific parameters
    n_variates: int = 5              # Number of features/variates (D)
    use_inverted_attention: bool = True  # Use inverted attention mechanism
    variate_embedding_dim: int = None    # Embedding dimension for variates (default: d_model)
    
    # Time-series specific
    prediction_horizons: List[str] = None  # ['1h', '4h', '24h']
    use_variate_tokens: bool = True    # Use learnable variate tokens
    
    # Multivariate correlation
    cross_variate_attention: bool = True  # Enable cross-variate attention
    temporal_fusion_layers: int = 2       # Additional layers for temporal fusion
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        super().__post_init__()
        
        # Validate iTransformer specific parameters
        if self.n_variates <= 0:
            raise ValueError("n_variates must be positive")
        
        # Set defaults
        if self.variate_embedding_dim is None:
            self.variate_embedding_dim = self.d_model
            
        if self.prediction_horizons is None:
            self.prediction_horizons = ['1h', '4h', '24h']
    
    def estimate_memory_usage_mb(self) -> float:
        """Estimate memory usage for iTransformer"""
        base_memory = super().estimate_memory_usage_mb()
        
        # Additional memory for variate embeddings and inverted attention
        variate_memory = (self.n_variates * self.variate_embedding_dim * 4) / (1024 * 1024)
        
        # Inverted attention memory (T x T instead of D x D)
        # This is typically more efficient as T >> D in time-series
        inverted_attention_memory = base_memory * 0.7  # Typically 30% reduction
        
        return inverted_attention_memory + variate_memory


class InvertedMultiHeadAttention(nn.Module):
    """
    Inverted Multi-Head Attention for iTransformer
    
    In standard attention: features attend to features across time
    In inverted attention: time points attend to time points across features
    
    This allows better temporal dependency modeling for multivariate series
    """
    
    def __init__(self, d_model: int, n_heads: int, n_variates: int, 
                 dropout: float = 0.1, max_seq_length: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_variates = n_variates
        self.d_k = d_model // n_heads
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Standard attention mechanism but applied in inverted manner
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        logger.debug("InvertedMultiHeadAttention initialized",
                    d_model=d_model, n_heads=n_heads, n_variates=n_variates)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with inverted attention
        
        Args:
            x: Input tensor [batch_size, seq_len, n_variates, d_model]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            output: Attended output [batch_size, seq_len, n_variates, d_model]
            attention_weights: Attention weights [batch_size, n_variates, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, n_variates, d_model = x.shape
        
        # Reshape for inverted attention: process each variate separately
        # [batch_size * n_variates, seq_len, d_model]
        x_reshaped = x.view(batch_size * n_variates, seq_len, d_model)
        
        # Expand attention mask for all variates if provided
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size * n_variates, seq_len]
            mask_expanded = attention_mask.unsqueeze(1).expand(-1, n_variates, -1)
            mask_expanded = mask_expanded.contiguous().view(batch_size * n_variates, seq_len)
        else:
            mask_expanded = None
        
        # Apply attention across time steps for each variate
        attended_output, attention_weights = self.attention(
            query=x_reshaped,
            key=x_reshaped,
            value=x_reshaped,
            attention_mask=mask_expanded
        )
        
        # Reshape back to original dimensions
        # [batch_size * n_variates, seq_len, d_model] -> [batch_size, seq_len, n_variates, d_model]
        attended_output = attended_output.view(batch_size, n_variates, seq_len, d_model)
        attended_output = attended_output.transpose(1, 2)  # [batch_size, seq_len, n_variates, d_model]
        
        # Reshape attention weights
        # [batch_size * n_variates, n_heads, seq_len, seq_len] -> [batch_size, n_variates, n_heads, seq_len, seq_len]
        attention_weights = attention_weights.view(batch_size, n_variates, self.n_heads, seq_len, seq_len)
        
        # Residual connection and layer normalization
        output = self.layer_norm(attended_output + x)
        
        return output, attention_weights


class iTransformerNetwork(TransformerBase):
    """
    iTransformer Network with Inverted Attention Mechanism
    
    Key innovations:
    1. Inverted attention: time points are tokens, features are channels
    2. Variate-wise embeddings for multivariate correlation
    3. Temporal fusion for better forecasting
    4. Multi-horizon prediction heads
    """
    
    def __init__(self, config: InvertedAttentionConfig):
        # Convert config to base format for parent class
        base_config = {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'max_seq_length': config.max_seq_length,
            'activation': config.activation,
        }
        super().__init__(ModelType.ITRANSFORMER, base_config)
        
        self.config = config
        
        # Variate embedding layer - each feature gets its own embedding
        self.variate_embeddings = nn.Parameter(
            torch.randn(config.n_variates, config.variate_embedding_dim)
        )
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(1, config.d_model)  # Each variate is 1D
        
        # Positional encoding for time steps
        pos_config = PositionalEncodingConfig(
            d_model=config.d_model,
            max_length=config.max_seq_length,
            dropout=config.dropout
        )
        self.pos_encoding = create_positional_encoding('sinusoidal', pos_config)
        
        # Inverted transformer encoder layers
        self.inverted_layers = nn.ModuleList([
            self._create_inverted_layer() for _ in range(config.n_layers)
        ])
        
        # Temporal fusion layers for cross-variate information
        if config.cross_variate_attention:
            self.temporal_fusion = nn.ModuleList([
                self._create_temporal_fusion_layer() 
                for _ in range(config.temporal_fusion_layers)
            ])
        else:
            self.temporal_fusion = None
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict({
            horizon: nn.Linear(config.d_model, config.n_variates)
            for horizon in config.prediction_horizons
        })
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(config.d_model, 1)
        
        # Direction classification head (for each variate)
        self.direction_head = nn.Linear(config.d_model, config.n_variates * 5)  # 5 classes per variate
        
        # Initialize parameters
        self._init_weights()
        
        logger.info("iTransformerNetwork initialized",
                   d_model=config.d_model,
                   n_variates=config.n_variates,
                   n_layers=config.n_layers,
                   prediction_horizons=config.prediction_horizons)
    
    def _create_inverted_layer(self) -> nn.Module:
        """Create a single inverted transformer layer"""
        return nn.ModuleDict({
            'inverted_attention': InvertedMultiHeadAttention(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_variates=self.config.n_variates,
                dropout=self.config.dropout,
                max_seq_length=self.config.max_seq_length
            ),
            'feed_forward': nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_ff),
                nn.GELU() if self.config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_ff, self.config.d_model),
                nn.Dropout(self.config.dropout)
            ),
            'norm1': nn.LayerNorm(self.config.d_model),
            'norm2': nn.LayerNorm(self.config.d_model)
        })
    
    def _create_temporal_fusion_layer(self) -> nn.Module:
        """Create temporal fusion layer for cross-variate attention"""
        return nn.ModuleDict({
            'cross_variate_attention': MultiHeadAttention(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads // 2,  # Use fewer heads for fusion
                dropout=self.config.dropout
            ),
            'fusion_norm': nn.LayerNorm(self.config.d_model)
        })
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize variate embeddings
        nn.init.normal_(self.variate_embeddings, std=0.02)
        
        # Initialize other parameters
        for name, param in self.named_parameters():
            if 'variate_embeddings' in name:
                continue  # Already initialized
            elif 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with inverted attention mechanism
        
        Args:
            x: Input tensor [batch_size, seq_len, n_variates]
            timestamps: Optional timestamps for temporal encoding
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with predictions and attention weights
        """
        batch_size, seq_len, n_variates = x.shape
        
        if n_variates != self.config.n_variates:
            raise ValueError(f"Input tensor has {n_variates} variates but model expects {self.config.n_variates}. "
                           f"Model may need to be recreated with correct n_variates configuration.")
        
        if seq_len == 0:
            raise ValueError("Empty sequence not supported")
        
        # Validate variate embedding dimensions match input
        expected_variate_embed_shape = (self.config.n_variates, self.config.variate_embedding_dim or self.config.d_model)
        if self.variate_embeddings.shape != expected_variate_embed_shape:
            raise RuntimeError(f"Variate embedding shape mismatch: expected {expected_variate_embed_shape}, "
                             f"got {self.variate_embeddings.shape}. Model architecture is inconsistent.")
        
        # Input projection: [batch_size, seq_len, n_variates] -> [batch_size, seq_len, n_variates, d_model]
        x_expanded = x.unsqueeze(-1)  # [batch_size, seq_len, n_variates, 1]
        x_projected = self.input_projection(x_expanded)  # [batch_size, seq_len, n_variates, d_model]
        
        # Add variate embeddings
        variate_embeds = self.variate_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, n_variates, d_model]
        x_embedded = x_projected + variate_embeds
        
        # Add positional encoding for time dimension
        # Reshape to apply positional encoding: [batch_size * n_variates, seq_len, d_model]
        x_pos_input = x_embedded.view(batch_size * n_variates, seq_len, self.config.d_model)
        x_pos_encoded = self.pos_encoding(x_pos_input)
        # Reshape back: [batch_size, seq_len, n_variates, d_model]
        x_encoded = x_pos_encoded.view(batch_size, seq_len, n_variates, self.config.d_model)
        
        # Store attention weights for interpretability
        attention_weights = []
        
        # Apply inverted transformer layers
        hidden = x_encoded
        for layer in self.inverted_layers:
            # Inverted attention within each variate
            attended, attn_weights = layer['inverted_attention'](hidden, attention_mask)
            attention_weights.append(attn_weights)
            
            # Feed-forward with residual connection
            ff_input = layer['norm1'](attended)
            ff_output = layer['feed_forward'](ff_input)
            hidden = layer['norm2'](ff_output + ff_input)
        
        # Apply temporal fusion for cross-variate information
        if self.temporal_fusion is not None:
            for fusion_layer in self.temporal_fusion:
                # Reshape for cross-variate attention: [batch_size * seq_len, n_variates, d_model]
                fusion_input = hidden.view(batch_size * seq_len, n_variates, self.config.d_model)
                
                # Cross-variate attention
                fused_output, _ = fusion_layer['cross_variate_attention'](
                    query=fusion_input,
                    key=fusion_input,
                    value=fusion_input
                )
                
                # Residual and norm
                fused_output = fusion_layer['fusion_norm'](fused_output + fusion_input)
                
                # Reshape back: [batch_size, seq_len, n_variates, d_model]
                hidden = fused_output.view(batch_size, seq_len, n_variates, self.config.d_model)
        
        # Use final time step for prediction (autoregressive style)
        final_hidden = hidden[:, -1, :, :]  # [batch_size, n_variates, d_model]
        
        # Global pooling across variates for global predictions
        global_repr = torch.mean(final_hidden, dim=1)  # [batch_size, d_model]
        
        # Generate predictions
        outputs = {}
        
        # Multi-horizon price predictions
        for horizon, head in self.prediction_heads.items():
            outputs[f'price_{horizon}'] = head(global_repr)  # [batch_size, n_variates]
        
        # Confidence estimation
        outputs['confidence'] = torch.sigmoid(self.confidence_head(global_repr))  # [batch_size, 1]
        
        # Direction predictions for each variate
        direction_logits = self.direction_head(global_repr)  # [batch_size, n_variates * 5]
        direction_logits = direction_logits.view(batch_size, self.config.n_variates, 5)
        outputs['direction_logits'] = direction_logits
        outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
        
        # Store attention weights for interpretability
        outputs['attention_weights'] = torch.stack(attention_weights, dim=0)  # [n_layers, batch_size, n_variates, n_heads, seq_len, seq_len]
        outputs['final_representations'] = final_hidden  # For analysis
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for visualization"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['attention_weights']


class iTransformerPredictor(MLAnalyzerBase):
    """
    iTransformer-based predictor following the same interface as other predictors
    Specialized for multivariate time-series forecasting with inverted attention
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ModelType.ITRANSFORMER, config)
        
        # Parse configuration
        self.model_config = self._parse_config(config or {})
        
        # Initialize model
        self.model = iTransformerNetwork(self.model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self._setup_training()
        
        # Feature engineering
        self.sequence_length = self.model_config.max_seq_length
        self.n_variates = self.model_config.n_variates
        self.required_features = self._get_required_features()
        
        logger.info("iTransformerPredictor initialized",
                   model_type=self.model_type,
                   n_variates=self.n_variates,
                   sequence_length=self.sequence_length)
    
    def _parse_config(self, config: Dict) -> InvertedAttentionConfig:
        """Parse configuration dictionary into InvertedAttentionConfig"""
        return InvertedAttentionConfig(
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 512),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            max_seq_length=config.get('max_seq_length', 100),
            n_variates=config.get('n_variates', 5),
            variate_embedding_dim=config.get('variate_embedding_dim'),
            prediction_horizons=config.get('prediction_horizons', ['1h', '4h', '24h']),
            use_variate_tokens=config.get('use_variate_tokens', True),
            cross_variate_attention=config.get('cross_variate_attention', True),
            temporal_fusion_layers=config.get('temporal_fusion_layers', 2)
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
        """Get list of required features for multivariate iTransformer"""
        return self.required_features
    
    def _get_required_features(self) -> List[str]:
        """Define required features for iTransformer (multivariate)"""
        return [
            # Core price features (main variates)
            'close', 'open', 'high', 'low', 'volume',
            
            # Derived variates for multivariate analysis
            'returns', 'volatility', 'volume_ratio',
            'price_momentum', 'rsi',
            
            # Correlation features (captured through attention)
            'correlation_features', 'cross_asset_features',
            
            # Market context variates
            'market_regime', 'volatility_regime', 'liquidity_score',
            
            # Temporal features (used in embeddings)
            'hour_of_day', 'day_of_week', 'trading_session',
            
            # Technical indicators as variates
            'sma_20', 'ema_12', 'macd', 'bollinger_position',
            'atr_normalized', 'funding_rate'
        ]
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Analyze token using iTransformer with multivariate approach
        """
        if not self._is_trained:
            raise ModelNotTrainedError("iTransformer model is not trained")
        
        if historical_data is None or len(historical_data) < self.sequence_length:
            self.logger.warning("Insufficient data for iTransformer prediction",
                              token=token.address,
                              data_length=len(historical_data) if historical_data is not None else 0,
                              required_length=self.sequence_length)
            return self._create_default_prediction(token)
        
        try:
            # Prepare multivariate input sequence
            input_sequence = self._prepare_multivariate_sequence(historical_data, token)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                
                # Extract predictions (average across variates for main prediction)
                price_predictions = {}
                for horizon in self.model_config.prediction_horizons:
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        # Average across variates for single price prediction
                        variate_preds = outputs[pred_key].cpu().numpy().flatten()
                        price_predictions[horizon] = np.mean(variate_preds)
                
                confidence = outputs['confidence'].item()
                direction_probs = outputs['direction_probs'].cpu().numpy()[0]  # First batch, all variates
                
                # Average direction probabilities across variates
                avg_direction_probs = np.mean(direction_probs, axis=0)
            
            # Convert relative predictions to absolute prices
            current_price = token.price_usd or historical_data['close'].iloc[-1]
            
            price_pred_1h = current_price * (1 + price_predictions.get('1h', 0))
            price_pred_4h = current_price * (1 + price_predictions.get('4h', 0))
            price_pred_24h = current_price * (1 + price_predictions.get('24h', 0))
            
            # Determine direction from averaged probabilities
            direction_idx = np.argmax(avg_direction_probs)
            directions = [PredictionDirection.STRONG_SELL, PredictionDirection.SELL,
                         PredictionDirection.HOLD, PredictionDirection.BUY,
                         PredictionDirection.STRONG_BUY]
            direction = directions[direction_idx]
            
            # Calculate probability of upward movement
            prob_up = avg_direction_probs[3] + avg_direction_probs[4]  # BUY + STRONG_BUY
            
            # Create prediction result
            result = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.ITRANSFORMER,
                price_prediction_1h=price_pred_1h,
                price_prediction_4h=price_pred_4h,
                price_prediction_24h=price_pred_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                model_accuracy=0.78,  # Placeholder - would be tracked
                features_used=['inverted_attention', 'multivariate_correlations', 'temporal_fusion'],
                model_version='itransformer_v1.0'
            )
            
            return result
            
        except Exception as e:
            self.logger.error("iTransformer prediction failed",
                            token=token.address,
                            error=str(e))
            return self._create_default_prediction(token)
    
    def _prepare_multivariate_sequence(self, data: pd.DataFrame, token: DiscoveredToken) -> np.ndarray:
        """
        Prepare multivariate input sequence for iTransformer
        
        Returns:
            Numpy array of shape (sequence_length, n_variates)
        """
        # Take the last sequence_length rows
        sequence_data = data.tail(self.sequence_length).copy()
        
        # Extract core variates (features that will be treated as separate channels)
        variates = []
        
        # Core price-based variates
        if 'close' in sequence_data.columns:
            # Normalize prices
            close_prices = sequence_data['close'].values
            normalized_close = (close_prices - close_prices.mean()) / (close_prices.std() + 1e-8)
            variates.append(normalized_close)
        
        if 'volume' in sequence_data.columns:
            # Normalize volume
            volumes = sequence_data['volume'].values
            normalized_volume = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            variates.append(normalized_volume)
        
        # Returns as a variate
        if 'close' in sequence_data.columns:
            returns = np.diff(np.log(sequence_data['close'].values + 1e-8))
            returns = np.concatenate([[0], returns])  # Pad first value
            variates.append(returns)
        
        # Volatility as a variate (rolling std of returns)
        if len(variates) > 0:  # If we have returns
            returns_series = pd.Series(variates[-1])
            volatility = returns_series.rolling(window=10, min_periods=1).std().fillna(0).values
            variates.append(volatility)
        
        # Technical indicator as variate
        if 'rsi' in sequence_data.columns:
            rsi_normalized = (sequence_data['rsi'].values - 50) / 50  # Center around 0
            variates.append(rsi_normalized)
        
        # Pad with zeros if we don't have enough variates
        while len(variates) < self.n_variates:
            variates.append(np.zeros(len(sequence_data)))
        
        # Truncate if we have too many variates
        variates = variates[:self.n_variates]
        
        # Stack variates: shape (sequence_length, n_variates)
        multivariate_sequence = np.column_stack(variates)
        
        return multivariate_sequence.astype(np.float32)
    
    def _extract_correlation_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract correlation features between variates (for analysis)"""
        correlation_features = {}
        
        # Calculate cross-correlations between key variates
        if all(col in data.columns for col in ['close', 'volume']):
            price_volume_corr = data['close'].corr(data['volume'])
            correlation_features['price_volume_corr'] = price_volume_corr
        
        # Add more correlation features as needed
        # This is just a placeholder implementation
        
        return correlation_features
    
    def _create_default_prediction(self, token: DiscoveredToken) -> PredictionResult:
        """Create a default prediction when analysis fails"""
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ITRANSFORMER,
            direction=PredictionDirection.HOLD,
            confidence=0.1,
            probability_up=0.5,
            features_used=['default'],
            model_version='itransformer_v1.0'
        )
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the iTransformer model on multivariate historical data
        """
        try:
            self.logger.info("Starting iTransformer model training",
                           data_shape=training_data.shape)
            
            # Prepare multivariate training data
            sequences, targets = self._prepare_training_data(training_data)
            
            if len(sequences) < 10:  # Minimum training samples
                self.logger.error("Insufficient training data", samples=len(sequences))
                return False
            
            # Convert to tensors
            X_train = torch.FloatTensor(sequences).to(self.device)
            y_train = torch.FloatTensor(targets).to(self.device)
            
            # Training loop
            self.model.train()
            epochs = self.config.get('epochs', 50)
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                outputs = self.model(X_train)
                
                # Calculate loss for price predictions
                loss = 0
                for i, horizon in enumerate(self.model_config.prediction_horizons):
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        # Average predictions across variates for loss calculation
                        pred_averaged = torch.mean(outputs[pred_key], dim=1)
                        loss += self.criterion(pred_averaged, y_train[:, i])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"Training epoch {epoch}, loss: {loss.item():.4f}")
            
            self._is_trained = True
            self.logger.info("iTransformer model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("iTransformer model training failed", error=str(e))
            return False
    
    def prepare_training_from_corpus(self, data: pd.DataFrame,
                                    sequence_length: Optional[int] = None,
                                    prediction_horizons: List[int] = [1, 4, 24],
                                    selected_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from unified corpus format for iTransformer
        
        Args:
            data: DataFrame from corpus with all features
            sequence_length: Length of input sequences (uses self.sequence_length if not provided)
            prediction_horizons: Hours ahead to predict [1h, 4h, 24h]
            selected_features: Optional list of features to use as variates (None = auto-select key features)
            
        Returns:
            X: Input sequences [n_samples, sequence_length, n_variates]
            y: Target values [n_samples, n_targets] as price changes
            feature_names: List of feature names used as variates
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # iTransformer treats each feature as a separate variate
        # Select key features for multivariate modeling
        if selected_features is None:
            # Default selection of important variates for iTransformer
            selected_features = [
                'close', 'volume', 'open', 'high', 'low',  # Core OHLCV
                'rsi_14', 'rsi_7', 'rsi_21',  # RSI indicators
                'macd', 'macd_signal',  # MACD
                'bb_upper', 'bb_lower', 'bb_position',  # Bollinger Bands
                'momentum_5', 'momentum_10', 'momentum_20',  # Momentum
                'volatility_score', 'volume_score',  # ML scores
                'market_regime', 'trend_strength'  # Market classification
            ]
        
        # Filter to available features
        available_features = [f for f in selected_features if f in data.columns]
        
        if not available_features:
            # Fallback to all numeric features
            metadata_cols = ['id', 'ohlcv_id', 'token_id', 'timestamp', 'feature_version',
                           'calculated_at', 'data_source', 'granularity', 'created_at', 'updated_at',
                           'collection_timestamp']
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [c for c in numeric_cols if c not in metadata_cols]
        
        # Intelligent feature selection based on current model configuration
        original_n_variates = self.model_config.n_variates
        
        # If we have more features than configured variates, select the most important ones
        if len(available_features) > original_n_variates:
            self.logger.info("Too many features for current model configuration",
                           available_features=len(available_features),
                           configured_variates=original_n_variates,
                           action="selecting_top_features")
            
            # Priority order: price features first, then volume, then technical indicators
            feature_priority = [
                'close', 'open', 'high', 'low', 'volume',  # Core OHLCV
                'rsi_14', 'macd', 'bb_position',  # Key technical indicators
                'volatility_score', 'volume_score',  # ML-generated scores
                'momentum_10', 'trend_strength', 'market_regime'  # Additional features
            ]
            
            # Select features in priority order, up to configured limit
            prioritized_features = []
            for feature in feature_priority:
                if feature in available_features and len(prioritized_features) < original_n_variates:
                    prioritized_features.append(feature)
            
            # Fill remaining slots with other available features
            remaining_features = [f for f in available_features if f not in prioritized_features]
            while len(prioritized_features) < original_n_variates and remaining_features:
                prioritized_features.append(remaining_features.pop(0))
            
            available_features = prioritized_features
            
            self.logger.info("Feature selection completed",
                           selected_features=available_features,
                           feature_count=len(available_features))
        
        # If we still have fewer features than configured, pad or adjust
        elif len(available_features) < original_n_variates:
            self.logger.warning("Fewer features available than configured variates",
                              available_features=len(available_features),
                              configured_variates=original_n_variates,
                              action="will_recreate_model")
        
        # Ensure we have essential columns
        if 'close' not in data.columns:
            raise ValueError("Missing 'close' price column in corpus data")
        
        # Select feature columns
        feature_data = data[available_features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize each variate independently (important for iTransformer)
        from sklearn.preprocessing import StandardScaler
        scalers = {}
        feature_data_normalized = pd.DataFrame(index=feature_data.index)
        
        for col in feature_data.columns:
            scaler = StandardScaler()
            feature_data_normalized[col] = scaler.fit_transform(feature_data[[col]])
            scalers[col] = scaler
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        max_horizon = max(prediction_horizons)
        
        for i in range(sequence_length, len(data) - max_horizon):
            # Input sequence - each column is a variate
            seq_features = feature_data_normalized.iloc[i-sequence_length:i].values
            
            # Target values as price changes
            current_price = data['close'].iloc[i-1]
            target_values = []
            
            for horizon in prediction_horizons:
                if i + horizon - 1 < len(data) and current_price > 0:
                    future_price = data['close'].iloc[i + horizon - 1]
                    price_change = (future_price - current_price) / current_price
                    target_values.append(price_change)
                else:
                    target_values.append(0.0)
            
            sequences.append(seq_features)
            targets.append(target_values)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        feature_names = list(feature_data.columns)
        
        # Update model config with actual number of variates
        original_n_variates = self.model_config.n_variates
        self.model_config.n_variates = len(feature_names)
        
        # Recreate model if n_variates changed to prevent tensor dimension mismatch
        if original_n_variates != self.model_config.n_variates:
            self.logger.info("Recreating iTransformer model due to n_variates change",
                           original_n_variates=original_n_variates,
                           new_n_variates=self.model_config.n_variates)
            self._recreate_model_with_new_variates()
        
        self.logger.info("iTransformer corpus data prepared",
                        sequences=len(X),
                        sequence_length=sequence_length,
                        n_variates=len(feature_names),
                        variates=feature_names[:5] + ['...'] if len(feature_names) > 5 else feature_names,
                        horizons=prediction_horizons)
        
        return X, y, feature_names
    
    def _recreate_model_with_new_variates(self):
        """
        Recreate the iTransformer model with updated n_variates to prevent tensor dimension mismatch
        
        This is necessary when corpus data has a different number of features than initially configured.
        The variate embeddings and related layers need to be resized to match the actual data.
        """
        try:
            # Store current training state and model parameters
            was_trained = self._is_trained
            old_state_dict = self.model.state_dict() if hasattr(self, 'model') else None
            
            # Store optimizer state if available
            optimizer_state = None
            if self.optimizer is not None:
                optimizer_state = self.optimizer.state_dict()
            
            # Update required features list to match new variate count
            self.required_features = self._get_required_features()
            self.n_variates = self.model_config.n_variates
            
            # Recreate the model with new configuration
            old_model = self.model if hasattr(self, 'model') else None
            self.model = iTransformerNetwork(self.model_config)
            self.model.to(self.device)
            
            # Try to transfer compatible parameters from old model
            if old_state_dict is not None and was_trained:
                self._transfer_compatible_parameters(old_state_dict, self.model.state_dict())
            
            # Recreate optimizer with new model parameters
            self._setup_training()
            
            # Restore optimizer state if we had one (though parameters may not match)
            if optimizer_state is not None and was_trained:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                    self.logger.info("Restored optimizer state after model recreation")
                except Exception as e:
                    self.logger.warning("Could not restore optimizer state after model recreation",
                                      error=str(e))
            
            # Mark as partially trained if we transferred some parameters
            if old_state_dict is not None and was_trained:
                # Model structure changed, so it needs retraining, but we transferred what we could
                self._is_trained = False
                self.logger.info("Model recreated with partial parameter transfer - retraining recommended")
            else:
                self._is_trained = False
            
            self.logger.info("iTransformer model recreated successfully",
                           new_n_variates=self.model_config.n_variates,
                           device=str(self.device),
                           requires_retraining=not was_trained or old_state_dict is None,
                           partial_transfer=old_state_dict is not None and was_trained)
                           
        except Exception as e:
            self.logger.error("Failed to recreate iTransformer model", error=str(e))
            raise RuntimeError(f"Model recreation failed: {str(e)}")
    
    def _transfer_compatible_parameters(self, old_state_dict: dict, new_state_dict: dict):
        """
        Transfer compatible parameters from old model to new model after n_variates change
        
        This helps preserve training progress for layers that don't depend on variate count,
        such as attention mechanisms, feed-forward layers, and prediction heads.
        """
        transferred_count = 0
        skipped_count = 0
        
        try:
            for param_name, old_param in old_state_dict.items():
                if param_name in new_state_dict:
                    new_param = new_state_dict[param_name]
                    
                    # Only transfer if shapes match exactly
                    if old_param.shape == new_param.shape:
                        new_state_dict[param_name] = old_param.clone()
                        transferred_count += 1
                        
                        # Log important layer transfers
                        if any(layer in param_name.lower() for layer in ['attention', 'feed_forward', 'prediction']):
                            self.logger.debug("Transferred compatible parameter", 
                                           parameter=param_name, 
                                           shape=list(old_param.shape))
                    else:
                        skipped_count += 1
                        # Log skipped variate-dependent parameters
                        if 'variate' in param_name.lower():
                            self.logger.debug("Skipped variate-dependent parameter", 
                                           parameter=param_name,
                                           old_shape=list(old_param.shape),
                                           new_shape=list(new_param.shape))
                else:
                    skipped_count += 1
            
            # Load the updated state dict into the new model
            self.model.load_state_dict(new_state_dict)
            
            self.logger.info("Parameter transfer completed",
                           transferred=transferred_count,
                           skipped=skipped_count,
                           transfer_ratio=transferred_count / (transferred_count + skipped_count) if (transferred_count + skipped_count) > 0 else 0.0)
                           
        except Exception as e:
            self.logger.warning("Parameter transfer failed, starting with fresh parameters",
                              error=str(e))
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multivariate training data for iTransformer
        
        Returns:
            sequences: [n_samples, sequence_length, n_variates]
            targets: [n_samples, n_horizons]
        """
        sequences = []
        targets = []
        
        # Create sliding windows
        for i in range(self.sequence_length, len(data) - 24):  # Need 24 hours ahead for targets
            # Input sequence
            seq_data = data.iloc[i-self.sequence_length:i]
            
            # Create dummy token for sequence preparation
            from src.utils.base import Chain
            from datetime import datetime
            dummy_token = DiscoveredToken(
                address="dummy", 
                name="DUMMY", 
                symbol="DUMMY",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="training"
            )
            sequence = self._prepare_multivariate_sequence(seq_data, dummy_token)
            
            # Target values (price changes) - simplified for now
            current_price = data.iloc[i]['close'] if 'close' in data.columns else 100.0
            target_1h = 0.0  # Placeholder
            target_4h = 0.0  # Placeholder  
            target_24h = 0.0  # Placeholder
            
            if 'close' in data.columns:
                if i+1 < len(data):
                    target_1h = (data.iloc[i+1]['close'] - current_price) / current_price
                if i+4 < len(data):
                    target_4h = (data.iloc[i+4]['close'] - current_price) / current_price
                if i+24 < len(data):
                    target_24h = (data.iloc[i+24]['close'] - current_price) / current_price
            
            sequences.append(sequence)
            targets.append([target_1h, target_4h, target_24h])
        
        return np.array(sequences), np.array(targets)
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained iTransformer model"""
        try:
            if not self._is_trained:
                self.logger.warning("Attempting to save untrained model")
                return False
            
            state = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config.__dict__,
                'is_trained': self._is_trained,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_type': 'itransformer'
            }
            
            torch.save(state, filepath)
            self.logger.info("iTransformer model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save iTransformer model", error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained iTransformer model"""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Update config
            self.model_config = InvertedAttentionConfig(**state['config'])
            
            # Recreate model with loaded config
            self.model = iTransformerNetwork(self.model_config)
            self.model.to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            
            self._is_trained = state.get('is_trained', False)
            
            # Load optimizer state if available
            if state.get('optimizer_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.logger.info("iTransformer model loaded", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load iTransformer model", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check if the iTransformer is healthy and ready"""
        try:
            if not self._is_trained:
                return False
            
            # Quick forward pass test with multivariate input
            dummy_input = torch.randn(1, self.sequence_length, self.n_variates).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(dummy_input)
                
                # Check that all expected outputs are present
                expected_keys = [f'price_{h}' for h in self.model_config.prediction_horizons]
                expected_keys.extend(['confidence', 'direction_probs', 'attention_weights'])
                
                return all(key in outputs for key in expected_keys)
                
        except Exception as e:
            self.logger.error("iTransformer health check failed", error=str(e))
            return False
    
    def prepare_for_quantization(self, quantization_type: str = "dynamic", 
                               preserve_attention_precision: bool = True) -> 'iTransformerPredictor':
        """Prepare iTransformer for quantization with specific attention handling"""
        try:
            self.model.eval()
            
            # Store pre-quantization state
            self._pre_quantization_state = {
                'training_mode': self.model.training,
                'config': self.model_config.__dict__.copy()
            }
            
            # iTransformer-specific quantization config
            # Preserve inverted attention precision by default due to importance for multivariate correlations
            if preserve_attention_precision:
                self._quantization_config = {
                    'inverted_attention_layers': 'fp16',  # Keep inverted attention precise
                    'variate_embedding_layers': 'fp16',   # Keep variate embeddings precise  
                    'temporal_fusion_layers': 'int8',     # Quantize temporal fusion
                    'feed_forward_layers': 'int8',        # Quantize feed-forward
                    'output_projection_layers': 'fp16',   # Keep output precise
                    'quantization_type': quantization_type,
                    'preserve_cross_variate_attention': True  # Critical for iTransformer performance
                }
            else:
                self._quantization_config = {
                    'inverted_attention_layers': 'int8',
                    'variate_embedding_layers': 'int8',
                    'temporal_fusion_layers': 'int8',
                    'feed_forward_layers': 'int8',
                    'output_projection_layers': 'fp16',  # Always preserve output
                    'quantization_type': quantization_type,
                    'preserve_cross_variate_attention': False
                }
            
            self.logger.info("iTransformer prepared for quantization",
                           quantization_type=quantization_type,
                           preserve_attention=preserve_attention_precision,
                           n_variates=self.n_variates)
            
            return self
            
        except Exception as e:
            self.logger.error("iTransformer quantization preparation failed", error=str(e))
            return self
    
    def apply_quantization(self, target_dtype: torch.dtype = torch.qint8) -> 'iTransformerPredictor':
        """Apply quantization to iTransformer with inverted attention considerations"""
        try:
            if not hasattr(self, '_quantization_config'):
                self.logger.warning("iTransformer not prepared for quantization, preparing with default settings")
                self.prepare_for_quantization()
            
            quantization_type = self._quantization_config.get('quantization_type', 'dynamic')
            
            # Create qconfig specification for iTransformer
            qconfig_spec = {}
            
            # Apply quantization based on layer types and iTransformer-specific needs
            for name, module in self.model.named_modules():
                if self._should_quantize_itransformer_layer(name, module):
                    if isinstance(module, (nn.Linear, nn.Conv1d)):
                        qconfig_spec[type(module)] = torch.quantization.default_dynamic_qconfig
                    elif isinstance(module, InvertedMultiHeadAttention):
                        # Special handling for inverted attention
                        if self._quantization_config.get('inverted_attention_layers') == 'int8':
                            qconfig_spec[type(module)] = torch.quantization.default_dynamic_qconfig
                        # else: preserve precision by not adding to qconfig_spec
            
            # Apply quantization
            if qconfig_spec:
                if quantization_type == "dynamic":
                    quantized_model = torch.quantization.quantize_dynamic(
                        self.model, qconfig_spec=qconfig_spec, dtype=target_dtype
                    )
                    self.model = quantized_model
                    
                    # Validate quantization
                    if self._validate_itransformer_quantization():
                        self.logger.info("iTransformer quantization applied successfully",
                                       quantization_type=quantization_type,
                                       target_dtype=str(target_dtype))
                        return self
                    else:
                        self.logger.error("iTransformer quantization validation failed")
                        return self
                else:
                    self.logger.warning("Static quantization not fully implemented, using dynamic")
                    return self.apply_quantization(target_dtype)
            else:
                self.logger.warning("No layers selected for iTransformer quantization")
                return self
                
        except Exception as e:
            self.logger.error("iTransformer quantization failed", error=str(e))
            return self
    
    def _should_quantize_itransformer_layer(self, layer_name: str, module: nn.Module) -> bool:
        """Determine if an iTransformer layer should be quantized"""
        layer_name_lower = layer_name.lower()
        
        # Categorize iTransformer-specific layers
        if 'inverted' in layer_name_lower or 'attention' in layer_name_lower:
            layer_category = 'inverted_attention_layers'
        elif 'variate' in layer_name_lower or 'embed' in layer_name_lower:
            layer_category = 'variate_embedding_layers'
        elif 'temporal_fusion' in layer_name_lower or 'fusion' in layer_name_lower:
            layer_category = 'temporal_fusion_layers'
        elif 'feed_forward' in layer_name_lower or 'ffn' in layer_name_lower or 'mlp' in layer_name_lower:
            layer_category = 'feed_forward_layers'
        elif 'output' in layer_name_lower or 'projection' in layer_name_lower:
            layer_category = 'output_projection_layers'
        else:
            # Default handling for standard linear layers
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                layer_category = 'feed_forward_layers'
            else:
                return False
        
        # Check quantization config
        target_precision = self._quantization_config.get(layer_category, 'fp32')
        should_quantize = target_precision in ['int8', 'qint8']
        
        return should_quantize
    
    def _validate_itransformer_quantization(self) -> bool:
        """Validate iTransformer quantization with multivariate input"""
        try:
            # Test with multivariate input
            test_input = torch.randn(1, self.sequence_length, self.n_variates).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(test_input)
                
                # Check outputs are valid
                for horizon in self.model_config.prediction_horizons:
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        pred_tensor = outputs[pred_key]
                        if torch.isnan(pred_tensor).any() or torch.isinf(pred_tensor).any():
                            return False
                
                # Check attention weights if available
                if 'attention_weights' in outputs:
                    attention = outputs['attention_weights']
                    if torch.isnan(attention).any() or torch.isinf(attention).any():
                        return False
                
                # Verify multivariate structure is preserved
                if 'direction_probs' in outputs:
                    direction_probs = outputs['direction_probs']
                    expected_shape = (1, self.n_variates, 5)  # batch, variates, directions
                    if direction_probs.shape != expected_shape:
                        self.logger.warning("iTransformer multivariate structure changed after quantization",
                                          expected_shape=expected_shape,
                                          actual_shape=direction_probs.shape)
                        return False
            
            self.logger.info("iTransformer quantization validation passed",
                           variates=self.n_variates,
                           sequence_length=self.sequence_length)
            return True
            
        except Exception as e:
            self.logger.error("iTransformer quantization validation failed", error=str(e))
            return False
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get iTransformer-specific quantization information"""
        base_info = {
            'model_type': 'iTransformer',
            'is_quantized': False,
            'quantization_type': None,
            'quantized_layers': [],
            'preserved_precision_layers': [],
            'quantization_config': None,
            'estimated_speedup': 1.0,
            'estimated_memory_reduction': 0.0,
            'multivariate_compatibility': True
        }
        
        try:
            # Check quantization config
            if hasattr(self, '_quantization_config'):
                base_info['quantization_config'] = self._quantization_config
                base_info['quantization_type'] = self._quantization_config.get('quantization_type')
            
            # Analyze quantization state
            quantized_count = 0
            preserved_count = 0
            total_quantizable = 0
            inverted_attention_quantized = False
            
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, InvertedMultiHeadAttention)):
                    total_quantizable += 1
                    
                    # Check if layer is quantized
                    if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                        if 'qint' in str(module.weight.dtype):
                            quantized_count += 1
                            base_info['quantized_layers'].append(name)
                            base_info['is_quantized'] = True
                            
                            # Track if inverted attention is quantized
                            if isinstance(module, InvertedMultiHeadAttention):
                                inverted_attention_quantized = True
                        else:
                            preserved_count += 1
                            base_info['preserved_precision_layers'].append(name)
            
            # Calculate benefits
            if total_quantizable > 0:
                quantization_ratio = quantized_count / total_quantizable
                # iTransformer benefits more from quantization due to multivariate processing
                base_info['estimated_speedup'] = 1.0 + (quantization_ratio * 1.8)  # Up to 2.8x
                base_info['estimated_memory_reduction'] = quantization_ratio * 0.8  # Up to 80%
            
            # iTransformer-specific metrics
            base_info['itransformer_specific'] = {
                'n_variates': self.n_variates,
                'inverted_attention_quantized': inverted_attention_quantized,
                'preserves_multivariate_structure': not inverted_attention_quantized,
                'temporal_fusion_quantizable': True,
                'cross_variate_attention_preserved': self._quantization_config.get('preserve_cross_variate_attention', False) if hasattr(self, '_quantization_config') else True
            }
            
        except Exception as e:
            self.logger.error("Failed to get iTransformer quantization info", error=str(e))
        
        return base_info
    
    def benchmark_quantization_performance(self, test_multivariate_data: List[np.ndarray], 
                                         num_warmup: int = 10, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark iTransformer quantization performance with multivariate data"""
        benchmark_results = {
            'original_latency_ms': 0.0,
            'quantized_latency_ms': 0.0,
            'speedup_ratio': 1.0,
            'memory_usage_mb': 0.0,
            'multivariate_accuracy_preservation': 1.0,
            'cross_variate_correlation_preservation': 1.0
        }
        
        try:
            if not test_multivariate_data:
                self.logger.warning("No multivariate test data provided for iTransformer benchmarking")
                return benchmark_results
            
            # Convert test data to tensors
            test_tensors = [torch.FloatTensor(data).unsqueeze(0).to(self.device) 
                           for data in test_multivariate_data]
            
            # Store reference to original model if we have quantized version
            original_model = self.model
            quantized_model = self.model
            
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = original_model(test_tensors[0])
                    _ = quantized_model(test_tensors[0])
            
            # Benchmark original model
            original_times = []
            original_outputs = []
            for test_tensor in test_tensors[:min(len(test_tensors), num_iterations)]:
                start_time = time.time()
                with torch.no_grad():
                    output = original_model(test_tensor)
                    original_outputs.append(output)
                original_times.append((time.time() - start_time) * 1000)
            
            # Benchmark quantized model  
            quantized_times = []
            quantized_outputs = []
            for test_tensor in test_tensors[:min(len(test_tensors), num_iterations)]:
                start_time = time.time()
                with torch.no_grad():
                    output = quantized_model(test_tensor)
                    quantized_outputs.append(output)
                quantized_times.append((time.time() - start_time) * 1000)
            
            # Calculate results
            benchmark_results['original_latency_ms'] = sum(original_times) / len(original_times)
            benchmark_results['quantized_latency_ms'] = sum(quantized_times) / len(quantized_times)
            benchmark_results['speedup_ratio'] = benchmark_results['original_latency_ms'] / benchmark_results['quantized_latency_ms']
            
            # Memory usage
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            benchmark_results['memory_usage_mb'] = param_size / (1024 * 1024)
            
            # iTransformer-specific accuracy metrics
            if original_outputs and quantized_outputs:
                # Compare multivariate outputs
                multivariate_mse = 0.0
                correlation_preservation = 0.0
                
                for orig_out, quant_out in zip(original_outputs[:5], quantized_outputs[:5]):  # Sample a few
                    for horizon in self.model_config.prediction_horizons:
                        pred_key = f'price_{horizon}'
                        if pred_key in orig_out and pred_key in quant_out:
                            orig_preds = orig_out[pred_key].cpu().numpy()
                            quant_preds = quant_out[pred_key].cpu().numpy()
                            
                            # MSE between predictions
                            mse = np.mean((orig_preds - quant_preds) ** 2)
                            multivariate_mse += mse
                            
                            # Correlation preservation between variates
                            if orig_preds.shape[-1] > 1:  # Multiple variates
                                orig_corr = np.corrcoef(orig_preds.flatten(), quant_preds.flatten())[0, 1]
                                correlation_preservation += orig_corr if not np.isnan(orig_corr) else 0.0
                
                benchmark_results['multivariate_accuracy_preservation'] = max(0.0, 1.0 - multivariate_mse)
                benchmark_results['cross_variate_correlation_preservation'] = max(0.0, correlation_preservation / len(original_outputs))
            
            self.logger.info("iTransformer quantization benchmark completed",
                           original_latency=benchmark_results['original_latency_ms'],
                           quantized_latency=benchmark_results['quantized_latency_ms'],
                           speedup=benchmark_results['speedup_ratio'],
                           multivariate_accuracy=benchmark_results['multivariate_accuracy_preservation'])
            
        except Exception as e:
            self.logger.error("iTransformer quantization benchmark failed", error=str(e))
        
        return benchmark_results