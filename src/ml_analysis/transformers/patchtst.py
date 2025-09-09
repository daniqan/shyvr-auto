"""
PatchTST Implementation for Long-Horizon Time-Series Forecasting

PatchTST uses patch-based tokenization to handle long sequences efficiently,
treating time-series patches as tokens. This approach enables better long-horizon
forecasting while maintaining computational efficiency.

Key Features:
1. Patch-based tokenization for long sequences
2. Channel independence for multivariate time series
3. Efficient attention computation on patches rather than raw time points
4. Multi-horizon prediction heads (1h, 4h, 24h)

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
"""

import math
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
from ..base import MLAnalyzerBase, ModelType, PredictionResult, PredictionDirection, ModelNotTrainedError
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()

# Control verbosity - set to INFO level by default to reduce debug spam
import logging
import os
if os.getenv('PATCHTST_DEBUG', 'false').lower() != 'true':
    # Suppress debug messages unless explicitly enabled
    logging.getLogger('src.ml_analysis.transformers.patchtst').setLevel(logging.INFO)


@dataclass
class PatchTSTConfig(TransformerConfig):
    """Configuration for PatchTST with patch-based tokenization"""
    
    # PatchTST specific parameters
    patch_length: int = 16              # Length of each patch
    stride: int = 8                     # Stride between patches
    n_channels: int = 15                # Number of channels/features - increased for multi-channel support
    channel_independence: bool = True    # Process channels independently
    
    # Patch-specific model parameters
    d_model: int = 128                  # Smaller default for efficiency
    n_layers: int = 3                   # Fewer layers than standard transformer
    max_seq_length: int = 1000          # Maximum input sequence length
    
    # Multi-horizon forecasting
    prediction_horizons: List[str] = None  # ['1h', '4h', '24h']
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        super().__post_init__()
        
        # Validate patch parameters
        if self.patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.stride > self.patch_length:
            raise ValueError("patch_length must be greater than stride")
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        
        # Set defaults
        if self.prediction_horizons is None:
            self.prediction_horizons = ['1h', '4h', '24h']
    
    def get_n_patches(self, seq_length: Optional[int] = None) -> int:
        """Calculate number of patches for a given sequence length"""
        seq_len = seq_length or self.max_seq_length
        if seq_len < self.patch_length:
            return 0
        return (seq_len - self.patch_length) // self.stride + 1
    
    def estimate_memory_usage_mb(self) -> float:
        """Estimate memory usage for PatchTST"""
        base_memory = super().estimate_memory_usage_mb()
        
        # PatchTST is more memory efficient due to patch-based attention
        # Attention operates on patches rather than full sequence
        n_patches = self.get_n_patches()
        patch_attention_memory = n_patches * n_patches * 4 / (1024 * 1024)  # Much smaller than full sequence
        
        # Channel independence reduces memory if enabled
        channel_factor = 1.0 if not self.channel_independence else 0.8
        
        return (base_memory * 0.6 + patch_attention_memory) * channel_factor  # ~40% reduction


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts time series patches to embeddings
    Each patch of length patch_length is projected to d_model dimensions
    """
    
    def __init__(self, patch_length: int, d_model: int):
        super().__init__()
        self.patch_length = patch_length
        self.d_model = d_model
        
        # Linear projection from patch to embedding
        self.projection = nn.Linear(patch_length, d_model)
        
        logger.debug("PatchEmbedding initialized",
                    patch_length=patch_length, d_model=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series patches to embeddings
        
        Args:
            x: Input tensor [batch_size, n_patches, patch_length]
            
        Returns:
            Embedded patches [batch_size, n_patches, d_model]
        """
        return self.projection(x)


class PatchTSTNetwork(TransformerBase):
    """
    PatchTST Network with patch-based tokenization
    
    Key innovations:
    1. Patch-based tokenization for long sequences
    2. Channel independence for multivariate processing
    3. Efficient attention on patches rather than time points
    4. Multi-horizon prediction capability
    """
    
    def __init__(self, config: PatchTSTConfig):
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
        super().__init__(ModelType.PATCHTST, base_config)
        
        self.config = config
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            patch_length=config.patch_length,
            d_model=config.d_model
        )
        
        # Channel embeddings for multivariate data
        if config.channel_independence:
            self.channel_embeddings = nn.Embedding(config.n_channels, config.d_model)
        else:
            self.channel_embeddings = None
        
        # Positional encoding for patch positions
        pos_config = PositionalEncodingConfig(
            d_model=config.d_model,
            max_length=config.get_n_patches() + 10,  # Extra buffer for varying lengths
            dropout=config.dropout
        )
        self.pos_encoding = create_positional_encoding('sinusoidal', pos_config)
        
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
        
        # Multi-horizon prediction heads
        if config.channel_independence:
            # Separate predictions for each channel
            self.prediction_heads = nn.ModuleDict({
                horizon: nn.Linear(config.d_model, config.n_channels)
                for horizon in config.prediction_horizons
            })
        else:
            # Single prediction aggregated across channels
            self.prediction_heads = nn.ModuleDict({
                horizon: nn.Linear(config.d_model, 1)
                for horizon in config.prediction_horizons
            })
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(config.d_model, 1)
        
        # Direction classification head
        if config.channel_independence:
            self.direction_head = nn.Linear(config.d_model, config.n_channels * 5)  # 5 classes per channel
        else:
            self.direction_head = nn.Linear(config.d_model, 5)  # Single prediction
        
        # Initialize parameters
        self._init_weights()
        
        logger.info("PatchTSTNetwork initialized",
                   d_model=config.d_model,
                   n_channels=config.n_channels,
                   patch_length=config.patch_length,
                   stride=config.stride,
                   prediction_horizons=config.prediction_horizons)
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patches from input time series
        
        Args:
            x: Input tensor [batch_size, seq_len, n_channels]
            
        Returns:
            Patches tensor [batch_size, n_channels, n_patches, patch_length]
        """
        batch_size, seq_len, n_channels = x.shape
        
        if seq_len < self.config.patch_length:
            raise ValueError(f"Sequence length {seq_len} too short for patch_length {self.config.patch_length}")
        
        # Calculate number of patches
        n_patches = (seq_len - self.config.patch_length) // self.config.stride + 1
        
        # Create patches using unfold
        # x: [batch_size, seq_len, n_channels] -> [batch_size, n_channels, seq_len]
        x_transposed = x.transpose(1, 2)
        
        # Create patches: [batch_size, n_channels, n_patches, patch_length]
        patches = x_transposed.unfold(2, self.config.patch_length, self.config.stride)
        
        return patches
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with patch-based processing
        
        Args:
            x: Input tensor [batch_size, seq_len, n_channels]
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with predictions and features
        """
        batch_size, seq_len, n_channels = x.shape
        
        if n_channels != self.config.n_channels:
            raise ValueError(f"Expected {self.config.n_channels} channels, got {n_channels}")
        
        # Create patches: [batch_size, n_channels, n_patches, patch_length]
        patches = self._create_patches(x)
        _, _, n_patches, patch_length = patches.shape
        
        outputs = {}
        
        if self.config.channel_independence:
            # Process each channel independently
            channel_outputs = []
            
            for channel_idx in range(n_channels):
                # Get patches for this channel: [batch_size, n_patches, patch_length]
                channel_patches = patches[:, channel_idx, :, :]
                
                # Embed patches: [batch_size, n_patches, d_model]
                embedded_patches = self.patch_embedding(channel_patches)
                
                # Add channel embedding
                if self.channel_embeddings is not None:
                    channel_emb = self.channel_embeddings(torch.tensor(channel_idx, device=x.device))
                    embedded_patches = embedded_patches + channel_emb.unsqueeze(0).unsqueeze(0)
                
                # Add positional encoding
                embedded_patches = self.pos_encoding(embedded_patches)
                
                # Apply transformer
                encoded_patches = self.transformer_encoder(embedded_patches)
                
                # Global pooling across patches for prediction
                channel_repr = torch.mean(encoded_patches, dim=1)  # [batch_size, d_model]
                channel_outputs.append(channel_repr)
            
            # Stack channel representations: [batch_size, n_channels, d_model]
            all_channel_repr = torch.stack(channel_outputs, dim=1)
            
            # Generate predictions for each channel
            for horizon, head in self.prediction_heads.items():
                # Apply head to each channel representation
                channel_preds = []
                for channel_idx in range(n_channels):
                    channel_pred = head(all_channel_repr[:, channel_idx, :])
                    if channel_pred.dim() == 2 and channel_pred.size(1) > 1:
                        # If head outputs multiple values, take the channel-specific one
                        channel_pred = channel_pred[:, channel_idx:channel_idx+1]
                    channel_preds.append(channel_pred)
                
                outputs[f'price_{horizon}'] = torch.cat(channel_preds, dim=-1)
            
            # Aggregate for confidence and direction
            global_repr = torch.mean(all_channel_repr, dim=1)  # [batch_size, d_model]
            
        else:
            # Process all channels together
            # Reshape patches: [batch_size * n_channels, n_patches, patch_length]
            all_patches = patches.view(batch_size * n_channels, n_patches, patch_length)
            
            # Embed patches
            embedded_patches = self.patch_embedding(all_patches)
            
            # Add positional encoding
            embedded_patches = self.pos_encoding(embedded_patches)
            
            # Apply transformer
            encoded_patches = self.transformer_encoder(embedded_patches)
            
            # Reshape back and aggregate across channels and patches
            encoded_patches = encoded_patches.view(batch_size, n_channels, n_patches, self.config.d_model)
            global_repr = torch.mean(encoded_patches, dim=(1, 2))  # [batch_size, d_model]
            
            # Generate single predictions
            for horizon, head in self.prediction_heads.items():
                outputs[f'price_{horizon}'] = head(global_repr)
        
        # Confidence estimation
        outputs['confidence'] = torch.sigmoid(self.confidence_head(global_repr))
        
        # Direction predictions
        direction_logits = self.direction_head(global_repr)
        if self.config.channel_independence:
            direction_logits = direction_logits.view(batch_size, self.config.n_channels, 5)
            outputs['direction_logits'] = direction_logits
            outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
        else:
            outputs['direction_logits'] = direction_logits
            outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for visualization (simplified)"""
        with torch.no_grad():
            # This is a simplified implementation
            # In practice, would need to modify transformer encoder to return attention weights
            batch_size, seq_len, n_channels = x.shape
            n_patches = (seq_len - self.config.patch_length) // self.config.stride + 1
            return torch.ones(1, self.config.n_heads, n_patches, n_patches)


class PatchTSTPredictor(MLAnalyzerBase):
    """
    PatchTST-based predictor for long-horizon time series forecasting
    Specialized for handling long sequences through patch-based tokenization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ModelType.PATCHTST, config)
        
        # Parse configuration
        self.model_config = self._parse_config(config or {})
        
        # Initialize model
        self.model = PatchTSTNetwork(self.model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self._setup_training()
        
        # Feature engineering
        self.patch_length = self.model_config.patch_length
        self.stride = self.model_config.stride
        self.n_channels = self.model_config.n_channels
        self.required_features = self._get_required_features()
        
        logger.info("PatchTSTPredictor initialized",
                   model_type=self.model_type,
                   patch_length=self.patch_length,
                   stride=self.stride,
                   n_channels=self.n_channels)
    
    def _parse_config(self, config: Dict) -> PatchTSTConfig:
        """Parse configuration dictionary into PatchTSTConfig"""
        return PatchTSTConfig(
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 3),
            d_ff=config.get('d_ff', 512),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            max_seq_length=config.get('max_seq_length', 1000),
            patch_length=config.get('patch_length', 16),
            stride=config.get('stride', 8),
            n_channels=config.get('n_channels', 15),
            channel_independence=config.get('channel_independence', True),
            prediction_horizons=config.get('prediction_horizons', ['1h', '4h', '24h'])
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
        """Get list of required features for PatchTST"""
        return self.required_features
    
    def _get_required_features(self) -> List[str]:
        """Define required features for PatchTST (patch-based processing)"""
        return [
            # Core price features for patch creation
            'close', 'open', 'high', 'low', 'volume',
            
            # Derived features for patches
            'returns', 'volatility', 'price_momentum',
            
            # Technical indicators
            'sma_20', 'ema_12', 'rsi', 'macd',
            
            # Patch-specific features
            'patch_sequence_data', 'long_term_trends',
            
            # Market context for long-horizon forecasting
            'market_regime', 'volatility_regime',
            'funding_rate', 'volume_profile',
            
            # Temporal features
            'hour_of_day', 'day_of_week', 'trading_session'
        ]
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Analyze token using PatchTST with patch-based processing
        """
        if not self._is_trained:
            raise ModelNotTrainedError("PatchTST model is not trained")
        
        min_required_length = self.patch_length * 2  # Need at least 2 patches
        if historical_data is None or len(historical_data) < min_required_length:
            self.logger.warning("Insufficient data for PatchTST prediction",
                              token=token.address,
                              data_length=len(historical_data) if historical_data is not None else 0,
                              required_length=min_required_length)
            return self._create_default_prediction(token)
        
        try:
            # Prepare patch-based input sequence
            input_sequence = self._prepare_patch_sequence(historical_data, token)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                
                # Extract predictions
                price_predictions = {}
                for horizon in self.model_config.prediction_horizons:
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        pred_values = outputs[pred_key].cpu().numpy().flatten()
                        if self.model_config.channel_independence:
                            # Average across channels for single prediction
                            price_predictions[horizon] = np.mean(pred_values)
                        else:
                            price_predictions[horizon] = pred_values[0]
                
                confidence = outputs['confidence'].item()
                
                # Handle direction probabilities
                direction_probs = outputs['direction_probs'].cpu().numpy()
                if self.model_config.channel_independence and direction_probs.ndim == 3:
                    # Average across channels: [batch, channels, classes] -> [classes]
                    avg_direction_probs = np.mean(direction_probs[0], axis=0)
                else:
                    avg_direction_probs = direction_probs.flatten()
            
            # Convert relative predictions to absolute prices
            current_price = token.price_usd or historical_data['close'].iloc[-1]
            
            price_pred_1h = current_price * (1 + price_predictions.get('1h', 0))
            price_pred_4h = current_price * (1 + price_predictions.get('4h', 0))
            price_pred_24h = current_price * (1 + price_predictions.get('24h', 0))
            
            # Determine direction from probabilities
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
                model_type=ModelType.PATCHTST,
                price_prediction_1h=price_pred_1h,
                price_prediction_4h=price_pred_4h,
                price_prediction_24h=price_pred_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                model_accuracy=0.80,  # Placeholder - would be tracked
                features_used=['patch_tokenization', 'channel_independence', 'long_horizon_forecasting'],
                model_version='patchtst_v1.0'
            )
            
            return result
            
        except Exception as e:
            self.logger.error("PatchTST prediction failed",
                            token=token.address,
                            error=str(e))
            return self._create_default_prediction(token)
    
    def _prepare_patch_sequence(self, data: pd.DataFrame, token: DiscoveredToken) -> np.ndarray:
        """
        Prepare input sequence for patch-based processing
        
        Returns:
            Numpy array of shape (sequence_length, n_channels)
        """
        # Take sufficient data for patch creation
        max_needed = self.model_config.max_seq_length
        sequence_data = data.tail(max_needed).copy()
        
        # Extract channels for patch processing
        channels = []
        
        # Core price channel
        if 'close' in sequence_data.columns:
            close_prices = sequence_data['close'].values
            # Normalize for better patch processing
            normalized_close = (close_prices - close_prices.mean()) / (close_prices.std() + 1e-8)
            channels.append(normalized_close)
        
        # Volume channel
        if 'volume' in sequence_data.columns and self.n_channels > 1:
            volumes = sequence_data['volume'].values
            normalized_volume = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            channels.append(normalized_volume)
        
        # Returns channel
        if self.n_channels > 2 and 'close' in sequence_data.columns:
            returns = np.diff(np.log(sequence_data['close'].values + 1e-8))
            returns = np.concatenate([[0], returns])  # Pad first value
            channels.append(returns)
        
        # Pad with zeros if we don't have enough channels
        while len(channels) < self.n_channels:
            channels.append(np.zeros(len(sequence_data)))
        
        # Truncate if we have too many channels
        channels = channels[:self.n_channels]
        
        # Stack channels: shape (sequence_length, n_channels)
        patch_sequence = np.column_stack(channels)
        
        return patch_sequence.astype(np.float32)
    
    def _create_default_prediction(self, token: DiscoveredToken) -> PredictionResult:
        """Create a default prediction when analysis fails"""
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.PATCHTST,
            direction=PredictionDirection.HOLD,
            confidence=0.1,
            probability_up=0.5,
            features_used=['default'],
            model_version='patchtst_v1.0'
        )
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the PatchTST model on historical data
        """
        try:
            self.logger.info("Starting PatchTST model training",
                           data_shape=training_data.shape)
            
            # Prepare patch-based training data
            sequences, targets = self._prepare_training_data(training_data)
            
            min_samples = self.patch_length  # Need at least patch_length samples
            if len(sequences) < min_samples:
                self.logger.error("Insufficient training data for patches", 
                                samples=len(sequences), 
                                required=min_samples)
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
                        predictions = outputs[pred_key]
                        if self.model_config.channel_independence:
                            # Average across channels for loss calculation
                            pred_averaged = torch.mean(predictions, dim=1)
                        else:
                            pred_averaged = predictions.squeeze()
                        loss += self.criterion(pred_averaged, y_train[:, i])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"Training epoch {epoch}, loss: {loss.item():.4f}")
            
            self._is_trained = True
            self.logger.info("PatchTST model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("PatchTST model training failed", error=str(e))
            return False
    
    def prepare_training_from_corpus(self, data: pd.DataFrame,
                                    sequence_length: Optional[int] = None,
                                    prediction_horizons: List[int] = [1, 4, 24],
                                    n_channels: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from unified corpus format for PatchTST
        
        Args:
            data: DataFrame from corpus with all features
            sequence_length: Length of input sequences (uses patch_length * 4 if not provided)
            prediction_horizons: Hours ahead to predict [1h, 4h, 24h]
            n_channels: Number of channels to use (None = use all features)
            
        Returns:
            X: Input sequences [n_samples, sequence_length, n_channels]
            y: Target values [n_samples, n_targets] as price changes
            feature_names: List of feature names used as channels
        """
        if sequence_length is None:
            # PatchTST needs sufficient length for patch creation
            sequence_length = self.patch_length * 4
            
        if n_channels is None:
            n_channels = self.n_channels
            
        # Identify metadata columns to exclude
        metadata_cols = ['id', 'ohlcv_id', 'token_id', 'timestamp', 'feature_version',
                        'calculated_at', 'data_source', 'granularity', 'created_at', 'updated_at',
                        'collection_timestamp']
        
        # Get all numeric columns except metadata
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in numeric_cols if c not in metadata_cols]
        
        # Ensure we have essential columns
        if 'close' not in data.columns:
            raise ValueError("Missing 'close' price column in corpus data")
        
        # Select top features based on importance for patches
        # Prioritize features that capture different aspects of market behavior
        priority_features = [
            'close', 'volume', 'open', 'high', 'low',  # Core OHLCV
            'returns_1h', 'returns_24h', 'returns_7d',  # Returns at different scales
            'rsi_14', 'rsi_7', 'rsi_21',  # RSI variations
            'macd', 'macd_signal', 'macd_histogram',  # MACD components
            'bb_upper', 'bb_lower', 'bb_position',  # Bollinger Bands
            'atr', 'adx',  # Volatility measures
            'momentum_5', 'momentum_10', 'momentum_20',  # Momentum
            'volatility_score', 'volume_score', 'momentum_score',  # ML scores
            'market_regime', 'trend_strength'  # Market state
        ]
        
        # Select features based on availability and channel limit
        selected_features = []
        for feat in priority_features:
            if feat in available_features and len(selected_features) < n_channels:
                selected_features.append(feat)
        
        # Fill remaining channels with other available features
        for feat in available_features:
            if feat not in selected_features and len(selected_features) < n_channels:
                selected_features.append(feat)
        
        # Ensure we have at least one feature
        if not selected_features:
            selected_features = available_features[:n_channels]
        
        # Select feature columns
        feature_data = data[selected_features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize features for patch processing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_data_normalized = pd.DataFrame(
            scaler.fit_transform(feature_data),
            columns=feature_data.columns,
            index=feature_data.index
        )
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        max_horizon = max(prediction_horizons)
        
        for i in range(sequence_length, len(data) - max_horizon):
            # Input sequence for patch creation
            seq_features = feature_data_normalized.iloc[i-sequence_length:i].values
            
            # Ensure sequence has correct shape for patches
            if len(seq_features) < sequence_length:
                # Pad if necessary
                padding = np.zeros((sequence_length - len(seq_features), len(selected_features)))
                seq_features = np.vstack([padding, seq_features])
            
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
        
        # Update model config with actual dimensions
        self.n_channels = len(feature_names)
        
        self.logger.info("PatchTST corpus data prepared",
                        sequences=len(X),
                        sequence_length=sequence_length,
                        n_channels=len(feature_names),
                        patch_length=self.patch_length,
                        channels=feature_names[:5] + ['...'] if len(feature_names) > 5 else feature_names,
                        horizons=prediction_horizons)
        
        return X, y, feature_names
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for patch-based processing
        
        Returns:
            sequences: [n_samples, sequence_length, n_channels]
            targets: [n_samples, n_horizons]
        """
        sequences = []
        targets = []
        
        min_seq_length = self.patch_length * 4  # Need sufficient data for patches
        
        # Create sliding windows
        for i in range(min_seq_length, len(data) - 24):  # Need 24 hours ahead for targets
            # Input sequence
            seq_data = data.iloc[i-min_seq_length:i]
            
            # Create dummy token for sequence preparation
            from src.utils.base import Chain
            dummy_token = DiscoveredToken(
                address="dummy", 
                name="DUMMY", 
                symbol="DUMMY",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="training"
            )
            sequence = self._prepare_patch_sequence(seq_data, dummy_token)
            
            # Target values (price changes)
            current_price = data.iloc[i]['close'] if 'close' in data.columns else 100.0
            target_1h = 0.0
            target_4h = 0.0
            target_24h = 0.0
            
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
        """Save the trained PatchTST model"""
        try:
            if not self._is_trained:
                self.logger.warning("Attempting to save untrained model")
                return False
            
            state = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config.__dict__,
                'is_trained': self._is_trained,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_type': 'patchtst'
            }
            
            torch.save(state, filepath)
            self.logger.info("PatchTST model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save PatchTST model", error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained PatchTST model"""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Update config
            self.model_config = PatchTSTConfig(**state['config'])
            
            # Recreate model with loaded config
            self.model = PatchTSTNetwork(self.model_config)
            self.model.to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            
            self._is_trained = state.get('is_trained', False)
            
            # Load optimizer state if available
            if state.get('optimizer_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.logger.info("PatchTST model loaded", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load PatchTST model", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check if the PatchTST is healthy and ready"""
        try:
            if not self._is_trained:
                return False
            
            # Quick forward pass test with patch-compatible input
            test_seq_length = self.patch_length * 3  # Sufficient for patches
            dummy_input = torch.randn(1, test_seq_length, self.n_channels).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(dummy_input)
                
                # Check that all expected outputs are present
                expected_keys = [f'price_{h}' for h in self.model_config.prediction_horizons]
                expected_keys.extend(['confidence', 'direction_probs'])
                
                return all(key in outputs for key in expected_keys)
                
        except Exception as e:
            self.logger.error("PatchTST health check failed", error=str(e))
            return False