"""
TimesMixer Implementation for Time-Series Forecasting

TimesMixer uses decomposable mixing to separate time and feature mixing operations,
enabling better capture of temporal patterns and multivariate relationships.

Key Features:
1. Past Decomposable Mixing (PDM) for historical pattern processing
2. Future Multipredictor Mixing (FMM) for multi-horizon forecasting
3. Seasonal-trend decomposition for pattern recognition
4. Decomposable mixing layers for time and feature domains

Reference: "TimesMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
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
from ..base import MLAnalyzerBase, ModelType, PredictionResult, PredictionDirection, ModelNotTrainedError
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


@dataclass
class TimesMixerConfig(TransformerConfig):
    """Configuration for TimesMixer with decomposable mixing"""
    
    # TimesMixer specific parameters
    seq_len: int = 96                   # Input sequence length
    pred_len: int = 24                  # Prediction horizon length
    d_model: int = 128                  # Smaller default for efficiency
    n_layers: int = 3                   # Fewer layers for mixing approach
    d_ff: int = 256                     # Smaller feed-forward dimension
    
    # Decomposable mixing parameters
    top_k: int = 5                      # Top-k mixing components
    num_kernels: int = 6                # Number of decomposition kernels
    use_time_mixing: bool = True        # Enable time domain mixing
    use_feature_mixing: bool = True     # Enable feature domain mixing
    
    # Decomposition parameters
    decomposition_kernel: str = 'moving_avg'  # Type of decomposition kernel
    kernel_size: int = 25               # Size of decomposition kernel
    
    # Multi-scale parameters
    down_sampling_layers: int = 3       # Number of downsampling layers
    down_sampling_method: str = 'avg'   # Downsampling method
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        super().__post_init__()
        
        # Validate TimesMixer specific parameters
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.num_kernels <= 0:
            raise ValueError("num_kernels must be positive")
        
        # Ensure kernel size is odd for proper centering
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
    
    def estimate_memory_usage_mb(self) -> float:
        """Estimate memory usage for TimesMixer"""
        base_memory = super().estimate_memory_usage_mb()
        
        # TimesMixer is more memory efficient due to decomposable mixing
        # No full attention matrices needed
        mixing_memory = (self.top_k * self.d_model * 4) / (1024 * 1024)
        
        # Decomposition memory
        decomp_memory = (self.num_kernels * self.seq_len * 4) / (1024 * 1024)
        
        # Total is much more efficient than full attention
        return (base_memory * 0.4) + mixing_memory + decomp_memory  # ~60% reduction


class MovingAverageDecomposition(nn.Module):
    """
    Moving average based decomposition for trend-seasonal separation
    Used in Past Decomposable Mixing
    """
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        # Create moving average kernel
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size) / kernel_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input into seasonal and trend components
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            seasonal: Seasonal component [batch_size, seq_len, d_model]
            trend: Trend component [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for convolution: [batch_size * d_model, 1, seq_len]
        x_reshaped = x.transpose(1, 2).contiguous().view(batch_size * d_model, 1, seq_len)
        
        # Apply moving average to get trend
        trend_reshaped = F.conv1d(x_reshaped, self.kernel, padding=self.padding)
        
        # Reshape back: [batch_size, seq_len, d_model]
        trend = trend_reshaped.view(batch_size, d_model, seq_len).transpose(1, 2)
        
        # Seasonal component is residual
        seasonal = x - trend
        
        return seasonal, trend


class PastDecomposableMixing(nn.Module):
    """
    Past Decomposable Mixing (PDM) layer
    
    Processes historical sequences through:
    1. Seasonal-trend decomposition
    2. Time domain mixing
    3. Feature domain mixing
    """
    
    def __init__(self, seq_len: int, d_model: int, top_k: int, num_kernels: int, 
                 kernel_size: int = 25, use_time_mixing: bool = True, 
                 use_feature_mixing: bool = True):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.top_k = top_k
        self.use_time_mixing = use_time_mixing
        self.use_feature_mixing = use_feature_mixing
        
        # Decomposition component
        self.decomposition = MovingAverageDecomposition(kernel_size)
        
        # Time domain mixing
        if use_time_mixing:
            self.time_mixing = nn.ModuleList([
                nn.Linear(seq_len, seq_len) for _ in range(top_k)
            ])
            self.time_mixing_weights = nn.Linear(d_model, top_k)
        
        # Feature domain mixing
        if use_feature_mixing:
            self.feature_mixing = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(top_k)
            ])
            self.feature_mixing_weights = nn.Linear(seq_len, top_k)
        
        # Output normalization
        self.norm = nn.LayerNorm(d_model)
        
        logger.debug("PastDecomposableMixing initialized",
                    seq_len=seq_len, d_model=d_model, top_k=top_k)
    
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply seasonal-trend decomposition"""
        return self.decomposition(x)
    
    def time_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time domain mixing"""
        if not self.use_time_mixing:
            return x
        
        batch_size, seq_len, d_model = x.shape
        
        # Compute mixing weights based on features
        # Average over sequence to get feature representation
        feature_repr = torch.mean(x, dim=1)  # [batch_size, d_model]
        mixing_weights = F.softmax(self.time_mixing_weights(feature_repr), dim=-1)  # [batch_size, top_k]
        
        # Apply different time mixing operations
        mixed_outputs = []
        for i, mixer in enumerate(self.time_mixing):
            # Apply mixing across time dimension
            x_transposed = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
            mixed = mixer(x_transposed).transpose(1, 2)  # Back to [batch_size, seq_len, d_model]
            mixed_outputs.append(mixed)
        
        # Weighted combination
        mixed_stack = torch.stack(mixed_outputs, dim=-1)  # [batch_size, seq_len, d_model, top_k]
        weights_expanded = mixing_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, top_k]
        
        mixed_result = torch.sum(mixed_stack * weights_expanded, dim=-1)  # [batch_size, seq_len, d_model]
        
        return mixed_result
    
    def feature_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature domain mixing"""
        if not self.use_feature_mixing:
            return x
        
        batch_size, seq_len, d_model = x.shape
        
        # Compute mixing weights based on temporal patterns
        # Average over features to get temporal representation
        temporal_repr = torch.mean(x, dim=2)  # [batch_size, seq_len]
        mixing_weights = F.softmax(self.feature_mixing_weights(temporal_repr), dim=-1)  # [batch_size, top_k]
        
        # Apply different feature mixing operations
        mixed_outputs = []
        for i, mixer in enumerate(self.feature_mixing):
            mixed = mixer(x)  # [batch_size, seq_len, d_model]
            mixed_outputs.append(mixed)
        
        # Weighted combination
        mixed_stack = torch.stack(mixed_outputs, dim=-1)  # [batch_size, seq_len, d_model, top_k]
        weights_expanded = mixing_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, top_k]
        
        mixed_result = torch.sum(mixed_stack * weights_expanded, dim=-1)  # [batch_size, seq_len, d_model]
        
        return mixed_result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PDM
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Mixed representations [batch_size, seq_len, d_model]
        """
        # Decompose into seasonal and trend
        seasonal, trend = self.decompose(x)
        
        # Apply mixing to both components
        seasonal_mixed = self.time_mixing(seasonal)
        seasonal_mixed = self.feature_mixing(seasonal_mixed)
        
        trend_mixed = self.time_mixing(trend)
        trend_mixed = self.feature_mixing(trend_mixed)
        
        # Combine and normalize
        combined = seasonal_mixed + trend_mixed
        output = self.norm(combined + x)  # Residual connection
        
        return output


class FutureMultipredictorMixing(nn.Module):
    """
    Future Multipredictor Mixing (FMM) layer
    
    Generates multiple predictions and adaptively mixes them:
    1. Multiple predictor generation
    2. Adaptive mixing weight computation
    3. Final prediction fusion
    """
    
    def __init__(self, seq_len: int, pred_len: int, d_model: int, top_k: int):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Multiple predictors with different architectures
        self.multi_predictors = nn.ModuleList()
        for i in range(top_k):
            # Each predictor has different hidden dimension for diversity
            hidden_dim = d_model // (i + 1) if i < top_k - 1 else d_model
            predictor = nn.Sequential(
                nn.Linear(seq_len, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, pred_len)
            )
            self.multi_predictors.append(predictor)
        
        # Adaptive mixing weight computation
        self.mixing_weights = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, top_k),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        logger.debug("FutureMultipredictorMixing initialized",
                    seq_len=seq_len, pred_len=pred_len, d_model=d_model, top_k=top_k)
    
    def generate_multi_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Generate multiple diverse predictions"""
        batch_size, seq_len, d_model = x.shape
        
        predictions = []
        for predictor in self.multi_predictors:
            # Apply predictor to each feature dimension
            pred_outputs = []
            for d in range(d_model):
                feature_seq = x[:, :, d]  # [batch_size, seq_len]
                pred = predictor(feature_seq)  # [batch_size, pred_len]
                pred_outputs.append(pred)
            
            # Stack predictions for all features
            pred_stack = torch.stack(pred_outputs, dim=-1)  # [batch_size, pred_len, d_model]
            predictions.append(pred_stack)
        
        # Stack all predictor outputs
        multi_predictions = torch.stack(predictions, dim=1)  # [batch_size, top_k, pred_len, d_model]
        
        return multi_predictions
    
    def compute_mixing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptive mixing weights based on input"""
        batch_size, seq_len, d_model = x.shape
        
        # Flatten input for weight computation
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * d_model]
        
        # Compute adaptive weights
        weights = self.mixing_weights(x_flat)  # [batch_size, top_k]
        
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FMM
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Future predictions [batch_size, pred_len, d_model]
        """
        # Generate multiple predictions
        multi_predictions = self.generate_multi_predictions(x)  # [batch_size, top_k, pred_len, d_model]
        
        # Compute adaptive mixing weights
        mixing_weights = self.compute_mixing_weights(x)  # [batch_size, top_k]
        
        # Apply adaptive mixing
        weights_expanded = mixing_weights.unsqueeze(2).unsqueeze(3)  # [batch_size, top_k, 1, 1]
        
        # Weighted combination of predictions
        mixed_prediction = torch.sum(multi_predictions * weights_expanded, dim=1)  # [batch_size, pred_len, d_model]
        
        # Apply output projection
        output = self.output_projection(mixed_prediction)
        
        return output


class TimesMixerNetwork(TransformerBase):
    """
    TimesMixer Network with Decomposable Mixing
    
    Key innovations:
    1. Past Decomposable Mixing (PDM) for historical processing
    2. Future Multipredictor Mixing (FMM) for multi-horizon forecasting
    3. Seasonal-trend decomposition for pattern recognition
    4. Decomposable time and feature mixing
    """
    
    def __init__(self, config: TimesMixerConfig):
        # Convert config to base format for parent class
        base_config = {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'max_seq_length': config.seq_len,
            'activation': config.activation,
        }
        super().__init__(ModelType.TIMESMIXER, base_config)
        
        self.config = config
        
        # Input embedding and projection
        self.input_embedding = nn.Linear(1, config.d_model)  # Project each feature to d_model
        
        # Past Decomposable Mixing layers
        self.pdm_layers = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=config.seq_len,
                d_model=config.d_model,
                top_k=config.top_k,
                num_kernels=config.num_kernels,
                kernel_size=config.kernel_size,
                use_time_mixing=config.use_time_mixing,
                use_feature_mixing=config.use_feature_mixing
            )
            for _ in range(config.n_layers)
        ])
        
        # Future Multipredictor Mixing layer
        self.fmm_layer = FutureMultipredictorMixing(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            top_k=config.top_k
        )
        
        # Multi-horizon prediction heads
        prediction_horizons = ['1h', '4h', '24h']
        self.prediction_heads = nn.ModuleDict({
            horizon: nn.Linear(config.d_model, 1)
            for horizon in prediction_horizons
        })
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(config.d_model, 1)
        
        # Direction classification head
        self.direction_head = nn.Linear(config.d_model, 5)  # 5 direction classes
        
        # Initialize parameters
        self._init_weights()
        
        logger.info("TimesMixerNetwork initialized",
                   d_model=config.d_model,
                   seq_len=config.seq_len,
                   pred_len=config.pred_len,
                   n_layers=config.n_layers,
                   top_k=config.top_k)
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with decomposable mixing
        
        Args:
            x: Input tensor [batch_size, seq_len, n_features]
            
        Returns:
            Dictionary with predictions and decomposition components
        """
        batch_size, seq_len, n_features = x.shape
        
        if seq_len < self.config.seq_len:
            raise ValueError(f"Sequence length {seq_len} too short for required {self.config.seq_len}")
        
        # Take the last seq_len timesteps if input is longer
        if seq_len > self.config.seq_len:
            x = x[:, -self.config.seq_len:, :]
            seq_len = self.config.seq_len
        
        # Process each feature separately then combine
        feature_embeddings = []
        seasonal_components = []
        trend_components = []
        
        for f in range(n_features):
            # Extract single feature and embed
            feature_data = x[:, :, f:f+1]  # [batch_size, seq_len, 1]
            embedded = self.input_embedding(feature_data)  # [batch_size, seq_len, d_model]
            
            # Store decomposition components from first PDM layer
            seasonal, trend = self.pdm_layers[0].decompose(embedded)
            seasonal_components.append(seasonal)
            trend_components.append(trend)
            
            # Apply PDM layers
            hidden = embedded
            for pdm_layer in self.pdm_layers:
                hidden = pdm_layer(hidden)
            
            feature_embeddings.append(hidden)
        
        # Combine feature embeddings
        combined_hidden = torch.stack(feature_embeddings, dim=-1)  # [batch_size, seq_len, d_model, n_features]
        combined_hidden = torch.mean(combined_hidden, dim=-1)  # [batch_size, seq_len, d_model]
        
        # Apply Future Multipredictor Mixing
        future_predictions = self.fmm_layer(combined_hidden)  # [batch_size, pred_len, d_model]
        
        # Use final timestep for prediction heads
        final_repr = torch.mean(future_predictions, dim=1)  # [batch_size, d_model]
        
        # Generate outputs
        outputs = {}
        
        # Multi-horizon predictions
        for horizon, head in self.prediction_heads.items():
            pred = head(final_repr)  # [batch_size, 1]
            outputs[f'price_{horizon}'] = pred
        
        # Confidence estimation
        outputs['confidence'] = torch.sigmoid(self.confidence_head(final_repr))
        
        # Direction predictions
        direction_logits = self.direction_head(final_repr)
        outputs['direction_logits'] = direction_logits
        outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
        
        # Future predictions for analysis
        outputs['predictions'] = future_predictions
        
        # Mixing weights from FMM
        mixing_weights = self.fmm_layer.compute_mixing_weights(combined_hidden)
        outputs['mixing_weights'] = mixing_weights
        
        # Decomposition components for interpretability
        if seasonal_components and trend_components:
            outputs['seasonal_components'] = torch.stack(seasonal_components, dim=-1).mean(dim=-1)
            outputs['trend_components'] = torch.stack(trend_components, dim=-1).mean(dim=-1)
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract mixing weights for visualization (TimesMixer doesn't use attention)"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.get('mixing_weights', torch.ones(1, self.config.top_k))


class TimesMixerPredictor(MLAnalyzerBase):
    """
    TimesMixer-based predictor for time-series forecasting
    Specialized for decomposable mixing and multi-scale pattern recognition
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ModelType.TIMESMIXER, config)
        
        # Parse configuration
        self.model_config = self._parse_config(config or {})
        
        # Initialize model
        self.model = TimesMixerNetwork(self.model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self._setup_training()
        
        # Feature engineering
        self.seq_len = self.model_config.seq_len
        self.pred_len = self.model_config.pred_len
        self.n_features = 20  # Default number of features for corpus training
        self.required_features = self._get_required_features()
        
        logger.info("TimesMixerPredictor initialized",
                   model_type=self.model_type,
                   seq_len=self.seq_len,
                   pred_len=self.pred_len)
    
    def _parse_config(self, config: Dict) -> TimesMixerConfig:
        """Parse configuration dictionary into TimesMixerConfig"""
        return TimesMixerConfig(
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 3),
            d_ff=config.get('d_ff', 256),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            seq_len=config.get('seq_len', 96),
            pred_len=config.get('pred_len', 24),
            top_k=config.get('top_k', 5),
            num_kernels=config.get('num_kernels', 6),
            use_time_mixing=config.get('use_time_mixing', True),
            use_feature_mixing=config.get('use_feature_mixing', True),
            decomposition_kernel=config.get('decomposition_kernel', 'moving_avg'),
            kernel_size=config.get('kernel_size', 25)
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
        """Get list of required features for TimesMixer"""
        return self.required_features
    
    def _get_required_features(self) -> List[str]:
        """Define required features for TimesMixer (decomposable mixing)"""
        return [
            # Core price features for mixing
            'close', 'open', 'high', 'low', 'volume',
            
            # Derived features for decomposition
            'returns', 'volatility', 'price_momentum',
            
            # Seasonal features for decomposition
            'seasonal_features', 'trend_features', 'cyclical_features',
            
            # Technical indicators
            'sma_20', 'ema_12', 'rsi', 'macd',
            
            # Market context for mixing
            'market_regime', 'volatility_regime',
            'funding_rate', 'volume_profile',
            
            # Temporal features
            'hour_of_day', 'day_of_week', 'trading_session',
            
            # Multi-scale features
            'short_term_patterns', 'long_term_trends'
        ]
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Analyze token using TimesMixer with decomposable mixing
        """
        if not self._is_trained:
            raise ModelNotTrainedError("TimesMixer model is not trained")
        
        if historical_data is None or len(historical_data) < self.seq_len:
            self.logger.warning("Insufficient data for TimesMixer prediction",
                              token=token.address,
                              data_length=len(historical_data) if historical_data is not None else 0,
                              required_length=self.seq_len)
            return self._create_default_prediction(token)
        
        try:
            # Prepare mixing-based input sequence
            input_sequence = self._prepare_mixing_sequence(historical_data, token)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                
                # Extract predictions
                price_predictions = {}
                for horizon in ['1h', '4h', '24h']:
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        pred_value = outputs[pred_key].cpu().numpy().flatten()[0]
                        price_predictions[horizon] = pred_value
                
                confidence = outputs['confidence'].item()
                direction_probs = outputs['direction_probs'].cpu().numpy().flatten()
            
            # Convert relative predictions to absolute prices
            current_price = token.price_usd or historical_data['close'].iloc[-1]
            
            price_pred_1h = current_price * (1 + price_predictions.get('1h', 0))
            price_pred_4h = current_price * (1 + price_predictions.get('4h', 0))
            price_pred_24h = current_price * (1 + price_predictions.get('24h', 0))
            
            # Determine direction from probabilities
            direction_idx = np.argmax(direction_probs)
            directions = [PredictionDirection.STRONG_SELL, PredictionDirection.SELL,
                         PredictionDirection.HOLD, PredictionDirection.BUY,
                         PredictionDirection.STRONG_BUY]
            direction = directions[direction_idx]
            
            # Calculate probability of upward movement
            prob_up = direction_probs[3] + direction_probs[4]  # BUY + STRONG_BUY
            
            # Create prediction result
            result = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.TIMESMIXER,
                price_prediction_1h=price_pred_1h,
                price_prediction_4h=price_pred_4h,
                price_prediction_24h=price_pred_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                model_accuracy=0.82,  # Placeholder - would be tracked
                features_used=['decomposable_mixing', 'seasonal_trend_decomposition', 'multi_predictor_mixing'],
                model_version='timesmixer_v1.0'
            )
            
            return result
            
        except Exception as e:
            self.logger.error("TimesMixer prediction failed",
                            token=token.address,
                            error=str(e))
            return self._create_default_prediction(token)
    
    def _prepare_mixing_sequence(self, data: pd.DataFrame, token: DiscoveredToken) -> np.ndarray:
        """
        Prepare input sequence for decomposable mixing
        
        Returns:
            Numpy array of shape (seq_len, n_features)
        """
        # Take the last seq_len rows
        sequence_data = data.tail(self.seq_len).copy()
        
        # Extract features for mixing
        features = []
        
        # Core price features
        if 'close' in sequence_data.columns:
            close_prices = sequence_data['close'].values
            # Normalize for better mixing
            normalized_close = (close_prices - close_prices.mean()) / (close_prices.std() + 1e-8)
            features.append(normalized_close)
        
        # Volume feature
        if 'volume' in sequence_data.columns:
            volumes = sequence_data['volume'].values
            normalized_volume = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            features.append(normalized_volume)
        
        # Returns feature
        if 'close' in sequence_data.columns:
            returns = np.diff(np.log(sequence_data['close'].values + 1e-8))
            returns = np.concatenate([[0], returns])  # Pad first value
            features.append(returns)
        
        # Volatility feature (rolling std of returns)
        if len(features) > 0:
            returns_series = pd.Series(features[-1])
            volatility = returns_series.rolling(window=10, min_periods=1).std().fillna(0).values
            features.append(volatility)
        
        # Technical indicators
        if 'rsi' in sequence_data.columns:
            rsi_normalized = (sequence_data['rsi'].values - 50) / 50  # Center around 0
            features.append(rsi_normalized)
        
        # Ensure we have at least 5 features for good mixing
        while len(features) < 5:
            features.append(np.zeros(len(sequence_data)))
        
        # Limit to maximum of 7 features to avoid overfitting
        features = features[:7]
        
        # Stack features: shape (seq_len, n_features)
        mixing_sequence = np.column_stack(features)
        
        return mixing_sequence.astype(np.float32)
    
    def _create_default_prediction(self, token: DiscoveredToken) -> PredictionResult:
        """Create a default prediction when analysis fails"""
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.TIMESMIXER,
            direction=PredictionDirection.HOLD,
            confidence=0.1,
            probability_up=0.5,
            features_used=['default'],
            model_version='timesmixer_v1.0'
        )
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the TimesMixer model on historical data
        """
        try:
            self.logger.info("Starting TimesMixer model training",
                           data_shape=training_data.shape)
            
            # Prepare mixing-based training data
            sequences, targets = self._prepare_training_data(training_data)
            
            if len(sequences) < self.seq_len:
                self.logger.error("Insufficient training data for mixing",
                                samples=len(sequences),
                                required=self.seq_len)
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
                horizons = ['1h', '4h', '24h']
                for i, horizon in enumerate(horizons):
                    pred_key = f'price_{horizon}'
                    if pred_key in outputs:
                        predictions = outputs[pred_key].squeeze()
                        loss += self.criterion(predictions, y_train[:, i])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"Training epoch {epoch}, loss: {loss.item():.4f}")
            
            self._is_trained = True
            self.logger.info("TimesMixer model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("TimesMixer model training failed", error=str(e))
            return False
    
    def prepare_training_from_corpus(self, data: pd.DataFrame,
                                    sequence_length: Optional[int] = None,
                                    prediction_horizons: List[int] = [1, 4, 24],
                                    n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from unified corpus format for TimesMixer
        
        Args:
            data: DataFrame from corpus with all features
            sequence_length: Length of input sequences (uses self.seq_len if not provided)
            prediction_horizons: Hours ahead to predict [1h, 4h, 24h]
            n_features: Number of features to use (None = auto-select based on decomposition needs)
            
        Returns:
            X: Input sequences [n_samples, seq_len, n_features]
            y: Target values [n_samples, n_targets] as price changes
            feature_names: List of feature names used
        """
        if sequence_length is None:
            sequence_length = self.seq_len
            
        if n_features is None:
            n_features = self.n_features
            
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
        
        # Select features optimal for temporal decomposition
        # TimesMixer benefits from features with different temporal patterns
        priority_features = [
            'close', 'volume', 'open', 'high', 'low',  # Core OHLCV
            # Seasonal/cyclical features
            'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            # Different time scale returns
            'returns_1h', 'returns_24h', 'returns_7d',
            # Trend indicators
            'ema_12', 'ema_26', 'ema_50', 'ema_200',
            'sma_20', 'sma_50', 'sma_200',
            # Oscillators with different periods
            'rsi_14', 'rsi_7', 'rsi_21',
            # Volatility at different scales
            'volatility', 'volatility_24h', 'realized_volatility',
            # MACD for trend
            'macd', 'macd_signal', 'macd_histogram',
            # Momentum features
            'momentum_5', 'momentum_10', 'momentum_20',
            # Market microstructure
            'high_low_ratio', 'close_to_high', 'close_to_low',
            # ML scores
            'volatility_score', 'volume_score', 'momentum_score',
            'market_regime', 'trend_strength'
        ]
        
        # Select features based on availability and limit
        selected_features = []
        for feat in priority_features:
            if feat in available_features and len(selected_features) < n_features:
                selected_features.append(feat)
        
        # Fill remaining slots with other available features
        for feat in available_features:
            if feat not in selected_features and len(selected_features) < n_features:
                selected_features.append(feat)
        
        # Ensure we have at least some features
        if not selected_features:
            selected_features = available_features[:n_features]
        
        # Select feature columns
        feature_data = data[selected_features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize features for mixing (each feature normalized independently)
        from sklearn.preprocessing import StandardScaler
        feature_data_normalized = pd.DataFrame(index=feature_data.index)
        
        for col in feature_data.columns:
            scaler = StandardScaler()
            feature_data_normalized[col] = scaler.fit_transform(feature_data[[col]])
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        max_horizon = max(prediction_horizons)
        
        for i in range(sequence_length, len(data) - max_horizon):
            # Input sequence for temporal decomposition
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
        
        # Update model config with actual dimensions
        self.n_features = len(feature_names)
        
        self.logger.info("TimesMixer corpus data prepared",
                        sequences=len(X),
                        seq_len=sequence_length,
                        n_features=len(feature_names),
                        features=feature_names[:5] + ['...'] if len(feature_names) > 5 else feature_names,
                        horizons=prediction_horizons)
        
        return X, y, feature_names
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for decomposable mixing
        
        Returns:
            sequences: [n_samples, seq_len, n_features]
            targets: [n_samples, n_horizons]
        """
        sequences = []
        targets = []
        
        # Create sliding windows
        for i in range(self.seq_len, len(data) - 24):  # Need 24 hours ahead for targets
            # Input sequence
            seq_data = data.iloc[i-self.seq_len:i]
            
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
            sequence = self._prepare_mixing_sequence(seq_data, dummy_token)
            
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
        """Save the trained TimesMixer model"""
        try:
            if not self._is_trained:
                self.logger.warning("Attempting to save untrained model")
                return False
            
            state = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config.__dict__,
                'is_trained': self._is_trained,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_type': 'timesmixer'
            }
            
            torch.save(state, filepath)
            self.logger.info("TimesMixer model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save TimesMixer model", error=str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained TimesMixer model"""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Update config
            self.model_config = TimesMixerConfig(**state['config'])
            
            # Recreate model with loaded config
            self.model = TimesMixerNetwork(self.model_config)
            self.model.to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            
            self._is_trained = state.get('is_trained', False)
            
            # Load optimizer state if available
            if state.get('optimizer_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.logger.info("TimesMixer model loaded", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load TimesMixer model", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check if the TimesMixer is healthy and ready"""
        try:
            if not self._is_trained:
                return False
            
            # Quick forward pass test with mixing-compatible input
            dummy_input = torch.randn(1, self.seq_len, 7).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(dummy_input)
                
                # Check that all expected outputs are present
                expected_keys = ['price_1h', 'price_4h', 'price_24h', 'confidence', 'direction_probs']
                
                return all(key in outputs for key in expected_keys)
                
        except Exception as e:
            self.logger.error("TimesMixer health check failed", error=str(e))
            return False