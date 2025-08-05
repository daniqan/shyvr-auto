"""
TimesFM Wrapper Architecture for Zero-Shot Time Series Forecasting
Integrates Google's TimesFM foundation model with existing RLTE infrastructure
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import structlog
from transformers import AutoModel, AutoConfig, AutoTokenizer

try:
    from .base import TransformerBase, TransformerConfig
    from ..base import MLAnalyzerBase, ModelType, PredictionResult, PredictionDirection
    from src.discovery.base import DiscoveredToken
except ImportError:
    # For standalone testing - define minimal stubs
    from abc import ABC
    from typing import Any
    class TransformerBase(ABC):
        pass
    class TransformerConfig:
        pass
    class MLAnalyzerBase:
        pass
    class ModelType:
        pass
    class PredictionResult:
        pass
    class PredictionDirection:
        pass
    class DiscoveredToken:
        pass


logger = structlog.get_logger()


@dataclass
class TimesFMConfig:
    """Configuration for TimesFM wrapper"""
    
    # Model configuration
    model_name: str = "google/timesfm-1.0-200m"  # TimesFM model variant
    prediction_length: int = 24                  # Forecast horizon
    context_length: int = 512                    # Input context length
    
    # Zero-shot and fine-tuning
    use_zero_shot: bool = True                   # Enable zero-shot predictions
    fine_tune_enabled: bool = False              # Enable fine-tuning capability
    
    # Performance optimizations
    gcp_optimized: bool = True                   # GCP-specific optimizations
    batch_size: int = 32                         # Batch size for inference
    use_tpu: bool = False                        # TPU support (if available)
    memory_efficient: bool = True                # Memory-efficient inference
    
    # Multivariate and streaming
    multivariate_enabled: bool = True            # Support multivariate time series
    streaming_enabled: bool = False              # Real-time streaming predictions
    
    # Error handling
    enable_fallback: bool = True                 # Enable fallback mechanisms
    fallback_model: str = "simple_linear"       # Fallback model type
    
    # Tokenization
    max_sequence_length: int = 2048             # Maximum sequence length
    chunk_size: int = 512                       # Chunk size for long sequences
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.max_sequence_length < self.context_length:
            raise ValueError("max_sequence_length must be >= context_length")
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Base memory for model parameters (estimated for 200M parameters)
        model_memory_mb = 800  # ~200M parameters * 4 bytes
        
        # Context memory
        context_memory_mb = (self.context_length * self.batch_size * 768 * 4) / (1024 * 1024)
        
        # Prediction memory
        pred_memory_mb = (self.prediction_length * self.batch_size * 768 * 4) / (1024 * 1024)
        
        total_memory = model_memory_mb + context_memory_mb + pred_memory_mb
        
        # Apply optimizations
        if self.memory_efficient:
            total_memory *= 0.7  # 30% reduction with memory optimization
        
        return total_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimesFMConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class CryptoTimesFMTokenizer:
    """Crypto-specific tokenizer for TimesFM"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
            '<MASK>': 4
        }
        
        # Price ranges for tokenization
        self.price_ranges = self._create_price_ranges()
        self.volume_ranges = self._create_volume_ranges()
        self.regime_tokens = {
            'bull': 100, 'bear': 101, 'sideways': 102, 
            'volatile': 103, 'stable': 104
        }
    
    def _create_price_ranges(self) -> Dict[str, int]:
        """Create price range tokens"""
        ranges = {}
        base_token = 10
        
        # Price percentage change ranges
        for i, threshold in enumerate([-50, -20, -10, -5, -2, -1, -0.5, 0, 
                                      0.5, 1, 2, 5, 10, 20, 50]):
            ranges[f'price_change_{threshold}'] = base_token + i
        
        return ranges
    
    def _create_volume_ranges(self) -> Dict[str, int]:
        """Create volume range tokens"""
        ranges = {}
        base_token = 50
        
        # Volume percentile ranges
        for i, percentile in enumerate([0, 10, 25, 50, 75, 90, 95, 99]):
            ranges[f'volume_p{percentile}'] = base_token + i
        
        return ranges
    
    def tokenize_prices(self, prices: List[float]) -> torch.Tensor:
        """Tokenize price sequence"""
        if not prices:
            return torch.tensor([self.special_tokens['<UNK>']])
        
        tokens = []
        for i, price in enumerate(prices):
            if i == 0:
                tokens.append(self.special_tokens['<START>'])
            else:
                # Calculate percentage change
                pct_change = ((price - prices[i-1]) / prices[i-1]) * 100
                
                # Find appropriate price range token
                token = self.special_tokens['<UNK>']
                for range_name, range_token in self.price_ranges.items():
                    threshold = float(range_name.split('_')[-1])
                    if pct_change >= threshold:
                        token = range_token
                
                tokens.append(token)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokenize_volumes(self, volumes: List[float]) -> torch.Tensor:
        """Tokenize volume sequence"""
        if not volumes:
            return torch.tensor([self.special_tokens['<UNK>']])
        
        # Calculate volume percentiles
        volume_percentiles = np.percentile(volumes, [0, 10, 25, 50, 75, 90, 95, 99])
        
        tokens = []
        for volume in volumes:
            # Find volume percentile bucket
            percentile_idx = np.searchsorted(volume_percentiles, volume)
            percentile_idx = min(percentile_idx, len(volume_percentiles) - 1)
            
            token = self.volume_ranges[f'volume_p{[0, 10, 25, 50, 75, 90, 95, 99][percentile_idx]}']
            tokens.append(token)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokenize_market_regimes(self, regimes: List[str]) -> torch.Tensor:
        """Tokenize market regime sequence"""
        tokens = []
        for regime in regimes:
            token = self.regime_tokens.get(regime.lower(), self.special_tokens['<UNK>'])
            tokens.append(token)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokenize_market_data(self, market_data: Dict[str, List[float]]) -> torch.Tensor:
        """Tokenize complete market data"""
        all_tokens = []
        
        if 'price' in market_data:
            price_tokens = self.tokenize_prices(market_data['price'])
            all_tokens.append(price_tokens.unsqueeze(-1))
        
        if 'volume' in market_data:
            volume_tokens = self.tokenize_volumes(market_data['volume'])
            all_tokens.append(volume_tokens.unsqueeze(-1))
        
        if len(all_tokens) == 0:
            return torch.tensor([[self.special_tokens['<UNK>']]])
        
        # Combine all token sequences
        combined_tokens = torch.cat(all_tokens, dim=-1)
        return combined_tokens


class GCPTimesFMOptimizer:
    """GCP-specific optimizations for TimesFM deployment"""
    
    def __init__(self, batch_size: int = 32, use_tpu: bool = False, 
                 memory_efficient: bool = True):
        self.batch_size = batch_size
        self.use_tpu = use_tpu
        self.memory_efficient = memory_efficient
        
        # Performance tracking
        self._inference_times = []
        self._memory_usage = []
    
    def create_batches(self, data_list: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Create optimized batches for inference"""
        batches = []
        
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def optimize_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations"""
        if self.memory_efficient:
            # Enable gradient checkpointing if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Set to eval mode to disable dropout and batch norm updates
            model.eval()
        
        return model
    
    def estimate_memory_usage(self, sequence_length: int, batch_size: int) -> float:
        """Estimate memory usage for given parameters"""
        # Simplified memory estimation
        base_memory = 800  # MB for model
        sequence_memory = (sequence_length * batch_size * 768 * 4) / (1024 * 1024)
        
        total_memory = base_memory + sequence_memory
        
        if self.memory_efficient:
            total_memory *= 0.7
        
        return total_memory
    
    def clean_cache(self):
        """Clean GPU/TPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TimesFMWrapper(TransformerBase):
    """
    TimesFM wrapper for zero-shot time series forecasting
    Integrates with existing RLTE transformer infrastructure
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize TransformerBase
        transformer_config_dict = {
            'd_model': 768,  # TimesFM model dimension
            'n_heads': 12,   # TimesFM attention heads
            'n_layers': 12,  # TimesFM layers
            'max_seq_length': config.get('context_length', 512),
            'dropout': 0.0   # No dropout for inference
        }
        super().__init__(ModelType.TIMESFM, transformer_config_dict)
        
        # TimesFM specific configuration
        self.timesfm_config = TimesFMConfig.from_dict(config)
        
        # Initialize components
        self.tokenizer = CryptoTimesFMTokenizer()
        self.gcp_optimizer = GCPTimesFMOptimizer(
            batch_size=self.timesfm_config.batch_size,
            use_tpu=self.timesfm_config.use_tpu,
            memory_efficient=self.timesfm_config.memory_efficient
        )
        
        # Model components
        self._timesfm_model = None
        self._model_config = None
        self._is_fine_tuned = False
        
        # Streaming state
        self._streaming_context = None
        self._streaming_initialized = False
        
        # Feature engineering integration
        self._feature_engineer = None
        
        # Performance tracking
        self._inference_times = []
        self._prediction_cache = {}
        
        # Initialize logger first
        try:
            self.logger = structlog.get_logger().bind(
                model_type=ModelType.TIMESFM.value,
                model_name=self.timesfm_config.model_name
            )
        except:
            # Fallback logger for testing
            self.logger = logger
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TimesFM model"""
        try:
            # Note: This is a placeholder for actual TimesFM model loading
            # In practice, you would use the official TimesFM library
            self.logger.info("Initializing TimesFM model (placeholder implementation)")
            
            # For now, create a simple transformer as placeholder
            self._model_config = {
                'vocab_size': self.tokenizer.vocab_size,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12,
                'intermediate_size': 3072,
                'max_position_embeddings': self.timesfm_config.max_sequence_length
            }
            
            # Create placeholder model (in real implementation, use TimesFM)
            self._timesfm_model = self._create_placeholder_model()
            
            # Apply GCP optimizations
            self._timesfm_model = self.gcp_optimizer.optimize_memory(self._timesfm_model)
            
            self._is_trained = True  # TimesFM is pre-trained
            
        except Exception as e:
            self.logger.error("Failed to initialize TimesFM model", error=str(e))
            if self.timesfm_config.enable_fallback:
                self._initialize_fallback_model()
            else:
                raise
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create placeholder model (replace with actual TimesFM)"""
        class PlaceholderTimesFM(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config['hidden_size'],
                        nhead=config['num_attention_heads'],
                        dim_feedforward=config['intermediate_size'],
                        batch_first=True
                    ),
                    num_layers=config['num_hidden_layers']
                )
                self.output_projection = nn.Linear(config['hidden_size'], 1)
            
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embeddings(input_ids)
                transformer_output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
                predictions = self.output_projection(transformer_output[:, -1, :])  # Use last token
                return predictions
        
        return PlaceholderTimesFM(self._model_config)
    
    def _initialize_fallback_model(self):
        """Initialize fallback model when TimesFM fails"""
        self.logger.warning("Initializing fallback model")
        
        class SimpleFallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
            
            def forward(self, x):
                return self.linear(x.float().mean(dim=-1, keepdim=True))
        
        self._timesfm_model = SimpleFallbackModel()
        self._is_trained = True
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through TimesFM model"""
        if self._timesfm_model is None:
            raise RuntimeError("TimesFM model not initialized")
        
        # Convert input to appropriate format for TimesFM
        if x.dtype != torch.long:
            # For placeholder, convert to token format
            x_tokens = (x * 1000).long().clamp(0, self.tokenizer.vocab_size - 1)
        else:
            x_tokens = x
        
        return self._timesfm_model(x_tokens, attention_mask)
    
    def get_attention_weights(self) -> torch.Tensor:
        """Extract attention weights (placeholder implementation)"""
        # In real TimesFM implementation, this would extract actual attention weights
        batch_size = 1
        seq_len = self.timesfm_config.context_length
        n_heads = 12
        
        return torch.ones(batch_size, n_heads, seq_len, seq_len)
    
    def zero_shot_predict(self, time_series: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None,
                         horizon: int = 24) -> np.ndarray:
        """Make zero-shot predictions without training"""
        start_time = time.time()
        
        try:
            # Validate inputs
            if time_series.size == 0:
                raise ValueError("Empty time series provided")
            
            # Handle NaN/Inf values
            if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
                if self.timesfm_config.enable_fallback:
                    return self._fallback_prediction(time_series, horizon)
                else:
                    raise ValueError("Invalid values in time series")
            
            # Prepare input tensor
            input_tensor = torch.tensor(time_series, dtype=torch.float32)
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure proper sequence length
            if input_tensor.shape[1] > self.timesfm_config.context_length:
                input_tensor = input_tensor[:, -self.timesfm_config.context_length:, :]
            
            # Make predictions
            self._timesfm_model.eval()
            with torch.no_grad():
                predictions = []
                current_input = input_tensor
                
                for _ in range(horizon):
                    pred = self.forward(current_input)
                    predictions.append(pred.item())
                    
                    # Update input for next prediction (autoregressive)
                    new_input = torch.cat([
                        current_input[:, 1:, :],  # Remove first timestep
                        pred.unsqueeze(1).unsqueeze(-1).expand(-1, -1, current_input.shape[-1])  # Add prediction
                    ], dim=1)
                    current_input = new_input
            
            predictions = np.array(predictions)
            
            # Track performance
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)
            
            self.logger.info("Zero-shot prediction completed",
                           horizon=horizon,
                           inference_time_ms=inference_time * 1000)
            
            return predictions
            
        except Exception as e:
            self.logger.error("Zero-shot prediction failed", error=str(e))
            if self.timesfm_config.enable_fallback:
                return self._fallback_prediction(time_series, horizon)
            else:
                raise
    
    def _fallback_prediction(self, time_series: np.ndarray, horizon: int) -> np.ndarray:
        """Fallback prediction when main prediction fails"""
        self.logger.warning("Using fallback prediction")
        
        # Simple linear extrapolation
        if len(time_series) >= 2:
            last_values = time_series[-2:]
            trend = last_values[-1] - last_values[-2] if len(last_values) == 2 else 0
            last_value = time_series[-1] if len(time_series) > 0 else 0
            
            predictions = []
            for i in range(horizon):
                pred = last_value + (trend * (i + 1))
                predictions.append(pred)
            
            return np.array(predictions)
        else:
            # Return zeros if insufficient data
            return np.zeros(horizon)
    
    def predict_with_attention(self, time_series: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """Make predictions and return attention weights"""
        predictions = self.zero_shot_predict(time_series, timestamps, 
                                           self.timesfm_config.prediction_length)
        attention_weights = self.get_attention_weights()
        
        return predictions, attention_weights
    
    def batch_inference(self, batch_data: List[np.ndarray]) -> List[np.ndarray]:
        """Perform batch inference for multiple time series"""
        batches = self.gcp_optimizer.create_batches(batch_data)
        all_predictions = []
        
        for batch in batches:
            batch_predictions = []
            for time_series in batch:
                predictions = self.zero_shot_predict(time_series, None, 
                                                   self.timesfm_config.prediction_length)
                batch_predictions.append(predictions)
            all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def predict_multivariate(self, multivariate_data: Dict[str, np.ndarray], 
                           horizon: int = 24) -> Dict[str, np.ndarray]:
        """Predict multiple time series simultaneously"""
        predictions = {}
        
        for asset_name, time_series in multivariate_data.items():
            predictions[asset_name] = self.zero_shot_predict(time_series, None, horizon)
        
        return predictions
    
    def fine_tune(self, training_data: pd.DataFrame, epochs: int = 5, 
                 learning_rate: float = 1e-5) -> bool:
        """Fine-tune TimesFM on domain-specific data"""
        if not self.timesfm_config.fine_tune_enabled:
            self.logger.warning("Fine-tuning not enabled")
            return False
        
        try:
            self.logger.info("Starting TimesFM fine-tuning", 
                           epochs=epochs, learning_rate=learning_rate)
            
            # Prepare training data
            # Note: This is a simplified implementation
            # Real fine-tuning would require more sophisticated data preparation
            
            optimizer = torch.optim.AdamW(self._timesfm_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                total_loss = 0
                
                # Simple training loop (would need proper batching in practice)
                for i in range(len(training_data) - self.timesfm_config.context_length):
                    sequence = training_data.iloc[i:i+self.timesfm_config.context_length]
                    target = training_data.iloc[i+self.timesfm_config.context_length]['close']
                    
                    # Convert to tensor
                    input_tensor = torch.tensor(sequence['close'].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    target_tensor = torch.tensor([target], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    output = self.forward(input_tensor)
                    loss = criterion(output.squeeze(), target_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / (len(training_data) - self.timesfm_config.context_length)
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self._is_fine_tuned = True
            return True
            
        except Exception as e:
            self.logger.error("Fine-tuning failed", error=str(e))
            return False
    
    def save_fine_tuned_model(self, filepath: str) -> bool:
        """Save fine-tuned model"""
        if not self._is_fine_tuned:
            self.logger.warning("Model is not fine-tuned")
            return False
        
        try:
            state = {
                'model_state_dict': self._timesfm_model.state_dict(),
                'config': self.timesfm_config.to_dict(),
                'is_fine_tuned': self._is_fine_tuned,
                'model_config': self._model_config
            }
            
            torch.save(state, filepath)
            self.logger.info("Fine-tuned model saved", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save fine-tuned model", error=str(e))
            return False
    
    def load_fine_tuned_model(self, filepath: str) -> bool:
        """Load fine-tuned model"""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            # Update config
            self.timesfm_config = TimesFMConfig.from_dict(state['config'])
            self._model_config = state['model_config']
            
            # Recreate model and load state
            self._timesfm_model = self._create_placeholder_model()
            self._timesfm_model.load_state_dict(state['model_state_dict'])
            
            self._is_fine_tuned = state.get('is_fine_tuned', False)
            
            self.logger.info("Fine-tuned model loaded", filepath=filepath)
            return True
            
        except Exception as e:
            self.logger.error("Failed to load fine-tuned model", error=str(e))
            return False
    
    def init_streaming(self, context_data: np.ndarray):
        """Initialize streaming prediction context"""
        if not self.timesfm_config.streaming_enabled:
            raise ValueError("Streaming not enabled")
        
        self._streaming_context = torch.tensor(context_data, dtype=torch.float32)
        if len(self._streaming_context.shape) == 2:
            self._streaming_context = self._streaming_context.unsqueeze(0)
        
        # Keep only the last context_length points
        if self._streaming_context.shape[1] > self.timesfm_config.context_length:
            self._streaming_context = self._streaming_context[:, -self.timesfm_config.context_length:, :]
        
        self._streaming_initialized = True
        self.logger.info("Streaming context initialized")
    
    def update_stream(self, new_point: np.ndarray) -> np.ndarray:
        """Update streaming context and get prediction"""
        if not self._streaming_initialized:
            raise ValueError("Streaming not initialized")
        
        # Add new point to context
        new_tensor = torch.tensor(new_point, dtype=torch.float32)
        if len(new_tensor.shape) == 1:
            new_tensor = new_tensor.unsqueeze(0).unsqueeze(0)
        elif len(new_tensor.shape) == 2:
            new_tensor = new_tensor.unsqueeze(0)
        
        # Update context (sliding window)
        self._streaming_context = torch.cat([
            self._streaming_context[:, 1:, :],  # Remove oldest
            new_tensor  # Add newest
        ], dim=1)
        
        # Make prediction
        prediction = self.zero_shot_predict(self._streaming_context.squeeze(0).numpy(), None, 1)
        
        return prediction
    
    def get_streaming_prediction(self) -> np.ndarray:
        """Get current streaming prediction"""
        if not self._streaming_initialized:
            raise ValueError("Streaming not initialized")
        
        return self.zero_shot_predict(self._streaming_context.squeeze(0).numpy(), None, 
                                    self.timesfm_config.prediction_length)
    
    def set_feature_engineer(self, feature_engineer):
        """Set feature engineer for integration"""
        self._feature_engineer = feature_engineer
        self.logger.info("Feature engineer set for TimesFM integration")
    
    def prepare_features(self, time_series: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare features using existing feature engineering"""
        features = {'raw_series': time_series}
        
        if self._feature_engineer is not None:
            # Use existing feature engineering
            engineered_features = self._feature_engineer.extract_transformer_features(time_series)
            features.update(engineered_features)
        
        return features
    
    # Required methods for ensemble compatibility
    def get_ensemble_predictions(self, time_series: np.ndarray) -> Dict[str, float]:
        """Get predictions formatted for ensemble system"""
        predictions = self.zero_shot_predict(time_series, None, 24)
        
        return {
            'price_1h': predictions[0] if len(predictions) > 0 else 0.0,
            'price_4h': predictions[3] if len(predictions) > 3 else 0.0,
            'price_24h': predictions[-1] if len(predictions) > 0 else 0.0
        }
    
    def get_prediction_confidence(self, time_series: np.ndarray) -> float:
        """Get prediction confidence score"""
        # Simple confidence based on prediction variance
        predictions = self.zero_shot_predict(time_series, None, 10)
        variance = np.var(predictions)
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + variance)))
        
        return confidence
    
    def get_model_weight(self, market_conditions: Dict[str, Any]) -> float:
        """Get model weight for ensemble based on market conditions"""
        # TimesFM works well in all conditions due to pre-training
        base_weight = 0.3
        
        # Increase weight for stable conditions (TimesFM strength)
        volatility = market_conditions.get('volatility', 0.5)
        if volatility < 0.3:
            base_weight += 0.2
        
        return min(1.0, base_weight)
    
    # TransformerBase required methods
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Analyze token using TimesFM"""
        if historical_data is None or len(historical_data) < 10:
            # Use default prediction for insufficient data
            return self._create_default_prediction(token)
        
        try:
            # Prepare time series data
            time_series = historical_data[['close', 'volume', 'high', 'low']].values
            
            # Make predictions
            predictions = self.zero_shot_predict(time_series, None, 24)
            confidence = self.get_prediction_confidence(time_series)
            
            # Create prediction result
            current_price = token.price_usd or historical_data['close'].iloc[-1]
            price_24h = current_price + predictions[-1]
            
            # Determine direction
            price_change = (price_24h - current_price) / current_price
            if price_change > 0.05:
                direction = PredictionDirection.BUY
                prob_up = 0.7
            elif price_change < -0.05:
                direction = PredictionDirection.SELL
                prob_up = 0.3
            else:
                direction = PredictionDirection.HOLD
                prob_up = 0.5
            
            return PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.TIMESFM,
                price_prediction_24h=price_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                features_used=['timesfm_zero_shot'],
                model_version='timesfm_v1.0'
            )
            
        except Exception as e:
            self.logger.error("Token analysis failed", error=str(e))
            return self._create_default_prediction(token)
    
    def _create_default_prediction(self, token: DiscoveredToken) -> PredictionResult:
        """Create default prediction when analysis fails"""
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.TIMESFM,
            direction=PredictionDirection.HOLD,
            confidence=0.1,
            probability_up=0.5,
            features_used=['default'],
            model_version='timesfm_v1.0'
        )
    
    # Health and monitoring methods
    async def health_check(self) -> bool:
        """Check TimesFM model health"""
        try:
            if self._timesfm_model is None:
                return False
            
            # Test with dummy input
            dummy_input = torch.randn(1, 10, 1)
            with torch.no_grad():
                output = self.forward(dummy_input)
            
            return not torch.isnan(output).any()
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': ModelType.TIMESFM.value,
            'timesfm_version': self.timesfm_config.model_name,
            'parameter_count': sum(p.numel() for p in self._timesfm_model.parameters()) if self._timesfm_model else 0,
            'supports_zero_shot': self.timesfm_config.use_zero_shot,
            'is_fine_tuned': self._is_fine_tuned,
            'context_length': self.timesfm_config.context_length,
            'prediction_length': self.timesfm_config.prediction_length,
            'memory_usage_mb': self.get_memory_usage(),
            'avg_inference_time_ms': np.mean(self._inference_times) * 1000 if self._inference_times else 0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.timesfm_config.estimate_memory_usage()
    
    def get_required_features(self) -> List[str]:
        """Get required features for TimesFM"""
        return [
            'close', 'open', 'high', 'low', 'volume',
            'timestamp'  # For temporal context
        ]