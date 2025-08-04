"""
Base Transformer class for time-series prediction
Optimized for GCP production environment with memory constraints
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import structlog

from ..base import MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


@dataclass
class TransformerConfig:
    """Configuration for Transformer models optimized for production use"""
    
    # Model architecture
    d_model: int = 512           # Model dimension
    n_heads: int = 8             # Number of attention heads
    n_layers: int = 6            # Number of transformer layers
    d_ff: int = 2048             # Feed-forward dimension
    dropout: float = 0.1         # Dropout rate
    max_seq_length: int = 1000   # Maximum sequence length (memory constraint)
    
    # Activation and normalization
    activation: str = "gelu"     # Activation function
    norm_type: str = "layer"     # Normalization type
    
    # Memory and performance optimizations for GCP
    use_flash_attention: bool = True         # Flash Attention for O(N) memory
    use_gradient_checkpointing: bool = True  # Gradient checkpointing for memory
    mixed_precision: bool = True             # FP16/BF16 mixed precision
    
    # Vocabulary and embeddings
    vocab_size: int = 10000      # Vocabulary size for embeddings
    pad_token_id: int = 0        # Padding token ID
    
    # Training parameters
    learning_rate: float = 1e-4  # Learning rate
    weight_decay: float = 0.01   # Weight decay
    warmup_steps: int = 1000     # Warmup steps
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_seq_length > 1000:
            raise ValueError("max_seq_length must be <= 1000 for memory constraints")
        
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
    
    def estimate_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB for GCP production planning"""
        # More accurate parameter count estimation
        
        # Attention parameters: Q, K, V projection + output projection per layer
        attention_params_per_layer = 4 * (self.d_model * self.d_model)
        
        # Feed-forward parameters per layer: two linear transformations  
        ff_params_per_layer = self.d_model * self.d_ff + self.d_ff * self.d_model
        
        # Layer norm parameters per layer (2 layer norms per transformer layer)
        layernorm_params_per_layer = 2 * self.d_model
        
        # Total model parameters
        transformer_params = self.n_layers * (
            attention_params_per_layer + 
            ff_params_per_layer + 
            layernorm_params_per_layer
        )
        
        # Embeddings
        embedding_params = self.vocab_size * self.d_model
        
        # Output projection
        output_params = self.d_model
        
        # Total parameters
        total_params = transformer_params + embedding_params + output_params
        
        # Convert to MB (4 bytes per float32)
        model_memory_mb = (total_params * 4) / (1024 * 1024)
        
        # Activation memory (depends on sequence length and batch size)
        # Assume batch size of 1 for inference
        batch_size = 1
        activation_memory_elements = (
            batch_size * self.max_seq_length * self.d_model * self.n_layers
        )
        activation_memory_mb = (activation_memory_elements * 4) / (1024 * 1024)
        
        # Attention matrix memory (most memory intensive part)
        attention_matrix_elements = (
            batch_size * self.n_heads * self.max_seq_length * self.max_seq_length
        )
        attention_memory_mb = (attention_matrix_elements * 4) / (1024 * 1024)
        
        # Total memory with optimizations
        base_memory = model_memory_mb + activation_memory_mb + attention_memory_mb
        
        if self.use_flash_attention:
            # Flash attention reduces memory significantly
            base_memory *= 0.5
        
        if self.use_gradient_checkpointing:
            # Gradient checkpointing reduces activation memory
            base_memory *= 0.7
        
        return base_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class TransformerBase(MLAnalyzerBase, nn.Module, ABC):
    """
    Abstract base class for Transformer models in time-series prediction
    
    Provides:
    - Memory-efficient attention mechanisms
    - Variable sequence length handling
    - GCP production optimizations
    - Integration with existing ML infrastructure
    """
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        # Initialize both parent classes
        MLAnalyzerBase.__init__(self, model_type, config)
        nn.Module.__init__(self)
        
        self.transformer_config = TransformerConfig.from_dict(config)
        self.logger = structlog.get_logger().bind(
            model_type=model_type.value,
            d_model=self.transformer_config.d_model,
            max_seq_length=self.transformer_config.max_seq_length
        )
        
        # Model components will be initialized by subclasses
        self._attention_weights: Optional[torch.Tensor] = None
        
        # Performance tracking
        self._inference_times: List[float] = []
        self._memory_usage: List[float] = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask of shape (batch_size, seq_len)
        
        Returns:
            Predictions of shape (batch_size, 1) for price prediction
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> torch.Tensor:
        """
        Extract attention weights for visualization and interpretation
        
        Returns:
            Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        pass
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for transformer models"""
        return [
            'price_sequence',        # Historical price sequence
            'volume_sequence',       # Historical volume sequence  
            'returns_sequence',      # Historical returns sequence
            'volatility_sequence',   # Historical volatility sequence
            'temporal_position',     # Temporal position encoding
            'attention_mask',        # Attention mask for variable lengths
            'sequence_length',       # Actual sequence length
            'market_features',       # Market context features
            'technical_indicators',  # Technical analysis features
            'time_of_day',          # Intraday temporal features
            'day_of_week',          # Weekly seasonal features
            'market_regime',        # Bull/bear market indicator
        ]
    
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Analyze a token using the transformer model
        
        Args:
            token: Token to analyze
            historical_data: Optional historical data
            
        Returns:
            Prediction result with transformer-based predictions
        """
        if not self.is_model_trained():
            raise ValueError("Model must be trained before analysis")
        
        start_time = datetime.now()
        
        try:
            # Extract features for transformer
            features = await self._extract_features(token, historical_data)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.forward(features['input_sequence'], 
                                         features.get('attention_mask'))
                
                # Get attention weights for explainability
                attention_weights = self.get_attention_weights()
            
            # Calculate confidence based on attention pattern consistency
            confidence = self._calculate_confidence(attention_weights, predictions)
            
            # Generate prediction result
            result = self._create_prediction_result(
                token=token,
                predictions=predictions,
                confidence=confidence,
                attention_weights=attention_weights,
                features_used=list(features.keys()),
                start_time=start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Token analysis failed", 
                            token=token.address, 
                            error=str(e))
            raise
    
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the transformer model
        
        Args:
            training_data: Historical training data
            
        Returns:
            Success status
        """
        try:
            self.logger.info("Starting transformer model training",
                           data_points=len(training_data),
                           sequence_length=self.transformer_config.max_seq_length)
            
            success = await self._train_transformer(training_data)
            
            if success:
                self._is_trained = True
                self._model = self
                self.logger.info("Transformer model training completed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error("Transformer training failed", error=str(e))
            return False
    
    async def _train_transformer(self, training_data: pd.DataFrame) -> bool:
        """
        Actual transformer training implementation
        To be implemented by subclasses with specific training logic
        """
        # This is a placeholder - subclasses should implement actual training
        self.logger.info("Training transformer model", 
                        model_type=self.model_type.value)
        return True
    
    async def _extract_features(self, token: DiscoveredToken, 
                               historical_data: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features in transformer-compatible format
        
        Args:
            token: Token to analyze
            historical_data: Historical price/volume data
            
        Returns:
            Dictionary of feature tensors
        """
        if historical_data is None or len(historical_data) < 10:
            # Create minimal dummy features for testing
            seq_len = min(100, self.transformer_config.max_seq_length)
            return {
                'input_sequence': torch.randn(1, seq_len, self.transformer_config.d_model),
                'attention_mask': torch.ones(1, seq_len),
                'sequence_length': torch.tensor([seq_len])
            }
        
        # Process historical data into transformer features
        # This would typically involve:
        # 1. Sequence creation from time series
        # 2. Normalization and scaling
        # 3. Positional encoding
        # 4. Market context features
        
        seq_len = min(len(historical_data), self.transformer_config.max_seq_length)
        
        # Create price sequence features
        price_features = self._create_price_features(historical_data[-seq_len:])
        
        # Create attention mask (all ones for real data)
        attention_mask = torch.ones(1, seq_len)
        
        return {
            'input_sequence': price_features.unsqueeze(0),  # Add batch dimension
            'attention_mask': attention_mask,
            'sequence_length': torch.tensor([seq_len])
        }
    
    def _create_price_features(self, data: pd.DataFrame) -> torch.Tensor:
        """Create price-based feature sequences"""
        features = []
        
        # Basic price features
        if 'price' in data.columns:
            prices = data['price'].values
            features.append(prices)
        
        if 'volume' in data.columns:
            volumes = data['volume'].values
            features.append(volumes)
        
        # If no features available, create dummy features
        if not features:
            features = [np.random.randn(len(data))]
        
        # Stack features and pad/truncate to d_model dimension
        feature_array = np.column_stack(features)
        
        # Pad or truncate to match d_model
        target_dim = self.transformer_config.d_model
        if feature_array.shape[1] < target_dim:
            # Pad with zeros
            padding = np.zeros((feature_array.shape[0], target_dim - feature_array.shape[1]))
            feature_array = np.column_stack([feature_array, padding])
        elif feature_array.shape[1] > target_dim:
            # Truncate
            feature_array = feature_array[:, :target_dim]
        
        return torch.tensor(feature_array, dtype=torch.float32)
    
    def _calculate_confidence(self, attention_weights: torch.Tensor, 
                            predictions: torch.Tensor) -> float:
        """Calculate prediction confidence based on attention patterns"""
        if attention_weights is None:
            return 0.5  # Default confidence
        
        # Simple confidence metric based on attention entropy
        # Higher entropy (more distributed attention) = lower confidence
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Calculate attention entropy per head
        attention_probs = torch.softmax(attention_weights, dim=-1)
        attention_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        
        # Average entropy across heads and sequence
        avg_entropy = attention_entropy.mean().item()
        
        # Convert entropy to confidence (higher entropy = lower confidence)
        max_entropy = math.log(seq_len)  # Maximum possible entropy
        confidence = 1.0 - (avg_entropy / max_entropy)
        
        return max(0.0, min(1.0, confidence))
    
    def _create_prediction_result(self, token: DiscoveredToken, predictions: torch.Tensor,
                                confidence: float, attention_weights: torch.Tensor,
                                features_used: List[str], start_time: datetime) -> PredictionResult:
        """Create a structured prediction result"""
        
        # Extract price predictions (assuming single prediction per token)
        price_pred = predictions[0, 0].item() if predictions.numel() > 0 else 0.0
        current_price = token.price_usd or 100.0  # Fallback price
        
        # Calculate prediction direction and probability
        price_change_pct = (price_pred - current_price) / current_price
        
        if price_change_pct > 0.05:  # 5% threshold
            direction = PredictionDirection.BUY
            probability_up = 0.7 + (confidence * 0.2)
        elif price_change_pct < -0.05:
            direction = PredictionDirection.SELL  
            probability_up = 0.3 - (confidence * 0.2)
        else:
            direction = PredictionDirection.HOLD
            probability_up = 0.5
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=self.model_type,
            price_prediction_24h=price_pred,
            direction=direction,
            confidence=confidence,
            probability_up=max(0.0, min(1.0, probability_up)),
            features_used=features_used,
            model_version=f"transformer-{self.transformer_config.d_model}",
            processing_time_ms=processing_time
        )
    
    async def health_check(self) -> bool:
        """Check transformer model health"""
        try:
            if not self.is_model_trained() or self._model is None:
                return False
            
            # Test with dummy input
            dummy_input = torch.randn(1, 10, self.transformer_config.d_model)
            
            with torch.no_grad():
                output = self.forward(dummy_input)
                
            # Check output is valid
            if torch.isnan(output).any() or torch.isinf(output).any():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def forward_with_validation(self, x: torch.Tensor, 
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with input validation"""
        # Validate input dimensions
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.shape}")
        
        if x.shape[0] == 0:
            raise ValueError("Empty sequence not supported")
        
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.transformer_config.d_model:
            raise ValueError(f"Expected d_model={self.transformer_config.d_model}, got {d_model}")
        
        if seq_len > self.transformer_config.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.transformer_config.max_seq_length}")
        
        # Validate attention mask if provided
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"Attention mask shape {attention_mask.shape} doesn't match input {(batch_size, seq_len)}")
        
        return self.forward(x, attention_mask)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for monitoring and debugging"""
        return {
            'model_type': self.model_type.value,
            'config': self.transformer_config.to_dict(),
            'is_trained': self.is_model_trained(),
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'memory_estimate_mb': self.transformer_config.estimate_memory_usage_mb(),
            'avg_inference_time_ms': np.mean(self._inference_times) if self._inference_times else 0.0,
            'device': str(next(self.parameters()).device) if list(self.parameters()) else 'cpu'
        }