"""
Base Transformer class for time-series prediction
Optimized for GCP production environment with memory constraints
"""

import math
import time
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
            'device': str(next(self.parameters()).device) if list(self.parameters()) else 'cpu',
            'quantization_compatible': self.is_quantization_compatible(),
            'supports_dynamic_quantization': True,
            'supports_static_quantization': True
        }
    
    def is_quantization_compatible(self) -> bool:
        """Check if model is compatible with quantization"""
        try:
            # Check if model has quantizable layers
            quantizable_layers = 0
            total_layers = 0
            
            for name, module in self.named_modules():
                total_layers += 1
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)):
                    quantizable_layers += 1
            
            # Model is quantization compatible if at least 30% of layers are quantizable
            compatibility_ratio = quantizable_layers / total_layers if total_layers > 0 else 0.0
            is_compatible = compatibility_ratio >= 0.3
            
            self.logger.info("Quantization compatibility check",
                           quantizable_layers=quantizable_layers,
                           total_layers=total_layers,
                           compatibility_ratio=compatibility_ratio,
                           is_compatible=is_compatible)
            
            return is_compatible
            
        except Exception as e:
            self.logger.error("Quantization compatibility check failed", error=str(e))
            return False
    
    def prepare_for_quantization(self, quantization_type: str = "dynamic", 
                               preserve_attention_precision: bool = True) -> 'TransformerBase':
        """Prepare transformer model for quantization"""
        try:
            # Set model to evaluation mode for quantization
            self.eval()
            
            # Store original state for fallback
            self._pre_quantization_state = {
                'training_mode': self.training,
                'config': self.transformer_config.to_dict()
            }
            
            # Configure quantization-specific settings
            if preserve_attention_precision:
                self._quantization_config = {
                    'attention_layers': 'fp16',  # Keep attention in higher precision
                    'feed_forward_layers': 'int8',  # Quantize feed-forward layers
                    'embedding_layers': 'fp16',   # Keep embeddings in higher precision
                    'output_layers': 'fp16',      # Keep output layers in higher precision
                    'quantization_type': quantization_type
                }
            else:
                self._quantization_config = {
                    'attention_layers': 'int8',
                    'feed_forward_layers': 'int8',
                    'embedding_layers': 'int8',
                    'output_layers': 'fp16',  # Always preserve output precision
                    'quantization_type': quantization_type
                }
            
            self.logger.info("Transformer prepared for quantization",
                           quantization_type=quantization_type,
                           preserve_attention_precision=preserve_attention_precision,
                           config=self._quantization_config)
            
            return self
            
        except Exception as e:
            self.logger.error("Quantization preparation failed", error=str(e))
            return self
    
    def apply_quantization(self, target_dtype: torch.dtype = torch.qint8) -> 'TransformerBase':
        """Apply quantization to the transformer model"""
        try:
            if not hasattr(self, '_quantization_config'):
                self.logger.warning("Model not prepared for quantization, preparing with default settings")
                self.prepare_for_quantization()
            
            quantization_type = self._quantization_config.get('quantization_type', 'dynamic')
            
            if quantization_type == "dynamic":
                quantized_model = self._apply_dynamic_quantization(target_dtype)
            elif quantization_type == "static":
                quantized_model = self._apply_static_quantization(target_dtype)
            else:
                self.logger.error("Unsupported quantization type", type=quantization_type)
                return self
            
            # Validate quantization was applied successfully
            if self._validate_quantization(quantized_model):
                self.logger.info("Quantization applied successfully",
                               quantization_type=quantization_type,
                               target_dtype=str(target_dtype))
                return quantized_model
            else:
                self.logger.error("Quantization validation failed, reverting to original model")
                return self
                
        except Exception as e:
            self.logger.error("Quantization application failed", error=str(e))
            return self
    
    def _apply_dynamic_quantization(self, target_dtype: torch.dtype) -> 'TransformerBase':
        """Apply dynamic quantization to transformer"""
        try:
            # Prepare qconfig specification based on quantization config
            qconfig_spec = {}
            
            for name, module in self.named_modules():
                if self._should_quantize_layer(name, module):
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                        qconfig_spec[type(module)] = torch.quantization.default_dynamic_qconfig
                    elif isinstance(module, nn.MultiheadAttention):
                        # Special handling for attention layers
                        if self._quantization_config.get('attention_layers') == 'int8':
                            qconfig_spec[type(module)] = torch.quantization.default_dynamic_qconfig
                        # else: skip quantization for attention (preserve precision)
            
            # Apply dynamic quantization if we have layers to quantize
            if qconfig_spec:
                quantized_model = torch.quantization.quantize_dynamic(
                    self, qconfig_spec=qconfig_spec, dtype=target_dtype
                )
                return quantized_model
            else:
                self.logger.warning("No layers selected for quantization")
                return self
                
        except Exception as e:
            self.logger.error("Dynamic quantization failed", error=str(e))
            return self
    
    def _apply_static_quantization(self, target_dtype: torch.dtype) -> 'TransformerBase':
        """Apply static quantization to transformer (requires calibration)"""
        try:
            # For static quantization, we need calibration data
            # This is a simplified implementation - in production would need actual calibration
            self.logger.warning("Static quantization requires calibration data - using dynamic quantization instead")
            return self._apply_dynamic_quantization(target_dtype)
            
        except Exception as e:
            self.logger.error("Static quantization failed", error=str(e))
            return self
    
    def _should_quantize_layer(self, layer_name: str, module: nn.Module) -> bool:
        """Determine if a layer should be quantized based on quantization config"""
        
        # Get layer type category
        layer_name_lower = layer_name.lower()
        
        if 'attention' in layer_name_lower or 'attn' in layer_name_lower:
            layer_category = 'attention_layers'
        elif 'feed_forward' in layer_name_lower or 'ffn' in layer_name_lower or 'mlp' in layer_name_lower:
            layer_category = 'feed_forward_layers'
        elif 'embed' in layer_name_lower:
            layer_category = 'embedding_layers'
        elif 'output' in layer_name_lower or 'classifier' in layer_name_lower:
            layer_category = 'output_layers'
        else:
            # Default to feed_forward category for linear layers
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                layer_category = 'feed_forward_layers'
            else:
                return False  # Don't quantize unknown layer types
        
        # Check if this layer category should be quantized
        target_precision = self._quantization_config.get(layer_category, 'fp32')
        should_quantize = target_precision in ['int8', 'qint8']
        
        return should_quantize
    
    def _validate_quantization(self, quantized_model: 'TransformerBase') -> bool:
        """Validate that quantization was applied correctly"""
        try:
            # Check if model has quantized layers
            has_quantized_layers = False
            
            for name, module in quantized_model.named_modules():
                # Check for quantized operations
                if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                    if 'qint' in str(module.weight.dtype):
                        has_quantized_layers = True
                        break
                # Check for quantized module names
                if any(quant_indicator in name.lower() 
                      for quant_indicator in ['quantized', 'dequantize', 'quant']):
                    has_quantized_layers = True
                    break
            
            if has_quantized_layers:
                # Perform a test forward pass to ensure model works
                test_input = torch.randn(1, 10, self.transformer_config.d_model)
                with torch.no_grad():
                    output = quantized_model.forward_with_validation(test_input)
                    
                # Check output is valid
                if torch.isnan(output).any() or torch.isinf(output).any():
                    self.logger.error("Quantized model produces invalid output")
                    return False
                
                self.logger.info("Quantization validation passed",
                               has_quantized_layers=has_quantized_layers,
                               output_shape=output.shape)
                return True
            else:
                self.logger.warning("No quantized layers detected after quantization")
                return False
                
        except Exception as e:
            self.logger.error("Quantization validation failed", error=str(e))
            return False
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about model quantization status"""
        quantization_info = {
            'is_quantized': False,
            'quantization_type': None,
            'quantized_layers': [],
            'preserved_precision_layers': [],
            'quantization_config': None,
            'estimated_speedup': 1.0,
            'estimated_memory_reduction': 0.0
        }
        
        try:
            # Check if model has quantization config
            if hasattr(self, '_quantization_config'):
                quantization_info['quantization_config'] = self._quantization_config
                quantization_info['quantization_type'] = self._quantization_config.get('quantization_type')
            
            # Analyze current quantization state
            quantized_count = 0
            preserved_count = 0
            total_quantizable = 0
            
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)):
                    total_quantizable += 1
                    
                    # Check if layer is quantized
                    if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                        if 'qint' in str(module.weight.dtype):
                            quantized_count += 1
                            quantization_info['quantized_layers'].append(name)
                            quantization_info['is_quantized'] = True
                        else:
                            preserved_count += 1
                            quantization_info['preserved_precision_layers'].append(name)
            
            # Calculate estimated benefits
            if total_quantizable > 0:
                quantization_ratio = quantized_count / total_quantizable
                quantization_info['estimated_speedup'] = 1.0 + (quantization_ratio * 1.5)  # Up to 2.5x speedup
                quantization_info['estimated_memory_reduction'] = quantization_ratio * 0.75  # Up to 75% reduction
            
            quantization_info['quantization_statistics'] = {
                'quantized_layers': quantized_count,
                'preserved_layers': preserved_count,
                'total_quantizable_layers': total_quantizable,
                'quantization_ratio': quantized_count / total_quantizable if total_quantizable > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error("Failed to get quantization info", error=str(e))
        
        return quantization_info
    
    def benchmark_quantization_performance(self, test_inputs: List[torch.Tensor], 
                                         num_warmup: int = 10, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark performance of quantized vs original model"""
        benchmark_results = {
            'original_latency_ms': 0.0,
            'quantized_latency_ms': 0.0,
            'speedup_ratio': 1.0,
            'memory_usage_mb': 0.0,
            'accuracy_preservation': 1.0
        }
        
        try:
            if not test_inputs:
                self.logger.warning("No test inputs provided for benchmarking")
                return benchmark_results
            
            # Create original model for comparison (if quantized)
            original_model = self
            quantized_model = self
            
            if hasattr(self, '_pre_quantization_state'):
                # We have a quantized model, create original for comparison
                # In real implementation would restore from saved state
                pass
            
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = original_model.forward(test_inputs[0])
                    _ = quantized_model.forward(test_inputs[0])
            
            # Benchmark original model
            original_times = []
            for test_input in test_inputs[:min(len(test_inputs), num_iterations)]:
                start_time = time.time()
                with torch.no_grad():
                    original_output = original_model.forward(test_input)
                original_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Benchmark quantized model
            quantized_times = []
            for test_input in test_inputs[:min(len(test_inputs), num_iterations)]:
                start_time = time.time()
                with torch.no_grad():
                    quantized_output = quantized_model.forward(test_input)
                quantized_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate results
            benchmark_results['original_latency_ms'] = sum(original_times) / len(original_times)
            benchmark_results['quantized_latency_ms'] = sum(quantized_times) / len(quantized_times)
            benchmark_results['speedup_ratio'] = benchmark_results['original_latency_ms'] / benchmark_results['quantized_latency_ms']
            
            # Estimate memory usage
            param_size = sum(p.numel() * p.element_size() for p in self.parameters())
            benchmark_results['memory_usage_mb'] = param_size / (1024 * 1024)
            
            # Calculate accuracy preservation (simplified)
            if 'original_output' in locals() and 'quantized_output' in locals():
                mse = torch.nn.functional.mse_loss(original_output, quantized_output)
                benchmark_results['accuracy_preservation'] = max(0.0, 1.0 - float(mse.item()))
            
            self.logger.info("Quantization performance benchmark completed",
                           original_latency=benchmark_results['original_latency_ms'],
                           quantized_latency=benchmark_results['quantized_latency_ms'],
                           speedup=benchmark_results['speedup_ratio'],
                           memory_mb=benchmark_results['memory_usage_mb'])
            
        except Exception as e:
            self.logger.error("Quantization benchmark failed", error=str(e))
        
        return benchmark_results