"""
Completely isolated debug script to reproduce and fix the iTransformer tensor dimension mismatch
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_length: int = 1000
    activation: str = 'gelu'
    input_dim: int = None
    output_dim: int = None
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_length: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Multi-head projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        
        if attention_mask is not None:
            scores.masked_fill_(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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
        
        logger.debug(f"InvertedMultiHeadAttention initialized: d_model={d_model}, n_heads={n_heads}, n_variates={n_variates}")
    
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
        logger.debug(f"InvertedMultiHeadAttention forward: input_shape={x.shape}, expected_n_variates={self.n_variates}")
        logger.debug(f"  batch_size={batch_size}, seq_len={seq_len}, n_variates={n_variates}, d_model={d_model}")
        
        # Check that the input variates match the expected variates
        if n_variates != self.n_variates:
            raise RuntimeError(f"❌ TENSOR MISMATCH DETECTED IN InvertedMultiHeadAttention: "
                             f"Input variates {n_variates} != expected {self.n_variates}. "
                             f"This indicates the model was initialized with different n_variates than the data.")
        
        # Reshape for inverted attention: process each variate separately
        # [batch_size * n_variates, seq_len, d_model]
        x_reshaped = x.view(batch_size * n_variates, seq_len, d_model)
        logger.debug(f"Reshaped for attention: {x_reshaped.shape}, expected={(batch_size * n_variates, seq_len, d_model)}")
        
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
        logger.debug(f"Reshaped attended output intermediate: {attended_output.shape}, expected={(batch_size, n_variates, seq_len, d_model)}")
        
        attended_output = attended_output.transpose(1, 2)  # [batch_size, seq_len, n_variates, d_model]
        logger.debug(f"Final attended output after transpose: {attended_output.shape}, expected={(batch_size, seq_len, n_variates, d_model)}")
        
        # Reshape attention weights
        # [batch_size * n_variates, n_heads, seq_len, seq_len] -> [batch_size, n_variates, n_heads, seq_len, seq_len]
        attention_weights = attention_weights.view(batch_size, n_variates, self.n_heads, seq_len, seq_len)
        
        # Residual connection and layer normalization
        output = self.layer_norm(attended_output + x)
        
        return output, attention_weights

class iTransformerNetwork(nn.Module):
    """
    iTransformer Network with Inverted Attention Mechanism
    
    Key innovations:
    1. Inverted attention: time points are tokens, features are channels
    2. Variate-wise embeddings for multivariate correlation
    3. Temporal fusion for better forecasting
    4. Multi-horizon prediction heads
    """
    
    def __init__(self, config: InvertedAttentionConfig):
        super().__init__()
        
        self.config = config
        
        # Variate embedding layer - each feature gets its own embedding
        self.variate_embeddings = nn.Parameter(
            torch.randn(config.n_variates, config.variate_embedding_dim)
        )
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(1, config.d_model)  # Each variate is 1D
        
        # Positional encoding for time steps
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_length=config.max_seq_length,
            dropout=config.dropout
        )
        
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
        
        logger.info(f"iTransformerNetwork initialized: d_model={config.d_model}, n_variates={config.n_variates}, n_layers={config.n_layers}")
    
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
        logger.debug(f"🔍 iTransformer forward pass started:")
        logger.debug(f"  input_shape={x.shape}")
        logger.debug(f"  expected_n_variates={self.config.n_variates}")
        logger.debug(f"  batch_size={batch_size}, seq_len={seq_len}, n_variates={n_variates}")
        
        if n_variates != self.config.n_variates:
            raise ValueError(f"❌ TENSOR MISMATCH DETECTED IN iTransformerNetwork: "
                           f"Input tensor has {n_variates} variates but model expects {self.config.n_variates}. "
                           f"Model may need to be recreated with correct n_variates configuration.")
        
        if seq_len == 0:
            raise ValueError("Empty sequence not supported")
        
        # Validate variate embedding dimensions match input
        expected_variate_embed_shape = (self.config.n_variates, self.config.variate_embedding_dim or self.config.d_model)
        if self.variate_embeddings.shape != expected_variate_embed_shape:
            raise RuntimeError(f"❌ VARIATE EMBEDDING SHAPE MISMATCH: expected {expected_variate_embed_shape}, "
                             f"got {self.variate_embeddings.shape}. Model architecture is inconsistent.")
        
        # Input projection: [batch_size, seq_len, n_variates] -> [batch_size, seq_len, n_variates, d_model]
        x_expanded = x.unsqueeze(-1)  # [batch_size, seq_len, n_variates, 1]
        logger.debug(f"Input expansion completed: x_expanded_shape={x_expanded.shape}")
        
        x_projected = self.input_projection(x_expanded)  # [batch_size, seq_len, n_variates, d_model]
        logger.debug(f"Input projection completed: x_projected_shape={x_projected.shape}")
        
        # Add variate embeddings
        variate_embeds = self.variate_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, n_variates, d_model]
        logger.debug(f"Variate embeddings prepared:")
        logger.debug(f"  variate_embeds_shape={variate_embeds.shape}")
        logger.debug(f"  x_projected_shape={x_projected.shape}")
        
        # Check for dimension mismatch before addition
        if variate_embeds.shape[2] != x_projected.shape[2]:
            raise RuntimeError(f"❌ VARIATE DIMENSION MISMATCH BEFORE ADDITION: "
                             f"variate_embeds has {variate_embeds.shape[2]} variates, "
                             f"but input has {x_projected.shape[2]} variates. "
                             f"Expected: {self.config.n_variates}, Got: {n_variates}")
        
        try:
            x_embedded = x_projected + variate_embeds
            logger.debug(f"✅ Variate embeddings added successfully: x_embedded_shape={x_embedded.shape}")
        except Exception as e:
            logger.error(f"❌ ERROR DURING VARIATE EMBEDDING ADDITION:")
            logger.error(f"  x_projected.shape={x_projected.shape}")
            logger.error(f"  variate_embeds.shape={variate_embeds.shape}")
            logger.error(f"  Error: {e}")
            raise e
        
        # Add positional encoding for time dimension
        # Reshape to apply positional encoding: [batch_size * n_variates, seq_len, d_model]
        x_pos_input = x_embedded.view(batch_size * n_variates, seq_len, self.config.d_model)
        logger.debug(f"Reshaping for positional encoding:")
        logger.debug(f"  x_pos_input_shape={x_pos_input.shape}")
        logger.debug(f"  expected_shape={(batch_size * n_variates, seq_len, self.config.d_model)}")
        
        x_pos_encoded = self.pos_encoding(x_pos_input)
        logger.debug(f"Positional encoding applied: x_pos_encoded_shape={x_pos_encoded.shape}")
        
        # Reshape back: [batch_size, seq_len, n_variates, d_model]
        x_encoded = x_pos_encoded.view(batch_size, seq_len, n_variates, self.config.d_model)
        logger.debug(f"Reshaped back after positional encoding: x_encoded_shape={x_encoded.shape}")
        
        # Store attention weights for interpretability
        attention_weights = []
        
        # Apply inverted transformer layers
        hidden = x_encoded
        for i, layer in enumerate(self.inverted_layers):
            logger.debug(f"🔄 Processing inverted layer {i}")
            logger.debug(f"  hidden_input_shape={hidden.shape}")
            
            try:
                # Inverted attention within each variate
                attended, attn_weights = layer['inverted_attention'](hidden, attention_mask)
                attention_weights.append(attn_weights)
                logger.debug(f"  ✅ Inverted attention completed: attended_shape={attended.shape}")
                
                # Feed-forward with residual connection
                ff_input = layer['norm1'](attended)
                ff_output = layer['feed_forward'](ff_input)
                hidden = layer['norm2'](ff_output + ff_input)
                
                logger.debug(f"  ✅ Inverted layer {i} completed: hidden_shape={hidden.shape}")
            except Exception as e:
                logger.error(f"❌ ERROR IN INVERTED LAYER {i}:")
                logger.error(f"  hidden.shape={hidden.shape}")
                logger.error(f"  Error: {e}")
                raise e
        
        # Apply temporal fusion for cross-variate information
        if self.temporal_fusion is not None:
            for i, fusion_layer in enumerate(self.temporal_fusion):
                logger.debug(f"🔄 Processing temporal fusion layer {i}")
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
                
                logger.debug(f"  ✅ Temporal fusion layer {i} completed: hidden_shape={hidden.shape}")
        
        # Use final time step for prediction (autoregressive style)
        final_hidden = hidden[:, -1, :, :]  # [batch_size, n_variates, d_model]
        logger.debug(f"Final hidden extracted: final_hidden_shape={final_hidden.shape}")
        
        # Global pooling across variates for global predictions
        global_repr = torch.mean(final_hidden, dim=1)  # [batch_size, d_model]
        logger.debug(f"Global representation: global_repr_shape={global_repr.shape}")
        
        # Generate predictions
        outputs = {}
        
        # Multi-horizon price predictions
        logger.debug(f"🎯 Generating predictions: global_repr_shape={global_repr.shape}")
        for horizon, head in self.prediction_heads.items():
            try:
                pred_output = head(global_repr)  # [batch_size, n_variates]
                logger.debug(f"  Prediction head {horizon}: input_shape={global_repr.shape}, output_shape={pred_output.shape}")
                outputs[f'price_{horizon}'] = pred_output
            except Exception as e:
                logger.error(f"❌ ERROR IN PREDICTION HEAD {horizon}:")
                logger.error(f"  global_repr.shape={global_repr.shape}")
                logger.error(f"  Expected output shape: [{batch_size}, {self.config.n_variates}]")
                logger.error(f"  Error: {e}")
                raise e
        
        # Confidence estimation
        try:
            outputs['confidence'] = torch.sigmoid(self.confidence_head(global_repr))  # [batch_size, 1]
        except Exception as e:
            logger.error(f"❌ ERROR IN CONFIDENCE HEAD: {e}")
            raise e
        
        # Direction predictions for each variate
        try:
            direction_logits = self.direction_head(global_repr)  # [batch_size, n_variates * 5]
            logger.debug(f"Direction head output:")
            logger.debug(f"  direction_logits_shape={direction_logits.shape}")
            logger.debug(f"  expected_shape={(batch_size, self.config.n_variates * 5)}")
            
            # Check dimensions before reshape
            expected_size = self.config.n_variates * 5
            if direction_logits.shape[1] != expected_size:
                raise RuntimeError(f"❌ DIRECTION HEAD OUTPUT SIZE MISMATCH: "
                                 f"expected {expected_size}, got {direction_logits.shape[1]}. "
                                 f"Config n_variates: {self.config.n_variates}, Input n_variates: {n_variates}")
            
            direction_logits = direction_logits.view(batch_size, self.config.n_variates, 5)
            outputs['direction_logits'] = direction_logits
            outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
            logger.debug(f"  ✅ Direction predictions completed: direction_logits_shape={direction_logits.shape}")
        except Exception as e:
            logger.error(f"❌ ERROR IN DIRECTION HEAD:")
            logger.error(f"  global_repr.shape={global_repr.shape}")
            logger.error(f"  Expected output shape: [{batch_size}, {self.config.n_variates * 5}]")
            logger.error(f"  Error: {e}")
            raise e
        
        # Store attention weights for interpretability
        outputs['attention_weights'] = torch.stack(attention_weights, dim=0)  # [n_layers, batch_size, n_variates, n_heads, seq_len, seq_len]
        outputs['final_representations'] = final_hidden  # For analysis
        
        logger.debug(f"🎉 iTransformer forward pass completed successfully")
        return outputs

def test_itransformer_tensor_mismatch():
    """Test the specific tensor mismatch issue"""
    
    print("=" * 80)
    print("🔬 TESTING iTransformer TENSOR DIMENSION MISMATCH")
    print("=" * 80)
    
    # Test different n_variates configurations that should trigger the issue
    test_configs = [
        {
            'name': 'Original Failing Config',
            'n_variates': 20, 
            'batch_size': 32, 
            'sequence_length': 96,
            'actual_features': 25  # This mismatch should trigger the error
        },
        {
            'name': 'Matching Config',
            'n_variates': 25, 
            'batch_size': 32,
            'sequence_length': 96,
            'actual_features': 25  # This should work
        },
        {
            'name': 'Batch Size Match Test',
            'n_variates': 32, 
            'batch_size': 32,
            'sequence_length': 96,
            'actual_features': 32  # This should work
        }
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {test_config['name']}")
        print(f"  n_variates: {test_config['n_variates']}")
        print(f"  batch_size: {test_config['batch_size']}")
        print(f"  sequence_length: {test_config['sequence_length']}")
        print(f"  actual_features: {test_config['actual_features']}")
        print(f"{'='*60}")
        
        try:
            # Initialize iTransformer with test configuration
            config = InvertedAttentionConfig(
                d_model=128,
                n_heads=8,
                n_layers=2,
                d_ff=512,
                dropout=0.1,
                max_seq_length=test_config['sequence_length'],
                n_variates=test_config['n_variates'],
                prediction_horizons=['1h', '4h', '24h'],
                cross_variate_attention=True,
                temporal_fusion_layers=1
            )
            
            print(f"✅ Initializing iTransformer with:")
            print(f"   n_variates={config.n_variates}, d_model={config.d_model}")
            model = iTransformerNetwork(config)
            
            # Create dummy input data with the specified dimensions
            batch_size = test_config['batch_size']
            sequence_length = test_config['sequence_length']
            actual_features = test_config['actual_features']
            
            # Create input tensor with ACTUAL features (this might not match n_variates)
            batch_X = torch.randn(batch_size, sequence_length, actual_features)
            
            print(f"✅ Created test input tensor: {batch_X.shape}")
            print(f"   batch_size={batch_size}, seq_len={sequence_length}, n_features={actual_features}")
            
            # Test forward pass - this should fail if there's a mismatch
            model.eval()
            with torch.no_grad():
                print(f"🚀 Starting forward pass...")
                outputs = model(batch_X)
                print(f"✅ Forward pass successful! Output keys: {list(outputs.keys())}")
                
                # Check output shapes
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape}")
                            
            print(f"✅✅ TEST {i+1} PASSED - {test_config['name']}")
            
        except Exception as e:
            print(f"❌❌ TEST {i+1} FAILED - {test_config['name']}: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Check if this is the tensor mismatch we're looking for
            error_str = str(e)
            if "must match the size" in error_str and ("20" in error_str and "32" in error_str):
                print("\n🔍🔍 FOUND THE EXACT TENSOR MISMATCH!")
                print(f"Error details: {e}")
                print("This is the (20) vs (32) dimension issue we need to fix.")
                
            elif "TENSOR MISMATCH DETECTED" in error_str:
                print("\n🔍 FOUND A CONFIGURATION MISMATCH!")
                print(f"This is our added validation catching the issue.")
                print("The model expects a different number of variates than the input data.")
                
            import traceback
            traceback.print_exc()
            
            # Continue with other tests
            continue
    
    print(f"\n{'='*80}")
    print("🎯 ANALYSIS COMPLETE")
    print("The tensor mismatch occurs when the model is configured with n_variates")
    print("that doesn't match the actual number of features in the input data.")
    print("The error (20) vs (32) suggests:")
    print("  - Model configured with 20 variates")
    print("  - But receiving data with different dimensions")
    print("  - Batch size is 32, which might be getting confused with variates")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    test_itransformer_tensor_mismatch()