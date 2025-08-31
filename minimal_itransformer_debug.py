"""
Minimal debug script to isolate and fix the iTransformer tensor dimension mismatch issue
"""
import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up minimal logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import minimal required modules directly - avoid config dependencies
try:
    # Try to import the actual modules first
    from src.ml_analysis.transformers.base import TransformerConfig
    from src.ml_analysis.transformers.attention import MultiHeadAttention
    from src.ml_analysis.transformers.positional_encodings import create_positional_encoding, PositionalEncodingConfig
except ImportError as e:
    print(f"Could not import project modules: {e}")
    print("Creating minimal implementations...")
    
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
        
        def estimate_memory_usage_mb(self) -> float:
            return 100.0  # Placeholder
    
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
    class PositionalEncodingConfig:
        d_model: int = 128
        max_length: int = 1000
        dropout: float = 0.1
    
    def create_positional_encoding(encoding_type: str, config: PositionalEncodingConfig):
        return PositionalEncoding(config.d_model, config.max_length, config.dropout)

# Now define the iTransformer classes with debugging
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
        """
        batch_size, seq_len, n_variates, d_model = x.shape
        logger.debug(f"InvertedMultiHeadAttention forward: input_shape={x.shape}, expected_n_variates={self.n_variates}")
        
        # Check that the input variates match the expected variates
        if n_variates != self.n_variates:
            raise RuntimeError(f"Input variates mismatch: expected {self.n_variates}, got {n_variates}. "
                             f"This indicates the model was initialized with different n_variates than the data.")
        
        # Reshape for inverted attention: process each variate separately
        # [batch_size * n_variates, seq_len, d_model]
        x_reshaped = x.view(batch_size * n_variates, seq_len, d_model)
        logger.debug(f"Reshaped for attention: {x_reshaped.shape}, expected_shape={(batch_size * n_variates, seq_len, d_model)}")
        
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
        logger.debug(f"Reshaped attended output intermediate: {attended_output.shape}, expected_shape={(batch_size, n_variates, seq_len, d_model)}")
        
        attended_output = attended_output.transpose(1, 2)  # [batch_size, seq_len, n_variates, d_model]
        logger.debug(f"Final attended output after transpose: {attended_output.shape}, expected_shape={(batch_size, seq_len, n_variates, d_model)}")
        
        # Reshape attention weights
        # [batch_size * n_variates, n_heads, seq_len, seq_len] -> [batch_size, n_variates, n_heads, seq_len, seq_len]
        attention_weights = attention_weights.view(batch_size, n_variates, self.n_heads, seq_len, seq_len)
        
        # Residual connection and layer normalization
        output = self.layer_norm(attended_output + x)
        
        return output, attention_weights

class iTransformerNetwork(nn.Module):
    """
    iTransformer Network with Inverted Attention Mechanism
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
        """
        batch_size, seq_len, n_variates = x.shape
        logger.debug(f"iTransformer forward pass started: input_shape={x.shape}, expected_n_variates={self.config.n_variates}")
        
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
        logger.debug(f"Input expansion completed: x_expanded_shape={x_expanded.shape}")
        
        x_projected = self.input_projection(x_expanded)  # [batch_size, seq_len, n_variates, d_model]
        logger.debug(f"Input projection completed: x_projected_shape={x_projected.shape}")
        
        # Add variate embeddings
        variate_embeds = self.variate_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, n_variates, d_model]
        logger.debug(f"Variate embeddings prepared: variate_embeds_shape={variate_embeds.shape}, x_projected_shape={x_projected.shape}")
        
        # Check for dimension mismatch before addition
        if variate_embeds.shape[2] != x_projected.shape[2]:
            raise RuntimeError(f"Variate dimension mismatch: variate_embeds has {variate_embeds.shape[2]} variates, "
                             f"but input has {x_projected.shape[2]} variates. "
                             f"Expected: {self.config.n_variates}, Got: {n_variates}")
        
        x_embedded = x_projected + variate_embeds
        logger.debug(f"Variate embeddings added: x_embedded_shape={x_embedded.shape}")
        
        # Add positional encoding for time dimension
        # Reshape to apply positional encoding: [batch_size * n_variates, seq_len, d_model]
        x_pos_input = x_embedded.view(batch_size * n_variates, seq_len, self.config.d_model)
        logger.debug(f"Reshaping for positional encoding: x_pos_input_shape={x_pos_input.shape}, expected_shape={(batch_size * n_variates, seq_len, self.config.d_model)}")
        
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
            logger.debug(f"Processing inverted layer {i}")
            # Inverted attention within each variate
            attended, attn_weights = layer['inverted_attention'](hidden, attention_mask)
            attention_weights.append(attn_weights)
            
            # Feed-forward with residual connection
            ff_input = layer['norm1'](attended)
            ff_output = layer['feed_forward'](ff_input)
            hidden = layer['norm2'](ff_output + ff_input)
            
            logger.debug(f"Inverted layer {i} completed: hidden_shape={hidden.shape}")
        
        # Apply temporal fusion for cross-variate information
        if self.temporal_fusion is not None:
            for i, fusion_layer in enumerate(self.temporal_fusion):
                logger.debug(f"Processing temporal fusion layer {i}")
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
                
                logger.debug(f"Temporal fusion layer {i} completed: hidden_shape={hidden.shape}")
        
        # Use final time step for prediction (autoregressive style)
        final_hidden = hidden[:, -1, :, :]  # [batch_size, n_variates, d_model]
        logger.debug(f"Final hidden extracted: final_hidden_shape={final_hidden.shape}")
        
        # Global pooling across variates for global predictions
        global_repr = torch.mean(final_hidden, dim=1)  # [batch_size, d_model]
        logger.debug(f"Global representation: global_repr_shape={global_repr.shape}")
        
        # Generate predictions
        outputs = {}
        
        # Multi-horizon price predictions
        logger.debug(f"Generating predictions: global_repr_shape={global_repr.shape}")
        for horizon, head in self.prediction_heads.items():
            pred_output = head(global_repr)  # [batch_size, n_variates]
            logger.debug(f"Prediction head {horizon}: input_shape={global_repr.shape}, output_shape={pred_output.shape}")
            outputs[f'price_{horizon}'] = pred_output
        
        # Confidence estimation
        outputs['confidence'] = torch.sigmoid(self.confidence_head(global_repr))  # [batch_size, 1]
        
        # Direction predictions for each variate
        direction_logits = self.direction_head(global_repr)  # [batch_size, n_variates * 5]
        logger.debug(f"Direction head output: direction_logits_shape={direction_logits.shape}, expected_shape={(batch_size, self.config.n_variates * 5)}")
        
        # Check dimensions before reshape
        expected_size = self.config.n_variates * 5
        if direction_logits.shape[1] != expected_size:
            raise RuntimeError(f"Direction head output size mismatch: expected {expected_size}, "
                             f"got {direction_logits.shape[1]}. Config n_variates: {self.config.n_variates}, "
                             f"Input n_variates: {n_variates}")
        
        direction_logits = direction_logits.view(batch_size, self.config.n_variates, 5)
        outputs['direction_logits'] = direction_logits
        outputs['direction_probs'] = torch.softmax(direction_logits, dim=-1)
        logger.debug(f"Direction predictions completed: direction_logits_shape={direction_logits.shape}")
        
        # Store attention weights for interpretability
        outputs['attention_weights'] = torch.stack(attention_weights, dim=0)  # [n_layers, batch_size, n_variates, n_heads, seq_len, seq_len]
        outputs['final_representations'] = final_hidden  # For analysis
        
        logger.debug(f"iTransformer forward pass completed successfully")
        return outputs

def create_dummy_corpus_data(n_samples: int = 500, n_features: int = 25) -> pd.DataFrame:
    """Create dummy corpus data that mimics the real structure"""
    
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    # Core OHLCV features
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)  # Random walk prices
    data = {
        'close': prices,
        'open': prices + np.random.randn(n_samples) * 0.5,
        'high': prices + abs(np.random.randn(n_samples)) * 2,
        'low': prices - abs(np.random.randn(n_samples)) * 2,
        'volume': np.random.lognormal(10, 1, n_samples),
    }
    
    # Add technical indicators (to match corpus structure)
    feature_names = [
        'rsi_14', 'rsi_7', 'rsi_21',
        'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_position',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_score', 'volume_score',
        'market_regime', 'trend_strength',
        'sma_20', 'ema_12', 'atr_normalized',
        'funding_rate', 'correlation_score'
    ]
    
    # Generate random feature data
    for feature in feature_names:
        if 'rsi' in feature:
            data[feature] = np.random.uniform(0, 100, n_samples)
        elif 'bb_position' in feature:
            data[feature] = np.random.uniform(-1, 1, n_samples)
        elif 'regime' in feature or 'score' in feature:
            data[feature] = np.random.uniform(0, 1, n_samples)
        else:
            data[feature] = np.random.randn(n_samples)
    
    # Add metadata columns that would be filtered out
    data.update({
        'timestamp': timestamps,
        'token_id': 'WBTC',
        'feature_version': '1.0',
        'data_source': 'debug'
    })
    
    df = pd.DataFrame(data)
    print(f"Created dummy corpus with {len(df)} samples and {len(df.columns)} columns")
    print(f"Feature columns: {[c for c in df.columns if c not in ['timestamp', 'token_id', 'feature_version', 'data_source']][:10]}...")
    
    return df

def prepare_multivariate_sequence(data: pd.DataFrame, sequence_length: int = 96, n_variates: int = 20) -> np.ndarray:
    """
    Prepare multivariate input sequence for iTransformer
    
    Returns:
        Numpy array of shape (n_samples, sequence_length, n_variates)
    """
    # Take the features we need
    metadata_cols = ['timestamp', 'token_id', 'feature_version', 'data_source']
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]
    
    # Select top features
    selected_features = feature_cols[:n_variates]
    
    print(f"Selected {len(selected_features)} features for {n_variates} variates: {selected_features}")
    
    # Extract feature data
    feature_data = data[selected_features].copy()
    
    # Handle missing values
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Normalize each variate independently (important for iTransformer)
    from sklearn.preprocessing import StandardScaler
    feature_data_normalized = pd.DataFrame(index=feature_data.index)
    
    for col in feature_data.columns:
        scaler = StandardScaler()
        feature_data_normalized[col] = scaler.fit_transform(feature_data[[col]]).flatten()
    
    # Create sequences
    sequences = []
    
    for i in range(sequence_length, len(data)):
        # Input sequence - each column is a variate
        seq_features = feature_data_normalized.iloc[i-sequence_length:i].values
        sequences.append(seq_features)
    
    X = np.array(sequences, dtype=np.float32)
    print(f"Prepared sequences: X.shape={X.shape}")
    
    return X

def test_itransformer_tensor_mismatch():
    """Test the specific tensor mismatch issue"""
    
    print("=" * 60)
    print("TESTING iTransformer TENSOR DIMENSION MISMATCH")
    print("=" * 60)
    
    # Create dummy data
    corpus_data = create_dummy_corpus_data()
    
    # Test different n_variates configurations
    test_configs = [
        {'n_variates': 20, 'batch_size': 32},  # Current failing config
        {'n_variates': 25, 'batch_size': 32},  # Match actual features
        {'n_variates': 32, 'batch_size': 32},  # Match batch size
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n--- TEST {i+1}: n_variates={test_config['n_variates']}, batch_size={test_config['batch_size']} ---")
        
        try:
            # Initialize iTransformer with test configuration
            config = InvertedAttentionConfig(
                d_model=128,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                max_seq_length=96,
                n_variates=test_config['n_variates'],
                prediction_horizons=['1h', '4h', '24h'],
                cross_variate_attention=True,
                temporal_fusion_layers=1
            )
            
            print(f"Initializing iTransformer with config: n_variates={config.n_variates}, d_model={config.d_model}")
            model = iTransformerNetwork(config)
            
            # Prepare data
            print(f"Preparing training data from corpus...")
            X = prepare_multivariate_sequence(corpus_data, sequence_length=96, n_variates=test_config['n_variates'])
            
            # Create a small batch for testing
            batch_size = test_config['batch_size']
            if len(X) < batch_size:
                batch_size = len(X)
                
            batch_X = torch.FloatTensor(X[:batch_size])
            
            print(f"Testing with batch: X={batch_X.shape}")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                print("Starting forward pass...")
                outputs = model(batch_X)
                print(f"Forward pass successful! Output keys: {list(outputs.keys())}")
                
                # Check output shapes
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                            
            print(f"✅ TEST {i+1} PASSED")
            
        except Exception as e:
            print(f"❌ TEST {i+1} FAILED: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # If this is the tensor mismatch we're looking for, analyze it
            if "must match the size" in str(e) and ("20" in str(e) and "32" in str(e)):
                print("\n🔍 FOUND THE TENSOR MISMATCH!")
                print(f"Error details: {e}")
                print("This is the issue we need to fix.")
                return False
    
    return True

if __name__ == "__main__":
    success = test_itransformer_tensor_mismatch()
    if success:
        print("\n🎉 All tests passed! The tensor mismatch issue appears to be resolved.")
    else:
        print("\n🔧 Tensor mismatch issue identified. Ready to fix.")