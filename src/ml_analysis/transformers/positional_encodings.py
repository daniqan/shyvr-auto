"""
Positional encoding variants optimized for cryptocurrency time-series data
Supports sinusoidal, learnable, and relative positional encodings with crypto-specific features

Optimized for:
- Variable sequence lengths
- Memory efficiency for GCP deployment  
- Integration with existing TransformerBase
- Crypto market temporal patterns
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encodings"""
    
    # Model dimensions
    d_model: int = 512
    max_length: int = 1000  # Memory constraint for GCP
    dropout: float = 0.1
    
    # Sinusoidal encoding parameters
    temperature: float = 10000.0
    
    # Learnable encoding parameters
    learnable_temperature: float = 10000.0
    
    # Relative encoding parameters
    relative_max_distance: int = 128
    
    # Crypto-specific features
    crypto_aware_features: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # GCP memory constraints
        if self.max_length > 1000:
            raise ValueError("max_length must be <= 1000 for memory constraints")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time-series data
    
    Uses sine and cosine functions of different frequencies to encode positions.
    Provides good interpolation for unseen sequence lengths.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_length = config.max_length
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_length, config.d_model)
        position = torch.arange(0, config.max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for frequency scaling
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * 
            (-math.log(config.temperature) / config.d_model)
        )
        
        # Apply sine to even dimensions, cosine to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_length, d_model)
        
        logger.info(
            "Initialized sinusoidal positional encoding",
            d_model=config.d_model,
            max_length=config.max_length,
            temperature=config.temperature
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_length}")
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    
    Uses trainable embeddings for each position. Can adapt to specific
    temporal patterns in crypto data but may not generalize to unseen lengths.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_length = config.max_length
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(config.max_length, config.d_model)
        
        # Initialize with scaled random values
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        
        logger.info(
            "Initialized learnable positional encoding",
            d_model=config.d_model,
            max_length=config.max_length,
            parameters=self.position_embeddings.weight.numel()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_length}")
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add to input
        x = x + position_embeddings
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for attention mechanisms
    
    Encodes relative distances between positions rather than absolute positions.
    Better for capturing local temporal patterns in financial time-series.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_distance = config.relative_max_distance
        
        # Relative position embeddings
        # We need embeddings for distances from -max_distance to +max_distance
        vocab_size = 2 * config.relative_max_distance + 1
        self.relative_positions = nn.Embedding(vocab_size, config.d_model)
        
        # Initialize
        nn.init.normal_(self.relative_positions.weight, std=0.02)
        
        logger.info(
            "Initialized relative positional encoding",
            d_model=config.d_model,
            max_distance=config.relative_max_distance,
            vocab_size=vocab_size
        )
    
    def _get_relative_distances(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative distance matrix
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Distance matrix of shape (seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len)
        
        # Compute pairwise distances
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        
        # Clip to maximum distance
        distances = torch.clamp(distances, -self.max_distance, self.max_distance)
        
        # Shift to positive indices for embedding lookup
        distances = distances + self.max_distance
        
        return distances
    
    def get_attention_bias(self, seq_len: int) -> torch.Tensor:
        """
        Get attention bias matrix for relative positions
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias matrix of shape (seq_len, seq_len)
        """
        # Get relative distances
        distances = self._get_relative_distances(seq_len)
        
        # Get relative position embeddings
        relative_embeddings = self.relative_positions(distances)
        
        # Project to scalar bias (simple approach)
        # In practice, this would be integrated with attention computation
        bias = relative_embeddings.mean(dim=-1)  # (seq_len, seq_len)
        
        return bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - for now just returns input
        Relative encoding is typically applied during attention computation
        """
        return x


class CryptoAwarePositionalEncoding(nn.Module):
    """
    Crypto-specific positional encoding with market-aware features
    
    Combines standard positional encoding with crypto-specific temporal features:
    - Block time patterns
    - Trading session indicators
    - Crypto event embeddings
    - Seasonal patterns specific to crypto markets
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_length = config.max_length
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Base sinusoidal encoding
        self.base_encoding = SinusoidalPositionalEncoding(config)
        
        if config.crypto_aware_features:
            # Block time encoding (different for BTC, ETH, SOL, etc.)
            self.block_time_encoder = nn.Linear(1, config.d_model // 8)
            
            # Trading session encoding (Asian, European, US)
            self.trading_session_embedding = nn.Embedding(3, config.d_model // 8)
            
            # Crypto event encoding
            self.crypto_event_embedding = nn.Embedding(10, config.d_model // 8)  # 10 event types
            
            # Hour of day encoding (for intraday patterns)
            self.hour_embedding = nn.Embedding(24, config.d_model // 8)
            
            # Day of week encoding  
            self.day_embedding = nn.Embedding(7, config.d_model // 8)
            
            # Month encoding (for seasonal patterns)
            self.month_embedding = nn.Embedding(12, config.d_model // 8)
            
            # Combine all features
            feature_dim = config.d_model // 8 * 6  # 6 feature types
            self.feature_combiner = nn.Linear(feature_dim, config.d_model)
            
            logger.info(
                "Initialized crypto-aware positional encoding",
                d_model=config.d_model,
                crypto_features=True,
                feature_dim=feature_dim
            )
        else:
            logger.info("Initialized crypto-aware encoding without crypto features")
    
    def _extract_temporal_features(self, 
                                 timestamps: Optional[torch.Tensor] = None,
                                 block_times: Optional[torch.Tensor] = None,
                                 trading_sessions: Optional[torch.Tensor] = None,
                                 crypto_events: Optional[torch.Tensor] = None,
                                 hour_of_day: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract crypto-specific temporal features
        
        Args:
            timestamps: Unix timestamps
            block_times: Block time in seconds for each position
            trading_sessions: Trading session indicators (0=Asian, 1=European, 2=US)
            crypto_events: Crypto event types (0=normal, 1=halving, etc.)
            hour_of_day: Hour of day (0-23)
            
        Returns:
            Combined feature tensor
        """
        features = []
        batch_size, seq_len = timestamps.shape if timestamps is not None else (1, 1)
        
        # Block time features
        if block_times is not None:
            # Normalize and encode block times
            normalized_block_times = torch.log(block_times.float() + 1e-6).unsqueeze(-1)
            block_features = self.block_time_encoder(normalized_block_times)
            features.append(block_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        
        # Trading session features
        if trading_sessions is not None:
            session_features = self.trading_session_embedding(trading_sessions)
            features.append(session_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        
        # Crypto event features
        if crypto_events is not None:
            event_features = self.crypto_event_embedding(crypto_events)
            features.append(event_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        
        # Hour of day features
        if hour_of_day is not None:
            hour_features = self.hour_embedding(hour_of_day)
            features.append(hour_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        
        # Default day and month features (zeros for now)
        features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        features.append(torch.zeros(batch_size, seq_len, self.d_model // 8, device=timestamps.device))
        
        # Combine features
        combined_features = torch.cat(features, dim=-1)  # (batch_size, seq_len, feature_dim)
        
        # Project to d_model dimensions
        return self.feature_combiner(combined_features)
    
    def forward(self, 
                x: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                block_times: Optional[torch.Tensor] = None,
                trading_sessions: Optional[torch.Tensor] = None,
                crypto_events: Optional[torch.Tensor] = None,
                hour_of_day: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add crypto-aware positional encoding
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            timestamps: Optional unix timestamps
            block_times: Optional block times in seconds
            trading_sessions: Optional trading session indicators
            crypto_events: Optional crypto event types
            hour_of_day: Optional hour of day (0-23)
            
        Returns:
            Tensor with crypto-aware positional encoding added
        """
        # Start with base sinusoidal encoding
        x = self.base_encoding(x)
        
        # Add crypto-specific features if available
        if hasattr(self, 'feature_combiner') and timestamps is not None:
            crypto_features = self._extract_temporal_features(
                timestamps=timestamps,
                block_times=block_times,
                trading_sessions=trading_sessions,
                crypto_events=crypto_events,
                hour_of_day=hour_of_day
            )
            x = x + crypto_features
        
        return self.dropout(x)


def create_positional_encoding(encoding_type: str, 
                             config: PositionalEncodingConfig) -> nn.Module:
    """
    Factory function to create positional encoding instances
    
    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learnable', 'relative', 'crypto_aware')
        config: Configuration object
        
    Returns:
        Positional encoding module
        
    Raises:
        ValueError: If encoding_type is not supported
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(config)
    elif encoding_type == 'learnable':
        return LearnablePositionalEncoding(config)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(config)
    elif encoding_type == 'crypto_aware':
        return CryptoAwarePositionalEncoding(config)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


# Export all classes and functions
__all__ = [
    'PositionalEncodingConfig',
    'SinusoidalPositionalEncoding',
    'LearnablePositionalEncoding', 
    'RelativePositionalEncoding',
    'CryptoAwarePositionalEncoding',
    'create_positional_encoding'
]