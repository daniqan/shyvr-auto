"""
Transformer models for time-series prediction in crypto trading
"""

# Import order: base components first, then specific implementations
from .base import TransformerBase, TransformerConfig
from .attention import MultiHeadAttention, AttentionVisualization
from .positional_encodings import (
    PositionalEncodingConfig,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    RelativePositionalEncoding,
    CryptoAwarePositionalEncoding,
    create_positional_encoding
)
from .temporal_embeddings import (
    TemporalEmbeddingConfig,
    TemporalEmbedding,
    FinancialTemporalEmbedding,
    CryptoMarketEmbedding,
    TradingSessionEncoder,
    MarketRegimeEncoder,
    VolatilityRegimeEncoder
)
from .transformer_predictor import TransformerPredictor

__all__ = [
    # Base components
    'TransformerBase',
    'TransformerConfig',
    'MultiHeadAttention',
    'AttentionVisualization',
    # Positional encodings
    'PositionalEncodingConfig',
    'SinusoidalPositionalEncoding',
    'LearnablePositionalEncoding',
    'RelativePositionalEncoding',
    'CryptoAwarePositionalEncoding',
    'create_positional_encoding',
    # Temporal embeddings
    'TemporalEmbeddingConfig',
    'TemporalEmbedding',
    'FinancialTemporalEmbedding',
    'CryptoMarketEmbedding',
    'TradingSessionEncoder',
    'MarketRegimeEncoder',
    'VolatilityRegimeEncoder',
    # Model implementations
    'TransformerPredictor',
]