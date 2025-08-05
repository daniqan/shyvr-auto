"""
XAI Transformers Module - Attention-based Explainers for Transformer Models

This module provides specialized explainers for transformer models used in 
time-series forecasting and trading systems. It includes attention weight 
analysis, temporal pattern detection, and cross-asset relationship identification.

Components:
- AttentionExplainer: Base class for attention weight analysis
- TemporalAttentionAnalyzer: Time-series specific attention patterns
- CrossAttentionAnalyzer: Multi-asset relationship analysis
"""

from typing import TYPE_CHECKING

# Import explainer classes when available
try:
    from .attention_explainer import AttentionExplainer
    __all__ = ['AttentionExplainer']
except ImportError:
    __all__ = []

try:
    from .temporal_attention_analyzer import TemporalAttentionAnalyzer
    __all__.append('TemporalAttentionAnalyzer')
except ImportError:
    pass

try:
    from .cross_attention_analyzer import CrossAttentionAnalyzer
    __all__.append('CrossAttentionAnalyzer')
except ImportError:
    pass

# Version info
__version__ = '0.1.0'
__author__ = 'SHYVRAI RLTE Team'

# Module metadata
SUPPORTED_TRANSFORMER_TYPES = [
    'iTransformer',
    'PatchTST', 
    'TimesMixer',
    'TimesFM',
    'TransformerPredictor'
]

EXPLANATION_TYPES = [
    'attention',
    'temporal_attention', 
    'cross_attention'
]