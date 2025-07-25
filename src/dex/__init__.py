"""
DEX Integration Module

This module provides a unified interface for interacting with decentralized exchanges (DEXs)
across multiple blockchain networks. It implements a pluggable architecture that allows
seamless integration with different DEXs while maintaining a consistent API.

Supported DEXs:
- Jupiter (Solana) - Best routing and zero protocol fees ✅
- Hyperliquid (Multi-chain) - High-performance perpetuals ✅
- Uniswap V3 (Ethereum/Base) - Concentrated liquidity pools ✅

The module follows the established patterns from the wallet and discovery modules,
providing async operations, comprehensive error handling, and production-ready
implementations suitable for automated trading systems.
"""

from .base import (
    DEXBase,
    SwapQuote,
    SwapResult,
    SwapStatus,
    DEXConfig,
    DEXError,
    DEXConnectionError,
    DEXTransactionError,
    DEXRateLimitError,
    SwapType,
    PriceImpact,
)
from .jupiter_client import JupiterDEXClient
from .uniswap_v3_client import UniswapV3Client
from .hyperliquid_client import HyperliquidDEXClient

__all__ = [
    "DEXBase",
    "SwapQuote", 
    "SwapResult",
    "SwapStatus",
    "DEXConfig",
    "DEXError",
    "DEXConnectionError", 
    "DEXTransactionError",
    "DEXRateLimitError",
    "SwapType",
    "PriceImpact",
    "JupiterDEXClient",
    "UniswapV3Client",
    "HyperliquidDEXClient",
]