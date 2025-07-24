"""
DEX Integration Module

This module provides a unified interface for interacting with decentralized exchanges (DEXs)
across multiple blockchain networks. It implements a pluggable architecture that allows
seamless integration with different DEXs while maintaining a consistent API.

Supported DEXs:
- Jupiter (Solana) - Best routing and zero protocol fees
- Hyperliquid (Multi-chain) - High-performance perpetuals
- Uniswap V3 (Ethereum/Base) - Deep liquidity pools

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
]