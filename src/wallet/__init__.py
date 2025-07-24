"""
Wallet integration module for multi-chain trading operations.

This module provides secure wallet connectivity and transaction management
for Ethereum and Solana blockchain networks.
"""

from .base import (
    WalletBase,
    WalletConfig,
    WalletError,
    WalletConnectionError,
    WalletTransactionError,
    NetworkType,
    Chain,
    TransactionStatus,
    WalletBalance,
    TransactionResult,
)

from .config import WalletConfigManager

__all__ = [
    # Base classes and types
    "WalletBase",
    "WalletConfig", 
    "WalletError",
    "WalletConnectionError",
    "WalletTransactionError",
    "NetworkType",
    "Chain",
    "TransactionStatus",
    "WalletBalance",
    "TransactionResult",
    # Configuration
    "WalletConfigManager",
]