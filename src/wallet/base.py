"""
Base wallet interface and data structures for multi-chain wallet operations.

This module defines the abstract base class for wallet implementations and
common data structures used across different blockchain networks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Network type enumeration."""
    MAINNET = "mainnet"
    TESTNET = "testnet"
    DEVNET = "devnet"


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    BASE = "base"
    POLYGON = "polygon"


class TransactionStatus(Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class WalletConfig:
    """Wallet configuration data structure."""
    chain: Chain
    network: NetworkType
    private_key: Optional[str] = None
    mnemonic: Optional[str] = None
    wallet_address: Optional[str] = None
    rpc_url: Optional[str] = None
    api_key: Optional[str] = None
    gas_price_gwei: Optional[float] = None
    max_gas_limit: Optional[int] = None
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.private_key and not self.mnemonic and not self.wallet_address:
            raise ValueError("Must provide private_key, mnemonic, or wallet_address")


@dataclass
class WalletBalance:
    """Wallet balance information."""
    native_balance: Decimal  # ETH, SOL, etc.
    native_symbol: str
    token_balances: Dict[str, Decimal]  # Token address -> balance
    total_usd_value: Optional[Decimal] = None
    
    def __str__(self) -> str:
        return f"{self.native_balance} {self.native_symbol}"


@dataclass
class TransactionResult:
    """Transaction result data structure."""
    transaction_hash: str
    status: TransactionStatus
    gas_used: Optional[int] = None
    gas_price: Optional[Decimal] = None
    block_number: Optional[int] = None
    timestamp: Optional[int] = None
    error_message: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if transaction was successful."""
        return self.status == TransactionStatus.CONFIRMED
    
    @property
    def transaction_fee(self) -> Optional[Decimal]:
        """Calculate transaction fee if gas info available."""
        if self.gas_used is not None and self.gas_price is not None:
            return Decimal(self.gas_used) * self.gas_price
        return None


class WalletError(Exception):
    """Base wallet error."""
    pass


class WalletConnectionError(WalletError):
    """Wallet connection error."""
    pass


class WalletTransactionError(WalletError):
    """Wallet transaction error."""
    pass


class WalletBase(ABC):
    """
    Abstract base class for wallet implementations.
    
    This class defines the interface that all wallet implementations must follow,
    providing common operations for connecting, querying balances, and executing
    transactions across different blockchain networks.
    """
    
    def __init__(self, config: WalletConfig):
        """
        Initialize wallet with configuration.
        
        Args:
            config: Wallet configuration object
        """
        self.config = config
        self._connected = False
        self._wallet_address: Optional[str] = None
        
    @property
    def chain(self) -> Chain:
        """Get the blockchain network this wallet operates on."""
        return self.config.chain
    
    @property
    def network(self) -> NetworkType:
        """Get the network type (mainnet/testnet/devnet)."""
        return self.config.network
    
    @property
    def is_connected(self) -> bool:
        """Check if wallet is connected."""
        return self._connected
    
    @property
    def wallet_address(self) -> Optional[str]:
        """Get the wallet address."""
        return self._wallet_address
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the wallet.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            WalletConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the wallet."""
        pass
    
    @abstractmethod
    async def get_balance(self, token_address: Optional[str] = None) -> WalletBalance:
        """
        Get wallet balance.
        
        Args:
            token_address: Specific token address to query, None for native token
            
        Returns:
            WalletBalance object with balance information
            
        Raises:
            WalletError: If balance query fails
        """
        pass
    
    @abstractmethod
    async def get_native_balance(self) -> Decimal:
        """
        Get native token balance (ETH, SOL, etc.).
        
        Returns:
            Native token balance as Decimal
            
        Raises:
            WalletError: If balance query fails
        """
        pass
    
    @abstractmethod
    async def get_token_balance(self, token_address: str) -> Decimal:
        """
        Get specific token balance.
        
        Args:
            token_address: Token contract address
            
        Returns:
            Token balance as Decimal
            
        Raises:
            WalletError: If token balance query fails
        """
        pass
    
    @abstractmethod
    async def send_native_token(
        self,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send native tokens (ETH, SOL, etc.).
        
        Args:
            to_address: Recipient address
            amount: Amount to send
            gas_price: Optional gas price override
            
        Returns:
            TransactionResult with transaction details
            
        Raises:
            WalletTransactionError: If transaction fails
        """
        pass
    
    @abstractmethod
    async def send_token(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send ERC-20/SPL tokens.
        
        Args:
            token_address: Token contract address
            to_address: Recipient address
            amount: Amount to send
            gas_price: Optional gas price override
            
        Returns:
            TransactionResult with transaction details
            
        Raises:
            WalletTransactionError: If transaction fails
        """
        pass
    
    @abstractmethod
    async def estimate_gas(
        self,
        to_address: str,
        amount: Decimal,
        token_address: Optional[str] = None,
        data: Optional[bytes] = None
    ) -> int:
        """
        Estimate gas required for transaction.
        
        Args:
            to_address: Recipient address
            amount: Transaction amount
            token_address: Token address for token transfers
            data: Optional transaction data
            
        Returns:
            Estimated gas units required
            
        Raises:
            WalletError: If gas estimation fails
        """
        pass
    
    @abstractmethod
    async def get_gas_price(self) -> Decimal:
        """
        Get current gas price.
        
        Returns:
            Current gas price in appropriate units (gwei for ETH, lamports for SOL)
            
        Raises:
            WalletError: If gas price query fails
        """
        pass
    
    @abstractmethod
    async def get_transaction_status(self, transaction_hash: str) -> TransactionResult:
        """
        Get transaction status and details.
        
        Args:
            transaction_hash: Transaction hash to query
            
        Returns:
            TransactionResult with current status and details
            
        Raises:
            WalletError: If transaction query fails
        """
        pass
    
    @abstractmethod
    async def sign_message(self, message: str) -> str:
        """
        Sign a message with the wallet's private key.
        
        Args:
            message: Message to sign
            
        Returns:
            Signed message as hex string
            
        Raises:
            WalletError: If signing fails
        """
        pass
    
    @abstractmethod
    async def validate_address(self, address: str) -> bool:
        """
        Validate if an address is valid for this chain.
        
        Args:
            address: Address to validate
            
        Returns:
            True if address is valid, False otherwise
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform wallet health check.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            balance = await self.get_native_balance()
            gas_price = await self.get_gas_price()
            
            return {
                "connected": self.is_connected,
                "wallet_address": self.wallet_address,
                "native_balance": str(balance),
                "gas_price": str(gas_price),
                "chain": self.chain.value,
                "network": self.network.value,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Wallet health check failed: {e}")
            return {
                "connected": self.is_connected,
                "status": "unhealthy",
                "error": str(e)
            }
    
    def __str__(self) -> str:
        """String representation of wallet."""
        return f"{self.__class__.__name__}({self.chain.value}, {self.wallet_address})"
    
    def __repr__(self) -> str:
        """Detailed string representation of wallet."""
        return (
            f"{self.__class__.__name__}("
            f"chain={self.chain.value}, "
            f"network={self.network.value}, "
            f"address={self.wallet_address}, "
            f"connected={self.is_connected})"
        )