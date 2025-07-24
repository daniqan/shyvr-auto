"""
Base DEX interface and data structures for multi-chain DEX operations.

This module defines the abstract base class for DEX implementations and
common data structures used across different decentralized exchanges.
It follows the established patterns from wallet/base.py while providing
DEX-specific functionality for price quotes, swaps, and liquidity analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import structlog

from src.utils.base import Chain


logger = structlog.get_logger()


class SwapType(Enum):
    """Type of swap operation."""
    EXACT_INPUT = "exact_input"      # Exact amount in, variable amount out
    EXACT_OUTPUT = "exact_output"    # Variable amount in, exact amount out


class SwapStatus(Enum):
    """Swap transaction status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PriceImpact(Enum):
    """Price impact severity levels."""
    LOW = "low"          # < 1%
    MEDIUM = "medium"    # 1-3%
    HIGH = "high"        # 3-10%
    EXTREME = "extreme"  # > 10%


@dataclass
class DEXConfig:
    """DEX configuration data structure."""
    chain: Chain
    name: str                           # DEX name (jupiter, hyperliquid, uniswap_v3)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    wallet_address: Optional[str] = None
    max_slippage_bps: int = 50          # 0.5% default max slippage
    timeout_seconds: int = 30
    rate_limit_per_second: int = 10     # Conservative default
    enable_price_impact_warnings: bool = True
    max_price_impact_bps: int = 1000    # 10% max price impact
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_slippage_bps < 0 or self.max_slippage_bps > 10000:  # 0-100%
            raise ValueError("max_slippage_bps must be between 0 and 10000")
        if self.max_price_impact_bps < 0 or self.max_price_impact_bps > 10000:
            raise ValueError("max_price_impact_bps must be between 0 and 10000")


@dataclass
class SwapQuote:
    """Swap quote data structure with comprehensive pricing information."""
    input_token: str                    # Token address
    output_token: str                   # Token address
    input_amount: Decimal               # Amount in smallest units
    output_amount: Decimal              # Expected output amount
    price: Decimal                      # Price per unit (output/input)
    price_impact_bps: int              # Price impact in basis points
    slippage_bps: int                  # Slippage tolerance in basis points
    estimated_gas: Optional[int] = None
    gas_price: Optional[Decimal] = None
    route: Optional[List[str]] = None   # List of intermediate tokens
    valid_until: Optional[datetime] = None
    dex_name: str = ""
    quote_id: Optional[str] = None      # DEX-specific quote identifier
    additional_fees: Optional[Dict[str, Decimal]] = None  # Platform fees, etc.
    
    @property
    def price_impact(self) -> PriceImpact:
        """Get price impact severity level."""
        if self.price_impact_bps < 100:  # < 1%
            return PriceImpact.LOW
        elif self.price_impact_bps < 300:  # 1-3%
            return PriceImpact.MEDIUM
        elif self.price_impact_bps < 1000:  # 3-10%
            return PriceImpact.HIGH
        else:  # > 10%
            return PriceImpact.EXTREME
    
    @property
    def estimated_fee(self) -> Optional[Decimal]:
        """Calculate estimated transaction fee."""
        if self.estimated_gas and self.gas_price:
            return Decimal(self.estimated_gas) * self.gas_price
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if quote has expired."""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until
    
    @property
    def effective_price(self) -> Decimal:
        """Get effective price including slippage."""
        slippage_multiplier = Decimal(1) - (Decimal(self.slippage_bps) / Decimal(10000))
        return self.price * slippage_multiplier


@dataclass
class SwapResult:
    """Swap execution result data structure."""
    transaction_hash: str
    status: SwapStatus
    input_token: str
    output_token: str
    input_amount: Decimal
    actual_output_amount: Optional[Decimal] = None
    gas_used: Optional[int] = None
    gas_price: Optional[Decimal] = None
    block_number: Optional[int] = None
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    dex_name: str = ""
    quote_used: Optional[SwapQuote] = None
    actual_price_impact_bps: Optional[int] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if swap was successful."""
        return self.status == SwapStatus.CONFIRMED
    
    @property
    def transaction_fee(self) -> Optional[Decimal]:
        """Calculate transaction fee if gas info available."""
        if self.gas_used is not None and self.gas_price is not None:
            return Decimal(self.gas_used) * self.gas_price
        return None
    
    @property
    def actual_price(self) -> Optional[Decimal]:
        """Calculate actual price achieved (output/input)."""
        if self.actual_output_amount and self.input_amount > 0:
            return self.actual_output_amount / self.input_amount
        return None
    
    @property
    def slippage_vs_quote(self) -> Optional[Decimal]:
        """Calculate slippage vs original quote."""
        if not self.quote_used or not self.actual_output_amount:
            return None
        expected = self.quote_used.output_amount
        actual = self.actual_output_amount
        if expected > 0:
            return (expected - actual) / expected
        return None


class DEXError(Exception):
    """Base DEX error."""
    pass


class DEXConnectionError(DEXError):
    """DEX connection error."""
    pass


class DEXTransactionError(DEXError):
    """DEX transaction error."""
    pass


class DEXRateLimitError(DEXError):
    """DEX rate limit error."""
    pass


class DEXBase(ABC):
    """
    Abstract base class for DEX implementations.
    
    This class defines the interface that all DEX implementations must follow,
    providing common operations for getting quotes, executing swaps, and
    analyzing liquidity across different decentralized exchanges.
    """
    
    def __init__(self, config: DEXConfig):
        """
        Initialize DEX with configuration.
        
        Args:
            config: DEX configuration object
        """
        self.config = config
        self._connected = False
        self.logger = logger.bind(dex=config.name, chain=config.chain.value)
        
    @property
    def chain(self) -> Chain:
        """Get the blockchain network this DEX operates on."""
        return self.config.chain
    
    @property
    def name(self) -> str:
        """Get the DEX name."""
        return self.config.name
    
    @property
    def is_connected(self) -> bool:
        """Check if DEX is connected."""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the DEX.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            DEXConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the DEX."""
        pass
    
    @abstractmethod
    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        swap_type: SwapType = SwapType.EXACT_INPUT,
        slippage_bps: Optional[int] = None
    ) -> SwapQuote:
        """
        Get a quote for a token swap.
        
        Args:
            input_token: Input token address
            output_token: Output token address  
            amount: Amount to swap (in smallest units)
            swap_type: Type of swap (exact input/output)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            SwapQuote with pricing and routing information
            
        Raises:
            DEXError: If quote request fails
        """
        pass
    
    @abstractmethod
    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_address: str,
        signature_data: Optional[Dict[str, Any]] = None
    ) -> SwapResult:
        """
        Execute a token swap using a quote.
        
        Args:
            quote: SwapQuote from get_quote()
            wallet_address: Wallet address to execute swap from
            signature_data: Additional signature data required by some DEXs
            
        Returns:
            SwapResult with transaction details
            
        Raises:
            DEXTransactionError: If swap execution fails
        """
        pass
    
    @abstractmethod
    async def get_token_price(
        self,
        token_address: str,
        base_token: Optional[str] = None
    ) -> Decimal:
        """
        Get current price of a token.
        
        Args:
            token_address: Token address to get price for
            base_token: Base token to price against (USDC/USDT default)
            
        Returns:
            Token price as Decimal
            
        Raises:
            DEXError: If price query fails
        """
        pass
    
    @abstractmethod
    async def get_supported_tokens(self) -> List[Dict[str, Any]]:
        """
        Get list of tokens supported by this DEX.
        
        Returns:
            List of token information dictionaries
            
        Raises:
            DEXError: If token list query fails
        """
        pass
    
    @abstractmethod
    async def estimate_gas(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        wallet_address: str
    ) -> int:
        """
        Estimate gas required for a swap.
        
        Args:
            input_token: Input token address
            output_token: Output token address
            amount: Amount to swap
            wallet_address: Wallet address executing swap
            
        Returns:
            Estimated gas units required
            
        Raises:
            DEXError: If gas estimation fails
        """
        pass
    
    @abstractmethod
    async def get_transaction_status(self, transaction_hash: str) -> SwapResult:
        """
        Get transaction status and details.
        
        Args:
            transaction_hash: Transaction hash to query
            
        Returns:
            SwapResult with current status and details
            
        Raises:
            DEXError: If transaction query fails
        """
        pass
    
    async def validate_quote(self, quote: SwapQuote) -> bool:
        """
        Validate a quote before execution.
        
        Args:
            quote: SwapQuote to validate
            
        Returns:
            True if quote is valid, False otherwise
        """
        try:
            # Check if quote has expired
            if quote.is_expired:
                self.logger.warning("Quote has expired", quote_id=quote.quote_id)
                return False
            
            # Check price impact
            if quote.price_impact_bps > self.config.max_price_impact_bps:
                self.logger.warning(
                    "Price impact exceeds maximum",
                    price_impact_bps=quote.price_impact_bps,
                    max_allowed=self.config.max_price_impact_bps
                )
                return False
            
            # Check slippage
            if quote.slippage_bps > self.config.max_slippage_bps:
                self.logger.warning(
                    "Slippage exceeds maximum",
                    slippage_bps=quote.slippage_bps,
                    max_allowed=self.config.max_slippage_bps
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Quote validation failed", error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform DEX health check.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            # Try to get supported tokens as a simple connectivity test
            tokens = await self.get_supported_tokens()
            token_count = len(tokens) if tokens else 0
            
            return {
                "connected": self.is_connected,
                "dex_name": self.name,
                "chain": self.chain.value,
                "supported_tokens": token_count,
                "max_slippage_bps": self.config.max_slippage_bps,
                "rate_limit": self.config.rate_limit_per_second,
                "status": "healthy"
            }
        except Exception as e:
            self.logger.error("DEX health check failed", error=str(e))
            return {
                "connected": self.is_connected,
                "dex_name": self.name,
                "chain": self.chain.value,
                "status": "unhealthy",
                "error": str(e)
            }
    
    def __str__(self) -> str:
        """String representation of DEX."""
        return f"{self.__class__.__name__}({self.name}, {self.chain.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of DEX."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"chain={self.chain.value}, "
            f"connected={self.is_connected})"
        )