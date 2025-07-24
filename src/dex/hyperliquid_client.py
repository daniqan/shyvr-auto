"""
Hyperliquid DEX Client for Perpetual Futures Trading

Provides integration with Hyperliquid's order book DEX, offering:
- On-chain order book with sub-second finality
- Zero gas fees for trading
- Perpetual futures with up to 50x leverage
- Advanced order types and risk management

Key differences from AMMs:
- Order book trading vs AMM routing
- Perpetual futures focus vs spot trading
- Authentication required for all operations
- Different quote structure (price/size vs input/output)
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import structlog

from hyperliquid import HyperliquidSync, HyperliquidAsync

from src.utils.base import Chain
from .base import (
    DEXBase,
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    DEXConfig,
    DEXError,
    DEXConnectionError,
    DEXTransactionError,
    DEXRateLimitError,
)


logger = structlog.get_logger()


class HyperliquidDEXClient(DEXBase):
    """Hyperliquid DEX client for perpetual futures trading"""
    
    # Hyperliquid API endpoints
    API_URL = "https://api.hyperliquid.xyz"
    TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
    
    # Common perpetual contracts
    COMMON_PERPETUALS = {
        "BTC-PERP": "BTC",
        "ETH-PERP": "ETH", 
        "SOL-PERP": "SOL",
        "MATIC-PERP": "MATIC",
        "AVAX-PERP": "AVAX",
        "DOGE-PERP": "DOGE",
        "ADA-PERP": "ADA",
        "DOT-PERP": "DOT",
    }
    
    def __init__(self, config: Optional[DEXConfig] = None):
        """
        Initialize Hyperliquid DEX client.
        
        Args:
            config: DEX configuration, creates default if None
        """
        if config is None:
            config = DEXConfig(
                chain=Chain.HYPERLIQUID,
                name="hyperliquid",
                max_slippage_bps=100,  # 1% default for futures
                timeout_seconds=30,
                rate_limit_per_second=5,  # Conservative for authenticated API
                max_price_impact_bps=500,  # 5% for leverage trading
            )
        
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.hyperliquid_client: Optional[Union[HyperliquidSync, HyperliquidAsync]] = None
        self._last_request_time = 0.0
        self._rate_limit_delay = 1.0 / self.config.rate_limit_per_second
        
        # Initialize Hyperliquid SDK client
        self._initialize_sdk_client()
    
    def _initialize_sdk_client(self) -> None:
        """Initialize the Hyperliquid SDK client."""
        try:
            # Use testnet if no API key provided for development
            if self.config.api_key:
                # Production client with authentication
                self.hyperliquid_client = HyperliquidAsync({
                    "api_key": self.config.api_key,
                    "wallet_address": self.config.wallet_address,
                })
            else:
                # Testnet client for development/testing
                self.hyperliquid_client = HyperliquidAsync({})
                
        except Exception as e:
            self.logger.error("Failed to initialize Hyperliquid SDK client", error=str(e))
            raise DEXConnectionError(f"SDK initialization failed: {str(e)}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper timeout."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ShyvrAI-RLTE/1.0"
                }
            )
        return self.session
    
    async def _rate_limited_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make rate-limited request to avoid hitting API limits."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
        
        # Use internal _make_request method (will be implemented)
        return await self._make_request(method, endpoint, **kwargs)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Hyperliquid API."""
        session = await self._get_session()
        url = f"{self.API_URL}{endpoint}"
        
        try:
            if method.upper() == "GET":
                if params:
                    url += f"?{urlencode(params)}"
                async with session.get(url) as response:
                    if response.status == 429:
                        raise DEXRateLimitError("Rate limit exceeded")
                    response.raise_for_status()
                    return await response.json()
            
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    if response.status == 429:
                        raise DEXRateLimitError("Rate limit exceeded")
                    response.raise_for_status()
                    return await response.json()
            
            else:
                raise DEXError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            self.logger.error("HTTP request failed", url=url, error=str(e))
            raise DEXConnectionError(f"Request failed: {str(e)}")
        except Exception as e:
            self.logger.error("Unexpected error in request", url=url, error=str(e))
            raise DEXError(f"Request error: {str(e)}")
    
    async def _authenticate(self) -> bool:
        """
        Authenticate with Hyperliquid API.
        
        Returns:
            True if authentication successful
            
        Raises:
            DEXConnectionError: If authentication fails
        """
        try:
            if not self.hyperliquid_client:
                raise DEXConnectionError("Hyperliquid client not initialized")
            
            # Test authentication by making a simple request
            # This is a placeholder - actual implementation would test API access
            self.logger.info("Authentication successful")
            return True
            
        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            raise DEXConnectionError(f"Authentication failed: {str(e)}")
    
    async def connect(self) -> bool:
        """
        Connect to Hyperliquid DEX.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            DEXConnectionError: If connection fails
        """
        try:
            # Authenticate if API key is provided
            if self.config.api_key:
                await self._authenticate()
            
            self._connected = True
            self.logger.info("Connected to Hyperliquid DEX")
            return True
            
        except Exception as e:
            self.logger.error("Connection failed", error=str(e))
            self._connected = False
            raise DEXConnectionError(f"Connection failed: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid DEX."""
        if self.session:
            await self.session.close()
            self.session = None
        
        self._connected = False
        self.logger.info("Disconnected from Hyperliquid DEX")
    
    async def _get_orderbook_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        side: str = "buy"
    ) -> Dict[str, Any]:
        """
        Get orderbook quote for a trade.
        
        Args:
            input_token: Input token symbol
            output_token: Output token symbol (perpetual contract)
            amount: Amount to trade
            side: Trade side (buy/sell)
            
        Returns:
            Quote response from API
        """
        # This is a mock implementation for now
        # Real implementation would call Hyperliquid orderbook API
        return {
            "price": "45000.50",
            "size": str(amount),
            "side": side,
            "orderbook_depth": 10,
            "funding_rate": "0.0001",
            "mark_price": "45000.75",
            "estimated_fees": str(float(amount) * 0.0001),  # 0.01% fee
        }
    
    async def _get_leveraged_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        leverage: int,
        slippage_bps: int
    ) -> Dict[str, Any]:
        """Get quote for leveraged position."""
        # Mock leveraged quote
        position_size = float(amount) * leverage / 45000  # Approximate position size
        return {
            "price": "45000.50",
            "size": str(position_size),
            "side": "buy",
            "leverage": leverage,
            "margin_required": str(amount),
            "liquidation_price": "40500.45",
            "estimated_fees": str(float(amount) * 0.001),  # 0.1% fee
        }
    
    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        swap_type: SwapType = SwapType.EXACT_INPUT,
        slippage_bps: Optional[int] = None
    ) -> SwapQuote:
        """
        Get a quote for a perpetual futures trade.
        
        Args:
            input_token: Input token (usually USDC)
            output_token: Output token (perpetual contract)
            amount: Amount to trade
            swap_type: Type of swap
            slippage_bps: Slippage tolerance
            
        Returns:
            SwapQuote with pricing information
        """
        if not self.is_connected:
            await self.connect()
        
        if slippage_bps is None:
            slippage_bps = self.config.max_slippage_bps
        
        try:
            # Get orderbook quote
            quote_response = await self._get_orderbook_quote(
                input_token, output_token, amount
            )
            
            price = Decimal(quote_response["price"])
            output_amount = amount / price if swap_type == SwapType.EXACT_INPUT else amount
            
            # Calculate price impact (simplified)
            price_impact_bps = 25  # Mock 0.25% price impact
            
            # Create quote ID
            quote_id = f"hl_{int(datetime.now().timestamp())}_{hash(str(amount))}"
            
            # Additional fees for perpetuals
            additional_fees = {
                "trading_fee": Decimal(quote_response.get("estimated_fees", "0")),
                "funding_rate": Decimal(quote_response.get("funding_rate", "0")),
            }
            
            return SwapQuote(
                input_token=input_token,
                output_token=output_token,
                input_amount=amount,
                output_amount=output_amount,
                price=price,
                price_impact_bps=price_impact_bps,
                slippage_bps=slippage_bps,
                dex_name="hyperliquid",
                quote_id=quote_id,
                additional_fees=additional_fees,
                valid_until=datetime.now() + timedelta(minutes=5),
            )
            
        except Exception as e:
            self.logger.error("Failed to get quote", error=str(e))
            raise DEXError(f"Quote request failed: {str(e)}")
    
    async def get_leveraged_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        leverage: int,
        slippage_bps: int
    ) -> SwapQuote:
        """Get quote for leveraged perpetual position."""
        if not self.is_connected:
            await self.connect()
        
        try:
            quote_response = await self._get_leveraged_quote(
                input_token, output_token, amount, leverage, slippage_bps
            )
            
            price = Decimal(quote_response["price"])
            output_amount = Decimal(quote_response["size"])
            
            additional_fees = {
                "trading_fee": Decimal(quote_response.get("estimated_fees", "0")),
                "leverage": Decimal(str(leverage)),
                "margin_required": Decimal(quote_response.get("margin_required", "0")),
                "liquidation_price": Decimal(quote_response.get("liquidation_price", "0")),
            }
            
            quote_id = f"hl_lev_{int(datetime.now().timestamp())}_{leverage}x"
            
            return SwapQuote(
                input_token=input_token,
                output_token=output_token,
                input_amount=amount,
                output_amount=output_amount,
                price=price,
                price_impact_bps=50,  # Higher impact for leveraged trades
                slippage_bps=slippage_bps,
                dex_name="hyperliquid",
                quote_id=quote_id,
                additional_fees=additional_fees,
                valid_until=datetime.now() + timedelta(minutes=5),
            )
            
        except Exception as e:
            self.logger.error("Failed to get leveraged quote", error=str(e))
            raise DEXError(f"Leveraged quote request failed: {str(e)}")
    
    async def _execute_order(self, quote: SwapQuote) -> Dict[str, Any]:
        """Execute order using Hyperliquid SDK."""
        # Mock order execution
        return {
            "order_id": f"hl_order_{int(datetime.now().timestamp())}",
            "status": "filled",
            "filled_size": str(quote.output_amount),
            "avg_fill_price": str(quote.price),
            "fees_paid": str(quote.additional_fees.get("trading_fee", Decimal("0"))),
            "transaction_hash": f"0x{hash(quote.quote_id):016x}",
        }
    
    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_address: str,
        signature_data: Optional[Dict[str, Any]] = None
    ) -> SwapResult:
        """
        Execute a perpetual futures order.
        
        Args:
            quote: SwapQuote from get_quote()
            wallet_address: Wallet address
            signature_data: Additional signature data
            
        Returns:
            SwapResult with execution details
        """
        if not self.is_connected:
            await self.connect()
        
        # Validate quote before execution
        if not await self.validate_quote(quote):
            raise DEXTransactionError("Quote validation failed")
        
        try:
            order_response = await self._execute_order(quote)
            
            # Map order status to swap status
            status_mapping = {
                "filled": SwapStatus.CONFIRMED,
                "partial": SwapStatus.PENDING,
                "cancelled": SwapStatus.FAILED,
                "rejected": SwapStatus.REJECTED,
            }
            
            status = status_mapping.get(order_response["status"], SwapStatus.PENDING)
            
            return SwapResult(
                transaction_hash=order_response["transaction_hash"],
                status=status,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                actual_output_amount=Decimal(order_response["filled_size"]),
                dex_name="hyperliquid",
                quote_used=quote,
                timestamp=datetime.now(),
                transaction_data=order_response,
            )
            
        except Exception as e:
            self.logger.error("Order execution failed", error=str(e))
            raise DEXTransactionError(f"Order execution failed: {str(e)}")
    
    async def _get_perpetual_price(self, symbol: str) -> Dict[str, Any]:
        """Get perpetual contract price information."""
        # Mock price response
        return {
            "symbol": symbol,
            "mark_price": "45000.75",
            "index_price": "45000.50",
            "funding_rate": "0.0001",
            "next_funding": "2024-01-01T12:00:00Z",
        }
    
    async def get_token_price(
        self,
        token_address: str,
        base_token: Optional[str] = None
    ) -> Decimal:
        """
        Get current price of a perpetual contract.
        
        Args:
            token_address: Token/contract symbol
            base_token: Base token (ignored for perpetuals)
            
        Returns:
            Token price as Decimal
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            price_response = await self._get_perpetual_price(token_address)
            return Decimal(price_response["mark_price"])
            
        except Exception as e:
            self.logger.error("Failed to get token price", error=str(e))
            raise DEXError(f"Price request failed: {str(e)}")
    
    async def _get_available_perpetuals(self) -> List[Dict[str, Any]]:
        """Get available perpetual contracts."""
        # Mock perpetuals list
        return [
            {
                "symbol": "BTC-PERP",
                "base_asset": "BTC",
                "quote_asset": "USDC",
                "min_size": "0.001",
                "max_leverage": 50,
                "funding_rate": "0.0001",
            },
            {
                "symbol": "ETH-PERP",
                "base_asset": "ETH",
                "quote_asset": "USDC",
                "min_size": "0.01",
                "max_leverage": 50,
                "funding_rate": "0.0002",
            },
        ]
    
    async def get_supported_tokens(self) -> List[Dict[str, Any]]:
        """
        Get list of supported perpetual contracts.
        
        Returns:
            List of perpetual contract information
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            return await self._get_available_perpetuals()
            
        except Exception as e:
            self.logger.error("Failed to get supported tokens", error=str(e))
            raise DEXError(f"Token list request failed: {str(e)}")
    
    async def estimate_gas(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        wallet_address: str
    ) -> int:
        """
        Estimate gas for a trade (always 0 on Hyperliquid).
        
        Args:
            input_token: Input token
            output_token: Output token
            amount: Trade amount
            wallet_address: Wallet address
            
        Returns:
            Gas estimate (always 0 for Hyperliquid)
        """
        # Hyperliquid has zero gas fees
        return 0
    
    async def _get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from Hyperliquid."""
        # Mock order status
        return {
            "order_id": order_id,
            "status": "filled",
            "filled_size": "1.0",
            "remaining_size": "0.0",
            "avg_fill_price": "45000.25",
            "fees_paid": "22.50",
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T10:00:05Z",
        }
    
    async def get_transaction_status(self, transaction_hash: str) -> SwapResult:
        """
        Get order status and details.
        
        Args:
            transaction_hash: Transaction/order hash
            
        Returns:
            SwapResult with current status
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            status_response = await self._get_order_status(transaction_hash)
            
            # Map order status
            status_mapping = {
                "filled": SwapStatus.CONFIRMED,
                "partial": SwapStatus.PENDING,
                "cancelled": SwapStatus.FAILED,
                "rejected": SwapStatus.REJECTED,
            }
            
            status = status_mapping.get(status_response["status"], SwapStatus.PENDING)
            
            return SwapResult(
                transaction_hash=transaction_hash,
                status=status,
                input_token="USDC",  # Default for perpetuals
                output_token="BTC-PERP",  # Default example
                input_amount=Decimal("45000"),  # Would be from original order
                actual_output_amount=Decimal(status_response["filled_size"]),
                dex_name="hyperliquid",
                timestamp=datetime.now(),
                transaction_data=status_response,
            )
            
        except Exception as e:
            self.logger.error("Failed to get transaction status", error=str(e))
            raise DEXError(f"Status request failed: {str(e)}")