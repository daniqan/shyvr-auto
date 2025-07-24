"""
Jupiter DEX Client for Solana Token Swaps

Provides integration with Jupiter's V6 Swap API, offering:
- Best routing across Solana DEXs
- Zero protocol fees
- Comprehensive price impact analysis
- Production-ready async implementation

Jupiter is the optimal first choice for Solana trading due to its:
- Clean API design and excellent documentation
- Best routing algorithm for optimal prices
- Zero protocol fees maximizing profits
- Established reliability and uptime
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import structlog

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


class JupiterDEXClient(DEXBase):
    """Jupiter DEX client for Solana token swaps"""
    
    # Jupiter V6 API endpoints
    QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    SWAP_URL = "https://quote-api.jup.ag/v6/swap"
    SWAP_INSTRUCTIONS_URL = "https://quote-api.jup.ag/v6/swap-instructions"
    TOKENS_URL = "https://tokens.jup.ag/tokens"
    PRICE_URL = "https://api.jup.ag/price/v2"
    
    # Common Solana token addresses
    COMMON_TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    }
    
    def __init__(self, config: Optional[DEXConfig] = None):
        """
        Initialize Jupiter DEX client.
        
        Args:
            config: DEX configuration, creates default if None
        """
        if config is None:
            config = DEXConfig(
                chain=Chain.SOLANA,
                name="jupiter",
                max_slippage_bps=50,  # 0.5% default
                timeout_seconds=30,
                rate_limit_per_second=10
            )
        
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: datetime = datetime.min
        self._cache_duration = timedelta(minutes=30)  # Cache tokens for 30 minutes
        
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
    
    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Jupiter API with error handling.
        
        Args:
            method: HTTP method (GET, POST)
            url: Request URL
            params: URL parameters
            data: Request body data
            
        Returns:
            Response data as dictionary
            
        Raises:
            DEXRateLimitError: If rate limited
            DEXConnectionError: If connection fails
            DEXError: For other API errors
        """
        session = await self._get_session()
        
        try:
            async with session.request(method, url, params=params, json=data) as response:
                if response.status == 429:
                    self.logger.warning("Jupiter API rate limit exceeded")
                    raise DEXRateLimitError("Jupiter API rate limit exceeded")
                elif response.status == 400:
                    error_text = await response.text()
                    self.logger.error("Jupiter API bad request", error=error_text)
                    raise DEXError(f"Jupiter API bad request: {error_text}")
                elif response.status != 200:
                    error_text = await response.text()
                    self.logger.error("Jupiter API error", status=response.status, error=error_text)
                    raise DEXConnectionError(f"Jupiter API error {response.status}: {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            self.logger.error("Jupiter API request failed", error=str(e))
            raise DEXConnectionError(f"Jupiter API request failed: {str(e)}")
    
    async def connect(self) -> bool:
        """
        Connect to Jupiter API (test connectivity).
        
        Returns:
            True if connection successful
            
        Raises:
            DEXConnectionError: If connection fails
        """
        try:
            # Test connectivity by fetching tokens
            await self.get_supported_tokens()
            self._connected = True
            self.logger.info("Connected to Jupiter DEX")
            return True
        except Exception as e:
            self.logger.error("Failed to connect to Jupiter DEX", error=str(e))
            raise DEXConnectionError(f"Failed to connect to Jupiter: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Jupiter API (close session)."""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        self.logger.info("Disconnected from Jupiter DEX")
    
    def _resolve_token_address(self, token: str) -> str:
        """
        Resolve token symbol to address or return as-is if already an address.
        
        Args:
            token: Token symbol or address
            
        Returns:
            Token address
        """
        # Check if it's a known symbol
        if token.upper() in self.COMMON_TOKENS:
            return self.COMMON_TOKENS[token.upper()]
        
        # Assume it's already an address if it looks like a Solana address
        if len(token) >= 32 and all(c.isalnum() for c in token):
            return token
        
        # If not found, return as-is and let Jupiter handle the error
        return token
    
    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        swap_type: SwapType = SwapType.EXACT_INPUT,
        slippage_bps: Optional[int] = None
    ) -> SwapQuote:
        """
        Get a swap quote from Jupiter.
        
        Args:
            input_token: Input token symbol or address
            output_token: Output token symbol or address
            amount: Amount to swap (in smallest units for exact input)
            swap_type: Type of swap (exact input/output)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            SwapQuote with pricing information
            
        Raises:
            DEXError: If quote request fails
        """
        try:
            # Resolve token addresses
            input_addr = self._resolve_token_address(input_token)
            output_addr = self._resolve_token_address(output_token)
            
            # Use config default if slippage not specified
            if slippage_bps is None:
                slippage_bps = self.config.max_slippage_bps
            
            # Prepare quote parameters
            params = {
                "inputMint": input_addr,
                "outputMint": output_addr,
                "amount": str(int(amount)),
                "slippageBps": slippage_bps,
                "swapMode": "ExactIn" if swap_type == SwapType.EXACT_INPUT else "ExactOut"
            }
            
            # Make quote request
            response = await self._make_request("GET", self.QUOTE_URL, params=params)
            
            # Parse response
            input_amount = Decimal(response["inAmount"])
            output_amount = Decimal(response["outAmount"])
            price_impact_pct = response.get("priceImpactPct", "0")
            price_impact_bps = int(float(price_impact_pct) * 10000)  # Convert % to bps
            
            # Calculate price (output per input)
            price = output_amount / input_amount if input_amount > 0 else Decimal(0)
            
            # Extract route information
            route_plan = response.get("routePlan", [])
            route = []
            for step in route_plan:
                swap_info = step.get("swapInfo", {})
                if "outputMint" in swap_info:
                    route.append(swap_info["outputMint"])
            
            # Create quote with 60-second validity
            valid_until = datetime.now() + timedelta(seconds=60)
            
            quote = SwapQuote(
                input_token=input_addr,
                output_token=output_addr,
                input_amount=input_amount,
                output_amount=output_amount,
                price=price,
                price_impact_bps=price_impact_bps,
                slippage_bps=slippage_bps,
                route=route,
                valid_until=valid_until,
                dex_name="jupiter",
                quote_id=response.get("quoteResponse", {}).get("timeTaken", str(datetime.now().timestamp())),
                additional_fees={
                    "platformFee": Decimal(response.get("platformFee", {}).get("amount", "0"))
                }
            )
            
            self.logger.info(
                "Generated Jupiter quote",
                input_token=input_token,
                output_token=output_token,
                input_amount=str(input_amount),
                output_amount=str(output_amount),
                price_impact_bps=price_impact_bps
            )
            
            return quote
            
        except Exception as e:
            self.logger.error("Failed to get Jupiter quote", error=str(e))
            raise DEXError(f"Failed to get Jupiter quote: {str(e)}")
    
    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_address: str,
        signature_data: Optional[Dict[str, Any]] = None
    ) -> SwapResult:
        """
        Execute a token swap using Jupiter.
        
        Note: This method prepares the swap transaction but requires
        external signing and submission via the Solana wallet integration.
        
        Args:
            quote: SwapQuote from get_quote()
            wallet_address: Wallet address to execute swap from
            signature_data: Additional data (not used for Jupiter)
            
        Returns:
            SwapResult with transaction preparation details
            
        Raises:
            DEXTransactionError: If swap preparation fails
        """
        try:
            # Validate quote
            if not await self.validate_quote(quote):
                raise DEXTransactionError("Invalid or expired quote")
            
            # Prepare swap request
            swap_request = {
                "quoteResponse": {
                    "inputMint": quote.input_token,
                    "outputMint": quote.output_token,
                    "inAmount": str(int(quote.input_amount)),
                    "outAmount": str(int(quote.output_amount)),
                    "otherAmountThreshold": str(int(quote.output_amount * 
                                                  (Decimal('1') - Decimal(quote.slippage_bps) / Decimal('10000')))),
                    "swapMode": "ExactIn",
                    "slippageBps": quote.slippage_bps,
                    "priceImpactPct": str(Decimal(quote.price_impact_bps) / Decimal('10000'))
                },
                "userPublicKey": wallet_address,
                "wrapAndUnwrapSol": True,
                "useSharedAccounts": True,
                "feeAccount": None,
                "trackingAccount": None,
                "computeUnitPriceMicroLamports": None
            }
            
            # Get swap transaction
            response = await self._make_request("POST", self.SWAP_URL, data=swap_request)
            
            # Return result indicating transaction is prepared
            # The actual submission will be handled by the wallet module
            result = SwapResult(
                transaction_hash="pending",  # Will be updated after submission
                status=SwapStatus.PENDING,
                input_token=quote.input_token,
                output_token=quote.output_token,
                input_amount=quote.input_amount,
                dex_name="jupiter",
                quote_used=quote
            )
            
            # Store transaction data for wallet module
            result.transaction_data = {
                "swapTransaction": response.get("swapTransaction"),
                "lastValidBlockHeight": response.get("lastValidBlockHeight")
            }
            
            self.logger.info(
                "Prepared Jupiter swap transaction",
                input_token=quote.input_token,
                output_token=quote.output_token,
                wallet=wallet_address
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to execute Jupiter swap", error=str(e))
            raise DEXTransactionError(f"Failed to execute Jupiter swap: {str(e)}")
    
    async def get_token_price(
        self,
        token_address: str,
        base_token: Optional[str] = None
    ) -> Decimal:
        """
        Get current price of a token via Jupiter Price API.
        
        Args:
            token_address: Token address to get price for
            base_token: Base token (defaults to USDC)
            
        Returns:
            Token price as Decimal
            
        Raises:
            DEXError: If price query fails
        """
        try:
            # Resolve token address
            token_addr = self._resolve_token_address(token_address)
            
            # Use USDC as default base
            if base_token is None:
                base_token = "USDC"
            base_addr = self._resolve_token_address(base_token)
            
            # Build price request URL
            params = {
                "ids": token_addr,
                "vsToken": base_addr
            }
            
            # Make price request
            response = await self._make_request("GET", self.PRICE_URL, params=params)
            
            # Extract price
            token_data = response.get("data", {}).get(token_addr)
            if not token_data:
                raise DEXError(f"Price data not found for token {token_address}")
            
            price = Decimal(str(token_data.get("price", "0")))
            
            self.logger.debug(
                "Retrieved token price",
                token=token_address,
                base=base_token,
                price=str(price)
            )
            
            return price
            
        except Exception as e:
            self.logger.error("Failed to get token price", token=token_address, error=str(e))
            raise DEXError(f"Failed to get token price: {str(e)}")
    
    async def get_supported_tokens(self) -> List[Dict[str, Any]]:
        """
        Get list of tokens supported by Jupiter.
        
        Returns:
            List of token information dictionaries
            
        Raises:
            DEXError: If token list query fails
        """
        try:
            # Check cache first
            if datetime.now() < self._cache_expiry and self._token_cache:
                return list(self._token_cache.values())
            
            # Fetch tokens from Jupiter
            response = await self._make_request("GET", self.TOKENS_URL)
            
            # Update cache
            self._token_cache = {}
            for token in response:
                self._token_cache[token["address"]] = token
            
            self._cache_expiry = datetime.now() + self._cache_duration
            
            self.logger.info("Retrieved supported tokens", count=len(self._token_cache))
            return list(self._token_cache.values())
            
        except Exception as e:
            self.logger.error("Failed to get supported tokens", error=str(e))
            raise DEXError(f"Failed to get supported tokens: {str(e)}")
    
    async def estimate_gas(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        wallet_address: str
    ) -> int:
        """
        Estimate compute units for a Jupiter swap.
        
        Jupiter swaps typically use 100,000-200,000 compute units.
        This is an estimation since Solana uses compute units, not gas.
        
        Args:
            input_token: Input token address
            output_token: Output token address
            amount: Amount to swap
            wallet_address: Wallet address
            
        Returns:
            Estimated compute units
        """
        try:
            # Get a quote first to analyze route complexity
            quote = await self.get_quote(input_token, output_token, amount)
            
            # Base compute units
            base_units = 100000
            
            # Add units based on route complexity
            if quote.route:
                # Add 20k compute units per hop
                route_complexity = len(quote.route) * 20000
                base_units += route_complexity
            
            # Cap at reasonable maximum
            estimated_units = min(base_units, 300000)
            
            self.logger.debug(
                "Estimated compute units for Jupiter swap",
                estimated_units=estimated_units,
                route_hops=len(quote.route) if quote.route else 0
            )
            
            return estimated_units
            
        except Exception as e:
            self.logger.warning("Failed to estimate compute units, using default", error=str(e))
            return 150000  # Conservative default
    
    async def get_transaction_status(self, transaction_hash: str) -> SwapResult:
        """
        Get transaction status.
        
        Note: Jupiter doesn't provide transaction status endpoints.
        This should be handled by the Solana RPC integration.
        
        Args:
            transaction_hash: Transaction hash to query
            
        Returns:
            SwapResult with status information
            
        Raises:
            DEXError: Always, as Jupiter doesn't provide this
        """
        raise DEXError(
            "Transaction status checking not available via Jupiter API. "
            "Use Solana RPC integration instead."
        )
    
    async def get_best_route(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal
    ) -> List[str]:
        """
        Get the best routing path for a token swap.
        
        Args:
            input_token: Input token symbol or address
            output_token: Output token symbol or address
            amount: Amount to swap
            
        Returns:
            List of token addresses in the optimal route
            
        Raises:
            DEXError: If route calculation fails
        """
        try:
            quote = await self.get_quote(input_token, output_token, amount)
            return quote.route or [quote.input_token, quote.output_token]
        except Exception as e:
            self.logger.error("Failed to get best route", error=str(e))
            raise DEXError(f"Failed to get best route: {str(e)}")
    
    async def close(self):
        """Close the HTTP session and cleanup resources."""
        await self.disconnect()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()