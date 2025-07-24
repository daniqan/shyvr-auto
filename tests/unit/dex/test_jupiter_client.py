"""
Unit tests for Jupiter DEX Client.

Tests the Jupiter DEX client implementation for Solana token swaps,
including quote generation, transaction preparation, and API integration.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import aiohttp

from src.dex.jupiter_client import JupiterDEXClient
from src.dex.base import (
    DEXConfig,
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    DEXError,
    DEXConnectionError,
    DEXTransactionError,
    DEXRateLimitError,
)
from src.utils.base import Chain


class TestJupiterDEXClient:
    """Test Jupiter DEX client functionality."""
    
    @pytest.fixture
    def jupiter_client(self):
        """Create Jupiter client for testing."""
        config = DEXConfig(
            chain=Chain.SOLANA,
            name="jupiter",
            max_slippage_bps=50,
            timeout_seconds=30
        )
        return JupiterDEXClient(config)
    
    @pytest.fixture
    def default_client(self):
        """Create Jupiter client with default config."""
        return JupiterDEXClient()
    
    def test_initialization(self, jupiter_client):
        """Test Jupiter client initialization."""
        assert jupiter_client.chain == Chain.SOLANA
        assert jupiter_client.name == "jupiter"
        assert jupiter_client.config.max_slippage_bps == 50
        assert jupiter_client.session is None
        assert not jupiter_client.is_connected
    
    def test_default_initialization(self, default_client):
        """Test Jupiter client with default configuration."""
        assert default_client.chain == Chain.SOLANA
        assert default_client.name == "jupiter"
        assert default_client.config.max_slippage_bps == 50
    
    def test_token_address_resolution(self, jupiter_client):
        """Test token symbol to address resolution."""
        # Test known symbols
        assert jupiter_client._resolve_token_address("SOL") == "So11111111111111111111111111111111111111112"
        assert jupiter_client._resolve_token_address("USDC") == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        assert jupiter_client._resolve_token_address("sol") == "So11111111111111111111111111111111111111112"  # Case insensitive
        
        # Test address pass-through
        test_address = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
        assert jupiter_client._resolve_token_address(test_address) == test_address
        
        # Test unknown symbol
        assert jupiter_client._resolve_token_address("UNKNOWN") == "UNKNOWN"
    
    @pytest.mark.asyncio
    async def test_session_creation(self, jupiter_client):
        """Test aiohttp session creation."""
        session = await jupiter_client._get_session()
        assert isinstance(session, aiohttp.ClientSession)
        assert session.timeout.total == 30
        
        # Test session reuse
        session2 = await jupiter_client._get_session()
        assert session is session2
        
        await jupiter_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, jupiter_client):
        """Test successful API request."""
        mock_response_data = {"test": "data"}
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_request.return_value.__aenter__.return_value = mock_response
            
            result = await jupiter_client._make_request("GET", "https://test.com")
            assert result == mock_response_data
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, jupiter_client):
        """Test rate limit error handling."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(DEXRateLimitError):
                await jupiter_client._make_request("GET", "https://test.com")
    
    @pytest.mark.asyncio
    async def test_make_request_bad_request(self, jupiter_client):
        """Test bad request error handling."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad request error")
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(DEXError, match="Jupiter API bad request"):
                await jupiter_client._make_request("GET", "https://test.com")
    
    @pytest.mark.asyncio
    async def test_make_request_server_error(self, jupiter_client):
        """Test server error handling."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server error")
            mock_request.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(DEXConnectionError, match="Jupiter API error 500"):
                await jupiter_client._make_request("GET", "https://test.com")
    
    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, jupiter_client):
        """Test connection error handling."""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Connection failed")
            
            with pytest.raises(DEXConnectionError, match="Jupiter API request failed"):
                await jupiter_client._make_request("GET", "https://test.com")
    
    @pytest.mark.asyncio
    async def test_connect_success(self, jupiter_client):
        """Test successful connection."""
        with patch.object(jupiter_client, 'get_supported_tokens', return_value=[{"test": "token"}]):
            success = await jupiter_client.connect()
            assert success
            assert jupiter_client.is_connected
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, jupiter_client):
        """Test connection failure."""
        with patch.object(jupiter_client, 'get_supported_tokens', side_effect=Exception("Connection failed")):
            with pytest.raises(DEXConnectionError):
                await jupiter_client.connect()
            assert not jupiter_client.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, jupiter_client):
        """Test disconnection."""
        # Create a session first
        await jupiter_client._get_session()
        assert jupiter_client.session is not None
        
        await jupiter_client.disconnect()
        assert jupiter_client.session is None
        assert not jupiter_client.is_connected
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, jupiter_client):
        """Test successful quote generation."""
        mock_response = {
            "inAmount": "1000000",  # 1 SOL in lamports
            "outAmount": "100000000",  # 100 USDC in microunits
            "priceImpactPct": "0.5",  # 0.5%
            "routePlan": [
                {"swapInfo": {"outputMint": "intermediate_token"}},
                {"swapInfo": {"outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"}}
            ],
            "quoteResponse": {"timeTaken": "123456"},
            "platformFee": {"amount": "1000"}
        }
        
        with patch.object(jupiter_client, '_make_request', return_value=mock_response):
            quote = await jupiter_client.get_quote(
                input_token="SOL",
                output_token="USDC", 
                amount=Decimal('1000000'),
                slippage_bps=100
            )
            
            assert quote.input_token == "So11111111111111111111111111111111111111112"
            assert quote.output_token == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            assert quote.input_amount == Decimal('1000000')
            assert quote.output_amount == Decimal('100000000')
            assert quote.price_impact_bps == 5000  # 0.5% * 10000
            assert quote.slippage_bps == 100
            assert quote.dex_name == "jupiter"
            assert len(quote.route) == 2
            assert quote.additional_fees["platformFee"] == Decimal('1000')
    
    @pytest.mark.asyncio
    async def test_get_quote_with_defaults(self, jupiter_client):
        """Test quote generation with default parameters."""
        mock_response = {
            "inAmount": "1000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.1",
            "routePlan": []
        }
        
        with patch.object(jupiter_client, '_make_request', return_value=mock_response):
            quote = await jupiter_client.get_quote(
                input_token="SOL",
                output_token="USDC",
                amount=Decimal('1000000')
            )
            
            # Should use config default slippage
            assert quote.slippage_bps == 50
    
    @pytest.mark.asyncio
    async def test_get_quote_failure(self, jupiter_client):
        """Test quote generation failure."""
        with patch.object(jupiter_client, '_make_request', side_effect=Exception("API error")):
            with pytest.raises(DEXError, match="Failed to get Jupiter quote"):
                await jupiter_client.get_quote("SOL", "USDC", Decimal('1000000'))
    
    @pytest.mark.asyncio
    async def test_execute_swap_success(self, jupiter_client):
        """Test successful swap execution preparation."""
        quote = SwapQuote(
            input_token="So11111111111111111111111111111111111111112",
            output_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            input_amount=Decimal('1000000'),
            output_amount=Decimal('100000000'),
            price=Decimal('100'),
            price_impact_bps=50,
            slippage_bps=100,
            valid_until=datetime.now() + timedelta(minutes=5),
            dex_name="jupiter"
        )
        
        mock_response = {
            "swapTransaction": "encoded_transaction_data",
            "lastValidBlockHeight": 123456789
        }
        
        with patch.object(jupiter_client, 'validate_quote', return_value=True), \
             patch.object(jupiter_client, '_make_request', return_value=mock_response):
            
            result = await jupiter_client.execute_swap(
                quote=quote,
                wallet_address="test_wallet_address"
            )
            
            assert result.status == SwapStatus.PENDING
            assert result.transaction_hash == "pending"
            assert result.input_token == quote.input_token
            assert result.output_token == quote.output_token
            assert result.dex_name == "jupiter"
            assert result.transaction_data is not None
            assert "swapTransaction" in result.transaction_data
    
    @pytest.mark.asyncio 
    async def test_execute_swap_invalid_quote(self, jupiter_client):
        """Test swap execution with invalid quote."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=50
        )
        
        with patch.object(jupiter_client, 'validate_quote', return_value=False):
            with pytest.raises(DEXTransactionError, match="Invalid or expired quote"):
                await jupiter_client.execute_swap(quote, "wallet_address")
    
    @pytest.mark.asyncio
    async def test_get_token_price_success(self, jupiter_client):
        """Test successful token price retrieval."""
        mock_response = {
            "data": {
                "So11111111111111111111111111111111111111112": {
                    "price": "100.50"
                }
            }
        }
        
        with patch.object(jupiter_client, '_make_request', return_value=mock_response):
            price = await jupiter_client.get_token_price("SOL", "USDC")
            assert price == Decimal('100.50')
    
    @pytest.mark.asyncio
    async def test_get_token_price_not_found(self, jupiter_client):
        """Test token price not found."""
        mock_response = {"data": {}}
        
        with patch.object(jupiter_client, '_make_request', return_value=mock_response):
            with pytest.raises(DEXError, match="Price data not found"):
                await jupiter_client.get_token_price("UNKNOWN_TOKEN")
    
    @pytest.mark.asyncio
    async def test_get_supported_tokens_success(self, jupiter_client):
        """Test successful supported tokens retrieval."""
        mock_tokens = [
            {"address": "token1", "symbol": "T1", "name": "Token 1"},
            {"address": "token2", "symbol": "T2", "name": "Token 2"}
        ]
        
        with patch.object(jupiter_client, '_make_request', return_value=mock_tokens):
            tokens = await jupiter_client.get_supported_tokens()
            assert len(tokens) == 2
            assert tokens[0]["symbol"] == "T1"
            assert "token1" in jupiter_client._token_cache
    
    @pytest.mark.asyncio
    async def test_get_supported_tokens_cached(self, jupiter_client):
        """Test cached supported tokens retrieval."""
        # Set up cache
        jupiter_client._token_cache = {"token1": {"symbol": "T1"}}
        jupiter_client._cache_expiry = datetime.now() + timedelta(minutes=10)
        
        tokens = await jupiter_client.get_supported_tokens()
        assert len(tokens) == 1
        assert tokens[0]["symbol"] == "T1"
    
    @pytest.mark.asyncio
    async def test_estimate_gas_success(self, jupiter_client):
        """Test successful gas estimation."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=50,
            route=["intermediate1", "intermediate2"]  # 2 hops
        )
        
        with patch.object(jupiter_client, 'get_quote', return_value=quote):
            gas = await jupiter_client.estimate_gas("token1", "token2", Decimal('100'), "wallet")
            # Base 100k + 2 hops * 20k = 140k
            assert gas == 140000
    
    @pytest.mark.asyncio
    async def test_estimate_gas_failure(self, jupiter_client):
        """Test gas estimation failure fallback."""
        with patch.object(jupiter_client, 'get_quote', side_effect=Exception("Quote failed")):
            gas = await jupiter_client.estimate_gas("token1", "token2", Decimal('100'), "wallet")
            assert gas == 150000  # Default fallback
    
    @pytest.mark.asyncio
    async def test_get_transaction_status_not_supported(self, jupiter_client):
        """Test transaction status query (not supported by Jupiter)."""
        with pytest.raises(DEXError, match="Transaction status checking not available"):
            await jupiter_client.get_transaction_status("tx_hash")
    
    @pytest.mark.asyncio
    async def test_get_best_route_success(self, jupiter_client):
        """Test best route retrieval."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=50,
            route=["intermediate1", "token2"]
        )
        
        with patch.object(jupiter_client, 'get_quote', return_value=quote):
            route = await jupiter_client.get_best_route("token1", "token2", Decimal('100'))
            assert route == ["intermediate1", "token2"]
    
    @pytest.mark.asyncio
    async def test_get_best_route_no_route(self, jupiter_client):
        """Test best route with no intermediate tokens."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=50,
            route=None
        )
        
        with patch.object(jupiter_client, 'get_quote', return_value=quote):
            route = await jupiter_client.get_best_route("token1", "token2", Decimal('100'))
            assert route == ["token1", "token2"]
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, jupiter_client):
        """Test async context manager functionality."""
        with patch.object(jupiter_client, 'connect', return_value=True) as mock_connect, \
             patch.object(jupiter_client, 'close') as mock_close:
            
            async with jupiter_client as client:
                assert client is jupiter_client
                mock_connect.assert_called_once()
            
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close(self, jupiter_client):
        """Test close method."""
        with patch.object(jupiter_client, 'disconnect') as mock_disconnect:
            await jupiter_client.close()
            mock_disconnect.assert_called_once()