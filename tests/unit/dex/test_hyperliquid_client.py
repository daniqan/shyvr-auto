"""
Test suite for Hyperliquid DEX client implementation.

This test suite follows Test-Driven Development (TDD) methodology by defining
the expected behavior of the HyperliquidDEXClient before implementation.

Key differences from Jupiter (order book vs AMM):
- Hyperliquid uses order book trading vs Jupiter's AMM routing
- Perpetual futures focus vs spot trading
- Different quote structure and execution model
- Authentication required for trading vs public quotes
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.utils.base import Chain
from src.dex.base import (
    DEXBase,
    DEXConfig,
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    DEXError,
    DEXConnectionError,
    DEXTransactionError,
)


class TestHyperliquidDEXClientTDD:
    """Test-driven development tests for Hyperliquid DEX client."""
    
    @pytest.fixture
    def hyperliquid_config(self):
        """Create test configuration for Hyperliquid."""
        return DEXConfig(
            chain=Chain.HYPERLIQUID,  # Need to add this to Chain enum
            name="hyperliquid",
            api_key="test_api_key_here",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            max_slippage_bps=100,  # 1% for futures trading
            timeout_seconds=30,
            rate_limit_per_second=5,  # More conservative for authenticated API
            enable_price_impact_warnings=True,
            max_price_impact_bps=500,  # 5% for higher leverage trading
        )
    
    @pytest.fixture
    def mock_hyperliquid_client(self, hyperliquid_config):
        """Create mock Hyperliquid client for testing."""
        # This will fail initially - need to implement HyperliquidDEXClient
        from src.dex.hyperliquid_client import HyperliquidDEXClient
        return HyperliquidDEXClient(hyperliquid_config)
    
    def test_hyperliquid_client_initialization(self, hyperliquid_config):
        """Test that Hyperliquid client initializes correctly."""
        from src.dex.hyperliquid_client import HyperliquidDEXClient
        
        client = HyperliquidDEXClient(hyperliquid_config)
        
        assert client.config == hyperliquid_config
        assert client.chain == Chain.HYPERLIQUID
        assert client.name == "hyperliquid"
        assert not client.is_connected
        assert client.session is None
        assert hasattr(client, 'hyperliquid_client')  # Internal SDK client
    
    def test_hyperliquid_client_default_config(self):
        """Test that Hyperliquid client creates default config correctly."""
        from src.dex.hyperliquid_client import HyperliquidDEXClient
        
        client = HyperliquidDEXClient()
        
        assert client.config.chain == Chain.HYPERLIQUID
        assert client.config.name == "hyperliquid"
        assert client.config.max_slippage_bps == 100  # Higher default for futures
        assert client.config.rate_limit_per_second == 5  # Conservative rate limit
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_hyperliquid_client):
        """Test successful connection to Hyperliquid."""
        with patch.object(
            mock_hyperliquid_client, '_authenticate', new_callable=AsyncMock
        ) as mock_auth:
            mock_auth.return_value = True
            
            result = await mock_hyperliquid_client.connect()
            
            assert result is True
            assert mock_hyperliquid_client.is_connected
            mock_auth.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_hyperliquid_client):
        """Test connection failure handling."""
        with patch.object(
            mock_hyperliquid_client, '_authenticate', new_callable=AsyncMock
        ) as mock_auth:
            mock_auth.side_effect = DEXConnectionError("Authentication failed")
            
            with pytest.raises(DEXConnectionError):
                await mock_hyperliquid_client.connect()
            
            assert not mock_hyperliquid_client.is_connected
    
    @pytest.mark.asyncio
    async def test_get_quote_btc_usdc(self, mock_hyperliquid_client):
        """Test getting a quote for BTC/USDC perpetual."""
        # Mock the internal Hyperliquid SDK response
        mock_quote_response = {
            "price": "45000.50",
            "size": "1.0",
            "side": "buy",
            "orderbook_depth": 10,
            "funding_rate": "0.0001",
            "mark_price": "45000.75",
            "estimated_fees": "4.50",
        }
        
        with patch.object(
            mock_hyperliquid_client, '_get_orderbook_quote', new_callable=AsyncMock
        ) as mock_quote:
            mock_quote.return_value = mock_quote_response
            
            quote = await mock_hyperliquid_client.get_quote(
                input_token="USDC",
                output_token="BTC-PERP",
                amount=Decimal("45000"),
                swap_type=SwapType.EXACT_INPUT,
                slippage_bps=50
            )
            
            assert isinstance(quote, SwapQuote)
            assert quote.input_token == "USDC"
            assert quote.output_token == "BTC-PERP"
            assert quote.input_amount == Decimal("45000")
            assert quote.price == Decimal("45000.50")
            assert quote.slippage_bps == 50
            assert quote.dex_name == "hyperliquid"
            assert quote.quote_id is not None
    
    @pytest.mark.asyncio
    async def test_get_quote_with_leverage(self, mock_hyperliquid_client):
        """Test getting a quote with leverage (Hyperliquid-specific feature)."""
        mock_quote_response = {
            "price": "45000.50",
            "size": "10.0",  # 10x leverage
            "side": "buy",
            "leverage": 10,
            "margin_required": "4500.05",
            "liquidation_price": "40500.45",
            "estimated_fees": "45.00",
        }
        
        with patch.object(
            mock_hyperliquid_client, '_get_leveraged_quote', new_callable=AsyncMock
        ) as mock_quote:
            mock_quote.return_value = mock_quote_response
            
            quote = await mock_hyperliquid_client.get_leveraged_quote(
                input_token="USDC",
                output_token="BTC-PERP",
                amount=Decimal("4500"),  # Margin amount
                leverage=10,
                slippage_bps=100
            )
            
            assert isinstance(quote, SwapQuote)
            assert quote.input_amount == Decimal("4500")  # Margin
            assert quote.output_amount == Decimal("10.0")  # Position size
            assert "leverage" in quote.additional_fees
            assert quote.additional_fees["leverage"] == Decimal("10")
    
    @pytest.mark.asyncio
    async def test_execute_swap_perpetual_order(self, mock_hyperliquid_client):
        """Test executing a perpetual futures order."""
        # Create a test quote
        quote = SwapQuote(
            input_token="USDC",
            output_token="BTC-PERP",
            input_amount=Decimal("45000"),
            output_amount=Decimal("1.0"),
            price=Decimal("45000.50"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="hyperliquid",
            quote_id="hl_quote_123",
            additional_fees={"trading_fee": Decimal("22.50")},
        )
        
        # Mock successful order execution
        mock_order_response = {
            "order_id": "hl_order_456",
            "status": "filled",
            "filled_size": "1.0",
            "avg_fill_price": "45000.25",
            "fees_paid": "22.50",
            "transaction_hash": "0xabc123def456",
        }
        
        with patch.object(
            mock_hyperliquid_client, '_execute_order', new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_order_response
            
            result = await mock_hyperliquid_client.execute_swap(
                quote=quote,
                wallet_address="0x1234567890abcdef1234567890abcdef12345678"
            )
            
            assert isinstance(result, SwapResult)
            assert result.transaction_hash == "0xabc123def456"
            assert result.status == SwapStatus.CONFIRMED
            assert result.actual_output_amount == Decimal("1.0")
            assert result.dex_name == "hyperliquid"
    
    @pytest.mark.asyncio
    async def test_get_token_price_perpetual(self, mock_hyperliquid_client):
        """Test getting price for a perpetual futures contract."""
        mock_price_response = {
            "symbol": "BTC-PERP",
            "mark_price": "45000.75",
            "index_price": "45000.50",
            "funding_rate": "0.0001",
            "next_funding": "2024-01-01T12:00:00Z",
        }
        
        with patch.object(
            mock_hyperliquid_client, '_get_perpetual_price', new_callable=AsyncMock
        ) as mock_price:
            mock_price.return_value = mock_price_response
            
            price = await mock_hyperliquid_client.get_token_price(
                token_address="BTC-PERP",
                base_token="USDC"
            )
            
            assert price == Decimal("45000.75")  # Mark price for perpetuals
    
    @pytest.mark.asyncio
    async def test_get_supported_tokens_perpetuals(self, mock_hyperliquid_client):
        """Test getting supported perpetual contracts."""
        mock_tokens_response = [
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
        
        with patch.object(
            mock_hyperliquid_client, '_get_available_perpetuals', new_callable=AsyncMock
        ) as mock_tokens:
            mock_tokens.return_value = mock_tokens_response
            
            tokens = await mock_hyperliquid_client.get_supported_tokens()
            
            assert len(tokens) == 2
            assert tokens[0]["symbol"] == "BTC-PERP"
            assert tokens[0]["max_leverage"] == 50
            assert tokens[1]["symbol"] == "ETH-PERP"
    
    @pytest.mark.asyncio
    async def test_estimate_gas_zero_fees(self, mock_hyperliquid_client):
        """Test gas estimation (should be zero for Hyperliquid)."""
        # Hyperliquid has zero gas fees
        gas_estimate = await mock_hyperliquid_client.estimate_gas(
            input_token="USDC",
            output_token="BTC-PERP",
            amount=Decimal("45000"),
            wallet_address="0x1234567890abcdef1234567890abcdef12345678"
        )
        
        assert gas_estimate == 0  # Zero gas fees on Hyperliquid
    
    @pytest.mark.asyncio
    async def test_get_transaction_status_order_tracking(self, mock_hyperliquid_client):
        """Test getting transaction/order status."""
        mock_status_response = {
            "order_id": "hl_order_456",
            "status": "filled",
            "filled_size": "1.0",
            "remaining_size": "0.0",
            "avg_fill_price": "45000.25",
            "fees_paid": "22.50",
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T10:00:05Z",
        }
        
        with patch.object(
            mock_hyperliquid_client, '_get_order_status', new_callable=AsyncMock
        ) as mock_status:
            mock_status.return_value = mock_status_response
            
            result = await mock_hyperliquid_client.get_transaction_status("hl_order_456")
            
            assert isinstance(result, SwapResult)
            assert result.transaction_hash == "hl_order_456"
            assert result.status == SwapStatus.CONFIRMED
            assert result.actual_output_amount == Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self, mock_hyperliquid_client):
        """Test proper cleanup on disconnect."""
        # Set up connected state
        mock_hyperliquid_client._connected = True
        mock_session = AsyncMock()
        mock_hyperliquid_client.session = mock_session
        
        await mock_hyperliquid_client.disconnect()
        
        assert not mock_hyperliquid_client.is_connected
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_hyperliquid_client):
        """Test rate limiting functionality."""
        # This test ensures we respect Hyperliquid's rate limits
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Mock multiple rapid requests
            for _ in range(10):
                with patch.object(
                    mock_hyperliquid_client, '_make_request', new_callable=AsyncMock
                ) as mock_request:
                    mock_request.return_value = {"success": True}
                    await mock_hyperliquid_client._rate_limited_request("GET", "/test")
            
            # Should have called sleep due to rate limiting
            assert mock_sleep.call_count >= 5  # Depends on rate limit implementation
    
    def test_hyperliquid_specific_constants(self):
        """Test Hyperliquid-specific constants and configurations."""
        from src.dex.hyperliquid_client import HyperliquidDEXClient
        
        # These constants should exist in the implementation
        assert hasattr(HyperliquidDEXClient, 'API_URL')
        assert hasattr(HyperliquidDEXClient, 'TESTNET_URL')
        assert hasattr(HyperliquidDEXClient, 'COMMON_PERPETUALS')
        
        # Check that common perpetuals are defined
        common_perps = HyperliquidDEXClient.COMMON_PERPETUALS
        assert "BTC-PERP" in common_perps
        assert "ETH-PERP" in common_perps
        assert "SOL-PERP" in common_perps
    
    @pytest.mark.asyncio
    async def test_funding_rate_integration(self, mock_hyperliquid_client):
        """Test funding rate considerations in quotes."""
        # Hyperliquid perpetuals have funding rates that affect trading decisions
        mock_quote_response = {
            "price": "45000.50",
            "size": "1.0",
            "side": "buy",
            "funding_rate": "0.0001",  # 0.01% funding rate
            "time_to_funding": 3600,   # 1 hour to next funding
        }
        
        with patch.object(
            mock_hyperliquid_client, '_get_orderbook_quote', new_callable=AsyncMock
        ) as mock_quote:
            mock_quote.return_value = mock_quote_response
            
            quote = await mock_hyperliquid_client.get_quote(
                input_token="USDC",
                output_token="BTC-PERP",
                amount=Decimal("45000"),
                swap_type=SwapType.EXACT_INPUT
            )
            
            # Funding rate should be included in additional fees or metadata
            assert quote.additional_fees is not None
            assert "funding_rate" in quote.additional_fees
            assert quote.additional_fees["funding_rate"] == Decimal("0.0001")

    @pytest.mark.asyncio
    async def test_sdk_initialization_error(self, hyperliquid_config):
        """Test SDK initialization error handling (lines 105-107)."""
        with patch('src.dex.hyperliquid_client.HyperliquidAsync') as mock_sdk:
            mock_sdk.side_effect = Exception("SDK initialization failed")
            
            with pytest.raises(DEXConnectionError, match="SDK initialization failed"):
                from src.dex.hyperliquid_client import HyperliquidDEXClient
                HyperliquidDEXClient(hyperliquid_config)

    @pytest.mark.asyncio
    async def test_session_creation_error(self, mock_hyperliquid_client):
        """Test session creation errors (lines 111-120)."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.side_effect = Exception("Session creation failed")
            
            with pytest.raises(Exception, match="Session creation failed"):
                await mock_hyperliquid_client._get_session()

    @pytest.mark.asyncio
    async def test_http_request_unsupported_method(self, mock_hyperliquid_client):
        """Test unsupported HTTP method error (lines 164-165)."""
        with patch.object(mock_hyperliquid_client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session
            
            with pytest.raises(DEXError, match="Unsupported HTTP method: PATCH"):
                await mock_hyperliquid_client._make_request("PATCH", "/test")

    @pytest.mark.asyncio
    async def test_http_request_client_error(self, mock_hyperliquid_client):
        """Test HTTP client error handling (lines 167-169)."""
        import aiohttp
        
        with patch.object(mock_hyperliquid_client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session
            
            # Create a mock context manager for session.get()
            mock_context = AsyncMock()
            mock_context.__aenter__.side_effect = aiohttp.ClientError("Network error")
            mock_session.get.return_value = mock_context
            
            with pytest.raises(DEXConnectionError, match="Request failed: Network error"):
                await mock_hyperliquid_client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_http_request_rate_limit_error(self, mock_hyperliquid_client):
        """Test rate limit error handling (lines 152-153, 159-160)."""
        with patch.object(mock_hyperliquid_client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 429
            
            # Create a proper context manager mock
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None
            mock_session.get.return_value = mock_context
            mock_get_session.return_value = mock_session
            
            with pytest.raises(DEXRateLimitError, match="Rate limit exceeded"):
                await mock_hyperliquid_client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_http_request_unexpected_error(self, mock_hyperliquid_client):
        """Test unexpected error handling (lines 170-172)."""
        with patch.object(mock_hyperliquid_client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session
            
            # Create a mock context manager that raises an error
            mock_context = AsyncMock()
            mock_context.__aenter__.side_effect = RuntimeError("Unexpected error")
            mock_session.get.return_value = mock_context
            
            with pytest.raises(DEXError, match="Request error: Unexpected error"):
                await mock_hyperliquid_client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_authentication_no_client_error(self, mock_hyperliquid_client):
        """Test authentication with no client initialized (lines 185-186)."""
        mock_hyperliquid_client.hyperliquid_client = None
        
        with pytest.raises(DEXConnectionError, match="Hyperliquid client not initialized"):
            await mock_hyperliquid_client._authenticate()

    @pytest.mark.asyncio
    async def test_authentication_failure_error(self, mock_hyperliquid_client):
        """Test authentication failure handling (lines 193-195)."""
        # Set the client but make it raise an exception during authentication test
        mock_hyperliquid_client.hyperliquid_client = AsyncMock()
        
        # Patch the logger to trigger the exception path in _authenticate
        with patch.object(mock_hyperliquid_client.logger, 'info') as mock_logger:
            mock_logger.side_effect = Exception("Auth failed")
            
            with pytest.raises(DEXConnectionError, match="Authentication failed: Auth failed"):
                await mock_hyperliquid_client._authenticate()

    @pytest.mark.asyncio
    async def test_get_quote_error_handling(self, mock_hyperliquid_client):
        """Test quote generation error handling (lines 344-346)."""
        with patch.object(
            mock_hyperliquid_client, '_get_orderbook_quote', new_callable=AsyncMock
        ) as mock_quote:
            mock_quote.side_effect = Exception("Quote API failed")
            
            with pytest.raises(DEXError, match="Quote request failed: Quote API failed"):
                await mock_hyperliquid_client.get_quote(
                    input_token="USDC",
                    output_token="BTC-PERP",
                    amount=Decimal("45000"),
                    swap_type=SwapType.EXACT_INPUT
                )

    @pytest.mark.asyncio
    async def test_get_leveraged_quote_error_handling(self, mock_hyperliquid_client):
        """Test leveraged quote generation error handling (lines 391-393)."""
        with patch.object(
            mock_hyperliquid_client, '_get_leveraged_quote', new_callable=AsyncMock
        ) as mock_quote:
            mock_quote.side_effect = Exception("Leveraged quote failed")
            
            with pytest.raises(DEXError, match="Leveraged quote request failed: Leveraged quote failed"):
                await mock_hyperliquid_client.get_leveraged_quote(
                    input_token="USDC",
                    output_token="BTC-PERP",
                    amount=Decimal("4500"),
                    leverage=10,
                    slippage_bps=100
                )

    @pytest.mark.asyncio
    async def test_execute_swap_validation_failure(self, mock_hyperliquid_client):
        """Test swap execution validation failure (lines 428-429)."""
        quote = SwapQuote(
            input_token="USDC",
            output_token="BTC-PERP",
            input_amount=Decimal("45000"),
            output_amount=Decimal("1.0"),
            price=Decimal("45000.50"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="hyperliquid",
            quote_id="hl_quote_123",
        )
        
        with patch.object(mock_hyperliquid_client, 'validate_quote') as mock_validate:
            mock_validate.return_value = False
            
            with pytest.raises(DEXTransactionError, match="Quote validation failed"):
                await mock_hyperliquid_client.execute_swap(
                    quote=quote,
                    wallet_address="0x1234567890abcdef1234567890abcdef12345678"
                )

    @pytest.mark.asyncio
    async def test_execute_swap_execution_failure(self, mock_hyperliquid_client):
        """Test swap execution failure handling (lines 457-459)."""
        quote = SwapQuote(
            input_token="USDC",
            output_token="BTC-PERP",
            input_amount=Decimal("45000"),
            output_amount=Decimal("1.0"),
            price=Decimal("45000.50"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="hyperliquid",
            quote_id="hl_quote_123",
        )
        
        with patch.object(mock_hyperliquid_client, 'validate_quote') as mock_validate:
            mock_validate.return_value = True
            
            with patch.object(
                mock_hyperliquid_client, '_execute_order', new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.side_effect = Exception("Order execution failed")
                
                with pytest.raises(DEXTransactionError, match="Order execution failed: Order execution failed"):
                    await mock_hyperliquid_client.execute_swap(
                        quote=quote,
                        wallet_address="0x1234567890abcdef1234567890abcdef12345678"
                    )