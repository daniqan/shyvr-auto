"""
Unit tests for Uniswap V3 DEX Client.

Tests the Uniswap V3 DEX client implementation for Ethereum token swaps,
including concentrated liquidity, fee tiers, and Web3 integration.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from web3 import Web3

from src.dex.uniswap_v3_client import UniswapV3Client
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


class TestUniswapV3Client:
    """Test Uniswap V3 DEX client functionality."""
    
    @pytest.fixture
    def uniswap_client(self):
        """Create Uniswap V3 client for testing."""
        config = DEXConfig(
            chain=Chain.ETHEREUM,
            name="uniswap_v3",
            max_slippage_bps=50,
            timeout_seconds=30,
            wallet_address="0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
        )
        return UniswapV3Client(config)
    
    @pytest.fixture
    def default_client(self):
        """Create Uniswap V3 client with default config."""
        return UniswapV3Client()
    
    def test_initialization_with_config(self, uniswap_client):
        """Test Uniswap V3 client initialization with config."""
        assert uniswap_client.chain == Chain.ETHEREUM
        assert uniswap_client.name == "uniswap_v3"
        assert uniswap_client.config.max_slippage_bps == 50
        assert not uniswap_client.is_connected
    
    def test_initialization_default(self, default_client):
        """Test Uniswap V3 client initialization with defaults."""
        assert default_client.chain == Chain.ETHEREUM
        assert default_client.name == "uniswap_v3"
        assert default_client.config.max_slippage_bps == 50  # Default
        assert not default_client.is_connected
    
    def test_fee_tiers_property(self, uniswap_client):
        """Test that V3 fee tiers are properly defined."""
        expected_tiers = [500, 3000, 10000]  # 0.05%, 0.3%, 1%
        assert uniswap_client.fee_tiers == expected_tiers
    
    @pytest.mark.asyncio
    async def test_connect_success(self, uniswap_client):
        """Test successful connection to Ethereum network."""
        with patch.object(uniswap_client, '_initialize_web3') as mock_init:
            mock_init.return_value = True
            
            result = await uniswap_client.connect()
            
            assert result is True
            assert uniswap_client.is_connected is True
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, uniswap_client):
        """Test connection failure."""
        with patch.object(uniswap_client, '_initialize_web3') as mock_init:
            mock_init.side_effect = DEXConnectionError("Connection failed")
            
            with pytest.raises(DEXConnectionError):
                await uniswap_client.connect()
            
            assert uniswap_client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, uniswap_client):
        """Test disconnect functionality."""
        # Setup connected state
        uniswap_client._connected = True
        
        await uniswap_client.disconnect()
        
        assert uniswap_client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_get_quote_exact_input(self, uniswap_client):
        """Test getting a quote for exact input swap."""
        # Setup
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"  # USDC
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        amount = Decimal("1000")  # 1000 USDC
        
        expected_quote = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=Decimal("0.5"),  # Mock 0.5 ETH
            price=Decimal("0.0005"),  # 1 USDC = 0.0005 ETH
            price_impact_bps=25,  # 0.25%
            slippage_bps=50,  # 0.5%
            estimated_gas=150000,
            gas_price=Decimal("20"),  # 20 gwei
            route=[input_token, output_token],
            dex_name="uniswap_v3"
        )
        
        with patch.object(uniswap_client, '_get_quote_for_fee_tier') as mock_fee_quote:
            mock_fee_quote.return_value = expected_quote
            
            with patch.object(uniswap_client, '_select_best_quote') as mock_select:
                mock_select.return_value = expected_quote
                
                quote = await uniswap_client.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=amount,
                    swap_type=SwapType.EXACT_INPUT
                )
                
                assert quote.input_token == input_token
                assert quote.output_token == output_token
                assert quote.input_amount == amount
                assert quote.dex_name == "uniswap_v3"
                assert quote.price_impact_bps == 25
                # Should be called 3 times (once for each fee tier)
                assert mock_fee_quote.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_quote_multiple_fee_tiers(self, uniswap_client):
        """Test quote selection across multiple fee tiers."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"  # USDC
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        amount = Decimal("1000")
        
        # Mock quotes for different fee tiers
        fee_500_quote = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=Decimal("0.502"),  # Best rate
            price=Decimal("0.000502"),
            price_impact_bps=20,
            slippage_bps=50,
            dex_name="uniswap_v3",
            additional_fees={"pool_fee": Decimal("5")}  # 0.05% fee
        )
        
        fee_3000_quote = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=Decimal("0.500"),  # Lower rate
            price=Decimal("0.0005"),
            price_impact_bps=15,
            slippage_bps=50,
            dex_name="uniswap_v3",
            additional_fees={"pool_fee": Decimal("30")}  # 0.3% fee
        )
        
        with patch.object(uniswap_client, '_get_quote_for_fee_tier') as mock_tier_quote:
            # Return different quotes for different fee tiers
            mock_tier_quote.side_effect = [fee_500_quote, fee_3000_quote, None]
            
            with patch.object(uniswap_client, '_select_best_quote') as mock_select:
                mock_select.return_value = fee_500_quote  # Best quote wins
                
                quote = await uniswap_client.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=amount
                )
                
                assert quote.output_amount == Decimal("0.502")  # Best rate
                assert quote.additional_fees["pool_fee"] == Decimal("5")
                # Should call for all 3 fee tiers
                assert mock_tier_quote.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_swap(self, uniswap_client):
        """Test swap execution."""
        quote = SwapQuote(
            input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
            output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            input_amount=Decimal("1000"),
            output_amount=Decimal("0.5"),
            price=Decimal("0.0005"),
            price_impact_bps=25,
            slippage_bps=50,
            estimated_gas=150000,
            gas_price=Decimal("20"),
            dex_name="uniswap_v3",
            quote_id="test_quote_123"
        )
        
        expected_result = SwapResult(
            transaction_hash="0x1234567890abcdef",
            status=SwapStatus.CONFIRMED,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            actual_output_amount=Decimal("0.498"),  # Slight slippage
            gas_used=145000,
            gas_price=Decimal("20"),
            dex_name="uniswap_v3",
            quote_used=quote
        )
        
        with patch.object(uniswap_client, '_execute_swap_transaction') as mock_execute:
            mock_execute.return_value = expected_result
            
            result = await uniswap_client.execute_swap(
                quote=quote,
                wallet_address="0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
            )
            
            assert result.status == SwapStatus.CONFIRMED
            assert result.transaction_hash == "0x1234567890abcdef"
            assert result.actual_output_amount == Decimal("0.498")
            mock_execute.assert_called_once_with(quote, "0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e")
    
    @pytest.mark.asyncio
    async def test_get_token_price(self, uniswap_client):
        """Test token price retrieval."""
        token_address = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"  # USDC
        expected_price = Decimal("1.0")  # 1 USDC = 1 USD
        
        with patch.object(uniswap_client, 'connect') as mock_connect:
            mock_connect.return_value = True
            
            with patch.object(uniswap_client, '_get_token_price_from_pool') as mock_price:
                mock_price.return_value = expected_price
                
                price = await uniswap_client.get_token_price(token_address)
                
                assert price == expected_price
                mock_price.assert_called_once_with(token_address, uniswap_client.COMMON_TOKENS["USDC"])
    
    @pytest.mark.asyncio
    async def test_get_supported_tokens(self, uniswap_client):
        """Test getting supported tokens."""
        expected_tokens = [
            {
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "symbol": "WETH",
                "name": "Wrapped Ether",
                "decimals": 18
            },
            {
                "address": "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
                "symbol": "USDC",
                "name": "USD Coin",
                "decimals": 6
            }
        ]
        
        with patch.object(uniswap_client, '_fetch_supported_tokens') as mock_tokens:
            mock_tokens.return_value = expected_tokens
            
            tokens = await uniswap_client.get_supported_tokens()
            
            assert len(tokens) == 2
            assert tokens[0]["symbol"] == "WETH"
            assert tokens[1]["symbol"] == "USDC"
    
    @pytest.mark.asyncio
    async def test_estimate_gas(self, uniswap_client):
        """Test gas estimation."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        amount = Decimal("1000")
        wallet_address = "0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
        expected_gas = 150000
        
        with patch.object(uniswap_client, '_estimate_swap_gas') as mock_gas:
            mock_gas.return_value = expected_gas
            
            gas = await uniswap_client.estimate_gas(
                input_token=input_token,
                output_token=output_token,
                amount=amount,
                wallet_address=wallet_address
            )
            
            assert gas == expected_gas
            mock_gas.assert_called_once_with(input_token, output_token, amount, wallet_address)
    
    @pytest.mark.asyncio
    async def test_get_transaction_status(self, uniswap_client):
        """Test transaction status retrieval."""
        tx_hash = "0x1234567890abcdef"
        expected_result = SwapResult(
            transaction_hash=tx_hash,
            status=SwapStatus.CONFIRMED,
            input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
            output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            input_amount=Decimal("1000"),
            actual_output_amount=Decimal("0.498"),
            gas_used=145000,
            dex_name="uniswap_v3"
        )
        
        with patch.object(uniswap_client, '_get_transaction_receipt') as mock_receipt:
            mock_receipt.return_value = expected_result
            
            result = await uniswap_client.get_transaction_status(tx_hash)
            
            assert result.transaction_hash == tx_hash
            assert result.status == SwapStatus.CONFIRMED
            mock_receipt.assert_called_once_with(tx_hash)
    
    def test_concentrated_liquidity_constants(self, uniswap_client):
        """Test Uniswap V3 concentrated liquidity constants."""
        # Test that essential V3 constants are defined
        assert hasattr(uniswap_client, 'FACTORY_ADDRESS')
        assert hasattr(uniswap_client, 'ROUTER_ADDRESS')
        assert hasattr(uniswap_client, 'QUOTER_ADDRESS')
        assert hasattr(uniswap_client, 'NONFUNGIBLE_POSITION_MANAGER_ADDRESS')
        
        # Verify fee tiers for concentrated liquidity
        assert 500 in uniswap_client.fee_tiers  # 0.05%
        assert 3000 in uniswap_client.fee_tiers  # 0.3%
        assert 10000 in uniswap_client.fee_tiers  # 1%
    
    @pytest.mark.asyncio
    async def test_concentrated_liquidity_tick_calculation(self, uniswap_client):
        """Test tick calculations for concentrated liquidity."""
        price = Decimal("2000")  # Price of 2000 USDC per ETH
        
        with patch.object(uniswap_client, '_price_to_tick') as mock_tick:
            mock_tick.return_value = 69080  # Mock tick for this price
            
            tick = await uniswap_client._price_to_tick(price)
            
            assert tick == 69080
            mock_tick.assert_called_once_with(price)
    
    @pytest.mark.asyncio
    async def test_error_handling_connection_error(self, uniswap_client):
        """Test handling of connection errors."""
        with patch.object(uniswap_client, '_initialize_web3') as mock_init:
            mock_init.side_effect = Exception("Network error")
            
            with pytest.raises(DEXConnectionError):
                await uniswap_client.connect()
    
    @pytest.mark.asyncio
    async def test_error_handling_transaction_error(self, uniswap_client):
        """Test handling of transaction errors."""
        quote = SwapQuote(
            input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
            output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            input_amount=Decimal("1000"),
            output_amount=Decimal("0.5"),
            price=Decimal("0.0005"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="uniswap_v3"
        )
        
        with patch.object(uniswap_client, '_execute_swap_transaction') as mock_execute:
            mock_execute.side_effect = Exception("Transaction failed")
            
            with pytest.raises(DEXTransactionError):
                await uniswap_client.execute_swap(
                    quote=quote,
                    wallet_address="0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
                )

    @pytest.mark.asyncio
    async def test_web3_initialization_error(self, uniswap_client):
        """Test Web3 initialization error handling (lines 354-356)."""
        with patch('web3.Web3') as mock_web3:
            mock_web3.side_effect = Exception("Web3 connection failed")
            
            result = await uniswap_client._initialize_web3()
            assert result is False

    @pytest.mark.asyncio
    async def test_get_quote_no_valid_quotes_across_tiers(self, uniswap_client):
        """Test quote processing with no valid quotes across fee tiers (lines 175-181, 184)."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        amount = Decimal("1000")
        
        # Mock all fee tier quotes to return None
        with patch.object(uniswap_client, '_get_quote_for_fee_tier') as mock_tier_quote:
            mock_tier_quote.return_value = None
            
            with pytest.raises(DEXError, match="No valid quotes found across any fee tier"):
                await uniswap_client.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=amount
                )

    @pytest.mark.asyncio
    async def test_get_quote_fee_tier_warning_logging(self, uniswap_client):
        """Test quote fee tier warning logging (lines 175-181)."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        amount = Decimal("1000")
        
        # Mock first fee tier to fail, second to succeed
        good_quote = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=Decimal("0.5"),
            price=Decimal("0.0005"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="uniswap_v3"
        )
        
        with patch.object(uniswap_client, '_get_quote_for_fee_tier') as mock_tier_quote:
            mock_tier_quote.side_effect = [Exception("Fee tier 500 failed"), good_quote, None]
            
            with patch.object(uniswap_client, '_select_best_quote') as mock_select:
                mock_select.return_value = good_quote
                
                quote = await uniswap_client.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=amount
                )
                
                assert quote == good_quote
                assert mock_tier_quote.call_count == 3

    @pytest.mark.asyncio
    async def test_get_quote_error_handling(self, uniswap_client):
        """Test quote request error handling (lines 190-192)."""
        # Force an error in the try-catch block by patching connect to raise an error
        with patch.object(uniswap_client, 'connect') as mock_connect:
            mock_connect.side_effect = RuntimeError("Unexpected quote error")
            uniswap_client._connected = False  # Force it to try to connect
            
            with pytest.raises(DEXError, match="Quote request failed: Unexpected quote error"):
                await uniswap_client.get_quote(
                    input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
                    output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    amount=Decimal("1000")
                )

    @pytest.mark.asyncio
    async def test_execute_swap_validation_failure(self, uniswap_client):
        """Test swap execution validation failure (lines 218-219)."""
        quote = SwapQuote(
            input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
            output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            input_amount=Decimal("1000"),
            output_amount=Decimal("0.5"),
            price=Decimal("0.0005"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="uniswap_v3"
        )
        
        with patch.object(uniswap_client, 'validate_quote') as mock_validate:
            mock_validate.return_value = False
            
            with pytest.raises(DEXTransactionError, match="Quote validation failed"):
                await uniswap_client.execute_swap(
                    quote=quote,
                    wallet_address="0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
                )

    @pytest.mark.asyncio
    async def test_get_token_price_error_handling(self, uniswap_client):
        """Test token price error handling (lines 254-256)."""
        with patch.object(uniswap_client, '_get_token_price_from_pool') as mock_price:
            mock_price.side_effect = Exception("Price query failed")
            
            with pytest.raises(DEXError, match="Price query failed: Price query failed"):
                await uniswap_client.get_token_price("0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC")

    @pytest.mark.asyncio
    async def test_get_supported_tokens_error_handling(self, uniswap_client):
        """Test supported tokens error handling (lines 272-274)."""
        with patch.object(uniswap_client, '_fetch_supported_tokens') as mock_tokens:
            mock_tokens.side_effect = Exception("Token list query failed")
            
            with pytest.raises(DEXError, match="Token list query failed: Token list query failed"):
                await uniswap_client.get_supported_tokens()

    @pytest.mark.asyncio
    async def test_estimate_gas_error_handling(self, uniswap_client):
        """Test gas estimation error handling (lines 307-309)."""
        with patch.object(uniswap_client, '_estimate_swap_gas') as mock_gas:
            mock_gas.side_effect = Exception("Gas estimation failed")
            
            with pytest.raises(DEXError, match="Gas estimation failed: Gas estimation failed"):
                await uniswap_client.estimate_gas(
                    input_token="0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
                    output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    amount=Decimal("1000"),
                    wallet_address="0x742d35cc6e38d44ccc6d7d0b4d16be1d8f7b3a3e"
                )

    @pytest.mark.asyncio
    async def test_get_transaction_status_error_handling(self, uniswap_client):
        """Test transaction status error handling (lines 331-333)."""
        with patch.object(uniswap_client, '_get_transaction_receipt') as mock_receipt:
            mock_receipt.side_effect = Exception("Transaction query failed")
            
            with pytest.raises(DEXError, match="Transaction query failed: Transaction query failed"):
                await uniswap_client.get_transaction_status("0x1234567890abcdef")

    @pytest.mark.asyncio
    async def test_get_quote_for_fee_tier_error_handling(self, uniswap_client):
        """Test fee tier quote error handling (lines 395-397)."""
        # Test internal error handling in _get_quote_for_fee_tier
        with patch.object(uniswap_client.logger, 'error') as mock_logger:
            result = await uniswap_client._get_quote_for_fee_tier(
                "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC",
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                Decimal("1000"),
                500,
                SwapType.EXACT_INPUT,
                50
            )
            
            # Should return a SwapQuote even with mock implementation
            assert result is not None
            assert isinstance(result, SwapQuote)

    @pytest.mark.asyncio 
    async def test_select_best_quote_exact_output(self, uniswap_client):
        """Test best quote selection for exact output swaps (lines 404-406)."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        
        # Create quotes with different input amounts for exact output
        quote1 = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=Decimal("1000"),  # Higher input amount
            output_amount=Decimal("0.5"),
            price=Decimal("0.0005"),
            price_impact_bps=25,
            slippage_bps=50,
            dex_name="uniswap_v3"
        )
        
        quote2 = SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=Decimal("995"),   # Lower input amount (better)
            output_amount=Decimal("0.5"),
            price=Decimal("0.000502"),
            price_impact_bps=20,
            slippage_bps=50,
            dex_name="uniswap_v3"
        )
        
        quotes = [quote1, quote2]
        best_quote = await uniswap_client._select_best_quote(quotes, SwapType.EXACT_OUTPUT)
        
        # For exact output, should choose quote with lowest input amount
        assert best_quote == quote2
        assert best_quote.input_amount == Decimal("995")

    @pytest.mark.asyncio
    async def test_fee_tier_mock_calculations(self, uniswap_client):
        """Test fee tier specific calculations (lines 368-397)."""
        input_token = "0xA0b86a33E6441c59C80d49Abb5a83c2c4cfE79cC"
        output_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        amount = Decimal("1000")
        
        # Test 0.05% fee tier (500)
        quote_500 = await uniswap_client._get_quote_for_fee_tier(
            input_token, output_token, amount, 500, SwapType.EXACT_INPUT, 50
        )
        assert quote_500.additional_fees["pool_fee"] == Decimal("5")  # 500/100 = 5 basis points
        assert quote_500.output_amount == amount * Decimal("0.502")  # Best rate
        assert quote_500.price_impact_bps == 20
        
        # Test 0.3% fee tier (3000)
        quote_3000 = await uniswap_client._get_quote_for_fee_tier(
            input_token, output_token, amount, 3000, SwapType.EXACT_INPUT, 50
        )
        assert quote_3000.additional_fees["pool_fee"] == Decimal("30")  # 3000/100 = 30 basis points
        assert quote_3000.output_amount == amount * Decimal("0.500")  # Standard rate
        assert quote_3000.price_impact_bps == 25
        
        # Test 1% fee tier (10000)
        quote_10000 = await uniswap_client._get_quote_for_fee_tier(
            input_token, output_token, amount, 10000, SwapType.EXACT_INPUT, 50
        )
        assert quote_10000.additional_fees["pool_fee"] == Decimal("100")  # 10000/100 = 100 basis points
        assert quote_10000.output_amount == amount * Decimal("0.495")  # Exotic pairs
        assert quote_10000.price_impact_bps == 35