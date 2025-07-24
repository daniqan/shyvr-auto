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